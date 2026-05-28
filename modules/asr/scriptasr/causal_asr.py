from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.asr.llama.llama import LLaMa, ModelArgs as LLaMaModelArgs
from utils.nn.seq_utils import add_prefix_nd, sequence_mask, remove_prefix, remove_suffix
from utils.nn.generation_utils import sample_topk, sample
from utils.commons.dataset_utils import pad_or_cut_xd

@dataclass
class ModelArgs:
    backbone = 'llama'
    lm_config = LLaMaModelArgs()

    vocab_size: int = None
    padding_idx: int = None

    audio_encoder_type: str = 'wavlm'
    audio_encoder_ckpt: str = None
    init_pretrained: bool = True

    model_spk_diarization: bool = False
    spk_diarization_dim: int = 512
    spk_diarization_after_lm: bool = False
    max_spk_num: int = None

class CausalASRModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        if config.backbone == 'llama':
            self.lm_embed = nn.Embedding(config.vocab_size, config.lm_config.dim, config.padding_idx)
            self.lm = LLaMa(config.lm_config)
            self.lm_head = nn.Linear(config.lm_config.dim, config.vocab_size, bias=False)
        elif config.backbone == 'llama_seq2seq':
            from modules.asr.llama.llama_seq2seq import Seq2SeqLLaMA, ModelArgs as LLaMaS2SModelArgs
            self.lm_embed = nn.Embedding(config.vocab_size, config.lm_config.dim, config.padding_idx)
            self.lm = Seq2SeqLLaMA(config.lm_config)
            self.lm_head = nn.Linear(config.lm_config.dim, config.vocab_size, bias=False)
        elif config.backbone == 'qwen3':
            from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
            from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
            if config.init_pretrained:
                self.lm = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", use_safetensors=True, attn_implementation="flash_attention_2")
            else:
                self.lm = Qwen3ForCausalLM._from_config(Qwen3Config.from_pretrained("Qwen/Qwen3-0.6B"), attn_implementation="flash_attention_2")
            from utils.nn.embedding import resize_embedding_layer
            resize_embedding_layer(self.lm, config.vocab_size)
            config.lm_config.dim = self.lm.config.hidden_size

        if config.audio_encoder_type == 'wavlm':    # checkpoints/wavlm/WavLM-Large.pt
            from modules.asr.wavlm.WavLM import WavLM, WavLMConfig
            checkpoint = torch.load(config.audio_encoder_ckpt)
            cfg = WavLMConfig(checkpoint['cfg'])
            model = WavLM(cfg)
            if config.init_pretrained:
                model.load_state_dict(checkpoint['model'])
            self.audio_encoder_dim = cfg.encoder_embed_dim
            self.audio_encoder = model
            self.audio_encoder_hopsize = 320
            self.audio_encoder_sample_rate = 16000
            self.audio_proj = nn.Linear(self.audio_encoder_dim, config.lm_config.dim, bias=False)
        elif config.audio_encoder_type == 'xlsr-53':    # facebook/wav2vec2-large-xlsr-53
            from transformers import Wav2Vec2Model, Wav2Vec2Config
            if config.init_pretrained:
                self.audio_encoder = Wav2Vec2Model.from_pretrained(config.audio_encoder_ckpt, use_safetensors=True, attn_implementation="flash_attention_2")
            else:
                self.audio_encoder = Wav2Vec2Model._from_config(Wav2Vec2Config.from_pretrained(config.audio_encoder_ckpt), attn_implementation="flash_attention_2")
            self.audio_encoder_dim = self.audio_encoder.config.hidden_size
            self.audio_encoder_hopsize = 320
            self.audio_encoder_sample_rate = 16000
            self.audio_proj = nn.Linear(self.audio_encoder_dim, config.lm_config.dim, bias=False)

        if config.model_spk_diarization:
            from modules.flow_matching.llama import LLaMa as LLaMaSmall,  ModelArgs as ModelArgsSmall
            self.spk_proj_in = nn.Linear(self.audio_encoder_dim, config.spk_diarization_dim, bias=False)
            self.spk_decoder = LLaMaSmall(ModelArgsSmall(
                dim=config.spk_diarization_dim, n_layers=6, n_heads=8
            ))
            self.spk_proj_out = nn.Linear(config.spk_diarization_dim, config.max_spk_num, bias=False)

        self.config = config

    def forward_audio_encoder(self, wavs, wav_mask, do_checkpoint=False):
        if self.config.audio_encoder_type == 'wavlm':
            feat, feat_padding_mask = self.audio_encoder.extract_features(wavs, padding_mask=~(wav_mask.bool()), do_checkpoint=do_checkpoint)    # [B, T, C]
            feat_mask = ~feat_padding_mask  # [B, T]
        elif self.config.audio_encoder_type == 'xlsr-53':
            feat = self.audio_encoder(input_values=wavs, attention_mask=wav_mask).last_hidden_state
            feat_mask = wav_mask[:, ::self.audio_encoder_hopsize]
            feat_mask = pad_or_cut_xd(feat_mask, feat.shape[1], dim=1)
        feat = self.audio_proj(feat)
        return feat, feat_mask
    
    def forward_spk_decoder(self, feat, feat_mask):
        spk_mask_logits = self.spk_decoder(self.spk_proj_in(feat), feat_mask)
        spk_mask_logits = self.spk_proj_out(spk_mask_logits)    # [B, T, C]
        return spk_mask_logits
    
    def forward_txt_embed(self, txt_tokens):
        if self.config.backbone in ['llama', 'llama_seq2seq']:
            txt_embeds = self.lm_embed(txt_tokens)  # [B, T, C]
        elif self.config.backbone == 'qwen3':
            txt_embeds = self.lm.model.embed_tokens(txt_tokens)
        return txt_embeds
        
    def forward(self, inputs, do_checkpoint=False):
        wavs = inputs['wavs']
        wav_mask = inputs['wav_mask']
        txt_tokens = inputs['txt_tokens']   # <BOT> ... <EOT>
        txt_mask = inputs['txt_mask']
        txt_lens = txt_mask.sum(1)

        feat, feat_mask = self.forward_audio_encoder(wavs, wav_mask, do_checkpoint)
        feat_lens = feat_mask.sum(1)

        txt_embeds = self.forward_txt_embed(txt_tokens).to(feat)  # [B, T, C]

        if self.config.backbone == 'llama_seq2seq':

            x = self.lm(
                encoder_x=feat,
                encoder_padding_mask=feat_mask,
                decoder_x=txt_embeds,
                decoder_padding_mask=txt_mask,
                do_checkpoint=do_checkpoint
            )
            x = x[:, :-1]
            logits = self.lm_head(x)
        
        else:
            
            x = add_prefix_nd(feat, feat_lens, txt_embeds, txt_lens)
            attn_mask = sequence_mask(feat_lens + txt_lens)

            if self.config.backbone == 'llama':
                x = self.lm(x, attn_mask=attn_mask, do_checkpoint=do_checkpoint)
                x_ = remove_prefix(x, torch.maximum(feat_lens, torch.zeros_like(feat_lens)), txt_lens-1)
                logits = self.lm_head(x_)
            elif self.config.backbone == 'qwen3':
                x = self.lm.model(inputs_embeds=x, attention_mask=attn_mask, use_cache=False).last_hidden_state
                x_ = remove_prefix(x, torch.maximum(feat_lens, torch.zeros_like(feat_lens)), txt_lens-1)
                logits = self.lm.lm_head(x_)
            
        labels = txt_tokens[:, 1:]
        loss_mask = txt_mask[:, 1:]

        outputs = {
            'logits': logits,
            'labels': labels,
            'loss_mask': loss_mask,
            'ntokens': (feat_lens + txt_lens).sum()
        }

        if self.config.model_spk_diarization:
            if self.config.spk_diarization_after_lm:
                feat_ = remove_suffix(x, feat_lens)
                spk_mask_logits = self.forward_spk_decoder(feat_, feat_mask)
            else:
                spk_mask_logits = self.forward_spk_decoder(feat, feat_mask)
            spk_mask_labels = inputs['spk_mask'].to(spk_mask_logits.dtype)
            spk_mask_labels = spk_mask_labels[:, ::self.audio_encoder_hopsize]
            spk_mask_labels = pad_or_cut_xd(spk_mask_labels, spk_mask_logits.shape[1], dim=1)
            spk_mask_loss = F.binary_cross_entropy_with_logits(spk_mask_logits, spk_mask_labels, reduction='none')
            spk_mask_loss = spk_mask_loss.sum() / feat_mask.sum()
            outputs['spk_mask_loss'] = spk_mask_loss

        return outputs

    @torch.no_grad()
    def inference(self, wavs, txt_tokens, 
                  topk=1, temperature=0.1, max_new_tokens=512, eos_idx=None, 
                  diarization=False,
                  use_tqdm=True, print_candidates=False):
        wav_mask = torch.ones_like(wavs).int()
        feat, feat_mask = self.forward_audio_encoder(wavs, wav_mask)

        txt_mask = torch.ones_like(txt_tokens)
        txt_embeds = self.forward_txt_embed(txt_tokens).to(feat)  # [B, T, C]

        feat_lens = feat_mask.sum(1)
        txt_lens = txt_mask.sum(1)
        if self.config.backbone != 'llama_seq2seq':
            x = add_prefix_nd(feat, feat_lens, txt_embeds, txt_lens)

        if use_tqdm:
            from tqdm import tqdm
            it = tqdm(range(max_new_tokens), desc='| Generating')
        else:
            it = range(max_new_tokens)


        if self.config.backbone == 'llama':

            x_ = self.lm(x, attn_mask=sequence_mask(feat_lens + txt_lens), start_pos=0, use_cache=True)

            def forward_(token, start_pos):
                x = self.lm_embed(token)
                x = self.lm(x, attn_mask=None, start_pos=start_pos, use_cache=True)
                logits = self.lm_head(x)
                return logits

            start_pos = (feat_lens + txt_lens - 1).to(torch.int32)
            for step in it:
                logits = forward_(txt_tokens[:, -1:], start_pos)
                token_pred = self.sample(logits, topk, temperature, print_candidates)
                if token_pred[0] == eos_idx:
                    break
                txt_tokens = torch.cat([txt_tokens, token_pred], dim=1)
                start_pos = start_pos + 1

        elif self.config.backbone == 'llama_seq2seq':

            self.lm.reset_kv_cache()
            start_pos = 0

            enc_out = self.lm.encode(
                encoder_x=feat,
                encoder_padding_mask=feat_mask,
            )

            def forward_(token, start_pos):
                x = self.lm_embed(token)
                x = self.lm.decode(
                    decoder_x=x,
                    decoder_padding_mask=None,
                    enc_out=enc_out,
                    encoder_padding_mask=feat_mask,
                    start_pos=start_pos,
                    use_cache=True
                )
                logits = self.lm_head(x)
                return logits

            for step in it:
                logits = forward_(txt_tokens[:, -1:], start_pos)
                token_pred = self.sample(logits, topk, temperature, print_candidates)
                if token_pred[0] == eos_idx:
                    break
                txt_tokens = torch.cat([txt_tokens, token_pred], dim=1)
                start_pos = start_pos + 1

        elif self.config.backbone == 'qwen3':

            past_key_values = None
            for step in it:
                if step == 0:
                    input_ids = None
                    inputs_embeds = x
                    logits_to_keep = 0
                else:
                    input_ids = txt_tokens[:, -1:]
                    inputs_embeds = None
                    logits_to_keep = 1
                lm_outputs = self.lm(
                    input_ids=input_ids, inputs_embeds=inputs_embeds,
                    past_key_values=past_key_values,
                    use_cache=True, logits_to_keep=logits_to_keep,
                    output_hidden_states=True
                )
                if step == 0:
                    x_ = lm_outputs.hidden_states[-1]
                logits = lm_outputs.logits
                past_key_values = lm_outputs.past_key_values
                token_pred = self.sample(logits, topk, temperature, print_candidates)
                if token_pred[0] == eos_idx:
                    break
                txt_tokens = torch.cat([txt_tokens, token_pred], dim=1)
        
        if not diarization:
            return txt_tokens[:, txt_lens[0]:]
        
        if self.config.model_spk_diarization:
            if self.config.spk_diarization_after_lm:
                feat_ = remove_suffix(x_, feat_lens)
                spk_mask_logits = self.forward_spk_decoder(feat_, feat_mask)
            else:
                spk_mask_logits = self.forward_spk_decoder(feat, feat_mask)
            spk_mask_logits = torch.sigmoid(spk_mask_logits)

        return txt_tokens[:, txt_lens[0]:], spk_mask_logits


    def sample(self, logits, topk, temperature, print_candidates=False):
        if topk == 1:
            token_pred = torch.argmax(logits[:, -1:], dim=-1)   # [1, 1]
        else:
            # token_pred = sample_topk(logits, topk, temperature)[0, 0]
            token_pred = sample(logits[:, -1:], topk, 1.0, temperature)
        if print_candidates and hasattr(self, 'text_tokenizer'):
            topk_logits, topk_idxs = torch.topk(torch.softmax(logits[:, -1], dim=1), 5)
            candidates = []
            for topk_logit, topk_idx in zip(topk_logits[0].cpu().numpy().tolist(), topk_idxs[0].cpu().numpy().tolist()):
                candidates.append({self.text_tokenizer.decode([topk_idx]): round(topk_logit, 3)})
            print(candidates, end=' ')
            print(f'Chosen: {self.text_tokenizer.decode([token_pred[-1].item()])}')

        return token_pred
    
    @torch.no_grad()
    def inference_batch(self, wavs, wav_mask, txt_tokens, txt_mask, topk=1, temperature=0.1, 
                        max_new_tokens=512, eos_idx=None, use_tqdm=True):
        bsz = len(wavs)
        
        if use_tqdm:
            from tqdm import tqdm
            it = tqdm(range(max_new_tokens), desc='| Generating')
        else:
            it = range(max_new_tokens)
            
        if self.config.backbone == 'llama':
            # TODO
            feat, feat_mask = self.forward_audio_encoder(wavs, wav_mask)
            
            feat_lens = feat_mask.sum(1)
            txt_lens = txt_mask.sum(1)
            
        elif self.config.backbone == 'llama_seq2seq':
            feat, feat_mask = self.forward_audio_encoder(wavs, wav_mask)
            
            feat_lens = feat_mask.sum(1)
            txt_lens = txt_mask.sum(1)
            
            self.lm.reset_kv_cache()
            start_pos = 0
            
            enc_out = self.lm.encode(
                encoder_x=feat,
                encoder_padding_mask=feat_mask,
            )
            
            def forward_(token, start_pos):
                x = self.lm_embed(token)
                x = self.lm.decode(
                    decoder_x=x,
                    decoder_padding_mask=None,
                    enc_out=enc_out,
                    encoder_padding_mask=feat_mask,
                    start_pos=start_pos,
                    use_cache=True
                )
                logits = self.lm_head(x)
                return logits
            
            stop = torch.zeros(bsz).to(txt_tokens)
            for step in it:
                logits = forward_(txt_tokens[:, -1:], start_pos)
                tokens_pred = self.sample(logits, topk, temperature)
                stop[tokens_pred[..., 0] == eos_idx] = 1
                if stop.sum() == bsz:
                    break
                txt_tokens = torch.cat([txt_tokens, tokens_pred], dim=1)
                start_pos = start_pos + 1

        results = []
        for i in range(bsz):
            tokens_pred = txt_tokens[i, txt_lens[i]:]
            end_pos = torch.nonzero(tokens_pred == eos_idx)
            if len(end_pos) >= 1:
                tokens_pred = tokens_pred[:end_pos[0, 0]]
            # print(f"{len(end_pos) = }")
            # print(f"{end_pos = }")
            # print(f"{tokens_pred = }")
            results.append(tokens_pred)
            
        return results
