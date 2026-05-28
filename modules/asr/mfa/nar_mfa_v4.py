from dataclasses import dataclass
from typing import Any, Optional, Tuple
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.nn.seq_utils import add_prefix_nd, sequence_mask, remove_prefix, remove_suffix
from utils.nn.generation_utils import sample_topk, sample, stochastic_round
from utils.commons.dataset_utils import pad_or_cut_xd
from utils.audio.mel import MelNet

from modules.commons.pos_encoding import SinusoidalPositionalEmbedding
from modules.commons.conv import ConvBlocks
from modules.asr.mfa.nar_mfa_utils import dur_to_paraformer_label, aggregate_frame_by_dur, cif_durations_frames
from modules.flow_matching.mask_cfm import MaskFlowMatching


def build_nar_mfa_model(hparams, init_pretrained=True, vocab_size=None):
    model_config = ModelArgs(
        vocab_size=800 + 15000,
        mask_id=798,
        padding_id=797,
        dim=hparams.get('hidden_size', 1024),
        decoder_n_layers=hparams.get('decoder_n_layers', 12),
        audio_encoder_type=hparams.get('audio_encoder_type'),
        audio_encoder_ckpt=hparams.get('audio_encoder_ckpt'),
        init_pretrained=init_pretrained,
    )
    print('| Use model version v4!')
    model = MFAModelV4(model_config)
    return model

@dataclass
class ModelArgs:
    vocab_size: int = None
    mask_id: int = None
    padding_id: int = None
    
    # audio
    fft_size: int = 800
    audio_num_mel_bins: int = 160
    audio_sample_rate: int = 16000
    hop_size: int = 160
    win_size: int = 800
    fmin: int = 0
    fmax: int = 12000
    
    # model
    dim: int = 1024
    decoder_n_layers: int = 12
    
    audio_encoder_type: str = 'wavlm'
    audio_encoder_ckpt: str = None
    init_pretrained: bool = True
    
    # diffusion
    schedule: str = 'cosine'
    

class SafeEmbedding(nn.Embedding):
    def forward(self, input):
        if input.min() < 0 or input.max() >= self.num_embeddings:
            print(f"| ERROR: {self.num_embeddings = } {input.min() = } {input.max() = }")
        return F.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)


class MFAModelV4Backbone(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        
        from modules.asr.wavlm.WavLM import WavLM, WavLMConfig
        if config.init_pretrained:
            checkpoint = torch.load(config.audio_encoder_ckpt)
            cfg = WavLMConfig(checkpoint['cfg'])
            model = WavLM(cfg)
            model.load_state_dict(checkpoint['model'])
        else:
            cfg = json.load(open('checkpoints/wavlm/WavLM-Large-Config.json'))
            cfg = WavLMConfig(cfg)
            model = WavLM(cfg)
        self.audio_encoder_dim = cfg.encoder_embed_dim
        self.audio_encoder = model
        self.audio_encoder_hopsize = 320
        self.audio_encoder_sample_rate = 16000
        self.audio_encoder_upper = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.mel_net = MelNet(hparams=dict(
            fft_size=config.fft_size,
            audio_num_mel_bins=config.audio_num_mel_bins,
            audio_sample_rate=config.audio_sample_rate,
            hop_size=config.hop_size,
            win_size=config.win_size,
            fmin=config.fmin,
            fmax=config.fmax
        ))
        self.mel_pos_enc = SinusoidalPositionalEmbedding(config.dim)
        self.mel_prenet = nn.Linear(config.audio_num_mel_bins, config.dim)
        self.mel_encoder = ConvBlocks(config.dim, config.dim, dilations=None, 
                                      kernel_size=3, num_layers=2, act_type='swish')
        
        self.merge_proj = nn.Linear(config.dim + self.audio_encoder_dim, config.dim)
        
        self.agg_prob_proj = nn.Linear(config.dim, 1)
        nn.init.zeros_(self.agg_prob_proj.weight)
        nn.init.constant_(self.agg_prob_proj.bias, -3.0)

        from modules.tts.llama_dit.llama_ca import LLaMa, ModelArgs as LLaMaModelArgs
        self.decoder = LLaMa(LLaMaModelArgs(
            encoder_dim=config.dim,
            encoder_n_layers=config.decoder_n_layers,
            encoder_n_heads=16,
            use_causal_attn=False,
            crossattn_n_layers=config.decoder_n_layers,
            use_caption_pool_in_adaln=True,
            use_qk_norm=True
        ))
        from modules.tts.f5_dit.f5_modules import TimestepEmbedding
        self.f5_time_embed = TimestepEmbedding(config.dim)
        
        # self.token_embed = nn.Embedding(config.vocab_size, config.dim)
        self.token_embed = SafeEmbedding(config.vocab_size, config.dim)
        self.token_out = nn.Linear(config.dim, config.vocab_size, bias=False)
        
    def forward_encoder(self, wavs, wav_mask, do_checkpoint=False):
        with torch.no_grad():
            if self.mel_net.device != wavs.device:
                self.mel_net.to(wavs.device)
            mel = self.mel_net(wavs)
        mel_mask = wav_mask[:, ::self.config.hop_size]
        
        x_mel = self.mel_prenet(mel)
        x_mel = x_mel + self.mel_pos_enc(positions=torch.arange(x_mel.shape[1], device=mel.device), out_dtype=x_mel.dtype, device=mel.device)[None, ...]
        x_mel = self.mel_encoder(x_mel)
        
        feat, feat_padding_mask = self.audio_encoder.extract_features(wavs, padding_mask=~(wav_mask.bool()), do_checkpoint=do_checkpoint)    # [B, T, C]
        feat = self.audio_encoder_upper(feat.transpose(1, 2)).transpose(1, 2)
        feat = pad_or_cut_xd(feat, tgt_len=x_mel.shape[1], dim=1)
        
        x = self.merge_proj(torch.cat([x_mel, feat], dim=-1)) # [B, T, C]
        
        return x, mel_mask
    
    def forward_length_predictor(self, x, mel_mask):
        frame_logits = self.agg_prob_proj(x)[..., 0]
        frame_prob = torch.sigmoid(frame_logits)  # [B, T]
        agg_lens = (frame_prob * mel_mask).sum(1)   # [B]
        return agg_lens
        
    def forward(self, x, t, cond, attn_mask):  
        x_mel, mel_mask = cond['x_mel'], cond['mel_mask']
        t = self.f5_time_embed(t).to(x_mel.dtype)
        
        x = self.token_embed(x).bfloat16()
        x = self.decoder.forward(
            x, t, attn_mask, context=x_mel, context_lens=mel_mask.sum(1), 
            do_checkpoint=cond.get('do_checkpoint', False)
        )
        x = self.token_out(x)
        return x
    
    
class MFAModelV4(MaskFlowMatching):
    def __init__(self, config: ModelArgs):
        self.config = config
        backbone = MFAModelV4Backbone(config)
        super().__init__(
            vocab_size=config.vocab_size,
            mask_id=config.mask_id,
            backbone=backbone,
            schedule=config.schedule,
            use_cfm_weight=True,
            pad_id=config.padding_id,
            enforce_one_mask=True
        )

    def forward(self, inputs, do_checkpoint=False):
        wavs = inputs['wavs']
        wav_mask = inputs['wav_mask']
        txt_tokens = inputs['txt_tokens']   # [B, T]
        txt_mask = inputs['txt_mask']
        dur = inputs['dur']
                
        x_mel, mel_mask = self.backbone.forward_encoder(wavs, wav_mask, do_checkpoint)
        
        lens_pred = self.backbone.forward_length_predictor(x_mel, mel_mask)   # [B]
        txt_lens = txt_mask.sum(1).to(lens_pred.dtype)  # [B]
        # print(f"{lens_pred = } {txt_lens = }")
        len_pred_loss = ((lens_pred - txt_lens) / txt_lens).abs().sum() / txt_lens.shape[0]
        len_pred_loss_log = (torch.log1p(lens_pred) - torch.log1p(txt_lens)).pow(2).sum() / txt_lens.shape[0]
        len_pred_loss = len_pred_loss + len_pred_loss_log
        
        offset = torch.cumsum(dur, dim=1) + 800
        offset[dur == 0] = 0
        x = torch.zeros_like(dur).repeat(1, 2)
        x[:, 0::2] = txt_tokens
        x[:, 1::2] = offset
        x_mask = sequence_mask(txt_mask.sum(1) * 2, maxlen=x.shape[1])
        
        # print(f"{x = } {x.max() = } {x.min() = } {dur.shape = } {x_mel.shape = }")
        fm_result = self.compute_loss(
            x0=x,
            t=None,
            padding_mask=~x_mask,
            cond={
                'x_mel': x_mel,
                'mel_mask': mel_mask,
                'do_checkpoint': do_checkpoint
            }
        )
        fm_loss = fm_result['loss']
        
        ret = {
            'len_pred_loss': len_pred_loss,
            'ce_loss': fm_loss,
            'ntokens': mel_mask.sum()
        }
        return ret
        
    def inference(self, wavs, wav_mask=None, timesteps=50, temperature=1.0, token_topk=10):
        if wav_mask is None:
            wav_mask = torch.ones_like(wavs)
            
        x_mel, mel_mask = self.backbone.forward_encoder(wavs, wav_mask)
        lens_pred = self.backbone.forward_length_predictor(x_mel, mel_mask)   # [B]
        lens_pred = stochastic_round(lens_pred)
        
        x_mask = sequence_mask(lens_pred * 2)
        x_init = torch.full_like(x_mask.long(), self.config.mask_id)
        cond = {
            'x_mel': x_mel,
            'mel_mask': mel_mask,
        }
        
        x_pred = self.infer(
            x_init=x_init,
            steps=timesteps,
            temperature=temperature,
            topk_per_step=None,
            token_topk=token_topk,
            cond=cond,
            padding_mask=~x_mask,
        )
        
        txt_pred = x_pred[:, 0::2]
        offsets = x_pred[:, 1::2] - 800
        offsets = torch.cat([torch.zeros_like(offsets[:, :1]), offsets], dim=1)
        dur = offsets[:, 1:] - offsets[:, :-1]
        dur_mask = sequence_mask(lens_pred)
        
        print(f"{x_pred.cpu().numpy().tolist() = }")
        print(f"{dur.cpu().numpy().tolist() = }")
        
        ret = {
            'txt_pred': txt_pred,
            'dur': dur,
            'dur_mask': dur_mask
        }

        return ret
        
        
        
    