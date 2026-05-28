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
        vocab_size=800,
        mask_id=798,
        padding_id=797,
        dim=hparams.get('hidden_size', 768),
        decoder_n_layers=hparams.get('decoder_n_layers', 12),
        aligner_n_layers=hparams.get('aligner_n_layers', 4),
        audio_encoder_type=hparams.get('audio_encoder_type'),
        audio_encoder_ckpt=hparams.get('audio_encoder_ckpt'),
        init_pretrained=init_pretrained,
    )
    print('| Use model version v5!')
    model = MFAModelV5(model_config)
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
    fmax: int = 8000
    
    # model
    dim: int = 1024
    decoder_n_layers: int = 12
    aligner_n_layers: int = 4
    
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

        from modules.tts.llama_dit.llama_ca import LLaMa as LLaMaDiT, ModelArgs as LLaMaDiTModelArgs
        self.decoder = LLaMaDiT(LLaMaDiTModelArgs(
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
        
        from modules.asr.llama.llama import LLaMa, ModelArgs as LLaMaModelArgs
        self.aligner = LLaMa(LLaMaModelArgs(
            dim=config.dim,
            n_layers=config.aligner_n_layers,
            n_heads=16,
            use_causal_attn=False,
            crossattn_n_layers=config.aligner_n_layers,
            use_qk_norm=True
        ))
        self.aligner_proj = nn.Linear(config.dim, 1)
        
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
    
    def forward_aligner(self, x_mel, mel_mask, txt_tokens, txt_mask, do_checkpoint=False):
        txt_embed = self.token_embed(txt_tokens).bfloat16()
        x_mel = self.aligner.forward(
            x_mel, mel_mask, context=txt_embed, context_lens=txt_mask.sum(1), do_checkpoint=do_checkpoint
        )
        frame_logits = self.aligner_proj(x_mel)[..., 0]
        return frame_logits
        
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
    
    
class MFAModelV5(MaskFlowMatching):
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
        
        fm_result = self.compute_loss(
            x0=txt_tokens,
            t=None,
            padding_mask=~txt_mask.bool(),
            cond={
                'x_mel': x_mel,
                'mel_mask': mel_mask,
                'do_checkpoint': do_checkpoint
            }
        )
        fm_loss = fm_result['loss']
        
        # alpha aggregation loss
        frame_logits = self.backbone.forward_aligner(
            x_mel, mel_mask, txt_tokens, txt_mask, do_checkpoint
        )
        frame_prob = torch.sigmoid(frame_logits)  # [B, T]
        agg_prob, valid_dur_mask = aggregate_frame_by_dur(frame_prob, dur)
        agg_label = valid_dur_mask.to(agg_prob.dtype)
        agg_loss = (agg_prob - agg_label).pow(2)
        agg_loss = (agg_loss * agg_label).sum() / agg_label.sum().clamp_min(1)
        
        ret = {
            'len_pred_loss': len_pred_loss,
            'ce_loss': fm_loss,
            'bd_agg_loss': agg_loss,
            'ntokens': mel_mask.sum()
        }
        return ret
        
    def inference(self, wavs, wav_mask=None, timesteps=100, temperature=0.0, token_topk=1):
        if wav_mask is None:
            wav_mask = torch.ones_like(wavs)
            
        x_mel, mel_mask = self.backbone.forward_encoder(wavs, wav_mask)
        lens_pred = self.backbone.forward_length_predictor(x_mel, mel_mask) * 1.01   # [B]
        lens_pred = torch.ceil(lens_pred).long()
        
        txt_mask = sequence_mask(lens_pred)
        txt_init = torch.full_like(txt_mask.long(), self.config.mask_id)
        cond = {
            'x_mel': x_mel,
            'mel_mask': mel_mask,
        }
        
        txt_pred = self.infer(
            x_init=txt_init,
            steps=timesteps,
            temperature=temperature,
            topk_per_step=None,
            token_topk=token_topk,
            cond=cond,
            padding_mask=~txt_mask,
        )
        
        frame_logits = self.backbone.forward_aligner(
            x_mel, mel_mask, txt_pred, txt_mask
        )
        frame_prob = torch.sigmoid(frame_logits)    # [B, T]
        frame_prob = frame_prob / frame_prob.sum(1, keepdim=True).clamp_min(1) * lens_pred.to(frame_prob)
        dur, dur_mask = cif_durations_frames(frame_prob)
        
        print(f"{txt_pred.cpu().numpy().tolist() = }")
        print(f"{dur.cpu().numpy().tolist() = }")
        
        ret = {
            'txt_pred': txt_pred,
            'dur': dur,
            'dur_mask': dur_mask
        }

        return ret
        
        
        
    