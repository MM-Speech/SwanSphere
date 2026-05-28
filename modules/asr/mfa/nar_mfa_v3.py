from dataclasses import dataclass
from typing import Any, Optional, Tuple
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.nn.seq_utils import add_prefix_nd, sequence_mask, remove_prefix, remove_suffix
from utils.nn.generation_utils import sample_topk, sample
from utils.commons.dataset_utils import pad_or_cut_xd
from utils.audio.mel import MelNet

from modules.commons.pos_encoding import SinusoidalPositionalEmbedding
from modules.commons.conv import ConvBlocks
from modules.asr.llama.llama import LLaMa, ModelArgs as LLaMaModelArgs
from modules.asr.mfa.nar_mfa_utils import dur_to_paraformer_label, aggregate_frame_by_dur, cif_durations_frames


def build_nar_mfa_model(hparams, init_pretrained=True, vocab_size=None):
    model_config = ModelArgs(
        vocab_size=vocab_size,
        audio_encoder_type=hparams.get('audio_encoder_type'),
        audio_encoder_ckpt=hparams.get('audio_encoder_ckpt'),
        init_pretrained=init_pretrained,
    )
    print('| Use model version v3!')
    model = MFAModelV3(model_config)
    return model

@dataclass
class ModelArgs:
    vocab_size: int = None
    
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
    

class MFAModelV3(nn.Module):
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
        
        self.decoder = LLaMa(LLaMaModelArgs(
            dim=config.dim,
            n_layers=config.decoder_n_layers,
            n_heads=16,
            use_causal_attn=False,
            crossattn_n_layers=config.decoder_n_layers,
        ))
        
        self.txt_out = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        
    def forward_model(self, wavs, wav_mask, do_checkpoint=False):
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
        
    def forward(self, inputs, do_checkpoint=False):  
        wavs = inputs['wavs']
        wav_mask = inputs['wav_mask']
        txt_tokens = inputs['txt_tokens']   # [B, T]
        txt_mask = inputs['txt_mask']
        dur = inputs['dur']
                
        x, mel_mask = self.forward_model(wavs, wav_mask, do_checkpoint)
        frame_logits = self.agg_prob_proj(x)[..., 0]
        
        # alpha aggregation loss
        frame_prob = torch.sigmoid(frame_logits)  # [B, T]
        agg_prob, valid_dur_mask = aggregate_frame_by_dur(frame_prob, dur)
        agg_label = valid_dur_mask.to(agg_prob.dtype)
        agg_loss = (agg_prob - agg_label).pow(2)
        agg_loss = (agg_loss * agg_label).sum() / agg_label.sum().clamp_min(1)
        
        # hidden aggregation
        x_agg = x * frame_prob[..., None]
        denom = agg_prob[..., None]
        x_agg, _ = aggregate_frame_by_dur(x_agg, dur)
        x_agg = x_agg / denom.clamp_min(1e-5)
        x_agg = self.decoder(x_agg, valid_dur_mask, context=x, context_lens=mel_mask.sum(1), do_checkpoint=do_checkpoint)
        txt_logits = self.txt_out(x_agg)  # [B, T, V]
        ce_loss = F.cross_entropy(txt_logits.transpose(1, 2), txt_tokens.clamp(0, self.config.vocab_size - 1), reduction='none')
        ce_loss = (ce_loss * txt_mask * valid_dur_mask).sum() / (txt_mask * valid_dur_mask).sum()
        
        ret = {
            'bd_agg_loss': agg_loss,
            'ce_loss': ce_loss,
            'ntokens': mel_mask.sum()
        }
        return ret
    
    def inference(self, wavs, wav_mask=None, return_logprob=False):
        if wav_mask is None:
            wav_mask = torch.ones_like(wavs)
        x, mel_mask = self.forward_model(wavs, wav_mask, do_checkpoint=False)
        frame_logits = self.agg_prob_proj(x)[..., 0]
        frame_prob = torch.sigmoid(frame_logits)
        dur, dur_mask = cif_durations_frames(frame_prob)
        
        x_agg = x * frame_prob[..., None]
        denom, _ = aggregate_frame_by_dur(frame_prob, dur)
        x_agg, _ = aggregate_frame_by_dur(x_agg, dur)
        x_agg = x_agg / denom.clamp_min(1e-5)[..., None]
        x_agg = self.decoder(x_agg, dur_mask, context=x, context_lens=mel_mask.sum(1), do_checkpoint=False)
        txt_logits = self.txt_out(x_agg)

        txt_pred = torch.argmax(txt_logits, dim=-1)     # [B, T]
        
        ret = {
            'txt_pred': txt_pred,
            'dur': dur,
            'dur_mask': dur_mask
        }
        
        if return_logprob:
            txt_logits: torch.Tensor = torch.max(txt_logits, dim=-1)
            logprob = (txt_logits.sigmoid().log() * dur_mask.to(txt_logits)).sum(1)
            ret['logprob'] = logprob
            
        return ret
    