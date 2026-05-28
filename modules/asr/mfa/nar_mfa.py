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
from modules.asr.mfa.nar_mfa_utils import dur_to_paraformer_label, aggregate_frame_by_dur, cif_durations_frames


def build_nar_mfa_model(hparams, init_pretrained=True, vocab_size=None):
    model_config = ModelArgs(
        vocab_size=vocab_size,
        audio_encoder_type=hparams.get('audio_encoder_type'),
        audio_encoder_ckpt=hparams.get('audio_encoder_ckpt'),
        init_pretrained=init_pretrained,
        use_refiner=hparams.get('use_refiner', False),
        refiner_n_layers=hparams.get('refiner_n_layers', 4),
        use_mel_pos_enc=hparams.get('use_mel_pos_enc', False),
        use_bd_frame_loss=hparams.get('use_bd_frame_loss', True),
        use_total_agg_loss=hparams.get('use_total_agg_loss', False),
        use_agg_gate_proj=hparams.get('use_agg_gate_proj', True)
    )
        
    if hparams.get('model_version', 'v1') == 'v1':
        model = MFAModel(model_config)
    elif hparams.get('model_version', 'v1') == 'v2':
        print('| Use model version v2!')
        model_config.unet_dim = 1024
        model_config.unet_midnet_n_layers = hparams.get('encoder_n_layers', 16)
        model = MFAModelV2(model_config)
        for name, parameter in model.audio_encoder.named_parameters():
            parameter.requires_grad = False
    return model

@dataclass
class ModelArgs:
    vocab_size: int = None
    padding_idx: int = None
    
    # audio
    fft_size: int = 800
    audio_num_mel_bins: int = 160
    audio_sample_rate: int = 16000
    hop_size: int = 160
    win_size: int = 800
    fmin: int = 0
    fmax: int = 12000
    
    # Unet
    unet_dim: int = 768
    unet_midnet_n_layers: int = 8
    unet_updown_rates: tuple = (2, 2, 2)
    unet_channel_multiples: tuple = (1, 1, 1)
    unet_kernel_size: int = 3
    unet_constant_channels: bool = False
    unet_use_skip_layer: bool = False
    unet_skip_scale: int = 1

    audio_encoder_type: str = 'wavlm'
    audio_encoder_ckpt: str = None
    init_pretrained: bool = True
    
    use_refiner: bool = False
    refiner_n_layers: int = 4
    use_mel_pos_enc: bool = False
    use_agg_gate_proj: bool = True

    # loss
    use_bd_frame_loss: bool = True
    use_total_agg_loss: bool = False

class MFAModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        
        self.mel_net = MelNet(hparams=dict(
            fft_size=config.fft_size,
            audio_num_mel_bins=config.audio_num_mel_bins,
            audio_sample_rate=config.audio_sample_rate,
            hop_size=config.hop_size,
            win_size=config.win_size,
            fmin=config.fmin,
            fmax=config.fmax
        ))
        self.unet_prenet = nn.Linear(config.audio_num_mel_bins, config.unet_dim)
        if config.use_mel_pos_enc:
            self.mel_pos_enc = SinusoidalPositionalEmbedding(config.unet_dim)
        
        from modules.asr.llama.llama import LLaMa, ModelArgs as LLaMaModelArgs
        mid_net = LLaMa(LLaMaModelArgs(
            dim=config.unet_dim,
            n_layers=config.unet_midnet_n_layers,
            n_heads=16,
            use_causal_attn=False
        ))
        
        from modules.commons.unet import Unet
        self.unet = Unet(
            hidden_size=config.unet_dim,
            down_layers=len(config.unet_updown_rates),
            up_layers=len(config.unet_updown_rates),
            kernel_size=config.unet_kernel_size,
            updown_rates=config.unet_updown_rates,
            channel_multiples=config.unet_channel_multiples,
            dropout=0,
            is_BTC=True,
            constant_channels=config.unet_constant_channels,
            mid_net=mid_net,
            use_skip_layer=config.unet_use_skip_layer,
            skip_scale=config.unet_skip_scale
        )
        self.unet_stride = int(np.prod(self.config.unet_updown_rates))
        
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
        
        self.merge_proj = nn.Linear(config.unet_dim + self.audio_encoder_dim, config.unet_dim)
        
        self.bd_proj = nn.Linear(config.unet_dim, 1)
        
        if config.use_agg_gate_proj:
            self.agg_gate_proj = nn.Linear(config.unet_dim, 1)
        self.txt_out = nn.Linear(config.unet_dim, config.vocab_size, bias=False)
        
        if config.use_refiner:
            self.refiner = LLaMa(LLaMaModelArgs(
                dim=config.unet_dim,
                n_layers=config.refiner_n_layers,
                n_heads=16,
                use_causal_attn=False,
                crossattn_n_layers=4
            ))
        
    def forward_model(self, wavs, wav_mask, do_checkpoint=False):
        with torch.no_grad():
            if self.mel_net.device != wavs.device:
                self.mel_net.to(wavs.device)
            mel = self.mel_net(wavs)
        mel_mask = wav_mask[:, ::self.config.hop_size]
            
        x_mel = self.unet_prenet(mel)
        if self.config.use_mel_pos_enc:
            x_mel = x_mel + self.mel_pos_enc(positions=torch.arange(x_mel.shape[1], device=mel.device), out_dtype=x_mel.dtype, device=mel.device)[None, ...]
        x_mel = self.unet(
            x_mel,
            mid_kwargs={
                'attn_mask': mel_mask[:, ::self.unet_stride],
                'do_checkpoint': do_checkpoint
            }
        )
        
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
        dur_paraformer_label = inputs['dur_paraformer_label']
                
        x, mel_mask = self.forward_model(wavs, wav_mask, do_checkpoint)
        bd_logits = self.bd_proj(x)[..., 0]
        
        # boundary frame loss
        if self.config.use_bd_frame_loss:
            bd_frame_loss = F.binary_cross_entropy_with_logits(bd_logits, dur_paraformer_label.clamp(0, 1), reduction='none')
            bd_frame_loss = (bd_frame_loss * mel_mask).sum() / mel_mask.sum().clamp_min(1)

        # boundary aggregation loss
        bd_prob = torch.sigmoid(bd_logits)  # [B, T]
        bd_agg_prob, valid_dur_mask = aggregate_frame_by_dur(bd_prob, dur)
        bd_agg_label = valid_dur_mask.to(bd_agg_prob.dtype)
        bd_agg_loss = (bd_agg_prob - bd_agg_label).pow(2)
        bd_agg_loss = (bd_agg_loss * bd_agg_label).sum() / bd_agg_label.sum().clamp_min(1)
        
        if self.config.use_total_agg_loss:
            agg_lens = (bd_prob * mel_mask).sum(1)
            valid_dur_lens = bd_agg_label.sum(1)
            total_agg_loss = (agg_lens - valid_dur_lens).pow(2).sum() / agg_lens.shape[0]
        
        # hidden aggregation
        if self.config.use_agg_gate_proj:
            agg_gate = torch.sigmoid(self.agg_gate_proj(x)) # [B, T, 1]
            denom, _ = aggregate_frame_by_dur(agg_gate, dur)
        else:
            agg_gate = bd_prob[..., None]
            denom = bd_agg_prob[..., None]
        x_agg = x * agg_gate
        x_agg, _ = aggregate_frame_by_dur(x_agg, dur)
        x_agg = x_agg / denom.clamp_min(1e-5)
        if self.config.use_refiner:
            x_agg = self.refiner(x_agg, valid_dur_mask, context=x, context_lens=mel_mask.sum(1), do_checkpoint=do_checkpoint)
        txt_logits = self.txt_out(x_agg)  # [B, T, V]
        ce_loss = F.cross_entropy(txt_logits.transpose(1, 2), txt_tokens.clamp(0, self.config.vocab_size - 1), reduction='none')
        ce_loss = (ce_loss * txt_mask * valid_dur_mask).sum() / (txt_mask * valid_dur_mask).sum()
        
        ret = {
            'bd_agg_loss': bd_agg_loss,
            'ce_loss': ce_loss,
            'ntokens': mel_mask.sum()
        }
        if self.config.use_bd_frame_loss:
            ret['bd_frame_loss'] = bd_frame_loss
        if self.config.use_total_agg_loss:
            ret['total_agg_loss'] = total_agg_loss
        return ret
    
    def inference(self, wavs, wav_mask=None):
        if wav_mask is None:
            wav_mask = torch.ones_like(wavs)
        x, mel_mask = self.forward_model(wavs, wav_mask, do_checkpoint=False)
        bd_logits = self.bd_proj(x)[..., 0]
        bd_prob = torch.sigmoid(bd_logits)
        dur, dur_mask = cif_durations_frames(bd_prob)
        
        if self.config.use_agg_gate_proj:
            agg_gate = torch.sigmoid(self.agg_gate_proj(x))
        else:
            agg_gate = bd_prob[..., None]
        x_agg = x * agg_gate
        denom, _ = aggregate_frame_by_dur(agg_gate, dur)
        x_agg, _ = aggregate_frame_by_dur(x_agg, dur)
        x_agg = x_agg / denom.clamp_min(1e-5)
        if self.config.use_refiner:
            x_agg = self.refiner(x_agg, dur_mask, context=x, context_lens=mel_mask.sum(1), do_checkpoint=False)
        txt_logits = self.txt_out(x_agg)

        txt_pred = torch.argmax(txt_logits, dim=-1)     # [B, T]
        return txt_pred, dur, dur_mask
    
    
class MFAModelV2(MFAModel):
    def forward_model(self, wavs, wav_mask, do_checkpoint=False):
        with torch.no_grad():
            if self.mel_net.device != wavs.device:
                self.mel_net.to(wavs.device)
            mel = self.mel_net(wavs)
            mel_mask = wav_mask[:, ::self.config.hop_size]
            
            feat, feat_padding_mask = self.audio_encoder.extract_features(wavs, padding_mask=~(wav_mask.bool()))    # [B, T, C]
            feat = self.audio_encoder_upper(feat.transpose(1, 2)).transpose(1, 2)
            feat = pad_or_cut_xd(feat, tgt_len=mel.shape[1], dim=1)
            
        x = self.unet_prenet(mel)
        
        x = self.merge_proj(torch.cat([x, feat], dim=-1))
        
        x = self.unet(
            x,
            mid_kwargs={
                'attn_mask': mel_mask[:, ::self.unet_stride],
                'do_checkpoint': do_checkpoint
            }
        )

        return x, mel_mask

        