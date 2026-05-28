from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from modules.commons.unet import Unet
from modules.asr.llama.llama import LLaMa, ModelArgs as LLaMaModelArgs

def build_diarization_model(hparams, training=True):
    model_config = ModelArgs(
        backbone_config=LLaMaModelArgs(use_causal_attn=False),
        unet_dim=hparams.get('unet_dim', 512),
        unet_updown_rates=[int(s) for s in hparams.get('unet_updown_rates', '2-2-2').split('-')],
        unet_channel_multiples=[int(s) for s in hparams.get('unet_channel_multiples', '1-1-1').split('-')],
        unet_use_skip_layer=hparams.get('unet_use_skip_layer', True),
        unet_skip_scale=hparams.get('unet_skip_scale', 0.7071067812),
        in_channels=hparams.get('audio_num_mel_bins', 160),
        n_class=hparams.get('max_n_spk_diarization', 8)
    )
    if hparams.get('model_size', 'base') == 'small':
        model_config.backbone_config.n_layers = 12
        model_config.backbone_config.n_heads = 12
        model_config.backbone_config.dim = 768 
    elif hparams.get('model_size', 'base') == '1b':
        model_config.backbone_config.n_layers = 28
        model_config.backbone_config.n_heads = 16
        model_config.backbone_config.dim = 1536 
        
    model_config.audio_encoder_type = hparams.get('audio_encoder_type')
    if hparams.get('audio_encoder_type') == 'wavlm':
        model_config.audio_encoder_ckpt = hparams.get('audio_encoder_ckpt')
        if 'large' in Path(model_config.audio_encoder_ckpt).stem.lower():
            model_config.audio_encoder_dim = 1024
        elif 'base' in Path(model_config.audio_encoder_ckpt).stem.lower():
            model_config.audio_encoder_dim = 768
            
            
    model = SpkDiarizationE2E(model_config)
        
    return model

@dataclass
class ModelArgs:
    # backbone
    backbone_config: LLaMaModelArgs = None

    # Unet
    unet_dim: int = None
    unet_updown_rates: tuple = (2, 2, 2)
    unet_channel_multiples: tuple = (1, 1, 1)
    unet_kernel_size: int = 3
    unet_constant_channels: bool = False
    unet_use_skip_layer: bool = True
    unet_skip_scale: int = 1

    audio_encoder_type: str = None
    audio_encoder_ckpt: str = None
    audio_encoder_dim: int = None
    audio_encoder_distill: bool = True

    # head
    in_channels: int = 160
    n_class: int = 8

class DistillEncoder(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        self.model = LLaMa(config.backbone_config)
        
        self.stu_in = nn.Linear(config.audio_encoder_dim, 768, bias=False)
        self.student = LLaMa(LLaMaModelArgs(
            dim=768, n_layers=12, n_heads=12, use_causal_attn=False
        ))
        self.stu_out = nn.Linear(768, config.audio_encoder_dim, bias=False)
        self.audio_encoder_proj = nn.Linear(config.audio_encoder_dim, config.backbone_config.dim, bias=False)

    def forward(self, x, attn_mask, condition=None):
        if self.training:
            condition_ = self.stu_out(self.student(self.stu_in(x), attn_mask))
            distill_loss = (((condition_ - condition) ** 2) * attn_mask[..., None]).sum() / attn_mask.sum() / condition.shape[-1]
            condition = (condition_ + condition) / 2
            condition = self.audio_encoder_proj(condition)

            x = x + condition
            x = self.model(x, attn_mask)
            return x, distill_loss
        else:
            condition = self.stu_out(self.student(self.stu_in(x), attn_mask))
            condition = self.audio_encoder_proj(condition)
            x = x + condition
            x = self.model(x, attn_mask)
            return x

class SpkDiarizationE2E(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        if config.audio_encoder_type is None:
            self.mid_net = LLaMa(config.backbone_config)
        else:
            self.mid_net = DistillEncoder(config)

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
            mid_net=self.mid_net,
            use_skip_layer=config.unet_use_skip_layer,
            skip_scale=config.unet_skip_scale
        )

        self.proj_in = nn.Linear(config.in_channels, config.unet_dim)
        self.proj_out = nn.Linear(config.unet_dim, config.n_class)

    def forward(self, x, x_mask, condition=None):
        x = self.proj_in(x)
        unet_stride = int(np.prod(self.config.unet_updown_rates))
        x_mask = x_mask[:, ::unet_stride]

        if self.config.audio_encoder_type is None:
            x = self.unet(
                x,
                mid_kwargs={
                    'attn_mask': x_mask
                }
            )
        else:
            x, skips = self.unet.down(x)
            x = self.unet.mid.pre(x.transpose(1, 2)).transpose(1, 2)
            if self.training and condition is not None:
                x, distill_loss = self.unet.mid.net(x, attn_mask=x_mask, condition=condition)
            else:
                x = self.unet.mid.net(x, attn_mask=x_mask, condition=condition)
            x = self.unet.mid.post(x.transpose(1, 2)).transpose(1, 2)
            x = self.unet.up(x, skips)
        
        x = self.proj_out(x)

        if self.config.audio_encoder_type is not None and self.training and condition is not None:
            return x, distill_loss
        return x




