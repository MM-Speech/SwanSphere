from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg
import numpy as np

from modules.tts.wavvae.encoder.common_modules import SEANetEncoder
from modules.commons.conv import ConvBlocks
from modules.commons.rel_transformer import RelTransformerEncoder

class SpkEmbed(nn.Module):

    def __init__(self, n_mels=160, hidden_size=1024, wav_dowmsamples: List[int] = [6, 5, 4, 2]):
        super().__init__()

        self.wav_conv = SEANetEncoder(
            causal=False, n_residual_layers=1, norm='weight_norm', pad_mode='reflect', lstm=False,
            dimension=512, channels=1, n_filters=32, ratios=wav_dowmsamples, activation='ELU',
            kernel_size=7, residual_kernel_size=3, last_kernel_size=7, dilation_base=2,
            true_skip=False, compress=2
        )

        self.mel_conv = ConvBlocks(
            c_in=n_mels, hidden_size=512, out_dims=512, dilations=1, kernel_size=3, num_layers=2
        )

        self.x_proj = nn.Linear(2 * 512, hidden_size, bias=False)

        self.encoder = RelTransformerEncoder(
            n_vocab=-1, out_channels=hidden_size, hidden_channels=hidden_size,
            filter_channels=hidden_size, n_heads=8, n_layers=4,
            kernel_size=3, prenet=True, pre_ln=True)

        self.attn_pooling = nn.Linear(hidden_size, 1, bias=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, wavs, mels):
        x = self.encode(wavs, mels)
        return x, self.logit_scale.exp()
    
    def encode(self, wavs, mels):
        wavs = self.wav_conv(wavs.unsqueeze(1)).transpose(1, 2)
        mels = self.mel_conv(mels)
        x = self.x_proj(torch.cat([wavs, mels], dim=-1))
        x = self.encoder(x)

        x = x * torch.softmax(self.attn_pooling(x), dim=1)
        x = x.sum(1)    # [B, C]
        x = F.normalize(x, dim=-1)
        return x
