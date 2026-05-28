import os
import math
import typing as tp
from dataclasses import dataclass, replace
from typing import List, Optional, Union
import json

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
import numpy as np
from hydra.utils import instantiate

from utils.audiotools.ml import BaseModel
from utils.commons.ckpt_utils import repair_unmatched_state_dict

from modules.codec.dac.model.base import CodecMixin
from modules.codec.dac.nn.layers import Snake1d, WNConv1d, WNConvTranspose1d
from modules.codec.fish.modded_dac import (
    init_weights, Transformer, CausalWNConv1d, ResidualUnit, ModelArgs, CausalWNConvTranspose1d
)


"""
VAE adapted from fish codec
"""


def build_wavvae(hparams, init_pretrained=True, verbose=True):
    cfg = json.load(open('modules/vae/wavvae_v5.json'))
    model: torch.nn.Module = instantiate(cfg)
    if init_pretrained:
        from utils.commons.ckpt_utils import torch_load_dist
        pretrained_model_path = hparams.get('pretrained_model_path')
        ckpt = torch_load_dist(pretrained_model_path, mmap=True)
        # load_results = model.load_state_dict(ckpt['state_dict']['model'], strict=False)

        state_dict = ckpt['state_dict']['model']
        state_dict, _, _ = repair_unmatched_state_dict(model.state_dict(), state_dict, silent=not verbose)
        load_results = model.load_state_dict(state_dict, strict=False)

        if verbose:
            print(f"| loaded 'model' from '{pretrained_model_path}'.")
            missing_keys, unexpected_keys = load_results.missing_keys, load_results.unexpected_keys
            print(f"| Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
    return model


def lengths_to_key_mask(x_lens: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Build key-side valid mask from lengths.
    Returns: key_mask [B, 1, 1, L], where True means 'keep' (valid key positions).
    """
    # x_lens: [B], int64
    device = x_lens.device
    B = x_lens.shape[0]
    L = max_len
    ar = torch.arange(L, device=device).view(1, 1, 1, L)  # [1,1,1,L]
    key_valid = ar < x_lens.view(B, 1, 1, 1)              # [B,1,1,L]
    return key_valid


class WindowLimitedTransformer(Transformer):
    """
    Transformer with window limited attention, supports causal and non-causal.
    """

    def __init__(
        self,
        config: ModelArgs,
        input_dim: int = 512,
        window_size: Optional[int] = None,
        causal: bool = True,
        look_ahead_conv: nn.Module = None,
    ):
        super().__init__(config)
        self.window_size = window_size
        self.causal = causal
        self.channels_first = config.channels_first
        self.look_ahead_conv = (
            look_ahead_conv if look_ahead_conv is not None else nn.Identity()
        )
        self.input_proj = (
            nn.Linear(input_dim, config.dim)
            if input_dim != config.dim
            else nn.Identity()
        )
        self.output_proj = (
            nn.Linear(config.dim, input_dim)
            if input_dim != config.dim
            else nn.Identity()
        )

    def _base_window_mask(self, max_length: int) -> torch.Tensor:
        """
        Create a base [L, L] window mask (without lengths).
        """
        L = max_length
        if self.causal:
            # causal lower-triangular + window
            mask = torch.tril(torch.ones(L, L, dtype=torch.bool))
            if self.window_size is not None:
                row = torch.arange(L).view(-1, 1)
                col = torch.arange(L).view(1, -1)
                valid = (col <= row) & (col >= (row - self.window_size + 1))
                mask = mask & valid
        else:
            # non-causal window or full
            row = torch.arange(L).view(-1, 1)
            col = torch.arange(L).view(1, -1)
            if self.window_size is not None:
                mask = (torch.abs(row - col) < self.window_size)
            else:
                mask = torch.ones(L, L, dtype=torch.bool)
        return mask  # [L, L], bool

    def _make_attn_mask(
        self,
        max_length: int,
        x_lens: Optional[torch.Tensor],  # [B], lengths in current time resolution
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build final attention mask [B, 1, L, L] (bool), where True means 'keep'.
        We only mask Key-side padding to avoid rows that are all False.
        """
        base = self._base_window_mask(max_length).to(device)  # [L, L]
        if x_lens is None:
            return base.view(1, 1, max_length, max_length)  # [1,1,L,L]
        # [B,1,1,L]
        key_mask = lengths_to_key_mask(x_lens, max_length).to(device)
        # [B,1,L,L] & [B,1,1,L] -> [B,1,L,L]
        attn_mask = base.view(1, 1, max_length, max_length) & key_mask
        return attn_mask

    def forward(
        self,
        x: Tensor,
        x_lens: Optional[Tensor] = None,  # [B], lengths in samples at current resolution
    ) -> Tensor:
        if self.channels_first:
            x = x.transpose(1, 2)  # (B, T, C)  <- input is (B, C, T)

        x = self.input_proj(x)  # (B, T, D)
        x = self.look_ahead_conv(x)

        B, T, _ = x.shape
        device = x.device
        input_pos = torch.arange(T, device=device)

        # Build batch-wise attention mask that honors lengths
        attn_mask = self._make_attn_mask(T, x_lens, device)

        x = super().forward(x, input_pos, attn_mask)

        x = self.output_proj(x)  # (B, T, D)
        if self.channels_first:
            x = x.transpose(1, 2)  # (B, C, T)
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int = 16,
        stride: int = 1,
        causal: bool = False,
        n_t_layer: int = 0,
        transformer_general_config=None,
    ):
        super().__init__()
        conv_class = CausalWNConv1d if causal else WNConv1d
        transformer_module = (
            nn.Identity()
            if n_t_layer == 0
            else (
                WindowLimitedTransformer(
                    causal=causal,
                    input_dim=dim,
                    window_size=512,
                    config=replace(
                        transformer_general_config,
                        n_layer=n_t_layer,
                        n_head=dim // 64,
                        n_local_heads=-1,
                        dim=dim,
                        intermediate_size=dim * 3,
                    ),
                )
            )
        )
        self.has_transformer = n_t_layer > 0
        self.stride = stride

        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1, causal=causal),
            ResidualUnit(dim // 2, dilation=3, causal=causal),
            ResidualUnit(dim // 2, dilation=9, causal=causal),
            Snake1d(dim // 2),
            conv_class(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )
        self.transformer_module = transformer_module

    def forward(self, x, x_lens=None):
        x = self.block(x)
        if self.has_transformer:
            x = self.transformer_module(x, x_lens)
        if x_lens is not None:
            x_lens = torch.div(x_lens, self.stride, rounding_mode="floor")
        return x, x_lens


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
        n_transformer_layers: list = [0, 0, 4, 4],
        transformer_general_config: ModelArgs = None,
        causal: bool = False,
    ):
        super().__init__()
        conv_class = CausalWNConv1d if causal else WNConv1d
        
        # Create first convolution
        self.first_conv = conv_class(1, d_model, kernel_size=7, padding=3)

        self.block = []
        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride, n_t_layer in zip(strides, n_transformer_layers):
            d_model *= 2
            self.block += [
                EncoderBlock(
                    d_model,
                    stride=stride,
                    causal=causal,
                    n_t_layer=n_t_layer,
                    transformer_general_config=transformer_general_config,
                )
            ]
        self.block = nn.ModuleList(self.block)
        
        # Create last convolution
        self.last_conv = nn.Sequential(
            Snake1d(d_model),
            conv_class(d_model, d_latent, kernel_size=3, padding=1),
        )

        self.enc_dim = d_model

    def forward(self, x, x_lens):
        x = self.first_conv(x)
        for block in self.block:
            x, x_lens = block(x, x_lens)
        x = self.last_conv(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        stride: int = 1,
        causal: bool = False,
        n_t_layer: int = 0,
        transformer_general_config=None,
    ):
        super().__init__()
        conv_trans_class = CausalWNConvTranspose1d if causal else WNConvTranspose1d
        self.block = nn.Sequential(
            Snake1d(input_dim),
            # conv_trans_class(
            #     input_dim,
            #     output_dim,
            #     kernel_size=2 * stride,
            #     stride=stride,
            #     padding=math.ceil(stride / 2),
            # ),
            conv_trans_class(
                input_dim, output_dim,
                kernel_size=2 * stride,      # 10
                stride=stride,               # 5
                padding=math.ceil(stride / 2),   # 3
                output_padding=(2*math.ceil(stride/2) - stride)    # -> 1
            ),
            ResidualUnit(output_dim, dilation=1, causal=causal),
            ResidualUnit(output_dim, dilation=3, causal=causal),
            ResidualUnit(output_dim, dilation=9, causal=causal),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
        causal: bool = False,
        n_transformer_layers: list = [0, 0, 0, 0],
        transformer_general_config=None,
    ):
        super().__init__()
        conv_class = CausalWNConv1d if causal else WNConv1d
        # Add first conv layer
        layers = [conv_class(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, (stride, n_t_layer) in enumerate(zip(rates, n_transformer_layers)):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [
                DecoderBlock(
                    input_dim,
                    output_dim,
                    stride,
                    causal=causal,
                    n_t_layer=n_t_layer,
                    transformer_general_config=transformer_general_config,
                )
            ]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            conv_class(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    

class VAEBottleneck(nn.Module):
    def __init__(
        self, 
        input_dim=1024, 
        z_dim=32, 
        downsample_factor=3,
        pre_module: nn.Module = None, 
        post_module: nn.Module = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.ds = downsample_factor

        self.pre = pre_module if pre_module is not None else nn.Identity()

        # self.down = WNConv1d(input_dim, 2 * z_dim,
        #                      kernel_size=2 * self.ds,
        #                      stride=self.ds,
        #                      padding=math.ceil(self.ds / 2))

        self.down_mu = WNConv1d(input_dim, z_dim, kernel_size=2*self.ds, stride=self.ds, padding=math.ceil(self.ds / 2))
        self.down_lv = WNConv1d(input_dim, z_dim, kernel_size=2*self.ds, stride=self.ds, padding=math.ceil(self.ds / 2))
        nn.init.zeros_(self.down_mu.bias)
        nn.init.constant_(self.down_lv.bias, -2.0)

        # self.up = WNConvTranspose1d(z_dim, input_dim,
        #                             kernel_size=2 * self.ds,
        #                             stride=self.ds,
        #                             padding=math.ceil(self.ds / 2))
        self.up = WNConvTranspose1d(
            z_dim, input_dim,
            kernel_size=2 * self.ds,     # ds=3 -> kernel=6
            stride=self.ds,              # 3
            padding=math.ceil(self.ds / 2),  # 2
            output_padding=(2*math.ceil(self.ds/2) - self.ds)  # -> 1
        )

        self.post = post_module if post_module is not None else nn.Identity()

        self.adapter = nn.Sequential(
            nn.LayerNorm(self.input_dim),  # 假设张量是 [B,C,T]，注意重排
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=1)
        )

    def encode(self, x):
        # x: [B, C_in=1024, T_enc]
        h = self.pre(x)
        # stats = self.down(h)  # [B, 2*z_dim, T_code]
        # mu, logvar = stats.chunk(2, dim=1)
        # logvar = logvar.clamp(-30.0, 20.0)
        mu = self.down_mu(h)
        logvar = self.down_lv(h).clamp(-30.0, 20.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * eps  # [B, z_dim, T_code]
        return z

    def decode_to_latent(self, z):
        h = self.up(z)                    # [B, input_dim, T_enc]
        h = self.post(h)
        h = self.adapter[0](h.permute(0, 2, 1)).permute(0, 2, 1)         # LN over C
        h = self.adapter[1](h)         # 1x1 conv
        return h

    def forward(self, x, kl_weight=1.0):
        mu, logvar = self.encode(x)     # [B, C, T]
        z = self.reparameterize(mu, logvar)
        x_rec_latent = self.decode_to_latent(z)
        # KL loss
        with torch.cuda.amp.autocast(enabled=False):
            # kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean() * mu.shape[1]
        return x_rec_latent, z, mu, logvar, kl * kl_weight


class WavVAE(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 5],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [5, 8, 4, 2],
        bottleneck: torch.nn.Module = None,
        sample_rate: int = 24000,
        causal: bool = False,
        encoder_transformer_layers: List[int] = [0, 0, 0, 0],
        decoder_transformer_layers: List[int] = [0, 0, 0, 0],
        transformer_general_config=None,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(
            encoder_dim,
            encoder_rates,
            latent_dim,
            causal=causal,
            n_transformer_layers=encoder_transformer_layers,
            transformer_general_config=transformer_general_config,
        )

        self.bottleneck = bottleneck

        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            causal=causal,
            n_transformer_layers=decoder_transformer_layers,
            transformer_general_config=transformer_general_config,
        )
        self.sample_rate = sample_rate
        self.apply(init_weights)

        self.frame_length = self.hop_length * 3

    def preprocess(self, audio_data, audio_lengths=None):
        if audio_data.ndim == 2:
            audio_data = audio_data.unsqueeze(1)
        if audio_lengths is None:
            audio_lengths = torch.full((audio_data.shape[0],), audio_data.shape[-1], device=audio_data.device, dtype=torch.long)
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.frame_length) * self.frame_length - length
        if right_pad > 0:
            audio_data = nn.functional.pad(audio_data, (0, right_pad))
        audio_lengths[audio_lengths == length] = audio_data.shape[-1]
        return audio_data, audio_lengths

    def encode(
        self,
        audio_data: torch.Tensor,
        audio_lengths: torch.Tensor = None,
    ):
        audio_data, audio_lengths = self.preprocess(audio_data, audio_lengths)

        z_enc = self.encoder(audio_data, audio_lengths)                # [B, 1024, T_enc]
        x_rec_latent, z, mu, logvar, kl = self.bottleneck(z_enc)
        T_code = z.shape[-1]

        ret = {
            "z": z, 
            "mu": mu, 
            "logvar": logvar, 
            "kl": kl,
            "latents_for_decoder": x_rec_latent, 
        }

        return ret

    def encode_latent(self, audio_data, audio_lengths=None):
        audio_data, audio_lengths = self.preprocess(audio_data, audio_lengths)
        z_enc = self.encoder(audio_data, audio_lengths)                # [B, 1024, T_enc]
        mu, logvar = self.bottleneck.encode(z_enc)
        z = self.bottleneck.reparameterize(mu, logvar)
        return z.permute(0, 2, 1)   # [B, T, C]
        
    def decode(self, latent):
        z_latent = self.bottleneck.decode_to_latent(latent.permute(0, 2, 1))  # [B, 1024, T_enc]
        return self.decoder(z_latent)

    def forward(
        self,
        audio_data: torch.Tensor,
        audio_lengths=None,
        **kwargs,
    ):
        audio_data, audio_lengths = self.preprocess(audio_data, audio_lengths)
        enc = self.encode(audio_data, audio_lengths)
        x = self.decoder(enc["latents_for_decoder"])
        enc['recon'] = x
        return enc


if __name__ == '__main__':
    model = build_wavvae({'pretrained_model_path': 'pretrained_models/fish_codec/model.ckpt'})

    audio_data = torch.randn(1, 24000 * 10)
    model_outputs = model(audio_data)

    for k in model_outputs.keys():
        print(k, model_outputs[k].shape)
