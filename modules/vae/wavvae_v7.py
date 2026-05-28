import os
import math
import json
from typing import List

import torch
import torch.nn as nn
from hydra.utils import instantiate
import numpy as np

from modules.codec.dac.nn.layers import Snake1d, WNConv1d, WNConvTranspose1d
from modules.codec.dac.model.dac import Encoder, init_weights, ResidualUnit
from modules.vae.wavvae_v5 import WindowLimitedTransformer, lengths_to_key_mask
from modules.codec.fish.modded_dac import ModelArgs as TransformerArgs

from utils.commons.ckpt_utils import repair_unmatched_state_dict
from utils.nn.seq_utils import sequence_mask

"""
VAE adapted from DAC
"""

def build_wavvae(hparams, init_pretrained=True, verbose=True):
    cfg = json.load(open('modules/vae/wavvae_v7.json'))
    
    cfg['bottleneck']['logvar_min'] = hparams.get('logvar_min', -30)
    cfg['bottleneck']['logvar_max'] = hparams.get('logvar_max', 20)
    cfg['bottleneck']['logvar_init'] = hparams.get('logvar_init', 0.0)
    cfg['bottleneck']['mu_init'] = hparams.get('mu_init', 0.0)
    cfg['bottleneck']['transformer_pre_config']['n_layer'] = hparams.get('bottleneck_down_layers', 8)
    cfg['bottleneck']['transformer_post_config']['n_layer'] = hparams.get('bottleneck_up_layers', 8)
    cfg['bottleneck']['downsample_factor'] = 3


    model: torch.nn.Module = instantiate(cfg)
    if init_pretrained:
        from utils.commons.ckpt_utils import torch_load_dist
        pretrained_model_path = hparams.get('pretrained_model_path')
        ckpt = torch_load_dist(pretrained_model_path, mmap=True)
        # load_results = model.load_state_dict(ckpt['state_dict']['model'], strict=False)

        state_dict = ckpt['state_dict']
        state_dict, _, _ = repair_unmatched_state_dict(model.state_dict(), state_dict, silent=not verbose)
        load_results = model.load_state_dict(state_dict, strict=False)

        if verbose:
            print(f"| loaded 'model' from '{pretrained_model_path}'.")
            missing_keys, unexpected_keys = load_results.missing_keys, load_results.unexpected_keys
            print(f"| Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
    return model

class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
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
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
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
        transformer_pre_config=None,
        transformer_post_config=None,
        logvar_min=-30,
        logvar_max=20,
        logvar_init=0.0,
        mu_init=0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.ds = downsample_factor
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.logvar_init = logvar_init
        self.mu_init = mu_init

        self.pre = WindowLimitedTransformer(
            config=transformer_pre_config,
            input_dim=input_dim,
            window_size=128,
            causal=False,
        )

        self.down_mu = WNConv1d(input_dim, z_dim, kernel_size=2*self.ds, stride=self.ds, padding=math.ceil(self.ds / 2))
        self.down_lv = WNConv1d(input_dim, z_dim, kernel_size=2*self.ds, stride=self.ds, padding=math.ceil(self.ds / 2))
        nn.init.constant_(self.down_mu.bias, self.mu_init)
        nn.init.constant_(self.down_lv.bias, self.logvar_init)

        self.up = WNConvTranspose1d(
            z_dim, input_dim,
            kernel_size=2 * self.ds,     # ds=3 -> kernel=6
            stride=self.ds,              # 3
            padding=math.ceil(self.ds / 2),  # 2
            output_padding=(2*math.ceil(self.ds/2) - self.ds)  # -> 1
        )

        self.post = WindowLimitedTransformer(
            config=transformer_post_config,
            input_dim=input_dim,
            window_size=128,
            causal=False,
        )

        self.adapter = nn.Sequential(
            nn.LayerNorm(self.input_dim),  # 假设张量是 [B,C,T]，注意重排
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=1)
        )

    def encode(self, x, x_lens):
        # x: [B, C_in=1024, T_enc]
        h = self.pre(x, x_lens)
        # stats = self.down(h)  # [B, 2*z_dim, T_code]
        # mu, logvar = stats.chunk(2, dim=1)
        mu = self.down_mu(h)
        logvar = self.down_lv(h).clamp(self.logvar_min, self.logvar_max)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * eps  # [B, z_dim, T_code]
        return z

    def decode_to_latent(self, z, z_lens):
        h = self.up(z)                    # [B, input_dim, T_enc]
        h = self.post(h, z_lens * self.ds)
        h = self.adapter[0](h.permute(0, 2, 1)).permute(0, 2, 1)         # LN over C
        h = self.adapter[1](h)         # 1x1 conv
        return h

    def forward(self, x, x_lens, kl_weight=1.0):
        mu, logvar = self.encode(x, x_lens)
        z = self.reparameterize(mu, logvar)
        x_rec_latent = self.decode_to_latent(z, x_lens)

        # KL loss
        with torch.cuda.amp.autocast(enabled=False):
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())     # [B, C, T]
            kl = kl.transpose(1, 2) # [B, T, C]
            kl_mask = sequence_mask(x_lens // self.ds)[..., None].to(kl)
            kl = (kl * kl_mask).sum() / kl_mask.sum() / kl.shape[-1]

        return x_rec_latent, z, mu, logvar, kl * kl_weight

class WavVAE(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 5, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 5, 4, 2],
        bottleneck: torch.nn.Module = None,
        sample_rate: int = 24000,
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
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)

        self.bottleneck = bottleneck

        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
        )
        self.sample_rate = sample_rate
        
        self.apply(init_weights)
        nn.init.constant_(self.bottleneck.down_mu.bias, self.bottleneck.mu_init)
        nn.init.constant_(self.bottleneck.down_lv.bias, self.bottleneck.logvar_init)

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

        z_enc = self.encoder(audio_data)                # [B, 1024, T_enc]
        z_lens = audio_lengths // self.hop_length
        x_rec_latent, z, mu, logvar, kl = self.bottleneck(z_enc, z_lens)

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
        z_enc = self.encoder(audio_data)                # [B, 1024, T_enc]
        z_lens = audio_lengths // self.hop_length
        mu, logvar = self.bottleneck.encode(z_enc, z_lens)
        z = self.bottleneck.reparameterize(mu, logvar)
        return z.permute(0, 2, 1)   # [B, T, C]
        
    def decode(self, latent, z_lens=None):
        # latent [B, T, C]
        if z_lens is None:
            z_lens = torch.full((latent.shape[0],), latent.shape[1], device=latent.device, dtype=torch.long)
        z_latent = self.bottleneck.decode_to_latent(latent.permute(0, 2, 1), z_lens)  # [B, 1024, T_enc]
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
    model = build_wavvae({'pretrained_model_path': 'pretrained_models/dac_24k/weights_24khz.pth'}, init_pretrained=True)

    audio_data = torch.randn(1, 24000 * 10 + 8 * 240)

    print(f"{audio_data.shape = }")

    with torch.no_grad():
        model_outputs = model(audio_data)

    for k in model_outputs.keys():
        print(k, model_outputs[k].shape)

