import argparse
from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from utils.nn.seq_utils import sequence_mask

from modules.tts.wavvae.decoder.diag_gaussian import DiagonalGaussianDistribution
from modules.tts.wavvae.decoder.latent2wav.modules import Generator, Upsample
from modules.tts.wavvae.encoder.common_modules import SEANetEncoder
from modules.vae.wavvae_v5 import WindowLimitedTransformer, lengths_to_key_mask
from modules.codec.fish.modded_dac import ModelArgs as TransformerArgs

"""
VAE adapted from wavvae_v3, optimized, add transformers
"""

def build_wavvae(hparams, init_pretrained=False):
    model = WavVAE_V8(hparams=hparams)
    if init_pretrained:
        from utils.commons.ckpt_utils import load_ckpt
        load_ckpt_gen = hparams.get('load_ckpt_gen', './checkpoints/1231_megatts3_wavvae_v3_25hz_kl001_fix4')
        if '1117_melgan-nsf_full_1' in load_ckpt_gen:
            load_ckpt(model.decoder, load_ckpt_gen, 'model_gen', force=True, strict=True)
        else:
            load_ckpt(model, load_ckpt_gen, 'model_gen', strict=False)
    return model

class WavVAE_V8(nn.Module):
    def __init__(self, kp_nums=68, kp_channels=2,
                 latent_channels=64, hidden_size=512, use_firstframe_cond=False,
                 hparams=None):
        super().__init__()
        self.encoder = SEANetEncoder(causal=False, n_residual_layers=1, norm='weight_norm', pad_mode='reflect', lstm=2,
                                dimension=512, channels=1, n_filters=32, ratios=[6, 5, 4, 4, 2], activation='ELU',
                                kernel_size=7, residual_kernel_size=3, last_kernel_size=7, dilation_base=2,
                                true_skip=False, compress=2)
        # self.transformer = WindowLimitedTransformer(
        #     config=TransformerArgs(
        #         block_size=4096,
        #         n_layer=8,
        #         n_head=8,
        #         dim=1024,
        #         intermediate_size=3072,
        #         n_local_heads=-1,
        #         head_dim=64,
        #         rope_base=10000,
        #         norm_eps=1e-05,
        #         dropout_rate=0.1,
        #         attn_dropout_rate=0.1,
        #         channels_first=False
        #     ),
        #     input_dim=1024,
        #     window_size=128,
        #     causal=False
        # )
        self.proj_to_z = nn.Linear(512, 64)
        self.proj_to_decoder = nn.Linear(32, 320)

        config_path = hparams['melgan_config']
        args = argparse.Namespace()
        args.__dict__.update(config_path)
        self.latent_upsampler = Upsample(320, 4)
        self.decoder = Generator(
            input_size_=160, ngf=128, n_residual_layers=4,
            num_band=1, args=args, ratios=[5,4,4,3])

    def encode_latent(self, audio, audio_lens=None):
        posterior = self.encode(audio, audio_lens)
        latent = posterior.sample().permute(0, 2, 1)  # (b,t,latent_channel)
        return latent

    def encode(self, audio, audio_lens=None):
        if audio_lens is None:
            audio_lens = torch.ones(audio.shape[0], device=audio.device, dtype=torch.long) * audio.shape[1]
        x = self.encoder(audio.unsqueeze(1)).permute(0, 2, 1)   # [B, C, T] -> [B, T, C]
        audio_lens = audio_lens // 960
        audio_lens[audio_lens > x.shape[1]] = x.shape[1]
        # x = self.transformer(x, audio_lens)
        x = self.proj_to_z(x).permute(0, 2, 1)  # [B, C, T]
        poseterior = DiagonalGaussianDistribution(x)
        return poseterior

    def decode(self, latent):
        latent = self.proj_to_decoder(latent).permute(0, 2, 1)
        return self.decoder(self.latent_upsampler(latent))

    def forward(self, audio, audio_lens=None):
        if audio_lens is None:
            audio_lens = torch.ones(audio.shape[0], device=audio.device, dtype=torch.long) * audio.shape[1]
        posterior = self.encode(audio, audio_lens)
        latent = posterior.sample().permute(0, 2, 1)  # (b, t, latent_channel)
        recon_wav = self.decode(latent)

        return recon_wav, posterior

        kl = posterior.kl().transpose(1, 2)     # [B, T, C]
        latent_lens = audio_lens // 960
        latent_lens[latent_lens > latent.shape[1]] = latent.shape[1]
        kl_mask = sequence_mask(latent_lens)[..., None].to(kl)
        kl = (kl * kl_mask).sum() / kl_mask.sum() / kl.shape[-1]  # [B, T, C] -> [B]

        # mu = posterior.mean.permute(0, 2, 1)
        # logvar = posterior.logvar.permute(0, 2, 1)
        # print(f"{mu.mean().item() = } {logvar.mean().item() = }")

        outputs = {
            'z': latent.permute(0, 2, 1),
            'mu': posterior.mean.permute(0, 2, 1),
            'logvar': posterior.logvar.permute(0, 2, 1),
            'kl': kl,
            'recon': recon_wav
        }

        return outputs
    

if __name__ == '__main__':
    from utils.commons.hparams import hparams, set_hparams
    set_hparams('/mnt/bn/sa-ag-data/jiangziyue/MegaHuman/egs/tts/wavvae3.yaml')
    wavvae_v3 = WavVAE_V3(hparams=hparams)
    a = torch.ones(3, 23040)
    recon_wav, posterior = wavvae_v3(a)
    print(recon_wav.shape)
    print(posterior.kl().shape)
