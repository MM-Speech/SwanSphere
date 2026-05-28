import os
import math
import json

import torch
import torch.nn as nn
# from transformers import EncodecConfig
from hydra.utils import instantiate

from modules.codec.dac.nn.layers import Snake1d, WNConv1d, WNConvTranspose1d
from modules.codec.encodec.modeling_encodec import EncodecEncoder, EncodecDecoder, EncodecModel
from modules.codec.encodec.configuration_encodec import EncodecConfig
from modules.vae.wavvae_v5 import VAEBottleneck

from utils.commons.ckpt_utils import repair_unmatched_state_dict

"""
VAE adapted from encodec
"""

def build_wavvae(hparams, init_pretrained=True, verbose=True):
    cfg = json.load(open('modules/vae/wavvae_v6.json'))
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

class WavVAE(nn.Module):
    def __init__(self, config: EncodecConfig, ds=3, bottleneck=None):
        super().__init__()
        self.config = config
        self.encoder = EncodecEncoder(config)
        self.decoder = EncodecDecoder(config)

        self.hidden_size = config.hidden_size  # 128 for 24k
        self.frame_rate_enc = int(round(config.sampling_rate / math.prod(config.upsampling_ratios)))  # 75
        assert self.frame_rate_enc == 75, "Expect 24kHz EnCodec-24 config."

        # 75 Hz -> 25 Hz
        self.bottleneck = bottleneck
        self.ds = ds
        self.frame_length = self.frame_rate_enc // self.ds

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
    model = build_wavvae({'pretrained_model_path': 'pretrained_models/encodec_24k/model.ckpt'}, init_pretrained=False)

    audio_data = torch.randn(1, 24000 * 10 + 8 * 240)

    print(f"{audio_data.shape = }")

    with torch.no_grad():
        model_outputs = model(audio_data)

    for k in model_outputs.keys():
        print(k, model_outputs[k].shape)

