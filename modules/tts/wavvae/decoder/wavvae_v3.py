import argparse
import torch
from torch import nn
import torch.nn.functional as F

from modules.tts.wavvae.decoder.diag_gaussian import DiagonalGaussianDistribution
from modules.tts.wavvae.decoder.feature_extractors import EncodecFeatures
from modules.tts.wavvae.decoder.latent2wav.modules import Generator, Upsample

class WavVAE_V3(nn.Module):
    def __init__(self, kp_nums=68, kp_channels=2,
                 latent_channels=64, hidden_size=512, use_firstframe_cond=False,
                 hparams=None):
        super().__init__()
        self.encoder = EncodecFeatures(dowmsamples=[6, 5, 4, 4, 2])
        self.proj_to_z = nn.Linear(512, 64)
        self.proj_to_decoder = nn.Linear(32, 320)

        config_path = hparams['melgan_config']
        args = argparse.Namespace()
        args.__dict__.update(config_path)
        self.latent_upsampler = Upsample(320, 4)
        self.decoder = Generator(
            input_size_=160, ngf=128, n_residual_layers=4,
            num_band=1, args=args, ratios=[5,4,4,3])

    def encode_latent(self, audio):
        posterior = self.encode(audio)
        latent = posterior.sample().permute(0, 2, 1)  # (b,t,latent_channel)
        return latent

    def encode(self, audio):
        x = self.encoder(audio).permute(0, 2, 1)
        x = self.proj_to_z(x).permute(0, 2, 1)
        poseterior = DiagonalGaussianDistribution(x)
        return poseterior

    def decode(self, latent):
        latent = self.proj_to_decoder(latent).permute(0, 2, 1)
        return self.decoder(self.latent_upsampler(latent))

    def forward(self, audio):
        posterior = self.encode(audio)
        latent = posterior.sample().permute(0, 2, 1)  # (b, t, latent_channel)
        recon_wav = self.decode(latent)
        return recon_wav, posterior
    

if __name__ == '__main__':
    from utils.commons.hparams import hparams, set_hparams
    set_hparams('/mnt/bn/sa-ag-data/jiangziyue/MegaHuman/egs/tts/wavvae3.yaml')
    wavvae_v3 = WavVAE_V3(hparams=hparams)
    a = torch.ones(3, 23040)
    recon_wav, posterior = wavvae_v3(a)
    print(recon_wav.shape)
    print(posterior.kl().shape)
