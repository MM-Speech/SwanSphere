import librosa
import argparse
import torch
from functools import partial

from modules.tts.megatts3.latent2wav.modules import Generator
from utils.audio import torch_wav2spec


class WavVAEGenerator(torch.nn.Module):
    def __init__(self, latent_dim, hparams, left_pad=False):
        super().__init__()
        self.hparams = hparams
        self.latent_dim = latent_dim
        config_path = hparams['melgan_config']
        args = argparse.Namespace()
        args.__dict__.update(config_path)

        # Online load mel with GPU
        sample_rate = hparams["audio_sample_rate"]
        fft_size = hparams["win_size"]
        win_size = hparams["win_size"]
        hop_size = hparams["hop_size"]
        num_mels = hparams["audio_num_mel_bins"]
        fmin = hparams["fmin"]
        fmax = hparams["fmax"]
        mel_basis = librosa.filters.mel(
            sr=sample_rate, n_fft=fft_size, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        self.torch_wav2spec_ = partial(
            torch_wav2spec, mel_basis=mel_basis, fft_size=fft_size, hop_size=hop_size, win_length=win_size,
        )
        if left_pad:
            from modules.tts.megatts3.stable_diffusion.sd_vae_left_conv import SDEncoder, SDDecoder
            self.encoder = SDEncoder(
                ch=320, out_ch=3, ch_mult=(1, 2, 4), num_res_blocks=2,
                dropout=0.0, resamp_with_conv=True, in_channels=160, attn_levels=[],
                z_channels=self.latent_dim, double_z=True, use_linear_attn=False, attn_type="vanilla")
            self.latent2mel = SDDecoder(
                ch=320, out_ch=160, ch_mult=(1, 2, 4), num_res_blocks=2,
                attn_levels=[], dropout=0.0, resamp_with_conv=True,
                z_channels=self.latent_dim, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                attn_type="vanilla")
        else:
            from modules.tts.megatts3.stable_diffusion.sd_vae import SDEncoder, SDDecoder
            self.encoder = SDEncoder(
                ch=256, out_ch=3, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
                dropout=0.0, resamp_with_conv=True, in_channels=160, attn_levels=[],
                z_channels=latent_dim, double_z=True, use_linear_attn=False, attn_type="vanilla")
            self.latent2mel = SDDecoder(
                ch=256, out_ch=160, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
                attn_levels=[], dropout=0.0, resamp_with_conv=True,
                z_channels=latent_dim, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                attn_type="vanilla")
        self.mel2wav = Generator(
            input_size_=160, ngf=128, n_residual_layers=4,
            num_band=1, args=args, ratios=[5,4,4,3])
    
    def wav2mel(self, wav):
        mels = self.torch_wav2spec_(wav)
        return mels
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def mel2latent(self, mels):
        mels = mels[:, :mels.shape[1] // self.latent_dim * self.latent_dim]
        h = self.encoder(mels.transpose(1, 2))
        mu, logvar = h[:, :self.latent_dim], h[:, self.latent_dim:]
        z = self.reparameterize(mu, logvar)
        return z.transpose(1, 2)

    def latent2wav(self, latent):
        wav_out = self.mel2wav(self.latent2mel(latent))
        return wav_out

