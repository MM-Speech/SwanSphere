import torch
from torch import nn
import torch.nn.functional as F

from modules.tts.wavvae.decoder.diag_gaussian import DiagonalGaussianDistribution
from modules.tts.wavvae.decoder.feature_extractors import EncodecFeatures
from modules.tts.wavvae.decoder.models import VocosBackbone
from modules.tts.wavvae.decoder.heads import ISTFTHead
from modules.tts.ar_dur.commons.align_ops import expand_states


class WavVAE_V2(nn.Module):
    def __init__(self, kp_nums=68, kp_channels=2,
                 latent_channels=64, hidden_size=512, use_firstframe_cond=False,
                 use_content=False):
        super().__init__()
        self.encoder = EncodecFeatures(dowmsamples=[6, 5, 4, 4, 2])
        self.decoder = VocosBackbone(512, 768, 2304, 12)
        self.istft_head = ISTFTHead(768, 2400, 960)
        self.proj_to_z = nn.Linear(512, 64)
        self.proj_to_decoder = nn.Linear(32, 512)

        self.use_content = use_content
        if self.use_content:
            from modules.tts.ar_dur.commons.layers import Embedding
            from modules.tts.ar_dur.commons.nar_tts_modules import PosEmb
            from modules.tts.ar_dur.commons.rel_transformer import RelTransformerEncoder
            self.ph_encoder = RelTransformerEncoder(
                500, hidden_size, hidden_size,
                hidden_size * 2, 4, 6,
                3, 0.0, prenet=True, pre_ln=True)
            self.tone_embed = Embedding(100 + 3, hidden_size, padding_idx=0)
            self.ph_pos_embed = PosEmb(hidden_size)
            self.ling_pre_net = torch.nn.Sequential(*[
                torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=s * 2, stride=s, padding=s // 2)
                for i, s in enumerate([2, 2])
            ])
            self.timbre_encoder = EncodecFeatures(dowmsamples=[6, 5, 4, 4, 2])

    def encode_latent(self, audio):
        posterior = self.encode(audio)
        latent = posterior.sample().permute(0, 2, 1)  # (b,t,latent_channel)
        return latent

    def decode_latent(self, latent, txt_tokens=None, tone_tokens=None, mel2ph=None, wavs_timbre=None):
        return self.istft_head(self.decode(latent, txt_tokens, tone_tokens, mel2ph, wavs_timbre))
    
    def encode(self, audio):
        x = self.encoder(audio).permute(0, 2, 1)
        x = self.proj_to_z(x).permute(0, 2, 1)
        poseterior = DiagonalGaussianDistribution(x)
        return poseterior

    def decode(self, latent, txt_tokens=None, tone_tokens=None, mel2ph=None, wavs_timbre=None):
        latent = self.proj_to_decoder(latent).permute(0, 2, 1)
        if self.use_content:
            cond_feat = self.forward_ling_encoder(txt_tokens, tone_tokens, wavs_timbre)
            cond_feat = self.ling_pre_net(expand_states(cond_feat, mel2ph).transpose(1, 2))
            latent = latent + cond_feat
        return self.decoder(latent)

    def forward(self, audio, txt_tokens=None, tone_tokens=None, mel2ph=None, wavs_timbre=None):
        posterior = self.encode(audio)
        latent = posterior.sample().permute(0, 2, 1)  # (b, t, latent_channel)
        decoded_spec = self.decode(latent, txt_tokens, tone_tokens, mel2ph, wavs_timbre)
        recon_wav = self.istft_head(decoded_spec)
        return recon_wav, posterior
    
    def forward_ling_encoder(self, txt_tokens, tone_tokens, wavs_timbre):
        ph_tokens = txt_tokens
        ph_nonpadding = (ph_tokens > 0).float()[:, :, None]  # [B, T_phone, 1]
        x_spk = self.timbre_encoder(wavs_timbre).mean(dim=2)[:, None, :]

        # enc_ph
        ph_enc_oembed = self.tone_embed(tone_tokens)
        ph_enc_oembed = ph_enc_oembed + self.ph_pos_embed(
            torch.arange(0, ph_tokens.shape[1])[None,].to(ph_tokens.device))
        ph_enc_oembed = ph_enc_oembed + x_spk
        ph_enc_oembed = ph_enc_oembed * ph_nonpadding
        x_ling = self.ph_encoder(ph_tokens, other_embeds=ph_enc_oembed)
        x_ling = (x_ling + x_spk) * ph_nonpadding
        return x_ling


if __name__ == '__main__':
    wavvae_v2 = WavVAE_V2()
    a = torch.ones(1, 23041)
    recon_wav, posterior = wavvae_v2(a)
    print(recon_wav.shape)
    print(posterior.kl().shape)
