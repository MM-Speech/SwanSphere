from typing import List

import torch
import torchaudio
from torch import nn
import math
from modules.tts.wavvae.encoder.common_modules import SEANetEncoder
from modules.tts.wavvae.decoder.models import VocosBackbone
from modules.tts.wavvae.decoder.heads import ISTFTHead

class EncodecFeatures(nn.Module):
    def __init__(
        self,
        encodec_model: str = "encodec_24khz",
        dowmsamples: List[int] = [6, 5, 5, 4, 2],
        vq_bins: int = 16384,
        vq_kmeans: int = 800,
    ):
        super().__init__()

        # breakpoint()
        self.frame_rate = 25  # not use
        self.encoder = SEANetEncoder(causal=False, n_residual_layers=1, norm='weight_norm', pad_mode='reflect', lstm=2,
                                dimension=512, channels=1, n_filters=32, ratios=dowmsamples, activation='ELU',
                                kernel_size=7, residual_kernel_size=3, last_kernel_size=7, dilation_base=2,
                                true_skip=False, compress=2)

    def forward(self, audio: torch.Tensor):
        audio = audio.unsqueeze(1)                  # audio(16,24000)
        emb = self.encoder(audio)
        return emb

if __name__ == '__main__':
    encodec_model = EncodecFeatures()
    vocos_decoder = VocosBackbone(512, 768, 2304, 12)
    istft_head = ISTFTHead(768, 2400, 1200)

    a = torch.ones(1, 24000)
    emb = encodec_model(a)
    print(emb.shape)
    decoded = vocos_decoder(emb)
    print(decoded.shape)
    recon_wav = istft_head(decoded)
    print(recon_wav.shape)
