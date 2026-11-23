"""Frozen original Stable Audio stereo VAE applied independently to two FOA pairs."""

import json
import math
from pathlib import Path

import torch
from torch import nn


def sample_posterior(mean, std, use_mean=False, generator=None):
    """Sample packed stereo-pair statistics with the original Oobleck RNG layout.

    Both statistics are [B, 128, T]. Noise is drawn as [2B, 64, T], exactly
    as for the shared stereo encoder's posterior before packing the pairs.
    Callers preserve their microbatch sizes when comparing RNG streams.
    """
    if mean.ndim != 3 or mean.shape[1] != 128 or mean.shape[0] < 1 or mean.shape[-1] < 1:
        raise ValueError(f"Expected posterior mean [B, 128, T], got {tuple(mean.shape)}")
    if std.shape != mean.shape or std.device != mean.device or std.dtype != mean.dtype:
        raise ValueError("Posterior mean/std must have identical shape, device and dtype")
    if not mean.is_floating_point():
        raise ValueError("Posterior mean/std must be floating-point tensors")
    if use_mean:
        return mean
    from diffusers.utils.torch_utils import randn_tensor

    batch, _, frames = mean.shape
    noise = randn_tensor(
        (batch * 2, 64, frames), generator=generator,
        device=mean.device, dtype=mean.dtype,
    ).reshape_as(mean)
    return mean + std * noise


class StableAudioPairCodec(nn.Module):
    """Preserve file channel order: [0, 1] -> 64 latents, [2, 3] -> 64.

    The same original stereo VAE handles both pairs. The 128 latent channels
    are concatenated in pair order; audio channels are never remapped.
    Callers control waveform padding/cropping and latent normalization.
    """

    sample_rate = 44100
    downsample_factor = 2048
    latent_dim = 128
    pair_latent_dim = 64
    audio_channels = 4

    def __init__(self, model_dir):
        super().__init__()
        self.model_dir = str(Path(model_dir).expanduser().resolve())
        with (Path(self.model_dir) / "config.json").open(encoding="utf-8") as handle:
            config = json.load(handle)
        expected = {
            "_class_name": "AutoencoderOobleck",
            "audio_channels": 2,
            "decoder_input_channels": self.pair_latent_dim,
            "sampling_rate": self.sample_rate,
        }
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(
                    f"Original Stable Audio stereo VAE requires {key}={value!r}, "
                    f"got {config.get(key)!r}: {self.model_dir}"
                )
        ratios = config.get("downsampling_ratios", ())
        if not ratios or math.prod(ratios) != self.downsample_factor:
            raise ValueError(
                f"Expected Stable Audio downsampling factor {self.downsample_factor}, "
                f"got {ratios}: {self.model_dir}"
            )
        from diffusers import AutoencoderOobleck

        self.autoencoder = AutoencoderOobleck.from_pretrained(
            self.model_dir, local_files_only=True
        )
        self.autoencoder.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        # This codec is a fixed target space, including when its parent trains.
        return super().train(False)

    @torch.no_grad()
    def encode_posterior(self, audio):
        """Return actual posterior mean/std, packed as two 64-D channel groups.

        Oobleck stores a raw scale parameter internally; its standard deviation
        is softplus(scale) + 1e-4. Save posterior.std itself, not scale/logvar.
        """
        if audio.ndim != 3 or audio.shape[1] != self.audio_channels:
            raise ValueError(f"Expected audio [B, 4, L], got {tuple(audio.shape)}")
        if audio.shape[0] < 1 or audio.shape[-1] < self.downsample_factor:
            raise ValueError("Stable Audio encoding requires a nonempty batch and at least 2048 samples")
        batch, _, length = audio.shape
        pairs = audio.reshape(batch * 2, 2, length)
        posterior = self.autoencoder.encode(pairs).latent_dist
        mean, std = posterior.mean, posterior.std
        if mean.ndim != 3 or mean.shape[:2] != (batch * 2, self.pair_latent_dim):
            raise ValueError(f"Expected stereo posterior [2B, 64, T], got {tuple(mean.shape)}")
        if std.shape != mean.shape:
            raise ValueError("Stereo posterior mean/std shapes differ")
        return (
            mean.reshape(batch, self.latent_dim, mean.shape[-1]),
            std.reshape(batch, self.latent_dim, std.shape[-1]),
        )

    @torch.no_grad()
    def encode(self, audio, use_mean=False, generator=None):
        mean, std = self.encode_posterior(audio)
        return sample_posterior(mean, std, use_mean=use_mean, generator=generator)

    @torch.no_grad()
    def decode(self, latent):
        if latent.ndim != 3 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"Expected paired latents [B, 128, T], got {tuple(latent.shape)}")
        if latent.shape[0] < 1 or latent.shape[-1] < 1:
            raise ValueError("Stable Audio decoding requires a nonempty batch and latent sequence")
        batch, _, frames = latent.shape
        pairs = latent.reshape(batch * 2, self.pair_latent_dim, frames)
        audio = self.autoencoder.decode(pairs).sample
        if audio.ndim != 3 or audio.shape[:2] != (batch * 2, 2):
            raise ValueError(f"Expected stereo audio [2B, 2, L], got {tuple(audio.shape)}")
        return audio.reshape(batch, self.audio_channels, audio.shape[-1])
