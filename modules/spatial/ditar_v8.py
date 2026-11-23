"""DiTAR v8 with a robust, bottlenecked LM audio-history branch.

LocEnc, the causal LM, the DiT history bottleneck and flow targets come from
the shared architecture. LiveFoley-style noise is applied to a separate copy of model-space
history latents (raw paired Stable Audio latents in v8) BEFORE LocEnc; its
output then passes through a configurable narrow MLP. BOS and the video condition bypass both operations.
"""

import math
from dataclasses import dataclass
from numbers import Integral

import torch
from torch import nn

from modules.spatial.ditar_base import DiTARModel as BaseDiTARModel
from modules.spatial.ditar_base import ModelArgs as BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    # Concatenate two 64-D stereo Stable Audio encodings in channel order.
    in_channels: int = 128
    out_channels: int = 128
    # Preserve the 215 valid frames used per ten-second training example.
    latent_frame_rate: float = 21.5
    video_frame_rate: float = 4.0

    use_lm_history_bottleneck: bool = True
    lm_history_bottleneck_dim: int = 16

    # Independently draw one Bernoulli and one gamma per sample/patch.
    lm_history_noise_prob: float = 0.5
    lm_history_noise_gamma_min: float = 0.0
    lm_history_noise_gamma_max: float = 0.5
    # Inference perturbs every feedback patch at this fixed gamma; zero
    # disables inference noise. This is not the training Bernoulli probability.
    lm_history_noise_infer_gamma: float = 0.15

    @property
    def patches_per_frame(self):
        return self.latent_frame_rate / (self.video_frame_rate * self.patch_size)


class DiTARModel(BaseDiTARModel):
    def __init__(self, config: ModelArgs):
        if config.in_channels != 128 or config.out_channels != 128:
            raise ValueError("DiTAR v8 requires 128-D paired Stable Audio VAE latents")
        if not math.isclose(float(config.latent_frame_rate), 21.5, rel_tol=0, abs_tol=1e-8):
            raise ValueError("DiTAR v8 requires latent_frame_rate=21.5")
        if not math.isfinite(float(config.video_frame_rate)) or config.video_frame_rate <= 0:
            raise ValueError("video_frame_rate must be finite and positive")
        if not isinstance(config.patch_size, Integral) or isinstance(config.patch_size, bool) or config.patch_size <= 0:
            raise ValueError("patch_size must be a positive integer")
        dim = config.lm_history_bottleneck_dim
        if not isinstance(dim, Integral) or isinstance(dim, bool) or dim <= 0:
            raise ValueError("lm_history_bottleneck_dim must be a positive integer")
        for name in (
            "lm_history_noise_prob",
            "lm_history_noise_gamma_min",
            "lm_history_noise_gamma_max",
            "lm_history_noise_infer_gamma",
        ):
            value = float(getattr(config, name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if config.lm_history_noise_gamma_min > config.lm_history_noise_gamma_max:
            raise ValueError("lm_history_noise_gamma_min must not exceed gamma_max")

        super().__init__(config)
        if config.use_lm_history_bottleneck:
            # No residual bypass: every audio-history embedding must pass
            # through the requested narrow dimension. LocEnc itself is intact.
            self.lm_history_bottleneck = nn.Sequential(
                nn.LayerNorm(config.lm_dim),
                nn.Linear(config.lm_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, config.lm_dim),
            )
            if not config.use_lm_audio_history:
                self.lm_history_bottleneck.requires_grad_(False)
        else:
            self.lm_history_bottleneck = nn.Identity()

    def _corrupt_lm_history(self, x, x_mask, *, inference=False):
        """Perturb only valid LM history frames, without mutating either input.

        Training forward: per-patch Bernoulli with gamma sampled uniformly.
        Validation forward (eval mode): clean teacher history.
        Explicit inference: fixed gamma even if the caller left train mode on.
        """
        if x.ndim != 3 or x.shape[-1] != self.config.in_channels:
            raise ValueError("LM history must have shape [B, T, latent_dim]")
        if x_mask.shape != x.shape[:2]:
            raise ValueError("LM history mask must have shape [B, T]")
        if x.shape[1] % self.config.patch_size:
            raise ValueError("LM history length must be divisible by patch_size")
        if not torch.is_floating_point(x):
            raise TypeError("LM history must be floating point")
        if not self.config.use_lm_audio_history:
            return x
        if inference:
            if self.config.lm_history_noise_infer_gamma == 0:
                return x
        elif (
            not self.training
            or self.config.lm_history_noise_prob == 0
            or self.config.lm_history_noise_gamma_max == 0
        ):
            return x

        batch, frames, channels = x.shape
        patch_size = self.config.patch_size
        patches = frames // patch_size
        valid = x_mask.to(device=x.device, dtype=torch.bool).reshape(
            batch, patches, patch_size, 1
        )
        available = valid.any(dim=2, keepdim=True)
        if not available.any():
            return x
        gamma_shape = (batch, patches, 1, 1)
        if inference:
            gamma = torch.full(
                gamma_shape,
                float(self.config.lm_history_noise_infer_gamma),
                device=x.device,
                dtype=torch.float32,
            )
            selected = available
        else:
            selected = (
                torch.rand(gamma_shape, device=x.device)
                < self.config.lm_history_noise_prob
            ) & available
            gamma = torch.empty(gamma_shape, device=x.device).uniform_(
                self.config.lm_history_noise_gamma_min,
                self.config.lm_history_noise_gamma_max,
            )
        if not selected.any():
            return x
        patch_latents = x.reshape(batch, patches, patch_size, channels)
        noise = torch.randn(patch_latents.shape, device=x.device, dtype=torch.float32)
        corrupted = (
            (1.0 - gamma).sqrt() * patch_latents.float()
            + gamma.sqrt() * noise
        ).to(x.dtype)
        # The clean branch and invalid frames remain bitwise unchanged. This
        # new tensor is never used as the DiT history or flow-matching target.
        return torch.where(selected & valid, corrupted, patch_latents).reshape_as(x)

    def _encode_lm_history(self, x, x_mask, *, inference=False):
        history = self._corrupt_lm_history(x, x_mask, inference=inference)
        encoded, encoded_mask = self.forward_encoder(history, x_mask)
        return self.lm_history_bottleneck(encoded), encoded_mask
