from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from modules.foa_vae.wavvae_v4_post import FOA_WYZX, _project_foa_wyzx, _validate_foa


DAUDIO_DIRECTION_NAMES = ("x", "y", "z", "random")


class FOADAudioViewBuilder:
    """Build W and one antipodal pair of mono views from WYZX FOA audio."""

    _AXES = {
        "x": (1.0, 0.0, 0.0),
        "y": (0.0, 1.0, 0.0),
        "z": (0.0, 0.0, 1.0),
    }

    def __init__(
        self,
        gain: float = 1.0,
        direction_probabilities: Optional[Dict[str, float]] = None,
        eps: float = 1.0e-8,
    ):
        probabilities = dict(
            direction_probabilities
            or {"x": 0.2, "y": 0.2, "z": 0.2, "random": 0.4}
        )
        if set(probabilities) != set(DAUDIO_DIRECTION_NAMES):
            raise ValueError(
                "D_audio direction_probabilities must contain exactly x, y, z, and random"
            )
        values = [float(probabilities[name]) for name in DAUDIO_DIRECTION_NAMES]
        if any(value < 0.0 for value in values):
            raise ValueError("D_audio direction probabilities must be non-negative")
        if abs(sum(values) - 1.0) > 1.0e-6:
            raise ValueError("D_audio direction probabilities must sum to 1")
        if float(eps) <= 0.0:
            raise ValueError("D_audio eps must be positive")

        self.gain = float(gain)
        self.eps = float(eps)
        self.direction_probabilities = {
            name: value for name, value in zip(DAUDIO_DIRECTION_NAMES, values)
        }
        self._cumulative_probabilities = torch.tensor(values, dtype=torch.float64).cumsum(0)

    def sample_direction(
        self,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, str]:
        # Sampling stays on CPU so the task can use a deterministic rank-specific
        # generator without introducing a CUDA synchronization for one scalar.
        draw = float(torch.rand((), generator=generator))
        cumulative = self._cumulative_probabilities
        direction_name = DAUDIO_DIRECTION_NAMES[-1]
        for index, name in enumerate(DAUDIO_DIRECTION_NAMES):
            if draw < float(cumulative[index]):
                direction_name = name
                break

        if direction_name == "random":
            direction = torch.randn(1, 3, generator=generator)
            direction = F.normalize(direction, dim=-1, eps=self.eps).to(
                device=device,
                dtype=dtype,
            )
        else:
            direction = torch.tensor(
                [self._AXES[direction_name]],
                device=device,
                dtype=dtype,
            )
        return direction, direction_name

    def build_views(
        self,
        audio: torch.Tensor,
        direction: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return W [B,1,T] and [W+u, W-u] flattened to [2B,1,T]."""
        _validate_foa(audio)
        if direction.ndim == 1:
            direction = direction.unsqueeze(0)
        if direction.shape != (1, 3):
            raise ValueError(
                f"D_audio expects one direction shaped [1, 3], got {tuple(direction.shape)}"
            )
        direction = F.normalize(
            direction.to(device=audio.device, dtype=audio.dtype),
            dim=-1,
            eps=self.eps,
        )
        antipodal_directions = torch.cat([direction, -direction], dim=0)
        w = audio[:, FOA_WYZX["w"] : FOA_WYZX["w"] + 1]
        projections = _project_foa_wyzx(
            audio,
            antipodal_directions,
            gain=self.gain,
        )
        return w, projections


__all__ = ["DAUDIO_DIRECTION_NAMES", "FOADAudioViewBuilder"]
