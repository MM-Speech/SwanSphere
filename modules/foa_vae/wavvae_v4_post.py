from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_audio_tools.training.losses import auraloss


FOA_WYZX = {"w": 0, "y": 1, "z": 2, "x": 3}


def _validate_foa(audio: torch.Tensor) -> None:
    if audio.ndim != 3 or audio.shape[1] != 4:
        raise ValueError(f"Expected FOA tensor [B, 4, T] in WYZX order, got {tuple(audio.shape)}")


def _build_projection_directions(
    random_count: int,
    include_axes: bool,
    include_corners: bool,
    device,
    dtype,
) -> torch.Tensor:
    if random_count < 0:
        raise ValueError("FOA projection random_dirs must be non-negative")

    directions: List[List[float]] = []
    if include_axes:
        directions.extend(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )
    if include_corners:
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    directions.append([sx, sy, sz])

    if directions:
        output = torch.tensor(directions, device=device, dtype=dtype)
    else:
        output = torch.empty(0, 3, device=device, dtype=dtype)
    if random_count > 0:
        random_directions = torch.randn(random_count, 3, device=device, dtype=dtype)
        random_directions = F.normalize(random_directions, dim=-1, eps=1.0e-8)
        output = torch.cat([output, random_directions], dim=0)
    return F.normalize(output, dim=-1, eps=1.0e-8)


def _project_foa_wyzx(
    audio: torch.Tensor,
    directions: torch.Tensor,
    gain: float,
) -> torch.Tensor:
    _validate_foa(audio)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError(f"Expected directions [K, 3], got {tuple(directions.shape)}")

    directions = F.normalize(
        directions.to(device=audio.device, dtype=audio.dtype),
        dim=-1,
        eps=1.0e-8,
    )
    w = audio[:, FOA_WYZX["w"] : FOA_WYZX["w"] + 1]
    x = audio[:, FOA_WYZX["x"] : FOA_WYZX["x"] + 1]
    y = audio[:, FOA_WYZX["y"] : FOA_WYZX["y"] + 1]
    z = audio[:, FOA_WYZX["z"] : FOA_WYZX["z"] + 1]
    ux = directions[:, 0].view(1, -1, 1)
    uy = directions[:, 1].view(1, -1, 1)
    uz = directions[:, 2].view(1, -1, 1)
    projected = w + float(gain) * (ux * x + uy * y + uz * z)
    return projected.reshape(audio.shape[0] * directions.shape[0], 1, audio.shape[-1])


class FOAV4SpatialLoss(nn.Module):
    """Directional reconstruction losses for four-channel WYZX FOA audio."""

    _CROSS_PAIRS = (
        (FOA_WYZX["w"], FOA_WYZX["x"]),
        (FOA_WYZX["w"], FOA_WYZX["y"]),
        (FOA_WYZX["w"], FOA_WYZX["z"]),
        (FOA_WYZX["x"], FOA_WYZX["y"]),
        (FOA_WYZX["x"], FOA_WYZX["z"]),
        (FOA_WYZX["y"], FOA_WYZX["z"]),
    )

    def __init__(
        self,
        sample_rate: int,
        projection_mrstft_config: Dict,
        projection_config: Dict,
        spatial_config: Dict,
    ):
        super().__init__()
        self.projection_mrstft = auraloss.MultiResolutionSTFTLoss(
            sample_rate=int(sample_rate),
            **dict(projection_mrstft_config),
        )

        projection_config = dict(projection_config or {})
        self.projection_gain = float(projection_config.get("gain", 1.0))
        self.projection_include_axes = bool(projection_config.get("include_axes", True))
        self.projection_include_corners = bool(projection_config.get("include_corners", False))
        self.projection_random_dirs = int(projection_config.get("random_dirs", 0))
        self.projection_chunk_size = int(projection_config.get("mrstft_chunk_size", 2))
        if self.projection_random_dirs < 0:
            raise ValueError("FOA projection random_dirs must be non-negative")

        spatial_config = dict(spatial_config or {})
        self.fft_size = int(spatial_config.get("fft_size", 1024))
        self.hop_size = int(spatial_config.get("hop_size", 256))
        self.win_length = int(spatial_config.get("win_length", 1024))
        self.energy_percentile = float(spatial_config.get("energy_percentile", 0.5))
        self.eps = float(spatial_config.get("eps", 1.0e-7))
        if self.fft_size <= 0 or self.hop_size <= 0:
            raise ValueError("FOA spatial fft_size and hop_size must be positive")
        if not 0 < self.win_length <= self.fft_size:
            raise ValueError("FOA spatial win_length must be in (0, fft_size]")
        if not 0.0 <= self.energy_percentile <= 1.0:
            raise ValueError("FOA spatial energy_percentile must be in [0, 1]")
        if self.eps <= 0.0:
            raise ValueError("FOA spatial eps must be positive")
        self.register_buffer(
            "spatial_window",
            torch.hann_window(self.win_length, dtype=torch.float32),
            persistent=False,
        )

    def _projection_directions(self, audio: torch.Tensor) -> torch.Tensor:
        return _build_projection_directions(
            random_count=self.projection_random_dirs,
            include_axes=self.projection_include_axes,
            include_corners=self.projection_include_corners,
            device=audio.device,
            dtype=audio.dtype,
        )

    def _projection_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        directions = self._projection_directions(target)
        if directions.numel() == 0:
            return prediction.new_tensor(0.0)

        chunk_size = self.projection_chunk_size
        if chunk_size <= 0:
            chunk_size = directions.shape[0]
        chunk_size = min(max(1, chunk_size), directions.shape[0])

        total = prediction.new_tensor(0.0)
        direction_count = 0
        for direction_chunk in directions.split(chunk_size, dim=0):
            prediction_projection = _project_foa_wyzx(
                prediction,
                direction_chunk,
                gain=self.projection_gain,
            )
            target_projection = _project_foa_wyzx(
                target,
                direction_chunk,
                gain=self.projection_gain,
            )
            total = total + self.projection_mrstft(
                prediction_projection,
                target_projection,
            ) * direction_chunk.shape[0]
            direction_count += direction_chunk.shape[0]
        return total / max(1, direction_count)

    def _stft_foa(self, audio: torch.Tensor) -> torch.Tensor:
        _validate_foa(audio)
        batch_size, channels, length = audio.shape
        spectrum = torch.stft(
            audio.float().reshape(batch_size * channels, length),
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=self.spatial_window,
            return_complex=True,
        )
        return spectrum.reshape(batch_size, channels, spectrum.shape[-2], spectrum.shape[-1])

    @staticmethod
    def _active_intensity(spectrum: torch.Tensor) -> torch.Tensor:
        w = spectrum[:, FOA_WYZX["w"]]
        x = spectrum[:, FOA_WYZX["x"]]
        y = spectrum[:, FOA_WYZX["y"]]
        z = spectrum[:, FOA_WYZX["z"]]
        return torch.stack(
            [
                torch.real(torch.conj(w) * x),
                torch.real(torch.conj(w) * y),
                torch.real(torch.conj(w) * z),
            ],
            dim=1,
        )

    def _intensity_loss(
        self,
        prediction_spectrum: torch.Tensor,
        target_spectrum: torch.Tensor,
    ) -> torch.Tensor:
        prediction_direction = F.normalize(
            self._active_intensity(prediction_spectrum),
            dim=1,
            eps=self.eps,
        )
        target_direction = F.normalize(
            self._active_intensity(target_spectrum),
            dim=1,
            eps=self.eps,
        )
        direction_error = 1.0 - (
            prediction_direction * target_direction
        ).sum(dim=1).clamp(-1.0, 1.0)

        w_energy = target_spectrum[:, FOA_WYZX["w"]].abs().pow(2)
        if self.energy_percentile > 0.0:
            threshold = torch.quantile(w_energy.detach().flatten(), self.energy_percentile)
            mask = w_energy > threshold
        else:
            mask = w_energy > self.eps
        weights = w_energy * mask.to(w_energy.dtype)
        return (direction_error * weights).sum() / weights.sum().clamp_min(self.eps)

    def _normalized_cross(
        self,
        spectrum: torch.Tensor,
        first: int,
        second: int,
    ) -> torch.Tensor:
        cross = torch.conj(spectrum[:, first]) * spectrum[:, second]
        return cross / cross.abs().clamp_min(self.eps)

    def _cross_loss(
        self,
        prediction_spectrum: torch.Tensor,
        target_spectrum: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        for first, second in self._CROSS_PAIRS:
            prediction_cross = self._normalized_cross(prediction_spectrum, first, second)
            target_cross = self._normalized_cross(target_spectrum, first, second)
            losses.append(
                F.l1_loss(
                    torch.view_as_real(prediction_cross),
                    torch.view_as_real(target_cross),
                )
            )
        return torch.stack(losses).mean()

    def _normalized_scm(self, spectrum: torch.Tensor) -> torch.Tensor:
        ordered = spectrum[
            :,
            [FOA_WYZX["w"], FOA_WYZX["x"], FOA_WYZX["y"], FOA_WYZX["z"]],
        ].permute(0, 2, 3, 1)
        covariance = ordered.unsqueeze(-1) * torch.conj(ordered.unsqueeze(-2))
        denominator = covariance.abs().pow(2).sum(
            dim=(-1, -2),
            keepdim=True,
        ).sqrt().clamp_min(self.eps)
        return covariance / denominator

    def _covariance_loss(
        self,
        prediction_spectrum: torch.Tensor,
        target_spectrum: torch.Tensor,
    ) -> torch.Tensor:
        prediction_scm = self._normalized_scm(prediction_spectrum)
        target_scm = self._normalized_scm(target_spectrum)
        return F.l1_loss(
            torch.view_as_real(prediction_scm),
            torch.view_as_real(target_scm),
        )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        compute_projection: bool = True,
        compute_intensity: bool = True,
        compute_cross: bool = True,
        compute_covariance: bool = True,
    ) -> Dict[str, torch.Tensor]:
        _validate_foa(prediction)
        _validate_foa(target)
        if prediction.shape != target.shape:
            raise ValueError(
                f"FOA prediction and target shapes differ: {tuple(prediction.shape)} vs {tuple(target.shape)}"
            )

        losses = {}
        if compute_projection:
            losses["proj_mrstft"] = self._projection_loss(prediction, target)

        if compute_intensity or compute_cross or compute_covariance:
            prediction_spectrum = self._stft_foa(prediction)
            target_spectrum = self._stft_foa(target)
            if compute_intensity:
                losses["intensity_dir"] = self._intensity_loss(
                    prediction_spectrum,
                    target_spectrum,
                )
            if compute_cross:
                losses["cross_phase"] = self._cross_loss(
                    prediction_spectrum,
                    target_spectrum,
                )
            if compute_covariance:
                losses["cov"] = self._covariance_loss(
                    prediction_spectrum,
                    target_spectrum,
                )
        return losses


__all__ = ["FOAV4SpatialLoss"]
