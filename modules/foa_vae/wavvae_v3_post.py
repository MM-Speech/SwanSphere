from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_audio_tools.training.losses import auraloss

from modules.foa_vae.wavvae_v1 import (
    FOA_WYZX,
    build_projection_directions,
    project_foa_wyzx,
)


class FOAV3SpatialLoss(nn.Module):
    """Spatial reconstruction losses for WYZX first-order ambisonics."""

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
        self.register_buffer(
            "spatial_window",
            torch.hann_window(self.win_length, dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _validate_audio(audio: torch.Tensor) -> None:
        if audio.ndim != 3 or audio.shape[1] != 4:
            raise ValueError(f"Expected FOA tensor [B, 4, T] in WYZX order, got {tuple(audio.shape)}")

    def _projection_directions(self, audio: torch.Tensor) -> torch.Tensor:
        return build_projection_directions(
            random_count=self.projection_random_dirs,
            include_axes=self.projection_include_axes,
            include_corners=self.projection_include_corners,
            device=audio.device,
            dtype=audio.dtype,
        )

    def _projection_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        directions = self._projection_directions(target)
        if directions.numel() == 0:
            return pred.new_tensor(0.0)

        chunk_size = self.projection_chunk_size
        if chunk_size <= 0:
            chunk_size = directions.shape[0]
        chunk_size = min(max(1, chunk_size), directions.shape[0])

        total = pred.new_tensor(0.0)
        num_directions = 0
        for direction_chunk in directions.split(chunk_size, dim=0):
            pred_projection = project_foa_wyzx(
                pred,
                direction_chunk,
                gain=self.projection_gain,
                fold_batch=True,
            )
            target_projection = project_foa_wyzx(
                target,
                direction_chunk,
                gain=self.projection_gain,
                fold_batch=True,
            )
            total = total + self.projection_mrstft(pred_projection, target_projection) * direction_chunk.shape[0]
            num_directions += direction_chunk.shape[0]
        return total / max(1, num_directions)

    def _stft_foa(self, audio: torch.Tensor) -> torch.Tensor:
        self._validate_audio(audio)
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

    def _intensity_loss(self, pred_spec: torch.Tensor, target_spec: torch.Tensor) -> torch.Tensor:
        pred_direction = F.normalize(self._active_intensity(pred_spec), dim=1, eps=self.eps)
        target_direction = F.normalize(self._active_intensity(target_spec), dim=1, eps=self.eps)
        direction_error = 1.0 - (
            pred_direction * target_direction
        ).sum(dim=1).clamp(-1.0, 1.0)

        w_energy = target_spec[:, FOA_WYZX["w"]].abs().pow(2)
        if self.energy_percentile > 0.0:
            threshold = torch.quantile(w_energy.detach().flatten(), self.energy_percentile)
            mask = w_energy > threshold
        else:
            mask = w_energy > self.eps
        weights = w_energy * mask.to(w_energy.dtype)
        return (direction_error * weights).sum() / weights.sum().clamp_min(self.eps)

    def _normalized_cross(self, spectrum: torch.Tensor, first: int, second: int) -> torch.Tensor:
        cross = torch.conj(spectrum[:, first]) * spectrum[:, second]
        return cross / cross.abs().clamp_min(self.eps)

    def _cross_loss(self, pred_spec: torch.Tensor, target_spec: torch.Tensor) -> torch.Tensor:
        losses = []
        for first, second in self._CROSS_PAIRS:
            pred_cross = self._normalized_cross(pred_spec, first, second)
            target_cross = self._normalized_cross(target_spec, first, second)
            losses.append(
                F.l1_loss(torch.view_as_real(pred_cross), torch.view_as_real(target_cross))
            )
        return torch.stack(losses).mean()

    def _normalized_scm(self, spectrum: torch.Tensor) -> torch.Tensor:
        spectrum = spectrum[:, [FOA_WYZX["w"], FOA_WYZX["x"], FOA_WYZX["y"], FOA_WYZX["z"]]]
        spectrum = spectrum.permute(0, 2, 3, 1)
        scm = spectrum.unsqueeze(-1) * torch.conj(spectrum.unsqueeze(-2))
        denominator = scm.abs().pow(2).sum(dim=(-1, -2), keepdim=True).sqrt().clamp_min(self.eps)
        return scm / denominator

    def _covariance_loss(self, pred_spec: torch.Tensor, target_spec: torch.Tensor) -> torch.Tensor:
        pred_scm = self._normalized_scm(pred_spec)
        target_scm = self._normalized_scm(target_spec)
        return F.l1_loss(torch.view_as_real(pred_scm), torch.view_as_real(target_scm))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        compute_projection: bool = True,
        compute_intensity: bool = True,
        compute_cross: bool = True,
        compute_covariance: bool = True,
    ) -> Dict[str, torch.Tensor]:
        self._validate_audio(pred)
        self._validate_audio(target)
        if pred.shape != target.shape:
            raise ValueError(f"FOA prediction and target shapes differ: {tuple(pred.shape)} vs {tuple(target.shape)}")

        losses = {}
        if compute_projection:
            losses["proj_mrstft"] = self._projection_loss(pred, target)

        if compute_intensity or compute_cross or compute_covariance:
            pred_spec = self._stft_foa(pred)
            target_spec = self._stft_foa(target)
            if compute_intensity:
                losses["intensity_dir"] = self._intensity_loss(pred_spec, target_spec)
            if compute_cross:
                losses["cross_phase"] = self._cross_loss(pred_spec, target_spec)
            if compute_covariance:
                losses["cov"] = self._covariance_loss(pred_spec, target_spec)
        return losses
