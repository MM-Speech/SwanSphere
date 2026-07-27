import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch
import torch.nn as nn


DEFAULT_V4_DOWNSAMPLING_RATIOS = [2, 4, 4, 8, 8]
V4_SHAPE_MISMATCH_KEYS = {
    "encoder.conv1.weight_v",
    "decoder.conv2.weight_v",
    "decoder.conv2.weight_g",
}


def _nested_get(hparams, section: str, key: str, default=None):
    if hparams is None:
        return default
    section_value = hparams.get(section, {}) if hasattr(hparams, "get") else {}
    if isinstance(section_value, Mapping) and key in section_value:
        return section_value[key]
    return hparams.get(key, default) if hasattr(hparams, "get") else default


def trainable_param_report(model: nn.Module) -> int:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Total params: {total:,}")
    print(f"Trainable params (requires_grad=True): {trainable:,}")
    return trainable


class FOAWavVAEV4(nn.Module):
    """Task-facing wrapper around a native four-channel Oobleck VAE."""

    def __init__(self, autoencoder: nn.Module, input_channels: int = 4, output_channels: int = 4):
        super().__init__()
        self.autoencoder = autoencoder
        self.in_channels = int(input_channels)
        self.out_channels = int(output_channels)
        self.io_channels = int(output_channels)

    @property
    def encoder(self):
        return self.autoencoder.encoder

    @property
    def decoder(self):
        return self.autoencoder.decoder

    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        latent_dist = self.autoencoder.encode(audio).latent_dist
        latents = latent_dist.sample()
        reconstruction = self.autoencoder.decode(latents).sample

        if hasattr(latent_dist, "kl"):
            kl = latent_dist.kl().mean()
        else:
            mean = latent_dist.mean
            logvar = latent_dist.logvar
            kl = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1.0).mean()

        return {
            "recon": reconstruction,
            "kl": kl,
            "mu": latent_dist.mean,
            "logvar": latent_dist.logvar,
        }


def _as_int_list(values: Sequence[int]) -> list:
    return [int(value) for value in values]


def _validate_target_settings(
    input_channels: int,
    output_channels: int,
    latent_channels: int,
    downsampling_ratios: Sequence[int],
) -> None:
    if int(input_channels) != 4 or int(output_channels) != 4:
        raise ValueError(
            "wavvae_v4 requires input_channels=4 and output_channels=4, "
            f"got {input_channels} and {output_channels}"
        )
    if int(latent_channels) != 64:
        raise ValueError(f"wavvae_v4 requires latent_channels=64, got {latent_channels}")
    ratios = _as_int_list(downsampling_ratios)
    if ratios != DEFAULT_V4_DOWNSAMPLING_RATIOS:
        raise ValueError(
            "wavvae_v4 requires the original downsampling ratios "
            f"{DEFAULT_V4_DOWNSAMPLING_RATIOS}, got {ratios}"
        )


def _validate_source_config(config: Mapping) -> None:
    audio_channels = int(config.get("audio_channels", -1))
    latent_channels = int(config.get("decoder_input_channels", -1))
    ratios = _as_int_list(config.get("downsampling_ratios", []))
    if audio_channels != 2:
        raise ValueError(f"Stable Audio source must use audio_channels=2, got {audio_channels}")
    if latent_channels != 64:
        raise ValueError(f"Stable Audio source must use decoder_input_channels=64, got {latent_channels}")
    if ratios != DEFAULT_V4_DOWNSAMPLING_RATIOS:
        raise ValueError(
            "Stable Audio source must use downsampling_ratios="
            f"{DEFAULT_V4_DOWNSAMPLING_RATIOS}, got {ratios}"
        )


def _orthogonal_decoder_directions(target: torch.Tensor) -> torch.Tensor:
    directions = torch.empty_like(target)
    nn.init.orthogonal_(directions.flatten(1))
    return directions


def _energy_preserving_decoder_gain(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    gain = source.detach().float().square().sum().div(target.shape[0]).sqrt()
    return torch.ones_like(target) * gain.to(device=target.device, dtype=target.dtype)


def adapt_oobleck_v4_state_dict(
    target_model: nn.Module,
    source_state: Mapping[str, torch.Tensor],
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """Build a strict four-channel state dict from the original stereo state."""

    target_state = target_model.state_dict()
    patched_state: Dict[str, torch.Tensor] = {}
    observed_mismatches = set()
    copied = 0

    for key, target_value in target_state.items():
        source_value = source_state.get(key)
        if source_value is None:
            raise RuntimeError(f"Missing Stable Audio parameter for v4: {key}")

        if source_value.shape == target_value.shape:
            patched_state[key] = source_value
            copied += 1
            continue

        observed_mismatches.add(key)
        if key == "encoder.conv1.weight_v":
            patched_state[key] = target_value
        elif key == "decoder.conv2.weight_v":
            patched_state[key] = _orthogonal_decoder_directions(target_value)
        elif key == "decoder.conv2.weight_g":
            patched_state[key] = _energy_preserving_decoder_gain(source_value, target_value)
        else:
            raise RuntimeError(
                f"Unexpected v4 shape mismatch for {key}: "
                f"source={tuple(source_value.shape)} target={tuple(target_value.shape)}"
            )

    if observed_mismatches != V4_SHAPE_MISMATCH_KEYS:
        raise RuntimeError(
            f"v4 mismatch set must be {sorted(V4_SHAPE_MISMATCH_KEYS)}, "
            f"got {sorted(observed_mismatches)}"
        )

    if verbose:
        print(f"| FOA VAE v4 copied {copied} same-shape Stable Audio tensors")
        print("| FOA VAE v4 independently initialized tensors:")
        for key in sorted(observed_mismatches):
            print(f"|   {key}: {tuple(target_state[key].shape)}")
    return patched_state


def _read_source_config(model_dir: Path) -> dict:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Stable Audio VAE config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config = {key: value for key, value in config.items() if not key.startswith("_")}
    _validate_source_config(config)
    return config


def _load_v4_oobleck(
    model_dir: Path,
    init_pretrained: bool,
    target_channels: int,
    verbose: bool,
):
    from diffusers import AutoencoderOobleck

    source_config = _read_source_config(model_dir)
    target_config = dict(source_config)
    target_config["audio_channels"] = int(target_channels)
    target_model = AutoencoderOobleck(**target_config)

    if not init_pretrained:
        return target_model

    if verbose:
        print(
            "| Loading original Stable Audio Oobleck from "
            f"'{model_dir}' into native 4ch/ds2048/z64 v4"
        )
    source_model = AutoencoderOobleck.from_pretrained(str(model_dir))
    patched_state = adapt_oobleck_v4_state_dict(
        target_model=target_model,
        source_state=source_model.state_dict(),
        verbose=verbose,
    )
    target_model.load_state_dict(patched_state, strict=True)
    return target_model


def build_foa_wavvae_v4(hparams=None, init_pretrained: bool = True, verbose: bool = True):
    model_dir = Path(_nested_get(hparams, "foa_vae", "pretrained_model_dir", "checkpoints/vae"))
    input_channels = int(_nested_get(hparams, "foa_vae", "input_channels", 4))
    output_channels = int(_nested_get(hparams, "foa_vae", "output_channels", 4))
    latent_channels = int(_nested_get(hparams, "foa_vae", "latent_channels", 64))
    downsampling_ratios = _nested_get(
        hparams,
        "foa_vae",
        "downsampling_ratios",
        DEFAULT_V4_DOWNSAMPLING_RATIOS,
    )
    _validate_target_settings(
        input_channels=input_channels,
        output_channels=output_channels,
        latent_channels=latent_channels,
        downsampling_ratios=downsampling_ratios,
    )

    autoencoder = _load_v4_oobleck(
        model_dir=model_dir,
        init_pretrained=init_pretrained,
        target_channels=input_channels,
        verbose=verbose,
    )
    return FOAWavVAEV4(
        autoencoder=autoencoder,
        input_channels=input_channels,
        output_channels=output_channels,
    )


build_foa_wavvae = build_foa_wavvae_v4


__all__ = [
    "DEFAULT_V4_DOWNSAMPLING_RATIOS",
    "V4_SHAPE_MISMATCH_KEYS",
    "FOAWavVAEV4",
    "adapt_oobleck_v4_state_dict",
    "build_foa_wavvae",
    "build_foa_wavvae_v4",
    "trainable_param_report",
]
