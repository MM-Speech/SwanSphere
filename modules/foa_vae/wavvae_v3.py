import json
from pathlib import Path
from typing import Dict, Mapping

import torch
import torch.nn as nn

from modules.foa_vae.wavvae_v1 import _nested_get, trainable_param_report
from modules.foa_vae.wavvae_v2 import FOAWavVAE


DEFAULT_V3_DOWNSAMPLING_RATIOS = [2, 2, 4, 8, 8]
V3_RANDOM_INIT_SHAPE_MISMATCH_KEYS = {
    "encoder.block.1.conv1.weight_v",
    "decoder.block.3.conv_t1.weight_v",
}


def _copy_zero_extra_channels(source: torch.Tensor, target: torch.Tensor, dim: int) -> torch.Tensor:
    if target.shape[dim] < source.shape[dim]:
        raise ValueError(f"Cannot copy shape {tuple(source.shape)} into smaller target {tuple(target.shape)} on dim {dim}")
    patched = torch.zeros_like(target)
    slices = [slice(None)] * target.ndim
    slices[dim] = slice(0, source.shape[dim])
    patched[tuple(slices)] = source
    return patched


def _copy_repeated_channels(source: torch.Tensor, target: torch.Tensor, dim: int) -> torch.Tensor:
    if target.shape[dim] % source.shape[dim] != 0:
        raise ValueError(f"Cannot repeat shape {tuple(source.shape)} into {tuple(target.shape)} on dim {dim}")
    repeat_count = target.shape[dim] // source.shape[dim]
    repeats = [1] * source.ndim
    repeats[dim] = repeat_count
    patched = source.repeat(*repeats)
    if patched.shape != target.shape:
        raise ValueError(f"Repeated shape {tuple(patched.shape)} does not match target {tuple(target.shape)}")
    return patched


def adapt_oobleck_v3_state_dict(
    target_model: nn.Module,
    source_state: Mapping[str, torch.Tensor],
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """Warm-start a ds1024/z64/4ch Oobleck from the 2ch Stable Audio VAE.

    The v3 recipe follows the external ds1024_z64 setup:
    - new input channels are zero-initialized;
    - new output channels are repeated from the pretrained stereo output;
    - stride-shape-mismatch convs remain randomly initialized from the target model.
    """
    target_state = target_model.state_dict()
    patched_state = {}
    copied = 0
    random_init = []

    for key, target_param in target_state.items():
        source_param = source_state.get(key)
        if source_param is None:
            patched_state[key] = target_param
            random_init.append(key)
            continue

        if source_param.shape == target_param.shape:
            patched_state[key] = source_param
            copied += 1
            continue

        if key == "encoder.conv1.weight_v":
            patched_state[key] = _copy_zero_extra_channels(source_param, target_param, dim=1)
        elif key in {"decoder.conv2.weight_v", "decoder.conv2.weight_g", "decoder.conv2.bias"}:
            patched_state[key] = _copy_repeated_channels(source_param, target_param, dim=0)
        elif key in V3_RANDOM_INIT_SHAPE_MISMATCH_KEYS:
            patched_state[key] = target_param
            random_init.append(key)
        else:
            raise RuntimeError(
                f"Cannot adapt v3 parameter {key}: "
                f"source={tuple(source_param.shape)} target={tuple(target_param.shape)}"
            )

    if verbose:
        print(f"| FOA VAE v3 warm-start copied {copied} same-shape tensors")
        if random_init:
            print("| FOA VAE v3 random-init tensors:")
            for key in random_init:
                print(f"|   {key}: {tuple(target_state[key].shape)}")
    return patched_state


def _load_v3_oobleck(
    model_dir: Path,
    init_pretrained: bool,
    target_channels: int,
    downsampling_ratios,
    verbose: bool,
):
    from diffusers import AutoencoderOobleck

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Stable audio VAE config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}

    cfg["audio_channels"] = target_channels
    cfg["decoder_input_channels"] = int(cfg.get("decoder_input_channels", 64))
    cfg["downsampling_ratios"] = [int(x) for x in downsampling_ratios]

    model = AutoencoderOobleck(**cfg)
    if not init_pretrained:
        return model

    if verbose:
        print(
            "| Loading stable audio AutoencoderOobleck weights from "
            f"'{model_dir}' into v3 {target_channels}ch/z{cfg['decoder_input_channels']}/"
            f"ds{torch.prod(torch.tensor(cfg['downsampling_ratios'])).item()} model"
        )
    source_model = AutoencoderOobleck.from_pretrained(str(model_dir))
    patched_state = adapt_oobleck_v3_state_dict(
        model,
        source_model.state_dict(),
        verbose=verbose,
    )
    model.load_state_dict(patched_state, strict=True)
    return model


def build_foa_wavvae_v3(hparams=None, init_pretrained: bool = True, verbose: bool = True):
    model_dir = Path(_nested_get(hparams, "foa_vae", "pretrained_model_dir", "checkpoints/vae"))
    input_channels = int(_nested_get(hparams, "foa_vae", "input_channels", 4))
    output_channels = int(_nested_get(hparams, "foa_vae", "output_channels", 4))
    if input_channels != output_channels:
        raise ValueError(f"wavvae_v3 expects input_channels == output_channels, got {input_channels} and {output_channels}")
    latent_channels = int(_nested_get(hparams, "foa_vae", "latent_channels", 64))
    if latent_channels != 64:
        raise ValueError(f"wavvae_v3 follows ds1024_z64 and expects latent_channels=64, got {latent_channels}")
    downsampling_ratios = _nested_get(hparams, "foa_vae", "downsampling_ratios", DEFAULT_V3_DOWNSAMPLING_RATIOS)
    downsampling_ratio = 1
    for stride in downsampling_ratios:
        downsampling_ratio *= int(stride)
    if downsampling_ratio != 1024:
        raise ValueError(f"wavvae_v3 expects downsampling ratio 1024, got strides={downsampling_ratios}")

    autoencoder = _load_v3_oobleck(
        model_dir=model_dir,
        init_pretrained=init_pretrained,
        target_channels=input_channels,
        downsampling_ratios=downsampling_ratios,
        verbose=verbose,
    )
    return FOAWavVAE(autoencoder=autoencoder, input_channels=input_channels, output_channels=output_channels)


build_foa_wavvae = build_foa_wavvae_v3


__all__ = [
    "DEFAULT_V3_DOWNSAMPLING_RATIOS",
    "V3_RANDOM_INIT_SHAPE_MISMATCH_KEYS",
    "adapt_oobleck_v3_state_dict",
    "build_foa_wavvae",
    "build_foa_wavvae_v3",
    "trainable_param_report",
]
