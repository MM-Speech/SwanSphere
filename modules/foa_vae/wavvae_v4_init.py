from collections.abc import Mapping
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from modules.foa_vae.wavvae_v4 import FOAWavVAEV4, build_foa_wavvae_v4


DEFAULT_OMNIAUDIO_INIT_CKPT = "checkpoints/omniaudio/omniaudio_vae.state_dict.pt"
V4_INIT_BOUNDARY_PREFIXES = (
    "autoencoder.encoder.conv1.",
    "autoencoder.decoder.conv2.",
)


class FOAWavVAEV4Init(FOAWavVAEV4):
    """V4 architecture initialized from a native four-channel OmniAudio VAE."""


def extract_omniaudio_state_dict(checkpoint) -> Mapping[str, torch.Tensor]:
    """Return a raw generator state dict from either supported OmniAudio format."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "OmniAudio checkpoint must be a tensor state dict or a mapping containing model_gen"
        )

    state_dict = checkpoint.get("model_gen", checkpoint)
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise RuntimeError("OmniAudio checkpoint does not contain a non-empty tensor state dict")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state_dict.items()):
        raise RuntimeError("OmniAudio checkpoint does not contain a pure tensor state dict")
    return state_dict


def _load_checkpoint_file(path: Path):
    try:
        return torch.load(
            str(path),
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch versions that do not expose mmap or
        # weights_only. The configured training environment supports both.
        return torch.load(str(path), map_location="cpu")


def _state_dict_differences(
    target_state: Mapping[str, torch.Tensor],
    source_state: Mapping[str, torch.Tensor],
) -> Tuple[List[str], List[str], List[Tuple[str, tuple, tuple]]]:
    missing_keys = sorted(set(target_state) - set(source_state))
    unexpected_keys = sorted(set(source_state) - set(target_state))
    shape_mismatches = sorted(
        (
            key,
            tuple(source_state[key].shape),
            tuple(target_state[key].shape),
        )
        for key in set(target_state) & set(source_state)
        if source_state[key].shape != target_state[key].shape
    )
    return missing_keys, unexpected_keys, shape_mismatches


def load_omniaudio_weights(
    model: nn.Module,
    checkpoint_path,
    verbose: bool = True,
) -> Dict[str, object]:
    """Strictly load every OmniAudio VAE tensor into the native V4 model."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"OmniAudio VAE checkpoint not found: {path}")

    checkpoint = _load_checkpoint_file(path)
    source_state = extract_omniaudio_state_dict(checkpoint)
    target_state = model.state_dict()
    missing_keys, unexpected_keys, shape_mismatches = _state_dict_differences(
        target_state,
        source_state,
    )
    if missing_keys or unexpected_keys or shape_mismatches:
        details = []
        if missing_keys:
            details.append(f"missing keys: {missing_keys}")
        if unexpected_keys:
            details.append(f"unexpected keys: {unexpected_keys}")
        if shape_mismatches:
            details.append(f"shape mismatch: {shape_mismatches}")
        raise RuntimeError("OmniAudio VAE state dict is incompatible: " + "; ".join(details))

    model.load_state_dict(source_state, strict=True)
    report = {
        "checkpoint_path": str(path),
        "loaded_tensors": len(source_state),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatches": shape_mismatches,
    }
    if verbose:
        print(f"| Strictly loaded {len(source_state)} OmniAudio VAE tensors from '{path}'")
    return report


def set_v4_init_generator_trainability(model: nn.Module, mode: str) -> List[str]:
    """Apply frozen, boundary-only, or full Generator trainability."""

    if hasattr(model, "module"):
        model = model.module
    if mode not in {"frozen", "boundary", "full"}:
        raise ValueError(f"Unknown v4_init Generator trainability mode: {mode}")

    trainable_names = []
    for name, parameter in model.named_parameters():
        trainable = mode == "full" or (
            mode == "boundary" and name.startswith(V4_INIT_BOUNDARY_PREFIXES)
        )
        parameter.requires_grad_(trainable)
        if trainable:
            trainable_names.append(name)

    if mode == "boundary" and not trainable_names:
        raise RuntimeError(
            "No v4_init boundary parameters matched "
            f"prefixes={V4_INIT_BOUNDARY_PREFIXES}"
        )
    return trainable_names


def build_foa_wavvae_v4_init(
    hparams=None,
    init_omniaudio: bool = True,
    verbose: bool = True,
) -> FOAWavVAEV4Init:
    """Build native 4ch/ds2048/z64 V4 and optionally load all OmniAudio weights."""

    base_model = build_foa_wavvae_v4(
        hparams=hparams,
        init_pretrained=False,
        verbose=verbose,
    )
    model = FOAWavVAEV4Init(
        autoencoder=base_model.autoencoder,
        input_channels=base_model.in_channels,
        output_channels=base_model.out_channels,
    )

    if init_omniaudio:
        section = hparams.get("foa_vae", {}) if hasattr(hparams, "get") else {}
        checkpoint_path = section.get(
            "omniaudio_init_ckpt",
            DEFAULT_OMNIAUDIO_INIT_CKPT,
        )
        load_omniaudio_weights(model, checkpoint_path, verbose=verbose)
    return model


build_foa_wavvae = build_foa_wavvae_v4_init


__all__ = [
    "DEFAULT_OMNIAUDIO_INIT_CKPT",
    "V4_INIT_BOUNDARY_PREFIXES",
    "FOAWavVAEV4Init",
    "build_foa_wavvae",
    "build_foa_wavvae_v4_init",
    "extract_omniaudio_state_dict",
    "load_omniaudio_weights",
    "set_v4_init_generator_trainability",
]
