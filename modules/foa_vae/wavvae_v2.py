import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from modules.foa_vae.wavvae_v1 import (
    DEFAULT_STEREO_PROJECTION_PAIRS,
    FOA_WYZX,
    PHYSICAL_XYZ_IN_WYZX,
    _nested_get,
    active_intensity_direction_loss,
    build_projection_directions,
    build_stereo_projection_pairs,
    linear_ramp,
    normalized_cross_spectrum_loss,
    project_foa_stereo_wyzx,
    project_foa_wyzx,
    scheduled_value,
    set_trainable_by_prefixes,
    spatial_covariance_loss,
    trainable_param_report,
    unwrap_training_module,
    wyzx_components,
)


class FOAWavVAE(nn.Module):
    """Thin task-facing wrapper around a 4ch AutoencoderOobleck.

    The wrapper only normalizes the forward return value used by the existing
    FOA VAE tasks. It does not add input/output projectors or extra learnable
    layers outside the original VAE.
    """

    def __init__(self, autoencoder: nn.Module, input_channels: int = 4, output_channels: int = 4):
        super().__init__()
        self.autoencoder = autoencoder
        self.in_channels = input_channels
        self.out_channels = output_channels
        self.io_channels = output_channels

    @property
    def encoder(self):
        return self.autoencoder.encoder

    @property
    def decoder(self):
        return self.autoencoder.decoder

    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.autoencoder.encode(audio)
        latent_dist = encoded.latent_dist
        latents = latent_dist.sample()
        decoded = self.autoencoder.decode(latents)

        mu = latent_dist.mean
        logvar = latent_dist.logvar
        if hasattr(latent_dist, "kl"):
            kl = latent_dist.kl().mean()
        else:
            kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0).mean()

        return {
            "recon": decoded.sample,
            "kl": kl,
            "mu": mu,
            "logvar": logvar,
        }


def _base_autoencoder_with_prefix(model) -> Tuple[nn.Module, str]:
    model = unwrap_training_module(model)
    if hasattr(model, "autoencoder"):
        return model.autoencoder, "autoencoder."
    return model, ""


def _block_names(module) -> List[str]:
    if hasattr(module, "block"):
        return [name for name, _ in module.block.named_children()]
    return []


def adapter_param_prefixes(model) -> List[str]:
    """Native VAE interface layers used as the lightweight adaptation part."""
    base_model, base_prefix = _base_autoencoder_with_prefix(model)
    prefixes = []
    if hasattr(base_model, "encoder") and hasattr(base_model.encoder, "conv1"):
        prefixes.append(f"{base_prefix}encoder.conv1")
    if hasattr(base_model, "encoder") and hasattr(base_model.encoder, "conv2"):
        prefixes.append(f"{base_prefix}encoder.conv2")
    if hasattr(base_model, "decoder") and hasattr(base_model.decoder, "conv1"):
        prefixes.append(f"{base_prefix}decoder.conv1")
    if hasattr(base_model, "decoder") and hasattr(base_model.decoder, "conv2"):
        prefixes.append(f"{base_prefix}decoder.conv2")
    return prefixes


def io_adapter_param_prefixes(model) -> List[str]:
    """Native VAE audio I/O layers used for the first warmup segment."""
    base_model, base_prefix = _base_autoencoder_with_prefix(model)
    prefixes = []
    if hasattr(base_model, "encoder") and hasattr(base_model.encoder, "conv1"):
        prefixes.append(f"{base_prefix}encoder.conv1")
    if hasattr(base_model, "decoder") and hasattr(base_model.decoder, "conv2"):
        prefixes.append(f"{base_prefix}decoder.conv2")
    return prefixes


def latent_adapter_param_prefixes(model) -> List[str]:
    """Native VAE latent interface layers introduced by latent expansion."""
    base_model, base_prefix = _base_autoencoder_with_prefix(model)
    prefixes = []
    if hasattr(base_model, "encoder") and hasattr(base_model.encoder, "conv2"):
        prefixes.append(f"{base_prefix}encoder.conv2")
    if hasattr(base_model, "decoder") and hasattr(base_model.decoder, "conv1"):
        prefixes.append(f"{base_prefix}decoder.conv1")
    return prefixes


def partial_unfreeze_prefixes(model, encoder_last_n: int = 2, decoder_last_n: int = 2) -> List[str]:
    prefixes = adapter_param_prefixes(model)
    base_model, base_prefix = _base_autoencoder_with_prefix(model)
    enc_block_names = _block_names(base_model.encoder)
    dec_block_names = _block_names(base_model.decoder)
    prefixes.extend([f"{base_prefix}encoder.block.{name}" for name in enc_block_names[-encoder_last_n:]])
    prefixes.extend([f"{base_prefix}decoder.block.{name}" for name in dec_block_names[-decoder_last_n:]])
    return prefixes


def split_adapter_backbone_params(model) -> List[Dict]:
    model = unwrap_training_module(model)
    prefixes = tuple(adapter_param_prefixes(model))
    adapter, backbone = [], []
    for name, param in model.named_parameters():
        (adapter if name.startswith(prefixes) else backbone).append(param)
    groups = []
    if adapter:
        groups.append({"name": "adapter", "params": adapter})
    if backbone:
        groups.append({"name": "backbone", "params": backbone})
    return groups


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


def _copy_repeated_vae_moments(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.shape[0] % 2 != 0 or target.shape[0] % 2 != 0:
        raise ValueError(f"VAE moment channels must be even: source={tuple(source.shape)} target={tuple(target.shape)}")
    old_latent = source.shape[0] // 2
    new_latent = target.shape[0] // 2
    if new_latent % old_latent != 0:
        raise ValueError(f"Cannot repeat latent {old_latent} into {new_latent}")
    repeat_count = new_latent // old_latent
    patched = target.clone()
    patched[:new_latent] = source[:old_latent].repeat(repeat_count, *([1] * (source.ndim - 1)))
    patched[new_latent:] = source[old_latent:].repeat(repeat_count, *([1] * (source.ndim - 1)))
    return patched


def _copy_zero_extra_channels(
    source: torch.Tensor,
    target: torch.Tensor,
    dim: int,
    random_extra: bool = False,
    extra_fill_scale: float = 0.0,
) -> torch.Tensor:
    if target.shape[dim] < source.shape[dim]:
        raise ValueError(f"Cannot copy shape {tuple(source.shape)} into smaller target {tuple(target.shape)} on dim {dim}")
    patched = target.clone() if random_extra else torch.zeros_like(target)
    slices = [slice(None)] * target.ndim
    slices[dim] = slice(0, source.shape[dim])
    patched[tuple(slices)] = source
    if extra_fill_scale > 0.0 and target.shape[dim] > source.shape[dim]:
        extra_slices = [slice(None)] * target.ndim
        extra_slices[dim] = slice(source.shape[dim], target.shape[dim])
        fill_value = source.detach().abs().mean().to(device=target.device, dtype=target.dtype) * float(extra_fill_scale)
        patched[tuple(extra_slices)] = fill_value
    return patched


def _copy_zero_extra_vae_moments(
    source: torch.Tensor,
    target: torch.Tensor,
    random_extra: bool = False,
    extra_fill_scale: float = 0.0,
) -> torch.Tensor:
    if source.shape[0] % 2 != 0 or target.shape[0] % 2 != 0:
        raise ValueError(f"VAE moment channels must be even: source={tuple(source.shape)} target={tuple(target.shape)}")
    old_latent = source.shape[0] // 2
    new_latent = target.shape[0] // 2
    if new_latent < old_latent:
        raise ValueError(f"Cannot copy latent {old_latent} into smaller target latent {new_latent}")

    patched = target.clone() if random_extra else torch.zeros_like(target)
    patched[:old_latent] = source[:old_latent]
    patched[new_latent:new_latent + old_latent] = source[old_latent:]
    if extra_fill_scale > 0.0 and new_latent > old_latent:
        mean_fill = source[:old_latent].detach().abs().mean().to(device=target.device, dtype=target.dtype) * float(extra_fill_scale)
        logvar_fill = source[old_latent:].detach().abs().mean().to(device=target.device, dtype=target.dtype) * float(extra_fill_scale)
        patched[old_latent:new_latent] = mean_fill
        patched[new_latent + old_latent:] = logvar_fill
    return patched


def _adapt_oobleck_state_dict(
    target_model: nn.Module,
    source_state: Mapping[str, torch.Tensor],
    init_strategy: str = "repeat",
    zero_extra_weight_g_scale: float = 0.05,
) -> Dict[str, torch.Tensor]:
    if init_strategy not in {"repeat", "zero_extra"}:
        raise ValueError(f"Unknown FOA VAE init_strategy={init_strategy!r}; expected 'repeat' or 'zero_extra'")

    target_state = target_model.state_dict()
    patched_state = {}

    for key, target_param in target_state.items():
        if key not in source_state:
            patched_state[key] = target_param
            continue

        source_param = source_state[key]
        if source_param.shape == target_param.shape:
            patched_state[key] = source_param
            continue

        if key == "encoder.conv1.weight_v":
            if init_strategy == "repeat":
                patched_state[key] = _copy_repeated_channels(source_param, target_param, dim=1)
            else:
                patched_state[key] = _copy_zero_extra_channels(source_param, target_param, dim=1)
        elif key in {"encoder.conv2.weight_v", "encoder.conv2.weight_g", "encoder.conv2.bias"}:
            if init_strategy == "repeat":
                patched_state[key] = _copy_repeated_vae_moments(source_param, target_param)
            else:
                patched_state[key] = _copy_zero_extra_vae_moments(
                    source_param,
                    target_param,
                    random_extra=key.endswith("weight_v"),
                    extra_fill_scale=zero_extra_weight_g_scale if key.endswith("weight_g") else 0.0,
                )
        elif key == "decoder.conv1.weight_v":
            if init_strategy == "repeat":
                patched_state[key] = _copy_repeated_channels(source_param, target_param, dim=1)
            else:
                patched_state[key] = _copy_zero_extra_channels(source_param, target_param, dim=1)
        elif key in {"decoder.conv2.weight_v", "decoder.conv2.weight_g", "decoder.conv2.bias"}:
            if init_strategy == "repeat":
                patched_state[key] = _copy_repeated_channels(source_param, target_param, dim=0)
            else:
                patched_state[key] = _copy_zero_extra_channels(
                    source_param,
                    target_param,
                    dim=0,
                    random_extra=key.endswith("weight_v"),
                    extra_fill_scale=zero_extra_weight_g_scale if key.endswith("weight_g") else 0.0,
                )
        else:
            raise RuntimeError(
                f"Cannot adapt stable audio VAE parameter {key}: "
                f"source={tuple(source_param.shape)} target={tuple(target_param.shape)}"
            )

    return patched_state


def _replace_encoder_moments_conv(model: nn.Module, moments_channels: int) -> None:
    old_conv = model.encoder.conv2
    in_channels = old_conv.weight_v.shape[1]
    kernel_size = old_conv.weight_v.shape[2]
    padding = old_conv.padding[0] if isinstance(old_conv.padding, tuple) else old_conv.padding
    has_bias = old_conv.bias is not None
    model.encoder.conv2 = weight_norm(
        nn.Conv1d(
            in_channels,
            moments_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=has_bias,
        )
    )


def _load_latent_expanded_oobleck(
    model_dir: Path,
    init_pretrained: bool,
    target_channels: int,
    latent_channels: int,
    init_strategy: str,
    zero_extra_weight_g_scale: float,
    verbose: bool,
):
    from diffusers import AutoencoderOobleck

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Stable audio VAE config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}

    source_channels = int(cfg.get("audio_channels", 2))
    if target_channels % source_channels != 0:
        raise ValueError(f"target_channels={target_channels} must be divisible by source_channels={source_channels}")

    source_latent_channels = int(cfg.get("decoder_input_channels", 64))
    if latent_channels % source_latent_channels != 0:
        raise ValueError(
            f"latent_channels={latent_channels} must be divisible by source latent channels={source_latent_channels}"
        )

    cfg["audio_channels"] = target_channels
    cfg["decoder_input_channels"] = latent_channels
    model = AutoencoderOobleck(**cfg)
    _replace_encoder_moments_conv(model, moments_channels=latent_channels * 2)

    if not init_pretrained:
        return model

    if verbose:
        print(
            "| Loading stable audio AutoencoderOobleck weights from "
            f"'{model_dir}' into {target_channels}ch/{latent_channels}latent model "
            f"(init_strategy={init_strategy}, zero_extra_weight_g_scale={zero_extra_weight_g_scale})"
        )
    source_model = AutoencoderOobleck.from_pretrained(str(model_dir))
    patched_state = _adapt_oobleck_state_dict(
        model,
        source_model.state_dict(),
        init_strategy=init_strategy,
        zero_extra_weight_g_scale=zero_extra_weight_g_scale,
    )
    model.load_state_dict(patched_state, strict=True)
    return model


def build_foa_wavvae(hparams=None, init_pretrained: bool = True, verbose: bool = True):
    model_dir = Path(_nested_get(hparams, "foa_vae", "pretrained_model_dir", "checkpoints/vae"))
    input_channels = int(_nested_get(hparams, "foa_vae", "input_channels", 4))
    output_channels = int(_nested_get(hparams, "foa_vae", "output_channels", 4))
    if input_channels != output_channels:
        raise ValueError(f"wavvae_v2 expects input_channels == output_channels, got {input_channels} and {output_channels}")
    latent_channels = int(_nested_get(hparams, "foa_vae", "latent_channels", 128))
    init_strategy = str(_nested_get(hparams, "foa_vae", "init_strategy", "repeat"))
    zero_extra_weight_g_scale = float(_nested_get(hparams, "foa_vae", "zero_extra_weight_g_scale", 0.05))

    autoencoder = _load_latent_expanded_oobleck(
        model_dir=model_dir,
        init_pretrained=init_pretrained,
        target_channels=input_channels,
        latent_channels=latent_channels,
        init_strategy=init_strategy,
        zero_extra_weight_g_scale=zero_extra_weight_g_scale,
        verbose=verbose,
    )
    return FOAWavVAE(autoencoder=autoencoder, input_channels=input_channels, output_channels=output_channels)
