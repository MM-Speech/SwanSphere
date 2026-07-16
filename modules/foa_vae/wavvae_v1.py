import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


FOA_WYZX = {"w": 0, "y": 1, "z": 2, "x": 3}
PHYSICAL_XYZ_IN_WYZX = (FOA_WYZX["x"], FOA_WYZX["y"], FOA_WYZX["z"])
DEFAULT_STEREO_PROJECTION_PAIRS = [
    {
        "name": "front_back",
        "left": [1.0, 0.0, 0.0],
        "right": [-1.0, 0.0, 0.0],
    },
    {
        "name": "left_right",
        "left": [0.0, 1.0, 0.0],
        "right": [0.0, -1.0, 0.0],
    },
    {
        "name": "up_down",
        "left": [0.0, 0.0, 1.0],
        "right": [0.0, 0.0, -1.0],
    },
]


def _nested_get(hparams, section: str, key: str, default=None):
    if hparams is None:
        return default
    section_value = hparams.get(section, {}) if hasattr(hparams, "get") else {}
    if isinstance(section_value, Mapping) and key in section_value:
        return section_value[key]
    return hparams.get(key, default) if hasattr(hparams, "get") else default


def linear_ramp(
    step: int,
    start: int,
    duration: int,
    start_value: float = 0.0,
    end_value: float = 1.0,
) -> float:
    if duration <= 0:
        return float(end_value if step >= start else start_value)
    progress = min(max((step - start) / float(duration), 0.0), 1.0)
    return float(start_value + progress * (end_value - start_value))


def scheduled_value(value, step: int) -> float:
    if isinstance(value, Mapping):
        return linear_ramp(
            step,
            start=int(value.get("start", value.get("start_step", 0))),
            duration=int(value.get("duration", value.get("ramp_steps", 1))),
            start_value=float(value.get("start_value", value.get("from", 0.0))),
            end_value=float(value.get("end_value", value.get("to", 0.0))),
        )
    return float(value)


def _projector_activation(name: str) -> nn.Module:
    name = (name or "silu").lower()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name in {"identity", "none"}:
        return nn.Identity()
    raise ValueError(f"Unsupported projector activation: {name}")


class ChannelMLPProjector(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        activation: str = "silu",
        use_skip: bool = True,
    ):
        super().__init__()
        if use_skip and in_channels != out_channels:
            raise ValueError("Residual ChannelMLPProjector requires in_channels == out_channels")
        self.fc_in = nn.Linear(in_channels, hidden_channels)
        self.fc_hidden = nn.Linear(hidden_channels, hidden_channels)
        self.fc_out = nn.Linear(hidden_channels, out_channels)
        self.activation = _projector_activation(activation)
        self.use_skip = use_skip
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        hidden = audio.transpose(1, 2)
        hidden = self.activation(self.fc_in(hidden))
        hidden = self.activation(self.fc_hidden(hidden))
        out = self.fc_out(hidden).transpose(1, 2)
        return audio + out if self.use_skip else out


class FOAWavVAEWithProjectors(nn.Module):
    def __init__(
        self,
        autoencoder: nn.Module,
        input_channels: int = 4,
        output_channels: int = 4,
        projector_hidden_channels: int = 64,
        projector_activation: str = "silu",
        projector_use_skip: bool = True,
    ):
        super().__init__()
        if input_channels != output_channels:
            raise ValueError("FOA projector wrapper expects input_channels == output_channels")

        self.input_projector = ChannelMLPProjector(
            input_channels,
            input_channels,
            hidden_channels=projector_hidden_channels,
            activation=projector_activation,
            use_skip=projector_use_skip,
        )
        self.autoencoder = autoencoder
        self.output_projector = ChannelMLPProjector(
            output_channels,
            output_channels,
            hidden_channels=projector_hidden_channels,
            activation=projector_activation,
            use_skip=projector_use_skip,
        )
        self.in_channels = input_channels
        self.out_channels = output_channels
        self.io_channels = output_channels

    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        projected = self.input_projector(audio)
        encoded = self.autoencoder.encode(projected)
        latent_dist = encoded.latent_dist
        latents = latent_dist.sample()
        decoded = self.autoencoder.decode(latents)
        recon = self.output_projector(decoded.sample)

        mu = latent_dist.mean
        logvar = latent_dist.logvar
        if hasattr(latent_dist, "kl"):
            kl = latent_dist.kl().mean()
        else:
            kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0).mean()

        return {
            "recon": recon,
            "kl": kl,
            "mu": mu,
            "logvar": logvar,
        }


def wyzx_components(audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if audio.ndim != 3 or audio.shape[1] != 4:
        raise ValueError(f"Expected FOA tensor [B, 4, T] in WYZX order, got {tuple(audio.shape)}")
    w = audio[:, FOA_WYZX["w"] : FOA_WYZX["w"] + 1]
    y = audio[:, FOA_WYZX["y"] : FOA_WYZX["y"] + 1]
    z = audio[:, FOA_WYZX["z"] : FOA_WYZX["z"] + 1]
    x = audio[:, FOA_WYZX["x"] : FOA_WYZX["x"] + 1]
    return w, y, z, x


def project_foa_wyzx(
    audio: torch.Tensor,
    directions: torch.Tensor,
    gain: float = 1.0,
    fold_batch: bool = True,
) -> torch.Tensor:
    w, y, z, x = wyzx_components(audio)
    dirs = directions.to(device=audio.device, dtype=audio.dtype)
    if dirs.ndim != 2 or dirs.shape[1] != 3:
        raise ValueError(f"Expected directions [K, 3], got {tuple(dirs.shape)}")
    dirs = F.normalize(dirs, dim=-1, eps=1.0e-8)

    ux = dirs[:, 0].view(1, -1, 1)
    uy = dirs[:, 1].view(1, -1, 1)
    uz = dirs[:, 2].view(1, -1, 1)
    projected = w + gain * (ux * x + uy * y + uz * z)

    if fold_batch:
        return projected.reshape(audio.shape[0] * dirs.shape[0], 1, audio.shape[-1])
    return projected


def _direction_pair_from_config(pair) -> List[List[float]]:
    if isinstance(pair, Mapping):
        return [pair["left"], pair["right"]]
    if isinstance(pair, Sequence) and len(pair) == 2:
        return [pair[0], pair[1]]
    raise ValueError(f"Invalid stereo projection pair: {pair}")


def build_stereo_projection_pairs(
    pairs: Sequence = None,
    device=None,
    dtype=None,
) -> torch.Tensor:
    pair_cfg = pairs or DEFAULT_STEREO_PROJECTION_PAIRS
    dirs = [_direction_pair_from_config(pair) for pair in pair_cfg]
    out = torch.tensor(dirs, device=device, dtype=dtype or torch.float32)
    if out.ndim != 3 or out.shape[1:] != (2, 3):
        raise ValueError(f"Expected stereo projection pairs [P, 2, 3], got {tuple(out.shape)}")
    return F.normalize(out, dim=-1, eps=1.0e-8)


def project_foa_stereo_wyzx(
    audio: torch.Tensor,
    pairs: torch.Tensor,
    gain: float = 1.0,
    fold_batch: bool = True,
) -> torch.Tensor:
    if pairs.ndim != 3 or pairs.shape[1:] != (2, 3):
        raise ValueError(f"Expected stereo projection pairs [P, 2, 3], got {tuple(pairs.shape)}")
    bsz = audio.shape[0]
    pair_count = pairs.shape[0]
    flat_dirs = pairs.reshape(pair_count * 2, 3)
    projected = project_foa_wyzx(audio, flat_dirs, gain=gain, fold_batch=False)
    stereo = projected.reshape(bsz, pair_count, 2, audio.shape[-1])
    if fold_batch:
        return stereo.reshape(bsz * pair_count, 2, audio.shape[-1])
    return stereo


def build_projection_directions(
    random_count: int = 0,
    include_axes: bool = True,
    include_corners: bool = True,
    device=None,
    dtype=None,
) -> torch.Tensor:
    dirs: List[List[float]] = []
    if include_axes:
        dirs.extend(
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
                    dirs.append([sx, sy, sz])

    out = torch.tensor(dirs, device=device, dtype=dtype or torch.float32)
    if out.numel() == 0:
        out = torch.empty(0, 3, device=device, dtype=dtype or torch.float32)
    if random_count > 0:
        rand = torch.randn(random_count, 3, device=device, dtype=out.dtype)
        rand = F.normalize(rand, dim=-1, eps=1.0e-8)
        out = torch.cat([out, rand], dim=0)
    return F.normalize(out, dim=-1, eps=1.0e-8)


def _stft_foa(
    audio: torch.Tensor,
    fft_size: int = 1024,
    hop_size: int = 256,
    win_length: int = 1024,
) -> torch.Tensor:
    if audio.ndim != 3 or audio.shape[1] != 4:
        raise ValueError(f"Expected FOA tensor [B, 4, T], got {tuple(audio.shape)}")
    bsz, channels, length = audio.shape
    window = torch.hann_window(win_length, device=audio.device, dtype=torch.float32)
    flat_audio = audio.float().reshape(bsz * channels, length)
    spec = torch.stft(
        flat_audio,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    return spec.reshape(bsz, channels, spec.shape[-2], spec.shape[-1])


def _active_intensity(spec: torch.Tensor) -> torch.Tensor:
    w = spec[:, FOA_WYZX["w"]]
    x = spec[:, FOA_WYZX["x"]]
    y = spec[:, FOA_WYZX["y"]]
    z = spec[:, FOA_WYZX["z"]]
    return torch.stack(
        [
            torch.real(torch.conj(w) * x),
            torch.real(torch.conj(w) * y),
            torch.real(torch.conj(w) * z),
        ],
        dim=1,
    )


def active_intensity_direction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    fft_size: int = 1024,
    hop_size: int = 256,
    win_length: int = 1024,
    energy_percentile: float = 0.0,
    eps: float = 1.0e-7,
) -> torch.Tensor:
    pred_spec = _stft_foa(pred, fft_size, hop_size, win_length)
    target_spec = _stft_foa(target, fft_size, hop_size, win_length)
    pred_i = _active_intensity(pred_spec)
    target_i = _active_intensity(target_spec)

    pred_dir = F.normalize(pred_i, dim=1, eps=eps)
    target_dir = F.normalize(target_i, dim=1, eps=eps)
    loss = 1.0 - (pred_dir * target_dir).sum(dim=1).clamp(-1.0, 1.0)

    w_energy = target_spec[:, FOA_WYZX["w"]].abs().pow(2)
    if energy_percentile > 0:
        threshold = torch.quantile(w_energy.detach().flatten(), float(energy_percentile))
        mask = w_energy > threshold
    else:
        mask = w_energy > eps
    weights = w_energy * mask.to(w_energy.dtype)
    denom = weights.sum().clamp_min(eps)
    return (loss * weights).sum() / denom


def _normalized_cross(spec: torch.Tensor, first: int, second: int, eps: float) -> torch.Tensor:
    cross = torch.conj(spec[:, first]) * spec[:, second]
    return cross / cross.abs().clamp_min(eps)


def normalized_cross_spectrum_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    fft_size: int = 1024,
    hop_size: int = 256,
    win_length: int = 1024,
    eps: float = 1.0e-7,
) -> torch.Tensor:
    pred_spec = _stft_foa(pred, fft_size, hop_size, win_length)
    target_spec = _stft_foa(target, fft_size, hop_size, win_length)
    pairs = (
        (FOA_WYZX["w"], FOA_WYZX["x"]),
        (FOA_WYZX["w"], FOA_WYZX["y"]),
        (FOA_WYZX["w"], FOA_WYZX["z"]),
        (FOA_WYZX["x"], FOA_WYZX["y"]),
        (FOA_WYZX["x"], FOA_WYZX["z"]),
        (FOA_WYZX["y"], FOA_WYZX["z"]),
    )
    losses = []
    for first, second in pairs:
        pred_cross = _normalized_cross(pred_spec, first, second, eps)
        target_cross = _normalized_cross(target_spec, first, second, eps)
        losses.append(F.l1_loss(torch.view_as_real(pred_cross), torch.view_as_real(target_cross)))
    return torch.stack(losses).mean()


def spatial_covariance_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    fft_size: int = 1024,
    hop_size: int = 256,
    win_length: int = 1024,
    eps: float = 1.0e-7,
) -> torch.Tensor:
    pred_spec = _stft_foa(pred, fft_size, hop_size, win_length)[:, [0, 3, 1, 2]]
    target_spec = _stft_foa(target, fft_size, hop_size, win_length)[:, [0, 3, 1, 2]]

    def normalized_scm(spec: torch.Tensor) -> torch.Tensor:
        spec = spec.permute(0, 2, 3, 1)
        scm = spec.unsqueeze(-1) * torch.conj(spec.unsqueeze(-2))
        denom = scm.abs().pow(2).sum(dim=(-1, -2), keepdim=True).sqrt().clamp_min(eps)
        return scm / denom

    pred_scm = normalized_scm(pred_spec)
    target_scm = normalized_scm(target_spec)
    return F.l1_loss(torch.view_as_real(pred_scm), torch.view_as_real(target_scm))


def unwrap_training_module(model):
    return model.module if hasattr(model, "module") else model


def _base_autoencoder_with_prefix(model):
    model = unwrap_training_module(model)
    if hasattr(model, "autoencoder"):
        return model.autoencoder, "autoencoder."
    return model, ""


def _block_names(module) -> List[str]:
    if hasattr(module, "block"):
        return [name for name, _ in module.block.named_children()]
    return []


def adapter_param_prefixes(model) -> List[str]:
    model = unwrap_training_module(model)
    if hasattr(model, "input_projector") and hasattr(model, "output_projector"):
        return ["input_projector", "output_projector"]
    return []


def partial_unfreeze_prefixes(model, encoder_last_n: int = 2, decoder_last_n: int = 2) -> List[str]:
    prefixes = adapter_param_prefixes(model)
    base_model, base_prefix = _base_autoencoder_with_prefix(model)
    enc_block_names = _block_names(base_model.encoder)
    dec_block_names = _block_names(base_model.decoder)
    prefixes.extend([f"{base_prefix}encoder.block.{name}" for name in enc_block_names[-encoder_last_n:]])
    prefixes.extend([f"{base_prefix}decoder.block.{name}" for name in dec_block_names[-decoder_last_n:]])
    return prefixes


def set_trainable_by_prefixes(model, prefixes: Sequence[str], train_all: bool = False) -> None:
    model = unwrap_training_module(model)
    for name, param in model.named_parameters():
        param.requires_grad = train_all or any(name.startswith(prefix) for prefix in prefixes)


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


def trainable_param_report(model) -> int:
    total = 0
    trainable = 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    print(f"Total params: {total:,}")
    print(f"Trainable params (requires_grad=True): {trainable:,}")
    return trainable


def _load_channel_copied_oobleck(model_dir: Path, init_pretrained: bool, target_channels: int, verbose: bool):
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
    cfg["audio_channels"] = target_channels
    model = AutoencoderOobleck(**cfg)

    if not init_pretrained:
        return model

    if verbose:
        print(f"| Loading stable audio AutoencoderOobleck weights from '{model_dir}' into {target_channels}ch model")
    source_model = AutoencoderOobleck.from_pretrained(str(model_dir))
    source_state = source_model.state_dict()
    target_state = model.state_dict()
    repeat_count = target_channels // source_channels
    patched_state = {}
    for key, target_param in target_state.items():
        if key not in source_state:
            patched_state[key] = target_param
            continue
        source_param = source_state[key]
        if source_param.shape == target_param.shape:
            patched_state[key] = source_param
        elif key == "encoder.conv1.weight_v" and source_param.shape[1] == source_channels:
            patched_state[key] = source_param.repeat(1, repeat_count, 1)
        elif key == "decoder.conv2.weight_v" and source_param.shape[0] == source_channels:
            patched_state[key] = source_param.repeat(repeat_count, 1, 1)
        elif key == "decoder.conv2.weight_g" and source_param.shape[0] == source_channels:
            patched_state[key] = source_param.repeat(repeat_count, 1, 1)
        else:
            raise RuntimeError(
                f"Cannot adapt stable audio VAE parameter {key}: "
                f"source={tuple(source_param.shape)} target={tuple(target_param.shape)}"
            )
    model.load_state_dict(patched_state, strict=True)
    return model


def build_foa_wavvae(hparams=None, init_pretrained: bool = True, verbose: bool = True):
    model_dir = Path(_nested_get(hparams, "foa_vae", "pretrained_model_dir", "checkpoints/vae"))
    input_channels = int(_nested_get(hparams, "foa_vae", "input_channels", 4))
    autoencoder = _load_channel_copied_oobleck(
        model_dir=model_dir,
        init_pretrained=init_pretrained,
        target_channels=input_channels,
        verbose=verbose,
    )

    return FOAWavVAEWithProjectors(
        autoencoder=autoencoder,
        input_channels=input_channels,
        output_channels=int(_nested_get(hparams, "foa_vae", "output_channels", 4)),
        projector_hidden_channels=int(_nested_get(hparams, "foa_vae", "projector_hidden_channels", 64)),
        projector_activation=_nested_get(hparams, "foa_vae", "projector_activation", "silu"),
        projector_use_skip=bool(_nested_get(hparams, "foa_vae", "projector_use_skip", True)),
    )
