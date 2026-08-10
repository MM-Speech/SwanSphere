import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.foa_vae.wavvae_v4 import build_foa_wavvae_v4


DEFAULT_CONFIG = "egs/inference/inference_foa_vae_v4.yaml"
DEFAULT_PATH_FIELDS = [
    "target_wav_path",
    "wav_path",
    "foa_path",
    "audio_path",
    "target_path",
    "path",
    "file_path",
]


def _as_list(value):
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _resolve_repo_path(path: str) -> str:
    path = os.path.expanduser(str(path))
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return path
    return str(PROJECT_ROOT / candidate)


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _resolve_with_roots(path: str, roots: Optional[Iterable[str]] = None) -> str:
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return path
    for root in _as_list(roots):
        candidate = os.path.join(os.path.expanduser(str(root)), path)
        if os.path.exists(candidate):
            return candidate
    return path


def _first_path_from_item(item: Dict, fields: List[str]) -> Optional[str]:
    for field in fields:
        value = item.get(field)
        if value:
            return str(value)
    return None


def collect_input_paths(config: Dict) -> List[str]:
    paths: List[str] = []
    roots = config.get("input_roots", config.get("audio_roots", None))

    for path in _as_list(config.get("input_path")) + _as_list(config.get("input_paths")):
        paths.append(_resolve_with_roots(str(path), roots))

    for list_path in _as_list(config.get("input_list")):
        list_path = _resolve_repo_path(str(list_path))
        for path in _read_lines(str(list_path)):
            paths.append(_resolve_with_roots(path, roots))

    metadata_path = config.get("metadata_path", "")
    if metadata_path:
        metadata_path = _resolve_repo_path(str(metadata_path))
        fields = list(config.get("path_fields", DEFAULT_PATH_FIELDS))
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                path = _first_path_from_item(item, fields)
                if path:
                    item_roots = []
                    item_roots.extend(_as_list(roots))
                    item_roots.extend(_as_list(item.get("audio_root")))
                    item_roots.extend(_as_list(item.get("root")))
                    item_roots.extend(_as_list(item.get("base_dir")))
                    paths.append(_resolve_with_roots(path, item_roots))

    if config.get("deduplicate_inputs", False):
        seen = set()
        deduped = []
        for path in paths:
            if path not in seen:
                seen.add(path)
                deduped.append(path)
        paths = deduped

    return paths


def resolve_output_wav_path(
    input_path: str,
    out_path: str,
    input_index: int,
    num_inputs: int,
    suffix: str = "_recon",
) -> str:
    out_path = os.path.expanduser(str(out_path))
    if num_inputs == 1 and out_path.lower().endswith(".wav"):
        return out_path

    stem = Path(input_path).stem or f"sample_{input_index:06d}"
    filename = f"{stem}{suffix}.wav"
    return str(Path(out_path) / filename)


def make_unique_path(path: str, used_paths: set) -> str:
    if path not in used_paths:
        used_paths.add(path)
        return path

    base = Path(path)
    for idx in range(1, 1000000):
        candidate = str(base.with_name(f"{base.stem}_{idx:04d}{base.suffix}"))
        if candidate not in used_paths:
            used_paths.add(candidate)
            return candidate
    raise RuntimeError(f"Unable to make unique output path for {path}")


def resolve_reference_audio_path(output_path: str, input_path: str, recon_suffix: str, reference_suffix: str) -> str:
    output = Path(output_path)
    input_suffix = Path(input_path).suffix or ".wav"
    stem = output.stem
    if recon_suffix and stem.endswith(recon_suffix):
        stem = stem[: -len(recon_suffix)]
    return str(output.with_name(f"{stem}{reference_suffix}{input_suffix}"))


def resolve_checkpoint_path(ckpt_path: str, steps: Optional[int] = None, prefer_model_only_last: bool = True) -> str:
    ckpt_path = os.path.expanduser(str(ckpt_path))
    if os.path.isfile(ckpt_path):
        return ckpt_path
    if not os.path.isdir(ckpt_path):
        raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path}")

    if steps:
        candidate = os.path.join(ckpt_path, f"model_ckpt_steps_{int(steps)}.ckpt")
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(f"Checkpoint step {steps} not found under {ckpt_path}")

    model_only = os.path.join(ckpt_path, "model_only_last.ckpt")
    if prefer_model_only_last and os.path.exists(model_only):
        return model_only

    pattern = re.compile(r"model_ckpt_steps_(\d+)\.ckpt$")
    candidates = []
    for child in Path(ckpt_path).glob("model_ckpt_steps_*.ckpt"):
        match = pattern.search(child.name)
        if match:
            candidates.append((int(match.group(1)), str(child)))
    if not candidates:
        raise FileNotFoundError(f"No model checkpoint found under {ckpt_path}")
    return max(candidates, key=lambda item: item[0])[1]


EMA_STATE_KEYS = {
    "decay",
    "min_decay",
    "optimization_step",
    "update_after_step",
    "use_ema_warmup",
    "inv_gamma",
    "power",
}


def resolve_ema_checkpoint_path(resolved_checkpoint: str, explicit_path: str = "") -> str:
    if explicit_path:
        candidate = Path(_resolve_repo_path(explicit_path))
    else:
        checkpoint = Path(resolved_checkpoint)
        candidate = checkpoint.with_name(f"{checkpoint.stem}_model_gen_ema.ckpt")
    if not candidate.is_file():
        raise FileNotFoundError(f"EMA checkpoint not found: {candidate}")
    return str(candidate)


def load_ema_parameters_strict(model, ema_checkpoint: str, model_name: str = "model_gen_ema") -> int:
    import torch

    checkpoint = torch.load(ema_checkpoint, map_location="cpu", mmap=True, weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    if model_name in state:
        state = state[model_name]
    if not isinstance(state, dict):
        raise TypeError(f"EMA state must be a dict, got {type(state).__name__}")

    shadow_keys = sorted(
        (key for key in state if key.startswith("shadow_params.")),
        key=lambda key: int(key.rsplit(".", 1)[-1]),
    )
    expected_shadow_keys = [f"shadow_params.{idx:06d}" for idx in range(len(shadow_keys))]
    if shadow_keys != expected_shadow_keys:
        raise RuntimeError("EMA shadow parameter indices are missing, duplicated, or non-contiguous")

    state_keys = set(state)
    unexpected_keys = state_keys - EMA_STATE_KEYS - set(shadow_keys)
    missing_state_keys = EMA_STATE_KEYS - state_keys
    if unexpected_keys or missing_state_keys:
        raise RuntimeError(
            "EMA state metadata mismatch: "
            f"missing={sorted(missing_state_keys)}, unexpected={sorted(unexpected_keys)}"
        )

    parameters = list(model.parameters())
    if len(shadow_keys) != len(parameters):
        raise RuntimeError(
            f"EMA shadow parameter count mismatch: checkpoint={len(shadow_keys)}, model={len(parameters)}"
        )

    with torch.no_grad():
        for index, (key, parameter) in enumerate(zip(shadow_keys, parameters)):
            shadow = state[key]
            if not torch.is_tensor(shadow):
                raise TypeError(f"EMA value {key} is not a tensor: {type(shadow).__name__}")
            if shadow.shape != parameter.shape:
                raise RuntimeError(
                    f"EMA shape mismatch at index {index}: checkpoint={tuple(shadow.shape)}, "
                    f"model={tuple(parameter.shape)}"
                )
            parameter.copy_(shadow.to(device=parameter.device, dtype=parameter.dtype))
    return len(parameters)


def load_inference_config(config_path: str) -> Dict:
    config_path = _resolve_repo_path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config


def apply_dict_override(base: Dict, override: Dict):
    from utils.commons.hparams import override_config

    if override:
        override_config(base, override)
    return base


def load_model_hparams(config: Dict) -> Dict:
    from utils.commons.hparams import hparams, set_hparams

    model_config = str(config.get("model_config", "") or "").strip()
    if not model_config:
        model_config = "egs/foa_vae/wavvae_v4.yaml"
    model_config = _resolve_repo_path(model_config)
    config["resolved_model_config"] = model_config
    model_hparams = set_hparams(config=model_config, print_hparams=False, global_hparams=True)
    apply_dict_override(model_hparams, config.get("model_hparams", {}))
    hparams.clear()
    hparams.update(model_hparams)
    return model_hparams


def build_model(config: Dict, model_hparams: Dict, device: str):
    import torch

    from utils.commons.ckpt_utils import load_ckpt

    print("| FOA VAE inference model builder: modules.foa_vae.wavvae_v4.build_foa_wavvae_v4")
    model = build_foa_wavvae_v4(hparams=model_hparams, init_pretrained=False)
    ckpt_path = config.get("ckpt_path", config.get("load_ckpt", ""))
    if not ckpt_path:
        raise ValueError("Inference config must provide ckpt_path or load_ckpt.")
    ckpt_path = _resolve_repo_path(ckpt_path)
    resolved_ckpt = resolve_checkpoint_path(
        ckpt_path,
        steps=config.get("ckpt_steps", None),
        prefer_model_only_last=bool(config.get("prefer_model_only_last", True)),
    )
    print(f"| FOA VAE inference model config: {config.get('resolved_model_config', '')}")
    print("| FOA VAE checkpoint strict: True")
    load_ckpt(
        model,
        resolved_ckpt,
        config.get("model_name", "model_gen"),
        strict=True,
        force=True,
        map_location="cpu",
    )
    config["resolved_base_checkpoint"] = resolved_ckpt
    if bool(config.get("use_ema", False)):
        ema_checkpoint = resolve_ema_checkpoint_path(
            resolved_ckpt,
            explicit_path=str(config.get("ema_ckpt_path", "") or ""),
        )
        print("| FOA VAE EMA checkpoint strict: True")
        parameter_count = load_ema_parameters_strict(
            model,
            ema_checkpoint,
            model_name=str(config.get("ema_model_name", "model_gen_ema")),
        )
        print(f"| loaded EMA parameters from '{ema_checkpoint}'.")
        print(f"| EMA parameters: {parameter_count}, Missing keys: 0, Unexpected keys: 0")
        config["resolved_ema_checkpoint"] = ema_checkpoint
        resolved_ckpt = ema_checkpoint
    model.eval()
    model.to(torch.device(device))
    return model, resolved_ckpt


def _normalize_audio(wav, mode: str, eps: float = 1.0e-8):
    if mode in ("", None, "none", False):
        return wav
    if mode == "peak":
        return wav / wav.abs().amax().clamp_min(eps)
    if mode == "rms":
        return wav / wav.square().mean().sqrt().clamp_min(eps)
    raise ValueError(f"Unsupported normalize_mode: {mode}")


def load_audio(path: str, sample_rate: int, expected_channels: int, normalize_mode: str):
    import torch
    import torchaudio

    try:
        wav, sr = torchaudio.load(path)
    except Exception as exc:
        wav, sr = load_audio_with_ffmpeg(path, sample_rate=sample_rate, channels=expected_channels)
        print(f"| Loaded audio with ffmpeg fallback: {path} ({exc})")
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] != expected_channels:
        raise ValueError(f"Expected {expected_channels} channels, got {wav.shape[0]}: {path}")
    wav = wav.to(torch.float32)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
        sr = sample_rate
    wav = _normalize_audio(wav, normalize_mode)
    return wav, sr


def load_audio_with_ffmpeg(path: str, sample_rate: int, channels: int):
    import subprocess

    import numpy as np
    import torch

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        path,
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size % channels != 0:
        raise RuntimeError(f"Decoded audio size {audio.size} is not divisible by {channels}: {path}")
    wav = torch.from_numpy(audio.reshape(-1, channels).T.copy())
    return wav, sample_rate


def model_downsample_factor(model) -> int:
    autoencoder = getattr(model, "autoencoder", model)
    config = getattr(autoencoder, "config", None)
    ratios = getattr(config, "downsampling_ratios", None) if config is not None else None
    factor = 1
    if ratios:
        for ratio in ratios:
            factor *= int(ratio)
    return max(1, factor)


def pad_to_multiple(wav, multiple: int):
    import torch.nn.functional as F

    multiple = max(1, int(multiple))
    length = wav.shape[-1]
    pad = (-length) % multiple
    if pad == 0:
        return wav, length
    return F.pad(wav, (0, pad), "constant", 0.0), length


def autocast_dtype(precision: str):
    import torch

    if precision in ("bf16", "bfloat16"):
        return torch.bfloat16
    if precision in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


def reconstruct(model, wav, device: str, precision: str, pad_multiple):
    import torch

    original_len = wav.shape[-1]
    if pad_multiple in ("auto", True):
        multiple = model_downsample_factor(model)
    elif pad_multiple in ("", None, False):
        multiple = 1
    else:
        multiple = int(pad_multiple)
    wav, _ = pad_to_multiple(wav, multiple)

    audio = wav.unsqueeze(0).to(device)
    use_autocast = str(device).startswith("cuda") and precision not in ("fp32", "float32", "none", "")
    dtype = autocast_dtype(precision)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_autocast):
            outputs = model(audio)
    recon = outputs["recon"].detach().float().cpu().squeeze(0)
    return recon[..., :original_len]


def save_audio(path: str, wav, sample_rate: int, config: Dict):
    import torch
    import torchaudio

    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    wav = wav.detach().cpu().to(torch.float32)
    if config.get("clip_output", False):
        wav = wav.clamp(-1.0, 1.0)
    encoding = config.get("save_encoding", "PCM_F")
    bits_per_sample = int(config.get("bits_per_sample", 32))
    torchaudio.save(path, wav, sample_rate=sample_rate, encoding=encoding, bits_per_sample=bits_per_sample)


def run_inference(config_path: str):
    import torch

    os.chdir(PROJECT_ROOT)
    config = load_inference_config(config_path)
    inputs = collect_input_paths(config)
    if not inputs:
        raise ValueError("No inference inputs found. Set input_paths, input_list, or metadata_path.")
    out_path = config.get("out_path", "")
    if not out_path:
        raise ValueError("Inference config must provide out_path.")
    out_path = _resolve_repo_path(out_path)
    config["out_path"] = out_path

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    precision = config.get("precision", "bf16")
    model_hparams = load_model_hparams(config)
    model, resolved_ckpt = build_model(config, model_hparams, device)

    sample_rate = int(config.get("sample_rate", model_hparams.get("sample_rate", 44100)))
    expected_channels = int(config.get("expected_channels", model_hparams.get("foa_vae", {}).get("input_channels", 4)))
    normalize_mode = config.get("normalize_mode", config.get("foa_normalize_mode", "none"))
    suffix = config.get("output_suffix", "_recon")
    reference_suffix = config.get("input_copy_suffix", "_input")
    copy_input_audio = bool(config.get("copy_input_audio", True))
    overwrite = bool(config.get("overwrite", True))
    pad_multiple = config.get("pad_to_multiple", "auto")

    used_outputs = set()
    print(f"| FOA VAE inference config: {config_path}")
    print(f"| Loaded checkpoint: {resolved_ckpt}")
    print(f"| Inputs: {len(inputs)}")
    for idx, input_path in enumerate(inputs):
        output_path = resolve_output_wav_path(input_path, out_path, idx, len(inputs), suffix=suffix)
        output_path = make_unique_path(output_path, used_outputs)
        reference_path = None
        if copy_input_audio:
            reference_path = resolve_reference_audio_path(output_path, input_path, suffix, reference_suffix)
            reference_path = make_unique_path(reference_path, used_outputs)
        if os.path.exists(output_path) and not overwrite:
            print(f"| [{idx + 1}/{len(inputs)}] skip existing: {output_path}")
            continue
        wav, sr = load_audio(input_path, sample_rate, expected_channels, normalize_mode)
        recon = reconstruct(model, wav, device=device, precision=precision, pad_multiple=pad_multiple)
        save_audio(output_path, recon, sr, config)
        if reference_path is not None and (overwrite or not os.path.exists(reference_path)):
            os.makedirs(os.path.dirname(reference_path), exist_ok=True)
            shutil.copy2(input_path, reference_path)
        print(f"| [{idx + 1}/{len(inputs)}] {input_path} -> {output_path}  shape={tuple(recon.shape)} sr={sr}")
        if reference_path is not None:
            print(f"| [{idx + 1}/{len(inputs)}] copied input audio to: {reference_path}")


def main():
    parser = argparse.ArgumentParser(description="Reconstruct FOA wavs with a trained FOA Stable Audio VAE.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to FOA VAE inference yaml.")
    args = parser.parse_args()
    run_inference(args.config)


if __name__ == "__main__":
    main()
