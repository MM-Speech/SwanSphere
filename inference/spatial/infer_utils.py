"""Shared checkpoint, configuration and audio-output helpers for spatial inference."""

import os
import re
from pathlib import Path
from typing import Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_repo_path(path: str) -> str:
    path = os.path.expanduser(str(path))
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return path
    return str(PROJECT_ROOT / candidate)


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


def load_inference_config(config_path: str) -> Dict:
    config_path = _resolve_repo_path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config


def autocast_dtype(precision: str):
    import torch

    if precision in ("bf16", "bfloat16"):
        return torch.bfloat16
    if precision in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


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
