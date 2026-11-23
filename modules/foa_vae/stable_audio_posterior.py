"""Validated per-example Stable Audio pair posterior caches.

mean/std are posterior parameters [128, 215], NOT dataset normalization stats.
All caches retain the original FOA file's first-pair/second-pair channel order.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

SCHEMA_VERSION = 1
LATENT_DIM = 128
LATENT_FRAMES = 215
SAMPLE_RATE = 44100
SAMPLES = 441000
DOWNSAMPLE_FACTOR = 2048
CACHE_INFO_NAME = "cache_info.json"


def load_foa_waveform(path, sample_rate=SAMPLE_RATE, audio_seconds=10.0, min_audio_ratio=0.95):
    """Exactly the v8 online crop/resample/pad contract; no amplitude normalization."""
    import soundfile as sf
    from scipy.signal import resample_poly
    import torch

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing original FOA audio: {path}")
    info = sf.info(path)
    if info.channels != 4 or info.samplerate <= 0:
        raise ValueError(f"Expected four-channel FOA audio with a valid sample rate: {path}")
    source_samples = int(round(audio_seconds * info.samplerate))
    waveform, source_rate = sf.read(
        path, frames=source_samples, dtype="float32", always_2d=True
    )
    if waveform.shape[0] < int(source_samples * min_audio_ratio):
        raise ValueError(f"FOA audio is too short for a {audio_seconds}-second clip: {path}")
    if not np.isfinite(waveform).all():
        raise ValueError(f"FOA audio contains NaN or Inf: {path}")
    if source_rate != sample_rate:
        divisor = math.gcd(int(source_rate), int(sample_rate))
        waveform = resample_poly(
            waveform, sample_rate // divisor, source_rate // divisor, axis=0
        )
    target_samples = int(round(audio_seconds * sample_rate))
    valid_samples = min(waveform.shape[0], target_samples)
    result = np.zeros((4, target_samples), dtype=np.float32)
    result[:, :valid_samples] = waveform[:valid_samples].T
    if not np.isfinite(result).all():
        raise ValueError(f"Resampled FOA audio contains NaN or Inf: {path}")
    return torch.from_numpy(result), valid_samples


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_cache_info(model_dir, min_audio_ratio=0.95):
    """Content identity is independent of mount location and metadata sharding."""
    model_dir = Path(model_dir).expanduser().resolve()
    weight = model_dir / "diffusion_pytorch_model.safetensors"
    config = model_dir / "config.json"
    if not weight.is_file() or not config.is_file():
        raise FileNotFoundError(f"Expected original local Stable Audio weights and config: {model_dir}")
    contract = {
        "schema_version": SCHEMA_VERSION,
        "codec": "original_stable_audio_stereo_pair",
        "latent_dim": LATENT_DIM,
        "latent_frames": LATENT_FRAMES,
        "sample_rate": SAMPLE_RATE,
        "audio_samples": SAMPLES,
        "downsample_factor": DOWNSAMPLE_FACTOR,
        "channel_order": "file_first_two_then_last_two",
        "preprocess": "source_crop_10s_scipy_resample_poly_44100_zero_tail_pad_no_stride_pad",
        "min_audio_ratio": float(min_audio_ratio),
        "dtype": "float32",
        "parameters": ["mean", "std"],
        "model_config_sha256": sha256_file(config),
        "model_weights_sha256": sha256_file(weight),
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return {**contract, "fingerprint": hashlib.sha256(encoded).hexdigest(),
            "model_dir": str(model_dir)}


def load_cache_info(directory):
    path = Path(directory) / CACHE_INFO_NAME
    with path.open(encoding="utf-8") as handle:
        info = json.load(handle)
    if info.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported posterior cache schema: {path}")
    for key, expected in (("codec", "original_stable_audio_stereo_pair"),
                          ("channel_order", "file_first_two_then_last_two"),
                          ("preprocess", "source_crop_10s_scipy_resample_poly_44100_zero_tail_pad_no_stride_pad"),
                          ("downsample_factor", DOWNSAMPLE_FACTOR),
                          ("latent_dim", LATENT_DIM), ("latent_frames", LATENT_FRAMES),
                          ("sample_rate", SAMPLE_RATE), ("audio_samples", SAMPLES),
                          ("parameters", ["mean", "std"]), ("dtype", "float32")):
        if info.get(key) != expected:
            raise ValueError(f"Invalid posterior cache {key}: {path}")
    min_ratio = info.get("min_audio_ratio")
    if not isinstance(min_ratio, (int, float)) or not math.isfinite(min_ratio) or not 0 < min_ratio <= 1:
        raise ValueError(f"Invalid posterior minimum audio ratio: {path}")
    fingerprint = info.get("fingerprint", "")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError(f"Missing posterior cache fingerprint: {path}")
    contract = {key: value for key, value in info.items()
                if key not in ("fingerprint", "model_dir")}
    computed = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if computed != fingerprint:
        raise ValueError(f"Corrupt posterior cache provenance fingerprint: {path}")
    return info


def ensure_cache_info(directory, info):
    """Publish once atomically; refuse an incompatible directory on resume."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / CACHE_INFO_NAME
    fd, temporary = tempfile.mkstemp(prefix=".cache_info.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(info, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
        existing = load_cache_info(directory)
        if existing["fingerprint"] != info["fingerprint"]:
            raise ValueError(f"Refusing incompatible posterior cache directory: {directory}")
        return existing
    finally:
        Path(temporary).unlink(missing_ok=True)


def _scalar(value, name):
    value = np.asarray(value)
    if value.ndim != 0:
        raise ValueError(f"Posterior metadata {name} must be a scalar")
    return value.item()


def validate_posterior(value, expected_fingerprint=None):
    mean, std = np.asarray(value["mean"]), np.asarray(value["std"])
    for name, array in (("mean", mean), ("std", std)):
        if array.shape != (LATENT_DIM, LATENT_FRAMES) or array.dtype != np.float32:
            raise ValueError(f"Expected {name} float32 [{LATENT_DIM}, {LATENT_FRAMES}], got {array.dtype} {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"Posterior {name} contains NaN or Inf")
    if np.any(std < 0):
        raise ValueError("Posterior std must be non-negative, not log-variance")
    schema = _scalar(value["schema_version"], "schema_version")
    valid_samples = _scalar(value["valid_samples"], "valid_samples")
    valid_frames = _scalar(value["valid_frames"], "valid_frames")
    fingerprint = _scalar(value["fingerprint"], "fingerprint")
    if schema != SCHEMA_VERSION:
        raise ValueError(f"Unsupported posterior schema {schema}")
    if not isinstance(valid_samples, int) or not 0 < valid_samples <= SAMPLES:
        raise ValueError(f"Invalid posterior valid_samples: {valid_samples}")
    expected_frames = max(1, min(LATENT_FRAMES, valid_samples // DOWNSAMPLE_FACTOR))
    if not isinstance(valid_frames, int) or valid_frames != expected_frames:
        raise ValueError(f"Invalid posterior valid_frames {valid_frames}, expected {expected_frames}")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("Invalid posterior fingerprint")
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise ValueError("Posterior fingerprint differs from selected codec/preprocessing")
    return {"mean": mean, "std": std, "valid_samples": valid_samples,
            "valid_frames": valid_frames, "schema_version": schema, "fingerprint": fingerprint}


def load_posterior(path, expected_fingerprint=None):
    with np.load(path, allow_pickle=False) as archive:
        return validate_posterior(
            {key: archive[key] for key in
             ("mean", "std", "valid_samples", "valid_frames", "schema_version", "fingerprint")},
            expected_fingerprint=expected_fingerprint,
        )


def valid_posterior_file(path, expected_fingerprint):
    try:
        load_posterior(path, expected_fingerprint)
        return True
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return False


def atomic_save_posterior(path, mean, std, valid_samples, fingerprint):
    path = Path(path)
    valid_samples = int(valid_samples)
    value = {
        "mean": np.asarray(mean, dtype=np.float32),
        "std": np.asarray(std, dtype=np.float32),
        "valid_samples": np.int64(valid_samples),
        "valid_frames": np.int64(max(1, min(LATENT_FRAMES, valid_samples // DOWNSAMPLE_FACTOR))),
        "schema_version": np.int64(SCHEMA_VERSION),
        "fingerprint": np.asarray(fingerprint),
    }
    validate_posterior(value, fingerprint)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            # Uncompressed float32 preserves posterior values and avoids CPU bottlenecks.
            np.savez(handle, **value)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
