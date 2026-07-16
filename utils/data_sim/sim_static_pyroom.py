#!/usr/bin/env python3

import math
import os
from pathlib import Path

import numpy as np
import soundfile as sf


DEFAULT_ROOM_DIM = np.array([11.0, 11.0, 5.0], dtype=np.float64)
DEFAULT_LISTENER_POSITION = DEFAULT_ROOM_DIM / 2.0
DEFAULT_NUM_POSITIONS = 5
MIN_CENTER_DISTANCE = 0.5
WALL_INSET_METERS = 0.001
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_ABSORPTION = 0.80
DEFAULT_MAX_ORDER = 2
DEFAULT_SOUND_SPEED = 343.0
DEFAULT_IR_SECONDS = 0.10
DEFAULT_IR_TRIM_MS = 50.0
DEFAULT_PEAK_DBFS = -1.0
DEFAULT_FOA_ORDER = "WYZX"

_PRA = None


def sample_source_positions(
    row_index: int,
    seed: int,
    count: int = DEFAULT_NUM_POSITIONS,
) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(row_index)]))
    positions = []
    lower_bound = np.full(3, WALL_INSET_METERS, dtype=np.float64)
    upper_bound = DEFAULT_ROOM_DIM - WALL_INSET_METERS

    while len(positions) < count:
        candidate = rng.uniform(lower_bound, upper_bound)
        if np.linalg.norm(candidate - DEFAULT_LISTENER_POSITION) >= MIN_CENTER_DISTANCE:
            positions.append(candidate)

    return np.stack(positions)


def output_path_for_item(
    output_root: Path,
    source_fname: str,
    position_index: int,
) -> Path:
    return Path(output_root) / Path(source_fname).stem / f"{position_index}.flac"


def parse_labels(value: str | None) -> list[str]:
    return [label.strip() for label in str(value or "").split(",") if label.strip()]


def build_metadata_record(
    wav_path: Path,
    duration: float,
    position: np.ndarray,
    source_wav_path: Path,
    source_fname: str,
    position_index: int,
    labels: list[str] | tuple[str, ...],
) -> dict:
    return {
        "wav_path": str(wav_path),
        "duration": float(duration),
        "position": {
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2]),
        },
        "source_wav_path": str(source_wav_path),
        "source_fname": str(source_fname),
        "position_index": int(position_index),
        "labels": list(labels),
    }


def load_pyroomacoustics():
    global _PRA
    if _PRA is None:
        import pyroomacoustics as pra

        _PRA = pra
    return _PRA


def foa_gains_from_room_position(source_position: np.ndarray) -> np.ndarray:
    room_offset = np.asarray(source_position, dtype=np.float64) - DEFAULT_LISTENER_POSITION
    relative_source = np.array(
        [room_offset[0], room_offset[2], room_offset[1]],
        dtype=np.float64,
    )
    distance = float(np.linalg.norm(relative_source))
    if distance <= 0.0:
        raise ValueError("Source position cannot equal the listener position")

    unit = relative_source / distance
    return np.array(
        [
            1.0 / math.sqrt(2.0),
            -float(unit[0]),
            float(unit[1]),
            -float(unit[2]),
        ],
        dtype=np.float32,
    )


def find_ir_onset(ir: np.ndarray) -> int:
    peak = float(np.max(np.abs(ir))) if ir.size else 0.0
    if peak <= 0.0:
        return 0
    hits = np.flatnonzero(np.abs(ir) >= max(peak * 1.0e-3, 1.0e-8))
    return int(hits[0]) if hits.size else 0


def shape_ir(ir: np.ndarray) -> np.ndarray:
    shaped = np.asarray(ir, dtype=np.float32)
    if shaped.ndim != 1 or shaped.size == 0:
        raise ValueError("Expected a non-empty one-dimensional RIR")

    onset = find_ir_onset(shaped)
    trim_samples = max(
        1,
        int(round(DEFAULT_SAMPLE_RATE * DEFAULT_IR_TRIM_MS / 1000.0)),
    )
    ir_limit_samples = max(1, int(round(DEFAULT_SAMPLE_RATE * DEFAULT_IR_SECONDS)))
    end = min(shaped.shape[0], onset + min(trim_samples, ir_limit_samples))
    end = max(onset + 1, end)
    shaped = shaped[:end]
    if float(np.max(np.abs(shaped))) <= 0.0:
        raise ValueError("Pyroom produced a silent RIR")
    return shaped


def fft_convolve_1d(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    output_length = signal.shape[0] + kernel.shape[0] - 1
    fft_length = 1 << (output_length - 1).bit_length()
    output = np.fft.irfft(
        np.fft.rfft(signal, n=fft_length) * np.fft.rfft(kernel, n=fft_length),
        n=fft_length,
    )
    return output[:output_length].astype(np.float32)


def convolve_multichannel(mono_audio: np.ndarray, rirs: np.ndarray) -> np.ndarray:
    return np.stack(
        [fft_convolve_1d(mono_audio, rir) for rir in rirs],
        axis=1,
    )


def normalize_peak(audio: np.ndarray, peak_dbfs: float) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0.0:
        raise ValueError("Cannot normalize silent FOA audio")
    target_peak = 10.0 ** (float(peak_dbfs) / 20.0)
    return np.clip(audio * (target_peak / peak), -1.0, 1.0).astype(np.float32)


def render_foa(mono_audio: np.ndarray, source_position: np.ndarray) -> np.ndarray:
    mono = np.asarray(mono_audio, dtype=np.float32)
    if mono.ndim != 1 or mono.size == 0:
        raise ValueError("Expected non-empty mono audio")
    if float(np.max(np.abs(mono))) <= 0.0:
        raise ValueError("Cannot render silent mono audio")

    pra = load_pyroomacoustics()
    if hasattr(pra, "constants"):
        pra.constants.set("c", DEFAULT_SOUND_SPEED)
    room = pra.ShoeBox(
        DEFAULT_ROOM_DIM.tolist(),
        fs=DEFAULT_SAMPLE_RATE,
        materials=pra.Material(energy_absorption=DEFAULT_ABSORPTION),
        max_order=DEFAULT_MAX_ORDER,
    )
    room.add_source(np.asarray(source_position, dtype=float).tolist())
    microphones = DEFAULT_LISTENER_POSITION.reshape(3, 1)
    room.add_microphone_array(pra.MicrophoneArray(microphones, fs=DEFAULT_SAMPLE_RATE))
    room.compute_rir()

    center_rir = shape_ir(np.asarray(room.rir[0][0], dtype=np.float32))
    foa_rirs = foa_gains_from_room_position(source_position)[:, None] * center_rir[None, :]
    rendered = convolve_multichannel(mono, foa_rirs)
    return normalize_peak(rendered, DEFAULT_PEAK_DBFS)


def validate_existing_flac(path: Path) -> float:
    saved, sample_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )
    if (
        sample_rate != DEFAULT_SAMPLE_RATE
        or saved.shape[1] != 4
        or saved.shape[0] == 0
    ):
        raise ValueError(f"Invalid 4-channel 44.1 kHz FLAC: {path}")
    if float(np.max(np.abs(saved))) <= 0.0:
        raise ValueError(f"Silent FLAC output: {path}")
    return saved.shape[0] / sample_rate


def write_validated_flac(path: Path, audio: np.ndarray) -> float:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.tmp.flac"
    )
    try:
        sf.write(
            str(temporary_path),
            audio,
            DEFAULT_SAMPLE_RATE,
            format="FLAC",
            subtype="PCM_16",
        )
        duration = validate_existing_flac(temporary_path)
        os.replace(temporary_path, output_path)
        return duration
    finally:
        temporary_path.unlink(missing_ok=True)
