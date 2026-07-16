#!/usr/bin/env python3

from pathlib import Path

import numpy as np


DEFAULT_ROOM_DIM = np.array([11.0, 11.0, 5.0], dtype=np.float64)
DEFAULT_LISTENER_POSITION = DEFAULT_ROOM_DIM / 2.0
DEFAULT_NUM_POSITIONS = 5
MIN_CENTER_DISTANCE = 0.5
WALL_INSET_METERS = 0.001


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
