from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torchaudio

from data_gen.stable_audio_3.stable_audio_3 import StableAudioModel


DEFAULT_PROMPT = (
    "House music that captures the feeling of a sunny outdoor festival with "
    "friends, 124 BPM"
)
MODEL_CHOICES = (
    "medium",
    "medium-base",
    "small-music",
    "small-music-base",
    "small-sfx",
    "small-sfx-base",
)


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate audio with Stable Audio 3")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="medium")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--duration", type=positive_float, default=10.0)
    parser.add_argument("--steps", type=positive_int, default=8)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--output", default="output.wav")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    if args.model.startswith("medium") and not torch.cuda.is_available():
        raise RuntimeError("The medium model requires a CUDA GPU for inference")

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    model = StableAudioModel.from_pretrained(args.model)
    print(f"Generating {args.duration:g} seconds with {args.steps} steps")
    audio = model.generate(
        prompt=args.prompt,
        duration=args.duration,
        steps=args.steps,
        seed=args.seed,
        batch_size=1,
        sample_size=model.model_config["sample_size"],
    )

    waveform = audio[0].detach().cpu()
    torchaudio.save(str(output), waveform, model.model_config["sample_rate"])
    print(f"Saved {tuple(waveform.shape)} audio to: {output}")
    return output


if __name__ == "__main__":
    main()
