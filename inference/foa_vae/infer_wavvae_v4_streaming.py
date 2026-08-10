import argparse
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple

from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.foa_vae import infer_wavvae_v4 as base


DEFAULT_CONFIG = base.DEFAULT_CONFIG
STREAM_CHUNK_FRAMES = 10
# Ten latent frames cover the measured V4 decoder receptive field on either side.
STREAM_CONTEXT_FRAMES = 10


@dataclass(frozen=True)
class StreamingWindow:
    index: int
    core_start: int
    core_end: int
    window_start: int
    window_end: int
    crop_start: int
    crop_end: int


def iter_streaming_windows(
    total_frames: int,
    hop_length: int,
    chunk_frames: int = STREAM_CHUNK_FRAMES,
    context_frames: int = STREAM_CONTEXT_FRAMES,
) -> Iterator[StreamingWindow]:
    """Plan decoder windows without inventing latent padding at the boundaries."""
    total_frames = int(total_frames)
    hop_length = int(hop_length)
    chunk_frames = int(chunk_frames)
    context_frames = int(context_frames)
    if total_frames < 0:
        raise ValueError(f"total_frames must be non-negative, got {total_frames}")
    if hop_length <= 0:
        raise ValueError(f"hop_length must be positive, got {hop_length}")
    if chunk_frames <= 0:
        raise ValueError(f"chunk_frames must be positive, got {chunk_frames}")
    if context_frames < 0:
        raise ValueError(f"context_frames must be non-negative, got {context_frames}")

    for index, core_start in enumerate(range(0, total_frames, chunk_frames)):
        core_end = min(total_frames, core_start + chunk_frames)
        window_start = max(0, core_start - context_frames)
        window_end = min(total_frames, core_end + context_frames)
        crop_start = (core_start - window_start) * hop_length
        crop_end = crop_start + (core_end - core_start) * hop_length
        yield StreamingWindow(
            index=index,
            core_start=core_start,
            core_end=core_end,
            window_start=window_start,
            window_end=window_end,
            crop_start=crop_start,
            crop_end=crop_end,
        )


def decode_streaming_latents(
    autoencoder,
    latents,
    hop_length: int,
    chunk_frames: int = STREAM_CHUNK_FRAMES,
    context_frames: int = STREAM_CONTEXT_FRAMES,
) -> Iterator[Tuple[StreamingWindow, object]]:
    if latents.dim() != 3:
        raise ValueError(f"Expected [batch, channels, frames] latents, got {tuple(latents.shape)}")

    windows = iter_streaming_windows(
        total_frames=latents.shape[-1],
        hop_length=hop_length,
        chunk_frames=chunk_frames,
        context_frames=context_frames,
    )
    for window in windows:
        latent_window = latents[..., window.window_start : window.window_end]
        decoded_window = autoencoder.decode(latent_window).sample
        if decoded_window.shape[-1] < window.crop_end:
            raise RuntimeError(
                "Decoder output is shorter than its hop-aligned crop: "
                f"window={window.index} decoded={decoded_window.shape[-1]} crop_end={window.crop_end}"
            )
        segment = decoded_window[..., window.crop_start : window.crop_end]
        yield window, segment


def create_chunks_directory(output_path: str) -> Path:
    output = Path(output_path)
    stem_path = output.with_suffix("")
    root = stem_path.with_name(f"{stem_path.name}_chunks")
    for index in range(1000000):
        candidate = root if index == 0 else root.with_name(f"{root.name}_{index:04d}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Unable to create a unique chunks directory for {output_path}")


def _resolve_pad_multiple(model, pad_multiple) -> int:
    if pad_multiple in ("auto", True):
        return base.model_downsample_factor(model)
    if pad_multiple in ("", None, False):
        return 1
    return int(pad_multiple)


def reconstruct_streaming(
    model,
    wav,
    device: str,
    precision: str,
    pad_multiple,
    segment_callback: Optional[Callable[[int, object], None]] = None,
    progress_desc: Optional[str] = None,
):
    """Encode once and emit each fixed-lookahead decoder core immediately."""
    import torch

    original_len = int(wav.shape[-1])
    if original_len <= 0:
        raise ValueError("Cannot reconstruct an empty waveform.")

    multiple = _resolve_pad_multiple(model, pad_multiple)
    wav, _ = base.pad_to_multiple(wav, multiple)
    audio = wav.unsqueeze(0).to(device)
    autoencoder = getattr(model, "autoencoder", None)
    if autoencoder is None:
        raise ValueError("Streaming V4 inference requires model.autoencoder.")

    hop_length = base.model_downsample_factor(model)
    use_autocast = str(device).startswith("cuda") and precision not in ("fp32", "float32", "none", "")
    dtype = base.autocast_dtype(precision)
    segments = []
    progress = None
    try:
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_autocast):
                latent_dist = autoencoder.encode(audio).latent_dist
                # Shared posterior noise is essential: resampling overlapping windows creates seams.
                latents = latent_dist.sample()
                del latent_dist
                if progress_desc is not None:
                    progress = tqdm(
                        total=int(latents.shape[-1]),
                        desc=progress_desc,
                        unit="latent frame",
                        dynamic_ncols=True,
                    )

                for window, decoded_segment in decode_streaming_latents(
                    autoencoder=autoencoder,
                    latents=latents,
                    hop_length=hop_length,
                ):
                    global_start = window.core_start * hop_length
                    valid_length = min(decoded_segment.shape[-1], original_len - global_start)
                    if valid_length <= 0:
                        break
                    if decoded_segment.shape[0] != 1:
                        raise RuntimeError(
                            f"Streaming inference expects batch size 1, got {decoded_segment.shape[0]}"
                        )
                    segment = decoded_segment[..., :valid_length].detach().float().cpu().squeeze(0)
                    segments.append(segment)
                    if segment_callback is not None:
                        segment_callback(window.index, segment)
                    if progress is not None:
                        progress.update(window.core_end - window.core_start)
    finally:
        if progress is not None:
            progress.close()

    if not segments:
        raise RuntimeError("The V4 encoder produced no decodable latent frames.")
    reconstruction = torch.cat(segments, dim=-1)
    if reconstruction.shape[-1] < original_len:
        raise RuntimeError(
            "Streaming decoder output is shorter than the input: "
            f"decoded={reconstruction.shape[-1]} input={original_len}"
        )
    return reconstruction[..., :original_len]


def run_inference(config_path: str):
    import torch

    os.chdir(PROJECT_ROOT)
    config = base.load_inference_config(config_path)
    inputs = base.collect_input_paths(config)
    if not inputs:
        raise ValueError("No inference inputs found. Set input_paths, input_list, or metadata_path.")
    out_path = config.get("out_path", "")
    if not out_path:
        raise ValueError("Inference config must provide out_path.")
    out_path = base._resolve_repo_path(out_path)
    config["out_path"] = out_path

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    precision = config.get("precision", "bf16")
    model_hparams = base.load_model_hparams(config)
    model, resolved_ckpt = base.build_model(config, model_hparams, device)

    sample_rate = int(config.get("sample_rate", model_hparams.get("sample_rate", 44100)))
    expected_channels = int(
        config.get("expected_channels", model_hparams.get("foa_vae", {}).get("input_channels", 4))
    )
    normalize_mode = config.get("normalize_mode", config.get("foa_normalize_mode", "none"))
    suffix = config.get("output_suffix", "_recon")
    reference_suffix = config.get("input_copy_suffix", "_input")
    copy_input_audio = bool(config.get("copy_input_audio", True))
    overwrite = bool(config.get("overwrite", True))
    pad_multiple = config.get("pad_to_multiple", "auto")
    hop_length = base.model_downsample_factor(model)

    used_outputs = set()
    chunk_ms = 1000.0 * STREAM_CHUNK_FRAMES * hop_length / sample_rate
    print(f"| FOA VAE fixed-lookahead streaming config: {config_path}")
    print(f"| Loaded checkpoint: {resolved_ckpt}")
    print(f"| Inputs: {len(inputs)}")
    print(
        "| Streaming decoder: "
        f"chunk={STREAM_CHUNK_FRAMES} latent frames, "
        f"left_context={STREAM_CONTEXT_FRAMES}, lookahead={STREAM_CONTEXT_FRAMES}, "
        f"hop={hop_length}, nominal_chunk_ms={chunk_ms:.2f}"
    )

    for idx, input_path in enumerate(inputs):
        output_path = base.resolve_output_wav_path(input_path, out_path, idx, len(inputs), suffix=suffix)
        output_path = base.make_unique_path(output_path, used_outputs)
        reference_path = None
        if copy_input_audio:
            reference_path = base.resolve_reference_audio_path(output_path, input_path, suffix, reference_suffix)
            reference_path = base.make_unique_path(reference_path, used_outputs)
        if os.path.exists(output_path) and not overwrite:
            print(f"| [{idx + 1}/{len(inputs)}] skip existing: {output_path}")
            continue

        wav, sr = base.load_audio(input_path, sample_rate, expected_channels, normalize_mode)
        chunks_dir = create_chunks_directory(output_path)
        started_at = time.perf_counter()
        first_chunk_seconds = None

        def save_segment(segment_index, segment):
            nonlocal first_chunk_seconds
            segment_path = chunks_dir / f"segment_{segment_index:06d}.wav"
            base.save_audio(str(segment_path), segment, sr, config)
            elapsed = time.perf_counter() - started_at
            if first_chunk_seconds is None:
                first_chunk_seconds = elapsed
            tqdm.write(
                f"| [{idx + 1}/{len(inputs)}] saved segment {segment_index:06d}: "
                f"{segment_path} shape={tuple(segment.shape)} elapsed={elapsed:.3f}s"
            )

        reconstruction = reconstruct_streaming(
            model=model,
            wav=wav,
            device=device,
            precision=precision,
            pad_multiple=pad_multiple,
            segment_callback=save_segment,
            progress_desc=f"[{idx + 1}/{len(inputs)}] {Path(input_path).name} decode",
        )
        base.save_audio(output_path, reconstruction, sr, config)
        if reference_path is not None and (overwrite or not os.path.exists(reference_path)):
            os.makedirs(os.path.dirname(reference_path), exist_ok=True)
            shutil.copy2(input_path, reference_path)

        total_seconds = time.perf_counter() - started_at
        print(
            f"| [{idx + 1}/{len(inputs)}] merged {input_path} -> {output_path} "
            f"shape={tuple(reconstruction.shape)} sr={sr} total={total_seconds:.3f}s"
        )
        print(f"| [{idx + 1}/{len(inputs)}] segment directory: {chunks_dir}")
        if first_chunk_seconds is not None:
            print(f"| [{idx + 1}/{len(inputs)}] first decoded segment saved in {first_chunk_seconds:.3f}s")
        if reference_path is not None:
            print(f"| [{idx + 1}/{len(inputs)}] copied input audio to: {reference_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct FOA wavs with V4 fixed-lookahead streaming decode."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to FOA VAE inference yaml.")
    args = parser.parse_args()
    run_inference(args.config)


if __name__ == "__main__":
    main()
