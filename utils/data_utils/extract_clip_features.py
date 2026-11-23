"""Sharded CLIP extraction from already projected videos.

Run with torchrun; each rank independently processes ids[rank::world_size].
No distributed process group or GPU-to-GPU communication is needed.
Preprocessing matches user/visage/data/infer_clip.py: ffmpeg fps/scale/crop,
RGB / 255, CLIP image normalization, ViT-B/32 encode_image, no feature L2 norm.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "results/261015_ditar_v7_16k_eval/models/ViT-B-32.pt"


def load_ids(metadata):
    """Read CSV id or JSONL file_name/id; deduplicate while preserving order."""
    metadata = Path(metadata)
    if metadata.suffix.lower() == ".csv":
        with metadata.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    else:
        with metadata.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    ids, seen = [], set()
    for index, row in enumerate(rows, 1):
        value = row.get("file_name", row.get("id"))
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing string file_name/id at metadata row {index}")
        value = value.strip()
        if value.endswith((".npy", ".mp4")):
            value = value.rsplit(".", 1)[0]
        if value in {".", ".."} or "/" in value or "\\" in value or not value:
            raise ValueError(f"Expected a plain clip ID, got {value!r}")
        if value not in seen:
            ids.append(value)
            seen.add(value)
    if not ids:
        raise ValueError("Metadata contains no video IDs")
    return ids


def shard_ids(ids, rank, world_size):
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Require world_size >= 1 and 0 <= rank < world_size")
    return ids[rank::world_size]


def valid_feature(path, frames=40):
    """A partial, corrupt or wrong-shaped file must not count as completed."""
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if not isinstance(value, np.ndarray):
            if hasattr(value, "close"):
                value.close()
            return False
        return (
            value.shape == (frames, 512)
            and np.issubdtype(value.dtype, np.floating)
            and bool(np.isfinite(value).all())
        )
    except (OSError, ValueError, EOFError):
        return False


def resolve_ffmpeg(explicit=None):
    if explicit:
        found = shutil.which(explicit)
        if found:
            return found
        raise FileNotFoundError(f"ffmpeg executable not found: {explicit}")
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def decode_video(path, ffmpeg_exe, fps=4.0, frames=40, threads=1, timeout=120.0):
    """Decode uint8 [T, 3, 224, 224], without requiring ffprobe.

    Scale expressions preserve the original script's integer-floor dimensions.
    The requested leading frames match the 10-second, 4-fps training contract.
    """
    size = 224
    filters = (
        f"fps={fps},"
        f"scale=w='if(lt(iw,ih),{size},trunc({size}*iw/ih))':"
        f"h='if(gte(iw,ih),{size},trunc({size}*ih/iw))',"
        f"crop={size}:{size}:floor((iw-{size})/2):floor((ih-{size})/2)"
    )
    command = [
        str(ffmpeg_exe), "-hide_banner", "-loglevel", "error", "-nostdin",
        "-threads", str(threads), "-filter_threads", str(threads),
        "-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", filters, "-frames:v", str(frames), "-threads:v", str(threads),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, timeout=timeout)
    if result.returncode:
        detail = result.stderr.decode(errors="replace")[-2000:]
        raise RuntimeError(f"ffmpeg failed ({result.returncode}): {detail}")
    expected_bytes = frames * size * size * 3
    if len(result.stdout) != expected_bytes:
        actual = len(result.stdout) / (size * size * 3)
        raise ValueError(f"Decoded {actual:g} frames; expected {frames}")
    array = np.frombuffer(result.stdout, dtype=np.uint8).reshape(frames, size, size, 3)
    # Copy into writable, contiguous CHW memory before handing it to a worker queue.
    return torch.from_numpy(array.transpose(0, 3, 1, 2).copy())


class VideoDataset(Dataset):
    def __init__(self, ids, data_path, ffmpeg_exe, fps, frames, threads, timeout):
        self.ids = ids
        self.data_path = Path(data_path)
        self.ffmpeg_exe = ffmpeg_exe
        self.fps, self.frames = fps, frames
        self.threads, self.timeout = threads, timeout

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        clip_id = self.ids[index]
        path = self.data_path / f"{clip_id}.mp4"
        try:
            value = decode_video(
                path, self.ffmpeg_exe, self.fps, self.frames,
                self.threads, self.timeout,
            )
            return clip_id, value, None
        except Exception as exc:
            return clip_id, None, {
                "id": clip_id, "path": str(path),
                "error": f"{type(exc).__name__}: {exc}",
            }


def collate_videos(items):
    ids, images, errors = [], [], []
    for clip_id, value, error in items:
        if error is not None:
            errors.append(error)
        else:
            ids.append(clip_id)
            images.append(value)
    return {
        "ids": ids,
        "images": torch.stack(images) if images else None,
        "errors": errors,
    }


def init_worker(_worker_id):
    torch.set_num_threads(1)


def atomic_save(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, value, allow_pickle=False)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="JSONL file_name or CSV id")
    parser.add_argument("--data-path", required=True, help="Projected .mp4 directory")
    parser.add_argument("--save-path", required=True, help="Destination .npy directory")
    parser.add_argument("--clip-model", default=str(DEFAULT_MODEL), help="Local ViT-B/32 .pt")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8, help="Videos per CPU decode batch")
    parser.add_argument("--encode-batch-size", type=int, default=256, help="Frames per GPU forward")
    parser.add_argument("--workers", type=int, default=4, help="CPU decode workers per rank")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--ffmpeg-threads", type=int, default=1)
    parser.add_argument("--ffmpeg", default=None, help="Optional executable; imageio fallback supported")
    parser.add_argument("--decode-timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=None, help="First N metadata IDs, before sharding")
    parser.add_argument("--report-dir", default=None, help="Default: save-path/_extract_reports")
    parser.add_argument("--log-every", type=int, default=20, help="Log every N loader batches")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    for name in ("frames", "batch_size", "encode_batch_size", "prefetch_factor", "ffmpeg_threads", "log_every"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.workers < 0 or (args.limit is not None and args.limit < 1):
        raise ValueError("workers must be non-negative and limit must be positive")
    if not math.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not math.isfinite(args.decode_timeout) or args.decode_timeout <= 0:
        raise ValueError("decode-timeout must be finite and positive")
    if not Path(args.data_path).is_dir():
        raise NotADirectoryError(args.data_path)
    if not Path(args.clip_model).is_file():
        raise FileNotFoundError(f"Local CLIP checkpoint missing: {args.clip_model}")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ids = load_ids(args.metadata)
    if args.limit is not None:
        ids = ids[:args.limit]
    selected = shard_ids(ids, rank, world_size)
    output_dir = Path(args.save_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir) if args.report_dir else output_dir / "_extract_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / f"summary_rank{rank:03d}.json"
    errors_path = report_dir / f"errors_rank{rank:03d}.jsonl"
    started = time.monotonic()
    report = {
        "rank": rank, "world_size": world_size, "local_rank": local_rank,
        "metadata_count": len(ids), "shard_count": len(selected),
        "skipped_valid": 0, "invalid_existing": 0, "written": 0, "failed": 0,
        "arguments": vars(args), "errors_path": str(errors_path),
    }
    error_log = errors_path.open("w", encoding="utf-8")

    def record_error(error):
        report["failed"] += 1
        error_log.write(json.dumps(error, ensure_ascii=False) + "\n")
        error_log.flush()

    try:
        pending = []
        # Directory listings avoid one network-filesystem stat per metadata ID.
        with os.scandir(output_dir) as entries:
            existing_names = {entry.name for entry in entries if entry.name.endswith(".npy")}
        with os.scandir(args.data_path) as entries:
            source_names = {entry.name for entry in entries if entry.name.endswith(".mp4")}
        for clip_id in selected:
            destination = output_dir / f"{clip_id}.npy"
            if destination.name in existing_names:
                if valid_feature(destination, args.frames):
                    report["skipped_valid"] += 1
                    continue
                report["invalid_existing"] += 1
            source = Path(args.data_path) / f"{clip_id}.mp4"
            if source.name not in source_names:
                record_error({"id": clip_id, "path": str(source), "error": "Missing input video"})
                continue
            pending.append(clip_id)
        print(
            f"[rank {rank}/{world_size}] selected={len(selected)} "
            f"valid_skip={report['skipped_valid']} pending={len(pending)} "
            f"missing={report['failed']}", flush=True,
        )
        if pending:
            if not torch.cuda.is_available():
                raise RuntimeError("CLIP extraction requires CUDA")
            torch.cuda.set_device(local_rank)
            torch.set_num_threads(1)
            device = torch.device("cuda", local_rank)
            ffmpeg_exe = resolve_ffmpeg(args.ffmpeg)
            dataset = VideoDataset(
                pending, args.data_path, ffmpeg_exe, args.fps, args.frames,
                args.ffmpeg_threads, args.decode_timeout,
            )
            loader_kwargs = {}
            if args.workers:
                loader_kwargs.update(
                    multiprocessing_context="spawn",
                    prefetch_factor=args.prefetch_factor,
                    persistent_workers=True,
                )
            loader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True,
                collate_fn=collate_videos, worker_init_fn=init_worker,
                **loader_kwargs,
            )
            import clip
            model, preprocess = clip.load(str(Path(args.clip_model).resolve()), device=device, jit=False)
            model.requires_grad_(False).eval()
            normalize = preprocess.transforms[-1]
            print(
                f"[rank {rank}] device={device} model={args.clip_model} "
                f"ffmpeg={ffmpeg_exe} workers={args.workers}", flush=True,
            )
            with torch.inference_mode():
                for batch_index, batch in enumerate(loader):
                    for error in batch["errors"]:
                        record_error(error)
                    if batch["images"] is not None:
                        raw = batch["images"].flatten(0, 1)
                        encoded = []
                        for begin in range(0, raw.shape[0], args.encode_batch_size):
                            images = raw[begin:begin + args.encode_batch_size].to(
                                device, non_blocking=True,
                            ).float().div_(255.0)
                            # Match the original extractor: image normalization only.
                            features = model.encode_image(normalize(images))
                            encoded.append(features.cpu().numpy())
                        values = np.concatenate(encoded, axis=0).reshape(
                            len(batch["ids"]), args.frames, -1,
                        )
                        for clip_id, value in zip(batch["ids"], values):
                            if value.shape != (args.frames, 512) or not np.isfinite(value).all():
                                record_error({"id": clip_id, "error": f"Invalid CLIP features: {value.shape}"})
                                continue
                            atomic_save(output_dir / f"{clip_id}.npy", value)
                            report["written"] += 1
                    if (batch_index + 1) % args.log_every == 0:
                        elapsed = time.monotonic() - started
                        print(
                            f"[rank {rank}] written={report['written']}/{len(pending)} "
                            f"failed={report['failed']} elapsed={elapsed:.1f}s "
                            f"videos/s={report['written'] / max(elapsed, 1e-6):.2f}",
                            flush=True,
                        )
    except BaseException as exc:
        report["fatal_error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        error_log.close()
        write_json(summary_path, report)
        print(f"[rank {rank}] SUMMARY {json.dumps(report, ensure_ascii=False)}", flush=True)
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
