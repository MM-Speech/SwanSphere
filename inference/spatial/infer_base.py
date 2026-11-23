"""Shared non-streaming four-way history ablation for spatial DiTAR.

The first mode letter selects the LM feedback history and the second selects
the DiT latent-context history. ``T`` uses the ground-truth VAE latent and
``G`` uses the generated patch, yielding TT, TG, GT, and GG. Every mode uses
the same per-step initial diffusion noise, records its own current prediction,
and decodes the complete predicted sequence only after generation finishes.
"""

import argparse
import collections
import collections.abc
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# attrdict still imports aliases removed from collections in Python 3.10.
for type_name in collections.abc.__all__:
    if not hasattr(collections, type_name):
        setattr(collections, type_name, getattr(collections.abc, type_name))


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from inference.spatial import infer_utils as vae_infer
from modules.spatial.build_model_utils import DiTARBuildModelMixinS
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams, set_hparams


YT_AMBIGEN_ROOT = "/mnt/bn/sa-ag-data/leike/spatial/data/YT-Ambigen"
LOCAL_CLIP_DIR = os.path.join(YT_AMBIGEN_ROOT, "front_clip_10")
GLOBAL_CLIP_DIR = os.path.join(YT_AMBIGEN_ROOT, "pad_360_clip_10")
LATENT_DIR = os.path.join(YT_AMBIGEN_ROOT, "stable_audio_pair_posterior_10")
ORIGINAL_AUDIO_DIR = os.path.join(YT_AMBIGEN_ROOT, "audio_10")
HISTORY_MODES = {
    "TT": ("teacher", "teacher"),
    "TG": ("teacher", "generated"),
    "GT": ("generated", "teacher"),
    "GG": ("generated", "generated"),
}


def _resolve_repo_path(path):
    path = Path(os.path.expanduser(str(path)))
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


def _resolve_ditar_config(config_path, checkpoint_path, default_config):
    if config_path:
        resolved = _resolve_repo_path(config_path)
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"DiTAR config not found: {resolved}")
        return resolved

    checkpoint_path = Path(_resolve_repo_path(checkpoint_path))
    checkpoint_dir = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
    saved_config = checkpoint_dir / "config.yaml"
    if saved_config.is_file():
        return str(saved_config)

    fallback = _resolve_repo_path(default_config)
    if not os.path.isfile(fallback):
        raise FileNotFoundError(
            f"Neither checkpoint config nor default config exists: "
            f"{saved_config}, {fallback}"
        )
    return fallback


def _precision_dtype(precision):
    precision = str(precision).lower()
    if precision in ("bf16", "bfloat16"):
        return torch.bfloat16
    if precision in ("fp16", "float16", "half"):
        return torch.float16
    if precision in ("fp32", "float32", "none"):
        return torch.float32
    raise ValueError(f"Unsupported precision: {precision}")


def _autocast(device, dtype):
    device_type = torch.device(device).type
    enabled = device_type == "cuda" and dtype != torch.float32
    return torch.autocast(device_type=device_type, dtype=dtype, enabled=enabled)


def build_ditar(config_path, checkpoint_path, device, precision, *,
                model_version, default_config, expected_latent_dim):
    config_path = _resolve_ditar_config(config_path, checkpoint_path, default_config)
    hp = set_hparams(config=config_path, print_hparams=False)
    if int(hp.get("model_ver", -1)) != model_version:
        raise ValueError(
            f"Expected model_ver={model_version}, got {hp.get('model_ver')} from {config_path}"
        )
    if int(hp.get("latent_dim", hp.get("d_lat", -1))) != expected_latent_dim:
        raise ValueError(
            f"Spatial DiT-AR v{model_version} inference requires a {expected_latent_dim}-D latent"
        )

    # Build through the same mixin as training so every architectural field is
    # identical to the checkpoint (including otherwise easy-to-miss defaults).
    hparams["gradient_checkpointing"] = False
    builder = DiTARBuildModelMixinS()
    builder._build_model()
    model = builder.model

    dtype = _precision_dtype(precision)
    if dtype != torch.float32:
        model.to(dtype=dtype)

    checkpoint_path = _resolve_repo_path(checkpoint_path)
    resolved_checkpoint = vae_infer.resolve_checkpoint_path(checkpoint_path)
    print(f"| DiT-AR v{model_version} config: {config_path}")
    print(f"| DiT-AR v{model_version} checkpoint strict: True")
    load_ckpt(
        model,
        resolved_checkpoint,
        "model",
        strict=True,
        force=True,
        map_location="cpu",
        mmap=True,
    )
    model.eval().to(torch.device(device))
    return model, hp, resolved_checkpoint




def load_clip(path, frames, dim, name):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {name}: {path}")
    value = np.load(path, allow_pickle=False)
    if value.ndim != 2 or value.shape[0] < frames or value.shape[1] != dim:
        raise ValueError(
            f"{name} must be [T>={frames}, {dim}], got {value.shape}: {path}"
        )
    value = np.asarray(value[:frames], dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf: {path}")
    return torch.from_numpy(value).unsqueeze(0)


def load_latent_stats(mean_path, std_path, latent_dim):
    mean = np.asarray(np.load(_resolve_repo_path(mean_path)), dtype=np.float32).reshape(-1)
    std = np.asarray(np.load(_resolve_repo_path(std_path)), dtype=np.float32).reshape(-1)
    if mean.shape != (latent_dim,) or std.shape != (latent_dim,):
        raise ValueError(f"Expected {latent_dim}-D stats, got mean={mean.shape}, std={std.shape}")
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Latent statistics contain NaN or Inf")
    if np.any(std <= 0):
        raise ValueError("Latent standard deviation must be positive")
    return torch.from_numpy(mean), torch.from_numpy(std)




class SpatialDiTARInferBase:
    # Subclasses select a different timebase/checkpoint family without
    # duplicating or mutating the shared four-way ablation implementation.
    build_ditar = staticmethod(build_ditar)
    @staticmethod
    def build_vae(config_path, device):
        raise NotImplementedError("Subclasses must provide an audio codec.")

    vae_backend = vae_infer
    metrics_space = "codec latent"
    expected_latent_dim = None

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.dtype = _precision_dtype(args.precision)
        self.model, self.hp, self.ditar_checkpoint = self.build_ditar(
            args.ditar_config,
            args.ditar_ckpt,
            self.device,
            args.precision,
        )
        self.vae, self.vae_config, self.vae_checkpoint = self.build_vae(
            args.vae_config,
            self.device,
        )
        self.mean, self.std = self._load_latent_stats(args)

        self.clip_frames = int(self.hp["clip_frames"])
        self.clip_dim = int(self.hp["d_clip"])
        self.latent_frames = int(self.hp["latent_frames"])
        self.patch_size = int(self.hp.get("ditar_patch_size", 4))
        if self.latent_frames <= 0 or self.patch_size <= 0:
            raise ValueError(
                f"Invalid latent_frames={self.latent_frames}, patch_size={self.patch_size}"
            )
        self.max_new_patches = math.ceil(self.latent_frames / self.patch_size)
        latent_dir = args.latent_dir or self.hp.get("latent_base_path", LATENT_DIR)
        self.latent_dir = os.path.expanduser(str(latent_dir))
        if not os.path.isabs(self.latent_dir):
            self.latent_dir = _resolve_repo_path(self.latent_dir)

    def _load_latent_stats(self, args):
        return load_latent_stats(args.mean_path, args.std_path, self.expected_latent_dim)

    def teacher_latent_source(self, sample_id):
        raise NotImplementedError('Subclasses must identify their teacher latent source.')

    def _load_teacher_latent(self, sample_id):
        raise NotImplementedError('Subclasses must load their codec-specific teacher latents.')

    def _prepare_inputs(self, sample_id):
        if sample_id.endswith(".npy"):
            raise ValueError("--id values must not include the .npy suffix")
        file_name = f"{sample_id}.npy"
        local_clip = load_clip(
            os.path.join(self.args.local_clip_dir, file_name),
            self.clip_frames,
            self.clip_dim,
            "local CLIP",
        ).to(self.device)
        global_clip = load_clip(
            os.path.join(self.args.global_clip_dir, file_name),
            self.clip_frames,
            self.clip_dim,
            "global CLIP",
        ).to(self.device)
        teacher_latent = self._load_teacher_latent(sample_id)
        inputs = {
            "video_feats": local_clip,
            "v_mask": torch.ones(
                local_clip.shape[:2], dtype=torch.long, device=self.device
            ),
            "global_clip_embedding": global_clip,
        }
        return inputs, teacher_latent

    def _make_initial_noise(self, seed):
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        return torch.randn(
            self.max_new_patches,
            1,
            self.patch_size,
            int(self.model.config.out_channels),
            device=self.device,
            dtype=torch.float32,
        )

    def _validate_generated_latent(self, generated_latent):
        if generated_latent.ndim != 3 or generated_latent.shape[-1] != self.expected_latent_dim:
            raise ValueError(
                f"Expected generated latent [B, T, {self.expected_latent_dim}], got "
                f"{tuple(generated_latent.shape)}"
            )
        if generated_latent.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got {generated_latent.shape[0]}")
        if generated_latent.shape[1] < self.latent_frames:
            raise ValueError(
                f"Generated only {generated_latent.shape[1]} latent frames; "
                f"expected at least {self.latent_frames}"
            )

        # Generation is patch aligned; decoding/metrics use only valid frames.
        generated_latent = generated_latent[:, : self.latent_frames].float()
        if not torch.isfinite(generated_latent).all():
            raise ValueError("Generated latent contains NaN or Inf")
        return generated_latent

    def _decode_latent(self, generated_latent):
        raise NotImplementedError('Subclasses must decode their codec-specific latents.')

    @torch.inference_mode()
    def generate_modes(self, sample_id, seed, history_modes):
        inputs, teacher_latent = self._prepare_inputs(sample_id)
        teacher_valid = teacher_latent[:, : self.latent_frames].float()
        initial_noise = self._make_initial_noise(seed)
        results = {}

        for mode in history_modes:
            lm_history_source, dit_history_source = HISTORY_MODES[mode]
            print(
                f"| Generating {mode}: LM={lm_history_source}, "
                f"DiT={dit_history_source}"
            )
            with _autocast(self.device, self.dtype):
                generated_latent, _ = self.model.inference(
                    inputs,
                    timesteps=self.args.timesteps,
                    seq_cfg_w=tuple(self.args.seq_cfg_w),
                    timestep_annealing_w=tuple(self.args.timestep_annealing_w),
                    use_tqdm=not self.args.no_tqdm,
                    max_new_patches=self.max_new_patches,
                    teacher_latent=teacher_latent,
                    lm_history_source=lm_history_source,
                    dit_history_source=dit_history_source,
                    initial_noise=initial_noise,
                )
            generated_latent = self._validate_generated_latent(generated_latent)
            audio = self._decode_latent(generated_latent)
            results[mode] = {
                "latent": generated_latent.detach().cpu(),
                "audio": audio,
            }

        return teacher_valid.detach().cpu(), results, tuple(initial_noise.shape)


def _cosine_similarity(value, target):
    value = value.float().reshape(-1)
    target = target.float().reshape(-1)
    return F.cosine_similarity(value, target, dim=0, eps=1.0e-8).item()


def _pair_metrics(value, target):
    return {
        "mse": F.mse_loss(value.float(), target.float()).item(),
        "cosine": _cosine_similarity(value, target),
    }


def _prediction_stats(value):
    value = value.float()
    return {
        "mean": value.mean().item(),
        "std": value.std(unbiased=False).item(),
        "rms": value.square().mean().sqrt().item(),
    }


def _patch_stage(patch_index, total_patches):
    early_end = math.ceil(total_patches / 3)
    mid_end = math.ceil(2 * total_patches / 3)
    if patch_index < early_end:
        return "early"
    if patch_index < mid_end:
        return "mid"
    return "late"


def _segment_frame_ranges(latent_frames, patch_size):
    total_patches = math.ceil(latent_frames / patch_size)
    early_end = min(latent_frames, math.ceil(total_patches / 3) * patch_size)
    mid_end = min(latent_frames, math.ceil(2 * total_patches / 3) * patch_size)
    return {
        "overall": (0, latent_frames),
        "early": (0, early_end),
        "mid": (early_end, mid_end),
        "late": (mid_end, latent_frames),
    }


def _build_patch_rows(sample_id, results, teacher, latent_frames, patch_size):
    total_patches = math.ceil(latent_frames / patch_size)
    tt_latent = results.get("TT", {}).get("latent")
    rows = []
    for mode, result in results.items():
        lm_history_source, dit_history_source = HISTORY_MODES[mode]
        for patch_index in range(total_patches):
            frame_start = patch_index * patch_size
            frame_end = min(frame_start + patch_size, latent_frames)
            predicted_patch = result["latent"][:, frame_start:frame_end]
            teacher_patch = teacher[:, frame_start:frame_end]
            teacher_metrics = _pair_metrics(predicted_patch, teacher_patch)
            stats = _prediction_stats(predicted_patch)
            if tt_latent is None:
                tt_metrics = {"mse": "", "cosine": ""}
            else:
                tt_patch = tt_latent[:, frame_start:frame_end]
                tt_metrics = _pair_metrics(predicted_patch, tt_patch)
            rows.append(
                {
                    "sample_id": sample_id,
                    "mode": mode,
                    "lm_history_source": lm_history_source,
                    "dit_history_source": dit_history_source,
                    "patch_index": patch_index,
                    "stage": _patch_stage(patch_index, total_patches),
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "valid_frames": frame_end - frame_start,
                    "mse_to_teacher": teacher_metrics["mse"],
                    "cosine_to_teacher": teacher_metrics["cosine"],
                    "mse_to_tt": tt_metrics["mse"],
                    "cosine_to_tt": tt_metrics["cosine"],
                    "pred_mean": stats["mean"],
                    "pred_std": stats["std"],
                    "pred_rms": stats["rms"],
                    "teacher_rms": teacher_patch.float().square().mean().sqrt().item(),
                }
            )
    return rows


def _build_mode_summaries(results, teacher, latent_frames, patch_size):
    tt_latent = results.get("TT", {}).get("latent")
    frame_ranges = _segment_frame_ranges(latent_frames, patch_size)
    summaries = {}
    for mode, result in results.items():
        predicted = result["latent"]
        segments = {}
        for segment_name, (frame_start, frame_end) in frame_ranges.items():
            predicted_segment = predicted[:, frame_start:frame_end]
            teacher_segment = teacher[:, frame_start:frame_end]
            teacher_metrics = _pair_metrics(predicted_segment, teacher_segment)
            stats = _prediction_stats(predicted_segment)
            if tt_latent is None:
                tt_metrics = {"mse": None, "cosine": None}
            else:
                tt_metrics = _pair_metrics(
                    predicted_segment, tt_latent[:, frame_start:frame_end]
                )
            segments[segment_name] = {
                "frame_start": frame_start,
                "frame_end": frame_end,
                "mse_to_teacher": teacher_metrics["mse"],
                "cosine_to_teacher": teacher_metrics["cosine"],
                "mse_to_tt": tt_metrics["mse"],
                "cosine_to_tt": tt_metrics["cosine"],
                "pred_mean": stats["mean"],
                "pred_std": stats["std"],
                "pred_rms": stats["rms"],
            }
        lm_history_source, dit_history_source = HISTORY_MODES[mode]
        summaries[mode] = {
            "lm_history_source": lm_history_source,
            "dit_history_source": dit_history_source,
            "normalized_latent_shape": list(predicted.shape),
            "audio_shape": list(result["audio"].shape),
            "segments": segments,
        }
    return summaries


def _step_zero_consistency(results, patch_size, atol=1.0e-5):
    modes = list(results)
    reference_mode = "TT" if "TT" in results else modes[0]
    reference = results[reference_mode]["latent"][:, :patch_size]
    max_abs_diff_by_mode = {}
    for mode in modes:
        difference = (results[mode]["latent"][:, :patch_size] - reference).abs()
        max_abs_diff_by_mode[mode] = difference.max().item()
    max_abs_diff = max(max_abs_diff_by_mode.values(), default=0.0)
    return {
        "reference_mode": reference_mode,
        "max_abs_diff_by_mode": max_abs_diff_by_mode,
        "max_abs_diff": max_abs_diff,
        "atol": atol,
        "passed": max_abs_diff <= atol,
    }


def _artifact_paths(args, sample_id, history_modes):
    if args.output:
        output = Path(_resolve_repo_path(args.output))
        stem = output.stem
        return {
            "audio": {history_modes[0]: output},
            "patch_metrics": output.with_name(f"{stem}_patch_metrics.csv"),
            "summary": output.with_name(f"{stem}_summary.json"),
            "reference": output.with_name(f"{stem}_reference.opus"),
        }

    sample_dir = Path(_resolve_repo_path(args.output_dir)) / sample_id
    return {
        "audio": {mode: sample_dir / f"{mode}.wav" for mode in history_modes},
        "patch_metrics": sample_dir / "patch_metrics.csv",
        "summary": sample_dir / "summary.json",
        "reference": sample_dir / "reference.opus",
    }


def _write_patch_metrics(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, indent=2, ensure_ascii=False)
        file_handle.write("\n")


def run(args, infer_cls):
    if args.timesteps <= 0:
        raise ValueError(f"--timesteps must be positive, got {args.timesteps}")
    history_modes = list(args.history_modes)
    if len(set(history_modes)) != len(history_modes):
        raise ValueError(f"--history_modes contains duplicates: {history_modes}")
    if args.output and (len(args.ids) != 1 or len(history_modes) != 1):
        raise ValueError("--output requires exactly one --id and one --history_modes value")

    infer = infer_cls(args)
    print(f"| DiT-AR checkpoint: {infer.ditar_checkpoint}")
    print(f"| VAE checkpoint: {infer.vae_checkpoint}")
    print(
        f"| Non-streaming history ablation: "
        f"{infer.max_new_patches} patches x "
        f"{infer.patch_size} frames, crop to {infer.latent_frames} frames"
    )
    print(f"| History modes: {' '.join(history_modes)} (first=LM, second=DiT)")
    print(f"| Teacher latent directory: {infer.latent_dir}")

    for index, sample_id in enumerate(args.ids):
        seed = args.seed + index
        teacher, results, initial_noise_shape = infer.generate_modes(
            sample_id, seed=seed, history_modes=history_modes
        )
        step_zero = _step_zero_consistency(results, infer.patch_size)
        if not step_zero["passed"]:
            raise RuntimeError(
                "Ablation invariant failed: step 0 must be identical because no "
                f"predicted history exists yet; max abs diff={step_zero['max_abs_diff']}"
            )

        artifact_paths = _artifact_paths(args, sample_id, history_modes)
        sample_rate = int(infer.vae_config.get("sample_rate", 44100))
        for mode, result in results.items():
            output_path = artifact_paths["audio"][mode]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            infer.vae_backend.save_audio(
                str(output_path), result["audio"], sample_rate, infer.vae_config
            )
            print(f"| Saved {mode} audio: {output_path}")

        reference_output = None
        if args.copy_reference:
            reference_source = os.path.join(
                args.original_audio_dir, f"{sample_id}.opus"
            )
            if not os.path.isfile(reference_source):
                raise FileNotFoundError(f"Missing original audio: {reference_source}")
            reference_output = artifact_paths["reference"]
            reference_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(reference_source, reference_output)
            print(f"| Copied reference audio: {reference_output}")

        patch_rows = _build_patch_rows(
            sample_id,
            results,
            teacher,
            infer.latent_frames,
            infer.patch_size,
        )
        _write_patch_metrics(artifact_paths["patch_metrics"], patch_rows)

        mode_summaries = _build_mode_summaries(
            results,
            teacher,
            infer.latent_frames,
            infer.patch_size,
        )
        summary = {
            "sample_id": sample_id,
            "seed": seed,
            "mode_definition": {
                "first_letter": "LM feedback history",
                "second_letter": "DiT latent-context history",
                "T": "ground-truth teacher latent",
                "G": "generated latent patch",
            },
            "history_modes": history_modes,
            "generation": {
                "non_streaming": True,
                "timesteps": args.timesteps,
                "seq_cfg_w": list(args.seq_cfg_w),
                "timestep_annealing_w": list(args.timestep_annealing_w),
                "latent_frames": infer.latent_frames,
                "patch_size": infer.patch_size,
                "max_new_patches": infer.max_new_patches,
                "metrics_space": infer.metrics_space,
                **getattr(infer, "generation_metadata", {}),
            },
            "shared_initial_noise": {
                "seed": seed,
                "shape": list(initial_noise_shape),
                "step_zero_consistency": step_zero,
            },
            "checkpoints": {
                "ditar": str(infer.ditar_checkpoint),
                "vae": str(infer.vae_checkpoint),
            },
            "inputs": {
                "teacher_latent": infer.teacher_latent_source(sample_id),
                "reference_audio": (
                    os.path.join(args.original_audio_dir, f"{sample_id}.opus")
                    if args.copy_reference
                    else None
                ),
            },
            "artifacts": {
                "audio": {
                    mode: str(path) for mode, path in artifact_paths["audio"].items()
                },
                "patch_metrics": str(artifact_paths["patch_metrics"]),
                "reference": str(reference_output) if reference_output else None,
            },
            "modes": mode_summaries,
        }
        _write_summary(artifact_paths["summary"], summary)

        print(f"| ID: {sample_id}")
        print(
            f"| Shared-noise step-0 max abs diff: "
            f"{step_zero['max_abs_diff']:.8g}"
        )
        for mode in history_modes:
            overall = mode_summaries[mode]["segments"]["overall"]
            print(
                f"| {mode} {infer.metrics_space}: "
                f"MSE={overall['mse_to_teacher']:.6f}, "
                f"cosine={overall['cosine_to_teacher']:.6f}"
            )
        print(f"| Saved patch metrics: {artifact_paths['patch_metrics']}")
        print(f"| Saved summary: {artifact_paths['summary']}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Four-way non-streaming history ablation for spatial DiTAR"
    )
    parser.add_argument(
        "--ditar_ckpt",
        required=True,
        help="DiT-AR experiment directory or a specific checkpoint file",
    )
    parser.add_argument(
        "--ditar_config",
        default="",
        help="Defaults to <ditar_ckpt>/config.yaml, then the selected training config",
    )
    parser.add_argument(
        "--id",
        dest="ids",
        nargs="+",
        required=True,
        help="One or more YT-Ambigen IDs without .npy",
    )
    parser.add_argument(
        "--output_dir",
        default="results/spatial_ditar_v8",
        help="Root output directory; each ID gets its own TT/TG/GT/GG directory",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Exact output WAV path; requires one ID and one history mode",
    )
    parser.add_argument(
        "--history_modes",
        nargs="+",
        choices=tuple(HISTORY_MODES),
        default=list(HISTORY_MODES),
        help=(
            "History ablations to run. First letter selects LM history and "
            "second selects DiT context; T=teacher, G=generated"
        ),
    )
    parser.add_argument(
        "--vae_config", default="egs/inference/inference_ditar_v8_vae.yaml"
    )
    parser.add_argument("--mean_path", default="")
    parser.add_argument("--std_path", default="")
    parser.add_argument("--local_clip_dir", default=LOCAL_CLIP_DIR)
    parser.add_argument("--global_clip_dir", default=GLOBAL_CLIP_DIR)
    parser.add_argument(
        "--latent_dir",
        default="",
        help="Teacher posterior cache directory; defaults to training config",
    )
    parser.add_argument("--original_audio_dir", default=ORIGINAL_AUDIO_DIR)
    reference_group = parser.add_mutually_exclusive_group()
    reference_group.add_argument(
        "--copy_reference", "--copy-reference", dest="copy_reference", action="store_true"
    )
    reference_group.add_argument(
        "--no_copy_reference",
        "--no-copy-reference",
        dest="copy_reference",
        action="store_false",
    )
    parser.set_defaults(copy_reference=True)
    parser.add_argument("--timesteps", type=int, default=20)
    parser.add_argument(
        "--seq_cfg_w",
        type=float,
        nargs=2,
        metavar=("SEMANTIC", "HISTORY"),
        default=(1.4, 3.0),
    )
    parser.add_argument(
        "--timestep_annealing_w",
        type=float,
        nargs=3,
        metavar=("A", "B", "P"),
        default=(1.0, 0.0, 1.0),
    )
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_tqdm", action="store_true")
    return parser
