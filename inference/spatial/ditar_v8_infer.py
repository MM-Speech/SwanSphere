"""V8 paired Stable Audio VAE inference and four-way history evaluation."""

import json
import math
import os
import random
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.spatial import infer_base as shared
from modules.foa_vae.stable_audio_pair import StableAudioPairCodec, sample_posterior
from modules.foa_vae.stable_audio_posterior import (
    load_cache_info, load_foa_waveform, load_posterior, make_cache_info,
)


def build_ditar(config_path, checkpoint_path, device, precision):
    return shared.build_ditar(
        config_path, checkpoint_path, device, precision,
        model_version=8, default_config="egs/spatial/spatial_ditar_v8.yaml",
        expected_latent_dim=128,
    )


def build_vae(config_path, device):
    config = shared.vae_infer.load_inference_config(config_path)
    if config.get("vae_backend") != "stable_audio_pair":
        raise ValueError("V8 requires vae_backend: stable_audio_pair (original stereo VAE twice)")
    if int(config.get("sample_rate", 44100)) != 44100:
        raise ValueError("The original Stable Audio VAE requires sample_rate: 44100")
    if int(config.get("expected_channels", 4)) != 4 or int(config.get("latent_dim", 128)) != 128:
        raise ValueError("V8 requires four FOA audio channels and 128 paired latent channels")
    if config.get("normalize_mode", "none") not in ("none", None, False, ""):
        raise ValueError("V8 original audio must retain its amplitude (normalize_mode: none)")
    model_dir = shared._resolve_repo_path(config.get("model_dir", "checkpoints/vae"))
    codec = StableAudioPairCodec(model_dir).to(device).eval()
    config["device"] = str(device)
    return codec, config, model_dir


def noise_gamma(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("LM history inference gamma must be finite and in [0, 1]")
    return value


class DiTARV8Infer(shared.SpatialDiTARInferBase):
    build_ditar = staticmethod(build_ditar)
    build_vae = staticmethod(build_vae)
    expected_latent_dim = 128
    metrics_space = "raw Stable Audio paired latent (21.5 Hz, 128-D)"

    def __init__(self, args):
        override = getattr(args, "lm_history_noise_infer_gamma", None)
        if override is not None:
            override = noise_gamma(override)
        super().__init__(args)
        if self.hp.get("vae_backend") != "stable_audio_pair":
            raise ValueError("The v8 checkpoint must use vae_backend: stable_audio_pair")
        if int(self.hp.get("vae_sample_rate", 44100)) != self.vae.sample_rate:
            raise ValueError("Checkpoint audio sample rate differs from the Stable Audio VAE")
        self.audio_seconds = float(self.hp.get("vae_audio_seconds", 10.0))
        if self.audio_seconds != 10.0 or self.latent_frames != 215:
            raise ValueError("V8 paired VAE inference currently requires 10 s / 215 valid latent frames")
        self.sample_posterior = bool(self.hp.get(
            "vae_sample_posterior", self.vae_config.get("sample_posterior", True)
        ))
        self.teacher_cache = str(getattr(args, "teacher_latents", "") or args.latent_dir or "")
        if not self.teacher_cache and self.hp.get("vae_latent_cache_format") == "stable_audio_pair_posterior_v1":
            audio_dir = Path(shared._resolve_repo_path(args.original_audio_dir)).resolve()
            for audio_key, cache_key in (
                ("yt_audio_base_path", "latent_base_path"),
                ("sphere360_audio_base_path", "sphere360_latent_base_path"),
            ):
                configured_audio = self.hp.get(audio_key, "")
                if configured_audio and audio_dir == Path(shared._resolve_repo_path(configured_audio)).resolve():
                    self.teacher_cache = str(self.hp.get(cache_key, "") or "")
                    break
        if self.teacher_cache:
            self.teacher_cache = shared._resolve_repo_path(self.teacher_cache)
            if self.teacher_cache.endswith(".npy"):
                raise ValueError("V8 posterior caches must be .npz with mean/std; old sampled .npy latents are unsupported")
            if self.teacher_cache.endswith(".npz") and len(args.ids) != 1:
                raise ValueError("A single --teacher_latents .npz file requires exactly one --id")
        self._posterior_cache_infos = {}
        self._posterior_fingerprint = None
        self.latent_dir = self.teacher_cache or f"online audio: {args.original_audio_dir}"
        if override is not None:
            self.model.config.lm_history_noise_infer_gamma = override
        self.generation_metadata = {
            "vae_backend": "stable_audio_pair",
            "latent_dim": self.expected_latent_dim,
            "latent_norm": self.latent_norm,
            "vae_sample_posterior": self.sample_posterior,
            "teacher_posterior_seed_offset": 2_000_003,
            "lm_history_noise_infer_gamma": self.model.config.lm_history_noise_infer_gamma,
            "use_lm_history_bottleneck": self.model.config.use_lm_history_bottleneck,
            "lm_history_bottleneck_dim": self.model.config.lm_history_bottleneck_dim,
            "paired_lm_history_noise": True,
            "lm_history_noise_seed_offset": 1_000_003,
        }

    def _load_latent_stats(self, args):
        self.latent_norm = bool(self.hp.get("latent_norm", False))
        if not self.latent_norm:
            if args.mean_path or args.std_path:
                raise ValueError("latent_norm is false; omit --mean_path and --std_path for raw 128-D latents")
            self.metrics_space = "raw Stable Audio paired latent (21.5 Hz, 128-D)"
            return torch.zeros(self.expected_latent_dim), torch.ones(self.expected_latent_dim)
        norm_path = self.hp.get("norm_path", "")
        mean_path = args.mean_path or (str(Path(norm_path) / "mean.npy") if norm_path else "")
        std_path = args.std_path or (str(Path(norm_path) / "std.npy") if norm_path else "")
        if not mean_path or not std_path:
            raise ValueError("Normalized 128-D inference requires norm_path or both explicit statistic paths")
        self.metrics_space = "normalized Stable Audio paired latent (21.5 Hz, 128-D)"
        return shared.load_latent_stats(mean_path, std_path, self.expected_latent_dim)

    def teacher_latent_source(self, sample_id):
        if self.teacher_cache:
            if self.teacher_cache.endswith(".npz"):
                return self.teacher_cache
            cached = Path(self.teacher_cache) / f"{sample_id}.npz"
            if cached.is_file():
                return str(cached)
        return shared._resolve_repo_path(str(Path(self.args.original_audio_dir) / f"{sample_id}.opus"))

    def _load_cached_posterior(self, source):
        # A directory manifest identifies both weights and preprocessing.
        # Derive the current codec identity once; an optional checkpoint pin
        # must agree with it as well as with the selected cache directory.
        if self._posterior_fingerprint is None:
            self._posterior_fingerprint = make_cache_info(
                self.vae.model_dir,
                min_audio_ratio=float(self.hp.get("vae_min_audio_ratio", 0.95)),
            )["fingerprint"]
            pinned = self.hp.get("vae_posterior_fingerprint") or None
            if pinned and pinned != self._posterior_fingerprint:
                raise ValueError("Checkpoint posterior fingerprint differs from the selected original VAE")
        directory = str(Path(source).parent)
        if directory not in self._posterior_cache_infos:
            info = load_cache_info(directory)
            if info["fingerprint"] != self._posterior_fingerprint:
                raise ValueError(f"Posterior cache was produced by different weights or preprocessing: {directory}")
            self._posterior_cache_infos[directory] = info
        return load_posterior(source, expected_fingerprint=self._posterior_fingerprint)

    @torch.no_grad()
    def _load_teacher_latent(self, sample_id):
        source = self.teacher_latent_source(sample_id)
        if source.endswith(".npz"):
            posterior = self._load_cached_posterior(source)
            mean = torch.from_numpy(posterior["mean"]).unsqueeze(0).to(self.device)
            std = torch.from_numpy(posterior["std"]).unsqueeze(0).to(self.device)
            with shared._autocast(self.device, torch.float32):
                latent = sample_posterior(mean, std, use_mean=not self.sample_posterior)
            valid_samples = posterior["valid_samples"]
            valid_frames = posterior["valid_frames"]
        else:
            # Shared with extraction and online training, including source-rate
            # crop, resample_poly, exact 441000 samples and the short-file guard.
            audio, valid_samples = load_foa_waveform(
                source, sample_rate=self.vae.sample_rate,
                audio_seconds=self.audio_seconds,
                min_audio_ratio=float(self.hp.get("vae_min_audio_ratio", 0.95)),
            )
            audio = audio.unsqueeze(0).to(self.device)
            # Native Oobleck encoding maps 441000 samples to 215 frames.
            with shared._autocast(self.device, torch.float32):
                latent = self.vae.encode(audio, use_mean=not self.sample_posterior)
            valid_frames = max(1, min(self.latent_frames, valid_samples // self.vae.downsample_factor))
        if latent.shape[-1] < self.latent_frames:
            raise ValueError(f"Encoded only {latent.shape[-1]} frames from {source}")
        teacher = latent[:, :, :self.latent_frames].transpose(1, 2).contiguous().float()
        if not torch.isfinite(teacher).all():
            raise ValueError(f"Teacher latent contains NaN or Inf: {source}")
        if self.latent_norm:
            teacher = (teacher - self.mean.to(self.device)) / (self.std.to(self.device) + 1.0e-6)
        self.generation_metadata.update({
            "teacher_valid_samples": valid_samples,
            "teacher_valid_frames": valid_frames,
            "teacher_posterior_fingerprint": (
                self._posterior_fingerprint if source.endswith(".npz") else None
            ),
        })
        return F.pad(teacher, (0, 0, 0, (-self.latent_frames) % self.patch_size))

    def _decode_latent(self, generated_latent):
        latent = generated_latent
        if self.latent_norm:
            mean = self.mean.to(device=self.device)[None, None]
            scale = (self.std + 1.0e-6).to(device=self.device)[None, None]
            latent = latent * scale + mean
        latent = latent.transpose(1, 2).contiguous()
        vae_dtype = self.vae_backend.autocast_dtype(self.vae_config.get("precision", "fp32"))
        with shared._autocast(self.device, vae_dtype):
            audio = self.vae.decode(latent)
        audio = audio[0].detach().float().cpu()
        if audio.ndim != 2 or audio.shape[0] != 4:
            raise ValueError(f"Expected four-channel FOA audio, got {tuple(audio.shape)}")
        if not torch.isfinite(audio).all():
            raise ValueError("Decoded audio contains NaN or Inf")
        return audio

    @torch.inference_mode()
    def generate_modes(self, sample_id, seed, history_modes):
        # One reproducible teacher posterior sample shared by all modes.
        torch.manual_seed(seed + 2_000_003)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed + 2_000_003)
        inputs, teacher = self._prepare_inputs(sample_id)
        self.generation_metadata["teacher_latent_online_encoded"] = not self.teacher_latent_source(sample_id).endswith(".npz")
        initial_noise = self._make_initial_noise(seed)
        results = {}
        for mode in history_modes:
            lm_source, dit_source = shared.HISTORY_MODES[mode]
            # The same per-patch epsilon is used for every ablation mode.
            # Resetting also makes results independent of mode ordering.
            torch.manual_seed(seed + 1_000_003)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed + 1_000_003)
            with shared._autocast(self.device, self.dtype):
                generated, _ = self.model.inference(
                    inputs,
                    timesteps=self.args.timesteps,
                    seq_cfg_w=tuple(self.args.seq_cfg_w),
                    timestep_annealing_w=tuple(self.args.timestep_annealing_w),
                    use_tqdm=not self.args.no_tqdm,
                    max_new_patches=self.max_new_patches,
                    teacher_latent=teacher,
                    lm_history_source=lm_source,
                    dit_history_source=dit_source,
                    initial_noise=initial_noise,
                )
            generated = self._validate_generated_latent(generated)
            results[mode] = {
                "latent": generated.detach().cpu(),
                "audio": self._decode_latent(generated),
            }
        return teacher[:, :self.latent_frames].float().cpu(), results, tuple(initial_noise.shape)


class SphereTest32Infer(DiTARV8Infer):
    def generate_modes(self, sample_id, seed, history_modes):
        sample = self.args.sphere_samples[sample_id]
        self.args.local_clip_dir = Path(sample["local_clip"]).parent
        self.args.global_clip_dir = Path(sample["global_clip"]).parent
        result = super().generate_modes(sample_id, seed, history_modes)
        artifacts = shared._artifact_paths(self.args, sample_id, history_modes)
        output_dir = artifacts["summary"].parent
        video_name = f"{Path(self.args.output).stem}_video.mp4" if self.args.output else "video.mp4"
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample["video"], output_dir / video_name)
        return result


def sphere_test_candidates(config, sphere, audio_dir):
    import soundfile as sf
    from utils.data_utils.extract_clip_features import load_ids

    train_sources = {
        clip_id.rsplit("_", 1)[0]
        for dataset in config["datasets"].values()
        for manifest in dataset["train_sets"]
        for clip_id in load_ids(PROJECT_ROOT / manifest)
    }
    test_ids = load_ids(PROJECT_ROOT / "user/metadata/sphere360_test.jsonl")
    candidates = {i for i in test_ids if i.rsplit("_", 1)[0] not in train_sources}
    for directory, suffix in ((audio_dir, ".opus"), (sphere / "test_video_erp_10", ".mp4"),
                              (sphere / "test_front_erp_10", ".mp4")):
        candidates.intersection_update(p.stem for p in directory.glob(f"*{suffix}"))

    min_ratio = float(config.get("vae_min_audio_ratio", 0.95))
    seconds = float(config.get("vae_audio_seconds", 10))

    def has_reference(clip_id):
        try:
            info = sf.info(audio_dir / f"{clip_id}.opus")
        except (OSError, RuntimeError):
            return False
        return (info.channels == 4 and info.samplerate > 0 and
                info.frames >= int(round(seconds * info.samplerate) * min_ratio))

    ordered = sorted(candidates)
    by_video = defaultdict(list)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for clip_id, valid in zip(ordered, pool.map(has_reference, ordered)):
            if valid:
                by_video[clip_id.rsplit("_", 1)[0]].append(clip_id)
    return by_video


def configure_sphere_test32(args):
    from utils.commons.hparams import set_hparams
    from utils.data_utils.extract_clip_features import valid_feature

    checkpoint = Path(shared.vae_infer.resolve_checkpoint_path(
        shared._resolve_repo_path(args.ditar_ckpt), prefer_model_only_last=False,
    ))
    args.ditar_ckpt = str(checkpoint)
    saved_config = checkpoint.parent / "config.yaml"
    config_path = shared._resolve_ditar_config(
        saved_config if saved_config.is_file() else args.ditar_config, checkpoint,
        default_config="egs/spatial/spatial_ditar_v8.yaml",
    )
    config = set_hparams(config=config_path, global_hparams=False, print_hparams=False)
    sphere = Path(config["sphere360_audio_base_path"]).parent
    audio_dir = Path(shared._resolve_repo_path(args.original_audio_dir)) if args.original_audio_dir else sphere / "test_audio"
    args.original_audio_dir = str(audio_dir)
    by_video = sphere_test_candidates(config, sphere, audio_dir)
    explicit = args.ids is not None
    if explicit:
        eligible = {i for clips in by_video.values() for i in clips}
        missing = set(args.ids) - eligible
        if missing:
            raise ValueError(f"Ineligible Sphere360 test IDs: {sorted(missing)}")
    else:
        if len(by_video) < 32:
            raise ValueError(f"Need 32 distinct held-out videos; found {len(by_video)}")
        rng = random.Random(args.seed)
        args.ids = [rng.choice(by_video[video]) for video in rng.sample(sorted(by_video), 32)]

    if not args.output_dir:
        tag = checkpoint.stem.replace("model_ckpt_steps_", "step", 1)
        label = "sphere_ids" if explicit else "sphere_test32"
        args.output_dir = str(
            PROJECT_ROOT / "results" / checkpoint.parent.name / f"{label}_{tag}_seed{args.seed}"
        )
    output_root = (
        Path(shared._resolve_repo_path(args.output)).parent
        if args.output else Path(shared._resolve_repo_path(args.output_dir))
    )
    # New features belong to this inference output; older caches remain reusable.
    cache = output_root / "heldout_features"
    previous = [
        cache,
        checkpoint.parent / "inference/sphere360_clip_vitb32_4fps",
        *[PROJECT_ROOT / "results" / experiment / "heldout_features" for experiment in (
            "261015_ditar_v7_16k_eval", "261017_ditar_v7_improvements_eval",
        )],
    ]
    args.sphere_samples = {}
    for clip_id in args.ids:
        row = {
            "id": clip_id, "audio": audio_dir / f"{clip_id}.opus",
            "video": sphere / "test_video_erp_10" / f"{clip_id}.mp4",
            "local_video": sphere / "test_front_erp_10" / f"{clip_id}.mp4",
            "global_video": sphere / "test_pad_360_erp_10" / f"{clip_id}.mp4",
        }
        for stream, override in (("local_clip", args.local_clip_dir), ("global_clip", args.global_clip_dir)):
            name = f"{clip_id}.npy"
            if override:
                path = Path(shared._resolve_repo_path(override)) / name
                if not valid_feature(path):
                    raise ValueError(f"Invalid or missing {stream} override: {path}")
            else:
                path = next((root / stream / name for root in previous if valid_feature(root / stream / name)),
                            cache / stream / name)
            row[stream] = path
        args.sphere_samples[clip_id] = row

    args.selection = {
        "method": "explicit_ids" if explicit else "sample_distinct_videos_then_one_clip_per_video",
        "seed": args.seed, "checkpoint": str(checkpoint),
        "eligible_videos": len(by_video), "eligible_clips": sum(map(len, by_video.values())),
        "ids": args.ids,
    }


def save_sphere_selection(args):
    if args.output:
        output = Path(shared._resolve_repo_path(args.output))
        path = output.with_name(f"{output.stem}_selection.json")
    else:
        path = Path(shared._resolve_repo_path(args.output_dir)) / "selection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(args.selection, handle, indent=2)
            handle.write("\n")
            handle.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"| Sphere360 seed {args.seed}, {len(args.ids)} selected clips: {path}", flush=True)


def write_sphere_global_view(source, target, ffmpeg):
    """Match the RGB padding, area resize and H.264 encoding used by pad_360."""
    import subprocess
    from contextlib import closing
    import imageio_ffmpeg
    import numpy as np

    # Sphere360 stores 896x448 ERP at 8 fps. For 896 -> 224, OpenCV INTER_AREA
    # averages each 4x4 block and rounds ties to even. This also avoids libGL.
    frames = imageio_ffmpeg.read_frames(str(source), input_params=["-threads", "1"],
                                        output_params=["-threads", "1"])
    bgr_frames = bytearray()
    with closing(frames):
        metadata = next(frames)
        width, height = metadata["size"]
        if (width, height) != (896, 448):
            raise ValueError(f"Expected 896x448 Sphere360 ERP, got {width}x{height}: {source}")
        square = np.zeros((896, 896, 3), dtype=np.uint8)
        for frame in frames:
            square[224:672] = np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 3)
            sums = square.reshape(224, 4, 224, 4, 3).sum(axis=(1, 3), dtype=np.uint16)
            resized = np.rint(sums.astype(np.float32) / 16).astype(np.uint8)
            bgr_frames.extend(resized[:, :, ::-1].tobytes())
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-threads", "1", "-filter_threads", "1",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s:v", "224x224",
        "-framerate", str(metadata["fps"]), "-i", "pipe:0", "-an",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p", str(target),
    ]
    subprocess.run(command, input=bgr_frames, check=True, timeout=120)


@torch.inference_mode()
def prepare_sphere_features(samples, device):
    """Build just the missing local/global CLIP arrays for the selected clips."""
    from tempfile import TemporaryDirectory
    from utils.data_utils import extract_clip_features as extractor

    pending = []
    for sample in samples:
        for stream in ("local", "global"):
            target = Path(sample[f"{stream}_clip"])
            if not extractor.valid_feature(target):
                pending.append((sample, stream, target))
    if not pending:
        return

    import clip
    weight = extractor.DEFAULT_MODEL
    if not weight.is_file():
        raise FileNotFoundError(f"Missing CLIP ViT-B/32 weights: {weight}")
    ffmpeg = extractor.resolve_ffmpeg()
    model, preprocess = clip.load(str(weight), device=device, jit=False)
    model.eval().requires_grad_(False)
    normalize = preprocess.transforms[-1]
    print(f"| Preparing {len(pending)} Sphere360 CLIP features", flush=True)
    with TemporaryDirectory(prefix="ditar_v8_sphere_") as temp_dir:
        for sample, stream, target in pending:
            source = Path(sample[f"{stream}_video"])
            if stream == "global" and not source.is_file():
                source = Path(temp_dir) / "global.mp4"
                write_sphere_global_view(sample["video"], source, ffmpeg)
            frames = extractor.decode_video(source, ffmpeg)
            images = frames.to(device=device, dtype=torch.float32).div_(255.0)
            features = model.encode_image(normalize(images)).to(device="cpu", dtype=torch.float16)
            if features.shape != (40, 512) or not torch.isfinite(features).all():
                raise ValueError(f"Invalid {stream} CLIP output for {sample['id']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            extractor.atomic_save(target, features.numpy())


def build_parser(sphere_test32=False):
    parser = shared.build_parser()
    parser.description = "DiTAR v8: original Stable Audio VAE twice, LM bottleneck and history noise"
    parser.epilog = "Sphere360 held-out inference: ditar_v8_infer.py sphere-test32 --ditar_ckpt EXPERIMENT"
    parser.set_defaults(
        output_dir="results/spatial_ditar_v8",
        vae_config="egs/inference/inference_ditar_v8_vae.yaml",
        mean_path="",
        std_path="",
    )
    if sphere_test32:
        parser.prog += " sphere-test32"
        parser.description = "Randomly select 32 held-out Sphere360 videos and one clip per video using --seed."
        parser.set_defaults(output_dir=None, history_modes=["GG"], original_audio_dir=None,
                            local_clip_dir=None, global_clip_dir=None)
    parser.add_argument(
        "--teacher_latents", default="",
        help="Optional posterior .npz file or directory; otherwise use the matching configured cache, then original FOA audio",
    )
    parser.add_argument(
        "--lm_history_noise_infer_gamma", type=noise_gamma, default=None,
        help="Override checkpoint setting: 0 disables LM history noise; LiveFoley uses 0.15",
    )
    for action in parser._actions:
        if sphere_test32 and action.dest == "ids":
            action.required = False
            action.help = "Explicit eligible test IDs, bypassing random selection"
        elif sphere_test32 and action.dest == "seed":
            action.help = "Seed for choosing the 32 clips and the base generation seed (default: 42)"
        elif action.dest == "output_dir":
            default = (
                "results/<experiment>/sphere_test32_<step>_seed<seed> "
                "(sphere_ids_<step>_seed<seed> with --id)"
                if sphere_test32 else "results/spatial_ditar_v8"
            )
            action.help = (
                f"Output root; saves each ID in its own subdirectory (default: {default}). "
                "Relative paths resolve from the project root; --output takes precedence."
            )
        elif action.dest == "output":
            action.help = (
                "Exact output WAV path; requires one ID and one history mode. "
                "Takes precedence over --output_dir."
            )
        elif action.dest == "ditar_config":
            action.help = "Defaults to checkpoint config.yaml, then the v8 training config"
        elif action.dest in ("mean_path", "std_path"):
            action.help = "128-D statistics override only when checkpoint latent_norm is true"
        elif action.dest == "latent_dir":
            action.help = "Optional posterior mean/std .npz directory; missing entries are encoded from original audio"
    return parser


def main():
    argv = sys.argv[1:]
    if argv[:1] == ["sphere-test32"]:
        args = build_parser(sphere_test32=True).parse_args(argv[1:])
        configure_sphere_test32(args)
        save_sphere_selection(args)
        prepare_sphere_features(args.sphere_samples.values(), args.device)
        shared.run(args, infer_cls=SphereTest32Infer)
    else:
        shared.run(build_parser().parse_args(argv), infer_cls=DiTARV8Infer)


if __name__ == "__main__":
    main()
