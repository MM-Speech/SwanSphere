"""Video batches with optional GT posteriors for online v8 preferences.

The default path keeps generated-only candidates. With dpo_include_gt_candidate,
load the original Stable Audio mean/std; the task resamples on each training use.
Missing media/cache data is an error, never a change in the training population.
"""
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from tasks.spatial.dataset_utils.spatial_dataset_base import SpatialDatasetBase, _load_feature


class SpatialDiTARV8DPODataset(SpatialDatasetBase):
    def collater(self, samples):
        if len(samples) == 1 and isinstance(samples[0], list):
            samples = samples[0]
        batch = super().collater(samples)
        if not samples:
            return batch
        batch["dpo_reference_path"] = [s["dpo_reference_path"] for s in samples]
        batch["dpo_video_path"] = [s["dpo_video_path"] for s in samples]
        batch["dpo_valid_samples"] = torch.tensor([s["dpo_valid_samples"] for s in samples], dtype=torch.long)
        batch["lat_lens"] = torch.tensor([s["lat_lens"] for s in samples], dtype=torch.long)
        batch["v_mask"] = torch.ones(batch["inputs_embeds"].shape[:2], dtype=torch.long)
        modes = {
            "cached" if "vae_posterior_mean" in s else
            "online" if "audio_vae_wav" in s else "generated_only"
            for s in samples
        }
        if len(modes) != 1:
            raise ValueError("Cannot mix GT posterior, online audio and generated-only DPO items")
        if "cached" in modes:
            for key in ("vae_posterior_mean", "vae_posterior_std"):
                batch[key] = torch.stack([s[key] for s in samples], dim=0)
            batch["vae_posterior_valid_frames"] = torch.tensor(
                [s["vae_posterior_valid_frames"] for s in samples], dtype=torch.long
            )
            if not torch.equal(batch["vae_posterior_valid_frames"], batch["lat_lens"]):
                raise ValueError("GT posterior and DPO reference frame lengths differ")
        elif "online" in modes:
            batch["audio_vae_wav"] = torch.stack([s["audio_vae_wav"] for s in samples], dim=0)
            batch["audio_vae_wav_lens"] = torch.tensor(
                [s["audio_vae_wav_len"] for s in samples], dtype=torch.long
            )
            if not torch.equal(batch["audio_vae_wav_lens"], batch["dpo_valid_samples"]):
                raise ValueError("GT online audio and DPO reference sample lengths differ")
        return batch


def _process_items(raw_item, hp, audio_root, video_root, clip_root, global_root, latent_root=None):
    if int(hp.get("d_lat", 128)) != 128 or hp.get("latent_norm", False):
        raise ValueError("v8 DPO requires raw 128-D paired Stable Audio latents")
    frames = int(hp.get("latent_frames", 215))
    if frames != 215 or int(hp.get("vae_sample_rate", 44100)) != 44100:
        raise ValueError("v8 DPO requires 215 latent frames at 44100 Hz")
    clip_frames, clip_dim = int(hp.get("clip_frames", 40)), int(hp.get("d_clip", 512))
    minimum = float(hp.get("vae_min_audio_ratio", .95))
    include_gt = bool(hp.get("dpo_include_gt_candidate", False))
    online = bool(hp.get("vae_online_encode", False))
    fingerprint = None
    if include_gt and not online:
        from tasks.spatial.dataset_utils.ditar_v8_dataset import _cache_fingerprint
        if hp.get("vae_latent_cache_format") != "stable_audio_pair_posterior_v1":
            raise ValueError("GT candidates require original Stable Audio posterior mean/std caches")
        if not latent_root:
            raise ValueError("GT candidates require a posterior cache directory")
        fingerprint = _cache_fingerprint(latent_root, hp)
    items = []
    for row in raw_item:
        name = str(row.get("file_name", ""))
        stem = Path(name).stem
        if not stem or Path(name).name != name:
            raise ValueError(f"Invalid DPO clip ID: {name!r}")
        audio_path = Path(audio_root) / f"{stem}.opus"
        video_path = Path(video_root) / f"{stem}.mp4"
        if not audio_path.is_file() or not video_path.is_file():
            raise FileNotFoundError(f"DPO requires both reference and video: {audio_path}, {video_path}")
        info = sf.info(str(audio_path))
        if info.channels != 4 or info.samplerate <= 0:
            raise ValueError(f"Expected four-channel FOA: {audio_path}")
        source_samples = min(info.frames, round(10 * info.samplerate))
        valid_samples = min(441000, math.ceil(source_samples * 44100 / info.samplerate))
        if valid_samples < int(441000 * minimum):
            raise ValueError(f"Reference shorter than allowed minimum: {audio_path}")
        valid_frames = min(frames, max(1, valid_samples // 2048))
        gt_fields = {}
        if include_gt:
            if online:
                from modules.foa_vae.stable_audio_posterior import load_foa_waveform
                wave, wave_samples = load_foa_waveform(
                    str(audio_path), sample_rate=44100, audio_seconds=10.0,
                    min_audio_ratio=minimum,
                )
                if int(wave_samples) != valid_samples:
                    raise ValueError(f"GT online audio length differs from DPO reference: {audio_path}")
                gt_fields = {"audio_vae_wav": wave, "audio_vae_wav_len": valid_samples}
            else:
                from modules.foa_vae.stable_audio_posterior import load_posterior
                posterior_path = Path(latent_root) / f"{stem}.npz"
                posterior = load_posterior(posterior_path, expected_fingerprint=fingerprint)
                if (int(posterior["valid_samples"]) != valid_samples or
                        int(posterior["valid_frames"]) != valid_frames):
                    raise ValueError(
                        f"GT posterior length differs from DPO reference: {posterior_path}; "
                        f"cache={posterior['valid_samples']}/{posterior['valid_frames']}, "
                        f"audio={valid_samples}/{valid_frames}"
                    )
                gt_fields = {
                    "vae_posterior_mean": torch.from_numpy(posterior["mean"]),
                    "vae_posterior_std": torch.from_numpy(posterior["std"]),
                    "vae_posterior_valid_frames": valid_frames,
                }
        features = []
        for root, label in ((clip_root, "local CLIP"), (global_root, "global CLIP")):
            array = _load_feature(str(Path(root) / f"{stem}.npy"), label)
            if array.shape[0] < clip_frames or array.shape[1] != clip_dim:
                raise ValueError(f"Bad {label} dimensions for {stem}: {array.shape}")
            array = np.array(array[:clip_frames], dtype=np.float32, copy=True)
            if not np.isfinite(array).all():
                raise ValueError(f"Nonfinite {label} for {stem}")
            features.append(torch.from_numpy(array))
        items.append({
            "file_name": f"{stem}.npy", "len": frames,
            "inputs_embeds": features[0], "global_clip_embedding": features[1],
            "dpo_reference_path": str(audio_path),
            "dpo_video_path": str(video_path),
            "dpo_valid_samples": valid_samples,
            "lat_lens": valid_frames,
            **gt_fields,
        })
    return items


def processer_fn_yt_ambigen(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    del tgt_size, global_stores, skip_logger, i_worker, n_worker
    audio = hparams["yt_audio_base_path"]
    video = hparams.get("dpo_yt_video_base_path", str(Path(audio).parent / "erp_10"))
    return _process_items(
        raw_item, hparams, audio, video, hparams["clip_base_path"],
        hparams["global_clip_base_path"], hparams.get("latent_base_path"),
    )


def processer_fn_sphere360(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    del tgt_size, global_stores, skip_logger, i_worker, n_worker
    audio = hparams["sphere360_audio_base_path"]
    video = hparams.get("dpo_sphere360_video_base_path", str(Path(audio).parent / "train_video_erp_10"))
    return _process_items(
        raw_item, hparams, audio, video, hparams["sphere360_clip_base_path"],
        hparams["sphere360_global_clip_base_path"], hparams.get("sphere360_latent_base_path"),
    )
