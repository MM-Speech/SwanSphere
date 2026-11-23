"""DiTAR v8: paired Stable Audio posterior caches or original FOA audio."""

import math
import os

import numpy as np
import torch

from modules.foa_vae.stable_audio_posterior import (
    load_cache_info,
    load_foa_waveform,
    load_posterior,
)
from tasks.spatial.dataset_utils.spatial_dataset_base import (
    SpatialDatasetBase,
    _load_audio_semantic_w_channel,
    _load_feature,
)


_CACHE_INFO = {}
_CACHE_FORMAT = "stable_audio_pair_posterior_v1"


class SpatialDiTARV8Dataset(SpatialDatasetBase):
    """Keep spatial visual batching; defer posterior sampling to the training device."""

    def collater(self, samples):
        if len(samples) == 1 and isinstance(samples[0], list):
            samples = samples[0]
        batch = super().collater(samples)
        if not samples:
            return batch
        cached = "vae_posterior_mean" in samples[0]
        if any(("vae_posterior_mean" in item) != cached for item in samples):
            raise ValueError("Cannot mix cached posteriors and online audio in one batch")
        if cached:
            for key in ("vae_posterior_mean", "vae_posterior_std"):
                batch[key] = torch.stack([sample[key] for sample in samples], dim=0)
            batch["vae_posterior_valid_frames"] = torch.as_tensor(
                [sample["vae_posterior_valid_frames"] for sample in samples],
                dtype=torch.long,
            )
        else:
            batch["audio_vae_wav"] = torch.stack(
                [sample["audio_vae_wav"] for sample in samples], dim=0
            )
            batch["audio_vae_wav_lens"] = torch.as_tensor(
                [sample["audio_vae_wav_len"] for sample in samples], dtype=torch.long
            )
        return batch


def _load_foa_waveform(path, sample_rate, audio_seconds, min_audio_ratio):
    """Use the exact preprocessing shared with posterior extraction."""
    return load_foa_waveform(path, sample_rate, audio_seconds, min_audio_ratio)


def _cache_fingerprint(directory, hparams):
    directory = os.path.abspath(os.path.expanduser(str(directory)))
    if directory not in _CACHE_INFO:
        # The helper validates schema and encoding contract before any samples.
        _CACHE_INFO[directory] = load_cache_info(directory)
    fingerprint = str(_CACHE_INFO[directory]["fingerprint"])
    expected = str(hparams.get("vae_posterior_fingerprint", "") or "")
    if expected and fingerprint != expected:
        raise ValueError(
            f"Posterior cache fingerprint differs from the configured VAE: {directory}"
        )
    return fingerprint


def _process_items(
    raw_item, hparams, i_worker, n_worker,
    audio_base_path, clip_base_path, global_clip_base_path, latent_base_path="",
):
    online = bool(hparams.get("vae_online_encode", False))
    if int(hparams["d_lat"]) != 128:
        raise ValueError("The paired Stable Audio latent space requires d_lat=128")
    if hparams.get("latent_norm", False):
        raise ValueError("DiTAR v8 paired posteriors require latent_norm=false")
    if not online and hparams.get("vae_latent_cache_format") != _CACHE_FORMAT:
        raise ValueError(f"DiTAR v8 posterior cache format must be {_CACHE_FORMAT}")
    use_semantic = bool(hparams.get("use_audio_semantic_supervision", False))
    if (online or use_semantic) and not audio_base_path:
        raise ValueError("DiTAR v8 online encoding/audio teacher requires a FOA directory")
    if not online and not latent_base_path:
        raise ValueError("Cached DiTAR v8 requires a posterior cache directory")
    latent_frames = int(hparams.get("latent_frames", 215))
    clip_frames = int(hparams["clip_frames"])
    clip_dim = int(hparams["d_clip"])
    sample_rate = int(hparams.get("vae_sample_rate", 44100))
    audio_seconds = float(hparams.get("vae_audio_seconds", 10.0))
    min_ratio = float(hparams.get("vae_min_audio_ratio", 0.95))
    if sample_rate != 44100 or not math.isclose(audio_seconds, 10.0):
        raise ValueError("DiTAR v8 currently expects 10-second audio at 44100 Hz")
    if latent_frames != 215:
        raise ValueError("Paired Stable Audio posterior caches require 215 frames")
    if not 0 < min_ratio <= 1:
        raise ValueError("vae_min_audio_ratio must be in (0, 1]")
    fingerprint = None if online else _cache_fingerprint(latent_base_path, hparams)

    items = []
    attempted = 0
    for metadata_item in raw_item:
        attempted += 1
        file_name = metadata_item.get("file_name")
        if not file_name:
            print(
                f"| Skip metadata without file_name (worker {i_worker}/{n_worker})",
                flush=True,
            )
            continue
        if not file_name.endswith(".npy"):
            file_name = f"{file_name}.npy"
        stem = os.path.splitext(file_name)[0]
        audio_path = os.path.join(audio_base_path, stem + ".opus")
        clip_path = os.path.join(clip_base_path, file_name)
        global_clip_path = os.path.join(global_clip_base_path, file_name)
        try:
            if online:
                audio, valid_samples = _load_foa_waveform(
                    audio_path, sample_rate, audio_seconds, min_ratio
                )
                latent_fields = {
                    "audio_vae_wav": audio,
                    "audio_vae_wav_len": valid_samples,
                }
            else:
                posterior = load_posterior(
                    os.path.join(latent_base_path, stem + ".npz"),
                    expected_fingerprint=fingerprint,
                )
                if int(posterior["valid_samples"]) < int(
                    sample_rate * audio_seconds * min_ratio
                ):
                    raise ValueError("Cached posterior audio is below vae_min_audio_ratio")
                # No random draw in dataset workers: every reuse is resampled
                # on the training device with the same codec sampling function.
                latent_fields = {
                    "vae_posterior_mean": torch.from_numpy(posterior["mean"]),
                    "vae_posterior_std": torch.from_numpy(posterior["std"]),
                    "vae_posterior_valid_frames": int(posterior["valid_frames"]),
                }
            local_clip = _load_feature(clip_path, "local CLIP")
            global_clip = _load_feature(global_clip_path, "global CLIP")
            for feature, name in (
                (local_clip, "local CLIP"), (global_clip, "global CLIP")
            ):
                if feature.shape[0] < clip_frames or feature.shape[1] != clip_dim:
                    raise ValueError(
                        f"Expected {name} [T>={clip_frames}, {clip_dim}], "
                        f"got {feature.shape}"
                    )
            local_clip = np.array(local_clip[:clip_frames], dtype=np.float32, copy=True)
            global_clip = np.array(global_clip[:clip_frames], dtype=np.float32, copy=True)
            if not np.isfinite(local_clip).all() or not np.isfinite(global_clip).all():
                raise ValueError(f"CLIP features contain NaN or Inf: {file_name}")
            item = {
                "inputs_embeds": torch.from_numpy(local_clip),
                "global_clip_embedding": torch.from_numpy(global_clip),
                "file_name": file_name,
                "len": latent_frames,
                **latent_fields,
            }
            if use_semantic:
                semantic_rate = int(
                    hparams.get("audio_semantic_source_sample_rate", 48000)
                )
                semantic_samples = int(round(
                    semantic_rate * float(
                        hparams.get("audio_semantic_audio_seconds", 10.0)
                    )
                ))
                semantic_wav, semantic_len = _load_audio_semantic_w_channel(
                    audio_path, semantic_rate, semantic_samples,
                    float(hparams.get("audio_semantic_min_audio_ratio", 0.95)),
                )
                item.update({
                    "audio_semantic_wav": torch.from_numpy(semantic_wav),
                    "audio_semantic_wav_len": semantic_len,
                })
            items.append(item)
        except Exception as exc:
            print(
                f"| Skip {file_name} (worker {i_worker}/{n_worker}): {exc}",
                flush=True,
            )
    if attempted and not items:
        source = audio_base_path if online else latent_base_path
        raise RuntimeError(
            f"No valid DiTAR v8 examples in this {attempted}-item metadata chunk; "
            f"check the configured source {source} and both CLIP paths. "
            "The specific missing or invalid files were logged above."
        )
    return items


def processer_fn_yt_ambigen(
    raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker
):
    del tgt_size, global_stores, skip_logger
    return _process_items(
        raw_item, hparams, i_worker, n_worker,
        hparams.get("yt_audio_base_path", ""),
        hparams["clip_base_path"],
        hparams["global_clip_base_path"],
        hparams.get("latent_base_path", ""),
    )


def processer_fn_sphere360(
    raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker
):
    del tgt_size, global_stores, skip_logger
    return _process_items(
        raw_item, hparams, i_worker, n_worker,
        hparams.get("sphere360_audio_base_path", ""),
        hparams["sphere360_clip_base_path"],
        hparams["sphere360_global_clip_base_path"],
        hparams.get("sphere360_latent_base_path", ""),
    )
