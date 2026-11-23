import os

import numpy as np
import soundfile as sf
import torch

from tasks.spatial.dataset_utils.tts_fastdataset_v2 import BaseTTSShmDataset
from utils.commons.dataset_utils import collate_xd


_STAT_CACHE = {}


class SpatialDatasetBase(BaseTTSShmDataset):
    """Dataset wrapper that delegates feature loading to the configured processor."""

    def _process_item(
        self,
        processer_fn,
        raw_item,
        tgt_size,
        hparams,
        global_stores,
        i_worker,
        n_worker,
    ):
        items = processer_fn(
            raw_item,
            tgt_size,
            hparams,
            global_stores,
            None,
            i_worker,
            n_worker,
        )
        if not items:
            return

        for item in items:
            yield item

    def collater(self, samples):
        # ``BaseTTSShmDataset.collater_fast`` may wrap a complete batch as
        # ``[samples_list]``.  The parent collater unwraps only its own local
        # variable, so do it here before inspecting semantic-audio fields.
        if len(samples) == 1 and isinstance(samples[0], list):
            samples = samples[0]
        batch = super().collater(samples)
        if not samples:
            return batch
        if not isinstance(samples[0], dict):
            raise TypeError(
                "SpatialDatasetBase.collater expects sample dictionaries, "
                f"got {type(samples[0]).__name__}"
            )
        if samples[0].get("audio_semantic_wav") is not None:
            batch["audio_semantic_wav"] = collate_xd(
                [sample["audio_semantic_wav"] for sample in samples],
                pad_idx=0.0,
            )
            batch["audio_semantic_wav_lens"] = torch.as_tensor(
                [sample["audio_semantic_wav_len"] for sample in samples],
                dtype=torch.long,
            )
        return batch


def _load_stats(norm_path, latent_dim):
    cache_key = (norm_path, latent_dim)
    if cache_key not in _STAT_CACHE:
        mean_path = os.path.join(norm_path, "mean.npy")
        std_path = os.path.join(norm_path, "std.npy")

        mean = np.asarray(np.load(mean_path), dtype=np.float32).reshape(-1)
        std = np.asarray(np.load(std_path), dtype=np.float32).reshape(-1)

        if mean.shape != (latent_dim,):
            raise ValueError(
                f"Expected mean shape ({latent_dim},), got {mean.shape}: {mean_path}"
            )
        if std.shape != (latent_dim,):
            raise ValueError(
                f"Expected std shape ({latent_dim},), got {std.shape}: {std_path}"
            )
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError(f"Latent statistics contain NaN or Inf: {norm_path}")
        if np.any(std <= 0):
            raise ValueError(f"Latent statistics contain non-positive std: {norm_path}")

        _STAT_CACHE[cache_key] = (mean, std)

    return _STAT_CACHE[cache_key]


def _load_feature(path, name):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {name}: {path}")

    array = np.load(path, mmap_mode="r")
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got {array.shape}: {path}")
    return array


def _load_audio_semantic_w_channel(
    path,
    sample_rate,
    target_samples,
    min_audio_ratio,
):
    """Load one fixed-duration FOA W waveform for a sequence audio teacher."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing FOA waveform for audio semantic supervision: {path}"
        )

    waveform, file_sample_rate = sf.read(
        path,
        dtype="float32",
        always_2d=True,
    )
    if file_sample_rate != sample_rate:
        raise ValueError(
            f"Audio semantic waveform must be {sample_rate} Hz, got "
            f"{file_sample_rate}: {path}"
        )
    if waveform.ndim != 2 or waveform.shape[1] != 4:
        raise ValueError(
            "Expected FOA waveform [samples, 4] in WYZX order, got "
            f"{waveform.shape}: {path}"
        )
    if waveform.shape[0] < int(target_samples * min_audio_ratio):
        raise ValueError(
            f"Audio semantic waveform is too short ({waveform.shape[0]} "
            "samples; "
            f"expected about {target_samples}): {path}"
        )

    valid_samples = min(waveform.shape[0], target_samples)
    wav_w = np.zeros((target_samples,), dtype=np.float32)
    # FOA storage order is WYZX, so channel 0 is the omnidirectional W.
    wav_w[:valid_samples] = waveform[:valid_samples, 0]
    if not np.isfinite(wav_w).all():
        raise ValueError(
            f"Audio semantic W waveform contains NaN or Inf: {path}"
        )
    return wav_w, valid_samples


def _process_spatial_items(
    raw_item,
    hparams,
    i_worker,
    n_worker,
    latent_base_path,
    clip_base_path,
    global_clip_base_path,
    audio_base_path,
    *,
    expected_latent_dim=64,
):
    latent_dim = int(hparams["d_lat"])
    latent_frames = int(hparams["latent_frames"])
    clip_frames = int(hparams["clip_frames"])
    clip_dim = int(hparams["d_clip"])
    use_audio_semantic_supervision = bool(
        hparams.get("use_audio_semantic_supervision", False)
    )
    audio_source_sample_rate = int(
        hparams.get("audio_semantic_source_sample_rate", 48000)
    )
    audio_seconds = float(hparams.get("audio_semantic_audio_seconds", 10.0))
    audio_target_samples = int(round(audio_source_sample_rate * audio_seconds))
    audio_min_ratio = float(
        hparams.get("audio_semantic_min_audio_ratio", 0.95)
    )

    if latent_dim != expected_latent_dim:
        raise ValueError(
            f"This latent processor requires d_lat={expected_latent_dim}, "
            f"got {latent_dim}"
        )
    if use_audio_semantic_supervision and not audio_base_path:
        raise ValueError(
            "Audio semantic supervision requires an audio_base_path"
        )
    if audio_target_samples <= 0:
        raise ValueError(
            f"Invalid audio semantic target length: {audio_seconds} seconds "
            f"at {audio_source_sample_rate} Hz"
        )
    if not 0 < audio_min_ratio <= 1:
        raise ValueError(
            "audio_semantic_min_audio_ratio must be in (0, 1], got "
            f"{audio_min_ratio}"
        )

    mean = std = None
    if hparams.get("latent_norm", True):
        mean, std = _load_stats(hparams["norm_path"], latent_dim)

    items = []
    for metadata_item in raw_item:
        file_name = metadata_item.get("file_name")
        if not file_name:
            print(
                f"| Skip metadata without file_name "
                f"(worker {i_worker}/{n_worker}): {metadata_item}"
            )
            continue

        if not file_name.endswith(".npy"):
            file_name = f"{file_name}.npy"

        latent_path = os.path.join(latent_base_path, file_name)
        clip_path = os.path.join(clip_base_path, file_name)
        global_clip_path = os.path.join(global_clip_base_path, file_name)
        audio_file_name = os.path.splitext(file_name)[0] + ".opus"
        audio_path = (
            os.path.join(audio_base_path, audio_file_name)
            if use_audio_semantic_supervision
            else None
        )

        try:
            latent = _load_feature(latent_path, "VAE latent")
            local_clip = _load_feature(clip_path, "local CLIP")
            global_clip = _load_feature(global_clip_path, "global CLIP")
            if use_audio_semantic_supervision:
                audio_semantic_wav, audio_semantic_wav_len = (
                    _load_audio_semantic_w_channel(
                        audio_path,
                        sample_rate=audio_source_sample_rate,
                        target_samples=audio_target_samples,
                        min_audio_ratio=audio_min_ratio,
                    )
                )

            if latent.shape[0] != latent_dim:
                raise ValueError(
                    f"Expected latent [{latent_dim}, T], got {latent.shape}: "
                    f"{latent_path}"
                )
            if latent.shape[1] < latent_frames:
                raise ValueError(
                    f"Latent has {latent.shape[1]} frames, "
                    f"requires at least {latent_frames}: {latent_path}"
                )
            if local_clip.shape[0] < clip_frames:
                raise ValueError(
                    f"Local CLIP has {local_clip.shape[0]} frames, "
                    f"requires at least {clip_frames}: {clip_path}"
                )
            if global_clip.shape[0] < clip_frames:
                raise ValueError(
                    f"Global CLIP has {global_clip.shape[0]} frames, "
                    f"requires at least {clip_frames}: {global_clip_path}"
                )
            if local_clip.shape[1] != clip_dim:
                raise ValueError(
                    f"Expected local CLIP dim {clip_dim}, got {local_clip.shape}"
                )
            if global_clip.shape[1] != clip_dim:
                raise ValueError(
                    f"Expected global CLIP dim {clip_dim}, got {global_clip.shape}"
                )

            # Disk layout is [C, T]; the DiT consumes [T, C].
            latent = np.array(
                latent[:, :latent_frames].T,
                dtype=np.float32,
                copy=True,
            )
            local_clip = np.array(
                local_clip[:clip_frames],
                dtype=np.float32,
                copy=True,
            )
            global_clip = np.array(
                global_clip[:clip_frames],
                dtype=np.float32,
                copy=True,
            )

            if not np.isfinite(latent).all():
                raise ValueError(f"Latent contains NaN or Inf: {latent_path}")
            if not np.isfinite(local_clip).all():
                raise ValueError(f"Local CLIP contains NaN or Inf: {clip_path}")
            if not np.isfinite(global_clip).all():
                raise ValueError(
                    f"Global CLIP contains NaN or Inf: {global_clip_path}"
                )

            if mean is not None:
                latent = (latent - mean) / (std + 1e-6)

            item = {
                "inputs_embeds": torch.from_numpy(local_clip),
                "global_clip_embedding": torch.from_numpy(global_clip),
                "lat": torch.from_numpy(latent),
                "lat_lens": latent_frames,
                "file_name": file_name,
                "len": latent_frames,
            }
            if use_audio_semantic_supervision:
                item.update(
                    {
                        "audio_semantic_wav": torch.from_numpy(
                            audio_semantic_wav
                        ),
                        "audio_semantic_wav_len": audio_semantic_wav_len,
                    }
                )
            items.append(item)
        except Exception as exc:
            print(
                f"| Skip {file_name} (worker {i_worker}/{n_worker}): {exc}"
            )

    return items


def processer_fn_yt_ambigen(
    raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker
):
    del tgt_size, global_stores, skip_logger
    return _process_spatial_items(
        raw_item,
        hparams,
        i_worker,
        n_worker,
        hparams["latent_base_path"],
        hparams["clip_base_path"],
        hparams["global_clip_base_path"],
        hparams.get("audio_semantic_yt_audio_base_path"),
    )


def processer_fn_sphere360(
    raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker
):
    del tgt_size, global_stores, skip_logger
    return _process_spatial_items(
        raw_item,
        hparams,
        i_worker,
        n_worker,
        hparams["sphere360_latent_base_path"],
        hparams["sphere360_clip_base_path"],
        hparams["sphere360_global_clip_base_path"],
        hparams.get("audio_semantic_sphere360_audio_base_path"),
    )
