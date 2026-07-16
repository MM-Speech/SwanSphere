import os
import random
from pathlib import Path

try:
    from tasks.tts.dataset_utils.tts_fastdataset_v2 import BaseTTSShmDataset
    _BASE_DATASET_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    BaseTTSShmDataset = object
    _BASE_DATASET_IMPORT_ERROR = exc


DEFAULT_FOA_PATH_FIELDS = [
    "target_wav_path",
    "wav_path",
    "foa_path",
    "audio_path",
    "target_path",
    "path",
    "file_path",
]


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _first_non_empty(item, fields):
    for field in fields:
        value = item.get(field)
        if value:
            return str(value)
    return None


def resolve_audio_path_from_item(item, field_priority=None, roots=None):
    fields = field_priority or DEFAULT_FOA_PATH_FIELDS
    path = _first_non_empty(item, fields)
    if path is None:
        available = ", ".join(sorted(item.keys()))
        raise KeyError(f"Cannot resolve FOA audio path from fields {fields}; available keys: {available}")

    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path

    for root in _as_list(roots):
        if root:
            return str(Path(os.path.expanduser(str(root))) / path)
    return path


def compute_target_num_samples(
    sample_rate,
    target_seconds,
    frames_multiple=1,
    hop_size=1,
    align_to_frames_multiple=True,
):
    target = int(round(float(target_seconds) * int(sample_rate)))
    if align_to_frames_multiple:
        multiple = max(1, int(frames_multiple) * int(hop_size))
        target = max(multiple, target // multiple * multiple)
    return target


def _dataset_roots(item, hparams):
    roots = []
    roots.extend(_as_list(hparams.get("foa_audio_roots")))
    roots.extend(_as_list(hparams.get("audio_roots")))
    roots.extend(_as_list(hparams.get("audio_root")))
    roots.extend(_as_list(hparams.get("pyroom_audio_root")))
    roots.extend(_as_list(item.get("audio_root")))
    roots.extend(_as_list(item.get("root")))
    roots.extend(_as_list(item.get("base_dir")))
    return [str(root) for root in roots if root]


def _load_audio(path):
    import torch
    import torchaudio

    wav, sr = torchaudio.load(path)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    return wav.to(torch.float32), sr


def _resample_if_needed(wav, sr, sample_rate):
    if sr == sample_rate:
        return wav
    import torchaudio

    return torchaudio.functional.resample(wav, sr, sample_rate)


def _crop_or_pad(wav, target_length, random_crop):
    import torch.nn.functional as F

    current_length = wav.shape[-1]
    if current_length == target_length:
        return wav
    if current_length > target_length:
        if random_crop:
            start = random.randint(0, current_length - target_length)
        else:
            start = 0
        return wav[:, start : start + target_length]
    return F.pad(wav, (0, target_length - current_length), "constant", 0.0)


def _normalize_foa(wav, mode, eps=1.0e-8):
    if mode in (None, "", "none", False):
        return wav
    if mode == "peak":
        scale = wav.abs().amax().clamp_min(eps)
        return wav / scale
    if mode == "rms":
        scale = wav.square().mean().sqrt().clamp_min(eps)
        return wav / scale
    raise ValueError(f"Unsupported foa_normalize_mode: {mode}")


class FOAVAEV1Dataset(BaseTTSShmDataset):
    if _BASE_DATASET_IMPORT_ERROR is not None:

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "FOAVAEV1Dataset requires the existing fast dataloader stack. "
                "Importing BaseTTSShmDataset failed."
            ) from _BASE_DATASET_IMPORT_ERROR

    def get_batcher(self, hparams, global_stores):
        from utils.commons.base_shm_dataset import get_from_global_stores
        from utils.dataset.batcher import BucketBatcher

        return get_from_global_stores(
            "batcher",
            global_stores,
            lambda: BucketBatcher(
                buckets=[
                    50,
                    100,
                    150,
                    200,
                    250,
                    300,
                    350,
                    400,
                    450,
                    500,
                    550,
                    600,
                    650,
                    700,
                    750,
                    800,
                    850,
                    900,
                    950,
                    1000,
                    1200,
                    1400,
                    1600,
                    1800,
                    2000,
                    2400,
                    2800,
                    3000,
                    3500,
                    4000,
                    4500,
                    5000,
                    6000,
                    7000,
                    8000,
                    9000,
                    10000,
                    11000,
                    12000,
                    14000,
                    16000,
                    18000,
                    20000,
                ],
                dynamic_batch=hparams.get("dynamic_batch", True),
                batch_size=hparams["max_sentences"],
                maximum_bucket_size=hparams.get("max_tokens", None),
                length_fn=lambda x: x["len"],
            ),
        )

    def _process_item(self, processer_fn, raw_item, tgt_size, hparams, global_stores, i_worker, n_worker):
        from utils.commons.base_shm_dataset import get_from_global_stores
        from utils.commons.dataset_utils import SkipLogger

        skip_logger = get_from_global_stores(
            "skip_logger",
            global_stores,
            lambda: SkipLogger(
                [
                    "path_missing",
                    "audio_load_fail",
                    "channel_mismatch",
                    "processer_exception",
                ],
                interval=1000,
                i_worker=i_worker,
                n_worker=n_worker,
            ),
        )

        items = processer_fn(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker)
        if not items:
            return

        bucket_len = int(hparams.get("foa_bucket_len", 35))
        for item in items:
            wav = item.get("wav")
            if wav is None:
                continue
            yield {
                "wav": wav,
                "len": int(item.get("len", bucket_len)),
                "item_name": item.get("item_name", ""),
                "wav_path": item.get("wav_path", ""),
            }

    def collater(self, samples):
        import random as _random
        from utils.commons.dataset_utils import collate_xd

        if len(samples) == 1 and isinstance(samples[0], list):
            samples = samples[0]
        if len(samples) == 0:
            if hasattr(self, "backup_batch") and self.backup_batch is not None:
                print("use backup batch!")
                return self.backup_batch
            print("no batch to take!")
            return {}

        wavs = collate_xd([s["wav"] for s in samples], 0.0)
        batch = {
            "nsamples": len(samples),
            "wavs": wavs,
            "item_names": [s.get("item_name", "") for s in samples],
            "wav_paths": [s.get("wav_path", "") for s in samples],
        }
        if not hasattr(self, "backup_batch") or self.backup_batch is None or _random.random() < 0.001:
            self.backup_batch = batch
        return batch


def processer_fn_pyroom(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    sample_rate = int(hparams.get("sample_rate", hparams.get("audio_sample_rate", 44100)))
    variable_length = bool(hparams.get("foa_variable_length", False))
    target_length = None
    if not variable_length:
        target_seconds = float(hparams.get("foa_target_seconds", hparams.get("target_seconds", 10.0)))
        target_length = compute_target_num_samples(
            sample_rate=sample_rate,
            target_seconds=target_seconds,
            frames_multiple=int(hparams.get("frames_multiple", 1)),
            hop_size=int(hparams.get("hop_size", 1)),
            align_to_frames_multiple=bool(hparams.get("foa_align_to_frames_multiple", True)),
        )
    expected_channels = int(hparams.get("foa_expected_channels", 4))
    field_priority = hparams.get("foa_path_fields", DEFAULT_FOA_PATH_FIELDS)
    random_crop = bool(hparams.get("foa_random_crop", True))
    normalize_mode = hparams.get("foa_normalize_mode", "none")
    bucket_seconds = float(hparams.get("foa_bucket_seconds", 1.0))
    fixed_bucket_len = int(hparams.get("foa_bucket_len", 35))

    items = []
    for item_ in raw_item:
        try:
            wav_path = resolve_audio_path_from_item(
                item_,
                field_priority=field_priority,
                roots=_dataset_roots(item_, hparams),
            )
        except Exception:
            if skip_logger is not None:
                skip_logger.report(1, "path_missing")
            continue

        try:
            wav, sr = _load_audio(wav_path)
            wav = _resample_if_needed(wav, sr, sample_rate)
        except Exception:
            if skip_logger is not None:
                skip_logger.report(1, "audio_load_fail")
            continue

        if wav.shape[0] != expected_channels:
            if skip_logger is not None:
                skip_logger.report(1, "channel_mismatch")
            continue

        try:
            if target_length is not None:
                wav = _crop_or_pad(wav, target_length, random_crop=random_crop)
                item_len = fixed_bucket_len
            else:
                item_len = max(1, int(round(wav.shape[-1] / max(1.0, sample_rate * bucket_seconds))))
            wav = _normalize_foa(wav, normalize_mode)
            items.append(
                {
                    "wav": wav.contiguous(),
                    "len": item_len,
                    "item_name": item_.get("item_name", item_.get("sample_id", "")),
                    "wav_path": wav_path,
                    "source_type": item_.get("metadata_set", item_.get("source_type", "")),
                }
            )
            if skip_logger is not None:
                skip_logger.step(1)
        except Exception:
            if skip_logger is not None:
                skip_logger.report(1, "processer_exception")
            continue

    return items
