"""Frozen spatial, ImageBind and Audiobox rewards for online v8 DPO.

Protocol: six fixed cubemap faces, five two-frame clips per face; normalized
mean visual embedding versus the W-channel ImageBind embedding. No candidate
audio influences visual-view selection. Spatial and quality rewards use the
same aligned valid waveform extent. The weighted score is deterministic.
"""
from collections import OrderedDict
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types

import numpy as np
import torch
import torch.nn.functional as F

from modules.spatial.energy_map_reward import EnergyMapReward

ROOT = Path(__file__).resolve().parents[2]
VIEWS = ((0, 0), (180, 0), (-90, 0), (90, 0), (0, 90), (0, -90))
PROTOCOL = "cubemap6_uniform5x2_w3x2s_v1"


def resolve_path(value):
    path = Path(os.path.expanduser(str(value)))
    return path if path.is_absolute() else ROOT / path


def combine_rewards(energy_cc, imagebind_cosine, audiobox_pq, *, spatial_valid=True, weights=(.4, .4, .2)):
    """Fixed bounded normalization; invalid reference space is masked for all candidates."""
    raw = torch.as_tensor([energy_cc, imagebind_cosine, audiobox_pq], dtype=torch.float32)
    weight = torch.as_tensor(weights, dtype=torch.float32)
    if not torch.isfinite(raw).all() or not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("Nonfinite reward or invalid weights")
    if not spatial_valid:
        weight[0] = 0.
    if weight.sum() <= 0:
        raise ValueError("No active reward component")
    normalized = torch.stack(((raw[0].clamp(-1, 1) + 1) / 2,
                              (raw[1].clamp(-1, 1) + 1) / 2,
                              raw[2].clamp(0, 10) / 10))
    return float((normalized * weight).sum() / weight.sum())


def _audio_mels(wave, sample_rate, clips=3):
    """ImageBind's 16k/128-mel/204-frame preprocessing on three fixed 2s windows."""
    import torchaudio
    if wave.ndim != 1 or not torch.isfinite(wave).all() or wave.numel() == 0:
        raise ValueError("ImageBind audio requires a finite nonempty W channel")
    wave = torchaudio.functional.resample(wave.float(), sample_rate, 16000)
    if wave.numel() < 32000:
        wave = F.pad(wave, (0, 32000 - wave.numel()))
    starts = torch.linspace(0, wave.numel() - 32000, clips).round().long().tolist()
    values = []
    for start in starts:
        chunk = wave[start:start + 32000].unsqueeze(0).clone()
        chunk = chunk - chunk.mean()
        mel = torchaudio.compliance.kaldi.fbank(
            chunk, htk_compat=True, sample_frequency=16000, use_energy=False,
            window_type="hanning", num_mel_bins=128, dither=0.,
            frame_length=25, frame_shift=10,
        ).transpose(0, 1)
        mel = F.pad(mel, (0, max(0, 204 - mel.shape[-1])))[:, :204]
        values.append(((mel.unsqueeze(0) + 4.268) / 9.138))
    return torch.stack(values)


def _ffmpeg():
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _decode_view(video_path, yaw, pitch, duration, clips=5):
    command = [
        _ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin",
        "-threads", "1", "-filter_threads", "1", "-i", str(video_path),
        "-an", "-t", str(duration), "-vf",
        f"v360=input=equirect:output=flat:yaw={yaw}:pitch={pitch}:h_fov=90:v_fov=90:w=224:h=224,fps=2",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    raw = subprocess.run(command, check=True, capture_output=True, timeout=120).stdout
    array = np.frombuffer(raw, dtype=np.uint8)
    if array.size == 0 or array.size % (224 * 224 * 3):
        raise ValueError(f"Invalid ERP decode: {video_path}")
    frames = torch.from_numpy(array.copy().reshape(-1, 224, 224, 3)).permute(0, 3, 1, 2).float() / 255
    mean = frames.new_tensor((.48145466, .4578275, .40821073))[None, :, None, None]
    std = frames.new_tensor((.26862954, .26130258, .27577711))[None, :, None, None]
    frames = (frames - mean) / std
    indices = torch.linspace(0, len(frames) - 1, clips * 2).round().long()
    return frames[indices].reshape(clips, 2, 3, 224, 224).permute(0, 2, 1, 3, 4).contiguous()


class V8DPORewards:
    def __init__(self, hp, device):
        self.hp, self.device = hp, torch.device(device)
        self.weights = tuple(float(hp.get(k, v)) for k, v in (
            ("dpo_reward_energy_weight", .4), ("dpo_reward_imagebind_weight", .4),
            ("dpo_reward_audiobox_weight", .2)))
        if any(not np.isfinite(w) or w < 0 for w in self.weights) or sum(self.weights) <= 0:
            raise ValueError("Invalid DPO reward weights")
        self.energy = EnergyMapReward(self.device, float(hp.get("dpo_energy_angle_step_deg", 6)),
                                     float(hp.get("dpo_energy_window_s", .5)))
        self.order = str(hp.get("dpo_foa_channel_order", "WYZX")).upper()
        if sorted(self.order) != sorted("WYZX"):
            raise ValueError("FOA channel order must be a permutation of WYZX")
        self.cache_size = int(hp.get("dpo_reward_cache_size", 32))
        if self.cache_size < 1:
            raise ValueError("dpo_reward_cache_size must be positive")
        self._contexts = OrderedDict()
        self._gt_scores = OrderedDict()
        self.video_cache = resolve_path(hp.get("dpo_imagebind_video_cache", "checkpoints/imagebind/v8_video_cache"))
        self.video_cache.mkdir(parents=True, exist_ok=True)
        self.imagebind = None
        self.audiobox = None
        self._load_models()

    def _load_models(self):
        if self.weights[1] > 0:
            source = resolve_path(self.hp.get("dpo_imagebind_source", "user/av_precompute_bundle/vendor/ImageBind-main"))
            vendor = resolve_path(self.hp.get("dpo_imagebind_python_path", "user/av_precompute_bundle/vendor/python"))
            for path in (vendor, source):
                if not path.is_dir():
                    raise FileNotFoundError(path)
                sys.path.insert(0, str(path))
            if "imagebind" not in sys.modules:
                package = types.ModuleType("imagebind")
                package.__path__ = [str(source / "imagebind")]
                sys.modules["imagebind"] = package
            from imagebind.models import imagebind_model
            checkpoint = resolve_path(self.hp.get("dpo_imagebind_ckpt", "checkpoints/imagebind/imagebind_huge.pth"))
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            model = imagebind_model.imagebind_huge(pretrained=False)
            # Other modalities are unused. Prune before copying weight tensors/GPU transfer.
            for bank_name in ("modality_preprocessors", "modality_trunks", "modality_heads", "modality_postprocessors"):
                bank = getattr(model, bank_name)
                for key in list(bank.keys()):
                    if key not in ("vision", "audio"):
                        del bank[key]
            state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
            expected = model.state_dict()
            selected = {key: value for key, value in state.items() if key in expected}
            model.load_state_dict(selected, strict=True)
            self.imagebind = model.to(self.device).requires_grad_(False).eval()
            stat = checkpoint.stat()
            self._model_identity = f"{checkpoint.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
            del state, selected, expected
        if self.weights[2] > 0:
            python_path = resolve_path(self.hp.get("dpo_audiobox_python_path", "checkpoints/eval_metrics/grpo_python"))
            sys.path.insert(0, str(python_path))
            from audiobox_aesthetics.infer import initialize_predictor
            checkpoint = resolve_path(self.hp.get("dpo_audiobox_ckpt", "checkpoints/audiobox/checkpoint.pt"))
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            self.audiobox = initialize_predictor(ckpt=str(checkpoint))
            self.audiobox.device = self.device
            self.audiobox.model.to(self.device).requires_grad_(False).eval()

    def _video_feature(self, path, duration):
        if self.imagebind is None:
            return None
        stat = Path(path).stat()
        identity = json.dumps([PROTOCOL, self._model_identity, str(Path(path).resolve()),
                               stat.st_size, stat.st_mtime_ns, round(duration, 6)])
        filename = self.video_cache / (hashlib.sha256(identity.encode()).hexdigest() + ".pt")
        if filename.is_file():
            saved = torch.load(filename, map_location="cpu", weights_only=True)
            feature = saved.get("feature")
            if saved.get("identity") == identity and torch.is_tensor(feature) and feature.shape == (1024,) and torch.isfinite(feature).all():
                return feature.to(self.device)
        features = []
        for yaw, pitch in VIEWS:
            pixels = _decode_view(path, yaw, pitch, duration).unsqueeze(0).to(self.device)
            encoded = self.imagebind({"vision": pixels})["vision"][0].float()
            features.append(F.normalize(encoded, dim=-1))
        feature = F.normalize(torch.stack(features).mean(0), dim=-1)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.video_cache, suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                torch.save({"identity": identity, "feature": feature.cpu()}, handle)
            os.replace(temporary, filename)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return feature

    def _context(self, sample, index):
        import soundfile as sf
        import torchaudio
        audio_path = str(sample["dpo_reference_path"][index])
        video_path = str(sample["dpo_video_path"][index])
        frames = int(sample["lat_lens"][index])
        valid_samples = int(sample["dpo_valid_samples"][index])
        # Native decoder length is latent_frames * 2048; compare only valid output.
        length = min(valid_samples, frames * 2048)
        key = (audio_path, video_path, length)
        if key in self._contexts:
            self._contexts.move_to_end(key)
            return self._contexts[key]
        with sf.SoundFile(audio_path) as handle:
            if handle.channels != 4:
                raise ValueError(f"Expected four-channel FOA: {audio_path}")
            sr = handle.samplerate
            wave = handle.read(round(10 * sr), dtype="float32", always_2d=True)
        reference = torch.from_numpy(wave.T.copy()).to(self.device)
        reference = reference[[self.order.index(c) for c in "WYZX"]]
        reference = torchaudio.functional.resample(reference, sr, 16000)
        n16 = math_floor_resampled(length, 44100, 16000)
        if reference.shape[-1] < n16 or not torch.isfinite(reference).all():
            raise ValueError(f"Invalid or short reference FOA: {audio_path}")
        reference = reference[:, :n16]
        context = {"reference": reference, "video": self._video_feature(video_path, length / 44100),
                   "length": length, "samples16": n16}
        self._contexts[key] = context
        while len(self._contexts) > self.cache_size:
            self._contexts.popitem(last=False)
        return context

    @torch.no_grad()
    def prepare(self, sample):
        with torch.autocast(device_type=self.device.type, enabled=False):
            for index in range(len(sample["dpo_reference_path"])):
                self._context(sample, index)

    @torch.no_grad()
    def score_group(self, audios, sample, index, sample_rate=44100):
        import torchaudio
        with torch.autocast(device_type=self.device.type, enabled=False):
            context = self._context(sample, index)
            scores, records = [], []
            for generated in audios:
                generated = generated.to(self.device, dtype=torch.float32)
                if generated.ndim != 2 or generated.shape[0] != 4 or not torch.isfinite(generated).all():
                    raise ValueError("Generated reward audio must be finite [4, samples]")
                generated = generated[[self.order.index(c) for c in "WYZX"]]
                n = round(context["length"] * sample_rate / 44100)
                if generated.shape[-1] < n:
                    raise ValueError("Generated audio shorter than valid candidate duration")
                generated = generated[:, :n]
                x16 = torchaudio.functional.resample(generated, sample_rate, 16000)[:, :context["samples16"]]
                energy, available, coverage, invalid = self.energy.compare(context["reference"], x16, 16000)
                cosine = 0.
                if self.imagebind is not None:
                    mels = _audio_mels(generated[0], sample_rate).unsqueeze(0)
                    embedding = self.imagebind({"audio": mels})["audio"][0].float()
                    embedding = F.normalize(embedding, dim=-1)
                    cosine = float((embedding * context["video"]).sum().clamp(-1, 1))
                quality = 0.
                if self.audiobox is not None:
                    row = self.audiobox.forward([{"path": generated[:1], "sample_rate": sample_rate}])[0]
                    quality = float(row["PQ"])
                total = combine_rewards(energy, cosine, quality, spatial_valid=available, weights=self.weights)
                scores.append(total)
                records.append({"energy_cc": energy, "energy_valid": float(available),
                                "energy_coverage": coverage, "energy_invalid_generated": invalid,
                                "imagebind_cosine": cosine, "audiobox_pq": quality, "reward_total": total})
            return torch.tensor(scores, device=self.device, dtype=torch.float32), records

    @torch.no_grad()
    def score_gt(self, sample, index, sample_rate=44100):
        """Rank original GT audio, with self-correlation 1 only where valid.

        GT training targets are separately sampled from the paired VAE
        posterior. This score deliberately describes the original recording,
        not its VAE reconstruction, as requested for the GT candidate.
        """
        import soundfile as sf
        import torchaudio

        if int(sample_rate) != 44100:
            raise ValueError("GT candidate scoring uses the common 44100-Hz protocol")
        context = self._context(sample, index)
        audio_path = str(sample["dpo_reference_path"][index])
        video_path = str(sample["dpo_video_path"][index])
        stat = Path(audio_path).stat()
        key = (audio_path, stat.st_size, stat.st_mtime_ns,
               video_path, context["length"], int(sample_rate))
        if key in self._gt_scores:
            self._gt_scores.move_to_end(key)
            cached = dict(self._gt_scores[key])
            return float(cached["reward_total"]), cached
        with torch.autocast(device_type=self.device.type, enabled=False):
            with sf.SoundFile(audio_path) as stream:
                if stream.channels != 4:
                    raise ValueError(f"Expected four-channel GT audio: {audio_path}")
                source_rate = int(stream.samplerate)
                array = stream.read(round(10 * source_rate), dtype="float32", always_2d=True)
            # Keep stored channel order; score_group performs the WYZX mapping.
            wave = torch.from_numpy(array.T.copy()).to(self.device)
            wave = torchaudio.functional.resample(wave, source_rate, int(sample_rate))
            if wave.shape[-1] < context["length"] or not torch.isfinite(wave).all():
                raise ValueError(f"Invalid or short GT candidate audio: {audio_path}")
            wave = wave[:, :context["length"]].contiguous()
            _, records = self.score_group([wave], sample, index, sample_rate=int(sample_rate))
            record = dict(records[0])
            available = bool(record["energy_valid"])
            record["energy_cc_measured"] = float(record["energy_cc"])
            record["energy_cc"] = 1.0 if available else float(record["energy_cc"])
            record["reward_total"] = combine_rewards(
                record["energy_cc"], record["imagebind_cosine"], record["audiobox_pq"],
                spatial_valid=available, weights=self.weights,
            )
            record.update(candidate_type="ground_truth", audio_source="original_gt",
                          energy_self_correlation_override=available)
        self._gt_scores[key] = dict(record)
        while len(self._gt_scores) > self.cache_size:
            self._gt_scores.popitem(last=False)
        return float(record["reward_total"]), dict(record)



def math_floor_resampled(length, source_rate, target_rate):
    return max(1, int(length) * int(target_rate) // int(source_rate))
