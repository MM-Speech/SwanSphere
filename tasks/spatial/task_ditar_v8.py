"""Train DiTAR v8 on paired Stable Audio posteriors, cached or encoded online."""

import math

import torch

from tasks.spatial.ditar_base_task import SpatialDiTARBaseTask
from utils.commons.hparams import hparams


class SpatialDiTARV8Task(SpatialDiTARBaseTask):
    """Sample paired VAE targets, then use the shared training and loss path."""

    model_version = 8
    expected_latent_dim = 128

    @staticmethod
    def validate_config(config):
        for key in ("latent_dim", "d_lat"):
            if int(config.get(key, 128)) != 128:
                raise ValueError(f"DiTAR v8 requires {key}=128")
        if not config.get("vae_online_encode", False):
            if config.get("vae_latent_cache_format") != "stable_audio_pair_posterior_v1":
                raise ValueError(
                    "Cached DiTAR v8 requires "
                    "vae_latent_cache_format=stable_audio_pair_posterior_v1"
                )
        if config.get("vae_backend") != "stable_audio_pair":
            raise ValueError("DiTAR v8 requires vae_backend=stable_audio_pair")
        if not str(config.get("vae_model_dir", "")).strip():
            raise ValueError("DiTAR v8 requires a local vae_model_dir")
        if int(config.get("latent_frames", 215)) != 215:
            raise ValueError("DiTAR v8 uses 215 valid latent frames per 10-second clip")
        if not math.isclose(
            float(config.get("latent_frame_rate", 21.5)), 21.5,
            rel_tol=0.0, abs_tol=1e-6,
        ):
            raise ValueError("DiTAR v8 requires the 21.5 Hz Stable Audio latent space")
        if int(config.get("vae_sample_rate", 44100)) != 44100:
            raise ValueError("The original Stable Audio VAE requires 44100 Hz")
        if not math.isclose(
            float(config.get("vae_audio_seconds", 10.0)), 10.0,
            rel_tol=0.0, abs_tol=1e-6,
        ):
            raise ValueError("DiTAR v8 currently expects 10-second training clips")
        if int(config.get("vae_forward_batch_size", 1)) <= 0:
            raise ValueError("vae_forward_batch_size must be positive")
        if str(config.get("vae_model_dtype", "float32")) != "float32":
            raise ValueError("Online Stable Audio encoding currently requires float32")
        if config.get("latent_norm", False):
            raise ValueError(
                "DiTAR v8 uses raw paired Stable Audio latents; "
                "set latent_norm=false"
            )

    def build_model(self):
        self.validate_config(hparams)
        return super().build_model()

    def _get_vae_codec(self, device):
        """Keep the frozen per-rank codec outside the trainable module tree.

        BaseTask is an nn.Module: ordinary assignment would register the codec,
        let task.train() change its mode, and include it in task.state_dict().
        A private non-registered cache avoids all three and is created only
        after the trainer has moved a batch to its final device.
        """
        device = torch.device(device)
        cache = self.__dict__.get("_online_vae_cache")
        if cache is None or cache[0] != device:
            from modules.foa_vae.stable_audio_pair import StableAudioPairCodec

            codec = StableAudioPairCodec(hparams["vae_model_dir"])
            codec.requires_grad_(False)
            codec.to(device=device, dtype=torch.float32).eval()
            cache = (device, codec)
            self.__dict__["_online_vae_cache"] = cache
        return cache[1].eval()

    @torch.no_grad()
    def _encode_audio_sample(self, sample):
        if "audio_vae_wav" not in sample:
            raise KeyError(
                "Online DiTAR v8 requires audio_vae_wav from SpatialDiTARV8Dataset"
            )
        audio = sample["audio_vae_wav"]
        expected_samples = int(round(
            float(hparams.get("vae_audio_seconds", 10.0))
            * int(hparams.get("vae_sample_rate", 44100))
        ))
        if audio.ndim != 3 or audio.shape[1:] != (4, expected_samples):
            raise ValueError(
                f"Expected FOA audio [B, 4, {expected_samples}], got {tuple(audio.shape)}"
            )
        if audio.shape[0] == 0:
            raise ValueError("Cannot encode an empty audio batch")
        if not torch.isfinite(audio).all():
            raise ValueError("FOA audio contains NaN or Inf")
        codec = self._get_vae_codec(audio.device)
        batch_size = int(hparams.get("vae_forward_batch_size", 1))
        if batch_size <= 0:
            raise ValueError("vae_forward_batch_size must be positive")
        latent_frames = int(hparams.get("latent_frames", 215))
        use_mean = not bool(hparams.get("vae_sample_posterior", True))
        chunks = []
        # The trainer enters autocast before run_model; explicitly keep the
        # frozen codec in FP32. Do not use inference_mode: these latents feed
        # trainable projections that need to save their input for backward.
        with torch.autocast(device_type=audio.device.type, enabled=False):
            for chunk in audio.split(batch_size, dim=0):
                chunk = chunk.float()
                # Oobleck naturally maps 441000 samples to 215 frames.
                # Waveform padding would change its boundary context.
                latent = codec.encode(chunk, use_mean=use_mean)
                if (
                    latent.ndim != 3
                    or latent.shape[0] != chunk.shape[0]
                    or latent.shape[1] != self.expected_latent_dim
                    or latent.shape[-1] < latent_frames
                ):
                    raise ValueError(
                        f"Expected paired latents [B, 128, T>={latent_frames}], "
                        f"got {tuple(latent.shape)}"
                    )
                chunks.append(latent[:, :, :latent_frames])
        lat = torch.cat(chunks, dim=0).transpose(1, 2).contiguous()
        if not torch.isfinite(lat).all():
            raise ValueError("Online Stable Audio latents contain NaN or Inf")
        wav_lens = sample.get("audio_vae_wav_lens")
        if wav_lens is None:
            wav_lens = torch.full(
                (audio.shape[0],), expected_samples,
                device=audio.device, dtype=torch.long,
            )
        else:
            wav_lens = torch.as_tensor(
                wav_lens, device=audio.device, dtype=torch.long
            ).reshape(-1)
        if wav_lens.numel() != audio.shape[0]:
            raise ValueError("audio_vae_wav_lens must have one length per example")
        if (wav_lens <= 0).any() or (wav_lens > expected_samples).any():
            raise ValueError("audio_vae_wav_lens contains an invalid waveform length")
        lat_lens = torch.div(
            wav_lens, codec.downsample_factor, rounding_mode="floor"
        ).clamp(min=1, max=latent_frames)
        model_sample = dict(sample)
        model_sample.update({"lat": lat, "lat_lens": lat_lens})
        return model_sample

    @torch.no_grad()
    def _sample_cached_posterior(self, sample):
        """Resample the cached conditional posterior on every training use.

        These per-example standard deviations are not dataset normalization
        statistics. Keep the same pair layout and microbatch RNG boundaries
        as StableAudioPairCodec.encode so online and cached sampling agree.
        """
        from modules.foa_vae.stable_audio_pair import sample_posterior

        required = (
            "vae_posterior_mean", "vae_posterior_std", "vae_posterior_valid_frames"
        )
        for key in required:
            if key not in sample:
                raise KeyError(f"Cached DiTAR v8 requires {key} from its dataset")
        mean, std = sample["vae_posterior_mean"], sample["vae_posterior_std"]
        frames = int(hparams.get("latent_frames", 215))
        if (
            mean.ndim != 3 or mean.shape[0] < 1
            or mean.shape[1:] != (self.expected_latent_dim, frames)
            or std.shape != mean.shape
        ):
            raise ValueError(
                f"Expected posterior mean/std [B, 128, {frames}], "
                f"got {tuple(mean.shape)} and {tuple(std.shape)}"
            )
        if mean.dtype != torch.float32 or std.dtype != torch.float32:
            raise ValueError("Cached Stable Audio posterior mean/std must be float32")
        if mean.device != std.device:
            raise ValueError("Posterior mean and std must be on the same device")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("Cached posterior contains NaN or Inf")
        if (std < 0).any():
            raise ValueError("Cached posterior std must be non-negative")
        lengths = torch.as_tensor(
            sample["vae_posterior_valid_frames"], device=mean.device, dtype=torch.long
        ).reshape(-1)
        if lengths.numel() != mean.shape[0] or (lengths <= 0).any() or (lengths > frames).any():
            raise ValueError("Cached posterior valid frame lengths are invalid")
        batch_size = int(hparams.get("vae_forward_batch_size", 1))
        if batch_size <= 0:
            raise ValueError("vae_forward_batch_size must be positive")
        use_mean = not bool(hparams.get("vae_sample_posterior", True))
        chunks = []
        with torch.autocast(device_type=mean.device.type, enabled=False):
            for offset in range(0, mean.shape[0], batch_size):
                chunks.append(sample_posterior(
                    mean[offset:offset + batch_size],
                    std[offset:offset + batch_size],
                    use_mean=use_mean,
                ))
        lat = torch.cat(chunks, dim=0).transpose(1, 2).contiguous()
        if not torch.isfinite(lat).all():
            raise ValueError("Sampled Stable Audio latent contains NaN or Inf")
        model_sample = dict(sample)
        model_sample.update({"lat": lat, "lat_lens": lengths})
        return model_sample

    def run_model(self, sample, infer=False, infer_steps=None):
        if not infer:
            if hparams.get("vae_online_encode", False):
                sample = self._encode_audio_sample(sample)
            else:
                sample = self._sample_cached_posterior(sample)
        return super().run_model(sample, infer=infer, infer_steps=infer_steps)
