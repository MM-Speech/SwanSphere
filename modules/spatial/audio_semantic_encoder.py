"""Frozen sequence-level audio teachers for DiTAR h supervision.

Both supported teachers encode the complete W-channel clip once and return a
time sequence.  The shared wrapper then overlap-pools that sequence onto the
DiTAR latent-patch timeline.  In particular, no short waveform patch is ever
repeated or padded to a 10-second CLAP input.
"""

import importlib
import importlib.util
import os
import sys
from contextlib import nullcontext

import torch
from torch import nn


SUPPORTED_AUDIO_SEMANTIC_ENCODERS = ("m2d_clap", "atst_frame")


def normalize_audio_semantic_encoder(name: str) -> str:
    """Return the canonical encoder name used by configs and checkpoints."""
    name = str(name).strip().lower().replace("-", "_")
    aliases = {
        "m2d": "m2d_clap",
        "m2dclap": "m2d_clap",
        "atst": "atst_frame",
        "atstframe": "atst_frame",
    }
    name = aliases.get(name, name)
    if name not in SUPPORTED_AUDIO_SEMANTIC_ENCODERS:
        raise ValueError(
            f"Unsupported audio_semantic_encoder={name!r}; expected one of "
            f"{SUPPORTED_AUDIO_SEMANTIC_ENCODERS}"
        )
    return name


def default_audio_semantic_embedding_dim(name: str) -> int:
    """Default output dimension for the supported official base checkpoints."""
    name = normalize_audio_semantic_encoder(name)
    if name == "m2d_clap":
        # Official M2D-CLAP p16x16 convention: five frequency patches are
        # concatenated, rather than averaged: 5 * 768 = 3840.
        return 3840
    # ATST-Frame base, using the last block only.
    return 768


def _resolve_python_file(code_path: str, relative_path: str) -> str:
    code_path = os.path.abspath(os.path.expanduser(str(code_path)))
    path = code_path if os.path.isfile(code_path) else os.path.join(
        code_path, relative_path
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Could not find encoder runtime file {relative_path!r} under "
            f"audio_semantic_code_path={code_path!r}"
        )
    return path


def _load_python_file(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    # Some decorators inspect sys.modules while the module is executing.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


class FrozenSequenceAudioEncoder(nn.Module):
    """Base class that aligns a frozen teacher sequence to latent patches."""

    _DTYPES = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }

    def __init__(
        self,
        input_sample_rate: int,
        forward_batch_size: int,
        model_dtype: str,
    ):
        super().__init__()
        if input_sample_rate <= 0:
            raise ValueError(
                f"audio_semantic_source_sample_rate must be positive, got "
                f"{input_sample_rate}"
            )
        if forward_batch_size <= 0:
            raise ValueError(
                f"audio_semantic_forward_batch_size must be positive, got "
                f"{forward_batch_size}"
            )
        dtype_key = str(model_dtype).lower()
        if dtype_key not in self._DTYPES:
            raise ValueError(
                f"Unsupported audio_semantic_model_dtype={model_dtype!r}; "
                f"expected one of {sorted(self._DTYPES)}"
            )
        self.input_sample_rate = int(input_sample_rate)
        self.forward_batch_size = int(forward_batch_size)
        self.teacher_dtype = self._DTYPES[dtype_key]
        self.embedding_dim = -1
        self.model_sample_rate = -1

    def train(self, mode: bool = True):
        # A parent task's train() call must never enable teacher dropout.
        super().train(False)
        self._set_encoder_eval()
        return self

    def _set_encoder_eval(self):
        raise NotImplementedError

    def _autocast_context(self, device: torch.device):
        enabled = device.type == "cuda" and self.teacher_dtype in {
            torch.bfloat16,
            torch.float16,
        }
        if not enabled:
            return nullcontext()
        return torch.autocast(
            device_type="cuda",
            dtype=self.teacher_dtype,
            enabled=True,
        )

    def _resample(self, wav: torch.Tensor) -> torch.Tensor:
        if self.model_sample_rate == self.input_sample_rate:
            return wav
        try:
            import torchaudio.functional as audio_functional
        except ImportError as exc:
            raise ImportError(
                "Audio semantic supervision requires torchaudio for the "
                f"{self.input_sample_rate} -> {self.model_sample_rate} Hz "
                "W-channel resampling."
            ) from exc
        return audio_functional.resample(
            wav,
            orig_freq=self.input_sample_rate,
            new_freq=self.model_sample_rate,
        )

    def encode_sequence(self, wav_w: torch.Tensor) -> torch.Tensor:
        """Encode a complete clip and return ``[B, T_teacher, C_teacher]``."""
        raise NotImplementedError

    @staticmethod
    def _pool_sequence_to_latent_patches(
        sequence: torch.Tensor,
        source_num_samples: int,
        source_sample_rate: int,
        latent_lens: torch.Tensor,
        num_patches: int,
        patch_size: int,
        latent_frame_rate: float,
    ) -> torch.Tensor:
        """Overlap-pool teacher time bins onto exact latent-patch intervals."""
        if sequence.ndim != 3:
            raise ValueError(
                f"Expected teacher sequence [B, T, C], got {tuple(sequence.shape)}"
            )
        batch_size, teacher_steps, embedding_dim = sequence.shape
        if teacher_steps <= 0:
            raise ValueError("Audio teacher returned an empty time sequence")

        encoded_duration = float(source_num_samples) / float(source_sample_rate)
        if encoded_duration <= 0:
            raise ValueError(
                f"Invalid encoded waveform duration {encoded_duration} seconds"
            )

        output = sequence.new_zeros((batch_size, num_patches, embedding_dim))
        # M2D's official timestamp helper uses clip_duration / sequence_length.
        # Using the same convention also gives ATST a common, encoder-agnostic
        # alignment path.
        source_edges = torch.linspace(
            0.0,
            encoded_duration,
            teacher_steps + 1,
            device=sequence.device,
            dtype=torch.float32,
        )
        source_starts = source_edges[:-1]
        source_ends = source_edges[1:]

        latent_lens_list = (
            latent_lens.detach().to(device="cpu", dtype=torch.long).reshape(-1).tolist()
        )
        if len(latent_lens_list) != batch_size:
            raise ValueError(
                f"Expected {batch_size} latent lengths, got "
                f"{len(latent_lens_list)}"
            )

        for batch_idx, latent_len in enumerate(latent_lens_list):
            latent_len = int(latent_len)
            if latent_len <= 0:
                raise ValueError(
                    f"Invalid latent length at batch {batch_idx}: {latent_len}"
                )
            valid_patches = (latent_len + patch_size - 1) // patch_size
            if valid_patches > num_patches:
                raise ValueError(
                    f"Item {batch_idx} requires {valid_patches} patches, but "
                    f"num_patches={num_patches}"
                )

            patch_ids = torch.arange(
                valid_patches,
                device=sequence.device,
                dtype=torch.float32,
            )
            patch_starts = patch_ids * (patch_size / latent_frame_rate)
            patch_end_frames = torch.minimum(
                (patch_ids + 1.0) * patch_size,
                patch_ids.new_full((valid_patches,), float(latent_len)),
            )
            patch_ends = patch_end_frames / latent_frame_rate

            overlap = (
                torch.minimum(patch_ends[:, None], source_ends[None, :])
                - torch.maximum(patch_starts[:, None], source_starts[None, :])
            ).clamp_min(0.0)
            weight_sum = overlap.sum(dim=-1, keepdim=True)
            if torch.any(weight_sum <= 0):
                raise ValueError(
                    "Audio/latent timelines do not overlap for every valid "
                    f"patch at batch {batch_idx}: encoded_duration="
                    f"{encoded_duration:.6f}s, latent_duration="
                    f"{latent_len / latent_frame_rate:.6f}s"
                )
            weights = overlap / weight_sum
            output[batch_idx, :valid_patches] = (
                weights.to(sequence.dtype) @ sequence[batch_idx]
            )
        return output

    @torch.no_grad()
    def encode_patches(
        self,
        wav_w: torch.Tensor,
        wav_lens: torch.Tensor,
        latent_lens: torch.Tensor,
        num_patches: int,
        patch_size: int,
        latent_frame_rate: float,
    ) -> torch.Tensor:
        """Return one frozen semantic target for every valid DiTAR patch."""
        if wav_w.ndim != 2:
            raise ValueError(
                f"Expected W-channel waveform [B, samples], got {tuple(wav_w.shape)}"
            )
        if num_patches <= 0 or patch_size <= 0 or latent_frame_rate <= 0:
            raise ValueError(
                "num_patches, patch_size, and latent_frame_rate must be positive"
            )
        wav_lens = wav_lens.reshape(-1)
        if wav_lens.numel() != wav_w.shape[0]:
            raise ValueError(
                f"Expected {wav_w.shape[0]} waveform lengths, got "
                f"{wav_lens.numel()}"
            )
        if torch.any(wav_lens <= 0) or torch.any(wav_lens > wav_w.shape[1]):
            raise ValueError(
                f"Invalid waveform lengths for padded size {wav_w.shape[1]}"
            )

        sequence = self.encode_sequence(wav_w)
        if sequence.shape[0] != wav_w.shape[0]:
            raise ValueError(
                f"Teacher changed batch size from {wav_w.shape[0]} to "
                f"{sequence.shape[0]}"
            )
        if sequence.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Teacher returned dim={sequence.shape[-1]}, expected "
                f"embedding_dim={self.embedding_dim}"
            )
        return self._pool_sequence_to_latent_patches(
            sequence=sequence.float(),
            source_num_samples=wav_w.shape[1],
            source_sample_rate=self.input_sample_rate,
            latent_lens=latent_lens,
            num_patches=num_patches,
            patch_size=patch_size,
            latent_frame_rate=latent_frame_rate,
        )


class FrozenM2DCLAPEncoder(FrozenSequenceAudioEncoder):
    """Official M2D-CLAP portable runtime with 3840-D frame features."""

    def __init__(
        self,
        model_path: str,
        code_path: str,
        input_sample_rate: int = 48000,
        forward_batch_size: int = 8,
        model_dtype: str = "bfloat16",
    ):
        super().__init__(input_sample_rate, forward_batch_size, model_dtype)
        model_path = os.path.abspath(os.path.expanduser(str(model_path)))
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"M2D-CLAP checkpoint file does not exist: {model_path}"
            )
        portable_path = _resolve_python_file(
            code_path, os.path.join("examples", "portable_m2d.py")
        )
        try:
            portable_m2d = _load_python_file(
                "_swansphere_portable_m2d", portable_path
            )
        except ImportError as exc:
            raise ImportError(
                "Failed to import M2D PortableM2D. Install its minimal "
                "dependencies in spat_edit with: pip install timm einops "
                "nnAudio transformers"
            ) from exc

        self.encoder = portable_m2d.PortableM2D(
            model_path,
            flat_features=False,
        )
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.model_sample_rate = int(self.encoder.cfg.sample_rate)
        self.embedding_dim = int(self.encoder.cfg.feature_d)
        if self.embedding_dim != 3840:
            raise ValueError(
                "The selected M2D checkpoint/runtime does not expose the "
                "requested official p16x16 concatenated feature dimension: "
                f"got {self.embedding_dim}, expected 3840. Download "
                "m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025."
            )

    def _set_encoder_eval(self):
        if hasattr(self, "encoder"):
            self.encoder.eval()

    @torch.no_grad()
    def encode_sequence(self, wav_w: torch.Tensor) -> torch.Tensor:
        device = next(self.encoder.parameters()).device
        wav = self._resample(wav_w.detach().to(device=device, dtype=torch.float32))
        outputs = []
        self.encoder.eval()
        for start in range(0, wav.shape[0], self.forward_batch_size):
            chunk = wav[start : start + self.forward_batch_size]
            with self._autocast_context(device):
                embedding = self.encoder(chunk)
            if embedding.ndim != 3:
                raise ValueError(
                    "PortableM2D must return [B, T, C], got "
                    f"{tuple(embedding.shape)}"
                )
            outputs.append(embedding.detach().float())
        return torch.cat(outputs, dim=0)


class FrozenATSTFrameEncoder(FrozenSequenceAudioEncoder):
    """Official ATST-Frame timestamp embeddings from one or more blocks."""

    def __init__(
        self,
        model_path: str,
        code_path: str,
        input_sample_rate: int = 48000,
        forward_batch_size: int = 8,
        model_dtype: str = "bfloat16",
        num_blocks: int = 1,
    ):
        super().__init__(input_sample_rate, forward_batch_size, model_dtype)
        if num_blocks <= 0 or num_blocks > 12:
            raise ValueError(f"atst_frame_num_blocks must be in [1, 12], got {num_blocks}")
        model_path = os.path.abspath(os.path.expanduser(str(model_path)))
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"ATST-Frame checkpoint file does not exist: {model_path}"
            )
        code_path = os.path.abspath(os.path.expanduser(str(code_path)))
        package_dir = os.path.join(code_path, "audiossl")
        if not os.path.isdir(package_dir):
            raise FileNotFoundError(
                f"Could not find the audiossl package under {code_path}"
            )
        if code_path not in sys.path:
            sys.path.insert(0, code_path)
        try:
            embedding_module = importlib.import_module(
                "audiossl.methods.atstframe.embedding"
            )
        except ImportError as exc:
            raise ImportError(
                "Failed to import the official ATST-Frame runtime from "
                f"{code_path}. Install the audiossl requirements first."
            ) from exc

        embedding_module.N_BLOCKS = int(num_blocks)
        self._embedding_module = embedding_module
        self.num_blocks = int(num_blocks)
        self.encoder = embedding_module.load_model(model_path)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.model_sample_rate = int(getattr(self.encoder, "sample_rate", 16000))
        self.embedding_dim = int(self.encoder.embed_dim) * self.num_blocks

    def _set_encoder_eval(self):
        if hasattr(self, "encoder"):
            self.encoder.eval()

    @torch.no_grad()
    def encode_sequence(self, wav_w: torch.Tensor) -> torch.Tensor:
        device = next(self.encoder.parameters()).device
        wav = self._resample(wav_w.detach().to(device=device, dtype=torch.float32))
        outputs = []
        self.encoder.eval()
        for start in range(0, wav.shape[0], self.forward_batch_size):
            chunk = wav[start : start + self.forward_batch_size]
            with self._autocast_context(device):
                embedding, _ = self._embedding_module.get_timestamp_embedding(
                    chunk,
                    self.encoder,
                )
            if embedding.ndim == 4:
                embedding = embedding.flatten(start_dim=-2)
            elif embedding.ndim != 3:
                raise ValueError(
                    "ATST-Frame must return [B, T, C] or [B, T, L, C], got "
                    f"{tuple(embedding.shape)}"
                )
            outputs.append(embedding.detach().float())
        return torch.cat(outputs, dim=0)


def build_frozen_audio_semantic_encoder(
    encoder_name: str,
    model_path: str,
    code_path: str,
    input_sample_rate: int = 48000,
    forward_batch_size: int = 8,
    model_dtype: str = "bfloat16",
    atst_frame_num_blocks: int = 1,
) -> FrozenSequenceAudioEncoder:
    """Factory controlled by the ``audio_semantic_encoder`` hparam."""
    encoder_name = normalize_audio_semantic_encoder(encoder_name)
    common = dict(
        model_path=model_path,
        code_path=code_path,
        input_sample_rate=input_sample_rate,
        forward_batch_size=forward_batch_size,
        model_dtype=model_dtype,
    )
    if encoder_name == "m2d_clap":
        return FrozenM2DCLAPEncoder(**common)
    return FrozenATSTFrameEncoder(
        **common,
        num_blocks=atst_frame_num_blocks,
    )
