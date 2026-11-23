import collections
import collections.abc
import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW

# attrdict still imports the Python <3.10 collection aliases.  Install the
# compatibility aliases before importing the shared spatial task stack.
for type_name in collections.abc.__all__:
    if not hasattr(collections, type_name):
        setattr(collections, type_name, getattr(collections.abc, type_name))

from modules.spatial.build_model_utils import DiTARBuildModelMixinS
from tasks.spatial.dittask import SpatialDiTTask
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams
from utils.commons.io import print_once
from utils.nn.model_utils import unwrap_model


class SpatialDiTARBaseTask(DiTARBuildModelMixinS, SpatialDiTTask):
    """Shared spatial DiT-AR training and loss implementation.

    Each ten-second example has 215 valid latent frames. The task pads targets
    to a complete patch while preserving their valid lengths for attention and
    diffusion-loss masking.
    """

    model_version = 8
    expected_latent_dim = 128

    def build_model(self):
        latent_dim = int(hparams["latent_dim"])
        if latent_dim != self.expected_latent_dim:
            raise ValueError(
                f"Spatial DiT-AR v{self.model_version} requires "
                f"latent_dim={self.expected_latent_dim}, got {latent_dim}"
            )
        if int(hparams.get("d_lat", latent_dim)) != latent_dim:
            raise ValueError(
                "d_lat and latent_dim must describe the same VAE latent space"
            )
        if hparams.get("use_ema", False):
            raise NotImplementedError(f"EMA is not implemented for Spatial DiT-AR v{self.model_version}")

        print(
            f"| Building Spatial DiT-AR v{self.model_version} "
            f"(latent_dim={latent_dim}, patch_size="
            f"{hparams.get('ditar_patch_size', 4)})"
        )
        build_result = super().build_model()

        if hparams.get("use_audio_semantic_supervision", False):
            semantic_loss_weight = float(
                hparams.get("audio_semantic_loss_weight", 1.0)
            )
            if semantic_loss_weight < 0:
                raise ValueError(
                    "audio_semantic_loss_weight must be non-negative, got "
                    f"{semantic_loss_weight}"
                )

            from modules.spatial.audio_semantic_encoder import (
                build_frozen_audio_semantic_encoder,
                default_audio_semantic_embedding_dim,
                normalize_audio_semantic_encoder,
            )

            encoder_name = normalize_audio_semantic_encoder(
                hparams.get("audio_semantic_encoder", "m2d_clap")
            )
            model_path = str(
                hparams.get("audio_semantic_model_path", "")
            ).strip()
            code_path = str(
                hparams.get("audio_semantic_code_path", "")
            ).strip()
            if not model_path:
                raise ValueError(
                    "use_audio_semantic_supervision=true requires a local "
                    "audio_semantic_model_path"
                )
            if not code_path:
                raise ValueError(
                    "use_audio_semantic_supervision=true requires a local "
                    "audio_semantic_code_path"
                )
            model_path = os.path.abspath(os.path.expanduser(model_path))
            code_path = os.path.abspath(os.path.expanduser(code_path))
            self.audio_semantic_teacher = build_frozen_audio_semantic_encoder(
                encoder_name=encoder_name,
                model_path=model_path,
                code_path=code_path,
                input_sample_rate=int(
                    hparams.get("audio_semantic_source_sample_rate", 48000)
                ),
                forward_batch_size=int(
                    hparams.get("audio_semantic_forward_batch_size", 8)
                ),
                model_dtype=hparams.get(
                    "audio_semantic_model_dtype", "bfloat16"
                ),
                atst_frame_num_blocks=int(
                    hparams.get("atst_frame_num_blocks", 1)
                ),
            )
            expected_dim = int(
                hparams.get(
                    "audio_semantic_embedding_dim",
                    default_audio_semantic_embedding_dim(encoder_name),
                )
            )
            if self.audio_semantic_teacher.embedding_dim != expected_dim:
                raise ValueError(
                    "Audio teacher output dimension does not match "
                    "audio_semantic_embedding_dim: teacher="
                    f"{self.audio_semantic_teacher.embedding_dim}, "
                    f"config={expected_dim}"
                )

            print_once(
                f"| Frozen {encoder_name} W-channel sequence teacher: "
                f"{model_path} (dim={expected_dim}, "
                f"loss_weight={semantic_loss_weight})"
            )
            build_result["others"].append(self.audio_semantic_teacher)

        return build_result

    def load_model(self):
        if hparams.get("load_ckpt", ""):
            load_ckpt(
                self.model,
                hparams["load_ckpt"],
                "model",
                strict=False,
                mmap=True,
            )

    def build_optimizer(self):
        if not hparams.get("disable_weight_decay_on_bias_and_norm_and_embed", True):
            return AdamW(unwrap_model(self.model).parameters(), **self.config.optimizer)

        decay_params = []
        no_decay_params = []
        no_decay_names = ("bias", "norm", "embed", "speech_start_token")
        for name, param in unwrap_model(self.model).named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() == 1 or any(key in name for key in no_decay_names):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        print_once(f"| Weight decay is disabled for {len(no_decay_params)} params")
        optimizer_groups = [
            {
                "params": decay_params,
                "weight_decay": self.config.optimizer.weight_decay,
            },
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        return AdamW(optimizer_groups, **self.config.optimizer)

    def fsdp_optm2model(self):
        return [self.model]

    def fsdp_wrap_policy(self):
        from modules.commons.hf.transformer import DecoderLayer, EncoderLayer
        from modules.commons.hf.transformer_dit import EncoderLayer as DiTEncoderLayer

        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            return recurse or isinstance(
                module,
                (EncoderLayer, DecoderLayer, DiTEncoderLayer),
            )

        return custom_auto_wrap_policy

    @classmethod
    def _prepare_model_sample(cls, sample):
        lat = sample["lat"]
        if lat.ndim != 3:
            raise ValueError(f"Expected lat [B, T, C], got {tuple(lat.shape)}")
        if lat.shape[-1] != cls.expected_latent_dim:
            raise ValueError(
                f"Expected {cls.expected_latent_dim}-channel VAE latent, "
                f"got {tuple(lat.shape)}"
            )

        patch_size = int(hparams.get("ditar_patch_size", 4))
        if patch_size <= 0:
            raise ValueError(f"ditar_patch_size must be positive, got {patch_size}")
        pad_frames = (-lat.shape[1]) % patch_size
        if pad_frames:
            lat = F.pad(lat, (0, 0, 0, pad_frames), value=0.0)

        lat_lens = sample.get("lat_lens")
        if lat_lens is None:
            lat_lens = torch.full(
                (lat.shape[0],),
                lat.shape[1] - pad_frames,
                dtype=torch.long,
                device=lat.device,
            )
        elif not isinstance(lat_lens, torch.Tensor):
            lat_lens = torch.as_tensor(lat_lens, dtype=torch.long, device=lat.device)
        else:
            lat_lens = lat_lens.to(device=lat.device, dtype=torch.long)
        lat_lens = lat_lens.reshape(-1)
        if lat_lens.numel() != lat.shape[0]:
            raise ValueError(
                f"Expected {lat.shape[0]} latent lengths, got {lat_lens.numel()}"
            )
        if torch.any(lat_lens <= 0) or torch.any(lat_lens > lat.shape[1]):
            raise ValueError(
                f"Invalid latent lengths {lat_lens.tolist()} for T={lat.shape[1]}"
            )

        video_feats = sample["inputs_embeds"]
        if video_feats.ndim != 3:
            raise ValueError(
                f"Expected inputs_embeds [B, T, C], got {tuple(video_feats.shape)}"
            )
        v_mask = sample.get("v_mask")
        if v_mask is None:
            v_mask = torch.ones(
                video_feats.shape[:2], dtype=torch.long, device=video_feats.device
            )

        model_sample = dict(sample)
        model_sample.update(
            {
                "lat": lat,
                "lat_lens": lat_lens,
                "video_feats": video_feats,
                "v_mask": v_mask,
            }
        )
        return model_sample

    def run_model(self, sample, infer=False, infer_steps=None):
        del infer_steps
        if infer:
            return {}, {}

        sample = self._prepare_model_sample(sample)
        inputs = {
            "lat": sample["lat"],
            "lat_lens": sample["lat_lens"],
            "video_feats": sample["video_feats"],
            "v_mask": sample["v_mask"],
            "global_clip_embedding": sample["global_clip_embedding"],
        }

        if hparams.get("use_audio_semantic_supervision", False):
            if (
                "audio_semantic_wav" not in sample
                or "audio_semantic_wav_lens" not in sample
            ):
                raise KeyError(
                    "Audio semantic supervision requires audio_semantic_wav "
                    "and audio_semantic_wav_lens from the spatial dataset"
                )
            patch_size = int(hparams.get("ditar_patch_size", 4))
            if sample["lat"].shape[1] % patch_size != 0:
                raise ValueError(
                    f"Padded latent length {sample['lat'].shape[1]} is not "
                    f"divisible by patch_size={patch_size}"
                )
            inputs["audio_semantic_embeddings"] = (
                self.audio_semantic_teacher.encode_patches(
                    wav_w=sample["audio_semantic_wav"],
                    wav_lens=sample["audio_semantic_wav_lens"],
                    latent_lens=sample["lat_lens"],
                    num_patches=sample["lat"].shape[1] // patch_size,
                    patch_size=patch_size,
                    latent_frame_rate=float(
                        hparams.get("latent_frame_rate", 21.5)
                    ),
                )
            )

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
            outputs = self.model(inputs)

        loss_mask = outputs["loss_mask"]
        losses_out = {
            "diff_loss": outputs["diff_loss"],
            "stop_loss": outputs["stop_loss"],
            "bs": sample["lat"].shape[0],
            "ntokens": sample["lat_lens"].sum(),
        }
        if hparams.get("use_audio_semantic_supervision", False):
            losses_out["audio_semantic_loss"] = (
                outputs["audio_semantic_loss"]
                * float(hparams.get("audio_semantic_loss_weight", 1.0))
            )

        with torch.no_grad():
            denom = loss_mask.sum().clamp_min(1)
            losses_out["monitor/orig_loss"] = outputs["diff_loss"].detach()
            losses_out["monitor/pred"] = (
                outputs["pred"] * loss_mask
            ).sum() / denom
            losses_out["monitor/tgt"] = (
                outputs["target"] * loss_mask
            ).sum() / denom
            losses_out["monitor/loss_mask"] = loss_mask.sum()
            losses_out["monitor/ctx_mask"] = outputs["ctx_mask"].sum()
            # Verify that the bottlenecked DiT latent-context history follows
            # the requested dropout/corruption distribution during training.
            for metric_name in (
                "dit_history_available_fraction",
                "dit_history_drop_fraction",
                "dit_ctx_corrupt_fraction",
                "dit_ctx_noise_ratio",
                "dit_ctx_noise_ratio_max",
                "dit_ctx_noise_delta_rms",
            ):
                if metric_name in outputs:
                    losses_out[f"monitor/{metric_name}"] = outputs[
                        metric_name
                    ].detach()
            losses_out["monitor/stop_labels_sum"] = outputs["stop_labels"].sum()
            if hparams.get("use_audio_semantic_supervision", False):
                losses_out["monitor/audio_semantic_loss_raw"] = outputs[
                    "audio_semantic_loss"
                ].detach()
                losses_out["monitor/audio_semantic_cosine"] = outputs[
                    "audio_semantic_cosine"
                ]
                losses_out["monitor/audio_semantic_valid_patches"] = outputs[
                    "audio_semantic_valid_patches"
                ]

        if loss_mask.sum() <= 3:
            if self.trainer.proc_rank_local == 0:
                print(
                    f"| CRITICAL: loss_mask.sum()={loss_mask.sum()}. "
                    "Resetting diffusion loss."
                )
            losses_out["diff_loss"] = outputs["diff_loss"] * 0.0

        return losses_out, {}
