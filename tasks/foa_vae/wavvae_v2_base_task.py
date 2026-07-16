from attrdict import AttrDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_audio_tools.models.discriminators import EncodecDiscriminator
from stable_audio_tools.training.losses import auraloss

from modules.foa_vae.wavvae_v2 import (
    active_intensity_direction_loss,
    adapter_param_prefixes,
    build_foa_wavvae,
    build_projection_directions,
    build_stereo_projection_pairs,
    latent_adapter_param_prefixes,
    normalized_cross_spectrum_loss,
    partial_unfreeze_prefixes,
    project_foa_stereo_wyzx,
    project_foa_wyzx,
    scheduled_value,
    set_trainable_by_prefixes,
    spatial_covariance_loss,
    split_adapter_backbone_params,
    trainable_param_report,
)
from utils.commons.base_task import BaseTask
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams
from utils.commons.import_utils import import_module_bystr
from utils.nn.model_utils import unwrap_model
from utils.nn.schedulers import CosineAnnealingWarmRestartsWithWarmup
from tasks.tts.dataset_utils.dataset_mixin import FastDatasetMixin


class PhaseShiftedCosineWarmRestartsWithWarmup:
    def __init__(
        self,
        optimizer,
        lr_max,
        warmup_updates,
        total_updates,
        initial_period,
        period_mult=1.0,
        lr_min=1e-5,
        start_phase="max",
    ):
        self.optimizer = optimizer
        self.lr_max = float(lr_max)
        self.lr_min = float(lr_min)
        self.warmup_updates = int(warmup_updates)
        self.total_updates = int(total_updates)
        self.initial_period = int(initial_period)
        self.period_mult = float(period_mult)
        self.start_phase = str(start_phase or "max").lower()

        self.cycle_start = self.warmup_updates
        self.current_period = self.initial_period
        self.cycle_end = self.cycle_start + self.current_period
        self.cycle_count = 0
        self.assign_learning_rate(self.optimizer, self._calculate_lr(0))

    def assign_learning_rate(self, optimizer, new_lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

    def _cycle_position(self, num_updates):
        while num_updates >= self.cycle_end and num_updates < self.total_updates:
            self.cycle_count += 1
            self.cycle_start = self.cycle_end
            self.current_period = max(1, int(self.current_period * self.period_mult))
            self.cycle_end = min(self.cycle_start + self.current_period, self.total_updates)
        position = (num_updates - self.cycle_start) / max(1, self.cycle_end - self.cycle_start)
        return min(max(position, 0.0), 1.0)

    def _calculate_lr(self, num_updates):
        if num_updates < self.warmup_updates:
            ratio = num_updates / max(1, self.warmup_updates)
            return self.lr_min + (self.lr_max - self.lr_min) * ratio
        if num_updates >= self.total_updates:
            return self.lr_min

        position = self._cycle_position(num_updates)
        if self.start_phase in {"min", "min_to_max_to_min"}:
            cosine = 0.5 * (1.0 - math.cos(2.0 * math.pi * position))
        elif self.start_phase == "min_to_max":
            cosine = 0.5 * (1.0 - math.cos(math.pi * position))
        else:
            cosine = 0.5 * (1.0 + math.cos(math.pi * position))
        return self.lr_min + (self.lr_max - self.lr_min) * cosine

    def step(self, num_updates):
        lr = self._calculate_lr(num_updates)
        self.assign_learning_rate(self.optimizer, lr)
        return lr


def trim_to_shortest(a, b):
    if a.shape[-1] > b.shape[-1]:
        return a[:, :, : b.shape[-1]], b
    if b.shape[-1] > a.shape[-1]:
        return a, b[:, :, : a.shape[-1]]
    return a, b


class FOAWavVAEBaseTask(FastDatasetMixin, BaseTask):
    stage_name = "base"

    def __init__(self):
        super().__init__()
        if hparams.get("dataloader_version", "v1") == "v1":
            self.dataset_cls = import_module_bystr(hparams["dataset_cls"])
            self.val_dataset_cls = import_module_bystr(hparams["val_dataset_cls"])
        elif hparams.get("dataloader_version", "v1") == "v2":
            self.dataset_cls = import_module_bystr(hparams["dataset_cls"])
            self.train_dataloader = FastDatasetMixin.train_dataloader.__get__(self)
            self.test_dataloader = FastDatasetMixin.test_dataloader.__get__(self)
            self.val_dataloader = FastDatasetMixin.val_dataloader.__get__(self)
        self.hparams = hparams
        self.config = AttrDict(hparams)
        self.sample_rate = hparams["sample_rate"]
        self.latest_recon = None

    @property
    def stage_cfg(self):
        return hparams.get("foa_stage", {})

    @property
    def loss_cfg(self):
        return hparams.get("foa_losses", {})

    @property
    def lr_scheduler_cfg(self):
        return hparams.get("foa_lr_scheduler", {})

    @property
    def disc_lr_scheduler_cfg(self):
        return hparams.get("foa_disc_lr_scheduler", {})

    def build_model(self):
        init_pretrained = (
            self.stage_name in {"projector_warmup", "native_adapter_warmup"}
            and not hparams.get("from_scratch", False)
            and not hparams.get("load_ckpt", "")
            and not hparams.get("resume_from", "")
        )
        self.model_gen = build_foa_wavvae(hparams=hparams, init_pretrained=init_pretrained)
        self._apply_build_time_trainability()
        trainable_param_report(self.model_gen)

        self.model_disc = torch.nn.ModuleDict()
        disc_args = hparams["loss_configs"]["discriminator"]["config"]
        self.model_disc["projection_discriminator"] = EncodecDiscriminator(in_channels=1, **disc_args)
        if hparams.get("foa_stereo_projection", {}).get("enabled", False):
            self.model_disc["stereo_projection_discriminator"] = EncodecDiscriminator(in_channels=2, **disc_args)

        stft_args = hparams["loss_configs"]["spectral"]["config"]
        self.channel_mrstft = auraloss.MultiResolutionSTFTLoss(sample_rate=self.sample_rate, **stft_args)
        self.projection_mrstft = auraloss.MultiResolutionSTFTLoss(sample_rate=self.sample_rate, **stft_args)
        return {"trainable": [self.model_gen, self.model_disc], "others": []}

    def _apply_build_time_trainability(self):
        trainable_mode = self.stage_cfg.get("trainable", "full")
        if trainable_mode == "adapter":
            set_trainable_by_prefixes(self.model_gen, adapter_param_prefixes(self.model_gen))
        elif trainable_mode == "io_then_adapter":
            set_trainable_by_prefixes(self.model_gen, adapter_param_prefixes(self.model_gen))

    def load_model(self):
        ckpt = hparams.get("load_ckpt", "")
        if ckpt:
            load_ckpt(self.model_gen, ckpt, "model_gen", strict=hparams.get("load_ckpt_strict", False))
            disc_ckpt = hparams.get("load_ckpt_disc", "")
            if not disc_ckpt and self.stage_cfg.get("use_g_adv", False):
                disc_ckpt = ckpt
            if disc_ckpt:
                load_ckpt(self.model_disc, disc_ckpt, "model_disc", strict=False)

    def fsdp_optm2model(self):
        return [self.model_gen]

    def build_optimizer(self):
        gen_groups = split_adapter_backbone_params(self.model_gen)
        for group in gen_groups:
            group["lr"] = hparams.get("lr", 5.0e-5)
        optimizer_gen = torch.optim.AdamW(
            gen_groups,
            lr=hparams.get("lr", 5.0e-5),
            betas=[hparams["adam_b1"], hparams["adam_b2"]],
            weight_decay=hparams.get("weight_decay", 0.0),
        )
        optimizer_disc = torch.optim.AdamW(
            self.model_disc.parameters(),
            lr=hparams.get("disc_lr", hparams.get("lr", 5.0e-5)),
            betas=[hparams["adam_b1"], hparams["adam_b2"]],
            weight_decay=hparams.get("disc_weight_decay", 0.0),
        )
        return [optimizer_gen, optimizer_disc]

    def _build_cosine_scheduler(self, opt, scheduler_cfg):
        if scheduler_cfg.get("enabled", False):
            accumulate = max(1, int(hparams.get("accumulate_grad_batches", 1)))
            total_updates = int(
                scheduler_cfg.get(
                    "total_updates",
                    (int(hparams["max_updates"]) + accumulate - 1) // accumulate,
                )
            )
            return PhaseShiftedCosineWarmRestartsWithWarmup(
                opt,
                lr_max=float(scheduler_cfg["lr_max"]),
                lr_min=float(scheduler_cfg["lr_min"]),
                warmup_updates=int(scheduler_cfg.get("warmup_updates", 0)),
                total_updates=total_updates,
                initial_period=int(scheduler_cfg["initial_period"]),
                period_mult=float(scheduler_cfg.get("period_mult", 1.0)),
                start_phase=scheduler_cfg.get("start_phase", "max"),
            )
        return None

    def build_scheduler(self, optimizer):
        generator_scheduler = self._build_cosine_scheduler(optimizer[0], self.lr_scheduler_cfg)
        discriminator_scheduler = self._build_cosine_scheduler(optimizer[1], self.disc_lr_scheduler_cfg)
        return (generator_scheduler, discriminator_scheduler)

    def _loss_weight(self, name):
        return scheduled_value(self.loss_cfg.get(name, 0.0), self.global_step)

    def _set_optimizer_lrs(self, optimizer_idx):
        if not hasattr(self, "trainer") or not self.trainer.optimizers:
            return
        if optimizer_idx == 0 and self.lr_scheduler_cfg.get("enabled", False):
            return
        if optimizer_idx == 0:
            lr_cfg = self.stage_cfg.get("lr", {})
            adapter_lr = float(lr_cfg.get("adapter", hparams.get("lr", 5.0e-5)))
            backbone_lr = float(lr_cfg.get("backbone", hparams.get("lr", 5.0e-5)))
            for group in self.trainer.optimizers[0].param_groups:
                if group.get("name") == "adapter":
                    group["lr"] = adapter_lr
                elif group.get("name") == "backbone":
                    group["lr"] = backbone_lr
        else:
            if self.disc_lr_scheduler_cfg.get("enabled", False):
                return
            disc_lr = float(self.stage_cfg.get("lr", {}).get("disc", hparams.get("disc_lr", hparams.get("lr", 5.0e-5))))
            for group in self.trainer.optimizers[1].param_groups:
                group["lr"] = disc_lr

    def _lr_monitor_outputs(self, optimizer_idx):
        if not hasattr(self, "trainer") or optimizer_idx >= len(self.trainer.optimizers):
            return {}
        optimizer = self.trainer.optimizers[optimizer_idx]
        if optimizer_idx == 0:
            return {
                f"monitor/lr_{group.get('name', f'group{group_idx}')}": float(group["lr"])
                for group_idx, group in enumerate(optimizer.param_groups)
            }
        return {"monitor/lr_disc": float(optimizer.param_groups[0]["lr"])}

    def _apply_trainability(self, optimizer_idx):
        if optimizer_idx == 1:
            return
        mode = self.stage_cfg.get("trainable", "full")
        if mode == "adapter":
            set_trainable_by_prefixes(self.model_gen, adapter_param_prefixes(self.model_gen))
        elif mode == "io_then_adapter":
            set_trainable_by_prefixes(self.model_gen, adapter_param_prefixes(self.model_gen))
        elif mode == "partial_then_full":
            full_step = int(self.stage_cfg.get("full_unfreeze_step", 30000))
            if self.global_step < full_step:
                prefixes = partial_unfreeze_prefixes(
                    self.model_gen,
                    encoder_last_n=int(self.stage_cfg.get("encoder_last_blocks", 2)),
                    decoder_last_n=int(self.stage_cfg.get("decoder_last_blocks", 2)),
                )
                set_trainable_by_prefixes(self.model_gen, prefixes)
            else:
                set_trainable_by_prefixes(self.model_gen, [], train_all=True)
        elif mode == "full":
            set_trainable_by_prefixes(self.model_gen, [], train_all=True)
        else:
            raise ValueError(f"Unknown FOA VAE trainable mode: {mode}")

    def _mask_io_then_adapter_latent_grads(self, optimizer_idx):
        if optimizer_idx != 0:
            return
        if self.stage_cfg.get("trainable", "full") != "io_then_adapter":
            return
        if self.global_step >= int(self.stage_cfg.get("io_warmup_steps", 5000)):
            return
        prefixes = tuple(latent_adapter_param_prefixes(self.model_gen))
        model = unwrap_model(self.model_gen)
        for name, param in model.named_parameters():
            if param.grad is not None and name.startswith(prefixes):
                param.grad = None

    def _set_discriminator_trainability(self, optimizer_idx):
        unwrap_model(self.model_disc).requires_grad_(optimizer_idx == 1)

    def _projection_directions(self, device, dtype):
        proj_cfg = hparams.get("foa_projection", {})
        return build_projection_directions(
            random_count=int(proj_cfg.get("random_dirs", 4)),
            include_axes=bool(proj_cfg.get("include_axes", True)),
            include_corners=bool(proj_cfg.get("include_corners", True)),
            device=device,
            dtype=dtype,
        )

    def _projection_audio(self, audio, directions):
        gain = float(hparams.get("foa_projection", {}).get("gain", 1.0))
        return project_foa_wyzx(audio, directions, gain=gain, fold_batch=True)

    def _projection_chunk_size(self, num_directions):
        chunk_size = int(hparams.get("foa_projection", {}).get("mrstft_chunk_size", 0))
        if chunk_size <= 0:
            return max(1, int(num_directions))
        return min(chunk_size, max(1, int(num_directions)))

    def _projection_mrstft_loss(self, pred, target, directions):
        if directions.numel() == 0:
            return pred.new_tensor(0.0)
        chunk_size = self._projection_chunk_size(directions.shape[0])
        loss = pred.new_tensor(0.0)
        n_dirs = 0
        for dirs_chunk in directions.split(chunk_size, dim=0):
            pred_proj = self._projection_audio(pred, dirs_chunk)
            target_proj = self._projection_audio(target, dirs_chunk)
            loss = loss + self.projection_mrstft(pred_proj, target_proj) * dirs_chunk.shape[0]
            n_dirs += dirs_chunk.shape[0]
        return loss / max(1, n_dirs)

    def _projection_l1_loss(self, pred, target, directions):
        if directions.numel() == 0:
            return pred.new_tensor(0.0)
        chunk_size = self._projection_chunk_size(directions.shape[0])
        loss = pred.new_tensor(0.0)
        n_dirs = 0
        for dirs_chunk in directions.split(chunk_size, dim=0):
            pred_proj = self._projection_audio(pred, dirs_chunk)
            target_proj = self._projection_audio(target, dirs_chunk)
            loss = loss + F.l1_loss(pred_proj, target_proj) * dirs_chunk.shape[0]
            n_dirs += dirs_chunk.shape[0]
        return loss / max(1, n_dirs)

    def _channel_mrstft_weights(self, pred):
        weights = self.loss_cfg.get("channel_mrstft_weights", None)
        if weights is None:
            weights = [1.0] * pred.shape[1]
        weights = [float(weight) for weight in weights]
        if len(weights) != pred.shape[1]:
            raise ValueError(
                f"channel_mrstft_weights length must match audio channels: "
                f"got {len(weights)} weights for {pred.shape[1]} channels"
            )
        if any(weight < 0 for weight in weights):
            raise ValueError("channel_mrstft_weights must be non-negative")
        if sum(weights) <= 0:
            raise ValueError("channel_mrstft_weights must contain at least one positive value")
        return pred.new_tensor(weights)

    def _per_channel_mrstft_loss(self, pred, target, weights=None):
        if weights is None:
            weights = self._channel_mrstft_weights(pred)
        else:
            weights = pred.new_tensor(weights)
        loss = pred.new_tensor(0.0)
        for channel_idx, channel_weight in enumerate(weights):
            loss = loss + self.channel_mrstft(
                pred[:, channel_idx : channel_idx + 1],
                target[:, channel_idx : channel_idx + 1],
            ) * channel_weight
        return loss / weights.sum().clamp_min(1.0e-8)

    def _stereo_projection_enabled(self):
        return "stereo_projection_discriminator" in unwrap_model(self.model_disc)

    def _stereo_projection_pairs(self, device, dtype):
        stereo_cfg = hparams.get("foa_stereo_projection", {})
        return build_stereo_projection_pairs(
            pairs=stereo_cfg.get("pairs", None),
            device=device,
            dtype=dtype,
        )

    def _stereo_projection_audio(self, audio, pairs):
        gain = float(hparams.get("foa_stereo_projection", {}).get("gain", hparams.get("foa_projection", {}).get("gain", 1.0)))
        return project_foa_stereo_wyzx(audio, pairs, gain=gain, fold_batch=True)

    def _sample_items(self, items, count):
        count = int(count or 0)
        if count <= 0 or count >= items.shape[0]:
            return items
        if items.shape[0] == 0:
            return items
        indices = torch.randperm(items.shape[0], device=items.device)[:count]
        return items.index_select(0, indices)

    def _projection_directions_for(self, device, dtype, count_key):
        dirs = self._projection_directions(device, dtype)
        return self._sample_items(dirs, self.stage_cfg.get(count_key, 0))

    def _stereo_projection_pairs_for(self, device, dtype, count_key):
        pairs = self._stereo_projection_pairs(device, dtype)
        return self._sample_items(pairs, self.stage_cfg.get(count_key, 0))

    def _current_d_update_every(self):
        d_update_every = int(self.stage_cfg.get("d_update_every", 1))
        after_step = self.stage_cfg.get("d_update_every_after_step", None)
        if after_step is not None and self.global_step >= int(after_step):
            d_update_every = int(self.stage_cfg.get("d_update_every_after", d_update_every))
        return max(1, d_update_every)

    def _use_projection_branch(self, mode_key, branch):
        mode = str(self.stage_cfg.get(mode_key, "both")).lower()
        if mode in {"both", "all"}:
            return True
        if mode in {"mono", "projection"}:
            return branch == "mono"
        if mode == "stereo":
            return branch == "stereo"
        if mode == "alternate":
            step = self.global_step
            if mode_key.startswith("disc_"):
                step = self.global_step // self._current_d_update_every()
            return (step % 2 == 0 and branch == "mono") or (step % 2 == 1 and branch == "stereo")
        raise ValueError(f"Unknown projection branch mode for {mode_key}: {mode}")

    def _disc_discriminator_loss(self, discriminator, reals, fakes):
        if hasattr(discriminator, "discriminator_loss"):
            out = discriminator.discriminator_loss(reals=reals, fakes=fakes, return_scores=True)
            if isinstance(out, tuple):
                loss_dis, real_score, fake_score = out
                return loss_dis, {
                    "real_score": real_score.detach(),
                    "fake_score": fake_score.detach(),
                }
            return out, {}
        loss_dis, _, _ = discriminator.loss(reals=reals, fakes=fakes)
        return loss_dis, {}

    def _disc_generator_loss(self, discriminator, reals, fakes, compute_fm):
        if hasattr(discriminator, "generator_loss"):
            return discriminator.generator_loss(reals=reals, fakes=fakes, compute_fm=compute_fm)
        _, loss_adv, feature_matching = discriminator.loss(reals=reals, fakes=fakes)
        return loss_adv, feature_matching

    def _disc_generator_loss_fp32(self, discriminator, reals, fakes, compute_fm):
        autocast_device = "cuda" if fakes.is_cuda else "cpu"
        with torch.autocast(device_type=autocast_device, enabled=False):
            return self._disc_generator_loss(
                discriminator,
                reals=reals.float(),
                fakes=fakes.float(),
                compute_fm=compute_fm,
            )

    def _spatial_fft_kwargs(self):
        cfg = hparams.get("foa_spatial_loss", {})
        return {
            "fft_size": int(cfg.get("fft_size", 1024)),
            "hop_size": int(cfg.get("hop_size", 256)),
            "win_length": int(cfg.get("win_length", 1024)),
        }

    def _generator_recon_losses(self, pred, target, loss_output):
        dirs = None
        pred_proj = None
        target_proj = None

        lambda_ch = self._loss_weight("lambda_ch_mrstft")
        if lambda_ch:
            loss_output["loss_ch_mrstft"] = self.channel_mrstft(pred, target) * lambda_ch

        lambda_channel = self._loss_weight("lambda_channel_mrstft")
        if lambda_channel:
            loss_output["loss_channel_mrstft"] = self._per_channel_mrstft_loss(pred, target) * lambda_channel

        lambda_proj = self._loss_weight("lambda_proj_mrstft")
        if lambda_proj:
            dirs = self._projection_directions(target.device, target.dtype)
            loss_output["loss_proj_mrstft"] = self._projection_mrstft_loss(pred, target, dirs) * lambda_proj

        lambda_proj_l1 = self._loss_weight("lambda_proj_l1")
        if lambda_proj_l1:
            if dirs is None:
                dirs = self._projection_directions(target.device, target.dtype)
            loss_output["loss_proj_l1"] = self._projection_l1_loss(pred, target, dirs) * lambda_proj_l1

        lambda_time_l1 = self._loss_weight("lambda_time_l1")
        if lambda_time_l1:
            loss_output["loss_time_l1"] = F.l1_loss(pred, target) * lambda_time_l1

        spatial_kwargs = self._spatial_fft_kwargs()
        lambda_intensity = self._loss_weight("lambda_intensity_dir")
        if lambda_intensity:
            energy_percentile = float(hparams.get("foa_spatial_loss", {}).get("energy_percentile", 0.0))
            loss_output["loss_intensity_dir"] = (
                active_intensity_direction_loss(
                    pred,
                    target,
                    energy_percentile=energy_percentile,
                    **spatial_kwargs,
                )
                * lambda_intensity
            )

        lambda_cross = self._loss_weight("lambda_cross_phase")
        if lambda_cross:
            loss_output["loss_cross_phase"] = normalized_cross_spectrum_loss(pred, target, **spatial_kwargs) * lambda_cross

        lambda_cov = self._loss_weight("lambda_cov")
        if lambda_cov:
            loss_output["loss_cov"] = spatial_covariance_loss(pred, target, **spatial_kwargs) * lambda_cov

        return pred_proj, target_proj

    def _generator_adv_losses(self, pred, target, pred_proj, target_proj, loss_output):
        if not self.stage_cfg.get("use_g_adv", False):
            return
        disc = unwrap_model(self.model_disc)

        lambda_adv = self._loss_weight("lambda_adv")
        lambda_fm = self._loss_weight("lambda_fm")
        if (lambda_adv or lambda_fm) and self._use_projection_branch("g_adv_projection_mode", "mono"):
            if pred_proj is None or target_proj is None:
                dirs = self._projection_directions_for(target.device, target.dtype, "g_adv_projection_dirs_per_step")
                pred_proj = self._projection_audio(pred, dirs)
                target_proj = self._projection_audio(target, dirs)
            if target_proj.shape[0] > 0:
                loss_adv, feature_matching = self._disc_generator_loss_fp32(
                    disc["projection_discriminator"],
                    reals=target_proj,
                    fakes=pred_proj,
                    compute_fm=bool(lambda_fm),
                )
                if lambda_adv:
                    loss_output["loss_adv"] = loss_adv * lambda_adv
                if lambda_fm:
                    loss_output["feature_matching_distance"] = feature_matching * lambda_fm
                loss_output["monitor/g_adv_mono_dirs"] = pred_proj.shape[0] // max(1, pred.shape[0])

        lambda_stereo_adv = self._loss_weight("lambda_stereo_adv")
        lambda_stereo_fm = self._loss_weight("lambda_stereo_fm")
        if (
            self._stereo_projection_enabled()
            and (lambda_stereo_adv or lambda_stereo_fm)
            and self._use_projection_branch("g_adv_projection_mode", "stereo")
        ):
            pairs = self._stereo_projection_pairs_for(target.device, target.dtype, "g_adv_stereo_pairs_per_step")
            pred_stereo = self._stereo_projection_audio(pred, pairs)
            target_stereo = self._stereo_projection_audio(target, pairs)
            if target_stereo.shape[0] > 0:
                loss_stereo_adv, stereo_feature_matching = self._disc_generator_loss_fp32(
                    disc["stereo_projection_discriminator"],
                    reals=target_stereo,
                    fakes=pred_stereo,
                    compute_fm=bool(lambda_stereo_fm),
                )
                if lambda_stereo_adv:
                    loss_output["loss_stereo_adv"] = loss_stereo_adv * lambda_stereo_adv
                if lambda_stereo_fm:
                    loss_output["stereo_feature_matching_distance"] = stereo_feature_matching * lambda_stereo_fm
                loss_output["monitor/g_adv_stereo_pairs"] = pred_stereo.shape[0] // max(1, pred.shape[0])

    def _training_step(self, sample, batch_idx, optimizer_idx):
        self._set_optimizer_lrs(optimizer_idx)

        if optimizer_idx == 1 and not self.stage_cfg.get("train_d", False):
            return None
        d_update_every = self._current_d_update_every()
        if optimizer_idx == 1 and self.global_step % d_update_every != 0:
            return None

        self._set_discriminator_trainability(optimizer_idx)
        self._apply_trainability(optimizer_idx)
        sample["wavs"] = sample["wavs"].float()
        y = sample["wavs"]
        loss_output = {}
        amp_enabled = bool(hparams.get("amp", True)) and y.is_cuda
        autocast_device = "cuda" if y.is_cuda else "cpu"

        if optimizer_idx == 0:
            if not self.stage_cfg.get("train_g", True):
                self.latest_recon = None
                if self.stage_cfg.get("train_d", False) and self.global_step % d_update_every == 0:
                    with torch.no_grad():
                        with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16, enabled=amp_enabled):
                            model_outputs = self.model_gen(y)
                        y_hat = model_outputs["recon"]
                        y_hat, _ = trim_to_shortest(y_hat, y)
                        self.latest_recon = y_hat.detach()
                return None

            with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16, enabled=amp_enabled):
                model_outputs = self.model_gen(y)
            y_hat = model_outputs["recon"]
            y_hat, y = trim_to_shortest(y_hat, y)
            pred_proj, target_proj = self._generator_recon_losses(y_hat, y, loss_output)
            self._generator_adv_losses(y_hat, y, pred_proj, target_proj, loss_output)

            lambda_kl = self._loss_weight("lambda_kl")
            if lambda_kl:
                loss_output["kl_loss"] = model_outputs["kl"] * lambda_kl
                loss_output["monitor/lambda_kl"] = lambda_kl
            loss_output["monitor/mu"] = model_outputs["mu"].mean().detach()
            loss_output["monitor/logvar"] = model_outputs["logvar"].mean().detach()
            self.latest_recon = y_hat.detach()
        else:
            if self.latest_recon is None:
                return None
            y_hat = self.latest_recon
            y_hat, y = trim_to_shortest(y_hat, y)
            disc = unwrap_model(self.model_disc)

            lambda_dis = self._loss_weight("lambda_dis")
            if lambda_dis and self._use_projection_branch("disc_projection_mode", "mono"):
                dirs = self._projection_directions_for(y.device, y.dtype, "disc_projection_dirs_per_step")
                target_proj = self._projection_audio(y, dirs)
                pred_proj = self._projection_audio(y_hat.detach(), dirs)
                if target_proj.shape[0] > 0:
                    with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16, enabled=amp_enabled):
                        loss_dis, disc_scores = self._disc_discriminator_loss(
                            disc["projection_discriminator"],
                            reals=target_proj,
                            fakes=pred_proj,
                        )
                    loss_output["loss_dis"] = loss_dis * lambda_dis
                    loss_output["monitor/disc_mono_dirs"] = target_proj.shape[0] // max(1, y.shape[0])
                    if disc_scores:
                        loss_output["monitor/disc_real_score"] = disc_scores["real_score"]
                        loss_output["monitor/disc_fake_score"] = disc_scores["fake_score"]

            lambda_stereo_dis = self._loss_weight("lambda_stereo_dis")
            if (
                self._stereo_projection_enabled()
                and lambda_stereo_dis
                and self._use_projection_branch("disc_projection_mode", "stereo")
            ):
                pairs = self._stereo_projection_pairs_for(y.device, y.dtype, "disc_stereo_pairs_per_step")
                target_stereo = self._stereo_projection_audio(y, pairs)
                pred_stereo = self._stereo_projection_audio(y_hat.detach(), pairs)
                if target_stereo.shape[0] > 0:
                    with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16, enabled=amp_enabled):
                        loss_stereo_dis, stereo_disc_scores = self._disc_discriminator_loss(
                            disc["stereo_projection_discriminator"],
                            reals=target_stereo,
                            fakes=pred_stereo,
                        )
                    loss_output["loss_stereo_dis"] = loss_stereo_dis * lambda_stereo_dis
                    loss_output["monitor/disc_stereo_pairs"] = target_stereo.shape[0] // max(1, y.shape[0])
                    if stereo_disc_scores:
                        loss_output["monitor/stereo_disc_real_score"] = stereo_disc_scores["real_score"]
                        loss_output["monitor/stereo_disc_fake_score"] = stereo_disc_scores["fake_score"]

        loss_output.update(self._lr_monitor_outputs(optimizer_idx))
        loss_terms = [v for k, v in loss_output.items() if not k.startswith("monitor/")]
        if not loss_terms:
            return None
        total_loss = sum(loss_terms)
        loss_output["bs"] = sample["wavs"].shape[0]
        return total_loss, loss_output

    def on_before_optimization(self, opt_idx):
        self._mask_io_then_adapter_latent_grads(opt_idx)
        grad_norm_dict = super().on_before_optimization(opt_idx)
        if opt_idx == 0:
            nn.utils.clip_grad_norm_(self.model_gen.parameters(), hparams["generator_grad_norm"])
        else:
            nn.utils.clip_grad_norm_(self.model_disc.parameters(), hparams["discriminator_grad_norm"])
        return grad_norm_dict

    @torch.no_grad()
    def validation_step(self, sample, batch_idx):
        return {}

    @torch.no_grad()
    def test_step(self, sample, batch_idx):
        return {}
