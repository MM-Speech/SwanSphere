import collections
import collections.abc

for _type_name in ("Mapping", "MutableMapping", "Sequence"):
    if not hasattr(collections, _type_name):
        setattr(collections, _type_name, getattr(collections.abc, _type_name))

from attrdict import AttrDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_audio_tools.models.discriminators import EncodecDiscriminator
from stable_audio_tools.training.losses import auraloss

from modules.foa_vae.wavvae_v4_init import (
    V4_INIT_BOUNDARY_PREFIXES,
    build_foa_wavvae_v4_init,
    set_v4_init_generator_trainability,
)
from modules.foa_vae.wavvae_v4_post import FOAV4SpatialLoss
from tasks.tts.dataset_utils.dataset_mixin import FastDatasetMixin
from utils.audio.mel import MultiResolutionMultiBandMelLoss
from utils.commons.base_task import BaseTask
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams
from utils.commons.import_utils import import_module_bystr
from utils.nn.ema import EMAModel
from utils.nn.model_utils import unwrap_model
from utils.nn.schedulers import WarmupSchedule


class TensorStateEMAModel(EMAModel):
    """EMAModel variant whose state_dict is compatible with this trainer.

    The local trainer assumes every state_dict value supports `.to(dtype)`.
    utils.nn.ema.EMAModel stores Python scalars and a list of tensors, so saving
    it directly crashes during checkpointing. Flattening the state keeps EMA
    checkpoints tensor-only without changing the EMA update rule.
    """

    @staticmethod
    def _tensor_scalar(value, dtype=torch.float32):
        return torch.tensor(value, dtype=dtype)

    def state_dict(self, *args, **kwargs):
        state = {
            "decay": self._tensor_scalar(float(self.decay)),
            "min_decay": self._tensor_scalar(float(self.min_decay)),
            "optimization_step": self._tensor_scalar(float(self.optimization_step)),
            "update_after_step": self._tensor_scalar(float(self.update_after_step)),
            "use_ema_warmup": self._tensor_scalar(float(self.use_ema_warmup)),
            "inv_gamma": self._tensor_scalar(float(self.inv_gamma)),
            "power": self._tensor_scalar(float(self.power)),
        }
        for idx, param in enumerate(self.shadow_params):
            state[f"shadow_params.{idx:06d}"] = param
        return state

    def load_state_dict(self, state_dict, strict=None):
        if "shadow_params" in state_dict:
            return super().load_state_dict(state_dict, strict=strict)

        def scalar(name, default):
            value = state_dict.get(name, None)
            if value is None:
                return default
            if torch.is_tensor(value):
                return float(value.detach().float().cpu().item())
            return float(value)

        self.decay = scalar("decay", self.decay)
        self.min_decay = scalar("min_decay", self.min_decay)
        self.optimization_step = int(round(scalar("optimization_step", self.optimization_step)))
        self.update_after_step = int(round(scalar("update_after_step", self.update_after_step)))
        self.use_ema_warmup = bool(round(scalar("use_ema_warmup", float(self.use_ema_warmup))))
        self.inv_gamma = scalar("inv_gamma", self.inv_gamma)
        self.power = scalar("power", self.power)

        shadow_keys = sorted(
            [key for key in state_dict if key.startswith("shadow_params.")],
            key=lambda key: int(key.rsplit(".", 1)[-1]),
        )
        if shadow_keys:
            if strict and len(shadow_keys) != len(self.shadow_params):
                raise RuntimeError(
                    f"EMA shadow param count mismatch: checkpoint={len(shadow_keys)} current={len(self.shadow_params)}"
                )
            current = self.shadow_params
            self.shadow_params = []
            for idx, key in enumerate(shadow_keys):
                value = state_dict[key].detach().clone()
                if idx < len(current):
                    value = value.to(device=current[idx].device, dtype=current[idx].dtype)
                self.shadow_params.append(value)
        return None


class HighFrequencyExcessDBLoss(nn.Module):
    def __init__(
        self,
        sample_rate,
        min_frequency_hz=6000.0,
        margin_db=1.0,
        fft_size=2048,
        hop_size=512,
        win_length=2048,
        min_db=-80.0,
    ):
        super().__init__()
        sample_rate = float(sample_rate)
        min_frequency_hz = float(min_frequency_hz)
        fft_size = int(fft_size)
        hop_size = int(hop_size)
        win_length = int(win_length)
        if not 0.0 <= min_frequency_hz < sample_rate / 2.0:
            raise ValueError("min_frequency_hz must be between 0 and Nyquist")
        if float(margin_db) < 0.0:
            raise ValueError("margin_db must be non-negative")
        if fft_size <= 0 or hop_size <= 0 or not 0 < win_length <= fft_size:
            raise ValueError("invalid STFT configuration")

        frequencies = torch.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
        high_frequency_mask = frequencies > min_frequency_hz
        if not high_frequency_mask.any():
            raise ValueError("min_frequency_hz selects no STFT bins")

        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.margin_db = float(margin_db)
        self.min_amplitude = 10.0 ** (float(min_db) / 20.0)
        window = torch.hann_window(win_length)
        self.magnitude_scale = 2.0 / float(window.sum())
        self.register_buffer("window", window, persistent=False)
        self.register_buffer("high_frequency_mask", high_frequency_mask, persistent=False)

    def _magnitude_db(self, waveform):
        spectrum = torch.stft(
            waveform,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        magnitude = (spectrum.abs() * self.magnitude_scale).clamp_min(self.min_amplitude)
        return 20.0 * torch.log10(magnitude)

    def forward(self, reconstruction, target):
        if reconstruction.shape != target.shape:
            raise ValueError("reconstruction and target must have the same shape")
        if reconstruction.ndim != 3:
            raise ValueError("expected audio shaped [batch, channels, samples]")

        reconstruction = reconstruction.reshape(-1, reconstruction.shape[-1])
        target = target.reshape(-1, target.shape[-1])
        reconstruction_db = self._magnitude_db(reconstruction)
        target_db = self._magnitude_db(target)
        excess_db = reconstruction_db[:, self.high_frequency_mask] - target_db[:, self.high_frequency_mask]
        return F.relu(excess_db - self.margin_db).mean()


def trim_to_shortest(a, b):
    if a.shape[-1] > b.shape[-1]:
        return a[:, :, : b.shape[-1]], b
    if b.shape[-1] > a.shape[-1]:
        return a, b[:, :, : a.shape[-1]]
    return a, b


def resolve_v4_init_generator_stage(stage_config, global_step):
    """Resolve Generator trainability from config and checkpointed global step."""

    stage_config = dict(stage_config or {})
    mode = stage_config.get("mode", "full")
    if mode == "full":
        return "full"
    if mode != "boundary_then_full":
        raise ValueError(
            "foa_init_stage.mode must be 'full' or 'boundary_then_full', "
            f"got {mode!r}"
        )

    generator_start_step = int(stage_config.get("generator_start_step", 10000))
    full_unfreeze_step = int(stage_config.get("full_unfreeze_step", 20000))
    if generator_start_step < 0:
        raise ValueError("foa_init_stage.generator_start_step must be non-negative")
    if full_unfreeze_step <= generator_start_step:
        raise ValueError(
            "foa_init_stage.full_unfreeze_step must be greater than generator_start_step"
        )

    global_step = int(global_step)
    if global_step < generator_start_step:
        return "frozen"
    if global_step < full_unfreeze_step:
        return "boundary"
    return "full"


class FOAWavVAEV4InitTask(FastDatasetMixin, BaseTask):
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
        else:
            raise ValueError(f"Unknown dataloader_version={hparams.get('dataloader_version')}")
        self.hparams = hparams
        self.config = AttrDict(hparams)
        self.sample_rate = hparams["sample_rate"]
        self.latest_recon = None
        self.model_gen_ema = None
        self.mel_multiband_loss = None
        self.high_frequency_excess_db_loss = None
        self.spatial_post_loss = None
        self.generator_total_param_count = 0
        self.generator_boundary_param_count = 0

    @property
    def loss_cfg(self):
        return hparams.get("losses", {})

    def _loss_weight(self, name, default=0.0):
        return float(self.loss_cfg.get(name, default) or 0.0)

    def _gan_ramp(self, d_pretrain_steps):
        ramp_steps = int(hparams.get("gan_ramp_steps", 0) or 0)
        if self.global_step < d_pretrain_steps:
            return 0.0
        if ramp_steps <= 0:
            return 1.0
        return min(1.0, max(0.0, float(self.global_step - d_pretrain_steps) / float(ramp_steps)))

    @property
    def init_stage_cfg(self):
        return hparams.get("foa_init_stage", {"mode": "full"})

    def _generator_stage(self):
        return resolve_v4_init_generator_stage(self.init_stage_cfg, self.global_step)

    def _apply_generator_trainability(self):
        stage = self._generator_stage()
        set_v4_init_generator_trainability(self.model_gen, stage)
        return stage

    def _stage_monitor_outputs(self, reference, stage=None):
        stage = stage or self._generator_stage()
        stage_code = {"frozen": 0.0, "boundary": 1.0, "full": 2.0}[stage]
        total = int(getattr(self, "generator_total_param_count", 0) or 0)
        boundary = int(getattr(self, "generator_boundary_param_count", 0) or 0)
        trainable = {"frozen": 0, "boundary": boundary, "full": total}[stage]
        fraction = float(trainable) / float(total) if total > 0 else 0.0
        return {
            "monitor/generator_stage": reference.new_tensor(stage_code),
            "monitor/generator_trainable_params": reference.new_tensor(float(trainable)),
            "monitor/generator_trainable_fraction": reference.new_tensor(fraction),
        }

    def build_model(self):
        init_pretrained = (
            not hparams.get("from_scratch", False)
            and not hparams.get("load_ckpt", "")
            and not hparams.get("resume_from", "")
        )
        self.model_gen = build_foa_wavvae_v4_init(hparams=hparams, init_omniaudio=init_pretrained)
        self.generator_total_param_count = sum(
            parameter.numel() for parameter in self.model_gen.parameters()
        )
        self.generator_boundary_param_count = sum(
            parameter.numel()
            for name, parameter in self.model_gen.named_parameters()
            if name.startswith(V4_INIT_BOUNDARY_PREFIXES)
        )
        # DDP decides whether to wrap a module from requires_grad at build time.
        # Keep every Generator parameter enabled here; the exact stage is
        # reapplied inside every Generator step after the trainer's optimizer
        # routing has also toggled requires_grad.
        self.model_gen.requires_grad_(True)
        initial_stage = self._generator_stage()
        initial_trainable_count = {
            "frozen": 0,
            "boundary": self.generator_boundary_param_count,
            "full": self.generator_total_param_count,
        }[initial_stage]
        print(
            "| FOA VAE v4_init planned Generator stage="
            f"{initial_stage}, step-trainable="
            f"{initial_trainable_count:,}/"
            f"{self.generator_total_param_count:,}"
        )

        disc_args = hparams["loss_configs"]["discriminator"]["config"]
        self.model_disc = torch.nn.ModuleDict()
        self.model_disc["discriminator"] = EncodecDiscriminator(in_channels=self.model_gen.out_channels, **disc_args)

        stft_args = hparams["loss_configs"]["spectral"]["config"]
        self.mrstft = auraloss.MultiResolutionSTFTLoss(sample_rate=self.sample_rate, **stft_args)
        self.mel_multiband_loss = self._build_mel_multiband_loss()
        self.high_frequency_excess_db_loss = self._build_high_frequency_excess_db_loss()
        self.spatial_post_loss = FOAV4SpatialLoss(
            sample_rate=self.sample_rate,
            projection_mrstft_config=stft_args,
            projection_config=hparams.get("foa_projection", {}),
            spatial_config=hparams.get("foa_spatial_loss", {}),
        )

        others = []
        if hparams.get("use_ema", False):
            self.model_gen_ema = TensorStateEMAModel(
                self.model_gen.parameters(),
                decay=float(hparams.get("ema_decay", 0.9999)),
                update_after_step=int(hparams.get("ema_update_after_step", 1)),
                use_ema_warmup=bool(hparams.get("ema_use_warmup", True)),
                power=float(hparams.get("ema_power", 0.75)),
            )
            others.append(self.model_gen_ema)
        return {"trainable": [self.model_gen, self.model_disc], "others": others}

    def _build_mel_multiband_loss(self):
        if self._loss_weight("lambda_mel_multiband", 0.0) <= 0.0:
            return None
        cfg = hparams.get("loss_configs", {}).get("mel_multiband", {})
        mel_params = cfg.get("resolutions", [])
        if not mel_params:
            mel_params = [
                {"fft_size": 512, "hop_size": 128, "win_size": 512, "audio_num_mel_bins": 80, "fmin": 0, "fmax": 20000},
                {"fft_size": 1024, "hop_size": 256, "win_size": 1024, "audio_num_mel_bins": 128, "fmin": 0, "fmax": 20000},
                {"fft_size": 2048, "hop_size": 512, "win_size": 2048, "audio_num_mel_bins": 160, "fmin": 0, "fmax": 20000},
            ]
        mel_params = [
            {
                **dict(item),
                "audio_sample_rate": int(hparams.get("audio_sample_rate", hparams.get("sample_rate", 44100))),
            }
            for item in mel_params
        ]
        band_edges_hz = cfg.get("band_edges_hz", [0, 1000, 6000, 20000])
        band_weights = cfg.get("band_weights", [1.0, 1.0, 1.0])
        return MultiResolutionMultiBandMelLoss(mel_params, band_edges_hz, band_weights)

    def _build_high_frequency_excess_db_loss(self):
        if self._loss_weight("lambda_high_frequency_excess_db", 0.0) <= 0.0:
            return None
        cfg = hparams.get("loss_configs", {}).get("high_frequency_excess_db", {})
        return HighFrequencyExcessDBLoss(
            sample_rate=self.sample_rate,
            min_frequency_hz=cfg.get("min_frequency_hz", 6000.0),
            margin_db=cfg.get("margin_db", 1.0),
            fft_size=cfg.get("fft_size", 2048),
            hop_size=cfg.get("hop_size", 512),
            win_length=cfg.get("win_length", 2048),
            min_db=cfg.get("min_db", -80.0),
        )

    @staticmethod
    def _foa_as_mono_batch(wavs):
        return wavs.reshape(wavs.shape[0] * wavs.shape[1], wavs.shape[-1])

    def _compute_high_frequency_excess_db_loss(self, reconstruction, target):
        weight = self._loss_weight("lambda_high_frequency_excess_db", 0.0)
        if weight <= 0.0:
            return None
        if self.high_frequency_excess_db_loss is None:
            raise RuntimeError("high-frequency excess loss is enabled but not initialized")
        return self.high_frequency_excess_db_loss(reconstruction, target) * weight

    def load_model(self):
        ckpt = hparams.get("load_ckpt", "")
        if not ckpt:
            return
        load_ckpt(self.model_gen, ckpt, "model_gen", strict=hparams.get("load_ckpt_strict", False))
        if hparams.get("load_ckpt_disc", True):
            load_ckpt(self.model_disc, ckpt, "model_disc", strict=False, silent=True)
        if self.model_gen_ema is not None:
            load_ckpt(self.model_gen_ema, ckpt, "model_gen_ema", strict=False, silent=True)

    def fsdp_optm2model(self):
        return [self.model_gen]

    def build_optimizer(self):
        optimizer_gen = torch.optim.AdamW(
            self.model_gen.parameters(),
            lr=hparams.get("lr", 1.0e-5),
            betas=[hparams["adam_b1"], hparams["adam_b2"]],
            weight_decay=hparams.get("weight_decay", 0.0),
        )
        optimizer_disc = torch.optim.AdamW(
            self.model_disc.parameters(),
            lr=hparams.get("disc_lr", hparams.get("lr", 1.0e-5)),
            betas=[hparams["adam_b1"], hparams["adam_b2"]],
            weight_decay=hparams.get("disc_weight_decay", 0.0),
        )
        return [optimizer_gen, optimizer_disc]

    def build_scheduler(self, optimizer):
        warmup_updates = int(hparams.get("warmup_updates", 0))
        return (
            WarmupSchedule(optimizer[0], lr=hparams.get("lr", 1.0e-5), warmup_updates=warmup_updates),
            WarmupSchedule(
                optimizer[1],
                lr=hparams.get("disc_lr", hparams.get("lr", 1.0e-5)),
                warmup_updates=warmup_updates,
            ),
        )

    def _set_discriminator_trainability(self, optimizer_idx):
        unwrap_model(self.model_disc).requires_grad_(optimizer_idx == 1)

    def _disc_discriminator_loss(self, reals, fakes):
        discriminator = unwrap_model(self.model_disc)["discriminator"]
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

    def _disc_generator_loss(self, reals, fakes):
        discriminator = unwrap_model(self.model_disc)["discriminator"]
        _, loss_adv, feature_matching = discriminator.loss(reals=reals, fakes=fakes)
        return loss_adv, feature_matching

    def _lr_monitor_outputs(self, optimizer_idx):
        if not hasattr(self, "trainer") or optimizer_idx >= len(self.trainer.optimizers):
            return {}
        optimizer = self.trainer.optimizers[optimizer_idx]
        return {
            f"monitor/lr_optm{optimizer_idx}_pg{group_idx}": float(group["lr"])
            for group_idx, group in enumerate(optimizer.param_groups)
        }

    def _add_spatial_losses(self, reconstruction, target, loss_output):
        weights = {
            "proj_mrstft": self._loss_weight("lambda_proj_mrstft", 0.0),
            "intensity_dir": self._loss_weight("lambda_intensity_dir", 0.0),
            "cross_phase": self._loss_weight("lambda_cross_phase", 0.0),
            "cov": self._loss_weight("lambda_cov", 0.0),
        }
        if not any(weight > 0.0 for weight in weights.values()):
            return
        if self.spatial_post_loss is None:
            raise RuntimeError("FOA v4_init spatial loss is enabled but not initialized")

        raw_losses = self.spatial_post_loss(
            reconstruction,
            target,
            compute_projection=weights["proj_mrstft"] > 0.0,
            compute_intensity=weights["intensity_dir"] > 0.0,
            compute_cross=weights["cross_phase"] > 0.0,
            compute_covariance=weights["cov"] > 0.0,
        )
        for name, raw_loss in raw_losses.items():
            weight = weights[name]
            loss_output[f"loss_{name}"] = raw_loss * weight
            loss_output[f"monitor/raw_{name}"] = raw_loss.detach()
            loss_output[f"monitor/lambda_{name}"] = reconstruction.new_tensor(weight)

    def _training_step(self, sample, batch_idx, optimizer_idx):
        self._set_discriminator_trainability(optimizer_idx)
        sample["wavs"] = sample["wavs"].float()
        target = sample["wavs"]
        loss_output = {}
        amp_enabled = bool(hparams.get("amp", True)) and target.is_cuda
        autocast_device = "cuda" if target.is_cuda else "cpu"
        d_pretrain_steps = int(hparams.get("d_pretrain_steps", 7000))

        if optimizer_idx == 0:
            generator_stage = self._apply_generator_trainability()
            if generator_stage == "frozen":
                with torch.no_grad():
                    with torch.autocast(
                        device_type=autocast_device,
                        dtype=torch.bfloat16,
                        enabled=amp_enabled,
                    ):
                        model_outputs = self.model_gen(target)
                self.latest_recon = trim_to_shortest(
                    model_outputs["recon"],
                    target,
                )[0].detach()
                return None

            with torch.autocast(
                device_type=autocast_device,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                model_outputs = self.model_gen(target)
            reconstruction = model_outputs["recon"]
            reconstruction, target = trim_to_shortest(reconstruction, target)

            lambda_mrstft = self._loss_weight("lambda_mrstft", 1.0)
            if lambda_mrstft:
                with torch.autocast(device_type=autocast_device, enabled=False):
                    loss_output["loss_mrstft"] = self.mrstft(
                        reconstruction.float(),
                        target.float(),
                    ) * lambda_mrstft

            lambda_high_frequency = self._loss_weight("lambda_high_frequency_excess_db", 0.0)
            if lambda_high_frequency > 0.0:
                with torch.autocast(device_type=autocast_device, enabled=False):
                    loss_output["loss_high_frequency_excess_db"] = self._compute_high_frequency_excess_db_loss(
                        reconstruction.float(),
                        target.float(),
                    )
                loss_output["monitor/lambda_high_frequency_excess_db"] = reconstruction.new_tensor(
                    lambda_high_frequency
                )

            lambda_mel = self._loss_weight("lambda_mel_multiband", 0.0)
            if lambda_mel and self.mel_multiband_loss is not None:
                with torch.autocast(device_type=autocast_device, enabled=False):
                    loss_output["loss_mel_multiband"] = (
                        self.mel_multiband_loss(
                            self._foa_as_mono_batch(reconstruction.float()),
                            self._foa_as_mono_batch(target.float()),
                        )
                        * lambda_mel
                    )
                loss_output["monitor/lambda_mel_multiband"] = reconstruction.new_tensor(lambda_mel)

            with torch.autocast(device_type=autocast_device, enabled=False):
                self._add_spatial_losses(
                    reconstruction.float(),
                    target.float(),
                    loss_output,
                )

            lambda_kl = self._loss_weight("lambda_kl", 1.0e-5)
            if lambda_kl:
                loss_output["kl_loss"] = model_outputs["kl"] * lambda_kl
                loss_output["monitor/lambda_kl"] = reconstruction.new_tensor(lambda_kl)

            g_adv_active = self.global_step >= d_pretrain_steps
            gan_ramp = self._gan_ramp(d_pretrain_steps)
            if g_adv_active:
                lambda_adv = self._loss_weight("lambda_adv", 0.02) * gan_ramp
                lambda_feature_matching = self._loss_weight("lambda_feature_matching", 25.0) * gan_ramp
                if lambda_adv or lambda_feature_matching:
                    with torch.autocast(device_type=autocast_device, enabled=False):
                        loss_adv, feature_matching = self._disc_generator_loss(
                            target.float(),
                            reconstruction.float(),
                        )
                    if lambda_adv:
                        loss_output["loss_adv"] = loss_adv * lambda_adv
                    if lambda_feature_matching:
                        loss_output["feature_matching_distance"] = (
                            feature_matching * lambda_feature_matching
                        )

            loss_output["monitor/g_adv_active"] = reconstruction.new_tensor(float(g_adv_active))
            loss_output["monitor/d_pretrain_active"] = reconstruction.new_tensor(float(not g_adv_active))
            loss_output["monitor/gan_ramp"] = reconstruction.new_tensor(float(gan_ramp))
            loss_output["monitor/effective_lambda_adv"] = reconstruction.new_tensor(
                float(self._loss_weight("lambda_adv", 0.02) * gan_ramp)
            )
            loss_output["monitor/effective_lambda_feature_matching"] = reconstruction.new_tensor(
                float(self._loss_weight("lambda_feature_matching", 25.0) * gan_ramp)
            )
            loss_output["monitor/mu"] = model_outputs["mu"].mean().detach()
            loss_output["monitor/logvar"] = model_outputs["logvar"].mean().detach()
            loss_output.update(self._stage_monitor_outputs(reconstruction, generator_stage))
            self.latest_recon = reconstruction.detach()

        else:
            if self.latest_recon is None:
                with torch.no_grad():
                    with torch.autocast(
                        device_type=autocast_device,
                        dtype=torch.bfloat16,
                        enabled=amp_enabled,
                    ):
                        model_outputs = self.model_gen(target)
                    self.latest_recon = trim_to_shortest(
                        model_outputs["recon"],
                        target,
                    )[0].detach()

            reconstruction = self.latest_recon
            reconstruction, target = trim_to_shortest(reconstruction, target)
            lambda_dis = self._loss_weight("lambda_dis", 1.0)
            if lambda_dis:
                with torch.autocast(device_type=autocast_device, enabled=False):
                    loss_dis, disc_scores = self._disc_discriminator_loss(
                        target.float(),
                        reconstruction.detach().float(),
                    )
                loss_output["loss_dis"] = loss_dis * lambda_dis
                if disc_scores:
                    loss_output["monitor/disc_real_score"] = disc_scores["real_score"]
                    loss_output["monitor/disc_fake_score"] = disc_scores["fake_score"]
            loss_output.update(self._stage_monitor_outputs(reconstruction))

        loss_output.update(self._lr_monitor_outputs(optimizer_idx))
        loss_terms = [
            value
            for key, value in loss_output.items()
            if not key.startswith("monitor/")
        ]
        if not loss_terms:
            return None
        total_loss = sum(loss_terms)
        loss_output["bs"] = sample["wavs"].shape[0]
        return total_loss, loss_output

    def on_before_optimization(self, opt_idx):
        grad_norm_dict = super().on_before_optimization(opt_idx)
        if opt_idx == 0:
            nn.utils.clip_grad_norm_(self.model_gen.parameters(), hparams["generator_grad_norm"])
        else:
            nn.utils.clip_grad_norm_(self.model_disc.parameters(), hparams["discriminator_grad_norm"])
        return grad_norm_dict

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        super().on_after_optimization(epoch, batch_idx, optimizer, optimizer_idx)
        if optimizer_idx == 0 and self.model_gen_ema is not None:
            self.model_gen_ema.step(unwrap_model(self.model_gen).parameters())

    @torch.no_grad()
    def validation_step(self, sample, batch_idx):
        return {}

    @torch.no_grad()
    def test_step(self, sample, batch_idx):
        return {}


__all__ = ["FOAWavVAEV4InitTask", "HighFrequencyExcessDBLoss", "TensorStateEMAModel", "resolve_v4_init_generator_stage", "trim_to_shortest"]
