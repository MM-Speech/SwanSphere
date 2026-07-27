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

from modules.foa_vae.wavvae_v4 import build_foa_wavvae_v4, trainable_param_report
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


class FOAWavVAEV4Task(FastDatasetMixin, BaseTask):
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

    def build_model(self):
        init_pretrained = (
            not hparams.get("from_scratch", False)
            and not hparams.get("load_ckpt", "")
            and not hparams.get("resume_from", "")
        )
        self.model_gen = build_foa_wavvae_v4(hparams=hparams, init_pretrained=init_pretrained)
        self.model_gen.requires_grad_(True)
        trainable_param_report(self.model_gen)

        disc_args = hparams["loss_configs"]["discriminator"]["config"]
        self.model_disc = torch.nn.ModuleDict()
        self.model_disc["discriminator"] = EncodecDiscriminator(in_channels=self.model_gen.out_channels, **disc_args)

        stft_args = hparams["loss_configs"]["spectral"]["config"]
        self.mrstft = auraloss.MultiResolutionSTFTLoss(sample_rate=self.sample_rate, **stft_args)
        self.mel_multiband_loss = self._build_mel_multiband_loss()
        self.high_frequency_excess_db_loss = self._build_high_frequency_excess_db_loss()

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

    def _training_step(self, sample, batch_idx, optimizer_idx):
        self._set_discriminator_trainability(optimizer_idx)
        sample["wavs"] = sample["wavs"].float()
        y = sample["wavs"]
        loss_output = {}
        amp_enabled = bool(hparams.get("amp", True)) and y.is_cuda
        autocast_device = "cuda" if y.is_cuda else "cpu"
        d_pretrain_steps = int(hparams.get("d_pretrain_steps", 7000))

        if optimizer_idx == 0:
            with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16, enabled=amp_enabled):
                model_outputs = self.model_gen(y)
            y_hat = model_outputs["recon"]
            y_hat, y = trim_to_shortest(y_hat, y)

            lambda_mrstft = self._loss_weight("lambda_mrstft", 1.0)
            if lambda_mrstft:
                with torch.autocast(device_type=autocast_device, enabled=False):
                    loss_output["loss_mrstft"] = self.mrstft(y_hat.float(), y.float()) * lambda_mrstft

            lambda_high_frequency = self._loss_weight("lambda_high_frequency_excess_db", 0.0)
            if lambda_high_frequency > 0.0:
                with torch.autocast(device_type=autocast_device, enabled=False):
                    loss_output["loss_high_frequency_excess_db"] = self._compute_high_frequency_excess_db_loss(
                        y_hat.float(), y.float()
                    )
                loss_output["monitor/lambda_high_frequency_excess_db"] = y_hat.new_tensor(lambda_high_frequency)

            lambda_mel = self._loss_weight("lambda_mel_multiband", 0.0)
            if lambda_mel and self.mel_multiband_loss is not None:
                with torch.autocast(device_type=autocast_device, enabled=False):
                    loss_output["loss_mel_multiband"] = (
                        self.mel_multiband_loss(
                            self._foa_as_mono_batch(y_hat.float()),
                            self._foa_as_mono_batch(y.float()),
                        )
                        * lambda_mel
                    )
                loss_output["monitor/lambda_mel_multiband"] = y_hat.new_tensor(lambda_mel)

            lambda_kl = self._loss_weight("lambda_kl", 1.0e-5)
            if lambda_kl:
                loss_output["kl_loss"] = model_outputs["kl"] * lambda_kl
                loss_output["monitor/lambda_kl"] = y_hat.new_tensor(lambda_kl)

            g_adv_active = self.global_step >= d_pretrain_steps
            gan_ramp = self._gan_ramp(d_pretrain_steps)
            if g_adv_active:
                lambda_adv = self._loss_weight("lambda_adv", 0.02) * gan_ramp
                lambda_fm = self._loss_weight("lambda_feature_matching", 25.0) * gan_ramp
                if lambda_adv or lambda_fm:
                    with torch.autocast(device_type=autocast_device, enabled=False):
                        loss_adv, feature_matching = self._disc_generator_loss(y.float(), y_hat.float())
                    if lambda_adv:
                        loss_output["loss_adv"] = loss_adv * lambda_adv
                    if lambda_fm:
                        loss_output["feature_matching_distance"] = feature_matching * lambda_fm

            loss_output["monitor/g_adv_active"] = y_hat.new_tensor(float(g_adv_active))
            loss_output["monitor/d_pretrain_active"] = y_hat.new_tensor(float(not g_adv_active))
            loss_output["monitor/gan_ramp"] = y_hat.new_tensor(float(gan_ramp))
            loss_output["monitor/effective_lambda_adv"] = y_hat.new_tensor(float(self._loss_weight("lambda_adv", 0.02) * gan_ramp))
            loss_output["monitor/effective_lambda_feature_matching"] = y_hat.new_tensor(
                float(self._loss_weight("lambda_feature_matching", 25.0) * gan_ramp)
            )
            loss_output["monitor/mu"] = model_outputs["mu"].mean().detach()
            loss_output["monitor/logvar"] = model_outputs["logvar"].mean().detach()
            self.latest_recon = y_hat.detach()

        else:
            if self.latest_recon is None:
                with torch.no_grad():
                    with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16, enabled=amp_enabled):
                        model_outputs = self.model_gen(y)
                    self.latest_recon = trim_to_shortest(model_outputs["recon"], y)[0].detach()

            y_hat = self.latest_recon
            y_hat, y = trim_to_shortest(y_hat, y)
            lambda_dis = self._loss_weight("lambda_dis", 1.0)
            if lambda_dis:
                with torch.autocast(device_type=autocast_device, enabled=False):
                    loss_dis, disc_scores = self._disc_discriminator_loss(y.float(), y_hat.detach().float())
                loss_output["loss_dis"] = loss_dis * lambda_dis
                if disc_scores:
                    loss_output["monitor/disc_real_score"] = disc_scores["real_score"]
                    loss_output["monitor/disc_fake_score"] = disc_scores["fake_score"]

        loss_output.update(self._lr_monitor_outputs(optimizer_idx))
        loss_terms = [v for k, v in loss_output.items() if not k.startswith("monitor/")]
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
