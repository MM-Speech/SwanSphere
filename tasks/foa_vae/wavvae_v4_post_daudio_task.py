import torch
import torch.nn as nn

from stable_audio_tools.models.discriminators import EncodecDiscriminator

from modules.foa_vae.wavvae_v4_post_daudio import (
    DAUDIO_DIRECTION_NAMES,
    FOADAudioViewBuilder,
)
from tasks.foa_vae.wavvae_v4_post_task import FOAWavVAEV4PostTask
from tasks.foa_vae.wavvae_v4_task import trim_to_shortest
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams
from utils.nn.model_utils import unwrap_model
from utils.nn.schedulers import WarmupSchedule


class OffsetWarmupSchedule(WarmupSchedule):
    """Warm up a newly-added optimizer relative to a resumed global step."""

    def __init__(self, optimizer, lr, warmup_updates, start_global_step, accumulate_grad_batches):
        accumulate_grad_batches = max(1, int(accumulate_grad_batches))
        self.start_update = int(start_global_step) // accumulate_grad_batches
        super().__init__(optimizer, lr=lr, warmup_updates=warmup_updates)

    def step(self, num_updates):
        relative_updates = max(0, int(num_updates) - self.start_update)
        return super().step(relative_updates)


class FOAWavVAEV4PostDAudioTask(FOAWavVAEV4PostTask):
    """V4 post-training with a shared mono discriminator over W and FOA views."""

    def __init__(self):
        super().__init__()
        self.model_disc_audio = None
        self.daudio_view_builder = None
        self.daudio_w_loss_weight = 0.5
        self.daudio_projection_loss_weight = 0.5
        self._latest_recon_step = None
        self._latest_daudio_direction_step = None
        self._latest_daudio_direction = None
        self._latest_daudio_direction_name = None

    def build_model(self):
        build_result = super().build_model()
        disc_config = dict(hparams["loss_configs"]["discriminator"]["config"])
        self.model_disc_audio = EncodecDiscriminator(in_channels=1, **disc_config)

        daudio_config = dict(hparams.get("foa_daudio", {}) or {})
        self.daudio_view_builder = FOADAudioViewBuilder(
            gain=float(daudio_config.get("gain", 1.0)),
            direction_probabilities=daudio_config.get("direction_probabilities"),
            eps=float(daudio_config.get("eps", 1.0e-8)),
        )
        self.daudio_w_loss_weight = float(daudio_config.get("w_loss_weight", 0.5))
        self.daudio_projection_loss_weight = float(
            daudio_config.get("projection_loss_weight", 0.5)
        )
        if self.daudio_w_loss_weight < 0.0 or self.daudio_projection_loss_weight < 0.0:
            raise ValueError("D_audio view loss weights must be non-negative")
        if abs(self.daudio_w_loss_weight + self.daudio_projection_loss_weight - 1.0) > 1.0e-6:
            raise ValueError("D_audio W and projection loss weights must sum to 1")

        build_result["trainable"].append(self.model_disc_audio)
        return build_result

    def load_model(self):
        super().load_model()
        ckpt = hparams.get("load_ckpt", "")
        if ckpt and self.model_disc_audio is not None:
            load_ckpt(
                self.model_disc_audio,
                ckpt,
                "model_disc_audio",
                strict=False,
                silent=True,
            )

    def build_optimizer(self):
        optimizers = list(super().build_optimizer())
        optimizer_disc_audio = torch.optim.AdamW(
            self.model_disc_audio.parameters(),
            lr=hparams.get("disc_lr", hparams.get("lr", 1.0e-5)),
            betas=[hparams["adam_b1"], hparams["adam_b2"]],
            weight_decay=hparams.get("disc_weight_decay", 0.0),
        )
        optimizers.append(optimizer_disc_audio)
        return optimizers

    def build_scheduler(self, optimizer):
        schedulers = list(super().build_scheduler(optimizer[:2]))
        schedulers.append(
            OffsetWarmupSchedule(
                optimizer[2],
                lr=hparams.get("disc_lr", hparams.get("lr", 1.0e-5)),
                warmup_updates=int(hparams.get("warmup_updates", 0)),
                start_global_step=self._daudio_start_step(),
                accumulate_grad_batches=int(hparams.get("accumulate_grad_batches", 1)),
            )
        )
        return tuple(schedulers)

    def _set_discriminator_trainability(self, optimizer_idx):
        super()._set_discriminator_trainability(optimizer_idx)
        if self.model_disc_audio is not None:
            unwrap_model(self.model_disc_audio).requires_grad_(optimizer_idx == 2)

    @staticmethod
    def _daudio_start_step():
        return int(hparams.get("daudio_start_step", 0))

    def _daudio_relative_step(self):
        return int(self.global_step) - self._daudio_start_step()

    @staticmethod
    def _daudio_pretrain_steps():
        return max(0, int(hparams.get("daudio_pretrain_steps", 10000)))

    @staticmethod
    def _daudio_ramp_steps():
        return max(0, int(hparams.get("daudio_gan_ramp_steps", 10000)))

    def _daudio_pretraining(self):
        return self._daudio_relative_step() < self._daudio_pretrain_steps()

    def _daudio_gan_ramp(self):
        relative_adversarial_step = self._daudio_relative_step() - self._daudio_pretrain_steps()
        if relative_adversarial_step < 0:
            return 0.0
        ramp_steps = self._daudio_ramp_steps()
        if ramp_steps <= 0:
            return 1.0
        return min(1.0, max(0.0, float(relative_adversarial_step) / float(ramp_steps)))

    def _rank_specific_direction_generator(self):
        rank = int(getattr(getattr(self, "trainer", None), "proc_rank", 0))
        base_seed = int(hparams.get("seed", 1234))
        seed = (
            base_seed
            + int(self.global_step) * 1_000_003
            + rank * 97_409
        ) % (2**63 - 1)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return generator

    def _get_daudio_direction(self, reference):
        if self.daudio_view_builder is None:
            raise RuntimeError("D_audio view builder is not initialized")
        if self._latest_daudio_direction_step != int(self.global_step):
            direction, direction_name = self.daudio_view_builder.sample_direction(
                device=reference.device,
                dtype=reference.dtype,
                generator=self._rank_specific_direction_generator(),
            )
            self._latest_daudio_direction = direction
            self._latest_daudio_direction_name = direction_name
            self._latest_daudio_direction_step = int(self.global_step)
        return self._latest_daudio_direction, self._latest_daudio_direction_name

    def _daudio_views(self, reals, fakes):
        direction, direction_name = self._get_daudio_direction(reals)
        real_w, real_projections = self.daudio_view_builder.build_views(reals, direction)
        fake_w, fake_projections = self.daudio_view_builder.build_views(fakes, direction)
        return real_w, fake_w, real_projections, fake_projections, direction_name

    @staticmethod
    def _single_discriminator_loss(discriminator, reals, fakes):
        if hasattr(discriminator, "discriminator_loss"):
            output = discriminator.discriminator_loss(
                reals=reals,
                fakes=fakes,
                return_scores=True,
            )
            if isinstance(output, tuple):
                loss_dis, real_score, fake_score = output
                return loss_dis, real_score.detach(), fake_score.detach()
            return output, None, None
        loss_dis, _, _ = discriminator.loss(reals=reals, fakes=fakes)
        return loss_dis, None, None

    def _daudio_discriminator_loss(self, reals, fakes):
        discriminator = unwrap_model(self.model_disc_audio)
        real_w, fake_w, real_proj, fake_proj, direction_name = self._daudio_views(
            reals,
            fakes,
        )
        loss_w, real_score_w, fake_score_w = self._single_discriminator_loss(
            discriminator,
            real_w,
            fake_w,
        )
        loss_proj, real_score_proj, fake_score_proj = self._single_discriminator_loss(
            discriminator,
            real_proj,
            fake_proj,
        )
        loss = (
            self.daudio_w_loss_weight * loss_w
            + self.daudio_projection_loss_weight * loss_proj
        )
        details = {
            "loss_w": loss_w.detach(),
            "loss_projection": loss_proj.detach(),
            "direction_name": direction_name,
        }
        if real_score_w is not None:
            details.update(
                {
                    "real_score_w": real_score_w,
                    "fake_score_w": fake_score_w,
                    "real_score_projection": real_score_proj,
                    "fake_score_projection": fake_score_proj,
                }
            )
        return loss, details

    def _daudio_generator_loss(self, reals, fakes):
        discriminator = unwrap_model(self.model_disc_audio)
        real_w, fake_w, real_proj, fake_proj, direction_name = self._daudio_views(
            reals,
            fakes,
        )
        _, adv_w, fm_w = discriminator.loss(reals=real_w, fakes=fake_w)
        _, adv_proj, fm_proj = discriminator.loss(reals=real_proj, fakes=fake_proj)
        adv = (
            self.daudio_w_loss_weight * adv_w
            + self.daudio_projection_loss_weight * adv_proj
        )
        feature_matching = (
            self.daudio_w_loss_weight * fm_w
            + self.daudio_projection_loss_weight * fm_proj
        )
        return adv, feature_matching, {
            "adv_w": adv_w.detach(),
            "adv_projection": adv_proj.detach(),
            "fm_w": fm_w.detach(),
            "fm_projection": fm_proj.detach(),
            "direction_name": direction_name,
        }

    @staticmethod
    def _add_direction_monitors(loss_output, reference, direction_name):
        for name in DAUDIO_DIRECTION_NAMES:
            loss_output[f"monitor/daudio_direction_{name}"] = reference.new_tensor(
                float(direction_name == name)
            )

    def _add_daudio_stage_monitors(self, loss_output, reference):
        ramp = self._daudio_gan_ramp()
        loss_output["monitor/daudio_relative_step"] = reference.new_tensor(
            float(self._daudio_relative_step())
        )
        loss_output["monitor/daudio_pretrain_active"] = reference.new_tensor(
            float(self._daudio_pretraining())
        )
        loss_output["monitor/daudio_gan_ramp"] = reference.new_tensor(float(ramp))
        loss_output["monitor/effective_lambda_daudio_adv"] = reference.new_tensor(
            self._loss_weight("lambda_daudio_adv", 0.02) * ramp
        )
        loss_output["monitor/effective_lambda_daudio_feature_matching"] = reference.new_tensor(
            self._loss_weight("lambda_daudio_feature_matching", 10.0) * ramp
        )

    def _ensure_latest_reconstruction(self, target, amp_enabled, autocast_device):
        if self.latest_recon is None or self._latest_recon_step != int(self.global_step):
            with torch.no_grad():
                with torch.autocast(
                    device_type=autocast_device,
                    dtype=torch.bfloat16,
                    enabled=amp_enabled,
                ):
                    model_outputs = self.model_gen(target)
                self.latest_recon = trim_to_shortest(model_outputs["recon"], target)[0].detach()
                self._latest_recon_step = int(self.global_step)
        return self.latest_recon

    def _generator_step(self, sample, batch_idx):
        self._set_discriminator_trainability(0)
        sample["wavs"] = sample["wavs"].float()
        target = sample["wavs"]
        loss_output = {}
        amp_enabled = bool(hparams.get("amp", True)) and target.is_cuda
        autocast_device = "cuda" if target.is_cuda else "cpu"
        d_pretrain_steps = int(hparams.get("d_pretrain_steps", 7000))

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

        daudio_ramp = self._daudio_gan_ramp()
        lambda_daudio_adv = self._loss_weight("lambda_daudio_adv", 0.02) * daudio_ramp
        lambda_daudio_fm = (
            self._loss_weight("lambda_daudio_feature_matching", 10.0) * daudio_ramp
        )
        if lambda_daudio_adv or lambda_daudio_fm:
            with torch.autocast(device_type=autocast_device, enabled=False):
                daudio_adv, daudio_fm, daudio_details = self._daudio_generator_loss(
                    target.float(),
                    reconstruction.float(),
                )
            if lambda_daudio_adv:
                loss_output["loss_daudio_adv"] = daudio_adv * lambda_daudio_adv
            if lambda_daudio_fm:
                loss_output["feature_matching_distance_daudio"] = daudio_fm * lambda_daudio_fm
            loss_output["monitor/daudio_adv_w"] = daudio_details["adv_w"]
            loss_output["monitor/daudio_adv_projection"] = daudio_details["adv_projection"]
            loss_output["monitor/daudio_fm_w"] = daudio_details["fm_w"]
            loss_output["monitor/daudio_fm_projection"] = daudio_details["fm_projection"]
            self._add_direction_monitors(
                loss_output,
                reconstruction,
                daudio_details["direction_name"],
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
        self._add_daudio_stage_monitors(loss_output, reconstruction)
        loss_output["monitor/mu"] = model_outputs["mu"].mean().detach()
        loss_output["monitor/logvar"] = model_outputs["logvar"].mean().detach()
        self.latest_recon = reconstruction.detach()
        self._latest_recon_step = int(self.global_step)

        loss_output.update(self._lr_monitor_outputs(0))
        loss_terms = [
            value for key, value in loss_output.items() if not key.startswith("monitor/")
        ]
        if not loss_terms:
            return None
        total_loss = sum(loss_terms)
        loss_output["bs"] = sample["wavs"].shape[0]
        return total_loss, loss_output

    def _daudio_discriminator_step(self, sample, batch_idx):
        self._set_discriminator_trainability(2)
        sample["wavs"] = sample["wavs"].float()
        target = sample["wavs"]
        loss_output = {}
        amp_enabled = bool(hparams.get("amp", True)) and target.is_cuda
        autocast_device = "cuda" if target.is_cuda else "cpu"
        reconstruction = self._ensure_latest_reconstruction(
            target,
            amp_enabled=amp_enabled,
            autocast_device=autocast_device,
        )
        reconstruction, target = trim_to_shortest(reconstruction, target)

        lambda_dis = self._loss_weight("lambda_daudio_dis", 1.0)
        if lambda_dis:
            with torch.autocast(device_type=autocast_device, enabled=False):
                loss_dis, details = self._daudio_discriminator_loss(
                    target.float(),
                    reconstruction.detach().float(),
                )
            loss_output["loss_daudio_dis"] = loss_dis * lambda_dis
            loss_output["monitor/daudio_dis_w"] = details["loss_w"]
            loss_output["monitor/daudio_dis_projection"] = details["loss_projection"]
            if "real_score_w" in details:
                loss_output["monitor/daudio_w_real_score"] = details["real_score_w"]
                loss_output["monitor/daudio_w_fake_score"] = details["fake_score_w"]
                loss_output["monitor/daudio_projection_real_score"] = details[
                    "real_score_projection"
                ]
                loss_output["monitor/daudio_projection_fake_score"] = details[
                    "fake_score_projection"
                ]
            self._add_direction_monitors(
                loss_output,
                reconstruction,
                details["direction_name"],
            )

        self._add_daudio_stage_monitors(loss_output, reconstruction)
        loss_output.update(self._lr_monitor_outputs(2))
        loss_terms = [
            value for key, value in loss_output.items() if not key.startswith("monitor/")
        ]
        if not loss_terms:
            return None
        total_loss = sum(loss_terms)
        loss_output["bs"] = sample["wavs"].shape[0]
        return total_loss, loss_output

    def _training_step(self, sample, batch_idx, optimizer_idx):
        if self._daudio_pretraining():
            if optimizer_idx != 2:
                return None
            return self._daudio_discriminator_step(sample, batch_idx)

        if optimizer_idx == 0:
            return self._generator_step(sample, batch_idx)
        if optimizer_idx == 1:
            return super()._training_step(sample, batch_idx, optimizer_idx)
        if optimizer_idx == 2:
            return self._daudio_discriminator_step(sample, batch_idx)
        raise ValueError(f"Unsupported optimizer_idx={optimizer_idx}")

    def on_before_optimization(self, opt_idx):
        grad_norm_dict = super().on_before_optimization(opt_idx)
        if opt_idx == 2:
            nn.utils.clip_grad_norm_(
                self.model_disc_audio.parameters(),
                hparams["discriminator_grad_norm"],
            )
        return grad_norm_dict


__all__ = ["FOAWavVAEV4PostDAudioTask", "OffsetWarmupSchedule"]
