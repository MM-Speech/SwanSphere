import torch

from modules.foa_vae.wavvae_v4_post import FOAV4SpatialLoss
from tasks.foa_vae.wavvae_v4_task import FOAWavVAEV4Task, trim_to_shortest
from utils.commons.hparams import hparams


class FOAWavVAEV4PostTask(FOAWavVAEV4Task):
    """V4 continuation training with fixed-weight FOA spatial losses."""

    def __init__(self):
        super().__init__()
        self.spatial_post_loss = None

    def build_model(self):
        build_result = super().build_model()
        self.spatial_post_loss = FOAV4SpatialLoss(
            sample_rate=self.sample_rate,
            projection_mrstft_config=hparams["loss_configs"]["spectral"]["config"],
            projection_config=hparams.get("foa_projection", {}),
            spatial_config=hparams.get("foa_spatial_loss", {}),
        )
        return build_result

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
            raise RuntimeError("FOA V4 post spatial loss is enabled but not initialized")

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
        if optimizer_idx != 0:
            return super()._training_step(sample, batch_idx, optimizer_idx)

        self._set_discriminator_trainability(optimizer_idx)
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
        self.latest_recon = reconstruction.detach()

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


__all__ = ["FOAWavVAEV4PostTask"]
