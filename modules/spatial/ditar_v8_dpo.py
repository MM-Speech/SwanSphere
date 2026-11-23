"""Deterministic candidate scoring and explicit-noise rollouts for v8 DPO.

This subclass introduces no parameters or persistent buffers. Its checkpoints
remain strictly loadable by the ordinary v8 inference model. Candidate targets
and all diffusion/history random variables are supplied by the training task.
"""
import math
import torch
from torch.utils.checkpoint import checkpoint

from modules.spatial.ditar_v8 import DiTARModel as V8Model, ModelArgs


class DiTARModel(V8Model):
    def _history_with_explicit_noise(self, x, mask, noise):
        if noise.shape != x.shape:
            raise ValueError("LM history noise must have the same [B,T,C] shape as the candidate")
        if noise.device != x.device or not torch.is_floating_point(noise):
            raise ValueError("LM history noise must be floating point on the candidate device")
        gamma = float(self.config.lm_history_noise_infer_gamma)
        if not self.config.use_lm_audio_history or gamma == 0:
            return x
        perturbed = (math.sqrt(1 - gamma) * x.float() + math.sqrt(gamma) * noise.float()).to(x.dtype)
        return torch.where(mask.bool().unsqueeze(-1), perturbed, x)

    def _encode_lm_history(self, x, x_mask, *, inference=False):
        # The original v4 sampler calls this once after each generated patch.
        # Explicit tensor slices avoid RNG-layout changes between sequential
        # sampling and parallel candidate replay.
        rollout_noise = getattr(self, "_dpo_rollout_history_noise", None)
        if inference and rollout_noise is not None:
            begin = self._dpo_rollout_history_cursor
            end = begin + x.shape[1]
            if end > rollout_noise.shape[1]:
                raise RuntimeError("Rollout exhausted its explicit history noise")
            noisy = self._history_with_explicit_noise(x, x_mask, rollout_noise[:, begin:end])
            self._dpo_rollout_history_cursor = end
            encoded, encoded_mask = self.forward_encoder(noisy, x_mask)
            return self.lm_history_bottleneck(encoded), encoded_mask
        return super()._encode_lm_history(x, x_mask, inference=inference)

    @torch.no_grad()
    def dpo_rollout(
        self, inputs, *, initial_noise, lm_history_noise,
        timesteps=32, seq_cfg_w=(1.4, 3.0),
        timestep_annealing_w=(1.0, 0.0, 1.0),
    ):
        """Return full generated latents; each initial_noise batch item is unique.

        initial_noise: [patches, B, patch_size, 128].
        lm_history_noise: [B, patches*patch_size, 128]; the last patch is unused.
        No random number is drawn by the LM history hook.
        """
        if hasattr(self, "_dpo_rollout_history_noise"):
            raise RuntimeError("Nested DPO rollouts are unsupported")
        if initial_noise.ndim != 4 or initial_noise.shape[2:] != (
            self.config.patch_size, self.config.out_channels
        ):
            raise ValueError("initial_noise must be [patches,B,patch_size,128]")
        patches, batch, size, channels = initial_noise.shape
        if tuple(lm_history_noise.shape) != (batch, patches * size, channels):
            raise ValueError("Rollout history noise shape differs from initial_noise")
        if int(timesteps) != timesteps or timesteps <= 0:
            raise ValueError("timesteps must be a positive integer")
        if not torch.isfinite(lm_history_noise).all():
            raise ValueError("LM history noise contains non-finite values")
        was_training = self.training
        self.eval()
        self._dpo_rollout_history_noise = lm_history_noise
        self._dpo_rollout_history_cursor = 0
        try:
            latent, _ = super().inference(
                inputs, timesteps=int(timesteps), seq_cfg_w=tuple(seq_cfg_w),
                timestep_annealing_w=tuple(timestep_annealing_w),
                max_new_patches=patches, initial_noise=initial_noise,
                lm_history_source="generated", dit_history_source="generated",
                use_tqdm=False,
            )
            expected = (patches - 1) * size if self.config.use_lm_audio_history else 0
            if self._dpo_rollout_history_cursor != expected:
                raise RuntimeError("Sampler consumed an unexpected number of history patches")
            return latent
        finally:
            del self._dpo_rollout_history_noise
            del self._dpo_rollout_history_cursor
            self.train(was_training)

    def dpo_conditions(self, inputs):
        """Replay a candidate's own history in parallel, preserving gradients."""
        x, lengths = inputs["lat"], inputs["lat_lens"]
        if x.ndim != 3 or x.shape[-1] != self.config.in_channels:
            raise ValueError("Candidate latent must be [B,T,128]")
        batch, frames, channels = x.shape
        size = self.config.patch_size
        if frames % size:
            raise ValueError("Candidate length must be patch aligned")
        lengths = torch.as_tensor(lengths, device=x.device, dtype=torch.long).reshape(-1)
        if lengths.shape != (batch,) or bool(((lengths <= 0) | (lengths > frames)).any()):
            raise ValueError("Invalid candidate valid-frame lengths")
        mask = torch.arange(frames, device=x.device)[None] < lengths[:, None]
        patches = frames // size
        video, _ = self.forward_ling_encoder(inputs)
        indices = (torch.arange(patches, device=x.device).float() * video.shape[1] / patches).floor().long()
        video = video[:, indices.clamp_max(video.shape[1] - 1)]
        if self.config.use_lm_audio_history:
            noise = inputs["lm_history_noise"]
            history = self._history_with_explicit_noise(x, mask, noise)
            encoded, _ = self.forward_encoder(history, mask.long())
            encoded = self.lm_history_bottleneck(encoded)
        else:
            encoded = video.new_zeros(batch, patches, self.config.lm_dim)
        start = self.speech_start_token.expand(batch, 1, -1)
        lm_inputs = torch.cat((start, encoded[:, :-1]), dim=1) + video
        # Future padded positions cannot affect valid outputs of the causal LM.
        lm_mask = torch.ones((batch, patches), device=x.device, dtype=torch.long)
        hidden = self.lm(
            inputs_embeds=lm_inputs, attention_mask=lm_mask,
            encoder_hidden_states=None, encoder_attention_mask=None,
            use_cache=False,
        ).last_hidden_state
        hidden = self.lm_postnet(self.lm_head(hidden))
        history = torch.cat((x.new_zeros(batch, size, channels), x[:, :-size]), dim=1)
        history_mask = torch.cat((mask.new_zeros(batch, size), mask[:, :-size]), dim=1)
        history = history.reshape(batch * patches, size, channels)
        history_mask = history_mask.reshape(batch * patches, size)
        if self.config.ctx_n_patches == 0:
            history_mask = torch.zeros_like(history_mask)
        encoded_dit = self._encode_dit_history_patch(history, history_mask)
        null_dit = self._null_dit_history_embedding(
            batch * patches, dtype=encoded_dit.dtype, device=x.device
        )
        dit_condition = torch.where(history_mask.any(-1, keepdim=True), encoded_dit, null_dit)
        return {
            "h": hidden,
            "e_ctx": dit_condition.reshape(batch, patches, -1),
            "mask": mask,
        }

    def _dpo_score_batch(self, inputs):
        x = inputs["lat"]
        batch, frames, channels = x.shape
        size = self.config.patch_size
        patches = frames // size
        noise = inputs["flow_noise"]
        times = inputs["flow_t"]
        if noise.shape != x.shape or noise.device != x.device:
            raise ValueError("flow_noise must match candidate shape and device")
        if times.shape != (batch, patches) or times.device != x.device:
            raise ValueError("flow_t must be [B,patches] on candidate device")
        if not torch.is_floating_point(times) or not torch.is_floating_point(noise):
            raise TypeError("Flow times and noise must be floating point")
        if not torch.isfinite(times).all() or bool(((times < 0) | (times > 1)).any()):
            raise ValueError("Flow times must be finite and in [0,1]")
        conditions = self.dpo_conditions(inputs)
        target = x.reshape(batch * patches, size, channels).float()
        noise = noise.reshape_as(target).float()
        times = times.reshape(-1).float()
        noisy = times[:, None, None] * target + (1 - times[:, None, None]) * noise
        velocity = target - noise
        token_mask = conditions["mask"].reshape(batch * patches, size)
        hidden = conditions["h"].reshape(batch * patches, -1)
        context = conditions["e_ctx"].reshape(batch * patches, -1)
        # TimestepEmbedding contains ordinary linear layers; run the same FP32
        # time embedding as v4, without casting away the condition gradients.
        fp32 = torch.autocast(device_type=x.device.type, enabled=False)
        with fp32:
            embedded_time = self.f5_time_embed(times)
        adaln = (
            embedded_time
            + self._h_to_adaln_condition(hidden).to(embedded_time)
            + self._dit_history_to_adaln_condition(context).to(embedded_time)
        )
        tokens = self.decoder_prenet(noisy)
        tokens = tokens + self.ctx_mask_embed(torch.zeros_like(token_mask, dtype=torch.long)).to(tokens.dtype)
        prediction = self.decoder(
            inputs_embeds=tokens, time_step=adaln.to(tokens.dtype),
            attention_mask=token_mask.long(),
        ).last_hidden_state
        prediction = self.decoder_postnet(prediction)
        squared = (prediction.float() - velocity).square().reshape(batch, frames, channels)
        mask = conditions["mask"].float()
        denominator = mask.sum(dim=1).clamp_min(1) * channels
        per_sequence = (squared * mask[:, :, None]).sum(dim=(1, 2)) / denominator
        return {"per_sequence_loss": per_sequence}

    def forward(self, inputs):
        if not inputs.get("dpo_score", False):
            return super().forward(inputs)
        if self.training:
            raise RuntimeError("DPO scoring requires model.eval(); gradients remain enabled in eval mode")
        if self.config.use_caption_encoder:
            raise NotImplementedError("V8 DPO currently supports video conditioning only")
        batch = inputs["lat"].shape[0]
        microbatch = int(inputs.get("dpo_microbatch", 0) or batch)
        if microbatch <= 0:
            raise ValueError("dpo_microbatch must be non-negative")
        results = []
        for begin in range(0, batch, microbatch):
            end = min(batch, begin + microbatch)
            sliced = {
                key: value[begin:end] if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch else value
                for key, value in inputs.items()
            }
            if self.config.do_checkpoint and torch.is_grad_enabled():
                # Explicit non-reentrant checkpointing remains active in eval
                # mode, unlike HF layer checkpointing gated by module.training.
                result = checkpoint(self._dpo_score_batch, sliced, use_reentrant=False)
            else:
                result = self._dpo_score_batch(sliced)
            results.append(result["per_sequence_loss"])
        return {"per_sequence_loss": torch.cat(results, dim=0)}
