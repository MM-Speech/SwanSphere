"""Online video-to-FOA preference training for the checkpoint-compatible v8.

One loader batch is one optimizer update: B videos, G fresh candidates each,
one winner/loser pair per video. Optional GT competes only for the winner slot.
The only objectives are DPO and winner FM.
Frozen reference, decoder and reward networks stay outside the task state dict.
"""

import copy
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from tasks.spatial.task_ditar_v8 import SpatialDiTARV8Task
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams
from utils.nn.model_utils import unwrap_model


def preference_losses(policy_losses, reference_losses, valid, beta, global_count=None, world_size=1):
    """Inputs are [all winners, all losers]; returns globally normalized losses."""
    if policy_losses.ndim != 1 or reference_losses.shape != policy_losses.shape:
        raise ValueError("Expected matched one-dimensional candidate losses")
    batch = valid.numel()
    if policy_losses.numel() != 2 * batch or batch == 0:
        raise ValueError("Expected one winner and loser loss per video")
    if not math.isfinite(float(beta)) or beta <= 0:
        raise ValueError("dpo_beta must be finite and positive")
    winner, loser = policy_losses[:batch], policy_losses[batch:]
    ref_winner, ref_loser = reference_losses.detach().split(batch)
    logits = float(beta) * ((loser - ref_loser) - (winner - ref_winner))
    weights = valid.to(device=logits.device, dtype=logits.dtype)
    count = weights.sum() if global_count is None else global_count.to(logits)
    scale = float(world_size) / count.clamp_min(1)
    dpo = (F.softplus(-logits) * weights).sum() * scale
    fm = (winner * weights).sum() * scale
    return dpo, fm, logits


def select_candidate_pairs(scores, generated_count, min_reward_gap=1e-6):
    """Choose the best candidate, but choose the loser only from generations.

    Columns [0, generated_count) are generated samples. The optional last
    column is GT; ties retain the first (generated) maximum as torch.argmax does.
    """
    if scores.ndim != 2 or scores.shape[0] < 1:
        raise ValueError("Expected nonempty candidate scores [B,G] or [B,G+1]")
    if not isinstance(generated_count, int) or generated_count < 2:
        raise ValueError("generated_count must be an integer >= 2")
    if scores.shape[1] not in (generated_count, generated_count + 1):
        raise ValueError("Scores must contain G generations and at most one GT")
    if not torch.is_floating_point(scores) or not torch.isfinite(scores).all():
        raise ValueError("Candidate scores must be finite floating-point values")
    if not math.isfinite(float(min_reward_gap)) or min_reward_gap < 0:
        raise ValueError("min_reward_gap must be finite and nonnegative")
    winner = scores.argmax(dim=1)
    loser = scores[:, :generated_count].argmin(dim=1)
    rows = torch.arange(scores.shape[0], device=scores.device)
    gaps = scores[rows, winner] - scores[rows, loser]
    return winner, loser, gaps, gaps > float(min_reward_gap)


class SpatialDiTARV8DPOTask(SpatialDiTARV8Task):
    def __init__(self):
        super().__init__()
        if not hparams.get("ditar_dpo_enabled", False):
            raise ValueError("Online DPO requires ditar_dpo_enabled=true")
        if hparams.get("use_fsdp", False):
            raise ValueError("Online DPO currently uses DDP; use_fsdp must be false")
        if int(hparams.get("accumulate_grad_batches", 1)) != 1:
            raise ValueError("Each DPO batch is one update; use accumulate_grad_batches=1")
        if int(hparams.get("dpo_group_size", 8)) < 2:
            raise ValueError("dpo_group_size must be at least two")
        if int(hparams.get("dpo_sample_steps", 32)) <= 0:
            raise ValueError("dpo_sample_steps must be positive")
        if hparams.get("use_audio_semantic_supervision", False):
            raise ValueError("This task optimizes DPO + winner FM only")
        for key in ("dpo_loss_weight", "dpo_fm_loss_weight"):
            if float(hparams.get(key, 1.0)) != 1.0:
                raise ValueError(f"{key} must be 1 for the requested DPO + winner FM objective")
        self.__dict__["_dpo_reference"] = None
        self.__dict__["_dpo_rewards"] = None
        self._dpo_rollout_counter = 0
        self._dpo_last_batch_report = None

    def load_model(self):
        path = hparams.get("load_ckpt", "")
        if not path or not Path(path).is_file():
            raise ValueError("DPO load_ckpt must pin an existing checkpoint file")
        load_ckpt(self.model, path, "model", strict=True, force=True, mmap=True)

    def build_scheduler(self, optimizer):
        # Keep the configured learning rate, without the inherited SFT floor.
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    def on_train_start(self):
        # The legacy BaseTask hook kills placeholder GPU processes. This task
        # deliberately has no process cleanup side effects on a shared machine.
        if getattr(self.trainer, "proc_rank", 0) == 0:
            model = unwrap_model(self.model)
            count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"| V8 online DPO: {count:,} trainable parameters; LM + DiT", flush=True)
            print(f"| Fixed reference: {hparams['load_ckpt']}", flush=True)
            print("| Objective: dpo_loss + fm_loss(winner), coefficients 1 + 1", flush=True)
            print(f"| Per rank: {hparams['max_sentences']} videos x {hparams.get('dpo_group_size', 8)} candidates; one pair/video, one update/batch", flush=True)
            if hparams.get("dpo_include_gt_candidate", False):
                print("| GT candidate: reward-ranked winner only; loser always generated; no separate GT FM", flush=True)

    def _reference(self):
        if self.__dict__["_dpo_reference"] is None:
            ref = copy.deepcopy(unwrap_model(self.model)).requires_grad_(False).eval()
            # Reload the original anchor even if the policy resumed a DPO run.
            load_ckpt(ref, hparams["load_ckpt"], "model", strict=True, force=True, mmap=True)
            self.__dict__["_dpo_reference"] = ref
        return self.__dict__["_dpo_reference"]

    def _rewards(self, device):
        if self.__dict__["_dpo_rewards"] is None:
            from modules.spatial.dpo_rewards import V8DPORewards
            self.__dict__["_dpo_rewards"] = V8DPORewards(hparams, device)
        return self.__dict__["_dpo_rewards"]

    @staticmethod
    def _rank():
        return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    @staticmethod
    def _world_size():
        return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

    @staticmethod
    def _conditions(sample):
        video = sample.get("video_feats", sample.get("inputs_embeds"))
        if video is None or video.ndim != 3:
            raise ValueError("DPO requires video features [B,T,512]")
        mask = sample.get("v_mask")
        if mask is None:
            mask = torch.ones(video.shape[:2], device=video.device, dtype=torch.long)
        return {"video_feats": video, "v_mask": mask,
                "global_clip_embedding": sample["global_clip_embedding"]}

    @torch.no_grad()
    def _decode_group(self, latents, device):
        codec = self._get_vae_codec(device)
        frames = int(hparams.get("latent_frames", 215))
        audios = []
        with torch.autocast(device_type=device.type, enabled=False):
            for latent in latents:
                audio = codec.decode(latent[:frames].float().T.unsqueeze(0))[0].float()
                if audio.ndim != 2 or audio.shape[0] != 4 or not torch.isfinite(audio).all():
                    raise RuntimeError("DPO decoder produced invalid FOA audio")
                audios.append(audio)
        return audios

    @torch.no_grad()
    def _gt_candidate_latent(self, sample, device, padded_frames, seed):
        """Resample GT with an independent RNG, retaining its true frame mask."""
        device = torch.device(device)
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(int(seed))
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(int(seed))
            gt_sample = self._encode_audio_sample(sample) if hparams.get("vae_online_encode", False) else self._sample_cached_posterior(sample)
        latent = gt_sample["lat"].detach().float()
        lengths = torch.as_tensor(gt_sample["lat_lens"], device=device, dtype=torch.long).reshape(-1)
        expected_lengths = torch.as_tensor(sample["lat_lens"], device=device, dtype=torch.long).reshape(-1)
        expected_frames = int(hparams.get("latent_frames", 215))
        expected_channels = int(hparams["latent_dim"])
        if latent.device != device or latent.shape != (expected_lengths.numel(), expected_frames, expected_channels):
            raise ValueError("GT posterior must yield [B,215,128] on the rollout device")
        if not torch.equal(lengths, expected_lengths):
            raise ValueError("GT posterior valid lengths differ from the reward interval")
        if bool(((lengths <= 0) | (lengths > expected_frames)).any()) or padded_frames < expected_frames:
            raise ValueError("Invalid GT candidate frame lengths or patch padding")
        if not torch.isfinite(latent).all():
            raise ValueError("GT posterior sample contains nonfinite values")
        return F.pad(latent, (0, 0, 0, padded_frames - expected_frames))

    @torch.no_grad()
    def _collect_candidates(self, sample):
        started = time.monotonic()
        model = unwrap_model(self.model)
        conditions = self._conditions(sample)
        device = conditions["video_feats"].device
        batch = conditions["video_feats"].shape[0]
        group = int(hparams.get("dpo_group_size", 8))
        patch = int(hparams.get("ditar_patch_size", 4))
        frames = int(hparams.get("latent_frames", 215))
        padded_frames = math.ceil(frames / patch) * patch
        patches = padded_frames // patch
        channels = int(hparams["latent_dim"])
        include_gt = bool(hparams.get("dpo_include_gt_candidate", False))
        rank = self._rank()
        seed = (int(hparams.get("seed", 42)) + 1_000_003 * int(self.global_step)
                + 10_007 * rank + 1_000_000_007 * self._dpo_rollout_counter) % (2**63 - 1)
        self._dpo_rollout_counter += 1
        generator = torch.Generator(device=device).manual_seed(seed)
        # One explicit history-noise field per video, shared across its G
        # candidates. DiT generation noise is drawn FRESH for every candidate.
        history_noise = torch.randn(batch, padded_frames, channels, device=device, generator=generator)
        rewards = self._rewards(device)
        rewards.prepare(sample)
        if rank == 0:
            print(f"| DPO update {int(self.global_step)+1}: sampling {batch} videos x {group} candidates, {hparams.get('dpo_sample_steps',32)} ODE steps", flush=True)
        generated = []
        noise_checks = []
        for candidate in range(group):
            initial_noise = torch.randn(patches, batch, patch, channels, device=device, generator=generator)
            noise_checks.append(initial_noise[0, :, 0, :4].float().cpu().tolist())
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda", cache_enabled=False):
                latent = model.dpo_rollout(
                    conditions, initial_noise=initial_noise, lm_history_noise=history_noise,
                    timesteps=int(hparams.get("dpo_sample_steps", 32)),
                    seq_cfg_w=tuple(hparams.get("dpo_seq_cfg_w", [1.4, 3.0])),
                )
            if latent.shape != (batch, padded_frames, channels) or not torch.isfinite(latent).all():
                raise RuntimeError("Online DPO candidate has invalid shape or values")
            generated.append(latent.detach().float())
            if rank == 0:
                print(f"| DPO update {int(self.global_step)+1}: candidate pass {candidate+1}/{group} complete ({time.monotonic()-started:.1f}s)", flush=True)
        candidates = torch.stack(generated, dim=1)
        del generated
        scores, all_records = [], []
        for index in range(batch):
            audios = self._decode_group(candidates[index], device)
            values, records = rewards.score_group(audios, sample, index, sample_rate=44100)
            values = torch.as_tensor(values, device=device, dtype=torch.float32)
            if values.shape != (group,) or not torch.isfinite(values).all():
                raise RuntimeError("DPO rewards must return a finite score per candidate")
            scores.append(values)
            all_records.append(records)
            del audios
        scores = torch.stack(scores)
        gt_seed = None
        if include_gt:
            # Do not consume the independent generation-noise generator above.
            gt_seed = (seed + 2_000_033) % (2**63 - 1)
            gt_latent = self._gt_candidate_latent(sample, device, padded_frames, gt_seed)
            gt_scores = []
            for index in range(batch):
                value, record = rewards.score_gt(sample, index, sample_rate=44100)
                if not math.isfinite(float(value)):
                    raise RuntimeError("GT reward must be finite")
                gt_scores.append(float(value))
                all_records[index].append(dict(record))
            candidates = torch.cat((candidates, gt_latent.unsqueeze(1)), dim=1)
            scores = torch.cat((scores, scores.new_tensor(gt_scores).unsqueeze(1)), dim=1)
        winner_index, loser_index, gaps, valid = select_candidate_pairs(
            scores, group, float(hparams.get("dpo_min_reward_gap", 1e-6)),
        )
        rows = torch.arange(batch, device=device)
        global_count = valid.float().sum()
        if self._world_size() > 1:
            dist.all_reduce(global_count)
        report = {
            "update": int(self.global_step) + 1, "rank": rank, "seed": seed,
            "sample_ids": list(sample.get("file_name", [str(i) for i in range(batch)])),
            "scores": scores.cpu().tolist(), "winner_indices": winner_index.cpu().tolist(),
            "loser_indices": loser_index.cpu().tolist(), "reward_gaps": gaps.cpu().tolist(),
            "valid_pairs": valid.cpu().tolist(), "noise_samples": noise_checks,
            "reward_records": all_records,
            "generated_count": group,
            "candidate_types": ["generated"] * group + (["gt"] if include_gt else []),
            "gt_candidate_index": group if include_gt else None,
            "gt_posterior_seed": gt_seed,
            "gt_winner": (winner_index == group).cpu().tolist() if include_gt else [False] * batch,
            "gt_scores": scores[:, group].cpu().tolist() if include_gt else None,
            "gt_can_be_loser": False,
            "rollout_seconds": time.monotonic() - started,
        }
        self._dpo_last_batch_report = report
        self._write_report(report)
        if rank == 0:
            print(f"| DPO update {int(self.global_step)+1}: rewards ready, global valid pairs={int(global_count.item())}, {report['rollout_seconds']:.1f}s", flush=True)
        return {
            "conditions": conditions, "winner": candidates[rows, winner_index],
            "loser": candidates[rows, loser_index], "history_noise": history_noise,
            "lat_lens": torch.as_tensor(sample["lat_lens"], device=device, dtype=torch.long),
            "valid": valid, "global_count": global_count, "scores": scores,
            "gaps": gaps, "records": all_records, "seconds": report["rollout_seconds"],
            "seed": seed, "generated_count": group,
            "winner_indices": winner_index, "loser_indices": loser_index,
            "gt_candidate_index": group if include_gt else None,
        }

    def _write_report(self, report):
        directory = hparams.get("dpo_output_dir", "")
        every = int(hparams.get("dpo_log_every_updates", 1))
        if not directory or every <= 0 or report["update"] % every:
            return
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        with (target / f"rollouts_rank{report['rank']}.jsonl").open("a") as stream:
            stream.write(json.dumps(report, ensure_ascii=False, allow_nan=False) + "\n")

    def run_model(self, sample, infer=False, infer_steps=None):
        if infer:
            return {}, {}
        self.model.eval()
        reference = self._reference()
        data = None
        for attempt in range(int(hparams.get("dpo_max_pair_retries", 3))):
            data = self._collect_candidates(sample)
            if int(data["global_count"].item()) > 0:
                break
        else:
            raise RuntimeError("No reliable preference pairs after synchronized fresh rollouts")
        batch, frames, channels = data["winner"].shape
        device = data["winner"].device
        patches = frames // int(hparams.get("ditar_patch_size", 4))
        generator = torch.Generator(device=device).manual_seed((data["seed"] + 4_000_037) % (2**63 - 1))
        flow_noise = torch.randn(batch, frames, channels, device=device, generator=generator)
        # Preserve v8's time sampler while making each rank/update reproducible.
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed((data["seed"] + 8_000_081) % (2**63 - 1))
            flow_t = unwrap_model(self.model).flow_matcher.time_sampler.sample([batch * patches], device).reshape(batch, patches)
        inputs = {key: torch.cat([value, value], dim=0) for key, value in data["conditions"].items()}
        inputs.update({
            "dpo_score": True,
            "lat": torch.cat([data["winner"], data["loser"]], dim=0),
            "lat_lens": torch.cat([data["lat_lens"], data["lat_lens"]]),
            "lm_history_noise": torch.cat([data["history_noise"], data["history_noise"]]),
            "flow_noise": torch.cat([flow_noise, flow_noise]),
            "flow_t": torch.cat([flow_t, flow_t]),
            "dpo_microbatch": int(hparams.get("dpo_train_microbatch", 1)),
        })
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                            enabled=device.type == "cuda", cache_enabled=False):
            reference_losses = reference(inputs)["per_sequence_loss"]
        # Do not reuse detached AMP parameter casts populated during rollout.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda", cache_enabled=False):
            policy_losses = self.model(inputs)["per_sequence_loss"]
        if not torch.isfinite(policy_losses).all() or not torch.isfinite(reference_losses).all():
            raise RuntimeError("Nonfinite flow loss during online DPO")
        dpo, fm, logits = preference_losses(
            policy_losses, reference_losses, data["valid"], float(hparams.get("dpo_beta", 100.0)),
            global_count=data["global_count"], world_size=self._world_size(),
        )
        mask = data["valid"].float()
        denom = mask.sum().clamp_min(1)
        generated_count = data["generated_count"]
        generated_scores = data["scores"][:, :generated_count]
        rows = torch.arange(batch, device=device)
        losses = {
            "dpo_loss": dpo, "fm_loss": fm,
            "bs": batch, "ntokens": data["lat_lens"].sum().detach(),
            "monitor/dpo_logit": (logits.detach() * mask).sum() / denom,
            "monitor/dpo_accuracy": ((logits.detach() > 0).float() * mask).sum() / denom,
            "monitor/reward_total": generated_scores.mean(),
            "monitor/reward_winner": data["scores"].max(dim=1).values.mean(),
            "monitor/reward_loser": data["scores"][rows, data["loser_indices"]].mean(),
            "monitor/reward_gap": data["gaps"].mean(),
            "monitor/valid_pairs": mask.sum(),
            "monitor/global_valid_pairs": data["global_count"],
            "monitor/generated_candidates": data["scores"].new_tensor(generated_scores.numel()),
            "monitor/total_candidates": data["scores"].new_tensor(data["scores"].numel()),
            "monitor/rollout_seconds": data["scores"].new_tensor(data["seconds"]),
            "monitor/reference_fm": (reference_losses[:batch] * mask).sum() / denom,
        }
        for key in ("energy_cc", "imagebind_cosine", "audiobox_pq", "energy_coverage", "energy_invalid_generated"):
            values = [float(row[key]) for records in data["records"] for row in records[:generated_count] if key in row]
            if values:
                losses[f"monitor/{key}"] = data["scores"].new_tensor(values).mean()
        if data["gt_candidate_index"] is not None:
            gt_index = data["gt_candidate_index"]
            gt_winner = data["winner_indices"] == gt_index
            losses["monitor/gt_winner_rate"] = (gt_winner.float() * mask).sum() / denom
            losses["monitor/gt_winner_pairs"] = (gt_winner.float() * mask).sum()
            losses["monitor/gt_reward_total"] = data["scores"][:, gt_index].mean()
            for key in ("energy_cc", "imagebind_cosine", "audiobox_pq", "energy_valid"):
                values = [float(records[gt_index][key]) for records in data["records"]]
                losses[f"monitor/gt_{key}"] = data["scores"].new_tensor(values).mean()
        if device.type == "cuda":
            losses["monitor/peak_memory_gib"] = data["scores"].new_tensor(torch.cuda.max_memory_allocated(device) / 2**30)
        return losses, {"logits": logits.detach()}

    def _training_step(self, sample, batch_idx, optimizer_idx):
        losses, _ = self.run_model(sample)
        return losses["dpo_loss"] + losses["fm_loss"], losses

    def on_before_optimization(self, opt_idx):
        model = unwrap_model(self.model)
        groups = {"lm": [], "dit": [], "history": []}
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            key = "lm" if name.startswith(("lm.", "lm_head.", "lm_postnet.", "video_")) else (
                "history" if name.startswith(("encoder", "lm_history_bottleneck", "dit_history")) else "dit")
            groups[key].append(param.grad.detach().float().norm().square())
        metrics = {}
        for key, values in groups.items():
            norm = torch.stack(values).sum().sqrt() if values else next(model.parameters()).new_zeros(())
            if not torch.isfinite(norm):
                raise RuntimeError(f"Nonfinite gradient norm in {key}")
            metrics[f"monitor/grad_norm_{key}"] = float(norm)
        if self._rank() == 0:
            print(f"| DPO update {int(self.global_step)+1}: backward complete; LM grad={metrics['monitor/grad_norm_lm']:.5g}, DiT grad={metrics['monitor/grad_norm_dit']:.5g}, history grad={metrics['monitor/grad_norm_history']:.5g}", flush=True)
        return metrics

    def validation_step(self, sample, batch_idx):
        # Fixed-condition generation evaluation is separate from checkpointing.
        return {}
