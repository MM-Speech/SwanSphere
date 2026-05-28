import os
import random
import re

from attrdict import AttrDict
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np

from utils.commons.base_task import BaseTask
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.import_utils import import_module_bystr, get_class_from_module
from utils.commons.hparams import hparams, set_hparams
from utils.commons.os_utils import kill_void
from utils.commons.dataset_utils import data_loader, build_dataloader
from utils.commons.trainer import LOCAL_RANK
from utils.commons.io import print_once
from utils.nn.schedulers import WarmupSchedule, CosineSchedule, CosineAnnealingWarmRestartsWithWarmup
from utils.nn.seq_utils import sequence_mask, add_prefix
from utils.nn.model_utils import print_arch, num_params, unwrap_model
from utils.nn.ema import EMAModel, restore_ema

from modules.tts.scriptspeech.build_model_utils import build_vae, build_qwen3, shard_model_in_node, DiTBuildModelMixinV2
from tasks.tts.dataset_utils.dataset_mixin import FastDatasetMixin, TTSDatasetMixin
from tasks.tts.scriptspeech_task import ScriptSpeechBaseTask
import json
import time
import math
from pathlib import Path
import torch.distributed as dist


class MegaTTSDiTTask(DiTBuildModelMixinV2, ScriptSpeechBaseTask):
    def build_model(self):
        self._build_model()
        self.vae.to(self.trainer.device)

        if hparams.get('use_ema', False):
            print_once('| Building EMA model...')
            self.model_ema = EMAModel(
                unwrap_model(self.dit).parameters(),
                decay=hparams.get('ema_decay', 0.9999),
                update_after_step=hparams.get('ema_update_after_step', 100),
            )
            self.model_ema.to(self.trainer.device)
            
            return {'trainable': [self.dit], 'others': [self.model_ema]}

        return {'trainable': [self.dit], 'others': []}

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        super().on_after_optimization(epoch, batch_idx, optimizer, optimizer_idx)
        if hparams.get('use_ema', False):
            self.model_ema.step(unwrap_model(self.dit).parameters())
    
    def load_model(self):
        if hparams.get('load_ckpt', '') != '':
            load_ckpt(self.dit, hparams['load_ckpt'], 'dit', strict=False, mmap=True)

    def build_optimizer(self):
        # optimizer = AdamW(unwrap_model(self.dit).parameters(), **self.config.optimizer)
        
        decay_params = []
        no_decay_params = []
        for name, param in unwrap_model(self.dit).named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() == 1 or "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
                
        print_once(f"| Weight decay is canceld for {len(no_decay_params)} params")

        optimizer_groups = [
            {'params': decay_params, 'weight_decay': self.config.optimizer.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_groups, **self.config.optimizer)

        return optimizer

    def fsdp_optm2model(self):
        return [self.dit]

    def fsdp_wrap_policy(self):
        from torch.nn import Linear, Sequential, Conv1d, Conv2d, Embedding

        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            model_blocks = (
                Linear,
                Sequential,
                Conv1d,
                Conv2d,
                Embedding,
                get_class_from_module("transformers.models.qwen3.modeling_qwen3", "Qwen3DecoderLayer")
            )
            return recurse or isinstance(module, model_blocks)

        return custom_auto_wrap_policy
    
    def _training_step(self, sample, batch_idx, optimizer_idx):
        """
        修改版：首个 batch 直接 dump 到 'debug' 目录：
        - 将每条 wav 写成 16-bit PCM WAV 文件（也包含 ctx_wavs，如果有）
        - 其余字段写入 JSON
        - 然后立刻退出进程
        后续 batch（如果不退出）仍按原逻辑训练
        """
        # 仅在第一个 batch 执行一次
        if not getattr(self, "_first_batch_dumped", False) and batch_idx == 0:
            try:
                import sys, os, json, wave
                from pathlib import Path
                import numpy as np
                import torch

                def _to_serializable(o):
                    if isinstance(o, torch.Tensor):
                        return o.detach().cpu().tolist()
                    try:
                        import numpy as _np
                        if isinstance(o, _np.ndarray):
                            return o.tolist()
                        if isinstance(o, (_np.floating, _np.integer)):
                            return o.item()
                    except Exception:
                        pass
                    if isinstance(o, (str, int, float, bool)) or o is None:
                        return o
                    if isinstance(o, (list, tuple)):
                        return [_to_serializable(x) for x in o]
                    if isinstance(o, dict):
                        return {k: _to_serializable(v) for k, v in o.items()}
                    # 兜底：转成字符串
                    return str(o)

                def _write_wav(wav_t: torch.Tensor, path: Path, sr: int):
                    w = wav_t.detach().cpu().numpy()
                    # 统一到 (T, C) 结构，C 缺省为 1
                    if w.ndim == 1:
                        w = w[:, None]  # (T, 1)
                    elif w.ndim == 2:
                        # 可能是 (C, T)，尽量识别成 (T, C)
                        if w.shape[0] <= 8 and w.shape[0] < w.shape[1]:
                            w = w.T  # (T, C)
                    else:
                        # 非常规形状，展开到单声道
                        w = w.reshape(-1, 1)

                    # 限幅到 [-1, 1]，转 int16
                    w = np.clip(w, -1.0, 1.0)
                    w_i16 = (w * 32767.0).astype(np.int16)

                    with wave.open(str(path), "wb") as wf:
                        wf.setnchannels(w_i16.shape[1])
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(int(sr))
                        wf.writeframes(w_i16.tobytes())

                sr = int(hparams.get("audio_sample_rate", hparams.get("sampling_rate", 24000)))
                rank = getattr(self.trainer, "proc_rank_local", 0)

                dump_root = Path("debug")
                dump_dir = dump_root / f"proc{rank:02d}_batch{batch_idx:04d}"
                dump_dir.mkdir(parents=True, exist_ok=True)

                # 写 wavs
                wav_files = []
                if "wavs" in sample and isinstance(sample["wavs"], torch.Tensor):
                    B = sample["wavs"].shape[0]
                    for i in range(B):
                        fn = dump_dir / f"wav_{i:02d}.wav"
                        _write_wav(sample["wavs"][i].float(), fn, sr)
                        wav_files.append(fn.name)

                # 写 ctx_wavs
                ctx_wav_files = []
                if "ctx_wavs" in sample and isinstance(sample["ctx_wavs"], torch.Tensor):
                    Bc = sample["ctx_wavs"].shape[0]
                    for i in range(Bc):
                        fn = dump_dir / f"ctx_wav_{i:02d}.wav"
                        _write_wav(sample["ctx_wavs"][i].float(), fn, sr)
                        ctx_wav_files.append(fn.name)

                # 其余字段写 json
                meta = {}
                for k, v in sample.items():
                    if k in ("wavs", "ctx_wavs"):
                        continue
                    meta[k] = _to_serializable(v)

                meta["__dump_info__"] = {
                    "audio_sample_rate": sr,
                    "wav_files": wav_files,
                    "ctx_wav_files": ctx_wav_files,
                    "rank": rank,
                    "batch_idx": int(batch_idx),
                }

                with open(dump_dir / "batch_meta.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

                print_once(f"| Dumped first batch to: {dump_dir.resolve()}")
            finally:
                # 立刻退出（所有 rank 都会在各自首个 batch 退出）
                os._exit(0)

        # ====== 正常训练路径（非首个 batch）======
        if self.trainer.proc_rank_local == 0 and random.random() < 0.1:
            kill_void()
        loss_output, model_out = self.run_model(sample)
        loss_weights = {
            "diff_loss": 1.0,
        }
        total_loss = sum(
            [loss_weights.get(k, 1) * v for k, v in loss_output.items()
            if isinstance(v, torch.Tensor) and v.requires_grad]
        )
        return total_loss, loss_output


    def run_caption_encoder(self, captions, device):
        inputs = self.caption_tokenizer(
            captions,
            padding=True,
            return_tensors="pt",
        )
        input_ids, attention_masks = inputs.input_ids.to(device), inputs.attention_mask.to(device)
        encoder_hidden_states = self.caption_encoder(
            input_ids, return_dict=False,
            attention_mask=attention_masks,
        )[0]
        return encoder_hidden_states, attention_masks

    def run_model(self, sample, infer=False, infer_steps=None):
        model_out = {}
        losses_out = {}
        if infer:
            return losses_out, model_out
        if 'wavs' not in sample:
            return losses_out, model_out

        wavs = sample["wavs"].float()
        wav_lengths = sample["wav_lengths"]
        ctx_wavs = sample["ctx_wavs"]
        ctx_mask = sample["ctx_mask"]
        if len(ctx_mask.shape) == 2:
            ctx_mask = ctx_mask[:, :, None]
        text = sample['text']
        ph_tokens = sample["ph_tokens"]
        tone_tokens = sample["tone"]
        mel2ph = sample['mel2ph']

        lat_lens = wav_lengths // hparams['hop_size'] // hparams['vae_stride']
        # lat_lens = ((mel2ph > 0).sum(-1) // hparams['vae_stride']).long()

        device = wavs.device

        # Disable the English tone (set them to 3)
        en_tone_idx = ~((tone_tokens == 4) | ( (11 <= tone_tokens) & (tone_tokens <= 15)) | (tone_tokens == 0))
        tone_tokens[en_tone_idx] = 3

        # audio encode
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                lat = self.vae.encode_latent(wavs)
                lat_ctx = self.vae.encode_latent(ctx_wavs)
                lat_ctx = torch.nn.functional.pad(lat_ctx, (0,0,0,lat.size(1)-lat_ctx.size(1)), mode='constant', value=0)

        if random.random() < 0.001:
            print('| text sample', text[0])

        # text tokenize
        text_inputs = self.dit_text_tokenizer(text, padding=True, return_tensors="pt").to(device)
        txt_tokens = text_inputs['input_ids']   # [B, T]
        txt_mask = text_inputs['attention_mask'].bool()
        txt_tokens[~txt_mask] = self.cfg_mask_text_token
        txt_lens = txt_mask.int().sum(1)
        loss_mask = sequence_mask(lat_lens)[:, :, None] * (1-ctx_mask)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                caption_embs, caption_mask = self.run_caption_encoder(text, device)
                caption_embs = caption_embs * caption_mask[..., None]
                caption_lens = caption_mask.sum(-1)

        # check text validation
        if txt_tokens.shape[1] > lat.shape[1]:
            print(f'|Warning: text lengths [{txt_tokens.shape[1]}] > wav latent [{lat.shape[1]}], clipping...')
            line = txt_tokens[torch.argmax(txt_mask.sum(1))]
            print(self.dit_text_tokenizer.decode(line.detach().cpu().numpy().tolist()))

            line_idxs = np.argsort(txt_lens.cpu().numpy()).tolist()
            for line_idx in reversed(line_idxs):
                if txt_lens[line_idx] > lat.shape[1]:
                    loss_mask[line_idx] = 0.0
                else:
                    break

            txt_tokens = txt_tokens[:, :lat.shape[1]]
            txt_mask = txt_mask[:, :lat.shape[1]]
            txt_lens = txt_mask.sum(1).int()
        
        # CFG Mask
        lat_cfg_mask = torch.rand_like(txt_tokens[:, 0].float())[:, None]
        lat_cfg_mask = (lat_cfg_mask < 0.15).long()
        lat_ctx = (lat_ctx * ctx_mask * (1 - lat_cfg_mask)[:, :, None])

        txt_cfg_mask = torch.rand_like(txt_tokens[:, 0].float())[:, None]
        txt_cfg_mask = (txt_cfg_mask < 0.15).long()
        txt_tokens = txt_tokens * (1 - txt_cfg_mask) + self.cfg_mask_text_token * txt_cfg_mask
        
        caption_cfg_mask = torch.rand_like(caption_embs[:, 0].float())[:, None]
        caption_cfg_mask = (caption_cfg_mask < 0.15).long()
        caption_embs = caption_embs * (1 - caption_cfg_mask)
        
        if not hasattr(self, 'cfg_mask_token_phone'):
            self.cfg_mask_token_phone = 302 - 1
        if not hasattr(self, 'cfg_mask_token_tone'):
            self.cfg_mask_token_tone = 32 - 1
        ph_cfg_mask = torch.rand_like(ph_tokens[:, 0].float())[:, None]
        ph_cfg_mask = (ph_cfg_mask < 0.15).long()
        ph_tokens = ph_tokens * (1 - ph_cfg_mask) + self.cfg_mask_token_phone * ph_cfg_mask
        tone_tokens = tone_tokens * (1 - ph_cfg_mask) + self.cfg_mask_token_tone * ph_cfg_mask
        
        # 1) 取并校验 dtype/shape
        spk_ids = sample.get('spk_mask', None)
        assert spk_ids is not None, "batch 里没有 spk_mask"
        spk_ids = spk_ids.long().to(wavs.device)
        assert spk_ids.shape == ph_tokens.shape, f"spk_mask 和 ph_tokens 形状不一致: {spk_ids.shape} vs {ph_tokens.shape}"

        # 2) CFG 保持一致
        spk_cfg_mask = torch.rand_like(spk_ids[:, 0].float())[:, None]
        spk_cfg_mask = (spk_cfg_mask < 0.15).long()
        spk_ids = spk_ids * (1 - spk_cfg_mask)

        # 3) 作为输入传给模型
        inputs = {
            "phone": ph_tokens,
            "tone": tone_tokens,
            "txt_tokens": txt_tokens.long(),
            "txt_lens": txt_lens,
            "txt_mask": txt_mask,
            "lat": lat,
            "lat_lens": lat_lens,
            "lat_ctx": lat_ctx,
            "ctx_mask": ctx_mask,
            "caption_emb": caption_embs,
            "caption_lens": caption_lens,
            "mel2ph": mel2ph,
            "spk_ids": spk_ids,           # ← 新增
        }

        if hparams.get('use_sparse_dur', False):
            inputs['mel2ph_sparse'] = sample['mel2ph_sparse']

        if not infer:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                model_outputs, target = self.dit(inputs)

            loss = F.mse_loss(model_outputs.float(), target.float(), reduction='none')
            loss = (loss * loss_mask).sum() / loss_mask.sum() / target.shape[-1]
            losses_out['diff_loss'] = loss
            losses_out['bs'] = loss_mask.shape[0]
            losses_out['ntokens'] = sum(lat_lens)
            return losses_out, model_out
        else:
            return losses_out, model_out