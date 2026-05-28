import os
import random
import re

from attrdict import AttrDict
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import random_split
import numpy as np
from copy import deepcopy
import soundfile as sf

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
from utils.commons.tensor_utils import move_to_cpu, convert_to_np

from modules.tts.scriptspeech.build_model_utils import build_vae, build_qwen3, shard_model_in_node, DiTBuildModelMixinV2, DiTBuildModelMixinV4
from tasks.tts.dataset_utils.dataset_mixin import FastDatasetMixin, TTSDatasetMixin
from tasks.tts.scriptspeech_task import ScriptSpeechBaseTask
from tasks.tts.task_utils.prompttts_task_utils import build_dialogue_mask_from_ids


class FastDatasetDiTBaseTask(FastDatasetMixin, BaseTask):
    def __init__(self):
        self.dataset_cls = import_module_bystr(hparams['dataset_cls'])
        self.val_dataset_cls = import_module_bystr(hparams['val_dataset_cls'])
        self.hparams = hparams
        self.config = AttrDict(hparams)

        super().__init__()
    
    def build_ema_model(self):
        print_once(f'| Building EMA model with decay={self.config.ema_decay} ...')
        self.ema_model = deepcopy(self.dit)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False
        self.ema_model.to(self.trainer.device)
    
    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        super().on_after_optimization(epoch, batch_idx, optimizer, optimizer_idx)
        if hparams.get('use_ema', False):
            self.ema_update(self.ema_model, self.dit, self.config.ema_decay)
        
    @torch.no_grad()
    def ema_update(self, ema_model, model, decay):
        ema_params = dict(unwrap_model(ema_model).named_parameters())
        for n, p in unwrap_model(model).named_parameters():
            p_ema = ema_params[n]
            src = p.detach()
            if src.dtype != torch.float32:
                src = src.float()
            if p_ema.dtype != torch.float32:
                p_ema.data = p_ema.data.float()
            p_ema.mul_(decay).add_(src.to(p_ema.device), alpha=1.0 - decay)
            
    def load_model(self):
        if hparams.get('load_ckpt', '') != '':
            load_ckpt(self.dit, hparams['load_ckpt'], 'dit', strict=False, mmap=True)
            if hparams.get('use_ema', False):
                load_ckpt(self.ema_model, hparams['load_ckpt'], 'dit', strict=False, mmap=True)
                
    def build_optimizer(self):
        if not hparams.get('disable_weight_decay_on_bias_and_norm_and_embed', True):
            optimizer = AdamW(unwrap_model(self.dit).parameters(), **self.config.optimizer)
        
        else:
            def has_name(names, param_name):
                if not isinstance(names, list):
                    names = [names]
                for name in names:
                    if name in param_name:
                        return True
                return False
                
            decay_params = []
            no_decay_params = []
            for name, param in unwrap_model(self.dit).named_parameters():
                if not param.requires_grad:
                    continue
                if param.dim() == 1 or has_name(['bias', 'norm', 'text_embedder', 'tone_embed', 'ph_embed'], name):
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
    
    def build_scheduler(self, optimizer):
        return CosineAnnealingWarmRestartsWithWarmup(
            optimizer, lr_max=hparams['optimizer']['lr'], warmup_updates=hparams.get('warmup_updates', 5000), 
            total_updates=1000000, initial_period=hparams.get('scheduler_initial_period', 10000), 
            period_mult=hparams.get('scheduler_period_mult', 1.2), lr_min=hparams.get('scheduler_lr_min', 1.0e-5)
        )

    def fsdp_optm2model(self):
        return [self.dit]

    def fsdp_wrap_policy(self):
        from torch.nn import Linear, Sequential, Conv1d, Conv2d, Embedding
        import modules.flow_matching.llama
        import modules.asr.llama.llama_seq2seq
        import modules.tts.llama_dit.llama_ca

        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            model_blocks = (
                # Linear, Sequential, Conv1d, Conv2d, Embedding,
                modules.flow_matching.llama.TransformerBlock,
                modules.asr.llama.llama_seq2seq.DecoderBlock,
                modules.asr.llama.llama_seq2seq.EncoderBlock,
                modules.tts.llama_dit.llama_ca.TransformerBlock,
                get_class_from_module("transformers.models.qwen3.modeling_qwen3", "Qwen3DecoderLayer")
            )
            return recurse or isinstance(module, model_blocks)

        return custom_auto_wrap_policy
    
    def on_epoch_start(self):
        super().on_epoch_start()
        kill_void()

    @torch.no_grad()
    def validation_step(self, sample, batch_idx):
        infer_steps = self.hparams.get('infer_steps', 12)
        outputs = self._validation_step(sample, batch_idx, infer_steps)
        return outputs

    def _validation_step(self, sample, batch_idx, infer_steps):
        outputs = {}
        if self.trainer.proc_rank == 0:
            pass
        return outputs

    @torch.no_grad()
    def test_step(self, sample, batch_idx):
        infer_steps = hparams['infer_steps']
        return self._validation_step(sample, batch_idx, infer_steps)


class PromptAudioTask(DiTBuildModelMixinV2, FastDatasetDiTBaseTask):
    def build_model(self):
        self._build_model()
        self.vae.to(self.trainer.device)

        if hparams.get('use_ema', False):
            self.build_ema_model()
            
            return {'trainable': [self.dit, self.ema_model], 'others': []}

        return {'trainable': [self.dit], 'others': []}
    
    def load_model(self):
        if hparams.get('load_ckpt', '') != '':
            load_ckpt(self.dit, hparams['load_ckpt'], 'dit', strict=False, mmap=True)
            if hparams.get('use_ema', False):
                load_ckpt(self.ema_model, hparams['load_ckpt'], 'dit', strict=False, mmap=True)
                
        if hparams.get('reset_cross_attn', False):
            with torch.no_grad():
                reset_names = []
                for name, module in self.dit.named_modules():
                    if (
                        'caption_proj' in name or
                        'caption_text_mark_embed' in name or
                        'cross_attention' in name or
                        'cross_attention_norm' in name
                    ):
                        if hasattr(module, "reset_parameters"):
                            module.reset_parameters()
                            reset_names.append(name)
                            for p in module.parameters(recurse=True):
                                for optm in self.trainer.optimizers:
                                    st = optm.state.get(p, None)
                                    if st is not None:
                                        st.clear()
                if self.trainer.proc_rank_local == 0:
                    print_once(f"| Resetting following modules:")
                    for name in reset_names:
                        print(f"| - {name}")
        
    def _training_step(self, sample, batch_idx, optimizer_idx):
        """
        修改版：首个 batch 直接 dump 到 'debug' 目录（与 MegaTTSDiTTask 示例一致）：
        - 将每条 wav 写成 16-bit PCM WAV 文件（也包含 ctx_wavs，如果有）
        - 其余字段写入 JSON
        - 然后立刻退出进程
        后续 batch（如果不退出）仍按原逻辑训练
        """
        # 仅在第一个 batch 执行一次
        if not getattr(self, "_first_batch_dumped", False) and batch_idx == 0:
            try:
                import os, json, wave
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
                    # 兜底：转字符串
                    return str(o)

                def _write_wav(wav_t: torch.Tensor, path: Path, sr: int):
                    w = wav_t.detach().cpu().numpy()
                    # 统一为 (T, C)
                    if w.ndim == 1:
                        w = w[:, None]
                    elif w.ndim == 2:
                        # 可能 (C, T)，若 C 很小且 < T，则转置为 (T, C)
                        if w.shape[0] <= 8 and w.shape[0] < w.shape[1]:
                            w = w.T
                    else:
                        w = w.reshape(-1, 1)
                    # 限幅并转 int16
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
                self._first_batch_dumped = True
            finally:
                # 立刻退出（所有 rank 在各自首个 batch 退出）—与示例保持一致
                os._exit(0)

        # ====== 正常训练路径（非首个 batch）======
        if self.trainer.proc_rank_local == 0 and random.random() < 0.1:
            kill_void()
        loss_output, model_out = self.run_model(sample)
        loss_weights = {"diff_loss": 1.0}
        total_loss = sum(
            [loss_weights.get(k, 1) * v for k, v in loss_output.items()
             if isinstance(v, torch.Tensor) and v.requires_grad]
        )
        return total_loss, loss_output

    def run_caption_encoder(self, captions, device):
        bsz = len(captions)
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
        
        if hparams.get('use_caption_text_mark', False):
            caption_text_mark = build_dialogue_mask_from_ids(
                input_ids=input_ids,
                attention_mask=attention_masks,
                tokenizer=self.caption_tokenizer,
            ).to(device)
            
            return encoder_hidden_states, attention_masks, caption_text_mark

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
        ctx_mask = ctx_mask.float()
        if len(ctx_mask.shape) == 2:
            ctx_mask = ctx_mask[:, :, None]
        text = sample['text']
        ph_tokens = sample["ph_tokens"]
        tone_tokens = sample["tone"]
        mel2ph = sample['mel2ph']
        global_prompts = sample['global_prompt']
        local_prompts = sample['local_prompt']
        
        bsz, device = wavs.shape[0], wavs.device

        lat_lens = wav_lengths // hparams['hop_size'] // hparams['vae_stride']

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

        # caption
        captions = []
        for local_prompt, global_prompt in zip(local_prompts, global_prompts):
            global_prompt = '<GPROMPT>' + global_prompt + '</GPROMPT>' if global_prompt != '' else ''
            if not isinstance(local_prompt, str):
                print(f"{local_prompt = }")
                local_prompt = ''
            else:
                local_prompt = local_prompt.replace('<tag>', '<TAG>').replace('</tag>', '</TAG>') if local_prompt != '' else ''
            captions.append(global_prompt + local_prompt)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                caption_embs, caption_mask, caption_text_mark = self.run_caption_encoder(captions, device)
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
        
        # zeroshot_task_indices, prompttts_task_indices, all_task_indices = [list(l) for l in random_split(range(bsz), (0.1, 0.8, 0.1))]
        zeroshot_task_indices, prompttts_task_indices, all_task_indices = [list(l) for l in random_split(range(bsz), (0.01, 0.9, 0.09))]
        if len(prompttts_task_indices) == 0:
            prompttts_task_indices = list(range(bsz))
        ctx_mask[prompttts_task_indices] = 0    # no context
        
        # CFG Mask
        lat_cfg_mask = torch.rand_like(txt_tokens[:, 0].float())[:, None]
        lat_cfg_mask = (lat_cfg_mask < hparams.get('lat_cfg_prob', 0.15)).long()
        lat_ctx = (lat_ctx * ctx_mask * (1 - lat_cfg_mask)[:, :, None])

        # txt_cfg_mask = torch.LongTensor([text_ == '' for text_ in text]).to(device)
        txt_cfg_mask = torch.rand_like(txt_tokens[:, 0].float())[:, None]
        txt_cfg_mask = (txt_cfg_mask < hparams.get('txt_cfg_prob', 0.15)).long()
        txt_tokens = txt_tokens * (1 - txt_cfg_mask) + self.cfg_mask_text_token * txt_cfg_mask
        
        # caption_cfg_mask = torch.LongTensor([caption == '' for caption in captions]).to(device)
        caption_cfg_mask = torch.rand_like(caption_embs[:, 0].float())[:, None]
        caption_cfg_mask = (caption_cfg_mask < hparams.get('caption_cfg_prob', 0.15)).long()
        caption_embs = caption_embs * (1 - caption_cfg_mask)
        
        if not hasattr(self, 'cfg_mask_token_phone'):
            self.cfg_mask_token_phone = 302 - 1
        if not hasattr(self, 'cfg_mask_token_tone'):
            self.cfg_mask_token_tone = 32 - 1
        no_ph_mask = ph_tokens[:, 0] == self.cfg_mask_token_phone
        if no_ph_mask.float().mean() < 0.15:
            ph_cfg_mask = torch.rand_like(ph_tokens[:, 0].float())[:, None]
            ph_cfg_mask = (ph_cfg_mask < hparams.get('ph_cfg_prob', 0.15)).long()
            ph_tokens = ph_tokens * (1 - ph_cfg_mask) + self.cfg_mask_token_phone * ph_cfg_mask
            tone_tokens = tone_tokens * (1 - ph_cfg_mask) + self.cfg_mask_token_tone * ph_cfg_mask
        
        # print(f"{mel2ph.max() = }")
        # print(f"{mel2ph.shape = }")
        # print(f"{ph_tokens.shape = }")

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
            "caption_lens": caption_lens, # B
            "caption_text_mark": caption_text_mark,
            "vad_mask": sample['vad_mask'],
            "mel2ph": mel2ph,
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
            
            # monitor
            losses_out['monitor/orig_loss'] = loss.detach()
            losses_out['monitor/pred'] = (model_outputs.detach() * loss_mask).sum() / loss_mask.sum()
            losses_out['monitor/tgt'] = (target * loss_mask).sum() / loss_mask.sum()
            losses_out['monitor/loss_mask'] = loss_mask.sum() 
            losses_out['monitor/ctx_mask'] = ctx_mask.sum()
            
            if loss_mask.sum() <= 3:
                if self.trainer.proc_rank_local == 0:
                    print(f"\n| CRITICAL: loss_mask.sum() = {loss_mask.sum()}. Resetting loss.")
                losses_out['diff_loss'] = 0.0
            
            return losses_out, model_out
        else:
            return losses_out, model_out

class PromptAudioV4Task(DiTBuildModelMixinV4, PromptAudioTask):
    pass
