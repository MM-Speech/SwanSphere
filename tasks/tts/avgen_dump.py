import os
import random
import re

from attrdict import AttrDict
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np

from tasks.tts.dataset_utils.dataset_mixin import TTSDatasetMixin
from utils.commons.base_task_old import BaseTask
from utils.commons.ckpt_utils import load_ckpt, get_last_checkpoint
from utils.commons.import_utils import import_module_bystr
from utils.commons.hparams import hparams, set_hparams
from utils.nn.schedulers import WarmupSchedule, CosineSchedule, CosineAnnealingWarmRestartsWithWarmup
from utils.nn.seq_utils import sequence_mask, add_prefix
from utils.nn.model_utils import print_arch, num_params, unwrap_model
from utils.commons.os_utils import kill_void
from tasks.tts.dataset_utils.dataset_mixin import FastDatasetMixin
from utils.commons.dataset_utils import data_loader, build_dataloader
from utils.commons.trainer import LOCAL_RANK

from modules.tts.scriptspeech.build_model_utils import DiTBuildModelMixin, SemanticLMBuildModelMixin
from tasks.tts.task_utils.prompttts_task_utils import build_audio_mask_from_ids

def get_class_from_module(module_name, class_name):
    import importlib
    # Import the module dynamically
    module = importlib.import_module(module_name)
    # Get the class from the module
    cls = getattr(module, class_name)
    return cls

class ScriptSpeechBaseTask(FastDatasetMixin, TTSDatasetMixin, BaseTask):
    def __init__(self):
        self.dataset_cls = import_module_bystr(hparams['dataset_cls'])
        if hparams['use_audio_dataset']:
            self.val_dataset_cls = import_module_bystr(hparams['val_dataset_cls'])
            self.processer_fn = import_module_bystr(hparams['processer_fn'])
            self.build_fast_dataloader = import_module_bystr(hparams['build_fast_dataloader'])
            self.train_dataloader = TTSDatasetMixin.train_dataloader.__get__(self)
        else:
            self.train_dataloader = FastDatasetMixin.train_dataloader.__get__(self)
        self.hparams = hparams
        self.config = AttrDict(hparams)

        if hparams.get('use_global', False) and hparams.get('use_random_global', False):
            with open('egs/datasets/global_captions.txt', 'r', encoding='utf-8') as f:
                self.global_samples = [line.strip() for line in f if line.strip()]


        super().__init__()

    def build_scheduler(self, optimizer):
        return CosineAnnealingWarmRestartsWithWarmup(
            optimizer, lr_max=hparams['optimizer']['lr'], warmup_updates=hparams.get('warmup_updates', 5000), 
            total_updates=1000000, initial_period=hparams.get('scheduler_initial_period', 10000), 
            period_mult=hparams.get('scheduler_period_mult', 1.2), lr_min=hparams.get('scheduler_lr_min', 1.0e-5)
        )

    def fsdp_wrap_policy(self):
        from torch.nn import Linear, Sequential, Conv1d, Conv2d, Embedding
        from modules.flow_matching.llama import TransformerBlock
        from modules.tts.llama_dit.llama_avgen import TransformerBlock as TransformerBlock_ca
        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            model_blocks = (
                TransformerBlock,
                TransformerBlock_ca,
                get_class_from_module("transformers.models.qwen2.modeling_qwen2", "Qwen2DecoderLayer")
            )
            return recurse or isinstance(module, model_blocks)

        return custom_auto_wrap_policy

    ##########################
    # training and validation
    ##########################

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


class PromptAudioTask(DiTBuildModelMixin, ScriptSpeechBaseTask):
    
    def load_model(self):
        if hparams.get('load_ckpt', '') != '':
            load_ckpt(self.dit, hparams['load_ckpt'], 'dit', strict=False)
        pretrained_dict = hparams.get('load_pretrained_weights', {})
        if len(pretrained_dict) > 0:
            model_dict = self.dit.state_dict()
            for path, key_list in pretrained_dict.items():
                pretrained_state_dict, last_ckpt_path = get_last_checkpoint(path, map_location="cpu", return_step=False)
                if 'dit' in pretrained_state_dict:
                    pretrained_state_dict = pretrained_state_dict['dit']
                else:
                    pretrained_state_dict = pretrained_state_dict
                for key in key_list:
                    print(f"| loading pretrained weight '*{key}*' from '{last_ckpt_path}'.")
                    new_state_dict_ = {
                        k: v for k, v in pretrained_state_dict.items() if key in k and v.shape == model_dict[k].shape
                    }
                    print('| debug: update keys:', new_state_dict_.keys())
                    model_dict.update(new_state_dict_)
            load_results = self.dit.load_state_dict(model_dict, strict=False)
            missing_keys, unexpected_keys = load_results.missing_keys, load_results.unexpected_keys
            print(f"| Load pretrained_weights Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")

        if hparams.get('train_modules', []):
            # 先冻结所有参数
            for name, param in self.dit.named_parameters():
                param.requires_grad = False

            # 只解冻包含指定关键字的模块
            train_modules = hparams['train_modules']
            for name, param in self.dit.named_parameters():
                if any(k in name for k in train_modules):
                    param.requires_grad = True

    def build_optimizer(self):
        optimizer = AdamW(unwrap_model(self.dit).parameters(), **self.config.optimizer)
        return optimizer

    def fsdp_optm2model(self):
        return [self.dit]
        
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


    def run_goku_text_encoder(self, captions: list):
        inputs = self.goku_tokenizer(
            captions,
            padding=True,  # Dynamic / longest
            truncation=True,
            max_length=hparams['text_max_token_length'],
            return_tensors="pt",
        )

        input_ids, attention_masks = inputs.input_ids.cuda(), inputs.attention_mask.cuda()
        encoder_hidden_states = self.goku_text_encoder(
            input_ids,
            return_dict=False,
            attention_mask=attention_masks,
        )[0]  # [B, T, C]

        # 新的多类掩码（0/1/2）
        dialogue_mask = build_audio_mask_from_ids(
            input_ids=input_ids,
            attention_mask=attention_masks,
            tokenizer=self.goku_tokenizer,
        )  # [B, T], Long

        return encoder_hidden_states, dialogue_mask, attention_masks

    def run_model(self, sample, infer=False, infer_steps=None):
        model_out = {}
        losses_out = {}
        if infer:
            return losses_out, model_out
        if 'wavs' not in sample:
            return losses_out, model_out

        if 'ph_tokens' in sample:
            ph_tokens = sample["ph_tokens"]
            tone_tokens = sample["tone"]
        else:
            ph_tokens = None
            tone_tokens = None
        wavs = sample["wavs"].float()
        wav_lengths = sample["wav_lengths"]

        text = sample['text']

        captions = None
        if 'caption' in sample and hparams.get('use_caption', True):
            captions = sample['caption']

        lat_lens = wav_lengths // hparams['hop_size'] // hparams['vae_stride']
        device = wavs.device

        ctx_wavs = sample["ctx_wavs"]
        ctx_mask = sample["ctx_mask"]
        if len(ctx_mask.shape) == 2:
            ctx_mask = ctx_mask[:, :, None]

        use_ref = True
        if hparams.get('drop_ref_wav', 0.5) > random.random():
            use_ref = False
            ctx_mask = torch.zeros_like(ctx_mask)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                lat = self.vae.encode_latent(wavs)
                if use_ref:
                    lat_ctx = self.vae.encode_latent(ctx_wavs)
                    lat_ctx = torch.nn.functional.pad(lat_ctx, (0,0,0,lat.size(1)-lat_ctx.size(1)), mode='constant', value=0)
                else:
                    lat_ctx = torch.zeros_like(lat)   # 或者按需要构造一个 [B, T, C] 的零张量

        if random.random() < 0.001:
            print('| text sample', text[0])


        # text tokenize
        text_inputs = self.dit_text_tokenizer(text, padding=True, return_tensors="pt").to(device)
        txt_tokens = text_inputs['input_ids']   # [B, T]
        txt_mask = text_inputs['attention_mask'].bool()
        txt_tokens[~txt_mask] = self.cfg_mask_text_token
        txt_lens = txt_mask.int().sum(1)
        loss_mask = sequence_mask(lat_lens)[:, :, None] * (1-ctx_mask)

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
        lat_cfg_mask = torch.rand_like(lat_ctx[:, :1, 0].float())
        lat_cfg_mask = (lat_cfg_mask < 0.15).long()
        lat_ctx = (lat_ctx * ctx_mask * (1 - lat_cfg_mask)[:, :, None])

        txt_cfg_mask = torch.rand_like(txt_tokens[:, :1].float())
        txt_cfg_mask = (txt_cfg_mask < 0.15).long()
        txt_tokens = txt_tokens * (1 - txt_cfg_mask) + self.cfg_mask_text_token * txt_cfg_mask

        # get caption
        if captions is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                    text_embs, captions_cmask, text_att_mask = self.run_goku_text_encoder(captions)
                    # [B, T, C], [B, T], [B, T]
                    text_embs = text_embs * text_att_mask[..., None]
                    text_cfg_mask = torch.rand_like(text_embs[:, :1].float())
                    text_cfg_mask = (text_cfg_mask < 0.15).long()
                    text_embs = text_embs * (1 - text_cfg_mask)
                    caption_lens = text_att_mask.sum(-1)
                    cmask_feat = captions_cmask.to(text_embs.dtype).unsqueeze(-1)  # [B, T, 1]

        inputs = {
            "phone": ph_tokens,
            "tone": tone_tokens,
            "txt_tokens": txt_tokens.long() if not hparams.get('drop_xt', False) else None,
            "txt_lens": txt_lens,
            "txt_mask": txt_mask,
            "lat": lat,
            "lat_lens": lat_lens,
            "lat_ctx": lat_ctx,
            "ctx_mask": ctx_mask,
            "caption_emb": torch.cat([text_embs, cmask_feat], dim=-1) if captions is not None else None, # B, T(150/300?), C(3584)
            "caption_lens": caption_lens if captions is not None else None, # B
            "vad_mask": sample['vad_mask'].float() if sample['vad_mask'] is not None else None
        }

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