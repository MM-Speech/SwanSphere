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
        lc = hparams.get('load_ckpt', '')
        audio = hparams.get('load_ckpt_audio', '')
        speech = hparams.get('load_ckpt_speech', '')
        if audio and speech:
            spec = {'audio': audio, 'speech': speech}
            load_ckpt(self.dit, spec, 'dit', strict=False, mmap=True)
        elif isinstance(lc, (dict, list, tuple)):
            load_ckpt(self.dit, lc, 'dit', strict=False, mmap=True)
        elif isinstance(lc, str) and lc != '':
            load_ckpt(self.dit, lc, 'dit', strict=False, mmap=True)


    def build_optimizer(self):
        
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
        import modules.flow_matching.llama
        import modules.asr.llama.llama_seq2seq
        import modules.tts.llama_dit.llama_ca

        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            model_blocks = (
                # Linear, Sequential, Conv1d, Conv2d, Embedding,
                modules.flow_matching.llama.TransformerBlock,
                modules.asr.llama.llama_seq2seq.DecoderBlock,
                modules.asr.llama.llama_seq2seq.EncoderBlock,
                # modules.tts.llama_dit.llama_moe.TransformerBlock,
                modules.tts.llama_dit.llama_ca.TransformerBlock,
                get_class_from_module("transformers.models.qwen3.modeling_qwen3", "Qwen3DecoderLayer")
            )
            return recurse or isinstance(module, model_blocks)

        return custom_auto_wrap_policy
    
    def _training_step(self, sample, batch_idx, optimizer_idx):
        if self.trainer.proc_rank_local == 0 and random.random() < 0.1:
            kill_void()
        loss_output, model_out = self.run_model(sample)
        loss_weights = {
            'diff_loss': 1.0,
        }
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if
                          isinstance(v, torch.Tensor) and v.requires_grad])

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
        ctx_mask = sample["ctx_mask"]                       # 期望 [B, T_ctx] 或 [B, T_ctx, 1]
        if len(ctx_mask.shape) == 2:
            ctx_mask = ctx_mask[:, :, None]
        text = sample['text']                               # 文本（含 <Audio>/<BGM> 标签，位于末尾）
        captions = sample.get('caption', None)              # "speaker 1: xxx, bgm/audio: yyy"
        caption_audio_list = sample.get('caption_audio', None)  # 纯音频片段的文字描述（不带前缀）
        ph_tokens = sample["ph_tokens"]
        tone_tokens = sample["tone"]
        mel2ph = sample['mel2ph']

        device = wavs.device
        lat_lens = wav_lengths // hparams['hop_size'] // hparams['vae_stride']

        # 关闭英文 tone（保持原逻辑）
        en_tone_idx = ~((tone_tokens == 4) | ((11 <= tone_tokens) & (tone_tokens <= 15)) | (tone_tokens == 0))
        tone_tokens[en_tone_idx] = 3

        # 编码音频
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                lat = self.vae.encode_latent(wavs)             # [B, T_lat, D]
                lat_ctx = self.vae.encode_latent(ctx_wavs)     # ctx_wav 不含 BGM
                # 右侧 0 填充到与 lat 对齐
                if lat_ctx.size(1) < lat.size(1):
                    lat_ctx = torch.nn.functional.pad(lat_ctx, (0, 0, 0, lat.size(1) - lat_ctx.size(1)),
                                                    mode='constant', value=0)
                elif lat_ctx.size(1) > lat.size(1):
                    lat_ctx = lat_ctx[:, :lat.size(1)]

        # 文本 tokenizer（用于条件）
        text_inputs = self.dit_text_tokenizer(text, padding=True, return_tensors="pt").to(device)
        txt_tokens = text_inputs['input_ids']   # [B, T_txt]
        txt_mask = text_inputs['attention_mask'].bool()
        txt_tokens[~txt_mask] = self.cfg_mask_text_token
        txt_lens = txt_mask.int().sum(1)

        # 将 ctx_mask 对齐到 T_lat，并统一为 [B, T_lat, 1] 的 float
        T_lat = lat.size(1)

        def _align_mask(mask_tensor, name: str):
            """将 mask 对齐到 [B, T_lat, 1]，不足右填 0，超出右裁剪；返回 float 张量。"""
            if mask_tensor is None:
                return None
            m = mask_tensor.to(device)
            if len(m.shape) == 2:
                m = m[:, :, None]
            if m.size(1) < T_lat:
                m = torch.nn.functional.pad(m, (0, 0, 0, T_lat - m.size(1)), value=0.0)
            elif m.size(1) > T_lat:
                print_once(f"| WARN {name} length {m.size(1)} > latent length {T_lat}, clipping to T_lat")
                m = m[:, :T_lat]
            if m.size(-1) != 1:
                print_once(f"| WARN {name} last dim {m.size(-1)} != 1, try to squeeze/broadcast to 1")
                m = m[..., :1]
            return m.float()

        ctx_mask = _align_mask(ctx_mask, 'ctx_mask')

        # 标准 loss 区域：仅 target 段（非 ctx 段）
        loss_mask = sequence_mask(lat_lens)[:, :, None].to(device).float() * (1.0 - ctx_mask)

        # Caption encoder（文本 caption）
        cap_input = captions if captions is not None else text
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                caption_embs, caption_mask = self.run_caption_encoder(cap_input, device)   # [B, Lc, D], [B, Lc]
                caption_embs = caption_embs * caption_mask[..., None]
                caption_lens = caption_mask.sum(-1)

        # Caption encoder（音频片段的文字描述 caption_audio）
        B = wavs.size(0)
        if caption_audio_list is None:
            caption_audio_list = [''] * B
        else:
            if not isinstance(caption_audio_list, (list, tuple)):
                caption_audio_list = [str(caption_audio_list)] * B
        # 以 1% 概率仅在本地 rank0 打印当前 batch 第 1 条样本的 text / caption / caption_audio
        if self.trainer.proc_rank_local == 0 and random.random() < 0.001:
            txt0 = text[0] if isinstance(text, (list, tuple)) else text
            if captions is None:
                cap0 = ""
            else:
                cap0 = captions[0] if isinstance(captions, (list, tuple)) else captions
            cap_audio0 = caption_audio_list[0] if isinstance(caption_audio_list, (list, tuple)) and len(caption_audio_list) > 0 else ""
            print(f"| sample[0] text: {txt0}")
            print(f"| sample[0] caption: {cap0}")
            print(f"| sample[0] caption_audio: {cap_audio0}")

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                cap_audio_embs, cap_audio_mask = self.run_caption_encoder(caption_audio_list, device)  # [B, La, D], [B, La]
                cap_audio_embs = cap_audio_embs * cap_audio_mask[..., None]
                cap_audio_lens = cap_audio_mask.sum(-1)

        # 文本长度 > latent 长度的裁剪与 loss_mask 防护
        if txt_tokens.shape[1] > T_lat:
            print(f'|Warning: text lengths [{txt_tokens.shape[1]}] > wav latent [{T_lat}], clipping...')
            line = txt_tokens[torch.argmax(txt_mask.sum(1))]
            print(self.dit_text_tokenizer.decode(line.detach().cpu().numpy().tolist()))

            line_idxs = np.argsort(txt_lens.cpu().numpy()).tolist()
            for line_idx in reversed(line_idxs):
                if txt_lens[line_idx] > T_lat:
                    loss_mask[line_idx] = 0.0
                else:
                    break

            txt_tokens = txt_tokens[:, :T_lat]
            txt_mask = txt_mask[:, :T_lat]
            txt_lens = txt_mask.sum(1).int()

        # CFG masks
        lat_cfg_mask = torch.rand_like(txt_tokens[:, 0].float())[:, None]
        lat_cfg_mask = (lat_cfg_mask < 0.15).long()
        lat_ctx = (lat_ctx * ctx_mask * (1 - lat_cfg_mask)[:, :, None])

        txt_cfg_mask = torch.rand_like(txt_tokens[:, 0].float())[:, None]
        txt_cfg_mask = (txt_cfg_mask < 0.3).long()
        txt_tokens = txt_tokens * (1 - txt_cfg_mask) + self.cfg_mask_text_token * txt_cfg_mask

        caption_cfg_mask = torch.rand_like(caption_embs[:, 0].float())[:, None]
        caption_cfg_mask = (caption_cfg_mask < 0.1).long()
        caption_embs = caption_embs * (1 - caption_cfg_mask)

        # 对 caption_audio 也做 CFG 掩码
        caption_audio_cfg_mask = torch.rand_like(cap_audio_embs[:, 0].float())[:, None]
        caption_audio_cfg_mask = (caption_audio_cfg_mask < 0.1).long()
        cap_audio_embs = cap_audio_embs * (1 - caption_audio_cfg_mask)

        if not hasattr(self, 'cfg_mask_token_phone'):
            self.cfg_mask_token_phone = 302 - 1
        if not hasattr(self, 'cfg_mask_token_tone'):
            self.cfg_mask_token_tone = 32 - 1
        ph_cfg_mask = torch.rand_like(ph_tokens[:, 0].float())[:, None]
        ph_cfg_mask = (ph_cfg_mask < 0.3).long()
        ph_tokens = ph_tokens * (1 - ph_cfg_mask) + self.cfg_mask_token_phone * ph_cfg_mask
        tone_tokens = tone_tokens * (1 - ph_cfg_mask) + self.cfg_mask_token_tone * ph_cfg_mask

        # （可选）spk_ids：若数据没提供 spk_mask 就跳过
        spk_ids = sample.get('spk_mask', None)
        if spk_ids is not None:
            spk_ids = spk_ids.long().to(device)
            if spk_ids.shape != ph_tokens.shape:
                print_once(f"| WARN spk_mask shape {spk_ids.shape} != ph_tokens {ph_tokens.shape}, ignore spk_ids")
                spk_ids = None
            else:
                spk_cfg_mask = torch.rand_like(spk_ids[:, 0].float())[:, None]
                spk_cfg_mask = (spk_cfg_mask < 0.15).long()
                spk_ids = spk_ids * (1 - spk_cfg_mask)

        # 组装输入（已移除 audio_mask）
        inputs = {
            "phone": ph_tokens,
            "tone": tone_tokens,
            "txt_tokens": txt_tokens.long(),
            "txt_lens": txt_lens,
            "txt_mask": txt_mask,
            "lat": lat,
            "lat_lens": lat_lens,
            "lat_ctx": lat_ctx,                  # ctx 段来自无 BGM 的 ctx_wav
            "ctx_mask": ctx_mask,                # 允许全 0
            "caption_emb": caption_embs,         # 文本 caption
            "caption_lens": caption_lens,
            "caption_audio_emb": cap_audio_embs, # 音频片段的文字描述
            "caption_audio_lens": cap_audio_lens,
            "mel2ph": mel2ph,
        }
        if spk_ids is not None:
            inputs["spk_ids"] = spk_ids
        if hparams.get('use_sparse_dur', False):
            inputs['mel2ph_sparse'] = sample['mel2ph_sparse']

        # 前向 & 损失
        if not infer:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                model_outputs, target = self.dit(inputs)

            # 仅用 loss_mask（非 ctx 段）做均值
            loss_raw = F.mse_loss(model_outputs.float(), target.float(), reduction='none')  # [B, T, D]
            denom = loss_mask.sum().clamp_min(1.0)                                          # 防 div0
            loss = (loss_raw * loss_mask).sum() / denom / target.shape[-1]

            losses_out['diff_loss'] = loss
            losses_out['bs'] = loss_mask.shape[0]
            losses_out['ntokens'] = sum(lat_lens)
            return losses_out, model_out
        else:
            return losses_out, model_out