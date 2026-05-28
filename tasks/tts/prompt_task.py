import os
import re
import soundfile as sf
import random
from attrdict import AttrDict
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
from utils.commons.base_task_old import BaseTask
from utils.commons.ckpt_utils import load_ckpt, get_last_checkpoint, load_ckpt_moe
from utils.commons.import_utils import import_module_bystr, get_class_from_module
from utils.commons.hparams import hparams
from utils.commons.os_utils import kill_void
from utils.commons.io import print_once
from utils.commons.dataset_utils import data_loader, build_dataloader
from utils.commons.trainer import LOCAL_RANK
from utils.commons.tensor_utils import move_to_cpu, convert_to_np
from utils.nn.schedulers import WarmupSchedule, CosineSchedule, CosineAnnealingWarmRestartsWithWarmup
from utils.nn.model_utils import print_arch, num_params, unwrap_model, freeze_by_module_name
from utils.nn.ema import EMAModel, restore_ema
from utils.nn.seq_utils import sequence_mask, add_prefix

from modules.tts.scriptspeech.build_model_utils import build_vae, build_qwen3, shard_model_in_node, DiTBuildModelMixin
from tasks.tts.task_utils.prompttts_task_utils import build_audio_mask_from_ids
from tasks.tts.scriptspeech_task import ScriptSpeechBaseTask
from tasks.tts.dataset_utils.dataset_mixin import FastDatasetMixin, TTSDatasetMixin


class PromptBaseTask(FastDatasetMixin, ScriptSpeechBaseTask):
    def __init__(self):
        self.dataset_cls = import_module_bystr(hparams['dataset_cls'])
        if hparams['use_audio_dataset']:
            self.val_dataset_cls = import_module_bystr(hparams['val_dataset_cls'])
            self.processer_fn = import_module_bystr(hparams['processer_fn'])
            self.build_fast_dataloader = import_module_bystr(hparams['build_fast_dataloader'])
            self.train_dataloader = TTSDatasetMixin.train_dataloader.__get__(self)
        else:
            self.train_dataloader = FastDatasetMixin.train_dataloader.__get__(self)
            self.test_dataloader = FastDatasetMixin.test_dataloader.__get__(self)
            self.val_dataloader = FastDatasetMixin.val_dataloader.__get__(self)
        self.hparams = hparams
        self.config = AttrDict(hparams)

        super().__init__()

    def fsdp_wrap_policy(self):
        from torch.nn import Linear, Sequential, Conv1d, Conv2d, Embedding
        from modules.asr.llama.llama import TransformerBlock
        from modules.tts.llama_dit.llama_prompt import TransformerBlock as TransformerBlock_ca
        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            model_blocks = (
                TransformerBlock,
                TransformerBlock_ca,
                get_class_from_module("transformers.models.qwen3.modeling_qwen3", "Qwen3DecoderLayer")
            )
            return recurse or isinstance(module, model_blocks)

        return custom_auto_wrap_policy

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        super().on_after_optimization(epoch, batch_idx, optimizer, optimizer_idx)
        if hparams.get('use_ema', False):
            # for tgt, src in zip(unwrap_model(self.ema_model).parameters(), unwrap_model(self.dit).parameters()):
            #     tgt.data.lerp_(src.data.to(tgt), 1 - self.config.ema_decay)
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
        if not hparams.get('disable_weight_decay_on_bias_and_norm_and_embed', False):
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

    def fsdp_optm2model(self):
        return [self.dit]


class PromptDiTTask(DiTBuildModelMixin, PromptBaseTask):

    def build_model(self):
        self._build_model()
        self.vae.to(self.trainer.device)
        self.vae = torch.compile(self.vae, mode='max-autotune')
        
        if hparams.get('use_ema', False):
            print_once(f'| Building EMA model with decay={self.config.ema_decay} ...')
            self.ema_model = deepcopy(self.dit)
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad = False
            self.ema_model.to(self.trainer.device)
            
            return {'trainable': [self.dit, self.ema_model], 'others': []}

        return {'trainable': [self.dit], 'others': []}

    def load_model(self):
        ckpt_path = hparams.get("load_ckpt", "")
        ckpt_audio = hparams.get("load_ckpt_audio", "")

        if ckpt_path == "":
            return

        if hparams.get("use_moe_ffn", False):
            load_ckpt_moe(
                self.dit,
                ckpt_base_dir=ckpt_path,
                ckpt_audio_base_dir=ckpt_audio,
                model_name="dit",
                strict=False,
                mmap=True,
            )
            if hparams.get("use_ema", False):
                self.ema_model.load_state_dict(self.dit.state_dict(), strict=False)
        else:
            if ckpt_path != "":
                load_ckpt(self.dit, ckpt_path, "dit", strict=False, mmap=True)
                if hparams.get("use_ema", False):
                    load_ckpt(self.ema_model, ckpt_path, "dit", strict=False, mmap=True)

    def _training_step(self, sample, batch_idx, optimizer_idx):
        if self.trainer.proc_rank_local == 0 and random.random() < 0.0001:
            kill_void()
        loss_output, model_out = self.run_model(sample)

        # === moe_aux_loss 的权重，线性从1到0 ===
        base_moe_w = hparams.get('moe_aux_loss_weight', 1.0)
        anneal_steps = hparams.get('moe_aux_anneal_steps', 200000)
        if anneal_steps > 0 and base_moe_w > 0:
            ratio = max(0.0, 1.0 - min(self.global_step, anneal_steps) / float(anneal_steps))
            moe_w = base_moe_w * ratio
        else:
            moe_w = base_moe_w

        loss_weights = {
            'diff_loss': 1.0,
            'moe_aux_loss': moe_w,
        }
        # =========================================

        total_loss = sum([
            loss_weights.get(k, 1.0) * v
            for k, v in loss_output.items()
            if isinstance(v, torch.Tensor) and v.requires_grad
        ])

        if self.trainer.proc_rank == 0 and self.global_step < 10:
            save_dir = f'{hparams["work_dir"]}/sample_batches/step_{self.global_step}'
            os.makedirs(save_dir, exist_ok=True)
            sample = convert_to_np(move_to_cpu(sample))
            for i in range(sample['nsamples']):
                sf.write(f"{save_dir}/{i}.wav", sample['wavs'][i, :sample['wav_lengths'][i]], 24000, 'PCM_16')
                sf.write(f"{save_dir}/{i}_ctx.wav", sample['ctx_wavs'][i, :sample['wav_lengths'][i]], 24000, 'PCM_16')
                np.save(f"{save_dir}/{i}.npy", sample, allow_pickle=True)
            del sample['wavs']
            del sample['ctx_wavs']

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
        if hparams.get('use_caption_text_mark', False):
            caption_text_mark = build_audio_mask_from_ids(
                input_ids=input_ids,
                attention_mask=attention_masks,
                tokenizer=self.goku_tokenizer,
            )  
        else:
            caption_text_mark = None

        return encoder_hidden_states, caption_text_mark, attention_masks

    def run_model(self, sample, infer=False, infer_steps=None):
        model_out = {}
        losses_out = {}

        # 推理时外面自己走 inference，这里直接返回空
        if infer:
            return losses_out, model_out
        if 'wavs' not in sample:
            return losses_out, model_out

        wavs = sample["wavs"].float()
        wav_lengths = sample["wav_lengths"]
        text = sample['text']

        captions = None
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
                    lat_ctx = torch.nn.functional.pad(
                        lat_ctx, (0, 0, 0, lat.size(1) - lat_ctx.size(1)),
                        mode='constant', value=0
                    )
                else:
                    lat_ctx = torch.zeros_like(lat)

        if random.random() < 0.001:
            print('| text sample', text[0])

        # text tokenize
        if 'txt_tokens' not in sample:
            text_inputs = self.dit_text_tokenizer(text, padding=True, return_tensors="pt").to(device)
            txt_tokens = text_inputs['input_ids']   # [B, T]
            txt_mask = text_inputs['attention_mask'].bool()
            txt_tokens[~txt_mask] = self.cfg_mask_text_token
            txt_lens = txt_mask.int().sum(1)
        else:
            txt_tokens = sample['txt_tokens']
            txt_lens = sample['txt_lengths']
            txt_mask = sequence_mask(txt_lens, maxlen=txt_tokens.shape[1])
            txt_tokens[~txt_mask] = self.cfg_mask_text_token

        loss_mask = sequence_mask(lat_lens)[:, :, None] * (1 - ctx_mask)

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

        # CFG Mask on latent context
        lat_cfg_mask = torch.rand_like(lat_ctx[:, :1, 0].float())
        lat_cfg_mask = (lat_cfg_mask < 0.15).long()
        lat_ctx = (lat_ctx * ctx_mask * (1 - lat_cfg_mask)[:, :, None])

        # CFG Mask on text tokens
        txt_cfg_mask = torch.rand_like(txt_tokens[:, :1].float())
        txt_cfg_mask = (txt_cfg_mask < 0.15).long()
        txt_tokens = txt_tokens * (1 - txt_cfg_mask) + self.cfg_mask_text_token * txt_cfg_mask

        if hparams.get('use_caption', False):
            if hparams.get('use_global', False) and use_ref:
                captions = sample['local']       # task1: ctx + local
            else:
                captions = sample['caption']

        # ------------ encode caption ------------
        text_embs = None
        caption_text_mark = None
        caption_lens = None

        if captions is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                    text_embs, caption_text_mark, text_att_mask = self.run_goku_text_encoder(captions)
                    text_embs = text_embs * text_att_mask[..., None]
                    text_cfg_mask = torch.rand_like(text_embs[:, :1].float())
                    text_cfg_mask = (text_cfg_mask < 0.15).long()
                    text_embs = text_embs * (1 - text_cfg_mask)
                    caption_lens = text_att_mask.sum(-1)

        inputs = {
            "txt_tokens": txt_tokens.long() if not hparams.get('drop_xt', False) else None,
            "txt_lens": txt_lens,
            "txt_mask": txt_mask,
            'spk_mask': sample['spk_mask'] if 'spk_mask' in sample else None,
            "lat": lat,
            "lat_lens": lat_lens,
            "lat_ctx": lat_ctx,
            "ctx_mask": ctx_mask,
            "caption_emb": text_embs,
            "caption_lens": caption_lens,
            "caption_text_mark": caption_text_mark,
            "vad_mask": sample['vad_mask'].float() if ('vad_mask' in sample and sample['vad_mask'] is not None) else None,
        }

        if not infer:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                if hparams.get('use_moe_ffn', False):
                    model_outputs, target, moe_aux = self.dit(inputs)
                else:
                    model_outputs, target = self.dit(inputs)
                    moe_aux = None

            # 主 diffusion loss
            loss = F.mse_loss(model_outputs.float(), target.float(), reduction='none')
            loss = (loss * loss_mask).sum() / loss_mask.sum() / target.shape[-1]
            losses_out['diff_loss'] = loss

            # MoE aux loss（未乘权重，权重在 _training_step 里给）
            if moe_aux is not None:
                losses_out['moe_aux_loss'] = moe_aux

            losses_out['bs'] = loss_mask.shape[0]
            losses_out['ntokens'] = sum(lat_lens)
            return losses_out, model_out
        else:
            return losses_out, model_out
