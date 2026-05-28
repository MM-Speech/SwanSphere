import os
import random
import re

from attrdict import AttrDict
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np

from tasks.tts.dataset_utils.dataset_mixin import TTSDatasetMixin
from utils.commons.base_task import BaseTask
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.import_utils import import_module_bystr, get_class_from_module
from utils.commons.hparams import hparams, set_hparams
from utils.nn.schedulers import WarmupSchedule, CosineSchedule, CosineAnnealingWarmRestartsWithWarmup
from utils.nn.seq_utils import sequence_mask, add_prefix
from utils.nn.model_utils import print_arch, num_params, unwrap_model
from utils.commons.os_utils import kill_void
from tasks.tts.dataset_utils.dataset_mixin import FastDatasetMixin
from utils.commons.dataset_utils import data_loader, build_dataloader
from utils.commons.trainer import LOCAL_RANK

from modules.tts.scriptspeech.build_model_utils import DiTBuildModelMixin, SemanticLMBuildModelMixin


class ScriptSpeechBaseTask(FastDatasetMixin, TTSDatasetMixin, BaseTask):
    def __init__(self):
        self.dataset_cls = import_module_bystr(hparams['dataset_cls'])
        if hparams.get('use_megatts_base_dataset', True):
            self.val_dataset_cls = import_module_bystr(hparams['val_dataset_cls'])
            self.processer_fn = import_module_bystr(hparams['processer_fn'])
            self.build_fast_dataloader = import_module_bystr(hparams['build_fast_dataloader'])
            self.train_dataloader = TTSDatasetMixin.train_dataloader.__get__(self)
        else:
            self.train_dataloader = FastDatasetMixin.train_dataloader.__get__(self)
        self.hparams = hparams
        self.config = AttrDict(hparams)

        if hparams.get('use_global', False) and hparams.get('use_random_global', False):
            with open('egs/tts/datasets/global_captions.txt', 'r', encoding='utf-8') as f:
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

        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            model_blocks = (
                Linear,
                Sequential,
                Conv1d,
                Conv2d,
                Embedding,
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


class ScriptSpeechDiTTask(DiTBuildModelMixin, ScriptSpeechBaseTask):
    def load_model(self):
        if hparams.get('load_ckpt', '') != '':
            load_ckpt(self.dit, hparams['load_ckpt'], 'dit', strict=False, mmap=True)

    def build_optimizer(self):
        optimizer = AdamW(unwrap_model(self.dit).parameters(), **self.config.optimizer)
        return optimizer

    def fsdp_optm2model(self):
        return [self.dit]
    
    def _training_step(self, sample, batch_idx, optimizer_idx):
        if random.random() < 0.01:
            kill_void()
        loss_output, model_out = self.run_model(sample)
        loss_weights = {
            'diff_loss': 1.0,
        }
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if
                          isinstance(v, torch.Tensor) and v.requires_grad])

        return total_loss, loss_output

    def generate_mask(self, input_tokens, start_token=10, end_token=11):
        batch_size, seq_len = input_tokens.shape
        mask = torch.zeros_like(input_tokens, dtype=torch.bool)

        for i in range(batch_size):
            indices = (input_tokens[i] == start_token).nonzero(as_tuple=True)[0]
            for idx in indices:
                # 找到 start_token 后面最近的 end_token
                end_idx = (input_tokens[i, idx + 1:] == end_token).nonzero(as_tuple=True)[0]
                if len(end_idx) > 0:
                    j = idx + 1 + end_idx[0].item()
                    mask[i, idx + 1:j] = 1
        return mask

    def run_sd_text_encoder(self, captions: list):
        special_tokens = self.config.text.special_tokens
        token0 = special_tokens[0].token
        token1 = special_tokens[1].token
        captions = [re.sub(r'<W>(.*?)</W>', rf"{token0}\1{token1}", cur_t) for cur_t in captions]
        captions_out = self.sd_text_encoder(captions, special_tokens)
        captions_cmask = self.generate_mask(captions_out.input_token_ids,
                                            start_token=special_tokens[0].token_id,
                                            end_token=special_tokens[1].token_id)
        return captions_out, captions_cmask

    def run_goku_text_encoder(self, captions: list):
        special_token_ids = self.goku_special_token_ids
        inputs = self.goku_tokenizer(
            captions,
            max_length=hparams['text_max_token_length'],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids, attention_masks = inputs.input_ids.cuda(), inputs.attention_mask.cuda()
        encoder_hidden_states = self.goku_text_encoder(
            input_ids, return_dict=False,
            attention_mask=attention_masks,
        )[0]
        captions_cmask = self.generate_mask(input_ids,
                                            start_token=special_token_ids[0],
                                            end_token=special_token_ids[1])
        return encoder_hidden_states, captions_cmask, attention_masks

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
        if hparams.get('drop_ref_wav', 0.5) > random.random():
            ctx_mask = torch.zeros_like(ctx_mask)
        text = sample['text']

        captions = None
        if 'caption' in sample:
            captions = sample['caption']
        else:
            if hparams.get('use_random_global', False):
                captions = []
                for t_ in text:
                    caption = ''
                    global_prompts = random.choice(self.global_samples)
                    caption = caption + global_prompts
                    # not adapter to local
                    caption = caption + '<W>' + t_ + '</W>'
                    captions.append(caption)

        lat_lens = wav_lengths // hparams['hop_size'] // hparams['vae_stride']
        device = wavs.device

        # audio encode
        with torch.inference_mode():
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

        # get caption
        if captions is not None:
            if 'seedance' in self.config.model_size:
                text_output, captions_cmask = self.run_sd_text_encoder(captions)
                text_embs = text_output.embeddings * text_output.masks[:, :, None]
                text_cfg_mask = torch.rand_like(text_embs[:, 0].float())[:, None]
                text_cfg_mask = (text_cfg_mask < 0.15).long()
                text_embs = text_embs * (1 - text_cfg_mask)
                caption_lens = text_output.masks.sum(-1)
            elif 'goku' in self.config.model_size:
                text_embs, captions_cmask, text_att_mask = self.run_goku_text_encoder(captions)
                text_embs = text_embs * text_att_mask[..., None]
                text_cfg_mask = torch.rand_like(text_embs[:, 0].float())[:, None]
                text_cfg_mask = (text_cfg_mask < 0.15).long()
                text_embs = text_embs * (1 - text_cfg_mask)
                caption_lens = text_att_mask.sum(-1)
            else:
                raise NotImplementedError

        inputs = {
            "txt_tokens": txt_tokens.long() if not hparams.get('drop_xt', False) else None,
            "txt_lens": txt_lens,
            "txt_mask": txt_mask,
            "lat": lat,
            "lat_lens": lat_lens,
            "lat_ctx": lat_ctx,
            "ctx_mask": ctx_mask,
            "caption_emb": torch.cat([text_embs, captions_cmask[:, :, None]], -1) if captions is not None else None, # B, T(150/300?), C(3584)
            "caption_lens": caption_lens if captions is not None else None, # B
            "vad_mask": sample['vad_mask'][..., None]
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
