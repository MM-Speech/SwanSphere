import os
import random

from attrdict import AttrDict
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np

from tasks.tts.dataset_utils.dataset_mixin import TTSDatasetMixin, FastDatasetMixin
from utils.commons.base_task_old import BaseTask
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.import_utils import import_module_bystr
from utils.commons.hparams import hparams, set_hparams
from utils.nn.schedulers import WarmupSchedule, CosineSchedule, CosineAnnealingWarmRestartsWithWarmup
from utils.nn.seq_utils import sequence_mask, add_prefix, remove_prefix
from utils.nn.model_utils import print_arch, num_params, unwrap_model
from utils.commons.os_utils import kill_void
from utils.commons.dataset_utils import data_loader, build_dataloader
from utils.commons.trainer import LOCAL_RANK

from modules.tts.scriptspeech.build_model_utils import DiTBuildModelMixin, SemanticLMBuildModelMixin


class ScriptSpeechBaseTask(TTSDatasetMixin, BaseTask):
    def __init__(self):
        self.dataset_cls = import_module_bystr(hparams['dataset_cls'])
        self.val_dataset_cls = import_module_bystr(hparams['val_dataset_cls'])
        self.processer_fn = import_module_bystr(hparams['processer_fn']) if hparams.get('processer_fn') else None
        self.build_fast_dataloader = import_module_bystr(hparams['build_fast_dataloader']) if hparams.get('build_fast_dataloader') else None
        self.hparams = hparams
        self.config = AttrDict(hparams)

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
                Embedding
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

    def run_model(self, sample, infer=False, infer_steps=None):
        model_out = {}
        losses_out = {}

        if infer:
            return losses_out, model_out
        if 'wavs' not in sample:
            return losses_out, model_out

        wavs = sample["wavs"].float()
        wav_lengths = sample["wav_lengths"]
        ctx_wavs = sample["ctx_wavs"].float()
        ctx_mask = sample["ctx_mask"]
        if hparams.get('drop_ref_wav', 0.0) > random.random():
            ctx_mask = torch.zeros_like(ctx_mask)
        text = sample['text']
        lat_lens = wav_lengths // hparams['hop_size'] // hparams['vae_stride']
        device = wavs.device

        # audio tokenize
        if hparams.get('audio_tokenizer', 'glm4v') == 'glm4v':
            from modules.tts.semantic_encoders.glm4_tokenizer.call_utils import extract_speech_token_v2
            semantic_tokens, semantic_mask = extract_speech_token_v2(
                self.audio_tokenizer, self.audio_token_feature_extractor, 
                wavs=wavs, sample_rate=hparams['audio_sample_rate'], wav_lens=wav_lengths, device=device
            )
            semantic_tokens = semantic_tokens.clone().detach()
    
        # audio encode
        with torch.inference_mode():
            lat = self.vae.encode_latent(wavs)
            lat_ctx = self.vae.encode_latent(ctx_wavs)
            lat_ctx = torch.nn.functional.pad(lat_ctx, (0,0,0,lat.size(1)-lat_ctx.size(1)), mode='constant', value=0)

        # text tokenize
        if random.random() < 0.001:
            print('text sample:', text[0])
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
        
        semantic_cfg_mask = torch.rand_like(txt_tokens[:, 0].float())[:, None]
        semantic_cfg_mask = (semantic_cfg_mask < 0.15).long()
        semantic_tokens = semantic_tokens * (1 - semantic_cfg_mask) + self.cfg_mask_audio_token * semantic_cfg_mask

        inputs = {
            "txt_tokens": txt_tokens.long() if not hparams.get('drop_xt', False) else None,
            "txt_lens": txt_lens,
            "txt_mask": txt_mask,
            "lat": lat,
            "lat_lens": lat_lens,
            "lat_ctx": lat_ctx,
            "ctx_mask": ctx_mask,
            "semantic_tokens": semantic_tokens if not hparams.get('drop_st', False) else torch.zeros_like(semantic_tokens),
            "caption_emb": None,
            "caption_lens": None,
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

    
class ScriptSpeechSemanticLMTask(SemanticLMBuildModelMixin, ScriptSpeechBaseTask):
    def load_model(self):
        if hparams.get('load_ckpt', '') != '':
            load_ckpt(self.lm, hparams['load_ckpt'], 'lm', strict=False, mmap=True)

    def build_optimizer(self):
        optimizer = AdamW(unwrap_model(self.lm).parameters(), **self.config.optimizer)
        return optimizer

    def fsdp_optm2model(self):
        return [self.lm]
    
    def fsdp_wrap_policy(self):
        from torch.nn import Linear, Sequential, Conv1d, Conv2d, Embedding
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            model_blocks = (
                Linear,
                Sequential,
                Conv1d,
                Conv2d,
                Embedding,
                Qwen3DecoderLayer
            )
            return recurse or isinstance(module, model_blocks)

        return custom_auto_wrap_policy

    def _training_step(self, sample, batch_idx, optimizer_idx):
        if random.random() < 0.01:
            kill_void()
        loss_output, model_out = self.run_model(sample)
        loss_weights = {
            'ce_loss': 1.0,
        }
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if
                          isinstance(v, torch.Tensor) and v.requires_grad])

        if self.global_step % 100 == 0:
            torch.cuda.empty_cache()

        return total_loss, loss_output
    
    def run_model(self, sample, infer=False, infer_steps=None):
        model_out = {}
        losses_out = {}

        if infer:
            return losses_out, model_out
        if 'wavs' not in sample:
            return losses_out, model_out

        wavs = sample["wavs"].float()
        wav_lengths = sample["wav_lengths"]
        text = sample['text']
        device = wavs.device

        # audio tokenize
        if hparams.get('audio_tokenizer', 'glm4v') == 'glm4v':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                from modules.tts.semantic_encoders.glm4_tokenizer.call_utils import extract_speech_token_v2
                semantic_tokens, semantic_mask = extract_speech_token_v2(
                    self.audio_tokenizer, self.audio_token_feature_extractor, 
                    wavs=wavs, sample_rate=hparams['audio_sample_rate'], wav_lens=wav_lengths, device=device
                )
                semantic_tokens = semantic_tokens.clone().detach()
                semantic_tokens = semantic_tokens + self.speech_start_token
                semantic_lens = semantic_mask.int().sum(1)
    
        # text tokenize
        text = ['<BOT>' + text_ + '<BOS>' for text_ in text]
        if random.random() < 0.001:
            print('text sample:', text[0])
        text_inputs = self.lm_text_tokenizer(text, padding=True, return_tensors="pt").to(device)
        txt_tokens = text_inputs['input_ids']   # [B, T]
        txt_mask = text_inputs['attention_mask'].bool()
        txt_lens = txt_mask.int().sum(1)
        
        input_tokens = add_prefix(txt_tokens, txt_lens, semantic_tokens, semantic_lens, padding_value=self.padding_idx)

        # assign <EOS> token
        input_lens = txt_lens + semantic_lens
        input_tokens = F.pad(input_tokens, (0, 1), value=self.padding_idx)
        input_tokens[torch.arange(input_tokens.shape[0]).to(input_tokens.device), input_lens] = self.eos_idx
        loss_mask = sequence_mask(input_lens)
        input_lens += 1
        
        # check text validation
        if txt_tokens.shape[1] > semantic_tokens.shape[1] * 2:
            print(f'|Warning: text lengths [{txt_tokens.shape[1]}] > wav latent [{semantic_tokens.shape[1] * 2}], clipping...')
            line = txt_tokens[torch.argmax(txt_mask.sum(1))]
            print(self.lm_text_tokenizer.decode(line.detach().cpu().numpy().tolist()))

            line_idxs = np.argsort(txt_lens.cpu().numpy()).tolist()
            for line_idx in reversed(line_idxs):
                if txt_lens[line_idx] > semantic_tokens.shape[1] * 2:
                    loss_mask[line_idx] = 0.0
                else:
                    break

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            lm_outputs = self.lm(
                input_ids=input_tokens,
                attention_mask=sequence_mask(input_lens),
                use_cache=False
            )

        logits = lm_outputs.logits[:, :-1]
        labels = input_tokens[:, 1:]

        if hparams.get('lambda_text', 0.1) >= 0:
            loss = F.cross_entropy(logits.transpose(1, 2), labels, reduction='none')
            effective_n_tokens = 0
            text_loss_mask = sequence_mask(txt_lens-1, maxlen=loss.shape[1])   # must -1, because right shift
            loss[text_loss_mask] = loss[text_loss_mask] * hparams.get('lambda_text', 0.1)
            effective_n_tokens += text_loss_mask.sum() if hparams.get('lambda_text', 0.1) > 0 else 0
            audio_loss_mask = sequence_mask(txt_lens + semantic_lens)
            audio_loss_mask[text_loss_mask] = False
            loss[audio_loss_mask] = loss[audio_loss_mask] * hparams.get('lambda_audio', 1.0)
            effective_n_tokens += audio_loss_mask.sum()
            loss = loss * loss_mask
            loss = loss.sum() / effective_n_tokens
        else:
            logits = remove_prefix(logits, txt_lens-1, semantic_lens + 1)
            labels = remove_prefix(labels[..., None], txt_lens-1, semantic_lens + 1)[..., 0]
            loss_mask = sequence_mask(semantic_lens + 1)
            loss = F.cross_entropy(logits.transpose(1, 2), labels, reduction='none')
            loss = loss * loss_mask
            loss = loss.sum() / loss_mask.sum()

        losses_out['ce_loss'] = loss
        losses_out['bs'] = loss_mask.shape[0]
        losses_out['ntokens'] = loss_mask.sum()
        return losses_out, model_out
        

class ScriptSpeechPromptSemanticLMTask(ScriptSpeechSemanticLMTask):
    def build_optimizer(self):
        if hparams.get('use_ctx_mask', False):
            optimizer = AdamW(list(unwrap_model(self.lm).parameters()) + list(self.mask_proj.parameters()), **self.config.optimizer)
        else:
            optimizer = AdamW(list(unwrap_model(self.lm).parameters()), **self.config.optimizer)
        return optimizer

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

    def run_model(self, sample, infer=False, infer_steps=None):
        model_out = {}
        losses_out = {}
        if infer:
            return losses_out, model_out
        if 'wavs' not in sample:
            return losses_out, model_out
        wavs = sample["wavs"].float()
        wav_lengths = sample["seq_len"]
        if hparams.get('prompt_type', 'local') == 'local':
            text = sample['local_description']
        elif hparams.get('prompt_type', 'local') == 'global':
            text = [glb + asr for glb, asr in zip(sample['global_description'], sample['asr_text'])]
        elif hparams.get('prompt_type', 'local') == 'total':
            text = sample['caption']

        device = wavs.device

        # audio tokenize
        if hparams.get('audio_tokenizer', 'glm4v') == 'glm4v':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                from modules.tts.semantic_encoders.glm4_tokenizer.call_utils import extract_speech_token_v2
                semantic_tokens, semantic_mask = extract_speech_token_v2(
                    self.audio_tokenizer, self.audio_token_feature_extractor,
                    wavs=wavs, sample_rate=hparams['audio_sample_rate'], wav_lens=wav_lengths, device=device
                )
                semantic_tokens = semantic_tokens.clone().detach()
                semantic_tokens = semantic_tokens + self.speech_start_token
                semantic_lens = semantic_mask.int().sum(1)

        # text tokenize
        text = ['<BOT>' + text_ + '<BOS>' for text_ in text]
        text_inputs = self.lm_text_tokenizer(text, padding=True, return_tensors="pt").to(device)
        txt_tokens = text_inputs['input_ids']  # [B, T]
        txt_mask = text_inputs['attention_mask'].bool()
        txt_lens = txt_mask.int().sum(1)

        input_tokens = add_prefix(txt_tokens, txt_lens, semantic_tokens, semantic_lens, padding_value=self.padding_idx)
        # assign <EOS> token
        input_lens = txt_lens + semantic_lens
        input_tokens = F.pad(input_tokens, (0, 1), value=self.padding_idx)
        input_tokens[torch.arange(input_tokens.shape[0]).to(input_tokens.device), input_lens] = self.eos_idx
        ctx_masks = self.generate_mask(input_tokens, start_token=151680, end_token=151681)[:, :, None].to(torch.bfloat16)
        loss_mask = sequence_mask(input_lens)
        input_lens += 1
        # check text validation
        if txt_tokens.shape[1] > semantic_tokens.shape[1] * 8:
            print(
                f'| Warning: text lengths [{txt_tokens.shape[1]}] > wav latent [{semantic_tokens.shape[1] * 4}], clipping...')
            max_idx = torch.argmax(txt_mask.sum(1))
            line = txt_tokens[torch.argmax(txt_mask.sum(1))]
            print(self.lm_text_tokenizer.decode(line.detach().cpu().numpy().tolist()),
                  sample["gemini_version"][max_idx], sample["video_path"][max_idx])

            line_idxs = np.argsort(txt_lens.cpu().numpy()).tolist()
            for line_idx in reversed(line_idxs):
                if txt_lens[line_idx] > semantic_tokens.shape[1] * 2:
                    loss_mask[line_idx] = 0.0
                else:
                    break

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if hparams.get('use_ctx_mask', False):
                input_embs = unwrap_model(self.lm).model.embed_tokens(input_tokens) + self.mask_proj(ctx_masks)
            else:
                input_embs = unwrap_model(self.lm).model.embed_tokens(input_tokens)
            lm_outputs = self.lm(
                inputs_embeds=input_embs,
                attention_mask=sequence_mask(input_lens),
                use_cache=False
            )
        logits = lm_outputs.logits[:, :-1]
        labels = input_tokens[:, 1:]
        # loss = torch.stack([F.cross_entropy(logits[i, txt_lens[i]-1:], labels[i, txt_lens[i]-1:], reduction='mean') for i in range(logits.shape[0])]).mean()
        # loss = F.cross_entropy(logits.transpose(1, 2)[:, txt_lens-1:], labels[:, txt_lens:], reduction='none')

        logits = remove_prefix(logits, txt_lens-1, semantic_lens + 1)
        labels = remove_prefix(labels[..., None], txt_lens-1, semantic_lens + 1)[..., 0]
        loss_mask = sequence_mask(semantic_lens + 1)
        loss = F.cross_entropy(logits.transpose(1, 2), labels, reduction='none')
        loss = loss * loss_mask
        loss = loss.sum() / loss_mask.sum()

        losses_out['ce_loss'] = loss
        losses_out['bs'] = loss_mask.shape[0]
        losses_out['ntokens'] = loss_mask.sum()
        torch.cuda.empty_cache()
        return losses_out, model_out
