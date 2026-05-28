import os
import random
import re
import traceback

from attrdict import AttrDict
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
from copy import deepcopy

from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams
from utils.nn.model_utils import print_arch, num_params, unwrap_model
from utils.commons.os_utils import kill_void
from utils.commons.io import print_once
from utils.nn.seq_utils import sequence_mask
from utils.commons.import_utils import get_class_from_module
from utils.nn.fsdp_utils import shard_model_in_node
from utils.text.text_encoder import TokenTextEncoder
from utils.text import PHONE_VOCAB, TONE_VOCAB

from modules.tts.scriptspeech.build_model_utils import build_qwen3
from tasks.tts.dataset_utils.dataset_mixin import FastDatasetMixin
from tasks.tts.scriptspeech_task import ScriptSpeechBaseTask


class DurLMTask(ScriptSpeechBaseTask):
    def build_model(self):  # interface to BaseTask()
        from modules.tts.ar_dur.dur_lm import build_dur_model
        self.model = build_dur_model(hparams, vocab_size=810, padding_idx=797)
        
        if hparams.get('backbone', 'llama') == 'llama_txtcond_seq2seq':
            caption_encoder_name = hparams.get('cond_encoder_name')
            self.caption_encoder, self.caption_tokenizer = build_qwen3(hparams, caption_encoder_name)
            self.caption_encoder = self.caption_encoder.model
            self.caption_encoder.requires_grad_(False).eval()
            if hparams.get('use_fsdp', True):
                transformer_cls = get_class_from_module("transformers.models.qwen3.modeling_qwen3", "Qwen3DecoderLayer")
                self.caption_encoder = shard_model_in_node(self.caption_encoder, transformer_cls, int(os.environ.get("LOCAL_RANK", 0)))
            else:
                self.caption_encoder.to(self.trainer.device)
                
        self.ph_tokenizer = TokenTextEncoder(None, vocab_list=PHONE_VOCAB, replace_oov='<UNK>')
        self.tone_tokenizer = TokenTextEncoder(None, vocab_list=TONE_VOCAB, replace_oov='<UNK>')
        self.sil_ph = self.ph_tokenizer.sil_phonemes()
        
        return {'trainable': [self.model], 'others': []}

    def load_model(self):
        if hparams.get('load_ckpt', '') != '':
            load_ckpt(self.model, hparams['load_ckpt'], 'model', strict=False, mmap=True)

    def build_optimizer(self):
        if not hparams.get('disable_weight_decay_on_bias_and_norm_and_embed', False):
            
            optimizer = AdamW(unwrap_model(self.model).parameters(), **self.config.optimizer)
            
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
            for name, param in unwrap_model(self.model).named_parameters():
                if not param.requires_grad:
                    continue
                if param.dim() == 1 or has_name(['bias', 'norm', 'embed'], name):
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
        return [self.model]
    
    def fsdp_wrap_policy(self):
        from torch.nn import Linear, Sequential, Conv1d, Conv2d, Embedding
        import modules.flow_matching.llama
        import modules.asr.llama.llama_seq2seq
        import modules.tts.llama_dit.llama_ca

        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            model_blocks = (
                modules.flow_matching.llama.TransformerBlock,
                modules.asr.llama.llama_seq2seq.DecoderBlock,
                modules.asr.llama.llama_seq2seq.EncoderBlock,
                modules.tts.llama_dit.llama_ca.TransformerBlock,
                get_class_from_module("transformers.models.qwen3.modeling_qwen3", "Qwen3DecoderLayer")
            )
            return recurse or isinstance(module, model_blocks)

        return custom_auto_wrap_policy

    def _training_step(self, sample, batch_idx, optimizer_idx):
        if random.random() < 0.1:
            kill_void()
        loss_output, model_out = self.run_model(sample)
        loss_weights = {
            'nll': 1.0,
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
        
        inputs = {
            'merged_ph_tokens': sample['merged_ph_tokens'],
            'merged_ph_tokens_len': sample['merged_ph_tokens_len'],
            'dur_tokens': sample['dur'].float().clamp(0, hparams.get('dur_max_value', 128))
        }
        
        if hparams.get('balance_sil', False):
            loss_w_mask = sequence_mask(sample['merged_ph_tokens_len'])[..., None].float()
            for p in self.sil_ph:
                loss_w_mask[sample['ph_tokens'] == self.ph_tokenizer.encode(p)[0]] = 0.1
            inputs['loss_w_mask'] = loss_w_mask
        
        if hparams.get('backbone', 'llama') == 'llama_txtcond_seq2seq':
            text = sample['text']
            device = self.trainer.device
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                    caption_embs, caption_mask = self.run_caption_encoder(text, device)
                    caption_embs = caption_embs * caption_mask[..., None]
                    caption_lens = caption_mask.sum(-1)
                    inputs['condition'] = caption_embs
                    inputs['condition_lens'] = caption_lens

        if unwrap_model(self.model).config.loss_type == 'nll':
            
            try:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    model_outputs = self.model(inputs, instruction_finetuning=hparams.get('use_icl', False))
                    gt_dur = model_outputs['gt_dur'].detach().cpu()
                    mu = model_outputs['mu'].detach().cpu()
                    sigma = model_outputs['sigma'].detach().cpu()
                    r2 = model_outputs['r2']
                    m_log_sigma = model_outputs['m_log_sigma']
            except:
                traceback.print_exc()
                model_outputs = {'loss': 0, 'ntokens': 0}
                gt_dur, mu, sigma, r2, m_log_sigma = 0, 0, 0, 0, 0
            
            losses_out['nll'] = model_outputs['loss']
            losses_out['monitor/gt_dur'] = gt_dur
            losses_out['monitor/mu'] = mu
            losses_out['monitor/sigma'] = sigma
            losses_out['r2'] = r2
        
        elif unwrap_model(self.model).config.loss_type == 'mse':
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                model_outputs = self.model(inputs, instruction_finetuning=hparams.get('use_icl', False))
            losses_out['mse_loss'] = model_outputs['loss']
            
        elif unwrap_model(self.model).config.loss_type == 'ce':
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                model_outputs = self.model(inputs, instruction_finetuning=hparams.get('use_icl', False))
            losses_out['ce_loss'] = model_outputs['loss']
            
        losses_out['ntokens'] = model_outputs['ntokens']
        losses_out['bs'] = sample['merged_ph_tokens'].shape[0]
        return losses_out, model_out
    
    
class DurDiTTask(ScriptSpeechBaseTask):
    def build_model(self):  # interface to BaseTask()
        from modules.tts.ar_dur.dur_dit import build_dur_model
        self.model = build_dur_model(hparams, vocab_size=810, padding_idx=797)

        caption_encoder_name = hparams.get('cond_encoder_name')
        self.caption_encoder, self.caption_tokenizer = build_qwen3(hparams, caption_encoder_name)
        self.caption_encoder = self.caption_encoder.model
        self.caption_encoder.requires_grad_(False).eval()
        if hparams.get('use_fsdp', True):
            transformer_cls = get_class_from_module("transformers.models.qwen3.modeling_qwen3", "Qwen3DecoderLayer")
            self.caption_encoder = shard_model_in_node(self.caption_encoder, transformer_cls, int(os.environ.get("LOCAL_RANK", 0)))
        else:
            self.caption_encoder.to(self.trainer.device)
                
        self.ph_tokenizer = TokenTextEncoder(None, vocab_list=PHONE_VOCAB, replace_oov='<UNK>')
        self.tone_tokenizer = TokenTextEncoder(None, vocab_list=TONE_VOCAB, replace_oov='<UNK>')
        self.sil_ph = self.ph_tokenizer.sil_phonemes()

        if hparams.get('use_ema', False):
            print_once(f'| Building EMA model with decay={self.config.ema_decay} ...')
            self.ema_model = deepcopy(self.model)
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad = False
            self.ema_model.to(self.trainer.device)
            
            return {'trainable': [self.model, self.ema_model], 'others': []}
        
        return {'trainable': [self.model], 'others': []}
    
    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        super().on_after_optimization(epoch, batch_idx, optimizer, optimizer_idx)
        if hparams.get('use_ema', False):
            self.ema_update(self.ema_model, self.model, self.config.ema_decay)

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
            load_ckpt(self.model, hparams['load_ckpt'], 'model', strict=False, mmap=True)
            if hparams.get('use_ema', False):
                load_ckpt(self.ema_model, hparams['load_ckpt'], 'dit', strict=False, mmap=True)

    def build_optimizer(self):
        def has_name(names, param_name):
            if not isinstance(names, list):
                names = [names]
            for name in names:
                if name in param_name:
                    return True
            return False
            
        decay_params = []
        no_decay_params = []
        for name, param in unwrap_model(self.model).named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() == 1 or has_name(['bias', 'norm', 'text_embed'], name):
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
        return [self.model]

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
        
        inputs = {
            'merged_ph_tokens': sample['merged_ph_tokens'],
            'merged_ph_tokens_len': sample['merged_ph_tokens_len'],
            'dur_tokens': sample['dur'].float().clamp_min(0)
        }
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                caption_embs, caption_mask = self.run_caption_encoder(sample['text'], self.trainer.device)
                caption_embs = caption_embs * caption_mask[..., None]
                caption_lens = caption_mask.sum(-1)
                inputs['condition'] = caption_embs
                inputs['condition_lens'] = caption_lens
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = self.model(inputs)
        
        losses_out['diff_loss'] = loss
        losses_out['bs'] = inputs['dur_tokens'].shape[0]
        losses_out['ntokens'] = inputs['merged_ph_tokens_len'].sum()
        return losses_out, model_out
    