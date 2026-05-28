import os
import random

from attrdict import AttrDict
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np

from tasks.tts.dataset_utils.dataset_mixin import TTSDatasetMixin,FastDatasetMixin
from utils.commons.base_task_old import BaseTask
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.import_utils import import_module_bystr
from utils.commons.hparams import hparams, set_hparams
from utils.nn.schedulers import WarmupSchedule, CosineSchedule, CosineAnnealingWarmRestartsWithWarmup
from utils.nn.seq_utils import sequence_mask, add_prefix
from utils.nn.model_utils import print_arch, num_params, unwrap_model
from utils.commons.os_utils import kill_void
from utils.commons.dataset_utils import data_loader, build_dataloader
from utils.commons.trainer import LOCAL_RANK

from modules.tts.scriptspeech.build_model_utils import DiTBuildModelMixin, SemanticLMBuildModelMixin


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


class ScriptSpeechDiTTask(DiTBuildModelMixin, ScriptSpeechBaseTask):
    def load_model(self):
        if hparams.get('load_ckpt', '') != '':
            load_ckpt(self.dit, hparams['load_ckpt'], 'dit', strict=False)

    def build_optimizer(self):
        optimizer = AdamW(unwrap_model(self.dit).parameters(), **self.config.optimizer)
        return optimizer

    def fsdp_optm2model(self):
        return [self.dit]
    
    def _training_step(self, sample, batch_idx, optimizer_idx):
        if random.random() < 0.0001:
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

        if 'ph_tokens' in sample:
            ph_tokens = sample["ph_tokens"]
            tone_tokens = sample["tone"]

            en_tone_idx = ~((tone_tokens == 4) | ( (11 <= tone_tokens) & (tone_tokens <= 15)) | (tone_tokens == 0))
            tone_tokens[en_tone_idx] = 3

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
        else:
            ph_tokens=None
            tone_tokens=None
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
            "phone": ph_tokens,
            "tone": tone_tokens,
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
