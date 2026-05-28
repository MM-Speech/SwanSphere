import os
import random
import re

from attrdict import AttrDict
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import torchaudio
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
from utils.nn.model_utils import print_arch, num_params, unwrap_model, freeze_by_module_name
from utils.nn.ema import EMAModel, restore_ema
from utils.commons.tensor_utils import move_to_cpu, convert_to_np

from modules.tts.scriptspeech.build_model_utils import build_vae, build_qwen3, shard_model_in_node, \
    DiTBuildModelMixinV2, DiTBuildModelMixinV4, DiTBuildModelMixinV5
from tasks.tts.dataset_utils.dataset_mixin import FastDatasetMixin, TTSDatasetMixin
from tasks.tts.scriptspeech_task import ScriptSpeechBaseTask


class MegaTTSDiTTask(DiTBuildModelMixinV2, ScriptSpeechBaseTask):
    def __init__(self):
        if hparams.get('dataloader_version', 'v1') == 'v1':
            self.dataset_cls = import_module_bystr(hparams['dataset_cls'])
            self.val_dataset_cls = import_module_bystr(hparams['val_dataset_cls'])
            self.processer_fn = import_module_bystr(hparams['processer_fn']) if hparams.get('processer_fn') else None
            self.build_fast_dataloader = import_module_bystr(hparams['build_fast_dataloader']) if hparams.get('build_fast_dataloader') else None
        elif hparams.get('dataloader_version', 'v1') == 'v2':
            self.dataset_cls = import_module_bystr(hparams['dataset_cls'])
            self.train_dataloader = FastDatasetMixin.train_dataloader.__get__(self)
            self.test_dataloader = FastDatasetMixin.test_dataloader.__get__(self)
            self.val_dataloader = FastDatasetMixin.val_dataloader.__get__(self)
        self.hparams = hparams
        self.config = AttrDict(hparams)

        super().__init__()
        
    def build_model(self):
        self._build_model()
        self.vae.to(self.trainer.device)
        
        if hparams.get('freeze_ling_encoder', False) is True:
            frozen_modules = freeze_by_module_name(
                unwrap_model(self.dit),
                freeze_modules=[
                    'text_embedder', 'text_encoder', 'ph_encoder', 'tone_embed', 'ph_embed', 'ling_pre_net', 'f5_time_embed', 
                ]
            )
            cr = '\n'
            print_once(f"| Freeze following modules:{cr}{cr.join([f'| - {frozen_module}' for frozen_module in frozen_modules])}")

        if hparams.get('use_ema', False):
            print_once(f'| Building EMA model with decay={self.config.ema_decay} ...')
            self.ema_model = deepcopy(self.dit)
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad = False
            self.ema_model.to(self.trainer.device)
            
            return {'trainable': [self.dit, self.ema_model], 'others': []}

        return {'trainable': [self.dit], 'others': []}

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
    
    def _training_step(self, sample, batch_idx, optimizer_idx):
        if self.trainer.proc_rank_local == 0 and random.random() < 0.0001:
            kill_void()
        loss_output, model_out = self.run_model(sample)
        loss_weights = {
            'diff_loss': 1.0,
        }
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if
                          isinstance(v, torch.Tensor) and v.requires_grad])
        
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

    def additional_condition(self, sample, inputs, conditions=None):
        return inputs

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
        ph_mask = ph_tokens > 0
        mel2ph = sample['mel2ph']
        dur = sample['dur']

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
                if hparams.get('is_foundation', False):
                    lat_ctx = torch.zeros_like(lat)
                    ctx_mask = torch.zeros_like(ctx_mask)
                else:
                    lat_ctx = self.vae.encode_latent(ctx_wavs)
                    lat_ctx = torch.nn.functional.pad(lat_ctx, (0,0,0,lat.size(1)-lat_ctx.size(1)), mode='constant', value=0)
                if hparams.get('is_partial_foundation', -1) > 0:
                    foundation_prob = hparams.get('is_partial_foundation', -1)
                    if random.random() < foundation_prob:
                        lat_ctx = torch.zeros_like(lat)
                        ctx_mask = torch.zeros_like(ctx_mask)
                    else:
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


        if hparams.get('use_caption_encoder', True):
            with torch.no_grad(),torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
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
        lat_cfg_mask = (lat_cfg_mask < hparams.get('lat_cfg_prob', 0.15)).long()
        lat_ctx = (lat_ctx * ctx_mask * (1 - lat_cfg_mask)[:, :, None])

        txt_cfg_mask = torch.rand_like(txt_tokens[:, 0].float())[:, None]
        txt_cfg_mask = (txt_cfg_mask < hparams.get('txt_cfg_prob', 0.15)).long()
        txt_tokens = txt_tokens * (1 - txt_cfg_mask) + self.cfg_mask_text_token * txt_cfg_mask
        
        if hparams.get('use_caption_encoder', True):
            caption_cfg_mask = torch.rand_like(caption_embs[:, 0].float())[:, None]
            caption_cfg_mask = (caption_cfg_mask < hparams.get('caption_cfg_prob', 0.15)).long()
            caption_embs = caption_embs * (1 - caption_cfg_mask)
        
        dur_cfg_mask = torch.rand_like(txt_tokens[:, 0].float())[:, None]
        dur_cfg_mask = (dur_cfg_mask < hparams.get('dur_cfg_prob', 0.15)).long()
        dur = (dur * (1 - dur_cfg_mask)).long()
        
        if not hasattr(self, 'cfg_mask_token_phone'):
            self.cfg_mask_token_phone = 302 - 1
        if not hasattr(self, 'cfg_mask_token_tone'):
            self.cfg_mask_token_tone = 32 - 1
        ph_cfg_mask = torch.rand_like(ph_tokens[:, 0].float())[:, None]
        ph_cfg_mask = (ph_cfg_mask < hparams.get('ph_cfg_prob', 0.15)).long()
        ph_tokens = ph_tokens * (1 - ph_cfg_mask) + self.cfg_mask_token_phone * ph_cfg_mask
        tone_tokens = tone_tokens * (1 - ph_cfg_mask) + self.cfg_mask_token_tone * ph_cfg_mask

        inputs = {
            "phone": ph_tokens,
            "ph_mask": ph_mask,
            "tone": tone_tokens,
            "txt_tokens": txt_tokens.long(),
            "txt_lens": txt_lens,
            "txt_mask": txt_mask,
            "lat": lat,
            "lat_lens": lat_lens,
            "lat_ctx": lat_ctx,
            "ctx_mask": ctx_mask,
            # "vad_mask": sample['vad_mask'][..., None]
            "mel2ph": mel2ph,
            "dur": dur,
        }
        if hparams.get('use_sparse_dur', False):
            inputs['mel2ph_sparse'] = sample['mel2ph_sparse']
        if hparams.get('use_caption_encoder', True):
            inputs['caption_emb'] = caption_embs
            inputs['caption_lens'] = caption_lens
        inputs = self.additional_condition(sample, inputs)

        if not infer:
            if 'MegaTTSDiTV5Task' in hparams['task_cls']:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                    model_outputs, target, dur_loss, dur_total_loss = self.dit(inputs)
                loss = F.mse_loss(model_outputs.float(), target.float(), reduction='none')
                loss = (loss * loss_mask).sum() / loss_mask.sum() / target.shape[-1]
                losses_out['diff_loss'] = loss
                losses_out['dur_loss'] = dur_loss if dur_loss is not None else 0.0
                losses_out['dur_total_loss'] = dur_total_loss if dur_total_loss is not None else 0.0
                losses_out['bs'] = loss_mask.shape[0]
                losses_out['ntokens'] = sum(lat_lens)
            else:  
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
            
            # if loss > 0.65:
            #     losses_out['diff_loss'] = 0.0
            if loss_mask.sum() <= 3:
                if self.trainer.proc_rank_local == 0:
                    print(f"\n| CRITICAL: loss_mask.sum() = {loss_mask.sum()}. Resetting loss.")
                losses_out['diff_loss'] = 0.0
            
            return losses_out, model_out
        else:
            return losses_out, model_out


class MegaTTSDiTV4Task(DiTBuildModelMixinV4, MegaTTSDiTTask):
    pass


class MegaTTSDiTV5Task(DiTBuildModelMixinV5, MegaTTSDiTV4Task):
    def additional_condition(self, sample, inputs, conditions=None):
        ctx_wavs = sample["ctx_wavs"]
        ctx_mask = sample["ctx_mask"].squeeze()   # [B, T]
        ctx_wav_lengths = (ctx_mask.sum(1) * 4 * hparams['hop_size']).long()
        ctx_wavs = ctx_wavs[:, :torch.max(ctx_wav_lengths)]

        if not hasattr(self, 'resampler'):
            self.resampler = torchaudio.transforms.Resample(orig_freq=hparams['audio_sample_rate'], new_freq=16000).to(ctx_wavs.device)
        ctx_wavs = self.resampler(ctx_wavs)
        ctx_wav_lengths = (ctx_wav_lengths * 16000 / hparams['audio_sample_rate']).int()

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
            feat, feat_padding_mask = self.audio_encoder.extract_features(
                ctx_wavs, padding_mask=~(sequence_mask(ctx_wav_lengths)))    # [B, T, C]
        inputs['audio_ctx_feature'] = feat
        inputs['audio_ctx_mask'] = feat_padding_mask

        return inputs
        
