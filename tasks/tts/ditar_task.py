import os
import random
import re
import math

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
from utils.commons.dataset_utils import pad_or_cut_xd
from utils.commons.tensor_utils import move_to_cpu, convert_to_np
from utils.nn.schedulers import WarmupSchedule, CosineSchedule, CosineAnnealingWarmRestartsWithWarmup
from utils.nn.seq_utils import sequence_mask, add_prefix
from utils.nn.model_utils import print_arch, num_params, unwrap_model, freeze_by_module_name
from utils.nn.ema import EMAModel, restore_ema

from modules.tts.ditar.build_model_utils import DiTARBuildModelMixinS, DiTARBuildModelMixin#, DiTARBuildModelMixinV2, DiTARBuildModelMixinV3
from tasks.tts.scriptspeech_task import ScriptSpeechBaseTask
from tasks.tts.dataset_utils.dataset_mixin import FastDatasetMixin, TTSDatasetMixin


class DiTARTask(DiTARBuildModelMixin, ScriptSpeechBaseTask):
    def build_model(self):
        self._build_model()
        self.vae.to(self.trainer.device)

        if hparams.get('use_ema', False):
            print_once(f'| Building EMA model with decay={self.config.ema_decay} ...')
            self.ema_model = deepcopy(self.model)
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad = False
            self.ema_model.to(self.trainer.device)
            
            return {'trainable': [self.model, self.ema_model], 'others': []}

        return {'trainable': [self.model], 'others': []}
    
    def compute_grad_norm(self, optimizer, distributed=True, norm_type=2.0):
        """
        计算当前 optimizer 全部参数的全局 L2 grad norm。
        - 要求在 AMP 的 unscale_ 之后调用，保证是真实梯度。
        - 支持 DDP/FSDP：通过 all_reduce(sum of squares) 聚合各 rank。
        """
        if norm_type != 2.0:
            norm_type = 2.0

        device = torch.device(self.trainer.device) if isinstance(self.trainer.device, str) else self.trainer.device
        local_sq_sum = torch.zeros(1, device=device, dtype=torch.float32)
        has_grad = False

        for group in optimizer.param_groups:
            for p in group['params']:
                if p is None or p.grad is None:
                    continue
                g = p.grad
                # 稀疏梯度
                if g.is_sparse:
                    g = g.coalesce().values()
                # 统一到 float32 做范数更稳
                g = g.detach().float()
                local_sq_sum += torch.sum(g * g)
                has_grad = True

        if not has_grad:
            return 0.0

        if distributed and dist.is_initialized():
            dist.all_reduce(local_sq_sum, op=dist.ReduceOp.SUM)

        total_norm = torch.sqrt(local_sq_sum)
        return float(total_norm.item())
    
    def on_before_optimization(self, opt_idx):
        """
        在 AMP unscale 之后、梯度裁剪之前调用（推荐调整 Trainer 调用顺序到此处）。
        返回一个 dict，以便 Trainer 统一写入 TensorBoard / pbar。
        """
        # 频率控制（按“有效 step”，即累计边界）
        if getattr(self, 'log_grad_every_n_steps', 0) <= 0:
            print('| INFO: no log_grad_every_n_steps')
            return
        else:
            eff_step = (self.global_step + 1) // hparams.get('accumulate_grad_batches', 1)
            if eff_step % self.log_grad_every_n_steps != 0:
                print(f'| INFO: not eff_step')
                return None

        try:
            optimizer = self.trainer.optimizers[opt_idx]
            gnorm = self.compute_grad_norm(optimizer, distributed=False, norm_type=2.0)
            # print(f'| INFO: on_before_optimization did compute_grad_norm')
            return {f'monitor/grad_norm_optm{opt_idx}': gnorm}
        except Exception as e:
            if self.trainer.proc_rank == 0:
                print(f'| WARN: on_before_optimization compute_grad_norm failed: {e}')
            return None
        
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
                load_ckpt(self.ema_model, hparams['load_ckpt'], 'model', strict=False, mmap=True)

    def build_optimizer(self):
        if not hparams.get('disable_weight_decay_on_bias_and_norm_and_embed', True):
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
        return [self.model]

    def fsdp_wrap_policy(self):
        from torch.nn import Linear, Sequential, Conv1d, Conv2d, Embedding
        import modules.flow_matching.llama
        import modules.asr.llama.llama_seq2seq
        import modules.tts.llama_dit.llama_ca

        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            model_blocks = (
                # Linear,
                # Sequential,
                # Conv1d,
                # Conv2d,
                # Embedding,
                modules.flow_matching.llama.TransformerBlock,
                modules.asr.llama.llama_seq2seq.DecoderBlock,
                modules.asr.llama.llama_seq2seq.EncoderBlock,
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
            'stop_loss': hparams.get('lambda_stop_loss', 10.0),
            'diff_loss': 1.0,
        }
        if self.global_step < hparams.get('stop_loss_start_step', 2000):
            loss_weights['stop_loss'] = 0.0
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
        ph_tokens = sample["ph_tokens"]
        tone_tokens = sample["tone"]
        dur = sample['dur']
        device = wavs.device

        # Disable the English tone (set them to 3)
        en_tone_idx = ~((tone_tokens == 4) | ( (11 <= tone_tokens) & (tone_tokens <= 15)) | (tone_tokens == 0))
        tone_tokens[en_tone_idx] = 3

        # audio encode
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                lat = self.vae.encode_latent(wavs)
        lat_lens = wav_lengths // hparams['hop_size'] // hparams['vae_stride']
        patch_size = hparams.get('ditar_patch_size', 4)
        lat = pad_or_cut_xd(lat, math.ceil(lat.shape[1] / patch_size) * patch_size, dim=1)
        
        if random.random() < 0.001:
            print('| text sample:', text[0])

        # text tokenize
        text_inputs = self.ditar_text_tokenizer(text, padding=True, return_tensors="pt").to(device)
        txt_tokens = text_inputs['input_ids']   # [B, T]
        txt_mask = text_inputs['attention_mask'].bool()
        txt_lens = txt_mask.int().sum(1)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                caption_embs, caption_mask = self.run_caption_encoder(text, device)
                caption_embs = caption_embs * caption_mask[..., None]
                caption_lens = caption_mask.sum(-1)
        
        inputs = {
            "phone": ph_tokens,
            "tone": tone_tokens,
            "txt_tokens": txt_tokens.long(),
            "txt_lens": txt_lens,
            "txt_mask": txt_mask,
            "lat": lat,
            "lat_lens": lat_lens,
            "caption_emb": caption_embs,
            "caption_lens": caption_lens, # B
            "dur": dur,
        }

        if not infer:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                model_outputs = self.model(inputs)
            
            pred = model_outputs['pred']
            target = model_outputs['target']
            loss_mask = model_outputs['loss_mask']

            losses_out['diff_loss'] = model_outputs['diff_loss']
            losses_out['stop_loss'] = model_outputs['stop_loss']
            losses_out['bs'] = lat.shape[0]
            losses_out['ntokens'] = sum(lat_lens)
            
            # monitor
            with torch.no_grad():
                losses_out['monitor/orig_loss'] = losses_out['diff_loss']
                losses_out['monitor/pred'] = (pred * loss_mask).sum() / loss_mask.sum()
                losses_out['monitor/tgt'] = (target * loss_mask).sum() / loss_mask.sum()
                losses_out['monitor/loss_mask'] = loss_mask.sum() 
                losses_out['monitor/ctx_mask'] = model_outputs['ctx_mask'].sum()
                losses_out['monitor/stop_labels_sum'] = model_outputs['stop_labels'].sum()

            if loss_mask.sum() <= 3:
                if self.trainer.proc_rank_local == 0:
                    print(f"\n| CRITICAL: loss_mask.sum() = {loss_mask.sum()}. Resetting loss.")
                losses_out['diff_loss'] = 0.0
                losses_out['stop_loss'] = 0.0
            
            return losses_out, model_out
        else:
            return losses_out, model_out
        
        
# class DiTARV2Task(DiTARBuildModelMixinV2, DiTARTask):
#     def __init__(self):
#         super().__init__()
        
#         if hparams.get('dataloader_version', 'v1') == 'v1':
#             self.dataset_cls = import_module_bystr(hparams['dataset_cls'])
#             self.val_dataset_cls = import_module_bystr(hparams['val_dataset_cls'])
#             self.processer_fn = import_module_bystr(hparams['processer_fn']) if hparams.get('processer_fn') else None
#             self.build_fast_dataloader = import_module_bystr(hparams['build_fast_dataloader']) if hparams.get('build_fast_dataloader') else None
#         elif hparams.get('dataloader_version', 'v1') == 'v2':
#             self.dataset_cls = import_module_bystr(hparams['dataset_cls'])
#             self.train_dataloader = FastDatasetMixin.train_dataloader.__get__(self)
#             self.test_dataloader = FastDatasetMixin.test_dataloader.__get__(self)
#             self.val_dataloader = FastDatasetMixin.val_dataloader.__get__(self)
#         self.hparams = hparams
#         self.config = AttrDict(hparams)
        
#     def build_model(self):
#         self._build_model()
#         self.vae.to(self.trainer.device)

#         if hparams.get('use_ema', False):
#             print_once(f'| Building EMA model with decay={self.config.ema_decay} ...')
#             self.ema_model = deepcopy(self.model)
#             self.ema_model.eval()
#             for param in self.ema_model.parameters():
#                 param.requires_grad = False
#             self.ema_model.to(self.trainer.device)
            
#             return {'trainable': [self.model, self.ema_model], 'others': []}
        
#         return {'trainable': [self.model], 'others': []}
    
#     def run_model(self, sample, infer=False, infer_steps=None):
#         model_out = {}
#         losses_out = {}
#         if infer:
#             return losses_out, model_out
#         if 'wavs' not in sample:
#             return losses_out, model_out

#         wavs = sample["wavs"].float()
#         wav_lengths = sample["wav_lengths"]
#         wav_mask = sequence_mask(wav_lengths, maxlen=wavs.size(1))
        
#         text = sample['text']
#         txt_tokens = sample['txt_tokens']
#         txt_lens = sample['txt_lengths']
#         txt_mask = sequence_mask(txt_lens, txt_tokens.shape[1])
#         device = wavs.device
        
#         # audio encode
#         with torch.no_grad():
#             lat = self.vae.encode_latent(wavs)
#         lat_lens = wav_lengths // hparams['hop_size'] // hparams['vae_stride']
#         patch_size = hparams.get('ditar_patch_size', 4)
#         lat = pad_or_cut_xd(lat, math.ceil(lat.shape[1] / patch_size) * patch_size, dim=1)
        
#         if random.random() < 0.001:
#             print('| text sample:', text[0])
            
#         caption_embs, caption_lens, caption_mask = None, None, None
#         if hparams.get('use_caption_encoder', False):
#             with torch.no_grad():
#                 caption_embs, caption_mask = self.run_caption_encoder(text, device)
#                 caption_embs = caption_embs * caption_mask[..., None]
#                 caption_lens = caption_mask.sum(-1)
        
#         inputs = {
#             "txt_tokens": txt_tokens.long(),
#             "txt_lens": txt_lens,
#             "txt_mask": txt_mask,
#             "lat": lat,
#             "lat_lens": lat_lens,
#             "caption_emb": caption_embs,
#             "caption_lens": caption_lens, # B
#         }
        
#         model_outputs = self.model(inputs)
        
#         pred = model_outputs['pred']
#         target = model_outputs['target']
#         loss_mask = model_outputs['loss_mask']

#         losses_out['diff_loss'] = model_outputs['diff_loss']
#         losses_out['stop_loss'] = model_outputs['stop_loss']
#         losses_out['bs'] = lat.shape[0]
#         losses_out['ntokens'] = sum(lat_lens)
        
#         # monitor
#         with torch.no_grad():
#             losses_out['monitor/orig_loss'] = losses_out['diff_loss']
#             losses_out['monitor/pred'] = (pred * loss_mask).sum() / loss_mask.sum()
#             losses_out['monitor/tgt'] = (target * loss_mask).sum() / loss_mask.sum()
#             losses_out['monitor/loss_mask'] = loss_mask.sum() 
#             losses_out['monitor/ctx_mask'] = model_outputs['ctx_mask'].sum()
#             losses_out['monitor/stop_labels_sum'] = model_outputs['stop_labels'].sum()
        
#         return losses_out, model_out

        
# class DiTARV2WarmUpTask(DiTARV2Task):
#     def build_model(self):
        
#         from modules.asr.wavlm.WavLM import WavLM, WavLMConfig
#         checkpoint = torch.load('checkpoints/wavlm/WavLM-Large.pt')
#         cfg = WavLMConfig(checkpoint['cfg'])
#         model = WavLM(cfg)
#         model.load_state_dict(checkpoint['model'])
#         self.audio_feat_dim = cfg.encoder_embed_dim
#         self.audio_encoder = model
#         self.audio_encoder.to(self.trainer.device)
#         for param in self.audio_encoder.parameters():
#             param.requires_grad = False
#             param.grad = None
#         self.audio_encoder.eval()
#         self.audio_encoder = torch.compile(self.audio_encoder, mode='max-autotune')
        
#         self.ditar_text_tokenizer, self.ditar_vocab_size = self.build_ditar_text_tokenizer()
#         self.build_ditar(hparams)
        
#         if hparams.get('use_ema', False):
#             print_once(f'| Building EMA model with decay={self.config.ema_decay} ...')
#             self.ema_model = deepcopy(self.model)
#             self.ema_model.eval()
#             for param in self.ema_model.parameters():
#                 param.requires_grad = False
#             self.ema_model.to(self.trainer.device)
            
#             return {'trainable': [self.model, self.ema_model], 'others': []}
        
#         return {'trainable': [self.model], 'others': []}
    
#     def run_model(self, sample, infer=False, infer_steps=None):
#         model_out = {}
#         losses_out = {}
#         if infer:
#             return losses_out, model_out
#         if 'wavs' not in sample:
#             return losses_out, model_out

#         wavs = sample["wavs"].float()
#         wav_lengths = sample["wav_lengths"]
#         if not hasattr(self, 'resampler'):
#             self.resampler = torchaudio.transforms.Resample(orig_freq=hparams['audio_sample_rate'], new_freq=16000).to(wavs.device)
#         wavs = self.resampler(wavs)
#         wav_lengths = (wav_lengths * 16000 / hparams['audio_sample_rate']).int()
#         wav_mask = sequence_mask(wav_lengths, maxlen=wavs.size(1))
        
#         txt_tokens = sample['txt_tokens']
#         txt_lens = sample['txt_lengths']
#         txt_mask = sequence_mask(txt_lens, txt_tokens.shape[1])
#         device = wavs.device
        
#         # audio encode
#         with torch.no_grad():
#             audio_feat, audio_feat_padding_mask = self.audio_encoder.extract_features(wavs, padding_mask=~(wav_mask.bool()))
#             audio_feat_mask = ~audio_feat_padding_mask  # [B, T]
            
#             # downsample the audio_feat by the factor of 2, averaging pooling
#             audio_feat = F.avg_pool1d(audio_feat.transpose(1, 2), kernel_size=2, stride=2, ceil_mode=True).transpose(1, 2)
#             audio_feat_mask = audio_feat_mask[:, ::2]
        
#         inputs = {
#             'txt_tokens': txt_tokens,
#             'txt_mask': txt_mask,
#             'audio_feat': audio_feat,
#             'audio_feat_mask': audio_feat_mask,
#         }
        
#         model_outputs = self.model(inputs)
        
#         losses_out['stop_loss'] = model_outputs['stop_loss']
#         losses_out['cos'] = model_outputs['cos']
        
#         with torch.no_grad():
#             losses_out['bs'] = wavs.shape[0]
#             losses_out['ntokens'] = model_outputs['ntokens']
                
#         return losses_out, model_out 
    

# class DiTARV3Task(DiTARV2Task):
#     def __init__(self):
#         super().__init__()
#         self.build_ditar = DiTARBuildModelMixinV3.build_ditar.__get__(self)

#     def fsdp_wrap_policy(self):
#         import modules.commons.hf.transformer
#         import modules.commons.hf.transformer_moe
#         import modules.commons.hf.transformer_dit
#         import modules.commons.hf.transformer_dit_moe

#         def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
#             model_blocks = (
#                 modules.commons.hf.transformer.EncoderLayer,
#                 modules.commons.hf.transformer.DecoderLayer,
#                 modules.commons.hf.transformer_moe.EncoderLayer,
#                 modules.commons.hf.transformer_moe.DecoderLayer,
#                 modules.commons.hf.transformer_dit.EncoderLayer,
#                 modules.commons.hf.transformer_dit_moe.EncoderLayer,
#                 get_class_from_module("transformers.models.qwen3.modeling_qwen3", "Qwen3DecoderLayer")
#             )
#             return recurse or isinstance(module, model_blocks)

#         return custom_auto_wrap_policy


# class DiTARV3WarmUpTask(DiTARV2WarmUpTask):
#     def __init__(self):
#         super().__init__()
#         self.build_ditar = DiTARBuildModelMixinV3.build_ditar.__get__(self)

#     def fsdp_wrap_policy(self):
#         import modules.commons.hf.transformer
#         import modules.commons.hf.transformer_moe
#         import modules.commons.hf.transformer_dit
#         import modules.commons.hf.transformer_dit_moe

#         def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
#             model_blocks = (
#                 modules.commons.hf.transformer.EncoderLayer,
#                 modules.commons.hf.transformer.DecoderLayer,
#                 modules.commons.hf.transformer_moe.EncoderLayer,
#                 modules.commons.hf.transformer_moe.DecoderLayer,
#                 modules.commons.hf.transformer_dit.EncoderLayer,
#                 modules.commons.hf.transformer_dit_moe.EncoderLayer,
#                 get_class_from_module("transformers.models.qwen3.modeling_qwen3", "Qwen3DecoderLayer")
#             )
#             return recurse or isinstance(module, model_blocks)

#         return custom_auto_wrap_policy

class DiTARSpatialTask(DiTARBuildModelMixinS, ScriptSpeechBaseTask):
    def build_model(self):
        self._build_model()
        
        if hparams.get('use_ema', False):
            raise NotImplementedError("EMA not implemented for DiTAR")
        
        return {'trainable': [self.model], 'others': []}
    
    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        super().on_after_optimization(epoch, batch_idx, optimizer, optimizer_idx)
        if hparams.get('use_ema', False):
            raise NotImplementedError("EMA not implemented for DiTAR")
            # self.ema_update(self.ema_model, self.model, self.config.ema_decay)
    
    
    def load_model(self):
        if hparams.get('load_ckpt', '') != '':
            load_ckpt(self.model, hparams['load_ckpt'], 'model', strict=False, mmap=True)
            if hparams.get('use_ema', False):
                raise NotImplementedError("EMA not implemented for DiTAR")
                # load_ckpt(self.ema_model, hparams['load_ckpt'], 'model', strict=False, mmap=True)
    
    def build_optimizer(self):
        if not hparams.get('disable_weight_decay_on_bias_and_norm_and_embed', True):
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
        return [self.model]

    def fsdp_wrap_policy(self):
        from torch.nn import Linear, Sequential, Conv1d, Conv2d, Embedding
        import modules.flow_matching.llama
        import modules.asr.llama.llama_seq2seq
        import modules.tts.llama_dit.llama_ca

        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            model_blocks = (
                # Linear,
                # Sequential,
                # Conv1d,
                # Conv2d,
                # Embedding,
                modules.flow_matching.llama.TransformerBlock,
                modules.asr.llama.llama_seq2seq.DecoderBlock,
                modules.asr.llama.llama_seq2seq.EncoderBlock,
                modules.tts.llama_dit.llama_ca.TransformerBlock,
                get_class_from_module("transformers.models.qwen3.modeling_qwen3", "Qwen3DecoderLayer")
            )
            return recurse or isinstance(module, model_blocks)

        return custom_auto_wrap_policy
    
    def _training_step(self, sample, batch_idx, optimizer_idx):
        # if self.trainer.proc_rank_local == 0 and random.random() < 0.1:
        #     kill_void()
        loss_output, model_out = self.run_model(sample)
        loss_weights = {
            # 'stop_loss': hparams.get('lambda_stop_loss', 10.0),
            'stop_loss': 0.0,
            'diff_loss': 1.0,
        }
        if self.global_step < hparams.get('stop_loss_start_step', 2000):
            loss_weights['stop_loss'] = 0.0
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if
                          isinstance(v, torch.Tensor) and v.requires_grad])
        
        # if self.trainer.proc_rank == 0 and self.global_step < 10:
        #     save_dir = f'{hparams["work_dir"]}/sample_batches/step_{self.global_step}'
        #     os.makedirs(save_dir, exist_ok=True)
        #     sample = convert_to_np(move_to_cpu(sample))
        #     for i in range(sample['nsamples']):
        #         sf.write(f"{save_dir}/{i}.wav", sample['wavs'][i, :sample['wav_lengths'][i]], 24000, 'PCM_16')
        #         sf.write(f"{save_dir}/{i}_ctx.wav", sample['ctx_wavs'][i, :sample['wav_lengths'][i]], 24000, 'PCM_16')
        #         np.save(f"{save_dir}/{i}.npy", sample, allow_pickle=True)
        #     del sample['wavs']
        #     del sample['ctx_wavs']

        return total_loss, loss_output
    
    
    def run_model(self, sample, infer=False, infer_steps=None):
        model_out = {}
        losses_out = {}
        if infer:
            return losses_out, model_out
        
        lat = sample["lat"]
        lat_lens = sample["lat_lens"]
        video_feats = sample["video_feats"]
        v_mask = sample["v_mask"]
        # direction = sample['direction']
        # energy_map = sample['energy_map']
        global_clip_embedding = sample['global_clip_embedding']
        
        
        B, T, _ = lat.shape
        
        inputs = {
            "video_feats": video_feats,
            "v_mask": v_mask,
            "global_clip_embedding": global_clip_embedding,
            # "direction": direction,
            # "energy_map": energy_map,
            "lat": lat,
            "lat_lens": lat_lens,
            # "file_name": file_name
        }

        if not infer:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                model_outputs = self.model(inputs)
            
            pred = model_outputs['pred']
            target = model_outputs['target']
            loss_mask = model_outputs['loss_mask']

            losses_out['diff_loss'] = model_outputs['diff_loss']
            losses_out['stop_loss'] = model_outputs['stop_loss']
            losses_out['bs'] = lat.shape[0]
            losses_out['ntokens'] = sum(lat_lens)
            
            # monitor
            with torch.no_grad():
                losses_out['monitor/orig_loss'] = losses_out['diff_loss']
                losses_out['monitor/pred'] = (pred * loss_mask).sum() / loss_mask.sum()
                losses_out['monitor/tgt'] = (target * loss_mask).sum() / loss_mask.sum()
                losses_out['monitor/loss_mask'] = loss_mask.sum() 
                losses_out['monitor/ctx_mask'] = model_outputs['ctx_mask'].sum()
                losses_out['monitor/stop_labels_sum'] = model_outputs['stop_labels'].sum()

            if loss_mask.sum() <= 3:
                if self.trainer.proc_rank_local == 0:
                    print(f"\n| CRITICAL: loss_mask.sum() = {loss_mask.sum()}. Resetting loss.")
                losses_out['diff_loss'] = 0.0
                losses_out['stop_loss'] = 0.0
            
            return losses_out, model_out
        else:
            return losses_out, model_out
        