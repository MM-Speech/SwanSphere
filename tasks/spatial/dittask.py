import os
import random
import re

from attrdict import AttrDict
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np

import torch.distributed as dist
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
# from tasks.tts.task_utils.prompttts_task_utils import build_audio_mask_from_ids

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

        self.log_grad_every_n_steps = hparams.get('log_grad_every_n_steps', 1)

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


class SpatialDiTTask(ScriptSpeechBaseTask):
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
        if random.random() < 0.0001:
            kill_void()
        loss_output, model_out = self.run_model(sample)
        loss_weights = {
            'diff_loss': 1.0,
        }
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if
                          isinstance(v, torch.Tensor) and v.requires_grad])

        return total_loss, loss_output
    
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
    
    def run_goku_text_encoder(self, captions: list):
        raise NotImplementedError
    
    
    def run_model(self, sample, infer=False, infer_steps=None):
        model_out = {}
        losses_out = {}
        
        inputs_embeds = sample['inputs_embeds']
        global_clip_embedding = sample['global_clip_embedding']
        # direction = sample['direction']
        # energy_map = sample['energy_map']
        lat = sample['lat']
        file_name = sample['file_name']
        
        B, T, _ = lat.shape
        
        
        inputs = {
            "inputs_embeds": inputs_embeds,
            "global_clip_embedding": global_clip_embedding,
            # "direction": direction,
            # "energy_map": energy_map,
            "lat": lat,
            "file_name": file_name
        }
        
        if not infer:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                model_outputs, target = self.dit(inputs)
            
            loss = F.mse_loss(model_outputs.float(), target.float(), reduction='none')
            loss = loss.sum() / target.shape[-1] / (B*T) 

            losses_out['diff_loss'] = loss
            losses_out['ntokens'] = B*T
            return losses_out, None # 原本是返回model_out，但是没复制，后续也没用，这里先返回None 
        else:
            return losses_out, model_out