import argparse
import filecmp
import multiprocessing
import os
import subprocess
import librosa
from functools import partial
from multiprocessing import Pool, Process
import random
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW
from attrdict import AttrDict

from utils.audio import torch_wav2spec
from utils.audio.align import mel2token_to_dur
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams
from utils.commons.base_task import BaseTask
from utils.commons.import_utils import import_module_bystr
from utils.commons.os_utils import kill_void
from utils.commons.io import print_once
from utils.nn.schedulers import WarmupSchedule, CosineSchedule
from utils.nn.model_utils import unwrap_model

from tasks.tts.dataset_utils.dataset_mixin import FastDatasetMixin, TTSDatasetMixin

# from tasks.vae.utils import auraloss as auraloss
from stable_audio_tools.training.losses import auraloss as auraloss
from stable_audio_tools.models.discriminators import EncodecDiscriminator



def trim_to_shortest(a, b):
    """Trim the longer of two tensors to the length of the shorter one."""
    if a.shape[-1] > b.shape[-1]:
        return a[:,:,:b.shape[-1]], b
    elif b.shape[-1] > a.shape[-1]:
        return a, b[:,:,:a.shape[-1]]
    return a, b
def trainable_param_report(model):
    total = 0
    trainable = 0
    for n, p in model.named_parameters():
        num = p.numel()
        total += num
        if p.requires_grad:
            trainable += num
    print(f"Total params: {total:,}")
    print(f"Trainable params (requires_grad=True): {trainable:,}")
    return trainable
class SAVAETask(FastDatasetMixin, BaseTask):
    def __init__(self):
        super().__init__()
        if hparams.get('dataloader_version', 'v1') == 'v1':
            self.dataset_cls = import_module_bystr(hparams['dataset_cls'])
            self.val_dataset_cls = import_module_bystr(hparams['val_dataset_cls'])
            # self.processer_fn = import_module_bystr(hparams['processer_fn'])
            # self.build_fast_dataloader = import_module_bystr(hparams['build_fast_dataloader'])
        elif hparams.get('dataloader_version', 'v1') == 'v2':
            self.dataset_cls = import_module_bystr(hparams['dataset_cls'])
            self.train_dataloader = FastDatasetMixin.train_dataloader.__get__(self)
            self.test_dataloader = FastDatasetMixin.test_dataloader.__get__(self)
            self.val_dataloader = FastDatasetMixin.val_dataloader.__get__(self)
        self.hparams = hparams
        self.config = AttrDict(hparams)
        
        # Online load mel with GPU
        self.sample_rate = hparams["sample_rate"]
        # sample_size = hparams["sample_size"]
        # audio_channels = hparams["audio_channels"]
        
    def build_model(self):
        from modules.vae.autoencoders import build_wavvae
        self.model_gen = build_wavvae(hparams=None, init_pretrained=True)
        trainable_param_report(self.model_gen)

        # if hparams.get('train_bottleneck_only', False):
        #     frozen = 0
        #     for p in self.model_gen.encoder.parameters():
        #         p.requires_grad = False
        #         frozen += 1
        #     for p in self.model_gen.decoder.parameters():
        #         p.requires_grad = False
        #         frozen += 1
        #     print_once(f"| Freeze encoder and decoder for {frozen} params, only train bottleneck")

        self.model_disc = torch.nn.ModuleDict()
        stft_loss_args = self.hparams['loss_configs']['spectral']['config']
        disc_args = self.hparams['loss_configs']['discriminator']['config']
        self.model_disc['discriminator'] = EncodecDiscriminator(in_channels=self.model_gen.out_channels, **disc_args)

        self.sdstft = auraloss.MultiResolutionSTFTLoss(sample_rate=self.sample_rate, **stft_loss_args)

        return {'trainable': [self.model_gen, self.model_disc], 'others': []}


    def fsdp_optm2model(self):
        # FIXME
        return [self.model_gen]
    
    
    def fsdp_wrap_policy(self):
        pass
        # from modules.vae.wavvae_v5 import EncoderBlock, DecoderBlock
        # from modules.codec.fish.modded_dac import TransformerBlock

        # def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
        #     model_blocks = (
        #         EncoderBlock,
        #         DecoderBlock,
        #         TransformerBlock
        #     )
        #     return recurse or isinstance(module, model_blocks)

        # return custom_auto_wrap_policy

    def build_optimizer(self):
        gen_params = self.model_gen.parameters()
        optimizer_gen = torch.optim.AdamW(gen_params, lr=hparams['lr'],
                                        betas=[hparams['adam_b1'], hparams['adam_b2']])

        optimizer_disc = torch.optim.AdamW(self.model_disc.parameters(),
                                        lr=hparams.get('disc_lr', hparams['lr']),
                                        betas=[hparams['adam_b1'], hparams['adam_b2']])
        return [optimizer_gen, optimizer_disc]
    
    def build_scheduler(self, optimizer):
        return (
            WarmupSchedule(
                optimizer[0], lr=hparams['lr'], warmup_updates=hparams.get('warmup_updates', 0)
            ),
            WarmupSchedule(
                optimizer[1], lr=hparams.get('disc_lr', hparams['lr']), warmup_updates=hparams.get('warmup_updates', 0)
            ),
        )
        
    def _training_step(self, sample, batch_idx, optimizer_idx):
        '''
        sample里需要的东西：wavs, 
        '''
        # if self.trainer.proc_rank_local == 0 and random.random() < 0.0001:
        #     kill_void()

        sample['wavs'] = sample['wavs'].float()
        # import pdb; pdb.set_trace()
        # return None, {}

        amp_enabled = True
        # amp_dtype = torch.float16
        amp_dtype = torch.bfloat16

        y = sample['wavs']
        # y = y.reshape(-1, 2, y.shape[-1])
        loss_output = {}
        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                model_outputs = self.model_gen(y)
                
            y_ = model_outputs['recon'] # [b, 4, n]
            y_, y = trim_to_shortest(y_, y)
            if self.training and self.global_step >= hparams.get('l1_end_steps', 0):
                loss_output['l1'] = F.l1_loss(y_, y) * 0
            else:
                loss_output['l1'] = F.l1_loss(y_, y) * hparams['losses']['lambda_l1'] # 就是后面的wav loss
            
            if self.training and self.global_step >= hparams.get('disc_start_steps', 0):
                loss_dis, loss_adv, feature_matching_distance = unwrap_model(self.model_disc)['discriminator'].loss(reals=y, fakes=y_)
                # loss_output['loss_dis'] = loss_dis
                loss_output['loss_adv'] = loss_adv * hparams['losses']['lambda_adv']
                loss_output["feature_matching_distance"] = feature_matching_distance * hparams['losses']['lambda_feature_matching']
                
                mrstft_loss = self.sdstft(y_, y)
                loss_output['loss_mrstft'] = mrstft_loss * hparams['losses']['lambda_mrstft']
                

            kl_start_steps = hparams.get('kl_start_steps', 0)
            if self.global_step >= kl_start_steps:
                if 0 < self.global_step - kl_start_steps < hparams.get('kl_annealing_step', 0):
                    lambda_kl = hparams.get('lambda_kl', 0.001) * (self.global_step - kl_start_steps) / hparams.get('kl_annealing_step', 0)
                else:
                    lambda_kl = hparams.get('lambda_kl', 0.001)
                # import pdb; pdb.set_trace()
                loss_output['kl_loss'] = model_outputs['kl'] * hparams['losses']['lambda_kl']
                
            total_loss = sum(loss_output.values())
            loss_output['monitor/mu'] = model_outputs['mu'].mean().detach()
            loss_output['monitor/logvar'] = model_outputs['logvar'].mean().detach()
            
            self.y_ = y_.detach()

        else:
            #######################
            #    Discriminator    #
            #######################
            if self.global_step >= hparams.get('disc_start_steps', 0):
                if not self.training:
                    return None
                # y = y.unsqueeze(1)
                y_ = self.y_
                y_, y = trim_to_shortest(y_, y)
                
                with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                    loss_dis, loss_adv, feature_matching_distance = unwrap_model(self.model_disc)['discriminator'].loss(reals=y, fakes=y_)
                loss_output['loss_dis'] = loss_dis * hparams['losses']['lambda_dis']

            total_loss = sum(loss_output.values())


        loss_output['bs'] = sample['wavs'].shape[0]
        # import pdb; pdb.set_trace()
        # loss_output['ntokens'] = sample['wavs'].shape[0] * sample['wavs'].shape[1] // hparams['hop_size']

        return total_loss, loss_output
    
    def on_before_optimization(self, opt_idx):

        grad_norm_dict = super().on_before_optimization(opt_idx)

        if opt_idx == 0:
            # 仅对训练中的生成器参数做梯度裁剪（可能只有 decoder）
            freeze_enc = hparams.get('freeze_encoder', False)
            if freeze_enc:
                nn.utils.clip_grad_norm_(unwrap_model(self.model_gen).decoder.parameters(), hparams['generator_grad_norm'])
            else:
                nn.utils.clip_grad_norm_(self.model_gen.parameters(), hparams['generator_grad_norm'])
        else:
            nn.utils.clip_grad_norm_(self.model_disc.parameters(), hparams["discriminator_grad_norm"])

        return grad_norm_dict
    
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