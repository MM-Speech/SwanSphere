import argparse
import filecmp
import multiprocessing
import os
import subprocess
import librosa
from functools import partial
from multiprocessing import Pool, Process
import random

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
from modules.vocoder.commons.stft_loss import MultiResolutionSTFTLoss
from modules.vocoder.hifigan.hifigan import MultiPeriodDiscriminator, MultiScaleDiscriminator, \
    generator_loss, feature_loss, discriminator_loss
from modules.vocoder.hifigan.mel_utils import mel_spectrogram
from modules.vocoder.univnet.mrd import MultiResolutionDiscriminator


class WavVAETask(FastDatasetMixin, BaseTask):
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
        sample_rate = hparams["audio_sample_rate"]
        fft_size = hparams["win_size"]
        win_size = hparams["win_size"]
        hop_size = hparams["hop_size"]
        num_mels = hparams["audio_num_mel_bins"]
        fmin = hparams["fmin"]
        fmax = hparams["fmax"]
        mel_basis = librosa.filters.mel(
            sr=sample_rate, n_fft=fft_size, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        self.torch_wav2spec_ = partial(
            torch_wav2spec, mel_basis=mel_basis, fft_size=fft_size, hop_size=hop_size, win_length=win_size,
        )

    def build_model(self):
        if hparams.get('model_version', 'v5') == 'v5':
            from modules.vae.wavvae_v5 import build_wavvae
            self.model_gen = build_wavvae(hparams=hparams, init_pretrained=True)
        elif hparams.get('model_version', 'v6') == 'v6':
            from modules.vae.wavvae_v6 import build_wavvae
            self.model_gen = build_wavvae(hparams=hparams, init_pretrained=True)
        elif hparams.get('model_version', 'v7') == 'v7':
            from modules.vae.wavvae_v7 import build_wavvae
            self.model_gen = build_wavvae(hparams=hparams, init_pretrained=not hparams.get('from_scratch', False))
        elif hparams.get('model_version', 'v7') == 'v8':
            from modules.vae.wavvae_v8 import build_wavvae
            self.model_gen = build_wavvae(hparams=hparams, init_pretrained=True)

        if hparams.get('train_bottleneck_only', False):
            frozen = 0
            for p in self.model_gen.encoder.parameters():
                p.requires_grad = False
                frozen += 1
            for p in self.model_gen.decoder.parameters():
                p.requires_grad = False
                frozen += 1
            print_once(f"| Freeze encoder and decoder for {frozen} params, only train bottleneck")

        self.model_disc = torch.nn.ModuleDict()
        self.model_disc['mpd'] = MultiPeriodDiscriminator(hparams['mpd'], use_cond=hparams['use_cond_disc'])
        self.model_disc['msd'] = MultiScaleDiscriminator(use_cond=hparams['use_cond_disc'])
        if hparams['use_mrd']:
            self.model_disc['mrd'] = MultiResolutionDiscriminator(hparams)
        load_ckpt_disc = hparams.get('load_ckpt_disc', './checkpoints/1117_melgan-nsf_full_1')
        load_ckpt(self.model_disc, load_ckpt_disc, 'model_disc', force=True, strict=True)

        self.stft_loss = MultiResolutionSTFTLoss()

        return {'trainable': [self.model_gen, self.model_disc], 'others': []}

    def load_model(self):
        if hparams.get('load_ckpt', '') != '':
            load_ckpt(self.model_gen, hparams['load_ckpt'], 'model_gen', strict=False)

    def fsdp_optm2model(self):
        # FIXME
        return [self.model_gen]

    def fsdp_wrap_policy(self):
        from modules.vae.wavvae_v5 import EncoderBlock, DecoderBlock
        from modules.codec.fish.modded_dac import TransformerBlock

        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            model_blocks = (
                EncoderBlock,
                DecoderBlock,
                TransformerBlock
            )
            return recurse or isinstance(module, model_blocks)

        return custom_auto_wrap_policy

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
        if self.trainer.proc_rank_local == 0 and random.random() < 0.0001:
            kill_void()

        sample['wavs'] = sample['wavs'].float()
        # return None, {}

        amp_enabled = True
        # amp_dtype = torch.float16
        amp_dtype = torch.bfloat16

        y = sample['wavs']
        loss_output = {}
        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                model_outputs = self.model_gen(y)

            y_ = model_outputs['recon']     # [B, 1, T]
            y = y.unsqueeze(1)
            y_mel = mel_spectrogram(y.squeeze(1), hparams).transpose(1, 2)
            y_hat_mel = mel_spectrogram(y_.squeeze(1), hparams).transpose(1, 2)
            loss_output['mel'] = F.l1_loss(y_hat_mel, y_mel) * hparams['lambda_mel']
            loss_output['wav'] = F.l1_loss(y_, y) * hparams['lambda_wav']

            if self.training and self.global_step >= hparams.get('disc_start_steps', 0):
                with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                    _, y_p_hat_g, fmap_f_r, fmap_f_g = unwrap_model(self.model_disc)['mpd'](y, y_, None)
                    _, y_s_hat_g, fmap_s_r, fmap_s_g = unwrap_model(self.model_disc)['msd'](y, y_, None)
                loss_output['a_p'] = generator_loss(y_p_hat_g) * hparams['lambda_adv'] * hparams.get('lambda_mpd', 1.0)
                loss_output['a_s'] = generator_loss(y_s_hat_g) * hparams['lambda_adv'] * hparams.get('lambda_msd', 1.0)
                if hparams['use_mrd']:
                    with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                        y_r_hat_g = [x[1] for x in unwrap_model(self.model_disc)['mrd'](y_)]
                    loss_output['a_r'] = generator_loss(y_r_hat_g) \
                                        * hparams['lambda_adv'] * hparams.get('lambda_mrd', 1.0)
                if hparams['use_ms_stft']:
                    loss_output['sc'], loss_output['mag'] = self.stft_loss(y.squeeze(1), y_.squeeze(1))
                
            kl_start_steps = hparams.get('kl_start_steps', 0)
            if self.global_step >= kl_start_steps:
                if 0 < self.global_step - kl_start_steps < hparams.get('kl_annealing_step', 0):
                    lambda_kl = hparams.get('lambda_kl', 0.001) * (self.global_step - kl_start_steps) / hparams.get('kl_annealing_step', 0)
                else:
                    lambda_kl = hparams.get('lambda_kl', 0.001)
                loss_output['kl_loss'] = model_outputs['kl'] * lambda_kl
                loss_output['monitor/lambda_kl'] = lambda_kl

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
                y = y.unsqueeze(1)
                y_ = self.y_
                # MPD
                with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                    y_p_hat_r, y_p_hat_g, _, _ = unwrap_model(self.model_disc)['mpd'](y, y_.detach(), None)
                loss_output['r_p'], loss_output['f_p'] = discriminator_loss(y_p_hat_r, y_p_hat_g)
                # MSD
                with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                    y_s_hat_r, y_s_hat_g, _, _ = unwrap_model(self.model_disc)['msd'](y, y_.detach(), None)
                loss_output['r_s'], loss_output['f_s'] = discriminator_loss(y_s_hat_r, y_s_hat_g)
                # MRD
                if hparams['use_mrd']:
                    with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                        y_r_hat_r = [x[1] for x in unwrap_model(self.model_disc)['mrd'](y)]
                        y_r_hat_g = [x[1] for x in unwrap_model(self.model_disc)['mrd'](y_.detach())]
                    loss_output['r_r'], loss_output['f_r'] = discriminator_loss(y_r_hat_r, y_r_hat_g)

        total_loss = sum(loss_output.values())
        loss_output['bs'] = sample['wavs'].shape[0]
        loss_output['ntokens'] = sample['wavs'].shape[0] * sample['wavs'].shape[1] // hparams['hop_size']

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