import argparse
import filecmp
import multiprocessing
import os
import subprocess
import librosa
from functools import partial
from multiprocessing import Pool, Process

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW

from modules.vocoder.commons.stft_loss import MultiResolutionSTFTLoss
from modules.vocoder.hifigan.hifigan import MultiPeriodDiscriminator, MultiScaleDiscriminator, \
    generator_loss, feature_loss, discriminator_loss
from modules.vocoder.hifigan.mel_utils import mel_spectrogram
from modules.vocoder.univnet.mrd import MultiResolutionDiscriminator
from modules.tts.wavvae.decoder.wavvae_v3 import WavVAE_V3
from utils.audio import torch_wav2spec
from utils.audio.align import mel2token_to_dur
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams

from attrdict import AttrDict
from tasks.tts.dataset_utils.dataset_mixin import FastDatasetMixin, TTSDatasetMixin
from utils.commons.base_task import BaseTask
from utils.commons.import_utils import import_module_bystr
from utils.nn.schedulers import WarmupSchedule, CosineSchedule
from utils.nn.model_utils import unwrap_model


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
        if hparams.get('model_version') == 'v8':
            from modules.vae.wavvae_v8 import WavVAE_V8
            self.model_gen = WavVAE_V8(hparams=hparams)
        else:
            self.model_gen = WavVAE_V3(hparams=hparams)

        self.model_disc = torch.nn.ModuleDict()
        self.model_disc['mpd'] = MultiPeriodDiscriminator(hparams['mpd'], use_cond=hparams['use_cond_disc'])
        self.model_disc['msd'] = MultiScaleDiscriminator(use_cond=hparams['use_cond_disc'])
        if hparams['use_mrd']:
            self.model_disc['mrd'] = MultiResolutionDiscriminator(hparams)
        self.stft_loss = MultiResolutionSTFTLoss()

        load_ckpt_gen = hparams.get('load_ckpt_gen', './checkpoints/1231_megatts3_wavvae_v3_25hz_kl001_fix4')
        load_ckpt_disc = hparams.get('load_ckpt_disc', './checkpoints/1117_melgan-nsf_full_1')
        if '1117_melgan-nsf_full_1' in load_ckpt_gen:
            load_ckpt(self.model_gen.decoder, load_ckpt_gen, 'model_gen', force=True, strict=True)
        else:
            load_ckpt(self.model_gen, load_ckpt_gen, 'model_gen', strict=False)
        load_ckpt(self.model_disc, load_ckpt_disc, 'model_disc', force=True, strict=True)

        # 新增：根据开关冻结 encoder，仅训练 decoder
        freeze_enc = hparams.get('freeze_encoder', False)
        if freeze_enc:
            for p in self.model_gen.encoder.parameters():
                p.requires_grad = False

        # 判别器依然训练
        if hparams['use_mrd']:
            disc_trainables = [self.model_disc['mpd'], self.model_disc['msd'], self.model_disc['mrd']]
        else:
            disc_trainables = [self.model_disc['mpd'], self.model_disc['msd']]

        return {'trainable': [self.model_gen] + disc_trainables, 'others': []}

    def load_model(self):
        if hparams.get('load_ckpt', '') != '':
            load_ckpt(self.model_gen, hparams['load_ckpt'], 'model_gen', strict=False)

    def build_optimizer(self):
        # 根据开关仅优化 decoder 或整套生成器
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
        log_outputs = {}
        loss_weights = {}
        sample['wavs'] = sample['wavs'].float()
        # return None, {}

        y = sample['wavs']
        loss_output = {}
        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=False):
                y_, posterior = self.model_gen(y)
            y = y.unsqueeze(1)
            y_mel = mel_spectrogram(y.squeeze(1), hparams).transpose(1, 2)
            y_hat_mel = mel_spectrogram(y_.squeeze(1), hparams).transpose(1, 2)
            loss_output['mel'] = F.l1_loss(y_hat_mel, y_mel) * hparams['lambda_mel']
            if self.training and self.global_step >= hparams.get('disc_start_steps', 0):
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=False):
                    _, y_p_hat_g, fmap_f_r, fmap_f_g = self.model_disc['mpd'](y, y_, None)
                    _, y_s_hat_g, fmap_s_r, fmap_s_g = self.model_disc['msd'](y, y_, None)
                loss_output['a_p'] = generator_loss(y_p_hat_g) * hparams['lambda_adv'] * hparams.get('lambda_mpd', 1.0)
                loss_output['a_s'] = generator_loss(y_s_hat_g) * hparams['lambda_adv'] * hparams.get('lambda_msd', 1.0)
                if hparams['use_mrd']:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=False):
                        y_r_hat_g = [x[1] for x in self.model_disc['mrd'](y_)]
                    loss_output['a_r'] = generator_loss(y_r_hat_g) \
                                        * hparams['lambda_adv'] * hparams.get('lambda_mrd', 1.0)
                if hparams['use_ms_stft']:
                    loss_output['sc'], loss_output['mag'] = self.stft_loss(y.squeeze(1), y_.squeeze(1))
                
            kl_start_steps = hparams.get('kl_start_steps', 0)
            if self.global_step >= kl_start_steps:
                loss_output['kl_loss'] = posterior.kl().mean() * hparams.get('lambda_kl', 1.0)

            loss_output['monitor/mu'] = posterior.mean.mean().detach()
            loss_output['monitor/logvar'] = posterior.logvar.mean().detach()

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
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=False):
                    y_p_hat_r, y_p_hat_g, _, _ = self.model_disc['mpd'](y, y_.detach(), None)
                loss_output['r_p'], loss_output['f_p'] = discriminator_loss(y_p_hat_r, y_p_hat_g)
                # MSD
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=False):
                    y_s_hat_r, y_s_hat_g, _, _ = self.model_disc['msd'](y, y_.detach(), None)
                loss_output['r_s'], loss_output['f_s'] = discriminator_loss(y_s_hat_r, y_s_hat_g)
                # MRD
                if hparams['use_mrd']:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=False):
                        y_r_hat_r = [x[1] for x in self.model_disc['mrd'](y)]
                        y_r_hat_g = [x[1] for x in self.model_disc['mrd'](y_.detach())]
                    loss_output['r_r'], loss_output['f_r'] = discriminator_loss(y_r_hat_r, y_r_hat_g)

        total_loss = sum(loss_output.values())
        loss_output['bs'] = sample['wavs'].shape[0]
        loss_output['ntokens'] = sample['wavs'].shape[0] * sample['wavs'].shape[1] // hparams['hop_size']

        return total_loss, loss_output

    def save_valid_result(self, sample, batch_idx, model_out):
        sr = hparams['audio_sample_rate']
        mel_out = model_out.get('mel_out')
        f0 = sample.get('f0')
        f0_gt = sample.get('f0')
        if f0 is not None:
            f0_gt = f0_gt.cpu()[-1]
        if mel_out is not None:
            f0_pred = self.predict_f0(sample['mels'])
            self.plot_mel(batch_idx, sample['mels'], mel_out, f0s={'f0': f0_pred, 'f0g': f0_gt})
        # gt wav
        if self.global_step <= hparams['valid_infer_interval']:
            mel_gt = sample['mels'][-1].cpu()
            f0 = self.predict_f0(sample['mels'][-1:])
            wav_gt = self.vocoder.spec2wav(mel_gt, f0=f0)
            self.logger.add_audio(f'wav_gt_{batch_idx}', wav_gt, self.global_step, sr)

        if self.global_step >= 0:
            # with gt duration
            model_out = self.run_model(sample, infer=True, infer_use_gt_dur=True)
            # dur_info = self.get_plot_dur_info(sample, model_out)
            # del dur_info['dur_pred']
            dur_info = None

            f0 = self.predict_f0(model_out['mel_out'])
            wav_pred = self.vocoder.spec2wav(model_out['mel_out'][-1].cpu(), f0=f0)
            self.logger.add_audio(f'wav_gdur_{batch_idx}', wav_pred, self.global_step, sr)
            self.plot_mel(batch_idx, sample['mels'][-1:], model_out['mel_out'][-1], f'mel_gdur_{batch_idx}',
                          dur_info=dur_info, f0s={'f0': f0, 'f0g': f0_gt})

            # with pred duration
            if not hparams['use_gt_dur'] and not hparams['use_gt_latent']:
                model_out = self.run_model(sample, infer=True, infer_use_gt_dur=False)
                # dur_info = self.get_plot_dur_info(sample, model_out)
                dur_info = None
                f0 = self.predict_f0(model_out['mel_out'])
                self.plot_mel(
                    batch_idx, sample['mels'], model_out['mel_out'][-1], f'mel_pdur_{batch_idx}',
                    dur_info=dur_info, f0s={'f0': f0, 'f0g': f0_gt})
                wav_pred = self.vocoder.spec2wav(model_out['mel_out'][-1].cpu(), f0=f0)
                self.logger.add_audio(f'wav_pdur_{batch_idx}', wav_pred, self.global_step, sr)

    def get_plot_dur_info(self, sample, model_out):
        T_txt = sample['txt_tokens'].shape[1]
        dur_gt = mel2token_to_dur(sample['mel2ph'], T_txt)[-1]
        dur_pred = model_out['dur'] if 'dur' in model_out else dur_gt
        txt = self.token_encoder.decode(sample['txt_tokens'][-1].cpu().numpy())
        txt = txt.split(" ")
        return {'dur_gt': dur_gt, 'dur_pred': dur_pred, 'txt': txt}

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