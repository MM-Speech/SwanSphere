import os
import random
import math

from attrdict import AttrDict
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
import torch.distributed

from utils.commons.ckpt_utils import load_ckpt
from utils.commons.import_utils import import_module_bystr
from utils.commons.hparams import hparams, set_hparams
from utils.nn.seq_utils import sequence_mask, add_prefix
from utils.nn.model_utils import print_arch, num_params, unwrap_model
from utils.commons.os_utils import kill_void
from utils.commons.dataset_utils import data_loader, build_dataloader
from utils.commons.trainer import LOCAL_RANK
from utils.audio.mel import MelNet

from tasks.tts.scriptspeech_task import ScriptSpeechBaseTask
from modules.asr.diarization.e2e_model import build_diarization_model
from tasks.asr.task_utils.spk_diarization_utils import pit_bce_loss_bruteforce, pit_bce_loss_hungarian

class SpkDiarizationTask(ScriptSpeechBaseTask):
    def build_model(self):
        self.mel_net = MelNet(hparams)
        self.mel_net.to(self.trainer.device)
        self.model = build_diarization_model(hparams)
        self.model.train()
        
        if hparams.get('audio_encoder_type') == 'wavlm':    # checkpoints/wavlm/WavLM-Large.pt
            from modules.asr.wavlm.WavLM import WavLM, WavLMConfig
            checkpoint = torch.load(hparams.get('audio_encoder_ckpt'))
            cfg = WavLMConfig(checkpoint['cfg'])
            model = WavLM(cfg)
            print(f"{cfg = }")
            model.load_state_dict(checkpoint['model'])
            model.to(self.trainer.device)
            model.eval()
            self.audio_encoder = model
            self.audio_encoder_hopsize = 320
            self.audio_encoder_sample_rate = 16000
            self.resampler = torchaudio.transforms.Resample(orig_freq=24000, new_freq=16000).to(self.trainer.device)
        
        return {'trainable': [self.model], 'others': []}

    def load_model(self):
        if hparams.get('load_ckpt', '') != '':
            load_ckpt(self.model, hparams['load_ckpt'], 'model', strict=False)

    def build_optimizer(self):
        optimizer = AdamW(unwrap_model(self.model).parameters(), **self.config.optimizer)
        return optimizer

    def fsdp_optm2model(self):
        return [self.model]
    
    def _training_step(self, sample, batch_idx, optimizer_idx):
        if self.trainer.proc_rank_local == 0 and random.random() < 0.01:
            kill_void()
        loss_output, model_out = self.run_model(sample)
        loss_weights = {
            'bce': hparams.get('lambda_bce', 1.0),
            'cov': 1.0,
            'dist': 1.0,
        }
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if
                          isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['total_loss'] = total_loss.item()

        return total_loss, loss_output
    
    def forward_audio_encoder(self, wavs, wav_mask):
        unet_stride = int(np.prod(unwrap_model(self.model).config.unet_updown_rates))
        
        if hparams.get('audio_encoder_type') == 'wavlm':
            
            tgt_len = wavs.shape[1] // hparams['hop_size'] // unet_stride
            
            wavs = self.resampler(wavs)
            wav_mask = sequence_mask((wav_mask.sum(1) / 3 * 2).long())
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    feat, feat_padding_mask = self.audio_encoder.extract_features(wavs, padding_mask=~(wav_mask.bool()))    # [B, T, C]
            feat_mask = ~feat_padding_mask  # [B, T]
            
            feat = F.interpolate(feat.transpose(1, 2), size=tgt_len).transpose(1, 2)
            feat_mask = F.interpolate(feat_mask[:, None, :].float(), size=tgt_len)[:, 0, :].bool()

        return feat, feat_mask
    
    def run_model(self, sample, infer=False, infer_steps=None):
        model_out = {}
        losses_out = {}

        if infer:
            return losses_out, model_out
        if 'wavs' not in sample:
            return losses_out, model_out
        
        wavs = sample['wavs']           # [B, T]
        wav_lengths = sample["wav_lengths"]
        wav_mask = sequence_mask(wav_lengths)
        spk_mask = sample['spk_mask']
        device = wavs.device
        
        with torch.no_grad():
            mels = self.mel_net(wavs)   # [B, T, C]

        hop_size = hparams['hop_size']
        mel_lengths = (wav_lengths / hop_size).long()
        mel_mask = sequence_mask(mel_lengths).float()
        spk_mask = spk_mask[:, ::hop_size]  # [B, T, K]
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if hparams.get('audio_encoder_type') == 'wavlm':
                audio_feat, audio_feat_mask = self.forward_audio_encoder(wavs, wav_mask)
                logits, distill_loss = self.model(mels, mel_mask, audio_feat)
                losses_out['dist'] = distill_loss
            else:
                logits = self.model(mels, mel_mask)

        if hparams.get('max_n_spk_diarization', 8) <= 4:
            bce_loss = pit_bce_loss_bruteforce(
                logits, spk_mask, lengths=mel_lengths, pos_weight=2.0
            )
        else:
            bce_loss, _ = pit_bce_loss_hungarian(
                logits, spk_mask, lengths=mel_lengths, pos_weight=2.0
            )
        
        p = torch.sigmoid(logits)
        n_bt = spk_mask.sum(dim=-1)  # [B,T]
        cov = (p.sum(dim=-1) - n_bt) ** 2 * mel_mask
        cov_loss = cov.mean()
        
        losses_out['bce'] = bce_loss
        losses_out['cov'] = cov_loss
        losses_out['bs'] = logits.shape[0]
        losses_out['monitor/max_n_spk'] = (spk_mask.sum(1) > 0).long().sum(-1).max()
        losses_out['ntokens'] = mel_lengths.sum()

        return losses_out, model_out
    
