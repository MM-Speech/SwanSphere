import os
import random

from attrdict import AttrDict
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import torchaudio
import numpy as np

from utils.commons.ckpt_utils import load_ckpt
from utils.commons.import_utils import import_module_bystr
from utils.commons.hparams import hparams, set_hparams
from utils.nn.seq_utils import sequence_mask, add_prefix
from utils.nn.model_utils import print_arch, num_params, unwrap_model
from utils.commons.os_utils import kill_void
from utils.commons.dataset_utils import data_loader, build_dataloader
from utils.commons.trainer import LOCAL_RANK
from utils.commons.io import print_once

from tasks.tts.scriptspeech_task import ScriptSpeechBaseTask
from tasks.tts.dataset_utils.dataset_mixin import FastDatasetMixin

from modules.asr.tagger.model import build_tagger_model, stats_meta, TaggerTokenizer

class TaggerTask(ScriptSpeechBaseTask):
    def __init__(self):
        super(ScriptSpeechBaseTask, self).__init__()
        self.dataset_cls = import_module_bystr(hparams['dataset_cls'])
        self.train_dataloader = FastDatasetMixin.train_dataloader.__get__(self)
        self.test_dataloader = FastDatasetMixin.test_dataloader.__get__(self)
        self.val_dataloader = FastDatasetMixin.val_dataloader.__get__(self)
        self.hparams = hparams
        self.config = AttrDict(hparams)
        
    def build_model(self):
        self.model = build_tagger_model(hparams, init_pretrained=True)
        self.model.train()
        
        self.tagger_tokenizer = TaggerTokenizer()
        
        age_counts = [stats_meta['age'][self.tagger_tokenizer.decode([idx], 'age')[0]]['count'] for idx in range(len(self.tagger_tokenizer.age))]
        print(f"{age_counts = }")
        age_counts = torch.Tensor(age_counts)
        
        # 1) 简单逆频率权重（均值归一化，数值更稳）
        age_weights = age_counts.sum() / (age_counts.numel() * age_counts)
        age_weights = age_weights / age_weights.mean()
        self.age_weights = age_weights = torch.pow(age_weights, 0.5)
        # 2) 有效样本数权重（Class-Balanced）
        # beta = 0.999
        # age_weights = (1 - beta) / (1 - torch.pow(beta, age_counts))
        # self.age_weights = age_weights = age_weights / age_weights.mean()
        print_once(f"{age_weights = }")
        
        # emotion valid subsets
        self.emotion_valid_subsets = set([
            'casia', 'cremad', 'emns', 'emov-db', 'esd', 'expresso', 'iemocap', 'jlcorpus', 'm3ed', 'mead', 'meld', 'mer2023', 'msp-podcast', 'ravdess', 'ravdess', 'tess'
        ])   # (very good)
        self.emotion_valid_subsets2 = set([
            'dailytalk', 'emilia_en', 'emilia_zh', 'gigaspeech', 'hifi_tts', 'hq-conversations', 'librispeech', 'libritts_r', 'mls_english', 
        ])   # (just good)
        self.emotion_valid_subsets = self.emotion_valid_subsets | self.emotion_valid_subsets2
        self.emotion_weights = torch.ones(23)
        self.emotion_weights[:2] = 0.1
        self.emotion_weights[5:] = 1.5
        
        return {'trainable': [self.model], 'others': []}

    def load_model(self):
        if hparams.get('load_ckpt', '') != '':
            load_ckpt(self.model, hparams['load_ckpt'], 'model', strict=False)

    def build_optimizer(self):
        optimizer = AdamW(unwrap_model(self.model).parameters(), **self.config.optimizer)
        return optimizer
    
    def _training_step(self, sample, batch_idx, optimizer_idx):
        if random.random() < 0.01:
            kill_void()
        loss_output, model_out = self.run_model(sample)
        loss_weights = {
            'emotion': 1, 'pitch': 10, 'pitch_std': 100, 
        }
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if
                          isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['total_loss'] = total_loss.item()

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
        device = wavs.device

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            model_outputs = self.model(
                wavs, sequence_mask(wav_lengths),
                do_checkpoint=hparams.get('gradient_checkpointing', False)
            )

        age_logits = model_outputs['age_logits']
        gender_logits = model_outputs['gender_logits']
        emotion_logits = model_outputs['emotion_logits']
        pitch_pred = model_outputs['pitch']
        pitch_std_pred = model_outputs['pitch_std']
        speed_pred = model_outputs['speed']
        
        #######
        # age #
        #######
        age_gt = sample['age']
        age_loss = F.cross_entropy(age_logits, age_gt, weight=self.age_weights.to(device))
        
        ##########
        # gender #
        ##########
        gender_gt = sample['gender']
        gender_loss = F.cross_entropy(gender_logits, gender_gt)
        
        ##########
        # emotion #
        ##########
        emotion_gt = sample['emotion']
        emotion_valid_idxs = []
        for batch_idx, subset in enumerate(sample['subset']):
            if subset in self.emotion_valid_subsets:
                emotion_valid_idxs.append(batch_idx)
        if len(emotion_valid_idxs) > 0:
            emotion_gt = emotion_gt[emotion_valid_idxs]
            emotion_logits = emotion_logits[emotion_valid_idxs]
            emotion_loss = F.cross_entropy(emotion_logits, emotion_gt, weight=self.emotion_weights.to(device))
        else:
            emotion_loss = 0.0
        
        ##########
        # pitch #
        ##########
        pitch_gt = torch.log1p(sample['pitch'])
        pitch_loss = F.mse_loss(pitch_pred[..., 0], pitch_gt)
        
        ##########
        # pitch_std #
        ##########
        pitch_std_gt = sample['pitch_std']
        pitch_std_loss = F.mse_loss(pitch_std_pred[..., 0], pitch_std_gt)
        
        ##########
        # speed #
        ##########
        speed_gt = sample['speed']
        speed_loss = F.mse_loss(speed_pred[..., 0], speed_gt)
        
        losses_out['age'] = age_loss
        losses_out['gender'] = gender_loss
        losses_out['emotion'] = emotion_loss
        losses_out['pitch'] = pitch_loss
        losses_out['pitch_std'] = pitch_std_loss
        losses_out['speed'] = speed_loss
        
        losses_out['bs'] = wavs.shape[0]
        losses_out['ntokens'] = wav_lengths.sum() // hparams['hop_size']

        return losses_out, model_out

    