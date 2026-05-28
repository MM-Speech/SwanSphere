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

from tasks.tts.scriptspeech_task import ScriptSpeechBaseTask
from modules.asr.scriptasr.build_model_utils import build_asr_text_tokenizer, build_asr_model

class MFATask(ScriptSpeechBaseTask):
    def build_model(self):
        self.mfa_vocab_size = 6800
        self.model = build_asr_model(hparams, init_pretrained=True, vocab_size=self.mfa_vocab_size, padding_idx=797)
        self.model.train()

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
        if random.random() < 0.01:
            kill_void()
        loss_output, model_out = self.run_model(sample)
        loss_weights = {
            'ce_loss': 1.0,
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

        # resample
        if hparams.get('audio_encoder_type', 'wavlm') == 'wavlm':
            wavs = sample["wavs"].float()
            wav_lengths = sample["wav_lengths"]
            if not hasattr(self, 'resampler'):
                self.resampler = torchaudio.transforms.Resample(orig_freq=hparams['audio_sample_rate'], new_freq=16000).to(wavs.device)
            wavs = self.resampler(wavs)
            wav_lengths = (wav_lengths * 16000 / hparams['audio_sample_rate']).int()
        elif hparams.get('audio_encoder_type', 'wavlm') in ['xlsr-53', 'wavlm-hf']:
            wavs = sample["wavs_w2v2"].float()
            wav_lengths = sample["wav_w2v2_lengths"]

        txt_tokens = sample['ph_timestamp']
        txt_lens = sample['ph_timestamp_len']
        txt_mask = sequence_mask(txt_lens)

        inputs = {
            'wavs': wavs,
            'wav_mask': sequence_mask(wav_lengths),
            'txt_tokens': txt_tokens,
            'txt_mask': txt_mask
        }

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            model_outputs = self.model(
                inputs,
                do_checkpoint=hparams.get('gradient_checkpointing', False)
            )

        logits = model_outputs['logits']
        labels = model_outputs['labels']
        loss_mask = model_outputs['loss_mask']
        ntokens = model_outputs['ntokens']

        loss = F.cross_entropy(logits.transpose(1, 2), labels, reduction='none')
        loss = loss * loss_mask
        loss = loss.sum() / loss_mask.sum()

        losses_out['ce_loss'] = loss
        losses_out['bs'] = loss_mask.shape[0]
        losses_out['ntokens'] = ntokens

        return losses_out, model_out

    