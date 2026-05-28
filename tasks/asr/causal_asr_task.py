import os
import random

from attrdict import AttrDict
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

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

class CausalASRTask(ScriptSpeechBaseTask):
    def build_model(self):
        self.asr_text_tokenizer, self.asr_vocab_size = build_asr_text_tokenizer()
        self.model = build_asr_model(hparams, self.asr_text_tokenizer, init_pretrained=True)
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
        if 'spk_loss' in loss_output:
            loss_weights['spk_loss'] = 1.0
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

        text = sample['text']
        device = wavs.device
    
        # text tokenize
        text = ['<BOT>' + text_ + '<|endoftext|>' for text_ in text]
        if random.random() < hparams.get('print_text_prob', 0.001):
            print(f'step {self.global_step}:', text[0])
            # if 'spk_mask' in sample and sample['spk_mask'] is not None:
            #     os.makedirs(f"{hparams['work_dir']}/figs", exist_ok=True)
            #     fig = plt.figure()
            #     ax = fig.add_subplot(111)
            #     im = ax.imshow(sample['spk_mask'][0, ::320].repeat_interleave(100, dim=1).cpu().numpy())
            #     # print(sample['spk_mask'][0, ::320].repeat(1, 100).cpu().numpy())
            #     plt.colorbar(im)
            #     plt.savefig(f"{hparams['work_dir']}/figs/{self.global_step}.png")
        text_inputs = self.asr_text_tokenizer(text, padding=True, return_tensors="pt").to(device)
        txt_tokens = text_inputs['input_ids']   # [B, T]
        txt_mask = text_inputs['attention_mask'].bool()

        inputs = {
            'wavs': wavs,
            'wav_mask': sequence_mask(wav_lengths),
            'txt_tokens': txt_tokens,
            'txt_mask': txt_mask
        }
        if hparams.get('model_spk_diarization', False):
            inputs['spk_mask'] = sample['spk_mask']

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            model_outputs = self.model(inputs)

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

        if hparams.get('model_spk_diarization', False):
            spk_loss = model_outputs['spk_mask_loss']
            losses_out['spk_loss'] = spk_loss

        return losses_out, model_out

    