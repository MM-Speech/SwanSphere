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

from tasks.tts.scriptspeech_task import ScriptSpeechBaseTask
from modules.asr.scriptasr.build_model_utils import build_asr_text_tokenizer, build_asr_model

class SpkDiarizationTask(ScriptSpeechBaseTask):
    def build_model(self):
        from utils.audio.mel import MelNet
        self.mel_net = MelNet(hparams)

        if hparams.get('model_name', 'simple') == 'simple':
            from modules.asr.spk_embed.spk_embed_model import SpkEmbed
            self.model = SpkEmbed(
                n_mels=hparams['audio_num_mel_bins'],
                hidden_size=hparams.get('hidden_size', 1024),
                wav_dowmsamples=[6, 5, 4, 2],
            )
        elif hparams.get('model_name', 'simple') == 'campplus':
            from modules.asr.spk_embed.campplus import CAMPPlus
            self.model = CAMPPlus(
                input_size=hparams['audio_num_mel_bins'], embd_dim=hparams.get('hidden_size', 1024),
                growth_rate=32, bn_size=4, init_channels=128, config_str='batchnorm-relu'
            )
        elif hparams.get('model_name', 'simple') == 'eres2net':
            from modules.asr.spk_embed.eres2net import ERes2Net
            self.model = ERes2Net(
                input_size=hparams['audio_num_mel_bins'], embd_dim=hparams.get('hidden_size', 1024), m_channels=32
            )

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
            'clip_loss': 1.0,
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
        
        wavs = sample['wavs']           # [B, T]
        wav_lengths = sample["wav_lengths"]
        spk_ids = sample['spk_ids']     # [B]
        spk_names = sample['spk_names']
        voiced = sample['voiced']       # [B, T]
        device = wavs.device

        # spk_win_size = hparams.get('spk_win_size', 12000)
        # spk_hop_size = hparams.get('spk_hop_size', 6000)
        # wav_mask = sequence_mask(wav_lengths).float()   # [B, T]
        # wav_mask = F.unfold(wav_mask.unsqueeze(1).unsqueeze(-1), (spk_win_size, 1), padding=(0, 0), stride=spk_hop_size)    # [B, t, N]
        # spk_ids = spk_ids.repeat_interleave(wav_mask.shape[-1])
        # wav_mask = rearrange(wav_mask, 'b t n -> (b n) t')
        # wavs = F.unfold(wavs.unsqueeze(1).unsqueeze(-1), (spk_win_size, 1), padding=(0, 0), stride=spk_hop_size)    # [B, t, N]
        # wavs = rearrange(wavs, 'b t n -> (b n) t')
        # wavs = wavs[(wav_mask > 0).all(dim=1), :]
        # voiced = F.unfold(voiced.unsqueeze(1).unsqueeze(-1), (spk_win_size, 1), padding=(0, 0), stride=spk_hop_size)    # [B, t, N]
        # voiced = rearrange(voiced, 'b t n -> (b n) t')
        # voiced = voiced[(wav_mask > 0).all(dim=1), :]
        # spk_ids = spk_ids[(wav_mask > 0).all(dim=1)]

        if self.mel_net.device != device:
            self.mel_net.to(device)
        with torch.no_grad():
            mels = self.mel_net(wavs)   # [B, T, C]

        if hparams.get('model_name', 'simple') == 'simple':
            spk_embeds, logit_scale = self.model(wavs, mels)     # [B, C]
        elif hparams.get('model_name', 'simple') in ['campplus', 'eres2net']:
            spk_embeds, logit_scale = self.model(mels)     # [B, C]

        # print('spk_embeds.shape', spk_embeds.shape)

        local_loss = True
        world_size = self.trainer.num_local_gpus
        rank = self.trainer.proc_rank_local
        if world_size > 1:
            all_spk_embeds = [torch.zeros_like(spk_embeds) for _ in range(world_size)]
            all_spk_names = [None for _ in range(world_size)]
            all_voiced = [torch.zeros_like(voiced) for _ in range(world_size)]
            torch.distributed.all_gather(all_spk_embeds, spk_embeds)
            torch.distributed.all_gather_object(all_spk_names, spk_names)
            torch.distributed.all_gather(all_voiced, voiced)
            all_spk_embeds[rank] = spk_embeds
            all_spk_embeds = torch.cat(all_spk_embeds, dim=0)
            all_voiced = torch.cat(all_voiced, dim=0)
            spk_map = {}
            all_spk_ids = []
            spk_ids = []
            for rank_i in range(world_size):
                for spk_name in all_spk_names[rank_i]:
                    if spk_name not in spk_map:
                        spk_map[spk_name] = len(spk_map)
                    all_spk_ids.append(spk_map[spk_name])
                    if rank_i == rank:
                        spk_ids.append(spk_map[spk_name])
            all_spk_ids = torch.LongTensor(all_spk_ids).to(device)
            spk_ids = torch.LongTensor(spk_ids).to(device)

            if local_loss:
                
                logits = logit_scale * spk_embeds @ all_spk_embeds.T        # [B, 8B]

                labels = spk_ids[..., None] == all_spk_ids[..., None].T # [B, 8B]
                voiced_ratio = voiced.mean(dim=1)   # [B]
                all_voiced_ratio = all_voiced.mean(dim=1)   # [8B]
                all_voiced_mask = voiced_ratio[..., None] * all_voiced_ratio[..., None].T   # [B, 8B]
                labels = labels.to(logits) * all_voiced_mask    # [B, 8B]
                # print(list(logits.shape), list(labels.shape), labels.mean().item(), logits.mean().item())

                pos_mask = labels > 0   # [B, 8B]
                num_pos = pos_mask.sum().float()
                num_neg = torch.numel(labels) - num_pos
                num_pos = num_pos - pos_mask.diag().sum()   # remove diagonal
                # pos_weight = num_neg / (num_pos + 1e-7)
                # neg_weight = num_pos / (num_neg + 1e-7)
                pos_weight = (num_neg / (num_pos + 1e-7))**0.5
                neg_weight = 1 / (pos_weight + 1e-7)
                weights = torch.where(pos_mask, pos_weight, neg_weight)
                clip_loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights, reduction='none')    # [B, 8B]
                loss_mask = 1 - torch.eye(clip_loss.shape[0], clip_loss.shape[1]).to(clip_loss)     # remove diagonal
                clip_loss = clip_loss * loss_mask
                clip_loss = clip_loss.sum() / loss_mask.sum()
                n_spk = len(spk_map)

            else:
                logits = logit_scale * all_spk_embeds @ all_spk_embeds.T    # [8B, 8B]

                labels = all_spk_ids[..., None] == all_spk_ids[..., None].T # [8B, 8B]
                all_voiced_ratio = all_voiced.mean(dim=1)   # [8B]
                all_voiced_mask = all_voiced_ratio[..., None] * all_voiced_ratio[..., None].T
                labels = labels.to(logits) * all_voiced_mask

                clip_loss = F.binary_cross_entropy_with_logits(logits, labels)
                n_spk = len(spk_map)

        else:
            logits = logit_scale * spk_embeds @ spk_embeds.T    # [B, B]

            labels = spk_ids[..., None] == spk_ids[..., None].T # [B, B]
            voiced_ratio = voiced.mean(dim=1)
            voiced_mask = voiced_ratio[..., None] * voiced_ratio[..., None].T
            labels = labels.to(logits) * voiced_mask

            clip_loss = F.binary_cross_entropy_with_logits(logits, labels)
            n_spk = spk_ids.max().item() + 1
        
        losses_out['clip_loss'] = clip_loss
        losses_out['bs'] = logits.shape[0]
        losses_out['logit_scale'] = logit_scale.detach().item()
        losses_out['n_spk'] = n_spk

        return losses_out, model_out

    # def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        
    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        super().on_after_optimization(epoch, batch_idx, optimizer, optimizer_idx)
        with torch.no_grad():
            unwrap_model(self.model).logit_scale.clamp_(0, math.log(100))
    