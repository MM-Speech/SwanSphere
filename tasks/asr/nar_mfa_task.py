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

class MFATask(ScriptSpeechBaseTask):
    def build_model(self):
        self.mfa_vocab_size = 800
        if hparams.get('model_version', 'v1') in ['v1', 'v2']:
            from modules.asr.mfa.nar_mfa import build_nar_mfa_model
            self.model = build_nar_mfa_model(hparams, init_pretrained=True, vocab_size=self.mfa_vocab_size)
        elif hparams.get('model_version', 'v1') == 'v3':
            from modules.asr.mfa.nar_mfa_v3 import build_nar_mfa_model
            self.model = build_nar_mfa_model(hparams, init_pretrained=True, vocab_size=self.mfa_vocab_size)
        elif hparams.get('model_version', 'v1') == 'v4':
            from modules.asr.mfa.nar_mfa_v4 import build_nar_mfa_model
            self.model = build_nar_mfa_model(hparams, init_pretrained=True)
        elif hparams.get('model_version', 'v1') == 'v5':
            from modules.asr.mfa.nar_mfa_v5 import build_nar_mfa_model
            self.model = build_nar_mfa_model(hparams, init_pretrained=True)
        elif hparams.get('model_version', 'v1') == 'v6':
            from modules.asr.mfa.nar_mfa_v6 import build_nar_mfa_model
            self.model = build_nar_mfa_model(hparams, init_pretrained=True)
        self.model.train()

        return {'trainable': [self.model], 'others': []}

    def load_model(self):
        if hparams.get('load_ckpt', '') != '':
            load_ckpt(self.model, hparams['load_ckpt'], 'model', strict=False)

    def build_optimizer(self):
        if len(hparams.get('disable_weight_decay_names', [])) <= 0:
            
            optimizer = AdamW(unwrap_model(self.model).parameters(), **self.config.optimizer)
            
        else:
            
            def has_name(names, param_name):
                if not isinstance(names, list):
                    names = [names]
                for name in names:
                    if name in param_name:
                        return True
                return False
        
            disable_weight_decay_names = hparams.get('disable_weight_decay_names', [])
                
            decay_params = []
            no_decay_params = []
            no_decay_param_names = []
            for name, param in unwrap_model(self.model).named_parameters():
                if not param.requires_grad:
                    continue
                if has_name(disable_weight_decay_names, name):
                    no_decay_params.append(param)
                    no_decay_param_names.append(name)
                else:
                    decay_params.append(param)
                    
            print_once(f"| Weight decay is canceld for {len(no_decay_params)} params:")
            for pn in no_decay_param_names:
                print_once(f"| * {pn}")

            optimizer_groups = [
                {'params': decay_params, 'weight_decay': self.config.optimizer.weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0}
            ]

            optimizer = AdamW(optimizer_groups, **self.config.optimizer)
        
        return optimizer
    
    # def on_before_optimization(self, opt_idx):
    #     for name, p in unwrap_model(self.model).named_parameters():
    #         if p.grad is None:
    #             continue
    #         if p.shape != p.grad.shape:
    #             print(f"[Shape Mismatch] {name}: param shape={tuple(p.shape)}, grad shape={tuple(p.grad.shape)}")
    #     return super().on_before_optimization(opt_idx)
    
    def fsdp_wrap_policy(self):
        from modules.asr.wavlm.WavLM import TransformerSentenceEncoderLayer
        import modules.flow_matching.llama
        import modules.asr.llama.llama_seq2seq
        import modules.tts.llama_dit.llama_ca
        import modules.asr.llama.llama

        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            model_blocks = (
                modules.flow_matching.llama.TransformerBlock,
                modules.asr.llama.llama_seq2seq.DecoderBlock,
                modules.asr.llama.llama_seq2seq.EncoderBlock,
                modules.tts.llama_dit.llama_ca.TransformerBlock,
                modules.asr.llama.llama.TransformerBlock,
                TransformerSentenceEncoderLayer
            )
            return recurse or isinstance(module, model_blocks)

        return custom_auto_wrap_policy

    def fsdp_optm2model(self):
        return [self.model]
    
    def _training_step(self, sample, batch_idx, optimizer_idx):
        if random.random() < 0.01:
            kill_void()
        loss_output, model_out = self.run_model(sample)
        loss_weights = {
            'ce_loss': hparams.get('lambda_ce_loss', 1.0),
            'bd_frame_loss': hparams.get('lambda_bd_frame_loss', 0.5),
            'bd_agg_loss': hparams.get('lambda_bd_agg_loss', 0.5),
            'total_agg_loss': hparams.get('lambda_total_agg_loss', 1.0),
            'align_loss': hparams.get('lambda_align_loss', 1.0),
            'mono_loss': hparams.get('lambda_mono_loss', 1.0),
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
        wavs = sample["wavs"].float()
        wav_lengths = sample["wav_lengths"]
        if not hasattr(self, 'resampler'):
            self.resampler = torchaudio.transforms.Resample(orig_freq=hparams['audio_sample_rate'], new_freq=16000).to(wavs.device)
        wavs = self.resampler(wavs)
        wav_lengths = (wav_lengths * 16000 / hparams['audio_sample_rate']).int()
                
        txt_tokens = sample['merged_ph_tokens'].clamp(0, 799)
        txt_lens = sample['merged_ph_tokens_len']
        txt_mask = sequence_mask(txt_lens, maxlen=txt_tokens.shape[1])
        dur = sample['dur']
        dur_len = sample['dur_len']
        dur_paraformer_label = sample['dur_paraformer_label']
        dur_paraformer_label_len = sample['dur_paraformer_label_len']
        
        inputs = {
            'wavs': wavs,
            'wav_mask': sequence_mask(wav_lengths),
            'txt_tokens': txt_tokens,
            'txt_mask': txt_mask,
            'dur': dur,
            'dur_len': dur_len,
            'dur_paraformer_label': dur_paraformer_label,
            'dur_paraformer_label_len': dur_paraformer_label_len
        }

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            model_outputs = self.model(
                inputs,
                do_checkpoint=hparams.get('gradient_checkpointing', False)
            )

        ce_loss = model_outputs['ce_loss']
        ntokens = model_outputs['ntokens']

        losses_out['ce_loss'] = ce_loss
        losses_out['bs'] = wavs.shape[0]
        losses_out['ntokens'] = ntokens
        
        if hparams.get('use_bd_frame_loss', True) and 'bd_frame_loss' in model_outputs:
            losses_out['bd_frame_loss'] = model_outputs['bd_frame_loss']
        if 'bd_agg_loss' in model_outputs:
            losses_out['bd_agg_loss'] = model_outputs['bd_agg_loss']
        if 'len_pred_loss' in model_outputs:
            losses_out['len_pred_loss'] = model_outputs['len_pred_loss']
        if 'align_loss' in model_outputs:
            losses_out['align_loss'] = model_outputs['align_loss']
        if 'mono_loss' in model_outputs:
            losses_out['mono_loss'] = model_outputs['mono_loss']
        if 'gamma' in model_outputs:
            losses_out['monitor/gamma'] = model_outputs['gamma']

        return losses_out, model_out

    