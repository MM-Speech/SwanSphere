import random

from attrdict import AttrDict
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from tasks.tts.dataset_utils.dataset_mixin import TTSDatasetMixin
from utils.commons.base_task import BaseTask
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.import_utils import import_module_bystr
from utils.commons.hparams import hparams, set_hparams
from utils.nn.schedulers import WarmupSchedule, CosineSchedule, CosineAnnealingWarmRestartsWithWarmup
from utils.nn.seq_utils import sequence_mask, weights_nonzero_speech
from utils.nn.seq_utils import sequence_mask
from utils.nn.model_utils import print_arch, num_params, unwrap_model
from utils.commons.os_utils import kill_void
from utils.commons.dataset_utils import data_loader, build_dataloader
from utils.commons.trainer import LOCAL_RANK


class Qwen3TTSMimiTask(TTSDatasetMixin, BaseTask):
    def __init__(self):
        self.dataset_cls = import_module_bystr(hparams['dataset_cls'])
        self.val_dataset_cls = import_module_bystr(hparams['val_dataset_cls'])
        self.processer_fn = import_module_bystr(hparams['processer_fn'])
        self.build_fast_dataloader = import_module_bystr(hparams['build_fast_dataloader'])
        self.hparams = hparams
        self.config = AttrDict(hparams)

        super().__init__()

    def build_model(self):
        self.build_tokenizer()

        from transformers import AutoConfig, AutoModelForCausalLM
        from utils.nn.embedding import resize_embedding_layer
        backbone_name = hparams.get('backbone_name', "Qwen/Qwen3-0.6B")
        backbone = AutoModelForCausalLM.from_pretrained(backbone_name, attn_implementation="flash_attention_2")
        resize_embedding_layer(backbone, len(self.text_tokenizer))

        decoder_name = hparams.get('decoder_name', "Qwen/Qwen3-0.6B")
        decoder_config = AutoConfig.from_pretrained(decoder_name)
        decoder_config.head_dim = 64
        decoder_config.hidden_size = 512
        decoder_config.intermediate_size = 2048
        decoder_config.num_attention_heads = 8
        decoder_config.num_hidden_layers = 16
        decoder_config.num_key_value_heads = 8
        if 'qwen3' in decoder_name.lower():
            from transformers.models.qwen3 import Qwen3ForCausalLM
            decoder = Qwen3ForCausalLM._from_config(decoder_config, attn_implementation="flash_attention_2")
        elif 'qwen2' in decoder_name.lower():
            from transformers import Qwen2ForCausalLM
            decoder = Qwen2ForCausalLM._from_config(decoder_config, attn_implementation="flash_attention_2")
        resize_embedding_layer(decoder, 2048 * 31 + 2)

        from modules.tts.codeclm.qwen_tts import Qwen3TTSMimiModel
        self.model = Qwen3TTSMimiModel(
            backbone=backbone,
            decoder=decoder,
            text_acoustic_token=2048 * 31,
            audio_codebook_size=2048,
            acoustic_n_quantizers=31,
            semantic_start_idx=self.text_tokenizer.encode('<Reserved_TTS_0>')[0],
            eos_idx=self.text_tokenizer.encode('<|endoftext|>')[0],
            backbone_padding_idx=self.text_tokenizer.encode('<|endoftext|>')[0],
            decoder_padding_idx=2048 * 31 + 1,
            decoder_frame_ratio=hparams.get('decoder_frame_ratio', 1/16),
            decoder_proj_bias=hparams.get('decoder_proj_bias', True),
            acoustic_encoder_zero_init=hparams.get('acoustic_encoder_zero_init', True)
        )
        self.model.train()
        if hparams.get('gradient_checkpointing', False):
            self.model.backbone.model.gradient_checkpointing_enable()
            self.model.decoder.model.gradient_checkpointing_enable()

        return {'trainable': [self.model], 'others': [self.audio_tokenizer]}

    def build_tokenizer(self):
        from transformers import MimiModel, AutoFeatureExtractor
        self.audio_tokenizer = MimiModel.from_pretrained("kyutai/mimi")
        self.audio_tokenizer.eval()
        for param in self.audio_tokenizer.parameters():
            param.requires_grad = False
            param.grad = None

        from transformers import AutoTokenizer, Qwen2Tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        self.text_tokenizer.add_tokens(['<BOT>'], special_tokens=True)
        self.text_tokenizer.add_tokens(['<BOS>'], special_tokens=True)
        semantic_token_num = 2048 * 1
        self.text_tokenizer.add_tokens([f'<Reserved_TTS_{i}>' for i in range(semantic_token_num)], special_tokens=True)

    def load_model(self):
        if hparams.get('load_ckpt', '') != '':
            load_ckpt(self.model, hparams['load_ckpt'], 'model', strict=False)

    def build_optimizer(self):
        optimizer = AdamW(self.model.parameters(), **self.config.optimizer)
        return optimizer

    def build_scheduler(self, optimizer):
        # if hparams.get('warmup_updates', 5000) <= 10:
        #     return None
        # return CosineSchedule(
        #     optimizer, hparams['optimizer']['lr'], hparams.get('warmup_updates', 5000), total_updates=1000000)
        return CosineAnnealingWarmRestartsWithWarmup(
            optimizer, lr_max=hparams['optimizer']['lr'], warmup_updates=hparams.get('warmup_updates', 5000), 
            total_updates=1000000, initial_period=hparams.get('scheduler_initial_period', 10000), 
            period_mult=hparams.get('scheduler_period_mult', 1.2), lr_min=hparams.get('scheduler_lr_min', 1.0e-5)
        )

    def fsdp_optm2model(self):
        return [self.model]

    def fsdp_wrap_policy(self):
        from torch.nn import Linear, Sequential, Conv1d, Conv2d, Embedding
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3Attention, Qwen3MLP

        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            model_blocks = (
                Qwen3DecoderLayer,
                Qwen3Attention,
                Qwen3MLP,
                Linear,
                Sequential,
                Conv1d,
                Conv2d,
                Embedding
            )
            return recurse or isinstance(module, model_blocks)

        return custom_auto_wrap_policy

    @data_loader
    def train_dataloader(self):
        if hparams.get('use_fast_dataloader', True):
            return super().train_dataloader()
        else:
            train_dataset = Qwen3MimiOfflineDataset(metadata_path='/mnt/bn/sa-ag-data/liruiqi/code/LLaMA-Factory-Speech/user_cot/data/sa_data_v1.json')
            return build_dataloader(
                train_dataset, shuffle=True, use_ddp=self.trainer.use_ddp, max_tokens=hparams['max_tokens'], max_sentences=hparams['max_sentences'], 
                endless=hparams['endless_ds'], is_batch_by_size=hparams['batch_by_size'], num_workers=hparams['ds_workers'], training=True, prefetch_factor=hparams['prefetch_factor']
            )

    ##########################
    # training and validation
    ##########################

    def on_epoch_start(self):
        super().on_epoch_start()
        kill_void()

    def _training_step(self, sample, batch_idx, optimizer_idx):
        if random.random() < 0.01:
            kill_void()
        loss_output, model_out = self.run_model(sample)
        loss_weights = {
            'backbone_loss': 1.0,
            'decoder_loss': 1.0,
        }
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if
                          isinstance(v, torch.Tensor) and v.requires_grad])

        return total_loss, loss_output

    def run_model(self, sample, infer=False, infer_steps=None):
        model_out = {}
        losses_out = {}

        if infer:
            return losses_out, model_out
        if 'wavs' not in sample:
            return losses_out, model_out

        device = sample["wavs"].device
        text = sample['text']
        text_inputs = self.text_tokenizer(text, padding=True, return_tensors="pt").to(device)
        text_tokens = text_inputs['input_ids']
        text_mask = text_inputs['attention_mask']
        if hparams.get('model_debug', False):
            print()
            print('='*100)
            print('text', text[0])

        wavs = sample["wavs"]
        wav_lengths = sample["wav_lengths"]
        wav_mask = sequence_mask(wav_lengths).int()
        with torch.inference_mode():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                audio_tokens = self.audio_tokenizer(
                    input_values=wavs[:, None].to(torch.float16),
                    padding_mask=wav_mask
                ).audio_codes   # [B, 32, Tw//1920]
        audio_tokens_lengths = wav_lengths // 1920
        audio_mask = sequence_mask(audio_tokens_lengths).int()

        semantic_tokens = audio_tokens[:, 0:1]
        acoustic_tokens = audio_tokens[:, 1:]   # [B, 31, Ta]
        semantic_tokens = semantic_tokens + unwrap_model(self.model).semantic_start_idx
        acoustic_tokens = acoustic_tokens + torch.arange(31)[None, :, None].to(acoustic_tokens) * 2048
        audio_tokens = torch.cat([semantic_tokens, acoustic_tokens], dim=1).transpose(1, 2)     # [B, Ta, 32]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            model_outputs = self.model(text_tokens, audio_tokens, text_mask, audio_mask)

        backbone_logits = model_outputs['backbone_logits']
        backbone_labels = model_outputs['backbone_labels']
        decoder_logits = model_outputs['decoder_logits']
        decoder_labels = model_outputs['decoder_labels']

        if hparams.get('model_debug', False):
            loss_mask = sequence_mask(text_mask.sum(1) + audio_mask.sum(1))
            print('loss_mask.shape', loss_mask.shape)
            print('backbone_logits.shape', backbone_logits.shape)
            print('text_mask.sum(1)[0]', text_mask.sum(1)[0])
            print('audio_mask.sum(1)[0]', audio_mask.sum(1)[0])
            print('-'*100)
            print('gt token   ', self.text_tokenizer.decode(semantic_tokens[0, 0].cpu().numpy().tolist()))
            print('-'*100)
            print('label      ', self.text_tokenizer.decode(backbone_labels[0, text_mask[0].sum()-1:].cpu().numpy().tolist()))
            print('-'*100)
            print('label total', self.text_tokenizer.decode(backbone_labels[0].cpu().numpy().tolist()))
            print('-'*100)
            print('label token', backbone_labels[0].cpu().numpy())
            print('-'*100)
            backbone_output = torch.argmax(backbone_logits, dim=-1)[0].detach().cpu().numpy()
            print('pred       ', self.text_tokenizer.decode(backbone_output[text_mask[0].sum()-1:].tolist()))
            print('-'*100)
            print('pred total ', self.text_tokenizer.decode(backbone_output.tolist()))
            print('-'*100)
            print('pred token ', backbone_output)

        # backbone loss
        backbone_loss = F.cross_entropy(backbone_logits.transpose(1, 2), backbone_labels, reduction='none')     # [B, T]
        loss_mask = sequence_mask(text_mask.sum(1) + audio_mask.sum(1))
        effective_n_tokens = 0
        text_loss_mask = sequence_mask(text_mask.sum(1)-1, maxlen=backbone_loss.shape[1])   # must -1, because right shift
        backbone_loss[text_loss_mask] = backbone_loss[text_loss_mask] * hparams.get('lambda_text', 0.1)
        effective_n_tokens += text_loss_mask.sum() if hparams.get('lambda_text', 0.1) > 0 else 0
        audio_loss_mask = sequence_mask(text_mask.sum(1) + audio_mask.sum(1))
        audio_loss_mask[text_loss_mask] = False
        backbone_loss[audio_loss_mask] = backbone_loss[audio_loss_mask] * hparams.get('lambda_audio', 1.0)
        effective_n_tokens += audio_loss_mask.sum()
        backbone_loss = backbone_loss * loss_mask
        backbone_loss = backbone_loss.sum() / effective_n_tokens

        if hparams.get('model_debug', False):
            print('-'*100)
            print('decoder gt  ', decoder_labels[0].cpu().numpy())
            print('-'*100)
            print('decoder pred', torch.argmax(decoder_logits[0], dim=-1).detach().cpu().numpy())

        # decoder loss
        if self.trainer.global_step >= hparams.get('decoder_loss_start_steps', -1):
            decoder_loss = F.cross_entropy(decoder_logits.transpose(1, 2), decoder_labels, reduction='mean')
        else:
            decoder_loss = 0

        losses_out['backbone_loss'] = backbone_loss
        losses_out['decoder_loss'] = decoder_loss
        losses_out['bs'] = wavs.shape[0]
        losses_out['decoder_bs'] = decoder_logits.shape[0]
        losses_out['ntokens'] = loss_mask.sum()
        return losses_out, model_out


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
    

import json
from tqdm import tqdm
import numpy as np
import librosa
from utils.commons.multiprocess_utils import chunked_multiprocess_run
from utils.text.split_text import get_word_list
from utils.commons.dataset_utils import collate_xd

def compute_sample_len(item):
    num_tokens = 0
    num_tokens += len(get_word_list(item['text'])) + 1
    num_tokens += item['duration'] * 12.5
    return int(num_tokens)

class Qwen3MimiOfflineDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_path):
        metadata_path = metadata_path.split(',')

        print('metadata_path', metadata_path)
        meta_data = []
        for metadata_path_ in metadata_path:
            with open(metadata_path_, "r", encoding="utf-8") as f:
                meta_data = meta_data + json.load(f)
        self.meta_data = meta_data
        self.backup_batch = None

        meta_data = []
        sizes = []
        for item in self.meta_data:
            if item['duration'] < 0.08:
                continue
            meta_data.append(item)
        for sample_len in tqdm(chunked_multiprocess_run(
            compute_sample_len, args=meta_data, num_workers=16), 
                total=len(meta_data), desc='computing sample lengths'):
            sizes.append(sample_len)

        print('max(self.sizes)', max(sizes))
        print('min(self.sizes)', min(sizes))
        print('np.mean(self.sizes)', np.mean(sizes))
        self.meta_data = meta_data
        self.sizes = sizes

        print('total training samples:', len(self.meta_data))

    def __getitem__(self, index):
        meta_item = self.meta_data[index]
        fm = hparams['frames_multiple']
        fm_wav = hparams['frames_multiple'] * hparams['hop_size']
        wav, sr = librosa.load(meta_item['wav_path'], sr=24000)
        wav = wav[:wav.shape[0] // fm_wav * fm_wav]
        ret = {
            'index': index,
            'wav': torch.from_numpy(wav),
            'text': '<BOT>' + meta_item['text'] + '<BOS>',
        }
        return ret

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        wavs = collate_xd([s['wav'] for s in samples], 0.0)
        wav_lengths = torch.LongTensor([s['wav'].shape[0] for s in samples])
        batch = {
            'nsamples': len(samples),
            'wavs': wavs,
            'wav_lengths': wav_lengths,
            'text': [s['text'] for s in samples]
        }

        return batch

    def __len__(self):
        return len(self.meta_data)
    
    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        indices = np.random.permutation(len(self))
        indices = indices[np.argsort(np.array(self.sizes)[indices], kind='mergesort')]
        return indices
    
    def num_tokens(self, index):
        return self.sizes[index]


