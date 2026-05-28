import os
import random
import re
import math
import traceback
import json

from attrdict import AttrDict
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.distributed.fsdp import FullyShardedDataParallel
import torch.distributed
import torchaudio
import numpy as np
from copy import deepcopy
import soundfile as sf
from tqdm import tqdm

from utils.commons.base_task import BaseTask
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.import_utils import import_module_bystr, get_class_from_module
from utils.commons.hparams import hparams, set_hparams
from utils.commons.os_utils import kill_void
from utils.commons.dataset_utils import data_loader, build_dataloader, collate_xd
from utils.commons.trainer import LOCAL_RANK
from utils.commons.io import print_once, json_dump, json_dumps
from utils.commons.tensor_utils import move_to_cpu, move_to_cuda, convert_to_np, tensors_to_scalars
from utils.commons.seq_utils import seq_match
from utils.text.text_encoder import TokenTextEncoder
from utils.nn.schedulers import WarmupSchedule, CosineSchedule, CosineAnnealingWarmRestartsWithWarmup
from utils.nn.seq_utils import sequence_mask, add_prefix, remove_prefix
from utils.nn.model_utils import print_arch, num_params, unwrap_model, freeze_by_module_name

from modules.tts.scriptspeech.build_model_utils import build_vae, build_qwen3, shard_model_in_node, \
    DiTBuildModelMixinV2, DiTBuildModelMixinV4, DiTBuildModelMixinV5, DiTBuildModelMixinV6
from tasks.tts.dataset_utils.dataset_mixin import FastDatasetMixin, TTSDatasetMixin
from tasks.tts.scriptspeech_task import ScriptSpeechBaseTask


DEBUG = True


def return_decay(step, decay_type):
    if decay_type == 0:
        flat = 0
        uprate = 0.0
        uphold = 0.0
    elif decay_type == 1:
        flat = 0
        uprate = 0.001
        uphold = 0.5
    elif decay_type == 2:
        flat = 75
        uprate = 0.0075
        uphold = 0.999
    elif decay_type == 3:
        flat = 0
        uprate = 0.1
        uphold = 0.999
    else:
        assert False

    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)

def all_gather_varlen_tensor(t, dim=0):
    ws = torch.distributed.get_world_size()
    local_len = torch.tensor([t.size(dim)], device=t.device, dtype=torch.long)
    lens = [torch.zeros_like(local_len) for _ in range(ws)]
    torch.distributed.all_gather(lens, local_len)
    lens = torch.cat(lens).cpu().tolist()
    max_len = max(lens)

    if t.size(dim) < max_len:
        pad_shape = list(t.shape)
        pad_shape[dim] = max_len - t.size(dim)
        pad = torch.zeros(pad_shape, device=t.device, dtype=t.dtype)
        t_pad = torch.cat([t, pad], dim=dim)
    else:
        t_pad = t

    gathered = [torch.zeros_like(t_pad) for _ in range(ws)]
    torch.distributed.all_gather(gathered, t_pad)
    # 截断
    chunks = []
    for r, g in enumerate(gathered):
        sl = [slice(None)] * g.ndim
        sl[dim] = slice(0, lens[r])
        chunks.append(g[tuple(sl)])
    return torch.cat(chunks, dim=dim)

def tensor_mean_per_element(t, t_mask):
    if t_mask.ndim == t.ndim - 1:
        t_mask = t_mask.unsqueeze(-1)
    t_mask = t_mask.to(t.dtype)
    num = (t * t_mask).sum()
    denom = t_mask.sum() * t.shape[2]
    out = torch.where(denom > 0, num / denom, torch.zeros_like(num))
    return out

def tensor_mean_per_seq(t, t_mask):
    mask = t_mask
    if mask.ndim == t.ndim - 1:
        mask = mask.unsqueeze(-1)          # [B, T, 1]
    mask = mask.to(t.dtype)
    num = (t * mask).sum(dim=(1, 2))     # [B]，有效位置的总和
    steps = mask.sum(dim=1).squeeze(-1)    # [B]，每序列有效时间步数
    denom = steps * t.shape[2]             # 有效步数 × C
    out = torch.where(denom > 0, num / denom, torch.zeros_like(num))
    return out

class TTSPostTrainBaseTask(DiTBuildModelMixinV6, ScriptSpeechBaseTask):
    def __init__(self):
        super().__init__()

        if hparams.get('dataloader_version', 'v1') == 'v1':
            self.dataset_cls = import_module_bystr(hparams['dataset_cls'])
            self.val_dataset_cls = import_module_bystr(hparams['val_dataset_cls'])
            self.processer_fn = import_module_bystr(hparams['processer_fn']) if hparams.get('processer_fn') else None
            self.build_fast_dataloader = import_module_bystr(hparams['build_fast_dataloader']) if hparams.get('build_fast_dataloader') else None
        elif hparams.get('dataloader_version', 'v1') == 'v2':
            self.dataset_cls = import_module_bystr(hparams['dataset_cls'])
            self.train_dataloader = FastDatasetMixin.train_dataloader.__get__(self)
            self.test_dataloader = FastDatasetMixin.test_dataloader.__get__(self)
            self.val_dataloader = FastDatasetMixin.val_dataloader.__get__(self)
        self.hparams = hparams
        self.config = AttrDict(hparams)

    def build_model(self):

        trainable_modules = []
        other_modules = []

        self._build_model()
        trainable_modules.append(self.dit)

        self.vae.to(self.trainer.device)
        
        if hparams.get('freeze_ling_encoder', False) is True:
            frozen_modules = freeze_by_module_name(
                unwrap_model(self.dit),
                freeze_modules=[
                    'text_embedder', 'text_encoder', 'ph_encoder', 'tone_embed', 'ph_embed', 'ling_pre_net', 'f5_time_embed', 
                ]
            )
            cr = '\n'
            print_once(f"| Freeze following modules:{cr}{cr.join([f'| - {frozen_module}' for frozen_module in frozen_modules])}")

        self.dit_old = deepcopy(self.dit)
        self.dit_old.eval()
        for param in self.dit_old.parameters():
            param.requires_grad = False
        self.dit_old.to(self.trainer.device)
        other_modules.append(self.dit_old)

        self.dit_ref = deepcopy(self.dit)
        self.dit_ref.eval()
        for param in self.dit_ref.parameters():
            param.requires_grad = False
        self.dit_ref.to(self.trainer.device)
        # trainable_modules.append(self.dit_ref)

        if hparams.get('use_ema', False):
            print_once(f'| Building EMA model with decay={self.config.ema_decay} ...')
            self.ema_model = deepcopy(self.dit)
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad = False
            self.ema_model.to(self.trainer.device)

            trainable_modules.append(self.ema_model)

        self.reward_model_pack = self.build_reward_model()
        self.resamplers = {}

        ling_dict = json.load(open('egs/tts/megatts3_dict.json'))
        self.ling_dict = {k: TokenTextEncoder(None, vocab_list=ling_dict[k], replace_oov='<UNK>') for k in ['phone', 'tone']}

        # load pretrained ref models
        if hparams.get('load_ckpt', '') != '':
            load_ckpt(self.dit_ref, hparams['load_ckpt'], 'dit', strict=False, mmap=True)
        else:
            print(f"| Warning: Post Training without Pretrained Models!!!")
            
        return {'trainable': trainable_modules, 'others': other_modules}

    def build_reward_model(self):
        raise NotImplementedError("Reward model is not implemented for post training.")

    def run_reward_model(self, wavs, wav_lens, **kwargs):
        raise NotImplementedError("Reward model is not implemented for post training.")

    @torch.no_grad()
    def ema_update(self, ema_model, model, decay):
        ema_params = dict(unwrap_model(ema_model).named_parameters())
        for n, p in unwrap_model(model).named_parameters():
            p_ema = ema_params[n]
            src = p.detach()
            if src.dtype != torch.float32:
                src = src.float()
            if p_ema.dtype != torch.float32:
                p_ema.data = p_ema.data.float()
            p_ema.mul_(decay).add_(src.to(p_ema.device), alpha=1.0 - decay)

    def load_model(self):
        if hparams.get('load_ckpt', '') != '':
            load_ckpt(self.dit, hparams['load_ckpt'], 'dit', strict=False, mmap=True)
            load_ckpt(self.dit_old, hparams['load_ckpt'], 'dit', strict=False, mmap=True)
            # load_ckpt(self.dit_ref, hparams['load_ckpt'], 'dit', strict=False, mmap=True) # ref must be loaded everytime, so I move it to build_model
            if hparams.get('use_ema', False):
                load_ckpt(self.ema_model, hparams['load_ckpt'], 'dit', strict=False, mmap=True)
        else:
            print(f"| Warning: Post Training without Pretrained Models!!!")

    def build_optimizer(self):
        optimizer = AdamW(unwrap_model(self.dit).parameters(), **self.config.optimizer)
        return optimizer

    def build_scheduler(self, optimizer):
        return None

    def fsdp_optm2model(self):
        return [self.dit]

    def fsdp_wrap_policy(self):
        from torch.nn import Linear, Sequential, Conv1d, Conv2d, Embedding
        import modules.flow_matching.llama
        import modules.asr.llama.llama_seq2seq
        import modules.tts.llama_dit.llama_ca

        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            model_blocks = (
                # Linear, Sequential, Conv1d, Conv2d, Embedding,
                modules.flow_matching.llama.TransformerBlock,
                modules.asr.llama.llama_seq2seq.DecoderBlock,
                modules.asr.llama.llama_seq2seq.EncoderBlock,
                modules.tts.llama_dit.llama_ca.TransformerBlock,
                get_class_from_module("transformers.models.qwen3.modeling_qwen3", "Qwen3DecoderLayer")
            )
            return recurse or isinstance(module, model_blocks)

        return custom_auto_wrap_policy

    def prepare_cfg_input(self, inputs, device):
        inputs = deepcopy(inputs)

        if not hasattr(self, 'cfg_mask_token_phone'):
            self.cfg_mask_token_phone = 302 - 1
        if not hasattr(self, 'cfg_mask_token_tone'):
            self.cfg_mask_token_tone = 32 - 1

        inputs['phone'] = torch.cat([
            inputs['phone'], inputs['phone'],
            torch.full(inputs['phone'].size(), self.cfg_mask_token_phone, device=device)
        ], dim=0)
        inputs['tone'] = torch.cat([
            inputs['tone'], inputs['tone'],
            torch.full(inputs['tone'].size(), self.cfg_mask_token_tone, device=device)
        ], dim=0)
        inputs['ph_mask'] = torch.cat([inputs['ph_mask']] * 3, dim=0)
        inputs['ctx_ph_mask'] = torch.cat([inputs['ctx_ph_mask']] * 3, dim=0)
        
        inputs['txt_tokens'] = torch.cat([
            inputs['txt_tokens'], inputs['txt_tokens'],
            torch.full(inputs['txt_tokens'].size(), self.cfg_mask_text_token, device=device)
        ], dim=0)
        inputs['txt_mask'] = torch.cat([inputs['txt_mask']] * 3, dim=0)

        inputs['lat'] = torch.cat([
            inputs['lat'], torch.zeros_like(inputs['lat']), torch.zeros_like(inputs['lat'])
        ], dim=0)
        inputs['lat_ctx'] = torch.cat([
            inputs['lat_ctx'], torch.zeros_like(inputs['lat_ctx']), torch.zeros_like(inputs['lat_ctx'])
        ], dim=0)
        inputs['ctx_mask'] = torch.cat([inputs['ctx_mask']] * 3, dim=0)

        inputs['tgt_len'] = torch.cat([inputs['tgt_len']] * 3, dim=0)

        return inputs

    def training_step(self, sample, batch_idx, optimizer_idx):
        if self.trainer.proc_rank_local == 0 and random.random() < 0.0001:
            kill_void()

        hop_size = hparams['hop_size']
        vae_stride = hparams['vae_stride']
        num_batches_per_epoch = hparams['sample']['num_batches_per_epoch']
        num_train_timesteps = int(hparams['sample']['num_steps'] * hparams['train']['timestep_fraction'])

        self.posttrain_epoch = self.global_step // num_batches_per_epoch
        if not hasattr(self, 'samples_data_list'):
            self.samples_data_list = []

        # world_size = self.trainer.num_local_gpus
        # rank = self.trainer.proc_rank_local
        world_size = self.trainer.num_total_gpus
        rank = self.trainer.proc_rank

        ############
        # SAMPLING #
        ############

        # all_item_names = [None for _ in range(world_size)]
        # torch.distributed.all_gather_object(all_item_names, sample['item_name'])
        # print(f"{all_item_names = }")

        item_names = sample['item_name']
        wavs = sample["wavs"].float()
        wav_lengths = sample["wav_lengths"]
        ctx_wavs = sample["ctx_wavs"]
        ctx_mask = sample["ctx_mask"]
        ctx_mask = ctx_mask.float()
        if len(ctx_mask.shape) == 2:
            ctx_mask = ctx_mask[:, :, None]
        text = sample['text']
        ph_tokens = sample["ph_tokens"]
        tone_tokens = sample["tone"]
        ctx_ph_mask = sample['ctx_ph_mask']     # [B, T]
        en_tone_idx = ~((tone_tokens == 4) | ( (11 <= tone_tokens) & (tone_tokens <= 15)) | (tone_tokens == 0))
        tone_tokens[en_tone_idx] = 3
        ph_mask = ph_tokens > 0
        lat_lens = wav_lengths // hparams['hop_size'] // hparams['vae_stride']
        ctx_lens = ctx_mask.sum(1).squeeze(-1).long()
        ctx_wav_lens = ctx_lens * hop_size * vae_stride
        bsz = wavs.size(0)

        device = wavs.device

        # audio encode
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
            lat = self.vae.encode_latent(wavs)
            lat_ctx = self.vae.encode_latent(ctx_wavs)
            lat_ctx = torch.nn.functional.pad(lat_ctx, (0,0,0,lat.size(1)-lat_ctx.size(1)), mode='constant', value=0)
            
        # text tokenize
        text_inputs = self.dit_text_tokenizer(text, padding=True, return_tensors="pt").to(device)
        txt_tokens = text_inputs['input_ids']   # [B, T]
        txt_mask = text_inputs['attention_mask'].bool()
        txt_tokens[~txt_mask] = self.cfg_mask_text_token
        txt_lens = txt_mask.int().sum(1)
        loss_mask = sequence_mask(lat_lens, maxlen=lat.shape[1])[:, :, None] * (1-ctx_mask)

        inputs = {
            'item_names': item_names,
            'lat': lat,
            'lat_ctx': lat_ctx,
            'tgt_len': lat_lens,
            'ctx_mask': ctx_mask,
            'txt_tokens': txt_tokens,
            'txt_mask': txt_mask,
            'phone': ph_tokens,
            'tone': tone_tokens,
            'ph_mask': ph_mask,
            'ctx_ph_mask': ctx_ph_mask
        }

        infer_inputs = self.prepare_cfg_input(inputs, device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True), torch.no_grad():
            lat_pred, t_schedule = self.dit_old.inference(
                infer_inputs,
                timesteps=hparams['sample']['num_steps'],
                seq_cfg_w=hparams['sample']['guidance_scale'],
                timestep_annealing_w=(0.6, 0.6, 1.0),
                return_timesteps=True,
            )   # [B, T, C]
        
            lat_pred = lat_pred * (1 - ctx_mask) + lat_ctx * ctx_mask
            try:
                wav_pred = self.vae.decode(lat_pred)[:, 0]    # [B, 1, T] -> [B, T]
            except:
                if self.trainer.proc_rank_local == 0:
                    traceback.print_exc()
                    print(f"| ERROR: VAE decoding fail: lat_pred.shape={lat_pred.shape}, skip")
                    print(f"{item_names = }")
                return {'loss': None}

            wav_pred_lens = wav_lengths - ctx_wav_lens
            wav_pred = remove_prefix(wav_pred[..., None], ctx_wav_lens, wav_pred_lens)[..., 0].float()

            with torch.cuda.amp.autocast(enabled=False):
                rewards = self.run_reward_model(wav_pred, wav_pred_lens, inputs=inputs, sample=sample)

            if DEBUG:
                if (
                    self.trainer.proc_rank == 0 and 
                    self.global_step % num_batches_per_epoch == 0 and 
                    (self.posttrain_epoch < 10 or self.posttrain_epoch % 20 == 0)
                ):
                    save_dir = f'{hparams["work_dir"]}/sample_batches/step_{self.global_step}'
                    os.makedirs(save_dir, exist_ok=True)
                    for b_i in range(bsz):
                        wav_pred_ = wav_pred[b_i, :wav_pred_lens[b_i]].cpu().float().numpy()
                        sf.write(f"{save_dir}/wav_pred_{b_i}.wav", wav_pred_, 24000, 'PCM_16')
                        ph_lens = ph_mask.sum(1)
                        phone = ph_tokens[b_i, :ph_lens[b_i]].cpu().long().numpy()
                        phone = self.ling_dict['phone'].decode(phone).split(' ')
                        tone = tone_tokens[b_i, :ph_lens[b_i]].cpu().long().numpy()
                        tone = self.ling_dict['tone'].decode(tone).split(' ')
                        
                        json_dump({
                            'text': text[b_i],
                            'phone': phone,
                            'tone': tone,
                            # 'reward': rewards[b_i].item(),
                            **{f'reward_{k}': rewards[k][b_i].item() for k in rewards}
                        }, f"{save_dir}/phone_tone_{b_i}.json")
                    json_dump({
                        f"{b_i}": {f'reward_{k}': rewards[k][b_i].item() for k in rewards} for b_i in range(bsz)
                    }, f"{save_dir}/rewards.json")

            inputs.update({
                'wavs': wavs,
                'wav_lengths': wav_lengths,
                'timesteps': t_schedule,
                'next_timesteps': torch.cat([t_schedule[1:], torch.zeros_like(t_schedule[:1])], dim=0),
                'lat_pred': lat_pred,
                'rewards': rewards
            })

        inputs = move_to_cpu(inputs)
        # for b_i in range(bsz * num_generation_per_prompt):
        for b_i in range(bsz):
            self.samples_data_list.append({
                'item_names': inputs['item_names'][b_i],
                'lat': inputs['lat'][b_i],
                'lat_ctx': inputs['lat_ctx'][b_i],
                'tgt_len': inputs['tgt_len'][b_i],
                'ctx_mask': inputs['ctx_mask'][b_i],
                'txt_tokens': inputs['txt_tokens'][b_i],
                'txt_mask': inputs['txt_mask'][b_i],
                'phone': inputs['phone'][b_i],
                'tone': inputs['tone'][b_i],
                'ph_mask': inputs['ph_mask'][b_i],
                'ctx_ph_mask': inputs['ctx_ph_mask'][b_i],
                'wavs': inputs['wavs'][b_i],
                'wav_lengths': inputs['wav_lengths'][b_i],
                'timesteps': inputs['timesteps'],
                'next_timesteps': inputs['next_timesteps'],
                'lat_pred': inputs['lat_pred'][b_i],
                'rewards': {k: inputs['rewards'][k][b_i] for k in rewards}
            })
        # self.samples_data_list.append(inputs)

        ############
        # TRAINING #
        ############

        if (self.global_step + 1) % num_batches_per_epoch == 0:

            collated_samples = {}
            for k in self.samples_data_list[0].keys():
                if isinstance(self.samples_data_list[0][k], torch.Tensor):
                    if self.samples_data_list[0][k].ndim == 0:
                        collated_samples[k] = torch.stack([s[k] for s in self.samples_data_list])
                    else:
                        collated_samples[k] = collate_xd([s[k] for s in self.samples_data_list])
                elif k == 'rewards':
                    collated_samples[k] = {rk: torch.stack([s[k][rk] for s in self.samples_data_list]) for rk in self.samples_data_list[0][k]}
                else:
                    collated_samples[k] = [s[k] for s in self.samples_data_list]

            collated_samples = move_to_cuda(collated_samples, device=device)
            self.samples_data_list.clear()
            rewards = collated_samples['rewards']['sum']
            item_names = collated_samples['item_names']

            if hparams['per_prompt_stat_tracking']:
                if world_size > 1:
                    all_rewards = all_gather_varlen_tensor(rewards, dim=0)  # [B]
                    all_item_names = [None for _ in range(world_size)]
                    torch.distributed.all_gather_object(all_item_names, item_names)
                    all_item_names = [item_name for item_names in all_item_names for item_name in item_names]
                    assert len(all_rewards) == len(all_item_names)
                else:
                    all_rewards = rewards
                    all_item_names = item_names
                item_names_unique = list(set(all_item_names))
                item_name2idxs = {item_name: [] for item_name in item_names_unique}
                for item_name_idx, item_name in enumerate(all_item_names):
                    item_name2idxs[item_name].append(item_name_idx)
                item_name2avg = {}
                item_name2std = {}
                for item_name in item_names_unique:
                    item_name_idxs = item_name2idxs[item_name]
                    item_name2avg[item_name] = all_rewards[item_name_idxs].mean()
                    item_name2std[item_name] = all_rewards[item_name_idxs].std()
                rewards_avg, rewards_std = [], []
                for item_name in item_names:
                    rewards_avg.append(item_name2avg[item_name])
                    rewards_std.append(item_name2std[item_name])
                rewards_avg = torch.stack(rewards_avg)
                rewards_std = torch.stack(rewards_std)
            else:
                if world_size > 1:
                    all_rewards = all_gather_varlen_tensor(rewards, dim=0)
                else:
                    all_rewards = rewards
                rewards_avg = all_rewards.mean()
                rewards_std = all_rewards.std()
            advantages = (rewards - rewards_avg) / (rewards_std + 1e-4)
            collated_samples['advantages'] = advantages

            filtered_samples = collated_samples
            del filtered_samples['item_names']
            total_batch_size_filtered, num_timesteps_filtered = filtered_samples["timesteps"].shape

            num_batches = math.ceil(
                num_batches_per_epoch * hparams['sample']['train_batch_size'] * 
                hparams['sample']['num_generation_per_prompt'] / hparams['train']['batch_size']
            )
            
            effective_grad_accum_steps = hparams['train']['gradient_accumulation_steps'] * num_train_timesteps
            
            current_accumulated_steps = 0  # Counter for backward passes
            # gradient_update_times = 0

            for inner_epoch in range(hparams['train']['num_inner_epochs']):
                perm = torch.randperm(total_batch_size_filtered, device=device)
                # print(f'{filtered_samples["timesteps"].shape = }')
                # print(f"{total_batch_size_filtered = }")
                # print(f"{perm.shape = }")
                # print(f"{filtered_samples = }")
                # shuffled_filtered_samples = {k: v[perm] for k, v in filtered_samples.items()}
                shuffled_filtered_samples = {}
                for k, v in filtered_samples.items():
                    if k == 'rewards':
                        shuffled_filtered_samples[k] = {rk: v[rk][perm] for rk in v}
                    else:
                        shuffled_filtered_samples[k] = v[perm]

                perms_time = torch.stack(
                    [torch.randperm(num_timesteps_filtered, device=device) for _ in range(total_batch_size_filtered)]
                )
                for key in ["timesteps", "next_timesteps"]:
                    shuffled_filtered_samples[key] = shuffled_filtered_samples[key][
                        torch.arange(total_batch_size_filtered, device=device)[:, None], perms_time
                    ]

                training_batch_size = total_batch_size_filtered // num_batches
                samples_batched_list = []
                for k_batch in range(num_batches):
                    batch_dict = {}
                    start = k_batch * training_batch_size
                    end = (k_batch + 1) * training_batch_size
                    for key, val_tensor in shuffled_filtered_samples.items():
                        if key == 'rewards':
                            batch_dict[key] = {rk: val_tensor[rk][start:end] for rk in val_tensor}
                        else:
                            batch_dict[key] = val_tensor[start:end]
                    samples_batched_list.append(batch_dict)

                if self.trainer.proc_rank_local == 0:
                    inner_training_pbar = tqdm(
                        list(enumerate(samples_batched_list)),
                        desc=f"| PostTrain Epoch {self.posttrain_epoch}/{inner_epoch}: training",
                        position=0,
                    )
                else:
                    inner_training_pbar = list(enumerate(samples_batched_list))
                for i, train_sample_batch in inner_training_pbar:
                    # current_micro_batch_size = len(train_sample_batch["txt_tokens"])

                    print(
                        f"| rewards={train_sample_batch['rewards']['sum'].mean().detach():.4f}, "
                        f"rewards_std={train_sample_batch['rewards']['sum'].std().detach():.4f}, "
                        f"advantages={train_sample_batch['advantages'].mean().detach():.4f}"
                    )
                    print(f"| each reward: ", end='')
                    for rk in train_sample_batch['rewards']:
                        if rk != 'sum':
                            print(f"{rk}={train_sample_batch['rewards'][rk].mean().detach():.4f}, ", end='')
                    print()

                    timestep_progress_bar_log = {}

                    if self.trainer.proc_rank_local == 0:
                        inner_timestep_training_pbar = tqdm(
                            enumerate(range(num_train_timesteps)),
                            desc="| Training Timestep",
                            position=1,
                            leave=False,
                        )
                    else:
                        inner_timestep_training_pbar = enumerate(range(num_train_timesteps))
                    for j_idx, j_timestep_orig_idx in inner_timestep_training_pbar:
                        
                        assert j_idx == j_timestep_orig_idx

                        x1 = train_sample_batch['lat_pred']
                        t = train_sample_batch["timesteps"][:, j_idx]   # [B]
                        t_expanded = t[:, None, None]
                        x0 = torch.randn_like(x1)
                        xt = t_expanded * x1 + (1 - t_expanded) * x0

                        lat_lens = train_sample_batch['tgt_len']
                        ctx_mask = train_sample_batch['ctx_mask']
                        loss_mask = sequence_mask(lat_lens, maxlen=xt.shape[1])[:, :, None].to(xt) * (1-ctx_mask)
                        # loss_mask_sum = loss_mask.sum()
                        # loss_mask_scale = loss_mask.shape[0] * loss_mask.shape[1] / loss_mask_sum
                        loss_mask_batch_scale = loss_mask.shape[1] / loss_mask[..., 0].sum(dim=1)   # [B]

                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):

                            infer_inputs = self.prepare_cfg_input(train_sample_batch, device)
                            xt_input = torch.cat([xt] * 3, dim=0)
                            t_3 = t.unsqueeze(1).repeat(1,3).reshape(-1)
                            
                            with torch.no_grad():
                                old_prediction = self.model_forward_step(unwrap_model(self.dit_old), xt_input, t_3, infer_inputs)
                                ref_forward_prediction = self.model_forward_step(unwrap_model(self.dit_ref), xt_input, t_3, infer_inputs)

                            forward_prediction = self.model_forward_step(unwrap_model(self.dit), xt_input, t_3, infer_inputs)

                        loss_terms = {}

                        # Policy Gradient Loss
                        adv_clip_max = hparams['train']['adv_clip_max']
                        nft_beta = hparams['nft_beta']
                        advantages_clip = torch.clamp(
                            train_sample_batch["advantages"], -adv_clip_max, adv_clip_max,
                        )
                        if hasattr(hparams['train'], "adv_mode"):
                            if hparams['train']["adv_mode"] == "positive_only":
                                advantages_clip = torch.clamp(advantages_clip, 0, adv_clip_max)
                            elif hparams['train']["adv_mode"] == "negative_only":
                                advantages_clip = torch.clamp(advantages_clip, -adv_clip_max, 0)
                            elif hparams['train']["adv_mode"] == "one_only":
                                advantages_clip = torch.where(
                                    advantages_clip > 0, torch.ones_like(advantages_clip), torch.zeros_like(advantages_clip)
                                )
                            elif hparams['train']["adv_mode"] == "binary":
                                advantages_clip = torch.sign(advantages_clip)
                        loss_terms["monitor/advantage"] = train_sample_batch["advantages"].mean().detach()
                        loss_terms["monitor/rewards"] = train_sample_batch["rewards"]['sum'].mean().detach()
                        loss_terms["monitor/rewards_std"] = train_sample_batch["rewards"]['sum'].std().detach()
                        for rk in train_sample_batch['rewards']:
                            if rk != 'sum':
                                loss_terms[f"monitor/rewards_{rk}"] = train_sample_batch["rewards"][rk].mean().detach()
                                loss_terms[f"monitor/rewards_{rk}_std"] = train_sample_batch["rewards"][rk].std().detach()

                        # normalize advantage
                        normalized_advantages_clip = (advantages_clip / adv_clip_max) / 2.0 + 0.5
                        r = torch.clamp(normalized_advantages_clip, 0, 1)
                        positive_prediction = nft_beta * forward_prediction + (1 - nft_beta) * old_prediction.detach()
                        implicit_negative_prediction = (1.0 + nft_beta) * old_prediction.detach() - nft_beta * forward_prediction
                        
                        loss_terms["monitor/x1_norm"] = tensor_mean_per_element(x1**2, loss_mask).detach()
                        loss_terms["monitor/x1_norm_max"] = ( (x1**2) * loss_mask ).max().detach()
                        loss_terms["monitor/old_deviate"] = tensor_mean_per_element((forward_prediction - old_prediction) ** 2, loss_mask).detach()
                        loss_terms["monitor/old_deviate_max"] = torch.max((forward_prediction - old_prediction) ** 2 * loss_mask).detach()
                        loss_terms["monitor/advantages_clip"] = normalized_advantages_clip.mean().detach()

                        # adaptive weighting
                        x1_prediction = xt + (1 - t_expanded) * positive_prediction
                        with torch.no_grad():
                            weight_factor = (
                                torch.abs((x1_prediction.float() - x1.float()) * loss_mask)
                                .mean(dim=tuple(range(1, x1.ndim)), keepdim=True)
                                .clip(min=0.00001) * loss_mask_batch_scale[:, None, None]
                            )
                        positive_loss = tensor_mean_per_seq((x1_prediction - x1) ** 2 / weight_factor, loss_mask)
                        loss_terms["pos"] = positive_loss.mean().detach()

                        negative_x1_prediction = xt + (1 - t_expanded) * implicit_negative_prediction
                        with torch.no_grad():
                            negative_weight_factor = (
                                torch.abs((negative_x1_prediction.float() - x1.float()) * loss_mask)
                                .mean(dim=tuple(range(1, x1.ndim)), keepdim=True)
                                .clip(min=0.00001) * loss_mask_batch_scale[:, None, None]
                            )
                        negative_loss = tensor_mean_per_seq((negative_x1_prediction - x1) ** 2 / negative_weight_factor, loss_mask)
                        loss_terms['neg'] = negative_loss.mean().detach()

                        ori_policy_loss = r * positive_loss / nft_beta + (1.0 - r) * negative_loss / nft_beta
                        policy_loss = ori_policy_loss * adv_clip_max
                        policy_loss = policy_loss.mean()

                        loss = policy_loss
                        loss_terms["policy"] = policy_loss.detach()

                        kl_div_loss = tensor_mean_per_element((forward_prediction - ref_forward_prediction) ** 2, loss_mask)
                        loss += hparams['train']['beta'] * kl_div_loss
                        loss_terms["kl"] = kl_div_loss.detach()
                        loss_terms["kl_old"] = tensor_mean_per_element((old_prediction - ref_forward_prediction) ** 2, loss_mask)

                        # loss_terms["total_loss"] = loss.detach()

                        scaled_loss = loss / effective_grad_accum_steps

                        if self.trainer.amp and self.trainer.amp_scalar is not None:
                            self.trainer.amp_scalar.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()
                        current_accumulated_steps += 1

                        has_nan_grad = False
                        nan_params_names = []
                        for name, param in self.named_parameters():
                            if (param.grad is not None) and torch.isnan(param.grad.float()).any():
                                has_nan_grad = True
                                nan_params_names.append(name)
                        if has_nan_grad:
                            print(f"| WARN: found nan in grad! first nan params: {nan_params_names[0]}; last nan params: {nan_params_names[-1]}.")
                            self.trainer.optimizers[0].zero_grad()

                        if current_accumulated_steps % effective_grad_accum_steps == 0 and not has_nan_grad:
                            #################
                            # OPTIMIER STEP #
                            #################
                            if self.trainer.amp and self.trainer.amp_scalar is not None:
                                self.trainer.amp_scalar.unscale_(self.trainer.optimizers[0])
                            
                            grad_norm = self.compute_grad_norm(self.trainer.optimizers[0], distributed=True, norm_type=2.0)
                            loss_terms['monitor/grad_norm_optm0'] = grad_norm

                            if self.gradient_clip_norm > 0 or self.gradient_clip_val > 0:
                                for n in self.trainer.training_module_names:
                                    m = getattr(self, n)
                                    if self.gradient_clip_norm > 0:
                                        if isinstance(m, FullyShardedDataParallel):
                                            grad_norm = m.clip_grad_norm_(self.gradient_clip_norm)
                                        else:
                                            torch.nn.utils.clip_grad_norm_(m.parameters(), self.gradient_clip_norm)
                                    if self.gradient_clip_val > 0:
                                        assert not isinstance(m, FullyShardedDataParallel)
                                        torch.nn.utils.clip_grad_value_(m.parameters(), self.gradient_clip_val)

                            if self.trainer.amp and self.trainer.amp_scalar is not None:
                                self.trainer.amp_scalar.step(self.trainer.optimizers[0])
                                self.trainer.amp_scalar.update()
                            else:
                                self.trainer.optimizers[0].step()
                            self.trainer.optimizers[0].zero_grad()

                            if hparams.get('use_ema', False):
                                self.ema_update(self.ema_model, self.dit, self.config.ema_decay)

                        global_step = (
                            (self.posttrain_epoch * hparams['train']['num_inner_epochs'] + inner_epoch) * num_batches * num_train_timesteps + 
                            i * num_train_timesteps + j_idx 
                        )
                        loss_terms = tensors_to_scalars(loss_terms)
                        for params_group_i in range(len(self.trainer.optimizers[0].param_groups)):
                            loss_terms[f'lr/optimizer0_params_group{params_group_i}'] = self.trainer.optimizers[0].param_groups[params_group_i]['lr']
                        # add to progress bar
                        progress_bar_log = {}
                        for k, v in loss_terms.items():
                            if 'monitor/' in k:
                                continue
                            if '/' in k:
                                k_split = k.split("/")
                                assert len(k_split) == 2, "we only support one `/` in tag_name, i.e., `<tag>/<sub_tag>`"
                                k = k.replace("/", "_")
                            k = k.replace('optimizer', 'optm').replace('params_group', 'pg')
                            assert k not in progress_bar_log, f"we got duplicate tags in log_outputs, check this `{k}`"
                            progress_bar_log[k] = v
                        # progress_bar_log['gn'] = grad_norm
                        for k, v in progress_bar_log.items():
                            timestep_progress_bar_log[k] = timestep_progress_bar_log.get(k, 0.0) + v / num_train_timesteps
                        # add to tensorboard
                        tb_log = {}
                        for k, v in loss_terms.items():
                            if '/' in k:
                                tb_log[k] = v
                            else:
                                tb_log[f'tr/{k}'] = v
                        if self.trainer.proc_rank == 0:
                            for k, v in tb_log.items():
                                if isinstance(v, torch.Tensor):
                                    v = v.item()
                                self.trainer.logger.add_scalar(k, v, global_step)
                            if j_idx % hparams['train']['pbar_log_interval'] == 0:
                                progress_bar_log = ', '.join([f"{k}={v:.5f}" for k, v in progress_bar_log.items()])
                                print(f"| training_step:{self.posttrain_epoch}/{inner_epoch}/{i}/{j_idx} | global step:{global_step} | {progress_bar_log}")
                            # inner_timestep_training_pbar.
                    
                    timestep_progress_bar_log['gn'] = loss_terms['monitor/grad_norm_optm0']
                    timestep_progress_bar_log = ', '.join([f"{k}={v:.5f}" for k, v in timestep_progress_bar_log.items()])
                    print(f"\n| Training Timestep End | {timestep_progress_bar_log}")

            if world_size > 1:
                torch.distributed.barrier()

            with torch.no_grad():
                global_step = (self.posttrain_epoch + 1) * hparams['train']['num_inner_epochs'] * num_batches * num_train_timesteps
                decay = return_decay(global_step // effective_grad_accum_steps, hparams['decay_type'])
                print(f"| PostTrain Epoch:{self.posttrain_epoch} | global step:{global_step} | update_steps:{global_step // effective_grad_accum_steps} | inner_epochs end, updating old model with decay {decay}")
                online = dict(unwrap_model(self.dit).named_parameters())
                target = dict(unwrap_model(self.dit_old).named_parameters())
                for n, p_online in online.items():
                    if n in target:
                        p_tgt = target[n]
                        if p_tgt.dtype != p_online.dtype:
                            p_src = p_online.detach().to(p_tgt.dtype)
                        else:
                            p_src = p_online.detach()
                        p_tgt.data.copy_(p_tgt.data * decay + p_src * (1.0 - decay))

            print()

        return {'loss': None}   # poison pill to skip trainer

    def model_forward_step(self, dit, xt, t, inputs, do_checkpoint=False):
        tgt_len = inputs['tgt_len']     # reference + target
        x_mask = sequence_mask(tgt_len, maxlen=xt.shape[1])
        x_txt = dit.forward_ling_encoder(inputs, x_mask)
        (bsz, tgt_len, _), device = x_txt.shape, x_txt.device
        bsz = bsz // 3

        ctx_mask = inputs['ctx_mask']
        ctx_feature = inputs['lat_ctx'] * ctx_mask

        cond = {
            'ctx': ctx_feature,
            'ctx_mask': ctx_mask,
            'attn_mask': x_mask,
            'x_txt': x_txt,
        }

        pred = dit._forward(
            xt, cond, t, 
            seq_cfg_w=hparams['sample']['guidance_scale'], 
            timestep_annealing_w=(0.6, 0.6, 1.0)
        )

        return pred


class MegaTTSDiTPostTrainPhoneTask(TTSPostTrainBaseTask):
    def build_reward_model(self):
        from modules.asr.mfa.nar_mfa_v6 import build_nar_mfa_model
        from inference.asr.nar_mfa_infer import MFAInfer
        mfa_ckpt = 'checkpoints/251104_nar_mfa_v6_long_base_robust/model_ckpt_steps_100000.ckpt'
        model = MFAInfer(self.trainer.device, mfa_ckpt, torch_compile=False, precision=torch.bfloat16)
        return {'mfa': model}

    def run_reward_mfa_model(self, wavs, wav_lens, **kwargs):
        bsz = wavs.shape[0]

        mfa_model = self.reward_model_pack['mfa']
        
        wavs_np = []
        for i in range(bsz):
            wavs_np.append(wavs[i, :wav_lens[i]].cpu().numpy())
        
        mfa_outputs = mfa_model.forward_batch(wavs_np, sample_rates=24000, timesteps=12, max_batch_duration=200, max_batch_size=10, use_tqdm=False)

        inputs = kwargs['inputs']
        phone = inputs['phone'].cpu().numpy()
        tone = inputs['tone'].cpu().numpy()
        ph_mask = inputs['ph_mask'].cpu().numpy()
        ph_lens = ph_mask.sum(1)
        ctx_ph_mask = inputs['ctx_ph_mask'].cpu().numpy()
        ctx_ph_lens = ctx_ph_mask.sum(1)

        rewards = []

        def score_fn(x, y):
            if x[0] == y[0]:
                if x[1] == y[1]:
                    return 1.0
                else:
                    return 0.5
            else:
                return 0.0

        for i in range(bsz):
            if mfa_outputs[i] is None:
                rewards.append(0.0)
                continue

            ph_gt = phone[i, ctx_ph_lens[i]: ph_lens[i]]
            tone_gt = tone[i, ctx_ph_lens[i]: ph_lens[i]]
            ph_pred = mfa_model.ling_dict['phone'].encode(' '.join(mfa_outputs[i]['ph']))
            tone_pred = mfa_model.ling_dict['tone'].encode(' '.join(mfa_outputs[i]['tone']))

            ph_gt = [(p, t) for p, t in zip(ph_gt, tone_gt) if p != 145]
            ph_pred = [(p, t) for p, t in zip(ph_pred, tone_pred) if p != 145]

            try:
                reward, _ = seq_match(ph_gt, ph_pred, score_fn=score_fn, metric='mean')
            except IndexError:
                reward = 0.0

            rewards.append(reward)

            if DEBUG:
                if (
                    self.trainer.proc_rank == 0 and 
                    self.global_step % hparams['sample']['num_batches_per_epoch'] == 0 and 
                    (self.posttrain_epoch < 10 or self.posttrain_epoch % 20 == 0)
                ):
                    save_dir = f'{hparams["work_dir"]}/sample_batches/step_{self.global_step}'
                    os.makedirs(save_dir, exist_ok=True)
                    json_dump({
                        'ph_gt': mfa_model.ling_dict['phone'].decode([p[0] for p in ph_gt]).split(' '),
                        'tone_gt': mfa_model.ling_dict['tone'].decode([p[1] for p in ph_gt]).split(' '),
                        'ph_pred': mfa_model.ling_dict['phone'].decode([p[0] for p in ph_pred]).split(' '),
                        'tone_pred': mfa_model.ling_dict['tone'].decode([p[1] for p in ph_pred]).split(' '),
                        'reward': reward
                    }, f"{save_dir}/phone_tone_pred_{i}.json")

        return torch.Tensor(rewards).to(wavs.device)

    def run_reward_model(self, wavs, wav_lens, **kwargs):
        bsz = wavs.shape[0]

        rewards = {}
        rewards['phone'] = self.run_reward_mfa_model(wavs, wav_lens, **kwargs)

        rewards_agg = torch.zeros(bsz, device=wavs.device)
        for k in rewards:
            rewards_agg += rewards[k]
        rewards['sum'] = rewards_agg
        
        return rewards


class MegaTTSDiTPostTrainMultiRewardTask(MegaTTSDiTPostTrainPhoneTask):
    def build_reward_model(self):

        reward_model_pack = {}

        if hparams['reward']['phone']:
            from inference.asr.nar_mfa_infer import MFAInfer
            mfa_ckpt = 'checkpoints/251104_nar_mfa_v6_long_base_robust/model_ckpt_steps_100000.ckpt'
            model = MFAInfer(self.trainer.device, mfa_ckpt, torch_compile=False, precision=torch.bfloat16)
            reward_model_pack['mfa'] = model

        if hparams['reward']['sim']:
            from modules.asr.speaker_verification.ecapa_tdnn import ECAPA_TDNN_SMALL
            model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=None)
            state_dict = torch.load('checkpoints/wavlm/wavlm_large_finetune.pth')
            load_results = model.load_state_dict(state_dict['model'], strict=False)
            model.eval()
            model.to(self.trainer.device)
            reward_model_pack['sim'] = model

        if hparams['reward']['mos']:
            from torchaudio.pipelines import SQUIM_SUBJECTIVE
            subjective_model = SQUIM_SUBJECTIVE.get_model().to(self.trainer.device)
            subjective_model.eval()
            for param in subjective_model.parameters():
                param.requires_grad = False
                param.grad = None
            wav_nmr, sr_nmr = torchaudio.load(torchaudio.utils.download_asset("tutorial-assets/ctc-decoding/1688-142285-0007.wav"))
            wav_nmr = wav_nmr.to(self.trainer.device)
            if sr_nmr != 16000:
                wav_nmr = torchaudio.functional.resample(wav_nmr, sr_nmr, 16000)
            reward_model_pack['mos'] = {'model': subjective_model, 'wav_nmr': wav_nmr}
        
        if hparams['reward']['stoi'] or hparams['reward']['pesq']:
            from torchaudio.pipelines import SQUIM_OBJECTIVE
            objective_model = SQUIM_OBJECTIVE.get_model().to(self.trainer.device)
            objective_model.eval()
            for param in objective_model.parameters():
                param.requires_grad = False
                param.grad = None
            reward_model_pack['stoi_pesq'] = objective_model
            
        return reward_model_pack

    def run_reward_sim_model(self, wavs, wav_lens, **kwargs):
        sample = kwargs['sample']

        ctx_wavs = sample["ctx_wavs"]
        ctx_mask = sample["ctx_mask"]
        if len(ctx_mask.shape) == 3:
            ctx_mask = ctx_mask[:, :, 0]
        ctx_lens = ctx_mask.sum(dim=1) * hparams['hop_size'] * hparams['vae_stride']

        wavs = repeat_pad_wavs(wavs, wav_lens)
        ctx_wavs = repeat_pad_wavs(ctx_wavs, ctx_lens)

        if hparams['audio_sample_rate'] != 16000 and hparams['audio_sample_rate'] not in self.resamplers:
            self.resamplers[hparams['audio_sample_rate']] = torchaudio.transforms.Resample(orig_freq=hparams['audio_sample_rate'], new_freq=16000).to(self.trainer.device)

        wavs = self.resamplers[hparams['audio_sample_rate']](wavs)
        ctx_wavs = self.resamplers[hparams['audio_sample_rate']](ctx_wavs)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            embeds_pred = self.reward_model_pack['sim'](wavs)   # [B, 256]
            embeds_ctx = self.reward_model_pack['sim'](ctx_wavs)

        sim = F.cosine_similarity(embeds_pred, embeds_ctx)  # [B]

        return sim

    def run_reward_mos_model(self, wavs, wav_lens, **kwargs):     
        bsz = wavs.shape[0]  
        wavs = repeat_pad_wavs(wavs, wav_lens)

        if hparams['audio_sample_rate'] != 16000 and hparams['audio_sample_rate'] not in self.resamplers:
            self.resamplers[hparams['audio_sample_rate']] = torchaudio.transforms.Resample(orig_freq=hparams['audio_sample_rate'], new_freq=16000).to(self.trainer.device)

        wavs = self.resamplers[hparams['audio_sample_rate']](wavs)
        wav_nmr = torch.cat([self.reward_model_pack['mos']['wav_nmr']] * bsz, dim=0)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            mos = self.reward_model_pack['mos']['model'](wavs, wav_nmr)

        return mos / 5

    def run_reward_stoi_pesq_model(self, wavs, wav_lens, **kwargs):
        wavs = repeat_pad_wavs(wavs, wav_lens)

        if hparams['audio_sample_rate'] != 16000 and hparams['audio_sample_rate'] not in self.resamplers:
            self.resamplers[hparams['audio_sample_rate']] = torchaudio.transforms.Resample(orig_freq=hparams['audio_sample_rate'], new_freq=16000).to(self.trainer.device)

        wavs = self.resamplers[hparams['audio_sample_rate']](wavs)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            stoi, pesq, si_sdr = self.reward_model_pack['stoi_pesq'](wavs)

        return stoi, pesq / 5

    def run_reward_model(self, wavs, wav_lens, **kwargs):
        bsz = wavs.shape[0]

        rewards = {}

        if hparams['reward']['phone']:
            rewards['phone'] = self.run_reward_mfa_model(wavs, wav_lens, **kwargs)

        if hparams['reward']['sim']:
            rewards['sim'] = self.run_reward_sim_model(wavs, wav_lens, **kwargs)
        
        if hparams['reward']['mos']:
            rewards['mos'] = self.run_reward_mos_model(wavs, wav_lens, **kwargs)
        
        if hparams['reward']['stoi'] or hparams['reward']['pesq']:
            stoi, pesq = self.run_reward_stoi_pesq_model(wavs, wav_lens, **kwargs)
            if hparams['reward']['stoi']:
                rewards['stoi'] = stoi
            if hparams['reward']['pesq']:
                rewards['pesq'] = pesq
        
        rewards_agg = torch.zeros(bsz, device=wavs.device)
        for k in rewards:
            rewards_agg += rewards[k]
        rewards['sum'] = rewards_agg
        
        return rewards
    

def repeat_pad_wavs(wavs: torch.Tensor, wav_lens: torch.Tensor) -> torch.Tensor:
    """
    wavs: [B, T], 任意 dtype（float）和 device
    wav_lens: [B], int/long，表示每条音频有效长度；要求 >0 且 <= T
    返回：将右侧 padding 替换为循环重复后的 [B, T]
    """
    assert wavs.dim() == 2, "wavs 应为 [B, T]"
    B, T = wavs.shape
    device = wavs.device
    lengths = wav_lens.to(device)
    if torch.any(lengths <= 0):
        raise ValueError("wav_lens 必须 > 0")
    # 构造每行的索引：0..T-1 对每个长度取模
    idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # [B, T]
    idx = (idx % lengths.unsqueeze(1)).long()  # [B, T], 每行 < L_i
    # 按行 gather；因为 idx < L_i，只会从有效区域取值，不会取到原始 padding
    out = torch.gather(wavs, dim=1, index=idx)
    return out