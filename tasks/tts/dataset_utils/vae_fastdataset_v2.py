import collections
import collections.abc
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))

import os
import random
import json
from copy import deepcopy
import pickle
import re
import traceback
import math
import time
import tempfile
import uuid

import setproctitle
import torch
import torchaudio
import numpy as np
import torch.utils
import torch.utils.data
import librosa
from dataloader import FalconReader, KVReader

from utils.commons.import_utils import import_module_bystr
from utils.commons.hparams import hparams
from utils.commons.os_utils import multiprocess_glob, handle_exacption
from utils.commons.io import get_wav_duration, print_once
from utils.commons.base_shm_dataset import BaseFalconReaderShmDataset, get_from_global_stores, save_samples_to_shm
from utils.commons.dataset_utils import collate_xd, pad_or_cut_xd, SkipLogger
from utils.commons.tensor_utils import convert_to_tensor, convert_to_np
# from utils.commons.tos_utils_v2 import TosClient
# from utils.commons.hdfs_utils import HDFSClient
from utils.dataset.batcher import BucketBatcher
from utils.audio.vad import build_vad_model, run_vad_trim
from utils.audio.align import mel2token_to_dur
from utils.audio.align import mel2token_to_dur
from utils.text.split_text import get_word_list
from utils.text.ph_tone_convert import map_phone_to_tokendict
from utils.text.split_text import get_word_list, remove_spaces_between_chinese
from utils.text import is_chinese, is_english

from tasks.tts.dataset_utils.tts_datasets import MegaTTSDataset, FrontendLMDataset
from tasks.tts.dataset_utils.tts_fastdataset_v2 import BaseTTSShmDataset, safe_read_path
from modules.tts.ar_dur.commons.align_ops import compute_mel2aug_from_dur
from modules.tts.ar_dur.commons.nar_tts_modules import LengthRegulator

DEBUG = False

import math
import torch

def repeat_or_chunk_1d(
    x: torch.Tensor,
    tgt_size: int,
    drop_last: bool = True,
    fill_last: str = "repeat",  # 可选: "repeat" | "wrap" | "pad_zero"
) -> torch.Tensor:
    """
    将一维音频 tensor 按需求 repeat 或切块。

    参数:
      x: 形状 [T] 的一维 tensor
      tgt_size: 目标长度 t
      drop_last: 当 T > t 时，是否丢弃末尾不足 t 的残段
      fill_last: 当 drop_last=False 时用于填充末尾残段的策略:
                 - "repeat": 对残段自身重复直至长度 t，再截断
                 - "wrap":   从序列起点环绕填充至 t（类似循环缓冲）
                 - "pad_zero": 用 0 填充至 t

    返回:
      形状 [B, t] 的 tensor；若 T <= t 则 B=1
    """
    if x.dim() != 1:
        raise ValueError(f"expect 1D tensor [T], got shape {tuple(x.shape)}")
    if tgt_size <= 0:
        raise ValueError("tgt_size must be positive")

    T = x.numel()
    if T == 0:
        raise ValueError("input length T must be > 0")

    # 情况 1: T < t，循环 repeat 到 t，超出截断
    if T < tgt_size:
        nrep = math.ceil(tgt_size / T)
        y = x.repeat(nrep)[:tgt_size]         # [t]
        return y.unsqueeze(0)                 # [1, t]

    # 情况 2: T == t，直接返回 [1, t]
    if T == tgt_size:
        return x.unsqueeze(0)

    # 情况 3: T > t，切成不重叠 chunk
    n_full = T // tgt_size
    out = x[:n_full * tgt_size].view(n_full, tgt_size)  # [B_full, t]
    rem = T % tgt_size

    # 没有残段或选择丢弃残段
    if rem == 0 or drop_last:
        return out

    # 需要补齐最后一个 chunk
    tail = x[n_full * tgt_size:]        # [rem]
    need = tgt_size - rem               # 还需补的长度

    if fill_last == "repeat":
        # 对 tail 本身重复直到达到 t，再截断
        last = torch.cat([tail, tail.repeat(math.ceil(need / rem))])[:tgt_size]
    elif fill_last == "wrap":
        # 从序列开头环绕填充
        last = torch.cat([tail, x[:need]])
    elif fill_last == "pad_zero":
        # 用 0 填充
        last = torch.cat([tail, x.new_zeros(need)])
    else:
        raise ValueError(f"invalid fill_last='{fill_last}'")

    return torch.cat([out, last.unsqueeze(0)], dim=0)  # [B_full+1, t]


class VAEShmDataset(BaseTTSShmDataset):
    def get_batcher(self, hparams, global_stores):
        return get_from_global_stores(
            'batcher', global_stores,
            lambda: BucketBatcher(
                buckets=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550,
                         600, 650, 700, 750, 800, 850, 900, 950, 1000, 1200, 1400,
                         1600, 1800, 2000, 2400, 2800, 3000, 3500, 4000, 4500, 5000,
                         6000, 7000, 8000, 9000, 10000, 11000, 12000, 14000, 16000, 18000, 20000],
                dynamic_batch=hparams.get("dynamic_batch", True),   # False
                batch_size=hparams['max_sentences'],
                maximum_bucket_size=hparams.get('max_tokens', None),
                length_fn=lambda x: x['len'],
            )
        )
    
    def _process_item(self, processer_fn, raw_item, tgt_size, hparams, global_stores, i_worker, n_worker):
        fm = hparams['frames_multiple']
        hop_size = hparams['hop_size']
        fm_wav = fm * hop_size
        sr = hparams['audio_sample_rate']
        tgt_size = tgt_size // fm * fm
        
        skip_logger: SkipLogger = get_from_global_stores(
            'skip_logger', global_stores, 
            lambda: SkipLogger(interval=1000, i_worker=i_worker, n_worker=n_worker)
        )
        
        items = processer_fn(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker)
        if items is None or len(items) <= 0:
            return
        
        for item in items:
            wav = item['wav']
            chunked_wavs = repeat_or_chunk_1d(wav, tgt_size=tgt_size * hop_size, drop_last=False)
            # print(f"{chunked_wavs.shape = } {tgt_size = }")

            for i in range(chunked_wavs.shape[0]):
                yield {
                    'wav': chunked_wavs[i],
                    'len': len(chunked_wavs[i]) // hop_size
                }

        
    def collater(self, samples):
        if len(samples) == 1 and isinstance(samples[0], list):
            samples = samples[0]
        if len(samples) == 0:
            if hasattr(self, 'backup_batch') and self.backup_batch is not None:
                print('use backup batch!')
                return self.backup_batch
            else:
                print('no batch to take!')
                return {}
        wavs = collate_xd([s['wav'] for s in samples], 0.0) if 'wav' in samples[0] and samples[0]['wav'] is not None else None
        wav_lengths = torch.LongTensor([s['wav'].shape[0] for s in samples]) if wavs is not None else None
        
        batch = {
            'nsamples': len(samples),
            'wavs': wavs,
            'wav_lengths': wav_lengths,
        }
        if not hasattr(self, 'backup_batch') or self.backup_batch is None or random.random() < 0.001:
            self.backup_batch = batch

        return batch
    

def processer_fn_megatts3(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    items = []
    for item_ in raw_item:
        try:
            item = {}
            item['wav'] = torch.FloatTensor(item_['wav'])
            item['wav_len'] = item['wav'].shape[0]
            item['item_name'] = item_['item_name']
            item['spk_name'] = item_['spk_name']
            items.append(item)
            skip_logger.step(1)
        except:
            skip_logger.report(1, 'megatts3')
            continue
    return items

def processer_fn_prompttts(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    fm = hparams['frames_multiple']
    hop_size = hparams['hop_size']
    fm_wav = fm * hop_size
    sr = hparams['audio_sample_rate']
    
    items = []
    for item_ in raw_item:
        try:
            wav = (item_.get('wav') or np.zeros(0, dtype=float)).astype(float)
            org_sr = item_.get('sr', sr)

            if sr != org_sr and wav.size > 0:
                wav = librosa.resample(wav, orig_sr=org_sr, target_sr=sr)
            
            if wav.size > 0 and fm_wav > 0:
                wav = wav[: (len(wav) // fm_wav) * fm_wav]

            if wav.size == 0:
                return

            item = {}
            item['wav'] = torch.FloatTensor(wav)
            item['wav_len'] = item['wav'].shape[0]
            items.append(item)
            skip_logger.step(1)
        except:
            skip_logger.report(1, 'megatts3')
            continue

    return items

def processer_fn_mtg_jamendo(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    fm = hparams['frames_multiple']
    hop_size = hparams['hop_size']
    fm_wav = fm * hop_size
    sr = hparams['audio_sample_rate']

    hdfs_clients = get_from_global_stores(
        'hdfs_clients', global_stores,
        lambda: {}
    )
    
    with tempfile.TemporaryDirectory(dir='/dev/shm') as temp_dir:
        items = []
        for item_ in raw_item:
            wav_path = safe_read_path(item_['wav_path'], os.path.join(temp_dir, f"{uuid.uuid4()}.wav"), hdfs_clients)
            try:
                wav, _ = librosa.load(wav_path, sr=sr)
                if wav.size > 0 and fm_wav > 0:
                    wav = wav[: (len(wav) // fm_wav) * fm_wav]
                if wav.size == 0:
                    return
                item = {}
                item['wav'] = torch.FloatTensor(wav)
                item['wav_len'] = item['wav'].shape[0]
                items.append(item)
                skip_logger.step(1)
            except:
                skip_logger.report(1, 'mtg_jamendo')
                continue
        
    return items

def processer_fn_audioset(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    fm = hparams['frames_multiple']
    hop_size = hparams['hop_size']
    fm_wav = fm * hop_size
    sr = hparams['audio_sample_rate']

    items = []
    for item_ in raw_item:
        wav_path = item_.get('wav_24k_path', item_['wav_path'])
        try:
            wav, _ = librosa.load(wav_path, sr=sr)
            if wav.size > 0 and fm_wav > 0:
                wav = wav[: (len(wav) // fm_wav) * fm_wav]
            if wav.size == 0:
                return
            item = {}
            item['wav'] = torch.FloatTensor(wav)
            item['wav_len'] = item['wav'].shape[0]
            items.append(item)
            skip_logger.step(1)
        except:
            skip_logger.report(1, 'audioset')
            continue
    
    return items

if __name__ == '__main__':
    # client = HDFSClient(namespace='harunava')
    from tasks.tts.dataset_utils.tts_fastdataset_v2 import get_hdfs_file
    with tempfile.TemporaryDirectory(dir='/dev/shm') as temp_dir:
        wav_path = get_hdfs_file('hdfs://harunava/home/byte_advertising_genai/20250808/liruiqi/data/music/mtg-jamendo/train_sp/193/1024417[0012].wav', f"{temp_dir}/audio.wav", {})
        print(f"{wav_path = }")

