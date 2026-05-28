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
import torch.nn.functional as F
# from dataloader import FalconReader, KVReader

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
            
            yield {
                'wav': wav,
                'len': 35 # 无意义
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
        # wav_lengths = torch.LongTensor([s['wav'].shape[0] for s in samples]) if wavs is not None else None
        
        batch = {
            'nsamples': len(samples),
            'wavs': wavs,
            # 'wav_lengths': wav_lengths,
        }
        if not hasattr(self, 'backup_batch') or self.backup_batch is None or random.random() < 0.001:
            self.backup_batch = batch

        return batch
    
def load_opus(path):
    try:
        # sr=None: 保持原始采样率
        # mono=False: 保持多声道 (librosa默认会混合成单声道，必须关掉)
        waveform, sr = librosa.load(path, sr=None, mono=False)
        
        # 转为 Tensor
        waveform = torch.from_numpy(waveform)
        
        # librosa 读取单声道时形状是 [Time]，需要升维成 [1, Time] 以对齐 torchaudio
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        return waveform, sr
    except Exception as e:
        return None, None
    
def processer_fn_ytambigen(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    sample_rate = hparams['sample_rate']
    
    # === 1. 设定目标长度标准 ===
    target_seconds = 10.0
    # 计算目标采样点数：例如 10 * 44100 = 441000
    target_length = int(target_seconds * sample_rate)

    items = []
    for item_ in raw_item:
        wav_path = item_['wav_path']
        # wav, sr = load_opus(wav_path)
        wav, sr = torchaudio.load(wav_path)
        if wav is None or sr is None:
            # skip_logger.report(1, 'yt-ambigen')
            continue
        try:
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)
                sr = sample_rate
                
            #### 对齐
            current_length = wav.shape[-1]
            if current_length > target_length:
                #情况 A: 太长了 -> 裁剪末尾
                wav = wav[:, :target_length]
            elif current_length < target_length:
                # 情况 B: 太短了 -> 末尾补 0
                pad_amount = target_length - current_length
                # F.pad 的参数格式对于 2D 输入是 (pad_left, pad_right)
                # 这里的 "constant", 0 表示用常数 0 填充（即静音）
                wav = F.pad(wav, (0, pad_amount), "constant", 0)
            
            item = {}
            item['wav'] = torch.FloatTensor(wav)
            
            items.append(item)
            # skip_logger.step(1)

        except Exception as e:
            # skip_logger.report(1, 'yt-ambigen')
            continue
        
    return items

if __name__ == '__main__':
    raw_item = [
        {'wav_path': '/data/leike/spatial/YT-Ambigen/audio_10/VX_gOGFgt14_20.opus'}
    ]
    
    hparams = {
        'sample_rate': 44100
    }
    
    items = processer_fn_ytambigen(raw_item, None, hparams, None, None, None, None)
    import pdb; pdb.set_trace()