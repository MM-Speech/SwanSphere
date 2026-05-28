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
from pathlib import Path

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
from utils.commons.io import get_wav_duration, print_once, load_samples_from_tsv, load_samples_from_jsonl
from utils.commons.base_shm_dataset import BaseFalconReaderShmDataset, get_from_global_stores, save_samples_to_shm
from utils.commons.dataset_utils import collate_xd, pad_or_cut_xd, SkipLogger
from utils.commons.tensor_utils import convert_to_tensor, convert_to_np
from utils.commons.tos_utils_v2 import TosClient
from utils.commons.hdfs_utils import HDFSClient
from utils.dataset.batcher import BucketBatcher
from utils.audio.vad import build_vad_model, run_vad_trim
from utils.audio.align import mel2token_to_dur
from utils.audio.align import mel2token_to_dur
from utils.text.split_text import get_word_list
from utils.text.ph_tone_convert import map_phone_to_tokendict
from utils.text.split_text import get_word_list, remove_spaces_between_chinese
from utils.text import is_chinese, is_english

from tasks.tts.dataset_utils.tts_fastdataset_v2 import BaseTTSShmDataset
from modules.asr.tagger.model import TaggerTokenizer

DEBUG = False


class TaggerShmDataset(BaseTTSShmDataset):
    def get_batcher(self, hparams, global_stores):
        batcher = get_from_global_stores(
            'batcher', global_stores,
            lambda: BucketBatcher(
                buckets=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 
                            600, 650, 700, 750, 800, 850, 900, 950, 1000, 1200, 1400, 
                            1600, 1800, 2000, 2400, 2800, 3000, 4000],
                dynamic_batch=hparams.get("dynamic_batch", True),
                batch_size=hparams['max_sentences'],
                maximum_bucket_size=hparams['max_tokens'],
                length_fn=lambda x: x['len'],
            )
        )
        return batcher
    
    def _process_item(self, processer_fn, raw_item, tgt_size, hparams, global_stores, i_worker, n_worker):
        skip_logger: SkipLogger = get_from_global_stores(
            'skip_logger', global_stores, 
            lambda: SkipLogger(interval=1000, i_worker=i_worker, n_worker=n_worker)
        )
        
        items = processer_fn(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker)
        if items is None or len(items) <= 0:
            return
        
        for item in items:
            item['len'] = item['wav_len'] // 160
            yield item
            
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
            'age': torch.LongTensor([s['age'] for s in samples]),
            'gender': torch.LongTensor([s['gender'] for s in samples]),
            'emotion': torch.LongTensor([s['emotion'] for s in samples]),
            'pitch': torch.Tensor([s['pitch'] for s in samples]),
            'pitch_std': torch.Tensor([s['pitch_std'] for s in samples]),
            'speed': torch.Tensor([s['speed'] for s in samples]),
            'subset': [s['subset'] for s in samples],
        }
        
        if not hasattr(self, 'backup_batch') or self.backup_batch is None or random.random() < 0.001:
            self.backup_batch = batch

        return batch
    
    
def processer_fn_voxbox(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    
    tagger_tokenizer: TaggerTokenizer = get_from_global_stores(
        'tagger_tokenizer', global_stores, lambda: TaggerTokenizer()
    )
    
    items = []
    for item_ in raw_item:
        try:
            if item_['split'] == 'test':
                continue
            wav_path = item_['wav_path']
            wav_path = os.path.join('/mnt/bn/sa-ag-data/panchanghao/data', wav_path)
            if not os.path.isfile(wav_path):
                skip_logger.step(); continue
            wav, _ = librosa.load(wav_path, sr=16000)
            wav_len = len(wav)
            item = {
                'wav': wav,
                'wav_len': wav_len,
                'age': tagger_tokenizer.encode([item_['age']], 'age')[0],
                'gender': tagger_tokenizer.encode([item_['gender']], 'gender')[0],
                'emotion': tagger_tokenizer.encode([item_['emotion']], 'emotion')[0],
                'pitch': item_['pitch'],
                'pitch_std': item_['pitch_std'],
                'speed': item_['speed'],
                'subset': item_['wav_path'].split('/')[2]
            }
            items.append(item)
            skip_logger.step()
        except:
            traceback.print_exc()
            skip_logger.report()
            continue
        
    return items 
        
        
    