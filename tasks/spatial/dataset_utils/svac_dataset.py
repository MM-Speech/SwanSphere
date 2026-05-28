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

import torch
import torchaudio
import numpy as np
import torch.utils
import torch.utils.data
import librosa
import soundfile as sf
from audiotools import AudioSignal
# import dac as dac_module
# from dataloader import FalconReader, KVReader

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
from utils.text.split_text import get_word_list, remove_spaces_between_chinese
from utils.text.ph_tone_convert import map_phone_to_tokendict
from utils.text import is_chinese, is_english

from tasks.tts.dataset_utils.tts_datasets import MegaTTSDataset, FrontendLMDataset
from modules.tts.ar_dur.commons.align_ops import compute_mel2aug_from_dur
from modules.tts.ar_dur.commons.nar_tts_modules import LengthRegulator
from tasks.tts.dataset_utils.tts_fastdataset_v2 import get_hdfs_file,safe_read_path,BaseTTSShmDataset
from dataclasses import dataclass

DEBUG = False

# ======= Skip 打印辅助 =======
def _print_skip(reason: str, i_worker=None, n_worker=None, item_name: str = None, extra: str = ""):
    if DEBUG is False:
        return
    try:
        worker_info = f"{i_worker}/{n_worker}" if i_worker is not None and n_worker is not None else "-"
        name_info = f", item={item_name}" if item_name else ""
        extra_info = f", {extra}" if extra else ""
        print(f"[SKIP][{worker_info}] {reason}{name_info}{extra_info}")
    except Exception:
        # 避免打印本身异常影响主逻辑
        pass



class PromptAudioShmDataset(BaseTTSShmDataset):
    def _process_item(self, processer_fn, raw_item, tgt_size, hparams, global_stores, i_worker, n_worker):        
        ### Fix: 这个logger之后改
        skip_logger: SkipLogger = get_from_global_stores(
            'skip_logger', global_stores,
            lambda: SkipLogger([
                'no_score_cnt',
                'no_text_cnt',
                'no_caption_cnt',
                'no_phone_cnt',
            ], interval=1000, i_worker=i_worker, n_worker=n_worker)
        )

        items = processer_fn(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker)
        if DEBUG:
            print("after process_fn")
            print('items[0]:', items[0].keys())
        if items is None or len(items) == 0:
            _print_skip("processer_fn_returned_none_or_empty", i_worker, n_worker, extra=f"tgt_size={tgt_size}")
            return
        
        for item_tgt in items:
            
            item_tgt['len'] = 35 # 暂时无意义
            if DEBUG:
                print(f"in return: {item_tgt.keys() =}")
            yield item_tgt
            # skip_logger.step(1)


def processer_fn_sphere360(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    # audio_mae_base_path = '/data/leike/spatial/sphere360/test_audio_mae'
    # audio_time_mae_base_path = '/data/leike/spatial/sphere360/negative/time'
    # audio_rot_mae_base_path = '/data/leike/spatial/sphere360/negative/rot'
    
    # video_mae_base_path = '/data/leike/spatial/sphere360/test_front_erp_mae'
    # video_rot_mae_base_path = '/data/leike/spatial/sphere360/negative/vid_rot'
    
    audio_mae_base_path = '/data/leike/spatial/sphere360/negative/train_audio_mae'
    audio_time_mae_base_path = '/data/leike/spatial/sphere360/negative/train_time'
    audio_rot_mae_base_path = '/data/leike/spatial/sphere360/negative/train_rot'
    
    video_mae_base_path = '/data/leike/spatial/sphere360/train_front_erp_mae'
    video_rot_mae_base_path = '/data/leike/spatial/sphere360/negative/train_vid_rot'
    
    VIDEO_LEN = 40
    AUDIO_LEN = 64
    
    items = []
    
    for _item in raw_item:
        file_name = _item['file_name']
        
        try:
            vid_mae = np.load(os.path.join(video_mae_base_path, file_name))
            foa_mae = np.load(os.path.join(audio_mae_base_path, file_name))
            
            rot_vid = np.load(os.path.join(video_rot_mae_base_path, file_name))
            time_foa = np.load(os.path.join(audio_time_mae_base_path, file_name))
            rot_foa = np.load(os.path.join(audio_rot_mae_base_path, file_name))
        except Exception as e:
            continue
        
        L_v, *_ = vid_mae.shape
        if L_v >= VIDEO_LEN:
            vid_mae = vid_mae[:VIDEO_LEN]
            rot_vid = rot_vid[:VIDEO_LEN]
        else:
            # shape 都是[len, 256, 1408]，用0pad到VIDEO_LEN
            pad_len = VIDEO_LEN - L_v
            # np.pad((前补, 后补), (前补, 后补), ...)
            # 只在第0维(时间维)后面补0，其他维度不补
            pad_width_vid = ((0, pad_len), (0, 0), (0, 0))
            
            vid_mae = np.pad(vid_mae, pad_width_vid, mode='constant', constant_values=0)
            rot_vid = np.pad(rot_vid, pad_width_vid, mode='constant', constant_values=0)
        
        L_a, *_ = foa_mae.shape
        if L_a >= AUDIO_LEN:
            foa_mae = foa_mae[:AUDIO_LEN]
            time_foa = time_foa[:AUDIO_LEN]
            rot_foa = rot_foa[:AUDIO_LEN]
        else:
            # shape 都是[len, 4, 8, 768]，用0pad到AUDIO_LEN
            pad_len = AUDIO_LEN - L_a
            # 只在第0维(时间维)后面补0
            pad_width_aud = ((0, pad_len), (0, 0), (0, 0), (0, 0))
            
            foa_mae = np.pad(foa_mae, pad_width_aud, mode='constant', constant_values=0)
            time_foa = np.pad(time_foa, pad_width_aud, mode='constant', constant_values=0)
            rot_foa = np.pad(rot_foa, pad_width_aud, mode='constant', constant_values=0)
        
        item = {
            'vid_mae': torch.from_numpy(vid_mae),
            'rot_vid': torch.from_numpy(rot_vid),
            'foa_mae': torch.from_numpy(foa_mae),
            'time_foa': torch.from_numpy(time_foa),
            'rot_foa': torch.from_numpy(rot_foa),
        }
        if vid_mae is None or rot_vid is None or foa_mae is None or time_foa is None or rot_foa is None:
            continue
        
        items.append(item)
    return items

if __name__ == '__main__':
    raw_item = [
        {'file_name': '_css5l3_vpY_155.npy'}
    ]
    items = processer_fn_sphere360(raw_item, 0, None, None, None, 0, 0)

    import pdb; pdb.set_trace()