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

    def _process_item_old(self, processer_fn, raw_item, tgt_size, hparams, global_stores, i_worker, n_worker):

        hop_size = hparams['hop_size']
        fm = hparams['frames_multiple']
        fm_wav = hparams['frames_multiple'] * hparams['hop_size']
        sr = hparams['audio_sample_rate']

        speech_augmentor = None
        if hparams.get('wav_add_noise', False) or hparams.get('wav_add_effect', False):
            from tasks.tts.dataset_utils.augment import SpeechAugment
            speech_augmentor = get_from_global_stores(
                'speech_augmentor', global_stores,
                lambda: SpeechAugment(
                    hparams.get('wav_add_noise', False), hparams.get('wav_add_effect', False), hparams.get('musan_dir', None),
                    noise_prob=hparams.get('wav_add_noise_prob', 0.5), effect_prob=hparams.get('wav_add_effect_prob', 0.5),
                    noise_snr=(6.0, 20.0), with_speech=hparams.get('musan_with_speech', False)
                )
            )

        if hparams.get('add_vad_mask', False):
            from utils.audio.vad import get_vad_model
            vad_model = get_from_global_stores(
                'vad_model', global_stores,
                lambda: get_vad_model()
            )

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
        if items is None or len(items) == 0:
            _print_skip("processer_fn_returned_none_or_empty", i_worker, n_worker, extra=f"tgt_size={tgt_size}")
            return

        for item_tgt in items:
            mel_len_total = item_tgt['wav_len'] // hop_size
            if not (hparams['max_frames'] >= mel_len_total > hparams['min_frames']):
                skip_logger.update(1)
                _print_skip(
                    "frames_out_of_range",
                    i_worker, n_worker,
                    item_name=item_tgt.get('item_name', ''),
                    extra=f"mel_len={mel_len_total}, allowed=({hparams['min_frames']}, {hparams['max_frames']}]"
                )
                continue

            # text：caption 数据集可以直接用 processer_fn 写好的 txt，其它数据仍走 raw_text_process
            if item_tgt.get('use_raw_txt_as_text', False):
                txt = item_tgt['txt']
            else:
                txt = raw_text_process(item_tgt['txt'], wav_len=item_tgt['wav_len'])
            if txt is None:
                txt=''
            item_tgt['text'] = txt

            # phone / tone
            item_tgt['ph_token'] = item_tgt['phone']
            # 这里不再重复做 “phone_len vs latent_len” 的过滤，
            # 只保留在各自的 processer_fn 中做一次即可，避免重复计算。

            if hparams.get('load_wav', True):
                # 对齐到 frames_multiple * hop_size 的整数倍
                if fm_wav > 0:
                    item_tgt['wav'] = item_tgt['wav'][:item_tgt['wav'].shape[0] // fm_wav * fm_wav]
                if speech_augmentor is not None:
                    try:
                        item_tgt['wav'] = speech_augmentor(item_tgt['wav'], sr)
                    except Exception as e:
                        _print_skip(
                            "speech_augmentor_failed",
                            i_worker, n_worker,
                            item_name=item_tgt.get('item_name', ''),
                            extra=f"err={str(e)}"
                        )

            mel_len = item_tgt['wav'].shape[0] // hop_size

            # ctx 相关
            min_idx = max(int(mel_len * 0.1), 200)
            max_idx = min(int(mel_len * 0.9), mel_len - 200)
            if min_idx > max_idx:
                min_idx = int(mel_len * 0.4)
                max_idx = int(mel_len * 0.6)
            rand_length = random.randint(min_idx, max_idx) // fm * fm

            ctx_mask = torch.zeros((mel_len, 1))
            ctx_mask[:rand_length] = 1.0
            item_tgt['ctx_mask'] = ctx_mask[::hparams['vae_stride']]

            item_tgt['ctx_wav'] = deepcopy(item_tgt['wav'])
            item_tgt['ctx_wav'] = item_tgt['ctx_wav'][:rand_length * hparams['hop_size']]

            if hparams.get('add_vad_mask', False):
                try:
                    vad_start, vad_end = run_vad_trim(item_tgt['wav'], hparams['audio_sample_rate'], vad_model)
                    vm = hparams['hop_size'] * hparams['vae_stride']
                    wav_len = item_tgt['wav'].shape[0]
                    vad_mask = np.zeros((wav_len // vm))
                    vad_mask[int(vad_start * hparams['audio_sample_rate'] // vm): int(
                        vad_end * hparams['audio_sample_rate'] // vm)] = 1
                    item_tgt['vad_mask'] = vad_mask  # 直接是 lat 的 shape
                except Exception as e:
                    item_tgt['vad_mask'] = None
                    _print_skip(
                        "vad_mask_failed",
                        i_worker, n_worker,
                        item_name=item_tgt.get('item_name', ''),
                        extra=f"err={str(e)}"
                    )
            else:
                item_tgt['vad_mask'] = None

            item_tgt['len'] = mel_len // 4
            yield item_tgt
            skip_logger.step(1)

    def _process_item(self, processer_fn, raw_item, tgt_size, hparams, global_stores, i_worker, n_worker):
        dac_base_path = hparams['dac_base_path']
        rotation_base_path = hparams['rotation_base_path']
        clip_base_path = hparams['clip_base_path']
        energy_map_path = hparams['energy_map_path']
        
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


def prepare_ambi(file_path, augment=False):
    with open("/home/leike/spatial/visage/csv/yt_ambigen/ambi.json") as f:
        ambi_data = json.load(f)
    
    key = file_path.replace(".npy", "")
    elevation, azimuth = ambi_data[key]
    if augment:
        azimuth = azimuth + 45
        if azimuth > 179:
            azimuth = azimuth - 180
    direction = (
        roty((2 * azimuth - 179) * np.pi / 180)
        @ rotx((89 - 2 * elevation) * np.pi / 180)
        @ np.array([0.0, 0.0, 1.0])
    )
    direction = np.float32(direction)

    return direction

def roty(ang):
    return np.array(
        [[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]]
    )

def rotx(ang):
    return np.array(
        [[1, 0, 0], [0, np.cos(ang), -np.sin(ang)], [0, np.sin(ang), np.cos(ang)]]
    )
    
def prepare_dac(dac: np.ndarray, args) -> np.ndarray:
    """
    Input: DAC codes of shape [s, n]
    Output: Decoder input ids of shape [2s+1, n]
    """
    # DAC -> Pad & Shift
    decoder_input_ids = []
    residual_codebooks = tuple(range(1, args.dac_num_codebooks))

    for idx in range(dac.shape[1]):
        decoder_input_id = np.full(
            dac.shape[0] * 2 + 1, args.dac_pad_token_id, dtype=dac.dtype
        )

        if idx == 0:  # W_p
            decoder_input_id[::2][:-1] = dac[:, idx]
        elif (
            idx in residual_codebooks or idx % args.dac_num_codebooks == 0
        ):  # W_r and S_p
            decoder_input_id[1::2] = dac[:, idx]
        else:  # S_r
            decoder_input_id[2::2] = dac[:, idx]
        decoder_input_ids.append(decoder_input_id)

    decoder_input_ids = np.stack(decoder_input_ids, axis=1)

    return decoder_input_ids

def prepare_labels(decoder_input_ids: np.ndarray, args) -> np.ndarray:
    """
    Input: Decoder input ids of shape [2s+1, n]
    Output: Labels of shape [2s+1, n]
    """
    # Labels -> Add pad token for last time step
    labels = decoder_input_ids.copy()
    labels[labels == args.dac_pad_token_id] = args.label_pad_token_id

    return labels

@dataclass
class ProcesserArgs:
    dac_base_path: Path = None
    dac_z_base_path: Path = None
    clip_base_path: Path = None
    rotation_base_path: Path = None
    energy_map_path: Path = None
    raw_opus_base_path: Path = None
    global_clip_base_path: Path = None
    seconds_to_use: int = 10
    dac_pad_token_id: int = 1024
    dac_num_codebooks: int = 9
    label_pad_token_id: int = -100
    dac_frame_rate: int = 86
    stable_frame_rate: float = 21.5
    clip_frame_rate: int = 4
    ambi_path = "csv/yt_ambigen/ambi.json"
     
dac_ins = None   
latent_mean = None
latent_std = None
def processer_fn_spatial(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    
    processer_args = ProcesserArgs(
        dac_base_path = hparams['dac_base_path'],
        dac_z_base_path = hparams['dac_z_base_path'],
        rotation_base_path = hparams.get('rotation_base_path', None),
        clip_base_path = hparams['clip_base_path'],
        global_clip_base_path = hparams.get('global_clip_base_path', None),
        energy_map_path = hparams.get('energy_map_path', None),
        raw_opus_base_path = hparams.get('raw_opus_base_path', None),
        # 剩下的怎么赋值？
    )
    
    items = []
    
    for _item in raw_item:
        file_name = _item['file_name']
        # dac_path = os.path.join(processer_args.dac_base_path, file_name)
        # dac_rotate_path = os.path.join(processer_args.rotation_base_path, file_name)
        dac_z_path = os.path.join(processer_args.dac_z_base_path, file_name)
        clip_path = os.path.join(processer_args.clip_base_path, file_name)
        energy_map_path = os.path.join(processer_args.energy_map_path, file_name)
        if processer_args.global_clip_base_path is not None:
            global_clip_path = os.path.join(processer_args.global_clip_base_path, file_name)
        
        ### load dac
        # rnd = random.random()
        # if processer_args.rotation_base_path is not None and rnd < 0.5:
        #     dac = np.load(dac_rotate_path)
        #     augment = True
        # else:
        #     dac = np.load(dac_path)
        #     augment = False
        # dac = np.load(dac_path)
        
        # Fix：还没详细的处理
        dac_z = np.load(dac_z_path)
        
        augment = False
        
        # dac = np.transpose(dac, axes=(1, 0))
        # dac = dac[:, : processer_args.dac_num_codebooks * 4]
        dac_z = np.transpose(dac_z, axes=(1, 0))
        clip_embedding = np.load(clip_path)
        if processer_args.global_clip_base_path is not None:
            global_clip_embedding = np.load(global_clip_path)
        # if processer_args.energy_map_path is not None:
        #     energy_map = np.load(energy_map_path)
        # else:
        #     energy_map = None
        # direction = prepare_ambi(file_name, augment=augment)
        
        
        ### load clip embedding, energy map
        max_second = int(dac_z.shape[0] / processer_args.stable_frame_rate)
        # import pdb; pdb.set_trace()
        if max_second > processer_args.seconds_to_use:
            max_second = min(
                max_second, clip_embedding.shape[0] // processer_args.clip_frame_rate
            )
            start_second = random.randint(0, max_second - processer_args.seconds_to_use)
            clip_embedding = clip_embedding[
                start_second
                * processer_args.clip_frame_rate : (start_second + processer_args.seconds_to_use)
                * processer_args.clip_frame_rate
            ]
            global_clip_embedding = global_clip_embedding[
                start_second
                * processer_args.clip_frame_rate : (start_second + processer_args.seconds_to_use)
                * processer_args.clip_frame_rate
            ]
            # if energy_map is not None:
            #     energy_map = energy_map[
            #         start_second
            #         * processer_args.clip_frame_rate : (
            #             start_second + processer_args.seconds_to_use
            #         )
            #         * processer_args.clip_frame_rate
            #     ]
            # dac = dac[
            #     int(start_second
            #     * processer_args.stable_frame_rate) : int((start_second + processer_args.seconds_to_use)
            #     * processer_args.stable_frame_rate)
            # ]
            dac_z = dac_z[
                int(start_second
                * processer_args.stable_frame_rate) : int((start_second + processer_args.seconds_to_use)
                * processer_args.stable_frame_rate)
            ]
            import pdb; pdb.set_trace()
        else:
            clip_embedding = clip_embedding[
                : processer_args.seconds_to_use * processer_args.clip_frame_rate
            ]
            global_clip_embedding = global_clip_embedding[
                : processer_args.seconds_to_use * processer_args.clip_frame_rate
            ]
            # if energy_map is not None:
            #     energy_map = energy_map[
            #         : processer_args.seconds_to_use * processer_args.clip_frame_rate
            #     ]
            dac_pad = int(processer_args.seconds_to_use * processer_args.stable_frame_rate) - dac_z.shape[0]
            # import pdb; pdb.set_trace()
            if dac_pad > 0:
                # dac = np.pad(
                #     dac,
                #     ((0, dac_pad), (0, 0)),
                #     constant_values=processer_args.dac_pad_token_id,
                # )
                dac_z = np.pad(
                    dac_z,
                    ((0, dac_pad), (0, 0)),
                    constant_values=0,
                )
            else:
                # dac = dac[: int(processer_args.seconds_to_use * processer_args.stable_frame_rate)]
                dac_z = dac_z[: int(processer_args.seconds_to_use * processer_args.stable_frame_rate)]
                
        if hparams['latent_norm'] == True:
            global latent_mean
            global latent_std
            if latent_mean is None and latent_std is None:
                # latent_mean = torch.from_numpy(np.load('/mnt/bn/sa-ag-data/leike/spatial/data/YT-Ambigen/dac_z_stats/mean.npy'))
                # latent_std = torch.from_numpy(np.load('/mnt/bn/sa-ag-data/leike/spatial/data/YT-Ambigen/dac_z_stats/std.npy'))
                # latent_mean = torch.tensor(latent_mean, dtype=dac_z.dtype)
                # latent_std = torch.tensor(latent_std, dtype=dac_z.dtype)
                latent_mean = np.load(os.path.join(hparams['norm_path'], 'mean.npy'))
                latent_std = np.load(os.path.join(hparams['norm_path'], 'std.npy'))
                latent_mean = np.array(latent_mean, dtype=dac_z.dtype)
                latent_std = np.array(latent_std, dtype=dac_z.dtype)

            
            dac_z = (dac_z - latent_mean) / (latent_std + 1e-6)
        
        if dac_z.shape[0] != int(processer_args.stable_frame_rate * processer_args.seconds_to_use):
            raise ValueError(f"Error processing example {file_name}")
        
        
        inputs_embeds = clip_embedding
        # decoder_input_ids = prepare_dac(dac, processer_args)
        # labels = prepare_labels(decoder_input_ids, processer_args)
        
        # global dac_ins
        # if dac_ins is None:
        #     print(f"first load dac model")
        #     model_path = dac_module.utils.download(model_type="44khz")
        #     dac_ins = dac_module.DAC.load(model_path)
        #     dac_ins.eval()
        # raw_opus_path = os.path.join(processer_args.raw_opus_base_path, file_name)
        # raw_opus_path = raw_opus_path.replace('.npy', '.opus')
        # raw_audio, sr = sf.read(raw_opus_path)
        # raw_audio = np.transpose(raw_audio, (-1, -2))
        # signals = []
        # ### Fix: 还有rotate的要处理一下
        # for channel_idx, channel in enumerate(raw_audio):
        #     signal = AudioSignal(channel, sr)#.to(device)
        #     signal.resample(44100) # dac_sample_rate
        #     signals.append(signal)
            

        # batched_signal =  AudioSignal.batch(signals, pad_signals=True)
        # x = dac_ins.preprocess(batched_signal.audio_data, batched_signal.sample_rate)
        # with torch.no_grad():
        #     z, codes, latents, _, _ = dac_ins.encode(x, n_quantizers=9) # z: [4, 1024, t]
        

        item = {}
        item["inputs_embeds"] = torch.tensor(inputs_embeds)
        item["global_clip_embedding"] = torch.tensor(global_clip_embedding)
        # item["decoder_input_ids"] = torch.tensor(decoder_input_ids[:-1])
        # item["labels"] = torch.tensor(labels)
        # item["direction"] = torch.from_numpy(direction)
        # item["lat"] = z.reshape(1, 4*1024, -1).squeeze()
        item["lat"] = torch.tensor(dac_z)
        item['file_name'] = file_name
        # if energy_map is not None:
        #     item["energy_map"] = torch.tensor(energy_map)
        
        items.append(item)
        # skip_logger.step(1)
    

    return items
        
sphere360_lat_mean = None
sphere360_lat_std = None
def processer_fn_sphere360(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    
    processer_args = ProcesserArgs()
    
    stable_lat_base_path = '/data/leike/spatial/sphere360/train_stable_latents'
    clip_base_path = '/data/leike/spatial/sphere360/train_front_erp_clip'
    global_clip_base_path = '/data/leike/spatial/sphere360/train_pad_360_erp_clip'
    # raw_opus_base_path = ''

    items = []
    
    for _item in raw_item:
        file_name = _item['file_name']
        stable_lat_path = os.path.join(stable_lat_base_path, file_name)
        clip_path = os.path.join(clip_base_path, file_name)
        global_clip_path = os.path.join(global_clip_base_path, file_name)

        stable_lat = np.load(stable_lat_path)
        stable_lat = np.transpose(stable_lat, axes=(1, 0))
        
        clip_embedding = np.load(clip_path)
        global_clip_embedding = np.load(global_clip_path)
        
        max_second = stable_lat.shape[0] / processer_args.stable_frame_rate
        if max_second > processer_args.seconds_to_use:
            clip_embedding = clip_embedding[
                : processer_args.seconds_to_use * processer_args.clip_frame_rate
            ]
            global_clip_embedding = global_clip_embedding[
                : processer_args.seconds_to_use * processer_args.clip_frame_rate
            ]
            stable_lat = stable_lat[
                : int(processer_args.seconds_to_use * processer_args.stable_frame_rate)
            ]
        else:
            # 1. 算出目标长度
            T_clip = int(processer_args.seconds_to_use * processer_args.clip_frame_rate)
            T_lat  = int(processer_args.seconds_to_use * processer_args.stable_frame_rate)

            # 2. 定义通用补0函数 (自动适配任意维度、设备、dtype)
            def pad(x, target_len):
                diff = target_len - x.shape[0]
                if diff <= 0: return x
                zeros = np.zeros((diff, *x.shape[1:]), dtype=x.dtype)
                return np.concatenate([x, zeros], axis=0)

            # 3. 批量应用
            clip_embedding = pad(clip_embedding, T_clip)
            global_clip_embedding = pad(global_clip_embedding, T_clip)
            stable_lat = pad(stable_lat, T_lat)
            
        if hparams['latent_norm'] == True:
            global sphere360_lat_mean
            global sphere360_lat_std
            if sphere360_lat_mean is None or sphere360_lat_std is None:
                sphere360_lat_mean = np.load('/data/leike/spatial/sphere360/train_stable_latents_stats/mean.npy')
                sphere360_lat_std  = np.load('/data/leike/spatial/sphere360/train_stable_latents_stats/std.npy')
                sphere360_lat_mean = np.array(sphere360_lat_mean, dtype=stable_lat.dtype)
                sphere360_lat_std  = np.array(sphere360_lat_std, dtype=stable_lat.dtype)
            stable_lat = (stable_lat - sphere360_lat_mean) / (sphere360_lat_std + 1e-6)
        
        if stable_lat.shape[0] != int(processer_args.stable_frame_rate * processer_args.seconds_to_use):
            print(f"{stable_lat.shape =}, {processer_args.stable_frame_rate * processer_args.seconds_to_use =}")
            raise ValueError(f"Error processing example {file_name}")
            
        inputs_embeds = clip_embedding
        
        item = {
            'inputs_embeds': torch.tensor(inputs_embeds),
            'global_clip_embedding': torch.tensor(global_clip_embedding),
            'lat': torch.tensor(stable_lat),
            'file_name': file_name,
        }
        items.append(item)
    return items

        
            
# if __name__ == '__main__':
#     from attrdict import AttrDict
#     hparams = AttrDict({
#         'dac_base_path':        '/data/leike/spatial/YT-Ambigen/dac',
#         'dac_z_base_path':        '/data/leike/spatial/YT-Ambigen/stable_latents_10',
#         'clip_base_path':       '/data/leike/spatial/YT-Ambigen/erp_clip_10',
#         'global_clip_base_path': '/data/leike/spatial/YT-Ambigen/erp_clip_360_10',
#         # 'rotation_base_path':   '/mnt/bn/sa-ag-data/leike/spatial/data/YT-Ambigen/dac_rotated',
#         # 'energy_map_path':      '/mnt/bn/sa-ag-data/leike/spatial/data/YT-Ambigen/energy_map',
#         # 'raw_opus_base_path':   '/mnt/bn/sa-ag-data/leike/spatial/data/YT-Ambigen/audio',

#         'latent_norm':          True,
#         'norm_path':            '/data/leike/spatial/YT-Ambigen/stable_latents_stats_10',
        
#         'dac_num_codebooks':    9,
#     })
    
#     raw_item = [{'file_name': 'VX_gOGFgt14_46.npy'}]
#     items = processer_fn_spatial(raw_item, None, hparams, None, None, None, None)
#     import pdb; pdb.set_trace()
    

if __name__ == '__main__':
    from attrdict import AttrDict
    hparams = AttrDict({
        # 'dac_base_path':        '/data/leike/spatial/YT-Ambigen/dac',
        'dac_z_base_path':        '/data/leike/spatial/YT-Ambigen/stable_latents_10',
        'clip_base_path':       '/data/leike/spatial/YT-Ambigen/erp_clip_10',
        'global_clip_base_path': '/data/leike/spatial/YT-Ambigen/erp_clip_360_10',
        # 'rotation_base_path':   '/mnt/bn/sa-ag-data/leike/spatial/data/YT-Ambigen/dac_rotated',
        # 'energy_map_path':      '/mnt/bn/sa-ag-data/leike/spatial/data/YT-Ambigen/energy_map',
        # 'raw_opus_base_path':   '/mnt/bn/sa-ag-data/leike/spatial/data/YT-Ambigen/audio',

        'latent_norm':          True,
        'norm_path':            '/data/leike/spatial/YT-Ambigen/stable_latents_stats_10',
        
        'dac_num_codebooks':    9,
    })
    
    raw_item = [{'file_name': '-076WPWoCRE_557.npy'}]
    items = processer_fn_sphere360(raw_item, None, hparams, None, None, None, None)
    import pdb; pdb.set_trace()

# if __name__ == '__main__':
#     import soundfile as sf
#     from utils.commons.io import json_dump
#     from tqdm import tqdm
#     from utils.commons.hparams import hparams, set_hparams
#     set_hparams()
#     # meta_path = '/mnt/bn/sa-ag-data/leike/spatial/data/vdataset_1spk.json'
#     # metas = json.load(open(meta_path))

#     # print(len(metas))
#     dataset = PromptAudioShmDataset(
#         prefix='train', hparams=hparams, use_fast_dataloader=True, rank_id=0, world_size=1, batch_size=16
#     )
#     print('init dataset over')
#     dataloader = dataset.get_dataloader(seed=1234, num_workers=4)
#     print('init dataloader over')

#     temp_dir = 'user/temp/test_dl'

#     for idx, batch in tqdm(enumerate(dataloader)):
#         if idx == 0:
#             print(batch.keys())
#             # del batch['wavs']
#             serializable_batch = {}
#             print(f"{batch.keys() = }")
#             import pdb; pdb.set_trace()
#             for key, value in batch.items():
#                 if isinstance(value, torch.Tensor):
#                     serializable_batch[key] = value.cpu().numpy().tolist()
#                 elif isinstance(value, np.ndarray):
#                     serializable_batch[key] = value.tolist()
#                 # 可以添加对其他类型的处理
#                 else:
#                     # 假设其他类型都是兼容的
#                     serializable_batch[key] = value
#             # --- 预处理结束 ---
#             # 现在保存净化后的字典
#             with open('my_batch.json', 'w', encoding='utf-8') as f:
#                 json.dump(serializable_batch, f, indent=4, ensure_ascii=False)
#         break