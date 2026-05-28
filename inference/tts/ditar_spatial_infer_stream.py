import os
import collections
import collections.abc
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))

import sys
import json
from typing import List, Union, Dict
import argparse
import librosa
import numpy as np
import torch
import io
import threading
import traceback
import torch.nn.functional as F
# import whisper
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import subprocess
from pathlib import Path
import time
import math

from copy import deepcopy
# from langdetect import detect as classify_language, LangDetectException
from pydub import AudioSegment
import pyloudnorm as pyln
from tqdm import tqdm
# from tn.chinese.normalizer import Normalizer as ZhNormalizer
# from tn.english.normalizer import Normalizer as EnNormalizer

from utils.audio.align import mel2token_to_dur
from utils.audio.io import save_wav_bytes, to_wav_bytes, wav_bytes_to_mp3_bytes
from utils.text import is_english, YUNMU_ERHUA, SHENGMU
from utils.text.text_encoder import TokenTextEncoder
from utils.text.split_text import chunk_text_english, chunk_text_chinese, get_word_list, remove_space, remove_unprintable
from utils.text.ph_tone_convert import split_ph_timestamp, split_ph, map_phone_to_tokendict
from utils.text.ssml_utils import SSML
from utils.text.ph_alignment import align_word_phone, print_align, merge_norm_alignment
from utils.text.split_text import chunk_text_chinese, chunk_text_english, chunk_text_chinese_v2
from utils.commons.ckpt_utils import load_ckpt, get_all_ckpt_steps
from utils.commons.hparams import set_hparams, hparams
from utils.commons.meters import Timer
from utils.commons.os_utils import handle_exacption, kill_void
from utils.commons.io import print_once, json_dumps, get_wav_duration
from utils.commons.import_utils import import_module_bystr, get_class_from_module
from utils.commons.tensor_utils import move_to_cpu, move_to_cuda
from utils.commons.dataset_utils import pad_or_cut_xd

from modules.tts.ar_dur.commons.nar_tts_modules import LengthRegulator
from modules.tts.ar_dur.commons.align_ops import compute_mel2aug_from_dur
from modules.tts.ditar.build_model_utils import DiTARBuildModelMixinS# DiTARBuildModelMixinV2, DiTARBuildModelMixinV3
# from tasks.tts.dataset_utils.promptaudio_fastdataset_v2 import build_spk_mask_from_text_tokens, _get_sx_token_patterns, augment_text_with_pinyin_s1s2_safe

from tasks.spatial.dataset_utils.visage_like_dataset import prepare_ambi
from stable_audio_tools.models.autoencoders import create_autoencoder_from_config

DEBUG = False

class DiTARSInfer(DiTARBuildModelMixinS):
    def __init__(self, device, ckpt):
        self.device = device
        self.sample_rate = 44100
        self.frame_rate = 21.5
        self.patch_size = 4
        
        self.build_model(ckpt)
        self.sa_vae = self.build_StableAudioVAE()
    
    def build_model(self, ckpt):
        set_hparams(config=os.path.join(ckpt, 'config.yaml'), print_hparams=False)
        self._build_model()
        load_ckpt(self.model, ckpt, 'model', strict=True)
        self.model.eval()
        self.model.to(self.device)
    
    def build_StableAudioVAE(self):
        json_path = '/home/leike/spatial/stable-audio-tools/stable_audio_tools/checkpoints/vae_model_config.json'
        vae_path = '/home/leike/spatial/stable-audio-tools/stable_audio_tools/checkpoints/vae_model.ckpt'
        with open(json_path, "r", encoding="utf-8") as f:
            cfg: dict = json.load(f)
        
        vae = create_autoencoder_from_config(cfg)
        vae_checkpoint = torch.load(vae_path, map_location='cpu')
        if "state_dict" in vae_checkpoint:
            state_dict = vae_checkpoint["state_dict"]
        else:
            state_dict = vae_checkpoint
        vae.load_state_dict(state_dict, strict=True)
        vae = vae.to(self.device)
        vae.eval()
        print('vae loaded')
        return vae
    
    def preprocess(self, test_id):
        
        inputs_embeds = np.load(os.path.join('/data/leike/spatial/YT-Ambigen/erp_clip_10', test_id))[:40]
        global_clip_embedding = np.load(os.path.join('/data/leike/spatial/YT-Ambigen/erp_clip_360_10', test_id))[:40]
        direction = prepare_ambi(test_id, augment=False)
        energy_map = np.load(os.path.join('/data/leike/spatial/YT-Ambigen/energy_map', test_id))[:20]
        
        gt = np.load(os.path.join('/data/leike/spatial/YT-Ambigen/stable_latents_10',test_id))
        gt = torch.tensor(gt).unsqueeze(0).to(self.device)
        
        video_feats = torch.from_numpy(inputs_embeds).to(self.device, dtype=torch.float32)
        vid_len = video_feats.shape[0]
        v_mask = torch.ones(vid_len, device=self.device, dtype=torch.long)
        video_feats = video_feats.unsqueeze(0)  # Shape: [1, Target_Len, Dim]
        v_mask = v_mask.unsqueeze(0)            # Shape: [1, Target_Len]
        
        self.metadata = {
            'target_lat_len': int(vid_len/4*self.frame_rate),
        }
        
        return video_feats, v_mask, gt
        
    
    @torch.no_grad()
    def forward_stream(self, test_id):

        video_feats, v_mask, gt = self.preprocess(test_id)
        inputs = {
            'video_feats': video_feats,
            'v_mask': v_mask,
            'direction': None,
            'energy_map': None,
        }
        
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # V2A 逻辑: 自动根据视频长度推导音频长度，无需 ref_lat
                generated_lat, _ = self.model.inference(inputs, use_tqdm=True)
                
        latent_mean = np.load('/data/leike/spatial/YT-Ambigen/stable_latents_stats/mean.npy')
        latent_std = np.load('/data/leike/spatial/YT-Ambigen/stable_latents_stats/std.npy')
        latent_mean = torch.tensor(latent_mean, device=self.device, dtype=generated_lat.dtype)
        latent_std = torch.tensor(latent_std, device=self.device, dtype=generated_lat.dtype)

        ### slice
        x = generated_lat
        x = x[:, :self.metadata['target_lat_len'], :]
        
        ###
        x = x * latent_std + latent_mean
        x = x.permute(0, 2, 1)
        WX = x[:, :64, :]
        YZ = x[:, 64:, :]
        rec_WX = self.sa_vae.decode(WX)
        rec_YZ = self.sa_vae.decode(YZ)
        rec_audio = torch.cat([rec_WX, rec_YZ], dim=0)
        
        rec_audio = rec_audio.detach().to('cpu') 
        rec_audio = rec_audio.reshape(-1, rec_audio.shape[-1])
        
        
        return {'generated_lat': generated_lat, 'rec_audio': rec_audio}


        
if __name__ == '__main__':
    
    from datetime import datetime
    import torchaudio
    infer = DiTARSInfer(torch.device('cuda'), '/home/leike/spatial/ScriptSpeech/checkpoints/ditar_test')

    infer_lst = [
        'BI_heWaNfro_8', 'VX_gOGFgt14_46', 'dKye1dZuECk_89', 'bhAhh3dSzHI_85', # train里的
        'ECFTh6UdONY_158', 'gSRVPLekBwY_14', 'xZ3KECAMpwo_73', 'kha7D_Nt3QA_15', # test里的
        '0A4GRMrLpWI_259', '0BDCLo2pioo_43', '0BDCLo2pioo_130', '0DhWUtlWcA0_13', '03XzVqjmECw_63', '06av4szCH1s_157',   # train里的
        '0D4rxdOI5TM_13', # valid
        '0FB9jMXMP8A_31', 
        '0FB9jMXMP8A_43',  # test
        'gSRVPLekBwY_21', '_D7CJg5fvsE_99', 'IQifpz8nZDA_91', '0hCGacvtyNQ_26', 
        'MVPCbI71shM_19', 'tENB2euDcB4_227', '5rrCEo7Rwv8_88', '4EgRaySMjJQ_39',
        'sLSh-etPd4o_138', 'HK-eDj5gdPk_89', '85YGH9MdjLo_52', 'nNRoC0xn1Aw_25' # test
    ]
    

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    SAVE_DIR = os.path.join('results', timestamp)
    os.makedirs(SAVE_DIR, exist_ok=True)
    for item in infer_lst:
        print(f"infer {item}...")
        ret  = infer.forward(item + '.npy')
        
        rec_audio = ret['rec_audio']
        
        torchaudio.save(f'{SAVE_DIR}/{item}_rec.wav', rec_audio, 44100)
        os.system(f"cp /data/leike/spatial/YT-Ambigen/audio_10/{item}.opus {SAVE_DIR}")
        os.system(f"cp /data/leike/spatial/YT-Ambigen/erp_10/{item}.mp4 {SAVE_DIR}")
        
    import pdb; pdb.set_trace()