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
import whisper
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import subprocess
from pathlib import Path
import time
import math

from copy import deepcopy
from langdetect import detect as classify_language, LangDetectException
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
from modules.tts.ditar.build_model_utils import DiTARBuildModelMixinV2, DiTARBuildModelMixinV3
from tasks.tts.dataset_utils.promptaudio_fastdataset_v2 import build_spk_mask_from_text_tokens, _get_sx_token_patterns, augment_text_with_pinyin_s1s2_safe

DEBUG = False

if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

def convert_to_wav(wav_path):
    # Check if the file exists
    if not os.path.exists(wav_path):
        print(f"The file '{wav_path}' does not exist.")
        return

    # Check if the file already has a .wav extension
    if not wav_path.endswith(".wav"):
        # Define the output path with a .wav extension
        out_path = os.path.splitext(wav_path)[0] + ".wav"

        # Load the audio file using pydub and convert it to WAV
        audio = AudioSegment.from_file(wav_path)
        audio.export(out_path, format="wav")

        print(f"Converted '{wav_path}' to '{out_path}'")


def convert_to_wav_bytes(audio_binary):
    # Load the audio binary using pydub and convert it to WAV
    audio = AudioSegment.from_file(io.BytesIO(audio_binary))
    wav_bytes = io.BytesIO()
    audio.export(wav_bytes, format="wav")
    wav_bytes.seek(0)
    return wav_bytes

@contextmanager
def model_lock(lock):
    try:
        lock.acquire()
        yield
    finally:
        torch.cuda.synchronize()
        lock.release()
        
        
def move_to_shm(exp_name):
    os.makedirs('/dev/shm/mega_ckpt', exist_ok=True)
    shm_exp_name = f"/dev/shm/mega_ckpt/{Path(exp_name).stem}"
    if os.path.exists(shm_exp_name):
        return shm_exp_name
    subprocess.check_call(f"cp -r {exp_name} /dev/shm/mega_ckpt/", shell=True)
    return shm_exp_name


@dataclass
class MegaTTS3Output:
    wav_bytes: bytes = None
    wav: np.ndarray = None
    duration: float = None
    stop_probs: list = None


class DiTARV2Infer(DiTARBuildModelMixinV2):
    def __init__(
            self, 
            device=None,
            ditar_exp_name='checkpoints/251222_ditar_v2',
            wavvae_exp_name='checkpoints/251120_wavvae_v4_unfreeze',
            max_ref_duration=20,
            use_tqdm=True,
            use_ema=False,
            precision='bf16',
            **kwargs
    ):

        if precision == 'fp16':
            self.precision = torch.float16
        elif precision == 'bf16':
            self.precision = torch.bfloat16

        self.sr = 24000
        self.fm = 8
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.use_tqdm = use_tqdm
        self.use_ema = use_ema

        self.ditar_exp_name = ditar_exp_name
        self.wavvae_exp_name = wavvae_exp_name

        self.build_model(self.device)

        # init text normalizer
        # self.zh_normalizer = ZhNormalizer(overwrite_cache=True, remove_erhua=False, remove_interjections=False, full_to_half=False)
        # self.en_normalizer = EnNormalizer(overwrite_cache=True)
        self.ph_replace_table = {}

        # break (silence)
        self.max_silence_alive = 1.28    # 1.28s
        self.max_ref_duration = max_ref_duration
        self.chunk_num_words_zh = 60
        self.chunk_num_words_en = 130

    def build_model(self, device):
        self.device = device

        if self.ditar_exp_name.endswith('.ckpt'):
            set_hparams(f'{Path(self.ditar_exp_name).parent}/config.yaml', print_hparams=False)
        else:
            set_hparams(f'{self.ditar_exp_name}/config.yaml', print_hparams=False)
        hparams['use_fsdp'] = False

        ''' Load DiT '''
        self._build_model()
        if self.use_ema and hparams.get('use_ema', False):
            load_ckpt(self.ditar, self.ditar_exp_name, 'ema_model', strict=False, mmap=True)
        else:
            load_ckpt(self.ditar, self.ditar_exp_name, 'model', strict=False, mmap=True)
        self.vae.eval()
        self.vae.to(self.device, dtype=self.precision)
        self.ditar.eval()
        self.ditar.to(device, dtype=self.precision)
        
        ''' ASR '''
        from modules.asr.sensevoice.sensevoice_api import build_asr_model
        self.asr_model = build_asr_model(device)

        ''' VAD '''
        from silero_vad import load_silero_vad
        self.vad_model = load_silero_vad()

        self.vae = torch.compile(self.vae, mode='max-autotune')
        self.ditar = torch.compile(self.ditar, mode='max-autotune')
        self.lock = threading.Lock()

    def trim_silence_vad(self, wav_16k=None, speech_timestamps=None, max_duration=60, max_silence=1.28):
        if speech_timestamps is None and wav_16k is not None:
            from silero_vad import load_silero_vad, get_speech_timestamps
            wav_16k = wav_16k[:int(16000 * max_duration * 1.2)]
            speech_timestamps = get_speech_timestamps(
                wav_16k,
                self.vad_model,
                return_seconds=True,  # Return speech timestamps in seconds (default is samples)
            )
        from utils.audio.diarization_utils import merge_segments
        segments = merge_segments(speech_timestamps, max_silence=max_silence, max_voiced_duration=max_duration)
        cur_duration = 0
        for i in range(len(segments)):
            if (cur_duration + segments[i]['end'] - segments[i]['start'] > max_duration) and cur_duration > 0:
                return segments[:i]
            cur_duration += segments[i]['end'] - segments[i]['start']
        return segments

    def preprocess(self, audio_bytes, wav_path=None, topk_dur=1, **kwargs):
        wav_bytes = convert_to_wav_bytes(audio_bytes)

        device = self.device

        wav, _ = librosa.load(wav_bytes, sr=self.sr)
        
        wav_16k = librosa.resample(wav, orig_sr=self.sr, target_sr=16000)
        max_ref_duration = self.max_ref_duration

        segments = self.trim_silence_vad(wav_16k, max_duration=max_ref_duration)
        wav = np.concatenate([wav[int(segment['start'] * self.sr): int(segment['end'] * self.sr)] for segment in segments])
        wav_16k = np.concatenate([wav_16k[int(segment['start'] * 16000): int(segment['end'] * 16000)] for segment in segments])
        # remove gap leak
        wav = np.concatenate([wav, np.zeros(int(self.sr * 0.2))])
        wav_16k = np.concatenate([wav_16k, np.zeros(int(16000 * 0.2))])

        fm_wav = hparams['frames_multiple'] * hparams['hop_size'] * hparams['ditar_patch_size']
        fm_wav_16k = fm_wav * 16000 // hparams['audio_sample_rate']
        wav = pad_or_cut_xd(torch.from_numpy(wav), math.ceil(len(wav) / fm_wav) * fm_wav, dim=-1).numpy()
        wav_16k = pad_or_cut_xd(torch.from_numpy(wav_16k), math.ceil(len(wav_16k) / fm_wav_16k) * fm_wav_16k, dim=-1).numpy()

        with torch.inference_mode():
            ''' Forward WavVAE to obtain: prompt latent '''
            wav = torch.tensor(wav, dtype=self.precision)[None].to(device)
            with torch.autocast(device_type='cuda', dtype=self.precision):
                vae_latent = self.vae.encode_latent(wav)

            ''' ASR '''
            from modules.asr.sensevoice.sensevoice_api import run_asr_model
            text = run_asr_model([wav_16k], self.asr_model, with_segments=False)[0]['text_normed']

        ret = {
            'text_ref': text,
            'vae_latent': vae_latent.cpu(),
        }

        return ret
    
    def forward(
            self, resource_context, input_text, time_step, seq_cfg_w=(1.4, 3), 
            speech_rate=1, timestep_annealing_w=(1.0, 0.0, 1.0), use_amo_sampler=False, 
            return_format='wav', custom_ph_table=None, num_parallel_workers=5, 
            chunk_num_words_zh=60, chunk_num_words_en=130, **kwargs
    ):
        
        device = self.device

        profile = os.environ.get('MEGA_PROFILE', 'false').strip().lower() == 'true'

        with torch.inference_mode():
            ''' Generating '''
            text = input_text

            text_ref = resource_context['text_ref']
            vae_latent = resource_context['vae_latent'].detach().clone().to(device)

            txt_tokens = self.ditar_text_tokenizer(text_ref + text, padding=True, return_tensors='pt')
            txt_mask = txt_tokens['attention_mask'].bool().to(self.device)
            txt_tokens = txt_tokens['input_ids'].to(self.device)

            with Timer('DiT target speech generation', enable=profile):
                
                inputs = {
                    "ref_lat": vae_latent,
                    "ref_lat_lens": torch.LongTensor([vae_latent.size(1)]).to(self.device),
                    "txt_tokens": txt_tokens,
                    'txt_mask': txt_mask,
                    'txt_lens': txt_mask.sum(1),
                }

                with model_lock(self.lock), torch.autocast(device_type='cuda', dtype=self.precision):
                    if self.__class__.__name__ == 'DiTARV2Infer':
                        x, stop_probs = self.ditar.inference(
                            inputs, timesteps=time_step, 
                            seq_cfg_w=seq_cfg_w, 
                            timestep_annealing_w=timestep_annealing_w,
                            start_pos=0, use_tqdm=True, max_new_patches=256
                        )
                    elif self.__class__.__name__ == 'DiTARV3Infer':
                        x, stop_probs = self.ditar.inference(
                            inputs, timesteps=time_step, 
                            seq_cfg_w=seq_cfg_w, 
                            timestep_annealing_w=timestep_annealing_w,
                            past_key_values=None, use_tqdm=True, max_new_patches=256
                        )

            # WavVAE decode
            with Timer('WavVAE decode', enable=profile):
                x = torch.cat([vae_latent, x], dim=1)
                with model_lock(self.lock), torch.autocast(device_type='cuda', dtype=self.precision):
                    wav_pred = self.vae.decode(x)[0,0].to(torch.float32)

            with Timer('Post-processing', enable=profile):
                hop_size = self.hp_vae['hop_size']
                vae_stride = self.hp_vae['vae_stride']
                wav_pred = wav_pred[vae_latent.size(1)*vae_stride*hop_size:]
                if wav_pred.abs().max() > 1:
                    print('Wav amplitude exceed 1, clip it.')
                    wav_pred = wav_pred / (wav_pred.abs().max())

                wav_pred = wav_pred.cpu().numpy()
                print(f"{wav_pred.shape = } {text = }")

            wav_bytes = to_wav_bytes(wav_pred.astype(float), self.sr)
            if return_format == 'mp3':
                wav_bytes = wav_bytes_to_mp3_bytes(wav_bytes)

            output = MegaTTS3Output(
                wav_bytes=wav_bytes,
                duration=wav_pred.shape[-1] / self.sr,
                stop_probs=stop_probs
            )

            return output


class DiTARV3Infer(DiTARV2Infer):
    def __init__(self, **kwargs):
        self.build_ditar = DiTARBuildModelMixinV3.build_ditar.__get__(self)
        super().__init__(**kwargs)



if __name__ == '__main__':
    if os.path.isfile('.env.local'):
        from dotenv import load_dotenv
        load_dotenv('.env.local')

    # ditar_exp_name = 'checkpoints/251222_ditar_v2'
    ditar_exp_name = 'checkpoints/251224_ditar_v3'

    cls_kwargs = dict(
        ditar_exp_name=ditar_exp_name,
        max_ref_duration=10,
        use_ema=False,
    )

    infer_kwargs = dict(
        time_step=20, 
        seq_cfg_w=(1.5, 3.0), 
        timestep_annealing_w=(0.6, 0.6, 1.0)
    )

    if 'ditar_v2' in ditar_exp_name:
        infer_ins = DiTARV2Infer(device='cuda', **cls_kwargs)
    elif 'ditar_v3' in ditar_exp_name:
        infer_ins = DiTARV3Infer(device='cuda', **cls_kwargs)

    wav_path = 'user/prompts/mega_eval_prompt0731/0_vocal.wav'
    resource_context = infer_ins.preprocess(open(wav_path, 'rb').read())
    
    test_text = [
        # '家人们，我必须给大家分享一下这款宠物鱼油胶囊，如果你家的猫狗掉毛严重，到处都是毛，那你一定得试试这个。'
        # '嗨，家人们，我必须给大家分享一下这款宠物鱼油胶囊，如果你家的猫狗掉毛严重，到处都是毛，那你一定得试试这个。'
        '嗨，家人们，我必须给大家分享一下这款宠物鱼油胶囊。如果你家的猫狗掉毛严重，到处都是毛，那你一定得试试这个。它富含高浓度的 omega 三和各种维生素，口感很好，可以直接喂食，也可以拌在猫粮或喜欢的零食里。这是专为猫狗设计的，每一瓶都有八十粒，只需要一杯奶茶的钱。现在厂家直接发货，性价比超高，赶紧下单吧！'
    ]

    out_path = f'infer_out/tts/infer_once/{Path(ditar_exp_name).stem}/step{get_all_ckpt_steps(ditar_exp_name)[-1]}'
    
    print(f"cls_kwargs = {json_dumps(cls_kwargs)}")
    print(f"infer_kwargs = {json_dumps(infer_kwargs)}")
    
    kill_void()
    pbar = tqdm(total=len(test_text))
    total_time_elapsed = 0
    total_duration_processed = 0
    for i, input_text in enumerate(test_text):
        print('='*100)
        print(f'| Generating text [{i}]: {input_text}')
        start_time = time.time()
        output = infer_ins.forward(
            resource_context, input_text, **infer_kwargs
        )
        total_time_elapsed += time.time() - start_time

        wav_bytes = output.wav_bytes
        gen_duration = output.duration
        stop_probs = output.stop_probs
        total_duration_processed += gen_duration

        os.makedirs(out_path, exist_ok=True)
        save_wav_bytes(wav_bytes, f'{out_path}/{input_text[:20]}.wav')
        
        import matplotlib.pyplot as plt
        plt.plot(stop_probs)
        plt.savefig(f'{out_path}/{input_text[:20]}.png')

        print(f'| Generation of text [{i}] succeed')
        pbar.update(1)
        
    print(f"{total_time_elapsed = }, {total_duration_processed = }, RTF = {total_time_elapsed / total_duration_processed}")

    # CUDA_VISIBLE_DEVICES=0 python inference/tts/ditar_v2_infer.py
