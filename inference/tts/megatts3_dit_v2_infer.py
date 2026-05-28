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

from copy import deepcopy
from langdetect import detect as classify_language, LangDetectException
from pydub import AudioSegment
import pyloudnorm as pyln
from tqdm import tqdm

from utils.audio.align import mel2token_to_dur
from utils.audio.io import save_wav_bytes, to_wav_bytes, wav_bytes_to_mp3_bytes
from utils.text import is_english, YUNMU_ERHUA, SHENGMU
from utils.text.text_encoder import TokenTextEncoder
from utils.text.split_text import chunk_text_english, chunk_text_chinese, get_word_list, remove_space, remove_unprintable
from utils.text.ph_tone_convert import split_ph_timestamp, split_ph, map_phone_to_tokendict
from utils.text.ssml_utils import SSML
from utils.text.ph_alignment import align_word_phone, print_align, merge_norm_alignment
from utils.text.cosyvoice2_tokenizer import CosyVoice2Tokenizer
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import set_hparams, hparams
from utils.commons.meters import Timer
from utils.commons.os_utils import handle_exacption
from utils.commons.io import print_once
from utils.commons.import_utils import import_module_bystr, get_class_from_module
from utils.commons.tensor_utils import move_to_cpu, move_to_cuda

from modules.tts.ar_dur.commons.nar_tts_modules import LengthRegulator
from modules.tts.ar_dur.commons.align_ops import compute_mel2aug_from_dur
from modules.tts.scriptspeech.build_model_utils import DiTBuildModelMixinV2, DiTBuildModelMixinV4, build_qwen3, DiTBuildModelMixinV6, DiTBuildModelMixinV7

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
    words_timestamps: Dict[str, List] = None
    words_timestamps_post: Dict[str, List] = None
    duration: float = None
    ph_pred: List[str] = None
    tone_pred: List[str] = None


class MegaTTS3DiTInfer(DiTBuildModelMixinV2):
    ''' 这里指定所用ckpt的路径 '''
    def __init__(
            self, 
            device=None,
            dit_exp_name='checkpoints/250818_megatts3_dit_v2',
            dur_exp_name='checkpoints/250826_dur_lm',
            frontend_exp_name='checkpoints/250820_lm_mfa_base_wavlmlarge',
            wavvae_exp_name='checkpoints/1231_megatts3_wavvae_v3_25hz_kl001_fix4',
            use_old_aligner=False,
            use_old_dur=False,
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

        self.dit_exp_name = dit_exp_name
        self.use_old_dur = use_old_dur
        if use_old_dur:
            self.dur_exp_name = 'checkpoints/megatts3_wavdit/duration_lm'
        else:
            self.dur_exp_name = dur_exp_name
        self.use_old_aligner = use_old_aligner
        if use_old_aligner:
            self.frontend_exp_name = 'checkpoints/megatts3_wavdit/aligner_lm'
        else:
            self.frontend_exp_name = frontend_exp_name
        self.wavvae_exp_name = wavvae_exp_name

        self.build_model(self.device)

        # break (silence)
        self.max_silence_alive = 1.28    # 1.28s
        self.max_ref_duration = max_ref_duration
        self.chunk_num_words_zh = 60
        self.chunk_num_words_en = 130

    def build_dur_model(self):
        self.length_regulator = LengthRegulator()
        if self.use_old_dur:
            from modules.tts.ar_dur.ar_dur_predictor import ARDurPredictor
            hp_dur_model = self.hp_dur_model = set_hparams(f'{self.dur_exp_name}/config.yaml', global_hparams=False)
            hp_dur_model['frames_multiple'] = hparams['frames_multiple']
            self.dur_model = ARDurPredictor(
                hp_dur_model, hp_dur_model['dur_txt_hs'], hp_dur_model['dur_model_hidden_size'],
                hp_dur_model['dur_model_layers'], len(self.token_encoder),
                hp_dur_model['dur_code_size'],
                use_rot_embed=hp_dur_model.get('use_rot_embed', False), 
                precision=self.precision
                )
            load_ckpt(self.dur_model, f'{self.dur_exp_name}', 'dur_model')
            self.dur_model.eval()
            self.dur_model.to(self.device, dtype=self.precision)
        else:
            if 'lm' in (dur_exp_name_eles := [n.lower() for n in self.dur_exp_name.split('_')]):
                if 'seq2seq' in dur_exp_name_eles:
                    self.dur_model_type = 'lm_seq2seq'
                else:
                    self.dur_model_type = 'lm'
            elif 'dit' in dur_exp_name_eles:
                self.dur_model_type = 'dit'
            if self.dur_model_type in ['lm', 'lm_seq2seq']:
                from modules.tts.ar_dur.dur_lm import build_dur_model
                hp_dur_model = self.hp_dur_model = set_hparams(f'{self.dur_exp_name}/config.yaml', global_hparams=False)
                self.dur_model = build_dur_model(hp_dur_model, vocab_size=810, padding_idx=797)
                self.dur_model.hparams = {}
                self.dur_model.eval()
                load_ckpt(self.dur_model, self.dur_exp_name, 'model', mmap=True)
                self.dur_model.to(self.device)
                if self.dur_model_type == 'lm_seq2seq':
                    if hp_dur_model['cond_encoder_name'] != hparams['caption_encoder_name']:
                        self.caption_encoder_dur_model, self.caption_tokenizer_dur_model = build_qwen3(hparams, hp_dur_model['cond_encoder_name'])
                        self.caption_encoder_dur_model = self.caption_encoder_dur_model.model
                        self.caption_encoder_dur_model.requires_grad_(False).eval()
                        self.caption_encoder_dur_model.to(self.device, dtype=self.precision)
            elif self.dur_model_type == 'dit':
                from modules.tts.scriptspeech.dit_dur import build_dur_model
                hp_dur_model = self.hp_dur_model = set_hparams(f'{self.dur_exp_name}/config.yaml', global_hparams=False)
                self.dur_model = build_dur_model(hp_dur_model)
                self.dur_model.hparams = {}
                self.dur_model.eval()
                load_ckpt(self.dur_model, self.dur_exp_name, 'model', strict=True, mmap=True)
                self.dur_model.to(self.device)

    def build_frontend_model(self):
        if self.use_old_aligner:
            from modules.tts.frontend_lm.whisper.whisper_small import Whisper
            self.aligner_lm = Whisper()
            load_ckpt(self.aligner_lm, f'{self.frontend_exp_name}', 'model')
            self.aligner_lm.eval()
            self.aligner_lm.to(self.device, dtype=self.precision)
            self.kv_cache = None
            self.hooks = None
        else:
            from modules.asr.scriptasr.build_model_utils import build_asr_model
            if self.frontend_exp_name.endswith('ckpt'):
                aligner_hparams = set_hparams(f'{Path(self.frontend_exp_name).parent}/config.yaml', global_hparams=False)
            else:
                aligner_hparams = set_hparams(f'{self.frontend_exp_name}/config.yaml', global_hparams=False)
            self.aligner_lm = build_asr_model(aligner_hparams, init_pretrained=False, vocab_size=6800, padding_idx=797)
            self.aligner_lm.eval()
            load_ckpt(self.aligner_lm, self.frontend_exp_name, 'model', strict=True, mmap=True)
            self.aligner_lm.to(self.device)

    ''' 加载模型参数 '''
    def build_model(self, device):
        self.device = device

        if self.dit_exp_name.endswith('.ckpt'):
            set_hparams(f'{Path(self.dit_exp_name).parent}/config.yaml', print_hparams=False)
        else:
            set_hparams(f'{self.dit_exp_name}/config.yaml', print_hparams=False)
        hparams['use_fsdp'] = False

        ''' Load Dict '''
        ling_dict = json.load(open('egs/tts/megatts3_dict.json'))
        self.ling_dict = {k: TokenTextEncoder(None, vocab_list=ling_dict[k], replace_oov='<UNK>') for k in ['phone', 'tone']}
        self.token_encoder = token_encoder = self.ling_dict['phone']
        # self.ph_replace_table = json.load(open(f"{current_dir}/ph_replace_table.json"))
        self.ph_replace_table = {'en': {}, 'zh': {}}
        self.sil_ph = self.ling_dict['phone'].sil_phonemes()

        ''' Load Duration LM '''
        self.build_dur_model()

        ''' Load DiT '''
        self._build_model()
        if self.use_ema and hparams.get('use_ema', False):
            load_ckpt(self.dit, f'{self.dit_exp_name}', 'ema_model', strict=False, mmap=True)
        else:
            load_ckpt(self.dit, f'{self.dit_exp_name}', 'dit', strict=False, mmap=True)
        self.vae.eval()
        self.vae.to(self.device, dtype=self.precision)
        self.dit.eval()
        self.dit.to(device, dtype=self.precision)
        self.cfg_mask_token_phone = 302 - 1
        self.cfg_mask_token_tone = 32 - 1
        
        if not hasattr(self, 'caption_encoder'):
            caption_encoder_name = hparams.get('caption_encoder_name')
            self.caption_encoder, self.caption_tokenizer = build_qwen3(hparams, caption_encoder_name)
            self.caption_encoder = self.caption_encoder.model
            self.caption_encoder.requires_grad_(False).eval()
        self.caption_encoder.to(device, dtype=self.precision)

        ''' Load Frontend LM '''
        self.build_frontend_model()

        ''' ASR '''
        from modules.asr.sensevoice.sensevoice_api import build_asr_model
        self.asr_model = build_asr_model(device)

        ''' VAD '''
        from silero_vad import load_silero_vad
        self.vad_model = load_silero_vad()

        self.vae = torch.compile(self.vae, mode='max-autotune')
        self.aligner_lm = torch.compile(self.aligner_lm, mode='max-autotune')
        self.dit = torch.compile(self.dit, mode='max-autotune')
        self.dur_model = torch.compile(self.dur_model, mode='max-autotune')
        self.lock = threading.Lock()


    ''' 文本预处理 '''
    def preprocess_text(self, input_text: SSML, ph_replace_table=None, use_sa_frontend=False, chunk_num_words_zh=60, chunk_num_words_en=130):

        def _normalize_text_en(text: str):
            text_norm = common_preprocess(text)
            if not use_sa_frontend:
                text_norm = self.en_normalizer.normalize(text_norm)
            if ph_replace_table is not None:
                for src, tgt in ph_replace_table['en'].items():
                    text_norm = text_norm.replace(src, tgt)
            text_norm = common_process(text_norm)
            # text_norm = common_postprocess(text_norm)
            return text_norm
            
        def _normalize_text_zh(text):
            text_norm = common_preprocess(text)

            if not use_sa_frontend:
                from opencc import OpenCC
                jp2t_converter = OpenCC('jp2t')
                t2s_converter = OpenCC('t2s')
                text_norm = t2s_converter.convert(jp2t_converter.convert(text_norm))
                
                text_norm = self.zh_normalizer.normalize(text_norm)

            if ph_replace_table is not None:
                for src, tgt in ph_replace_table['zh'].items():
                    text_norm = text_norm.replace(src, tgt)
            text_norm = common_process(text_norm)
            # text_norm = common_postprocess(text_norm)
            return text_norm

        def common_process(text: str):
            text_norm = text
            if not use_sa_frontend:
                # 处理特殊符号
                pause_punc = [
                    '~', '～', ':', '$', '¥', '&', '#', '@', '^', '・', '·', '‘', '’', '“', '”', "'", "'", '"', '"',
                    '（', '）', '(', ')', '【', '】', '{', '}', '「', '」', '[', ']', '<', '>', '《', '》',
                    '%', '*', '|', '｜', '\\', '/', '-', '+', '_', '=',
                    '²',
                ]
                text_norm = batch_replace(text_norm, pause_punc, tgt='')
            return text_norm
        
        def common_preprocess(text: str):
            special_symbols = [
                '&#34;'
            ]
            if use_sa_frontend:
                special_symbols.extend(['"'])   # 删除引号，否则json格式问题
            text_norm = batch_replace(text, special_symbols, tgt='')
            text_norm = batch_replace(text_norm, ['\n'], tgt=' ')
            return text_norm
        
        def common_postprocess(text: str):
            import re
            max_no_punct = 77
            errors = []
            no_punct_segments = re.split(r'[，。、；：！？,.;:!?]', text)
            for seg in no_punct_segments:
                len_seg = len(get_word_list(seg))
                if len_seg > max_no_punct:
                    errors.append(f"无效文本：存在{len_seg}个连续字符无标点（允许上限{max_no_punct}字）")
                    break
            if len(errors) > 0:
                if len(errors) > 1:
                    msg = '; '.join([f'{i+1}. {errors[i]}' for i in range(len(errors))])
                else:
                    msg = errors[0]
                raise RuntimeError(msg)
            return text
        
        def batch_replace(text: str, src: Union[str, List], tgt: str = ','):
            for p in src:
                text = text.replace(p, tgt)
            return text
        
        input_text.apply_sub()

        try:
            language_type = classify_language(input_text.text_str)
        except LangDetectException as err:
            handle_exacption(err, '无法检测语言，默认选择中文')
            language_type = 'zh'
        if language_type == 'en':
            input_text.normalize(_normalize_text_en)
            text_segs = SSML.chunk_text_with_breaks(input_text, limit=chunk_num_words_en, language_type='en', debug=False)
        else:
            # print('input_text', input_text)
            input_text.normalize(_normalize_text_zh)
            # print('input_text', input_text)
            text_segs = SSML.chunk_text_with_breaks(input_text, limit=chunk_num_words_zh, language_type='zh', debug=False)
            # print('text_segs', text_segs)

        return text_segs
    
    ''' 根据SSML修改发音 '''
    def refine_ph_tone(self, text: SSML, ph_pred: torch.Tensor, tone_pred: torch.Tensor):
        ph_tokens = ph_pred.squeeze().cpu().numpy()
        tone_tokens = tone_pred.squeeze().cpu().numpy()
        ph_tokens = self.ling_dict['phone'].decode(ph_tokens).split(' ')
        tone_tokens = self.ling_dict['tone'].decode(tone_tokens).split(' ')

        # FIXME：这可能导致表现力下降，但如果不这么做，ph2word可能将不再是单向的
        ph_tokens_ = []
        tone_tokens_ = []
        for p_i, p in enumerate(ph_tokens):
            # 额外考虑“这儿(zh er -> zh e er)”这种情况
            if (p_i > 0 and p == "C0er" and ph_tokens[p_i - 1] in SHENGMU) or (p in YUNMU_ERHUA):
                ph_tokens_.append(p[:-1])
                ph_tokens_.append("C0er")
                tone_tokens_.append(tone_tokens[p_i])
                tone_tokens_.append('5')    # 轻声？
            else:
                ph_tokens_.append(p)
                tone_tokens_.append(tone_tokens[p_i])
        ph_tokens = ph_tokens_
        tone_tokens = tone_tokens_

        text_, ph_tokens, ph2word = align_word_phone(text.text_str, ph_tokens)
        # print_align(text_, ph_tokens, ph2word)
        ph2word = [p-1 for p in ph2word]    # ignore "sil" at the begining

        ph_tokens, tone_tokens, ph2word = SSML.replace_ph_tone(text, ph_tokens, tone_tokens, ph2word)

        ph_tokens = self.ling_dict['phone'].encode(' '.join(ph_tokens))
        ph_pred = torch.LongTensor(ph_tokens)[None].to(ph_pred)
        tone_tokens = self.ling_dict['tone'].encode(' '.join(tone_tokens))
        tone_pred = torch.LongTensor(tone_tokens)[None].to(tone_pred)

        return ph_pred, tone_pred, ph2word
    
    ''' 根据SSML增加停顿 '''
    def add_breaks(self, text: SSML, ph_pred: torch.Tensor, tone_pred: torch.Tensor, dur_pred: torch.Tensor, 
                   ph2word: List, break_token=145, break_tone=3):
        ph_tokens = ph_pred.squeeze().cpu().numpy().tolist()
        tone_tokens = tone_pred.squeeze().cpu().numpy().tolist()
        dur_tokens = dur_pred.squeeze().cpu().numpy().tolist()
        ph_tokens, tone_tokens, ph2word, dur_tokens = SSML.add_breaks(
            text, ph_tokens, tone_tokens, ph2word, dur_tokens, break_token, break_tone, 0.01
        )
        ph_pred = torch.Tensor(ph_tokens)[None].to(ph_pred)
        tone_pred = torch.Tensor(tone_tokens)[None].to(tone_pred)
        dur_pred = torch.Tensor(dur_tokens)[None].to(dur_pred)

        return ph_pred, tone_pred, dur_pred, ph2word
    
    ''' 处理文字时间戳 '''
    def make_word_timestamps(self, text: SSML, dur_pred: np.ndarray, ph2word: List):
        dur_timestep = 0.01
        offsets = [0] + np.cumsum(dur_pred).tolist()
        words_to_get = get_word_list(text.text_str)
        ph2word = ph2word + [-3]
        words = []
        timestamps = []
        ph_start_idx = ph_end_idx = 0
        for ph_end_idx in range(1, len(ph2word)):
            if ph2word[ph_end_idx] != ph2word[ph_start_idx]:
                if ph2word[ph_start_idx] >= 0:
                    words.append(words_to_get[ph2word[ph_start_idx]])
                    timestamps.append(
                        [offsets[ph_start_idx] * dur_timestep, offsets[ph_end_idx] * dur_timestep]
                    )
                ph_start_idx = ph_end_idx


        text_merged, text_norm_merged, text_idx_merged, text_norm_idx_merged = merge_norm_alignment(
            text.origin.text_str, words, debug=False
        )

        words_merged = []
        timestamps_merged = []
        word_idx = 0
        for merge_idx in range(len(text_merged)):
            if isinstance(text_merged[merge_idx], list):
                word_merged = []
                timestamp_merged = []
                for i in range(len(text_merged[merge_idx])):
                    if len(word_merged) > 0 and is_english(word_merged[-1]) and is_english(text_merged[merge_idx][i]):
                        word_merged.append(' ')
                    word_merged.append(text_merged[merge_idx][i])
                for i in range(len(text_norm_merged[merge_idx])):
                    timestamp_merged.append(timestamps[word_idx])
                    word_idx += 1
                words_merged.append(''.join(word_merged))
                if len(timestamp_merged) > 0:
                    timestamps_merged.append([timestamp_merged[0][0], timestamp_merged[-1][-1]])
                else:
                    # 此时，raw text有符号，但norm后该符号被删除。该符号的时长暂定为0
                    if len(timestamps_merged) <= 0:
                        timestamps_merged.append([0.0, 0.0])
                    else:
                        timestamps_merged.append([timestamps_merged[-1][-1], timestamps_merged[-1][-1]])
            else:
                words_merged.append(text_merged[merge_idx])
                timestamps_merged.append(timestamps[word_idx])
                word_idx += 1

        return {
            'words': words_merged,
            'timestamps': timestamps_merged
        }
    
    ''' 拼接音频段，使用 crossfade 实现平滑过渡。 '''
    def combine_audio_segments(self, segments, words_timestamps=(), sil_pad_lst=(), crossfade_duration=0.32):
        window_length = int(self.sr * crossfade_duration)
        hanning_window = np.hanning(2 * window_length)
        return_timestamps = len(words_timestamps) > 0
        combined_words_timestamps = {'words': [], 'timestamps': []}
        # Combine
        for i, segment in enumerate(segments):
            if i == 0:
                combined_audio = segment
                if return_timestamps:
                    combined_words_timestamps['words'] = words_timestamps[i]['words']
                    combined_words_timestamps['timestamps'] = words_timestamps[i]['timestamps']
                sil_pad_start, sil_pad_end = sil_pad_lst[i]
                if sil_pad_start > 0:
                    combined_audio = np.concatenate([np.zeros((int(sil_pad_start * self.sr))), combined_audio])
                    combined_words_timestamps['timestamps'] = [[s[0] + sil_pad_start, s[1] + sil_pad_start] for s in combined_words_timestamps['timestamps']]
                if sil_pad_end > 0:
                    combined_audio = np.concatenate([combined_audio, np.zeros((int(sil_pad_end * self.sr)))])
            else:
                sil_pad_start, sil_pad_end = sil_pad_lst[i]
                if sil_pad_start > 0:
                    segment = np.concatenate([np.zeros((int(sil_pad_start * self.sr))), segment])
                if sil_pad_end > 0:
                    segment = np.concatenate([segment, np.zeros((int(sil_pad_end * self.sr)))])
                overlap = combined_audio[-window_length:] * hanning_window[window_length:] + segment[:window_length] * hanning_window[:window_length]
                offset = combined_audio[:-window_length].shape[0] + sil_pad_start * self.sr
                combined_audio = np.concatenate(
                    [combined_audio[:-window_length], overlap, segment[window_length:]]
                )
                if return_timestamps:
                    combined_words_timestamps['words'] = combined_words_timestamps['words'] + words_timestamps[i]['words']
                    timestamps = words_timestamps[i]['timestamps']
                    offset = offset / self.sr
                    timestamps = [[s[0] + offset, s[1] + offset] for s in timestamps]
                    combined_words_timestamps['timestamps'] = combined_words_timestamps['timestamps'] + timestamps
        return combined_audio, combined_words_timestamps

    def chunk_wavs_vad(self, wav_16k=None, speech_timestamps=None, chunk_duration=10, max_duration=60):
        if speech_timestamps is None and wav_16k is not None:
            from silero_vad import load_silero_vad, get_speech_timestamps
            wav_16k = wav_16k[:int(16000 * max_duration * 1.2)]
            speech_timestamps = get_speech_timestamps(
                wav_16k,
                self.vad_model,
                return_seconds=True,  # Return speech timestamps in seconds (default is samples)
            )
        timestamp_end_idx = -1   # include
        chunk_end_offsets = []
        last_end = 0
        for timestamp_idx, timestamp in enumerate(speech_timestamps):
            if timestamp['end'] > max_duration:
                timestamp_end_idx = timestamp_idx - 1
                break
            if timestamp['end'] - last_end > chunk_duration:
                chunk_end_offsets.append(timestamp['end'])
                last_end = timestamp['end']
        else:
            timestamp_end_idx = len(speech_timestamps) - 1
        if timestamp_end_idx == -1:
            chunk_end_offsets.append(max_duration)
        else:
            if len(chunk_end_offsets) > 0 and chunk_end_offsets[-1] != speech_timestamps[timestamp_end_idx]['end']:
                chunk_end_offsets.append(speech_timestamps[timestamp_end_idx]['end'])

        chunk_offsets = [speech_timestamps[0]['start']] + chunk_end_offsets
        chunk_offsets = [(chunk_offsets[i], chunk_offsets[i+1]) for i in range(len(chunk_offsets)-1)]
        
        if len(chunk_offsets) == 0:
            chunk_offsets = [(0, max_duration)]
        
        return chunk_offsets
    
    def trim_silence_vad(self, wav_16k=None, speech_timestamps=None, max_duration=60, max_silence=1.0):
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

        ''' 准备音频和各种参数 '''
        device = self.device

        # Process reference text and wav
        wav, _ = librosa.core.load(wav_bytes, sr=self.sr)
        # Pad wav if necessary
        ws = hparams['win_size']
        if len(wav) % ws < ws - 1:
            wav = np.pad(wav, (0, ws - 1 - (len(wav) % ws)), mode='constant', constant_values=0.0).astype(np.float32)
        wav = np.pad(wav, (0, 12000), mode='constant', constant_values=0.0).astype(np.float32)

        wav_16k = librosa.resample(wav, orig_sr=self.sr, target_sr=16000)
        max_ref_duration = self.max_ref_duration

        def _process_alignment(alignment_tokens):
            ''' 将aligner_lm的输出进行处理，得到音素、声调、持续时间 '''
            ph_ref, tone_ref, dur_ref, _ = split_ph_timestamp(deepcopy(alignment_tokens))
            dur_ref = dur_ref.clamp_min(0)
            ph_ref = torch.Tensor(ph_ref)[None].to(self.device)
            tone_ref = torch.Tensor(tone_ref)[None].to(self.device)
            if dur_ref.sum() < prompt_max_frame:
                dur_ref[-1] += prompt_max_frame - dur_ref.sum()
            elif dur_ref.sum() > prompt_max_frame:
                len_diff = dur_ref.sum() - prompt_max_frame
                while True:
                    for i in range(len(dur_ref)):
                        if dur_ref[i] > 0:
                            dur_ref[i] -= 1
                            len_diff -= 1
                        if len_diff == 0:
                            break
                    if len_diff == 0:
                        break
            return ph_ref, tone_ref, dur_ref
        
        if self.use_old_aligner:
            
            chunk_wav_offsets = self.chunk_wavs_vad(wav_16k, chunk_duration=10, max_duration=max_ref_duration)
            wav = wav[int(chunk_wav_offsets[0][0] * self.sr): int(chunk_wav_offsets[-1][-1] * self.sr)]
            wav_16k = wav_16k[int(chunk_wav_offsets[0][0] * 16000): int(chunk_wav_offsets[-1][-1] * 16000)]

            ph_ref_lst = []
            tone_ref_lst = []
            dur_ref_lst = []
            fm = 160 * 8
            for chunk_wav in chunk_wav_offsets:
                chunk_start, chunk_end = chunk_wav
                chunk_start = int((chunk_start * 16000) // fm * fm)
                chunk_end = int((chunk_end * 16000) // fm * fm)
                
                wav_16k_chunk = wav_16k[chunk_start: chunk_end]

                ''' 使用aligner_lm进行音频特征提取 '''
                with torch.inference_mode():
                    whisper_wav = wav_16k_chunk
                    mel = torch.tensor(whisper.log_mel_spectrogram(whisper_wav).T, dtype=self.precision).to(self.device)[None].transpose(1,2)
                    prompt_max_frame = mel.size(2) // self.fm * self.fm
                    mel = mel[:, :, :prompt_max_frame]
                    token = torch.LongTensor([[798]]).to(device)
                    audio_features = self.aligner_lm.embed_audio(mel)
                    # for i in tqdm(range(1024)):
                    for i in range(1024):
                        logits = self.aligner_lm.logits(token, audio_features, None)
                        token_pred = torch.argmax(F.softmax(logits[:, -1], dim=-1), 1)[None]
                        token = torch.cat([token, token_pred], dim=1)
                        if token_pred[0] == 799:
                            break
                    alignment_tokens = token[0, 1:-1]

                ph_ref, tone_ref, dur_ref = _process_alignment(alignment_tokens)
                ph_ref_lst.append(ph_ref)
                tone_ref_lst.append(tone_ref)
                dur_ref_lst.append(dur_ref)

            ph_ref = torch.cat(ph_ref_lst, dim=1)
            tone_ref = torch.cat(tone_ref_lst, dim=1)
            dur_ref = torch.cat(dur_ref_lst)
        
        else:
            segments = self.trim_silence_vad(wav_16k, max_duration=max_ref_duration)
            wav = np.concatenate([wav[int(segment['start'] * self.sr): int(segment['end'] * self.sr)] for segment in segments])
            wav_16k = np.concatenate([wav_16k[int(segment['start'] * 16000): int(segment['end'] * 16000)] for segment in segments])
            # remove gap leak
            wav = np.concatenate([wav, np.zeros(int(self.sr * 0.2))])
            wav_16k = np.concatenate([wav_16k, np.zeros(int(16000 * 0.2))])
            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=self.precision):
                # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                whisper_wav = torch.from_numpy(wav_16k)[None].to(self.device, self.precision)
                whisper_wav = whisper_wav[:, :whisper_wav.shape[-1] // 1280 * 1280]
                prompt_max_frame = whisper_wav.shape[-1] // 160 // 8 * 8
                token = torch.LongTensor([798])[None, :].to(self.device)
                token = self.aligner_lm.inference(
                    whisper_wav, token, topk=1, temperature=0.7,
                    max_new_tokens=16384, eos_idx=799, use_tqdm=self.use_tqdm
                )
            alignment_tokens = token[0]
            ph_ref, tone_ref, dur_ref = _process_alignment(alignment_tokens)

        if DEBUG:
            # print(f"{ph_ref.shape = }")
            # print(f"{dur_ref.shape = }")
            print(f"{dur_ref.max() = } {dur_ref.min() = }")
            # print(f"{dur_ref = }")

        mel2ph_ref = self.length_regulator(dur_ref[None]).to(self.device)
        mel2ph_ref = mel2ph_ref[:, :mel2ph_ref.size(1)//self.fm*self.fm]
        dur_ref = mel2token_to_dur(mel2ph_ref)

        if topk_dur > 1:
            self.dur_model.hparams["infer_top_k"] = topk_dur
        else:
            self.dur_model.hparams["infer_top_k"] = None

        with torch.inference_mode():
            ''' Forward WavVAE to obtain: prompt latent '''
            wav = torch.tensor(wav, dtype=self.precision)[None].to(device)
            with torch.autocast(device_type='cuda', dtype=self.precision):
                vae_latent = self.vae.encode_latent(wav)
            vae_latent = vae_latent[:, :mel2ph_ref.size(1)//4]
        
            ''' Duration Prompting '''
            if self.use_old_dur:
                dur_tokens_2d_ = mel2token_to_dur(mel2ph_ref, ph_ref.shape[1]).clamp(
                        max=self.hp_dur_model.get('dur_code_size', self.hp_dur_model.get('dur_max_value', 128)) - 1) + 1
    
                ctx_dur_tokens = dur_tokens_2d_.clone().flatten(0, 1).to(self.device)
                txt_tokens_flat_ = ph_ref.flatten(0, 1)
                ctx_dur_tokens = ctx_dur_tokens[txt_tokens_flat_ > 0][None]

                last_dur_pos_prompt = ctx_dur_tokens.shape[1]
                dur_spk_pos_ids_flat = range(0, last_dur_pos_prompt)
                dur_spk_pos_ids_flat = torch.LongTensor([dur_spk_pos_ids_flat]).to(mel2ph_ref.device)

                _, incremental_state_dur_prompt = self.dur_model.infer(
                    ph_ref, {'tone': tone_ref}, None, None, None,
                    ctx_vqcodes=ctx_dur_tokens, spk_pos_ids_flat=dur_spk_pos_ids_flat, return_state=True)
            else:
                if self.dur_model_type in ['lm', 'lm_seq2seq']:
                    merged_ph_tokens = map_phone_to_tokendict({'txt_token': ph_ref, 'tone': tone_ref}, pad_bos_eos=False)
                    try:
                        with torch.autocast(device_type='cuda', dtype=self.precision):
                            dur_start_pos = self.dur_model.prefill(merged_ph_tokens, dur_ref.clamp(0, self.dur_model.config.dur_max_value).to(self.device))
                    except:
                        print(f"{wav_path = }")
                        print(f"{merged_ph_tokens.shape = }")
                        print(f"{ph_ref.shape = }")
                        print(f"{tone_ref.shape = }")
                        print(f"{dur_ref.shape = }")
                        traceback.print_exc()
                        sys.exit()

            ''' ASR '''
            from modules.asr.sensevoice.sensevoice_api import run_asr_model
            text = run_asr_model([wav_16k], self.asr_model, with_segments=False)[0]['text_normed']

            # print('text_ref', text)
            # print(f'{dur_ref = }')

            ret = {
                'text_ref': text,
                'ph_ref': ph_ref.cpu(),
                'tone_ref': tone_ref.cpu(),
                'dur_ref': dur_ref.cpu(),
                'mel2ph_ref': mel2ph_ref.cpu(),
                'vae_latent': vae_latent.cpu(),
            }
            if self.use_old_dur:
                ret['incremental_state_dur_prompt'] = move_to_cpu(incremental_state_dur_prompt)
                ret['ctx_dur_tokens'] = ctx_dur_tokens.cpu()
            else:
                if self.dur_model_type in ['lm', 'lm_seq2seq']:
                    ret['dur_start_pos'] = dur_start_pos

            return ret

    @torch.inference_mode()
    def process_text_seg(self, t_i, text, len_text_segs, profile,
                         text_ref,
                         ph_ref, 
                         tone_ref,
                         dur_ref,
                         dur_start_pos,
                         mel2ph_ref,
                         vae_latent,
                         ctx_dur_tokens,
                         incremental_state_dur_prompt,
                         last_dur_pos_prompt,
                         wav_pred_,
                         sil_pad_lst,
                         ph_pred_lst,
                         tone_pred_lst,
                         words_timestamps,
                         words_timestamps_post,
                         dur_disturb,
                         dur_alpha,
                         normalize_dur,
                         return_timestamp,
                         timestamp_postprocess,
                         use_sa_frontend,
                         time_step,
                         w_all, w_txt, w_cap, w_ref, seq_cfg_w, timestep_annealing_w, use_amo_sampler,
                         global_prompt, local_prompt, prompttts_kwargs
                         ):

        # print(f'| Generating: {text.text_str}')
        
        is_prompttts = local_prompt is not None or global_prompt is not None
        
        if text.text_str.strip() == '':
            ph_pred_lst[t_i] = []
            tone_pred_lst[t_i] = []
            wav_pred_[t_i] = np.zeros(int(0.16 * self.sr))
            if return_timestamp:
                words_timestamps[t_i] = {'words': [], 'timestamps': []}
            sil_pad_lst[t_i] = (text.pause_at_start, text.pause_at_end)
            return

        ''' G2P '''
        with Timer('G2P', enable=profile):
            if not use_sa_frontend:
                with model_lock(self.lock):
                    ph_pred, tone_pred = self.g2p(text.text_str)
            else:
                # with model_lock(self.lock):
                from modules.tts.frontend_lm.sa_frontend import call_sa_frontend
                # print('text.sa_ssml_str', text.sa_ssml_str)
                sa_ret = call_sa_frontend(text.sa_ssml_str, debug=0)

                if sa_ret is None:  # 文本不合法，跳过
                    print(f'文本段落{t_i}/{len_text_segs}不合法，跳过')
                    return

                text_sa, ph_tokens, tone_tokens, alignment_sa = sa_ret
                
                # print(f"{text_sa = } {ph_tokens = }")
                
                # directly override
                new_text = SSML(text_sa)
                new_text.rate = text.rate
                new_text.pause_at_start = text.pause_at_start
                new_text.pause_at_end = text.pause_at_end
                text = new_text

                ph_pred = self.ling_dict['phone'].encode(' '.join(ph_tokens))
                ph_pred = torch.LongTensor(ph_pred)[None].to(self.device)
                tone_pred = self.ling_dict['tone'].encode(' '.join(tone_tokens))
                tone_pred = torch.LongTensor(tone_pred)[None].to(self.device)
                # print('ph_tokens', ph_tokens)
                # print('tone_tokens', tone_tokens)
                # print('text_sa', text_sa)
                # print('alignment_sa', alignment_sa)
        
        ''' Refeine Phonemes and Tones using SSML'''
        if not use_sa_frontend:
            # with Timer('Refeine Phonemes and Tones using SSML', enable=profile):
            ph_pred, tone_pred, ph2word = self.refine_ph_tone(text, ph_pred, tone_pred)
        # else:
        #     text_, ph_tokens, ph2word = align_word_phone(text.text_str, ph_tokens)
        #     print_align(text_, ph_tokens, ph2word)
        #     ph2word = [p-1 for p in ph2word]    # ignore "sil" at the begining
        
        ''' Caption encoding '''
        def run_caption_encoder(captions, device, caption_encoder=None, caption_tokenizer=None):
            caption_encoder = self.caption_encoder if caption_encoder is None else caption_encoder
            caption_tokenizer = self.caption_tokenizer if caption_tokenizer is None else caption_tokenizer
            inputs = caption_tokenizer(
                captions,
                padding=True,
                return_tensors="pt",
            )
            input_ids, attention_masks = inputs.input_ids.to(device), inputs.attention_mask.to(device)
            encoder_hidden_states = caption_encoder(
                input_ids, return_dict=False,
                attention_mask=attention_masks,
            )[0]
            return encoder_hidden_states, attention_masks
        with model_lock(self.lock):
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=self.precision, enabled=True):
                # with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
                caption_embs, caption_mask = run_caption_encoder([text_ref + text.text_str], self.device)
                caption_embs = caption_embs * caption_mask[..., None]
                caption_lens = caption_mask.sum(-1)
                if hasattr(self, "caption_encoder_dur_model"):
                    caption_embs_dur_model, caption_mask_dur_model = run_caption_encoder(
                        [text_ref + text.text_str], self.device, self.caption_encoder_dur_model, self.caption_tokenizer_dur_model
                    )
                    caption_embs_dur_model = caption_embs_dur_model * caption_mask_dur_model[..., None]
                    caption_lens_dur_model = caption_mask_dur_model.sum(-1)

        ''' Duration Prediction '''
        from utils.debug.cuda_index import IndexDebugMode
        with Timer('Duration Prediction', enable=profile):
            if self.use_old_dur:
                last_dur_token = ctx_dur_tokens[:, -1:]
                last_dur_pos = last_dur_pos_prompt
                txt_len = ph_pred.shape[1]
                dur_spk_pos_ids_flat = range(last_dur_pos, last_dur_pos + txt_len)
                dur_spk_pos_ids_flat = torch.LongTensor([dur_spk_pos_ids_flat]).to(self.device)
                last_dur_pos = last_dur_pos + txt_len
                inc_state = None
                if incremental_state_dur_prompt is not None:
                    gpu_index = None
                    if isinstance(self.device, str) and self.device.startswith("cuda"):
                        parts = self.device.split(":")
                        # "cuda" → 默认 0；"cuda:1" → 1
                        if len(parts) == 2 and parts[1] != "":
                            gpu_index = int(parts[1])
                        else:
                            gpu_index = 0
                    # 有增量状态时再搬到 CUDA 上
                    if gpu_index is not None:
                        inc_state = move_to_cuda(incremental_state_dur_prompt, gpu_index)
                    else:
                        inc_state = move_to_cuda(incremental_state_dur_prompt)
                with model_lock(self.lock):
                    dur_pred = self.dur_model.infer(
                        ph_pred, {'tone': tone_pred}, None, None, None,
                        incremental_state=inc_state,
                        first_decoder_inp=last_dur_token.to(self.device),
                        spk_pos_ids_flat=dur_spk_pos_ids_flat, use_tqdm=False
                    )
                dur_pred = dur_pred - 1
            else:
                if self.dur_model_type == 'lm':
                    merged_ph_tokens = map_phone_to_tokendict({'txt_token': ph_pred, 'tone': tone_pred}, pad_bos_eos=False)
                    with model_lock(self.lock), torch.autocast(device_type='cuda', dtype=self.precision):
                        # with torch.autocast(device_type='cuda', dtype=torch.float16):
                        dur_pred = self.dur_model.inference(
                            txt_tokens=merged_ph_tokens,
                            start_pos=dur_start_pos,
                            temperature=dur_disturb, use_tqdm=self.use_tqdm
                        )
                elif self.dur_model_type == 'lm_seq2seq':
                    merged_ph_tokens = map_phone_to_tokendict({'txt_token': ph_pred, 'tone': tone_pred}, pad_bos_eos=False)
                    with model_lock(self.lock):
                        dur_infer_kwargs = dict(
                            txt_tokens=merged_ph_tokens,
                            condition=caption_embs if not hasattr(self, "caption_encoder_dur_model") else caption_embs_dur_model,
                            start_pos=dur_start_pos,
                            temperature=dur_disturb, 
                            topk=5,
                            use_tqdm=self.use_tqdm
                        )
                        if self.dur_model.config.modeling_type == 'ar':
                            dur_infer_kwargs['dur_tokens'] = dur_ref.clamp(0, self.dur_model.config.dur_max_value)
                        with torch.autocast(device_type='cuda', dtype=self.precision):
                            dur_pred = self.dur_model.inference(**dur_infer_kwargs)
                elif self.dur_model_type == 'dit':
                    from inference.tts.dur_model_infer import forward_dur_dit
                    with model_lock(self.lock), torch.autocast(device_type='cuda', dtype=self.precision):
                        # with torch.autocast(device_type='cuda', dtype=torch.float16):
                        dur_pred = forward_dur_dit(
                            self.dur_model, ph_pred, tone_pred, text.text_str,
                            ph_ref, tone_ref, text_ref, dur_ref, 
                            caption_embs if not hasattr(self, "caption_encoder_dur_model") else caption_embs_dur_model, 
                            caption_lens if not hasattr(self, "caption_encoder_dur_model") else caption_lens_dur_model,
                            self.cfg_mask_text_token, self.cfg_mask_token_phone, self.cfg_mask_token_tone,
                            self.device
                        )

            if DEBUG:
                print(f"{dur_pred.max() = } {dur_pred.min() = }")
                # print(f"{dur_pred = }")
                # print(f"{dur_pred.max() = } {dur_pred.min() = } text_ref:{text_ref} text:{text}")
            if dur_pred.max() < 2 and dur_pred.min() == 0:
                # print(f"{dur_pred.max() = } {dur_pred.min() = } text_ref:{text_ref} text:{text}")
                print(f"{text_ref = }")
                print(f"{dur_ref = }")
                print(f"{text = }")
                print(f"{ph_tokens = }")
                print(f"{dur_pred = }")
                print(f"{dur_pred.max() = } {dur_pred.min() = }")
                
            # ref_ph_tokens_ = self.ling_dict['phone'].decode(ph_ref.squeeze().cpu().numpy()).split(' ')
            # ref_tone_tokens_ = self.ling_dict['tone'].decode(tone_ref.squeeze().cpu().numpy()).split(' ')
            # print(f"{ref_ph_tokens_ = } {ref_tone_tokens_ = } {dur_ref = } {dur_pred = }")
                
            ''' Normalize Speech Speed '''
            if normalize_dur and dur_pred.shape[1] > 10:
                sil_mask_pred = torch.zeros_like(ph_pred)
                sil_mask_ref = torch.zeros_like(ph_ref)
                for p in self.sil_ph:
                    sil_mask_pred[ph_pred == self.ling_dict['phone'].encode(p)[0]] = 1
                    sil_mask_ref[ph_ref == self.ling_dict['phone'].encode(p)[0]] = 1
                z_dur_pred = torch.log1p(dur_pred.float())
                z_dur_ref  = torch.log1p(dur_ref.float())
                diff_dur_sil = 0
                diff_dur_nonsil = 0
                if sil_mask_pred.sum() > 0 and sil_mask_ref.sum() > 0:
                    diff_dur_sil = z_dur_ref[sil_mask_ref == 1].mean() - z_dur_pred[sil_mask_pred == 1].mean()
                    z_dur_pred[sil_mask_pred == 1] = z_dur_pred[sil_mask_pred == 1] + diff_dur_sil
                if (sil_mask_pred != 1).sum() > 0 and (sil_mask_ref != 1).sum() > 0:
                    # print(f"{z_dur_ref[sil_mask_ref != 1].mean() = } {z_dur_pred[sil_mask_pred != 1].mean() = }")
                    diff_dur_nonsil = z_dur_ref[sil_mask_ref != 1].mean() - z_dur_pred[sil_mask_pred != 1].mean()
                    z_dur_pred[sil_mask_pred != 1] = z_dur_pred[sil_mask_pred != 1] + diff_dur_nonsil
                # if DEBUG:
                # print(f"{diff_dur_sil = } {diff_dur_nonsil = }")
                dur_pred = torch.expm1(z_dur_pred).clamp_min(0)
                d_floor = torch.floor(dur_pred)
                frac = (dur_pred - d_floor).clamp(0, 1)
                dur_pred = (d_floor + torch.bernoulli(frac)).long()
                
            ''' Control Speach Spead '''
            dur_pred = torch.round(dur_pred.float() / text.rate).long()

            dur_pred = dur_pred.clamp(0, self.hp_dur_model.get('dur_code_size', self.hp_dur_model.get('dur_max_value', 128)) - 1)
            if t_i < len_text_segs - 1:
                # add 0.32ms for crossfade
                dur_pred[:, -1] = dur_pred[:, -1] + 32
            else:
                dur_pred[:, -1] = dur_pred[:, -1].clamp(32, 80)

            if self.use_old_dur:
                dur_disturb_choice = (torch.rand_like(dur_pred.float()) > 0.5).float()
                dur_disturb_r = 1 + torch.rand_like(dur_pred.float()) * dur_disturb
                dur_pred = dur_pred * dur_disturb_r * dur_disturb_choice + \
                        dur_pred / dur_disturb_r * (1 - dur_disturb_choice)
                dur_pred = torch.round(dur_pred * dur_alpha).clamp(0, 127)
            # ['。', '！', '？', 'sil']
            for sil_token in [148, 153, 166, 145]:
                dur_pred[ph_pred==sil_token] = dur_pred[ph_pred==sil_token].clamp_min(32)
            # ['，', '；'] 
            for sil_token in [163, 165]:
                dur_pred[ph_pred==sil_token] = dur_pred[ph_pred==sil_token].clamp_min(16)
            dur_pred[:, 0] = 8

            if t_i == 0:
                dur_pred[:, 0] = 8
            # else:
            #     dur_pred[:, 0] = 48     # add 0.16ms for crossfade

            ''' Add Breaks '''
            if not use_sa_frontend:
                ph_pred, tone_pred, dur_pred, ph2word = self.add_breaks(
                    text, ph_pred, tone_pred, dur_pred, ph2word, break_token=163, break_tone=3
                )
            else:
                sil_pad_start, sil_pad_end = 0, 0

                if text.pause_at_start > 0:
                    if ph_pred[:, 0] == 145:
                        if dur_pred[:, 0]/100 + text.pause_at_start > self.max_silence_alive:
                            sil_pad_start = text.pause_at_start - (self.max_silence_alive - dur_pred[:, 0]/100).item()
                            dur_pred[:, 0] = round(self.max_silence_alive * 100)
                        else:
                            sil_pad_start = 0
                            dur_pred[:, 0] += round(text.pause_at_start * 100)
                    else:
                        ph_pred = torch.cat([torch.full((1, 1), 145).to(ph_pred), ph_pred], dim=1)
                        if text.pause_at_start > self.max_silence_alive:
                            sil_pad_start = text.pause_at_start - self.max_silence_alive
                            dur_pred = torch.cat([torch.full((1, 1), round(self.max_silence_alive * 100)).to(dur_pred), dur_pred], dim=1)
                        else:
                            sil_pad_start = 0
                            dur_pred = torch.cat([torch.full((1, 1), round(text.pause_at_start * 100)).to(dur_pred), dur_pred], dim=1)

                if text.pause_at_end > 0:
                    if ph_pred[:, -1].item() in [148, 163, 166, 153, 165, 147]:     # 。 ， ？ ！ ； 、
                        if dur_pred[:, -1]/100 + text.pause_at_end > self.max_silence_alive:
                            sil_pad_end = text.pause_at_end - (self.max_silence_alive - dur_pred[:, -1]/100).item()
                            dur_pred[:, -1] = round(self.max_silence_alive * 100)
                        else:
                            sil_pad_end = 0
                            dur_pred[:, -1] += round(text.pause_at_end * 100)
                    else:
                        ph_pred = torch.cat([ph_pred, torch.full((1, 1), 163).to(ph_pred)], dim=1)
                        if text.pause_at_end > self.max_silence_alive:
                            sil_pad_end = text.pause_at_end - self.max_silence_alive
                            dur_pred = torch.cat([dur_pred, torch.full((1, 1), round(self.max_silence_alive * 100)).to(dur_pred)], dim=1)
                        else:
                            sil_pad_end = 0
                            dur_pred = torch.cat([dur_pred, torch.full((1, 1), round(text.pause_at_end * 100)).to(dur_pred)], dim=1)
                
                sil_pad_lst[t_i] = (sil_pad_start, sil_pad_end)
            ''''''

            ph_pred_lst[t_i] = self.ling_dict['phone'].decode(ph_pred.squeeze().cpu().numpy()).split(' ')
            tone_pred_lst[t_i] = self.ling_dict['tone'].decode(tone_pred.squeeze().cpu().numpy()).split(' ')

            dur_sum = dur_pred.sum()
            vqs = hparams.get('vq_stride', 8)
            npad = vqs - dur_sum % vqs
            if npad < vqs:
                dur_pred[:, -1] += npad

            ''' Make words&timestamps '''
            if return_timestamp:
                try:
                    if not use_sa_frontend:
                        words_timestamps_cur = self.make_word_timestamps(text, dur_pred.squeeze().cpu().numpy(), ph2word)
                    else:
                        words = []
                        timestamps = []
                        offsets = [0] + np.cumsum(dur_pred.squeeze().cpu().numpy()).tolist()
                        dur_timestep = 0.01
                        for align_item in alignment_sa:
                            words.append(align_item['word'])
                            timestamps.append([offsets[align_item['phone_idx'][0]] * dur_timestep, 
                                               offsets[align_item['phone_idx'][-1] + 1] * dur_timestep])
                        words_timestamps_cur = {
                            'words': words,
                            'timestamps': timestamps
                        }
                    words_timestamps[t_i] = (words_timestamps_cur)
                except IndexError as err:
                    handle_exacption(err, text)
            ''''''
            mel2ph_pred = self.length_regulator(dur_pred).to(self.device)

        ''' DiT target speech generation '''
        with Timer('DiT target speech generation', enable=profile):
            
            if is_prompttts:
                
                def run_caption_encoder_prompttts(captions, device):
                    from tasks.tts.task_utils.prompttts_task_utils import build_dialogue_mask_from_ids
                    bsz = len(captions)
                    inputs = self.caption_tokenizer(
                        captions,
                        padding=True,
                        return_tensors="pt",
                    )
                    input_ids, attention_masks = inputs.input_ids.to(device), inputs.attention_mask.to(device)
                    encoder_hidden_states = self.caption_encoder(
                        input_ids, return_dict=False,
                        attention_mask=attention_masks,
                    )[0]
                    
                    if hparams.get('use_caption_text_mark', False):
                        caption_text_mark = build_dialogue_mask_from_ids(
                            input_ids=input_ids,
                            attention_mask=attention_masks,
                            tokenizer=self.caption_tokenizer,
                        ).to(device)
                        
                        return encoder_hidden_states, attention_masks, caption_text_mark

                    return encoder_hidden_states, attention_masks
                
                local_prompt = local_prompt if local_prompt is not None else ''
                global_prompt = global_prompt if global_prompt is not None else ''
                global_prompt = '<GPROMPT>' + global_prompt + '</GPROMPT>' if global_prompt != '' else ''
                local_prompt = local_prompt.replace('<tag>', '<TAG>').replace('</tag>', '</TAG>') if local_prompt != '' else ''
                captions = [global_prompt + local_prompt]
                with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                    caption_embs, caption_mask, caption_text_mark = run_caption_encoder_prompttts(captions, self.device)
                    caption_embs = caption_embs * caption_mask[..., None]
                    caption_lens = caption_mask.sum(-1)
                
            if prompttts_kwargs.get('ref_dur_only', False) or hparams.get('is_foundation') is True:
                mel2ph_pred = mel2ph_pred[:, :mel2ph_pred.size(1)//self.fm*self.fm].repeat(3, 1)
                target_size = mel2ph_pred.size(1)//4
                dur_seq = mel2token_to_dur(mel2ph_pred[:1])
                
                lat = torch.zeros((1, target_size, 32)).to(self.device)
                ctx_mask = torch.zeros((1, target_size, 1)).to(self.device)
                
                if isinstance(self.dit_text_tokenizer, CosyVoice2Tokenizer):
                    text_inputs = self.dit_text_tokenizer(text.text_str, padding=True, return_tensors='pt')
                    txt_tokens = text_inputs['input_ids'].to(self.device)
                    txt_mask = text_inputs['attention_mask'].bool().to(self.device)
                    txt_tokens[~txt_mask] = self.cfg_mask_text_token
                else:
                    text_inputs = self.dit_text_tokenizer(text.text_str, padding=True, return_tensors='pt').to(self.device)
                    txt_tokens = text_inputs['input_ids']
                    txt_mask = text_inputs['attention_mask'].bool()
                    txt_tokens[~txt_mask] = self.cfg_mask_text_token
                    
                ph_seq = ph_pred
                tone_seq = tone_pred
                en_tone_idx = ~((tone_seq == 4) | ( (11 <= tone_seq) & (tone_seq <= 15)) | (tone_seq == 0))
                tone_seq[en_tone_idx] = 3
                    
            else:
                # Prepare duration token 
                mel2ph_pred = torch.cat((mel2ph_ref, mel2ph_pred+ph_ref.size(1)), dim=1)
                if seq_cfg_w is None:
                    mel2ph_pred = mel2ph_pred[:, :mel2ph_pred.size(1)//self.fm*self.fm].repeat(5, 1)
                else:
                    mel2ph_pred = mel2ph_pred[:, :mel2ph_pred.size(1)//self.fm*self.fm].repeat(3, 1)
                target_size = mel2ph_pred.size(1)//4
                dur_seq = mel2token_to_dur(mel2ph_pred[:1])

                if hparams.get('use_sparse_dur', False):
                    dur_pred = torch.cat([dur_ref, dur_pred], dim=1)
                    mel2ph_sparse = compute_mel2aug_from_dur(
                        dur_pred.int().squeeze().cpu().numpy().tolist(),
                        gap_mode=hparams.get('sparse_dur_mode', 'proportional'),
                        gap_frames=hparams.get('sparse_dur_frames', 4),
                        gap_alpha=hparams.get('sparse_dur_alpha', 0.2),
                        min_keep=hparams.get('sparse_dur_min_keep', 1),
                        keep_ratio=hparams.get('sparse_dur_keep_ratio'),
                        symmetric=hparams.get('sparse_dur_symmetric', True),
                    )
                    if seq_cfg_w is None:
                        mel2ph_sparse = torch.stack([mel2ph_sparse] * 5).to(self.device)
                    else:
                        mel2ph_sparse = torch.stack([mel2ph_sparse] * 3).to(self.device)
                    mel2ph_sparse = mel2ph_sparse[:, :mel2ph_pred.shape[1]]

                ctx_mask = torch.ones_like(vae_latent[:, :, 0:1])
                lat = F.pad(vae_latent, (0, 0, 0, target_size - vae_latent.size(1)), mode='constant', value=0)
                ctx_mask = F.pad(ctx_mask, (0, 0, 0, target_size - ctx_mask.size(1)), mode='constant', value=0)
                                
                if isinstance(self.dit_text_tokenizer, CosyVoice2Tokenizer):
                    text_inputs = self.dit_text_tokenizer(text.text_str, padding=True, return_tensors='pt')
                    txt_tokens = text_inputs['input_ids'].to(self.device)
                    txt_mask = text_inputs['attention_mask'].bool().to(self.device)
                    txt_tokens[~txt_mask] = self.cfg_mask_text_token
                else:
                    text_inputs = self.dit_text_tokenizer(text.text_str, padding=True, return_tensors='pt').to(self.device)
                    txt_tokens = text_inputs['input_ids']
                    txt_mask = text_inputs['attention_mask'].bool()
                    txt_tokens[~txt_mask] = self.cfg_mask_text_token

                # Disable the English tone (set them to 3)"""
                ph_seq = torch.cat((ph_ref, ph_pred), dim=1)
                tone_seq = torch.cat((tone_ref, tone_pred), dim=1)
                en_tone_idx = ~((tone_seq == 4) | ( (11 <= tone_seq) & (tone_seq <= 15)) | (tone_seq == 0))
                tone_seq[en_tone_idx] = 3
            
            if seq_cfg_w is None:

                ph_seq = torch.cat([
                    ph_seq,
                    ph_seq,
                    torch.full(ph_seq.size(), self.cfg_mask_token_phone, device=self.device),
                    torch.full(ph_seq.size(), self.cfg_mask_token_phone, device=self.device),
                    torch.full(ph_seq.size(), self.cfg_mask_token_phone, device=self.device)
                ], dim=0)

                tone_seq = torch.cat([
                    tone_seq,
                    tone_seq,
                    torch.full(tone_seq.size(), self.cfg_mask_token_tone, device=self.device),
                    torch.full(tone_seq.size(), self.cfg_mask_token_tone, device=self.device),
                    torch.full(tone_seq.size(), self.cfg_mask_token_tone, device=self.device)
                ], dim=0)
                
                dur_seq = torch.cat([
                    dur_seq,
                    dur_seq,
                    torch.zeros_like(dur_seq),
                    torch.zeros_like(dur_seq),
                    torch.zeros_like(dur_seq),
                ], dim=0)

                txt_tokens = torch.cat([
                    txt_tokens,
                    txt_tokens,
                    torch.full(txt_tokens.size(), self.cfg_mask_text_token, device=self.device),
                    torch.full(txt_tokens.size(), self.cfg_mask_text_token, device=self.device),
                    torch.full(txt_tokens.size(), self.cfg_mask_text_token, device=self.device),
                ], dim=0)
                txt_mask = torch.cat([txt_mask] * 5, dim=0)

                caption_embs = torch.cat([
                    caption_embs,
                    torch.zeros_like(caption_embs),
                    caption_embs,
                    torch.zeros_like(caption_embs),
                    torch.zeros_like(caption_embs)
                ], dim=0)
                caption_lens = torch.cat([caption_lens] * 5, dim=0).to(torch.long)
                if is_prompttts:
                    caption_text_mark = torch.cat([caption_text_mark] * 5, dim=0)
                
                lat = torch.cat([
                    lat,
                    torch.zeros_like(lat),
                    torch.zeros_like(lat),
                    lat,
                    torch.zeros_like(lat)
                ], dim=0)
                ctx_mask = torch.cat([ctx_mask] * 5, dim=0)
                
                vad_mask = torch.ones_like(ctx_mask)[..., 0]
                vad_mask[:, :int(0.2 * 25)] = 0
                vad_mask[:, -int(0.2 * 25):] = 0
                
                seq_cfg_w = [w_all, w_txt, w_cap, w_ref]
                
            else:
                             
                if is_prompttts and prompttts_kwargs.get('drop_phone', False):
                    ph_seq = torch.cat([torch.full(ph_seq.size(), self.cfg_mask_token_phone, device=self.device)] * 3, dim=0)
                    tone_seq = torch.cat([torch.full(tone_seq.size(), self.cfg_mask_token_tone, device=self.device)] * 3, dim=0)
                    dur_seq = torch.cat([torch.full(dur_seq.size(), 0.0, device=self.device)] * 3, dim=0)
                    
                else:
                    ph_seq = torch.cat([
                        ph_seq,
                        ph_seq,
                        torch.full(ph_seq.size(), self.cfg_mask_token_phone, device=self.device)
                    ], dim=0)

                    tone_seq = torch.cat([
                        tone_seq,
                        tone_seq,
                        torch.full(tone_seq.size(), self.cfg_mask_token_tone, device=self.device)
                    ], dim=0)
                    
                    dur_seq = torch.cat([
                        dur_seq,
                        dur_seq,
                        torch.zeros_like(dur_seq),
                    ], dim=0)

                txt_tokens = torch.cat([
                    txt_tokens,
                    txt_tokens,
                    torch.full(txt_tokens.size(), self.cfg_mask_text_token, device=self.device)
                ], dim=0)
                txt_mask = torch.cat([txt_mask] * 3, dim=0)

                if is_prompttts:
                    caption_embs = torch.cat([
                        caption_embs,
                        torch.zeros_like(caption_embs),
                        torch.zeros_like(caption_embs),
                    ], dim=0)
                    caption_text_mark = torch.cat([caption_text_mark] * 3, dim=0)
                else:
                    caption_embs = torch.cat([
                        caption_embs,
                        caption_embs,
                        torch.zeros_like(caption_embs),
                    ], dim=0)
                caption_lens = torch.cat([caption_lens] * 3, dim=0).to(torch.long)                    
                
                lat = torch.cat([
                    lat,
                    torch.zeros_like(lat),
                    torch.zeros_like(lat)
                ], dim=0)
                ctx_mask = torch.cat([ctx_mask] * 3, dim=0)
                
                vad_mask = torch.ones_like(ctx_mask)[..., 0]
                vad_mask[:, :int(0.2 * 25)] = 0
                vad_mask[:, -int(0.2 * 25):] = 0

            inputs = {
                'phone': ph_seq,
                'tone': tone_seq,
                "lat_ctx": lat * ctx_mask,
                "ctx_mask": ctx_mask,
                "mel2ph": mel2ph_pred,
                "txt_tokens": txt_tokens,
                'txt_mask': txt_mask,
                "caption_emb": caption_embs,
                "caption_lens": caption_lens,  # B
                "vad_mask": vad_mask,
                "dur": dur_seq,
                "tgt_len": torch.LongTensor([target_size] * 3).to(self.device)
            }
            if hparams.get('use_sparse_dur', False):
                inputs['mel2ph_sparse'] = mel2ph_sparse
            if is_prompttts:
                inputs['caption_text_mark'] = caption_text_mark

            # Euler ODE solver
            with model_lock(self.lock), torch.autocast(device_type='cuda', dtype=self.precision):
                # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                x = self.dit.inference(
                    inputs, timesteps=time_step, 
                    seq_cfg_w=seq_cfg_w, 
                    timestep_annealing_w=timestep_annealing_w,
                    use_amo_sampler=use_amo_sampler
                )

        # WavVAE decode
        with Timer('WavVAE decode', enable=profile):
            if hparams.get('is_foundation') is False and not (is_prompttts and prompttts_kwargs.get('ref_dur_only', False)):
                x[:, :vae_latent.size(1)] = vae_latent
            with model_lock(self.lock):
                with torch.autocast(device_type='cuda', dtype=self.precision):
                    wav_pred = self.vae.decode(x)[0,0].to(torch.float32)
            
            ''' Post-processing '''
        with Timer('Post-processing', enable=profile):
            hop_size = self.hp_vae['hop_size']
            vae_stride = self.hp_vae['vae_stride']
            # Trim prompt wav
            if hparams.get('is_foundation') is False and not (is_prompttts and prompttts_kwargs.get('ref_dur_only', False)):
                wav_pred = wav_pred[vae_latent.size(1)*vae_stride*hop_size:]
            # clamp the maximum value
            if wav_pred.abs().max() > 1:
                print('Wav amplitude exceed 1, clip it.')
                wav_pred = wav_pred / (wav_pred.abs().max())

            wav_pred = wav_pred.cpu().numpy()
            wav_pred_[t_i] = wav_pred


    def forward(self, resource_context, input_text, time_step, w_all=1.0, w_txt=1.0, w_cap=1.0, w_ref=1.0, seq_cfg_w=None,
                speech_rate=1, timestep_annealing_w=(1.0, 0.0, 1.0), use_amo_sampler=False, return_timestamp=True, timestamp_postprocess=False, 
                return_format='wav', custom_ph_table=None, dur_disturb=0.1, dur_alpha=1.0, normalize_dur=False,
                num_parallel_workers=5, use_sa_frontend=True, chunk_num_words_zh=60, chunk_num_words_en=130, 
                global_prompt=None, local_prompt=None, prompttts_kwargs=None, **kwargs):
        """
        Args:
            resource_context (dict): resource context dict generated from self.preprocess(). 由 self.preprocess() 方法返回的资源包
            input_text (str): input text, with or without SSML format. 输入的文本，可支持 SSML 格式
            time_step (int): number of time steps for generation. 生成所需要的步数
            p_w (float): weight to control pronunciation. 控制发音清晰度的权重
            t_w (float): weight to control speaker similarity. 控制音色相似度的权重
            speech_rate (float): control speech rate, from 0.01 to 2, 1 for original rate. 控制语速，最低0.01倍，最高2倍速，默认1倍速
            return_timestamp (bool): if true, return raw-text timestamps. 控制是否返回与原始文本对齐的时间戳
            timestamp_postprocess (bool): [deprecated]. 已弃用. if true, also perform post-alignment to obtain more accurate timestamps. 控制是否额外使用后处理对齐的方式获得时间戳
            return_format (str): only support 'wav' or 'mp3', 'wav' recommended. 控制返回的文件格式，支持'wav'和'mp3'，推荐使用'wav'
            custom_ph_table (dict | None): whitelist for special pronunciation. 控制特殊发音规则的白名单
            dur_disturb (float): randomly disturb phoneme duration. 控制音素时长随机扰动的权重
            dur_alpha (float): [deprecated]. 已弃用

        Returns:
            output (MegaTTS3Output): contains wav_bytes, words_timestamps, and words_timestamps_post. 
                结构体包含音频文件的二进制内容(wav_bytes)、时间戳(words_timestamps)、后处理获得的时间戳(words_timestamps_post)，时间戳默认返回 None
        """
        device = self.device
        incremental_state_dur_prompt = resource_context.get('incremental_state_dur_prompt')
        last_dur_pos_prompt = resource_context['ctx_dur_tokens'].shape[1] if 'ctx_dur_tokens' in resource_context else None
        prompttts_kwargs = prompttts_kwargs if prompttts_kwargs is not None else {}

        profile = os.environ.get('MEGA_PROFILE', 'false').strip().lower() == 'true'

        with torch.inference_mode():
            ''' Generating '''
            # input_text = remove_space(input_text)
            # remove blank special symbols
            input_text = ''.join(c for c in input_text if c.isprintable())

            input_text = SSML(input_text)
            input_text.rate = float(speech_rate)

            ''' generate pure silence '''
            if input_text.text_str.strip() == '':
                from bs4 import Tag
                sil_time = 0
                for ele in input_text:
                    if ele.name == 'break' and isinstance(ele, Tag):
                        sil_time += float(ele.get('time')[:-1])
                if sil_time == 0:
                    raise RuntimeError('输入为空，输入不合法')
                wav_bytes = to_wav_bytes(np.zeros((int(sil_time * self.sr),)).astype(float), self.sr)
                if return_format == 'mp3':
                    wav_bytes = wav_bytes_to_mp3_bytes(wav_bytes)
                output = MegaTTS3Output(
                    wav_bytes=wav_bytes,
                    words_timestamps={'words': [], 'timestamps': []},
                    words_timestamps_post=None,
                    duration=sil_time,
                    ph_pred=[],
                    tone_pred=[]
                )
                return output
                        
            ''' preprocess text '''
            with Timer('preprocess text', enable=profile):
                ph_replace_table = deepcopy(self.ph_replace_table)
                if custom_ph_table is not None:
                    ph_replace_table.update(custom_ph_table)
                text_segs = self.preprocess_text(input_text, ph_replace_table, use_sa_frontend, chunk_num_words_zh, chunk_num_words_en)

            len_text_segs = len(text_segs)
            words_timestamps = [None] * len_text_segs
            words_timestamps_post = [None] * len_text_segs
            wav_pred_ = [None] * len_text_segs
            sil_pad_lst = [None] * len_text_segs
            ph_pred_lst = [None] * len_text_segs
            tone_pred_lst = [None] * len_text_segs

            with ThreadPoolExecutor(max_workers=num_parallel_workers) as executor:
                futs = []
                for t_i, text in enumerate(text_segs):
                    future = executor.submit(
                                self.process_text_seg, 
                                *(t_i, text, len_text_segs, profile), 
                                **{
                                    "text_ref": resource_context['text_ref'],
                                    "ph_ref": resource_context['ph_ref'].detach().clone().to(device), 
                                    "tone_ref": resource_context['tone_ref'].detach().clone().to(device),
                                    "dur_ref": resource_context['dur_ref'].detach().clone().to(device),
                                    "dur_start_pos": resource_context.get('dur_start_pos', None),
                                    "mel2ph_ref": resource_context['mel2ph_ref'].detach().clone().to(device),
                                    "vae_latent": resource_context['vae_latent'].detach().clone().to(device),
                                    "ctx_dur_tokens": resource_context['ctx_dur_tokens'].detach().clone().to(device) if 'ctx_dur_tokens' in resource_context else None,
                                    "incremental_state_dur_prompt": deepcopy(incremental_state_dur_prompt),
                                    "last_dur_pos_prompt": last_dur_pos_prompt,
                                    "wav_pred_": wav_pred_,
                                    "sil_pad_lst": sil_pad_lst,
                                    "ph_pred_lst": ph_pred_lst,
                                    "tone_pred_lst": tone_pred_lst,
                                    "words_timestamps": words_timestamps,
                                    "words_timestamps_post": words_timestamps_post,
                                    "dur_disturb": dur_disturb,
                                    "dur_alpha":dur_alpha,
                                    "normalize_dur": normalize_dur,
                                    "return_timestamp": return_timestamp,
                                    "timestamp_postprocess": timestamp_postprocess,
                                    "use_sa_frontend": use_sa_frontend,
                                    "time_step": time_step,
                                    "w_all": w_all,
                                    "w_txt": w_txt,
                                    "w_cap": w_cap,
                                    "w_ref": w_ref,
                                    "seq_cfg_w": seq_cfg_w,
                                    "timestep_annealing_w": timestep_annealing_w,
                                    "use_amo_sampler": use_amo_sampler,
                                    "global_prompt": global_prompt,
                                    "local_prompt": local_prompt,
                                    "prompttts_kwargs": prompttts_kwargs
                                })
                    futs.append(future)
                
                results = [f.result() for f in futs]

            words_timestamps = [s for s in words_timestamps if s is not None]
            wav_pred_ = [s for s in wav_pred_ if s is not None]
            ph_pred_lst = [s for s in ph_pred_lst if s is not None]
            tone_pred_lst = [s for s in tone_pred_lst if s is not None]

            # assert len(wav_pred_) == len(ph_pred_lst) == len(tone_pred_lst), f"{len(wav_pred_)}, {len(ph_pred_lst)}, {len(tone_pred_lst)}"

            for i in range(1, len(ph_pred_lst)):
                ph_pred_lst[i] = ph_pred_lst[i][1:]
            ph_pred_lst = np.concatenate(ph_pred_lst).tolist()
            for i in range(1, len(tone_pred_lst)):
                tone_pred_lst[i] = tone_pred_lst[i][1:]
            tone_pred_lst = np.concatenate(tone_pred_lst).tolist()
            
            words_timestamps_fail = (None in words_timestamps) or (len(words_timestamps) == 0) or (len(words_timestamps) != len(wav_pred_))
            if words_timestamps_fail:
                words_timestamps = []
                words_timestamps_post = []

            # normalize loudness
            if len(wav_pred_) > 1:
                silent_speech = False
                meter = pyln.Meter(self.sr)  # create BS.1770 meter
                j = 0
                while j < len(wav_pred_):
                    try:
                        loudness_1 = meter.integrated_loudness(wav_pred_[0].astype(float))
                        break
                    except:
                        pass
                    j += 1
                else:
                    silent_speech = True
                if not silent_speech:
                    for i in range(j+1, len(wav_pred_)):
                        wav_pred__ = wav_pred_[i]
                        loudness_pred = meter.integrated_loudness(wav_pred__)
                        wav_pred__ = pyln.normalize.loudness(wav_pred__, loudness_pred, loudness_1)
                        if np.abs(wav_pred__).max() >= 1:
                            wav_pred__ = wav_pred__ / np.abs(wav_pred__).max() * 0.95
                        wav_pred_[i] = wav_pred__

            if not timestamp_postprocess:
                wav_pred_, words_timestamps = self.combine_audio_segments(wav_pred_, words_timestamps, sil_pad_lst)
                words_timestamps_post = None
            else:
                _, words_timestamps = self.combine_audio_segments(wav_pred_, words_timestamps, sil_pad_lst)
                wav_pred_, words_timestamps_post = self.combine_audio_segments(wav_pred_, words_timestamps_post, sil_pad_lst)

            if not return_timestamp:
                words_timestamps = words_timestamps_post = None

            if words_timestamps_fail:
                words_timestamps = None

            wav_bytes = to_wav_bytes(wav_pred_.astype(float), self.sr)
            if return_format == 'mp3':
                wav_bytes = wav_bytes_to_mp3_bytes(wav_bytes)

            output = MegaTTS3Output(
                wav_bytes=wav_bytes,
                words_timestamps=words_timestamps,
                words_timestamps_post=words_timestamps_post,
                duration=wav_pred_.shape[-1] / self.sr,
                ph_pred=ph_pred_lst,
                tone_pred=tone_pred_lst
            )

            return output


class MegaTTS3DiTV4Infer(DiTBuildModelMixinV4, MegaTTS3DiTInfer):
    pass


class MegaTTS3DiTV6Infer(DiTBuildModelMixinV6, MegaTTS3DiTInfer):
    pass


class MegaTTS3DiTV7Infer(DiTBuildModelMixinV7, MegaTTS3DiTInfer):
    pass


if __name__ == '__main__':
    # 仅改动第一段 main：去掉 DataPipe/多进程/上传，改为单句（逐句）infer。
    # 其余参数与原脚本保持一致，不从外部传参。
    import os
    import tempfile
    from datetime import datetime
    from pathlib import Path

    import torch
    from tqdm import tqdm

    from utils.commons.os_utils import kill_void
    from utils.text.ssml_utils import SSML
    from utils.audio.io import save_wav_bytes

    # 如果存在本地环境变量文件，加载之
    if os.path.isfile('.env.local'):
        from dotenv import load_dotenv
        load_dotenv('.env.local')

    # ====== 原脚本中保持不变的检查点路径设置 ======
    # dit_exp_name = 'checkpoints/250909_megatts3_dit_v2_32layer_16ca_qknorm_fix'
    # dit_exp_name = 'checkpoints/250909_megatts3_dit_v2_32layer_16ca_sparsedur_qknorm_fix'
    dit_exp_name = 'checkpoints/251118_megatts3_dit_v4_32layer_qknorm_cross'

    # dur_exp_name = 'checkpoints/megatts3_wavdit/duration_lm'
    # dur_exp_name = 'checkpoints/250826_dur_lm'
    # dur_exp_name = 'checkpoints/250827_dur_lm_base'
    # dur_exp_name = 'checkpoints/250910_dur_lm_base_icl'
    # dur_exp_name = 'checkpoints/250911_dur_lm_base_seq2seq_txtcond'
    # dur_exp_name = 'checkpoints/250923_dur_lm_base_seq2seq_txtcond_bias'
    # dur_exp_name = 'checkpoints/250924_dur_lm_base_seq2seq_v2_txtcond_bias'
    # dur_exp_name = 'checkpoints/250925_dur_lm_base_seq2seq_v2_ce_txtcond_bias'
    # dur_exp_name = 'checkpoints/250928_dur_lm_base_seq2seq_v2_txtcond_bias'
    # dur_exp_name = 'checkpoints/250924_dur_lm_base_seq2seq_v2_txtcond_bias_fix'
    dur_exp_name = 'checkpoints/251009_dur_lm_base_seq2seq_v3_txtcond_bias'

    # frontend_exp_name = 'checkpoints/250823_lm_mfa_seq2seq_small_wavlmlarge'
    frontend_exp_name = 'checkpoints/250923_lm_mfa_seq2seq_small_wavlmlarge_long_robust'

    # ===== 推理器参数 / 超参（保持与原脚本一致） =====
    cls_kwargs = dict(
        # dit_exp_name=move_to_shm(dit_exp_name),
        # dur_exp_name=move_to_shm(dur_exp_name),
        # frontend_exp_name=move_to_shm(frontend_exp_name),
        # wavvae_exp_name=move_to_shm('checkpoints/1231_megatts3_wavvae_v3_25hz_kl001_fix4'),
        dit_exp_name=dit_exp_name,
        dur_exp_name=dur_exp_name,
        frontend_exp_name=frontend_exp_name,
        wavvae_exp_name='checkpoints/1231_megatts3_wavvae_v3_25hz_kl001_fix4',
        use_old_aligner=False,
        use_old_dur=True,
        max_ref_duration=60,
        use_tqdm=False,
        use_ema=True,
        precision='bf16'
    )

    infer_kwargs = dict(
        time_step=50,
        w_all=3,
        w_txt=-1,
        w_cap=0,
        w_ref=0,
        seq_cfg_w=(2.5, 5),
        use_amo_sampler=True,
        speech_rate=1.0,
        custom_ph_table={
            'en': {'@': 'at', '&': 'and'},
            'zh': {'@': '艾特', '&': '和'}
        },
        dur_disturb=0.2,
        timestep_annealing_w=(0.6, 0.6, 1.0),
        chunk_num_words_zh=80,
        num_parallel_workers=1,
        normalize_dur=True
    )

    # 与原脚本保持相同的细节调整逻辑
    if cls_kwargs['use_old_dur']:
        infer_kwargs['dur_disturb'] = 0.1
    if 'ce' in dur_exp_name:
        infer_kwargs['dur_disturb'] = 0.2
    if infer_kwargs.get('use_amo_sampler', False):
        infer_kwargs['seq_cfg_w'] = (1.5, 3)
    if cls_kwargs['use_old_dur']:
        cls_kwargs['max_ref_duration'] = 20

    version = '251013'
    time_version = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    version = f"{version}/{time_version}"

    # ======= 输入（保持与原脚本一致，不从命令行/外部读取） =======
    audio_paths = ['/mnt/bn/sa-ag-data/liruiqi/code/MegaTTS3-inference-safrontend/infer_out/prompts/top5音频0923/专家采访/专家采访2-情景原声_vocal.wav']

    texts = [
        # '四肢不胖肚子大，可能是你的肠胃在作怪，肠胃不好的人要注意了。如果你经常胃痛胃胀，反酸烧心，消化不良，恶心呕吐，打嗝嗳气，腹胀腹痛，肚子大爱放屁，排便困难，有这些问题啊，说明你的肠胃问题啊，已经很严重了，千万要引起重视，可不能再拖了。不用东奔西跑，不用花大价钱，在家就可以试试这个万太医肠胃贴，精选8大肠胃友好成分，贴在肚脐处的神阙穴，使用方便不用入口，温和调理你的肠胃问题，万太医啊在肠胃领域深耕了多年，正规医疗器械，经常肠胃不舒服，一吃点凉的辣的、喝点酒就难受的朋友点击视频下方链接查看详情吧。',
        '1955-1980年出生的您注意了！您的2025新款超声波洗菜盆专属优惠，已送达！\r\n别划走！这份福利正等着您签收！如果这次您再错过，优惠名额将转给他人——但现在，它依然属于您！点击下方链接，立即锁定您的优惠！名额已发送至您的手机。今天不参与，资格即失效！无论多忙，只需几秒就能抢到手！无需转发！无需下载！看到即得！所见即所得！现在参与，最快明天到货！名额告急，错过后悔莫及！赶紧点击下方链接，抢占最后名额！',
        # '你这首独奏弹得不错呦，我以为年轻人都在搞电子乐呢',
    ]

    # ======= 单句（逐句）推理：不使用 DataPipe / 多进程 / 上传 =======
    print('| 初始化推理器...')
    infer_ins = MegaTTS3DiTV4Infer(**cls_kwargs)
    kill_void()

    out_root = Path(f'infer_out/tts/single_infer/{version}')
    out_root.mkdir(parents=True, exist_ok=True)

    for audio_idx, audio_path in enumerate(audio_paths):
        print('=' * 100)
        print(f'| 参考音频[{audio_idx}]: {audio_path}')
        # 预处理仅做一次（单句/逐句推理）
        with open(audio_path, 'rb') as f:
            file_content = f.read()
        print('| 开始 preprocess...')
        resource_context = infer_ins.preprocess(file_content, audio_path)
        ref_out_dir = out_root / Path(audio_path).stem.replace('_vocal', '')
        ref_out_dir.mkdir(parents=True, exist_ok=True)

        pbar = tqdm(total=len(texts), desc='| 生成中')
        for t_i, input_text in enumerate(texts):
            print('-' * 80)
            print(f'| 文本[{t_i}]: {input_text[:80]}...')
            output = infer_ins.forward(resource_context, input_text, **infer_kwargs)

            wav_bytes = output.wav_bytes
            text_clean = SSML(input_text).text_str
            out_path = ref_out_dir / f"{Path(audio_path).stem.replace('_vocal', '')}+{text_clean[:20]}.wav"
            save_wav_bytes(wav_bytes, str(out_path))
            print(f'| 成功 -> {out_path}')
            pbar.update(1)
        pbar.close()

    print('\n| 全部生成完成。')
