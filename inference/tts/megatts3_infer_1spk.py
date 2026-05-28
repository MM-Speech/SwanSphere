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
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import set_hparams, hparams
from utils.commons.meters import Timer
from utils.commons.os_utils import handle_exacption
from utils.commons.io import print_once
from utils.commons.import_utils import import_module_bystr, get_class_from_module
from utils.nn.ema import restore_ema

from modules.tts.ar_dur.commons.nar_tts_modules import LengthRegulator
from modules.tts.ar_dur.commons.align_ops import compute_mel2aug_from_dur
from modules.tts.scriptspeech.build_model_utils import DiTBuildModelMixinV2

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
            dit_exp_name='/mnt/bn/genai-data2/liruiqi/code/ScriptSpeech/checkpoints/250827_megatts3_dit_v2_32layer_16ca',
            dur_exp_name='checkpoints/250826_dur_lm',
            frontend_exp_name='checkpoints/250820_lm_mfa_base_wavlmlarge',
            use_old_aligner=True,
            use_old_dur=True,
            max_ref_duration=20,
            use_tqdm=True,
            **kwargs
        ):

        self.sr = 24000
        self.fm = 8
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.use_tqdm = use_tqdm

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
        self.wavvae_exp_name = 'checkpoints/1231_megatts3_wavvae_v3_25hz_kl001_fix4'

        self.build_model(self.device)

        # break (silence)
        self.max_silence_alive = 1.28    # 1.28s
        self.max_ref_duration = max_ref_duration

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
            from modules.tts.ar_dur.dur_lm import DurationLM, build_dur_model
            hp_dur_model = self.hp_dur_model = set_hparams(f'{self.dur_exp_name}/config.yaml', global_hparams=False)
            self.dur_model = build_dur_model(hp_dur_model, vocab_size=810, padding_idx=797)
            self.dur_model.hparams = {}
            self.dur_model.eval()
            load_ckpt(self.dur_model, self.dur_exp_name, 'model', mmap=True)
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
            aligner_hparams = set_hparams(f'{self.frontend_exp_name}/config.yaml', global_hparams=False)
            self.aligner_lm = build_asr_model(aligner_hparams, init_pretrained=False, vocab_size=6800, padding_idx=797)
            self.aligner_lm.eval()
            load_ckpt(self.aligner_lm, self.frontend_exp_name, 'model', strict=True, mmap=True)
            self.aligner_lm.to(self.device)

    ''' 加载模型参数 '''
    def build_model(self, device):
        self.device = device
        self.precision = torch.bfloat16

        set_hparams(f'{self.dit_exp_name}/config.yaml', print_hparams=False)
        hparams['use_fsdp'] = False

        ''' Load Dict '''
        ling_dict = json.load(open('egs/tts/megatts3_dict.json'))
        self.ling_dict = {k: TokenTextEncoder(None, vocab_list=ling_dict[k], replace_oov='<UNK>') for k in ['phone', 'tone']}
        self.token_encoder = token_encoder = self.ling_dict['phone']
        # self.ph_replace_table = json.load(open(f"{current_dir}/ph_replace_table.json"))
        self.ph_replace_table = {'en': {}, 'zh': {}}

        ''' Load Duration LM '''
        self.build_dur_model()

        ''' Load DiT '''
        self._build_model()
        # if hparams.get('use_ema', False):
        #     restore_ema(self.dit, f'{self.dit_exp_name}', 'model_ema', hparams, self.device)
        # else:
        load_ckpt(self.dit, f'{self.dit_exp_name}', 'dit', strict=False, mmap=True)
        self.vae.eval()
        self.vae.to(self.device, dtype=self.precision)
        self.dit.eval()
        self.dit.to(device, dtype=self.precision)
        self.cfg_mask_token_phone = 302 - 1
        self.cfg_mask_token_tone = 32 - 1
        self.caption_encoder.to(device, dtype=self.precision)

        ''' Load Frontend LM '''
        self.build_frontend_model()

        ''' ASR '''
        from modules.asr.sensevoice.sensevoice_api import build_asr_model
        self.asr_model = build_asr_model(device)

        ''' VAD '''
        from silero_vad import load_silero_vad
        self.vad_model = load_silero_vad()

        self.lock = threading.Lock()


    ''' 文本预处理 '''
    def preprocess_text(self, input_text: SSML, ph_replace_table=None, use_sa_frontend=False):

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
            text_segs = SSML.chunk_text_with_breaks(input_text, limit=130, language_type='en', debug=False)
        else:
            # print('input_text', input_text)
            input_text.normalize(_normalize_text_zh)
            # print('input_text', input_text)
            text_segs = SSML.chunk_text_with_breaks(input_text, limit=60, language_type='zh', debug=False)
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


    def chunk_wavs_vad(self, wav_16k=None, speech_timestamps=None, 
                    chunk_duration=10, max_duration=60, 
                    vad_thresholds=(0.50, 0.35, 0.25),
                    min_speech_ms=150, min_silence_ms=100):
        """
        返回 [(start_sec, end_sec), ...]；永不返回空列表。
        策略：
        1) 用多组 threshold 递减重试 VAD；
        2) 仍为空则回退到 [0, min(max_duration, 音频总长)]；
        3) 用简单等长切分保证下游不会越界。
        """
        if speech_timestamps is None and wav_16k is not None:
            from silero_vad import get_speech_timestamps
            # 限长，避免过长音频卡住
            wav_16k = wav_16k[: int(16000 * max_duration * 1.2)]

            # 递降阈值重试
            for thr in vad_thresholds:
                st = get_speech_timestamps(
                    wav_16k, self.vad_model, return_seconds=True,
                    threshold=thr,
                    min_speech_duration_ms=min_speech_ms,
                    min_silence_duration_ms=min_silence_ms
                )
                if len(st) > 0:
                    speech_timestamps = st
                    print_once(f'| VAD detected {len(st)} segments with threshold={thr}')
                    break
            else:
                speech_timestamps = []

        # 取有效起止，并按 chunk_duration 等长切分
        start = max(0.0, float(speech_timestamps[0]['start']))
        end   = min(float(speech_timestamps[-1]['end']), float(max_duration))

        # 保护：end 可能被 clamp 后 <= start
        if end <= start:
            total_dur = (len(wav_16k) / 16000.0) if wav_16k is not None else float(max_duration)
            end = max(start + 0.1, min(total_dur, max_duration))

        # 生成等长切分
        offs = []
        cur = start
        while cur < end - 1e-6:
            nxt = min(end, cur + float(chunk_duration))
            offs.append((cur, nxt))
            if nxt == cur:  # 罕见保护：避免死循环
                break
            cur = nxt
        return offs


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

        chunk_wav_offsets = self.chunk_wavs_vad(wav_16k, chunk_duration=10, max_duration=max_ref_duration)
        wav = wav[int(chunk_wav_offsets[0][0] * self.sr): int(chunk_wav_offsets[-1][-1] * self.sr)]
        wav_16k = wav_16k[int(chunk_wav_offsets[0][0] * 16000): int(chunk_wav_offsets[-1][-1] * 16000)]

        def _process_alignment(alignment_tokens):
            ''' 将aligner_lm的输出进行处理，得到音素、声调、持续时间 '''
            ph_ref, tone_ref, dur_ref, _ = split_ph_timestamp(deepcopy(alignment_tokens))
            ph_ref = torch.Tensor(ph_ref)[None].to(self.device)
            tone_ref = torch.Tensor(tone_ref)[None].to(self.device)
            if dur_ref.sum() < prompt_max_frame:
                dur_ref[-1] += prompt_max_frame - dur_ref.sum()
            elif dur_ref.sum() > prompt_max_frame:
                len_diff = dur_ref.sum() - prompt_max_frame
                while True:
                    for i in range(len(dur_ref)):
                        dur_ref[i] -= 1
                        len_diff -= 1
                        if len_diff == 0:
                            break
                    if len_diff == 0:
                        break
            return ph_ref, tone_ref, dur_ref
        
        if self.use_old_aligner:

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
                    for i in tqdm(range(1024)):
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
            with torch.inference_mode():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    whisper_wav = torch.from_numpy(wav_16k)[None].to(self.device)
                    whisper_wav = whisper_wav[:, :whisper_wav.shape[-1] // 1280 * 1280]
                    prompt_max_frame = whisper_wav.shape[-1] // 160 // 8 * 8
                    token = torch.LongTensor([798])[None, :].to(self.device)
                    token = self.aligner_lm.inference(
                        whisper_wav, token, topk=1, temperature=0.7,
                        max_new_tokens=16384, eos_idx=799, use_tqdm=self.use_tqdm
                    )
            alignment_tokens = token[0]
            ph_ref, tone_ref, dur_ref = _process_alignment(alignment_tokens)

        # print(f"{ph_ref = }")
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
                merged_ph_tokens = map_phone_to_tokendict({'txt_token': ph_ref, 'tone': tone_ref}, pad_bos_eos=False)
                try:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        dur_start_pos = self.dur_model.prefill(merged_ph_tokens, dur_ref.to(self.device))
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

            ret = {
                'text_ref': text,
                'ph_ref': ph_ref.cpu(),
                'tone_ref': tone_ref.cpu(),
                'dur_ref': dur_ref.cpu(),
                'mel2ph_ref': mel2ph_ref.cpu(),
                'vae_latent': vae_latent.cpu(),
            }
            if self.use_old_dur:
                ret['incremental_state_dur_prompt'] = incremental_state_dur_prompt
                ret['ctx_dur_tokens'] = ctx_dur_tokens
            else:
                ret['dur_start_pos'] = dur_start_pos

            return ret

    def _get_spk_open_id_cached(self, device):
        """缓存并返回 <SPK> 的 token id（标量 LongTensor）。要求 <SPK> 是单一 token。"""
        if not hasattr(self, "_spk_open_id_cpu"):
            tid = self.dit_text_tokenizer.convert_tokens_to_ids("<SPK>")
            need_fallback = (
                tid is None
                or (hasattr(self.dit_text_tokenizer, "unk_token_id") and tid == self.dit_text_tokenizer.unk_token_id)
            )
            if need_fallback:
                ids = self.dit_text_tokenizer.encode("<SPK>")
                if not isinstance(ids, (list, tuple)) or len(ids) != 1:
                    raise RuntimeError(f"<SPK> 未被注册为单 token，encode 返回：{ids}")
                tid = ids[0]
            self._spk_open_id_cpu = torch.tensor(tid, dtype=torch.long)  # 常驻 CPU
        return self._spk_open_id_cpu.to(device=device, non_blocking=True)


    @torch.no_grad()
    def _build_txt_spk_mask_strict(self, txt_tokens: torch.Tensor, txt_mask: torch.Tensor,
                                seg_spk_masks, device, raw_texts=None):
        """
        段级 spk_mask (1-based: 1..K) -> token 级 mask：
        - <SPK> 作为段起点；段内正文 token = 段级 spk_id(1..K)；
        - PAD 与 <SPK> 本体 = 0。
        返回：txt_spk_mask [B, T] (long)
        """
        if seg_spk_masks is None:
            raise ValueError("[批次] 缺少 spk_mask。")

        B, T = txt_tokens.shape
        spk_tok = int(self._get_spk_open_id_cached(device).item())
        starts_mask = (txt_tokens == spk_tok)               # [B,T]
        starts_cnt  = starts_mask.sum(1)                    # [B]
        starts_cnt_cpu = starts_cnt.to("cpu").tolist()
        for b in range(B):
            if starts_cnt_cpu[b] == 0:
                rt = (raw_texts[b] if (raw_texts is not None and b < len(raw_texts)) else "<hidden>")
                raise ValueError(f"[样本 {b}] 未找到任何 '<SPK>'，文本片段：{str(rt)[:120]}")

        # 段索引：段前为 -1，第一段为 0，第二段为 1，……
        seg_index = torch.cumsum(starts_mask.to(torch.int64), dim=1) - 1   # [-1,0,1,...]
        pos = torch.arange(T, device=device, dtype=torch.int64).unsqueeze(0).expand(B, T)
        last_start = torch.cummax((starts_mask * (pos + 1)).to(torch.int64), dim=1)[0] - 1
        has_started = seg_index >= 0
        valid = has_started & (pos >= (last_start + 1)) & txt_mask.bool()

        max_N = int(starts_cnt.max().item())
        # 第 0 列保留为 0（哨兵列）：段前 / 非正文 / PAD / <SPK> 本体 都会落到 0
        spk_ids_pad = torch.zeros((B, max_N + 1), dtype=torch.long, device=device)

        for b in range(B):
            entry = seg_spk_masks[b]
            seg_ids = entry.tolist() if hasattr(entry, "tolist") else list(entry)  # 期望 1-based: 1..K
            N = starts_cnt_cpu[b]
            if len(seg_ids) != N:
                rt = (raw_texts[b] if (raw_texts is not None and b < len(raw_texts)) else "<hidden>")
                print(f"[warn] 样本 {b}: <SPK> 次数({N}) 与 spk_mask 段数({len(seg_ids)}) 不一致；文本：{str(rt)[:100]}")
            if N > 0:
                spk_ids_pad[b, 1:N+1] = torch.as_tensor(seg_ids, dtype=torch.long, device=device)

        seg_index_shift = torch.clamp(seg_index + 1, min=0, max=max_N)
        out = torch.gather(spk_ids_pad, 1, seg_index_shift)  # 不再 +1：保持 1..K；0 留给 PAD/<SPK>/无效
        out = out * valid.long()                             # 只在正文 token 生效
        out = out.masked_fill(~txt_mask.bool(), 0)           # PAD=0
        return out


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
                        return_timestamp,
                        timestamp_postprocess,
                        use_sa_frontend,
                        time_step,
                        w_all, w_txt, w_cap, w_ref, w_spk,   # <<< 新增 w_spk
                        timestep_annealing_w):


        # —— 以下与“参考版”一致：空文本处理、G2P、SSML refine、时长预测、断句/停顿、时间戳、mel2ph —— #
        if text.text_str.strip() == '':
            ph_pred_lst[t_i] = []
            tone_pred_lst[t_i] = []
            wav_pred_[t_i] = np.zeros(int(0.16 * self.sr))
            if return_timestamp:
                words_timestamps[t_i] = {'words': [], 'timestamps': []}
            sil_pad_lst[t_i] = (text.pause_at_start, text.pause_at_end)
            return

        with Timer('G2P', enable=profile):
            if not use_sa_frontend:
                with model_lock(self.lock):
                    ph_pred, tone_pred = self.g2p(text.text_str)
            else:
                from modules.tts.frontend_lm.sa_frontend import call_sa_frontend
                sa_ret = call_sa_frontend(text.sa_ssml_str, debug=0)
                if sa_ret is None:
                    print(f'文本段落{t_i}/{len_text_segs}不合法，跳过')
                    return
                text_sa, ph_tokens, tone_tokens, alignment_sa = sa_ret
                new_text = SSML(text_sa)
                new_text.rate = text.rate
                new_text.pause_at_start = text.pause_at_start
                new_text.pause_at_end = text.pause_at_end
                text = new_text
                ph_pred = self.ling_dict['phone'].encode(' '.join(ph_tokens))
                ph_pred = torch.LongTensor(ph_pred)[None].to(self.device)
                tone_pred = self.ling_dict['tone'].encode(' '.join(tone_tokens))
                tone_pred = torch.LongTensor(tone_pred)[None].to(self.device)

        if not use_sa_frontend:
            ph_pred, tone_pred, ph2word = self.refine_ph_tone(text, ph_pred, tone_pred)

        with Timer('Duration Prediction', enable=profile):
            if self.use_old_dur:
                last_dur_token = ctx_dur_tokens[:, -1:]
                last_dur_pos = last_dur_pos_prompt
                txt_len = ph_pred.shape[1]
                dur_spk_pos_ids_flat = range(last_dur_pos, last_dur_pos + txt_len)
                dur_spk_pos_ids_flat = torch.LongTensor([dur_spk_pos_ids_flat]).to(self.device)
                last_dur_pos = last_dur_pos + txt_len
                with model_lock(self.lock):
                    dur_pred = self.dur_model.infer(
                        ph_pred, {'tone': tone_pred}, None, None, None,
                        incremental_state=incremental_state_dur_prompt,
                        first_decoder_inp=last_dur_token,
                        spk_pos_ids_flat=dur_spk_pos_ids_flat, use_tqdm=False
                    )
                dur_pred = dur_pred - 1
            else:
                merged_ph_tokens = map_phone_to_tokendict({'txt_token': ph_pred, 'tone': tone_pred}, pad_bos_eos=False)
                with model_lock(self.lock):
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        dur_pred = self.dur_model.inference(
                            txt_tokens=merged_ph_tokens,
                            start_pos=dur_start_pos,
                            temperature=dur_disturb, use_tqdm=self.use_tqdm
                        )

            dur_pred = torch.round(dur_pred / text.rate).int()
            dur_pred = dur_pred.clamp(0, self.hp_dur_model.get('dur_code_size', self.hp_dur_model.get('dur_max_value', 128)) - 1)
            if t_i < len_text_segs - 1:
                dur_pred[:, -1] = dur_pred[:, -1] + 32
            else:
                dur_pred[:, -1] = dur_pred[:, -1].clamp(32, 80)

            if self.use_old_dur:
                dur_disturb_choice = (torch.rand_like(dur_pred.float()) > 0.5).float()
                dur_disturb_r = 1 + torch.rand_like(dur_pred.float()) * dur_disturb
                dur_pred = dur_pred * dur_disturb_r * dur_disturb_choice + \
                        dur_pred / dur_disturb_r * (1 - dur_disturb_choice)
                dur_pred = torch.round(dur_pred * dur_alpha).clamp(0, 127)
            for sil_token in [148, 153, 166, 145]:
                dur_pred[ph_pred==sil_token] = dur_pred[ph_pred==sil_token].clamp_min(32)
            for sil_token in [163, 165]:
                dur_pred[ph_pred==sil_token] = dur_pred[ph_pred==sil_token].clamp_min(16)
            dur_pred[:, 0] = 8
            if t_i == 0:
                dur_pred[:, 0] = 8

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
                    if ph_pred[:, -1].item() in [148, 163, 166, 153, 165, 147]:
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

            ph_pred_lst[t_i] = self.ling_dict['phone'].decode(ph_pred.squeeze().cpu().numpy()).split(' ')
            tone_pred_lst[t_i] = self.ling_dict['tone'].decode(tone_pred.squeeze().cpu().numpy()).split(' ')

            dur_sum = dur_pred.sum()
            vqs = hparams.get('vq_stride', 8)
            npad = vqs - dur_sum % vqs
            if npad < vqs:
                dur_pred[:, -1] += npad


            tag = f"seg {t_i+1}/{len_text_segs}"
            self._print_ph_dur_debug(tag, ph_pred, dur_pred)
            seg_total_cs = int(dur_pred.sum().item())
            print(f"[PH/DUR DEBUG][{tag}] ph_len={ph_pred.shape[1]}, dur_len={dur_pred.shape[1]}, "
                    f"sum_dur={seg_total_cs} cs ≈ {seg_total_cs/100:.3f}s")

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

            mel2ph_pred = self.length_regulator(dur_pred).to(self.device)

        # ====== 仅此处与“参考版”不同：caption / text / spk_mask ======
        # Caption：前缀 speaker 1:
        def run_caption_encoder(captions, device):
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
            return encoder_hidden_states, attention_masks

        with model_lock(self.lock):
            caption_str = f"speaker 1: {text_ref + text.text_str}"
            caption_embs, caption_mask = run_caption_encoder([caption_str], self.device)
        caption_embs = caption_embs * caption_mask[..., None]
        caption_lens = caption_mask.sum(-1)

        with Timer('DiT target speech generation', enable=profile):
            # Prepare duration token 
            mel2ph_pred = torch.cat((mel2ph_ref, mel2ph_pred+ph_ref.size(1)), dim=1)
            mel2ph_pred = mel2ph_pred[:, :mel2ph_pred.size(1)//self.fm*self.fm].repeat(6, 1)
            target_size = mel2ph_pred.size(1)//4

            if hparams.get('use_sparse_dur', False):
                dur_pred_all = torch.cat([dur_ref, dur_pred], dim=1)
                mel2ph_sparse = compute_mel2aug_from_dur(
                    dur_pred_all.int().squeeze().cpu().numpy().tolist(),
                    gap_mode=hparams.get('sparse_dur_mode', 'proportional'),
                    gap_frames=hparams.get('sparse_dur_frames', 4),
                    gap_alpha=hparams.get('sparse_dur_alpha', 0.2),
                    min_keep=hparams.get('sparse_dur_min_keep', 1),
                    keep_ratio=hparams.get('sparse_dur_keep_ratio'),
                    symmetric=hparams.get('sparse_dur_symmetric', True),
                )
                mel2ph_sparse = torch.stack([mel2ph_sparse] * 6).to(self.device)
                mel2ph_sparse = mel2ph_sparse[:, :mel2ph_pred.shape[1]]

            ctx_mask = torch.ones_like(vae_latent[:, :, 0:1])
            lat = F.pad(vae_latent, (0, 0, 0, target_size - vae_latent.size(1)), mode='constant', value=0)
            ctx_mask = F.pad(ctx_mask, (0, 0, 0, target_size - ctx_mask.size(1)), mode='constant', value=0)
            
            # Text：在原 text_ref + text.text_str 前加 "<SPK> "
            prompt_text = "<SPK> " + (text_ref + text.text_str)
            text_inputs = self.dit_text_tokenizer(prompt_text, padding=True, return_tensors='pt').to(self.device)
            tag = f"seg {t_i+1}/{len_text_segs}"
            self._print_dit_inputs_debug(tag, caption_str, prompt_text, text_inputs=text_inputs)
            txt_tokens = text_inputs['input_ids']
            txt_mask = text_inputs['attention_mask'].bool()
            txt_tokens[~txt_mask] = self.cfg_mask_text_token

            # Disable the English tone (set them to 3)
            ph_seq = torch.cat((ph_ref, ph_pred), dim=1)
            tone_seq = torch.cat((tone_ref, tone_pred), dim=1)
            en_tone_idx = ~((tone_seq == 4) | ( (11 <= tone_seq) & (tone_seq <= 15)) | (tone_seq == 0))
            tone_seq[en_tone_idx] = 3
            
            # 1) text tokens（txt）与 phone/tone（ph）
            # 先得到 cfg 掩码占位
            txt_mask_token_block = torch.full(txt_tokens.size(), self.cfg_mask_text_token, device=self.device)
            ph_mask_token_block  = torch.full(ph_seq.size(), self.cfg_mask_token_phone, device=self.device)
            tone_mask_token_block= torch.full(tone_seq.size(), self.cfg_mask_token_tone,  device=self.device)

            # 六路：all, txt, cap, ref, spk, uncond
            # - all: txt+ph+cap+ref+spk 全开
            # - txt: txt+ph，其他关（spk 关）
            # - cap: 仅 caption
            # - ref: 仅 reference（lat_ctx）
            # - spk: 仅 spk
            # - uncond: 全关
            txt_tokens = torch.cat([
                txt_tokens,           # all
                txt_tokens,           # txt
                txt_mask_token_block, # cap
                txt_mask_token_block, # ref
                txt_mask_token_block, # spk
                txt_mask_token_block  # uncond
            ], dim=0)
            txt_mask = torch.cat([txt_mask] * 6, dim=0)

            ph_seq = torch.cat([
                ph_seq,               # all
                ph_seq,               # txt
                ph_mask_token_block,  # cap
                ph_mask_token_block,  # ref
                ph_mask_token_block,  # spk
                ph_mask_token_block   # uncond
            ], dim=0)

            tone_seq = torch.cat([
                tone_seq,             # all
                tone_seq,             # txt
                tone_mask_token_block,# cap
                tone_mask_token_block,# ref
                tone_mask_token_block,# spk
                tone_mask_token_block # uncond
            ], dim=0)

            # 2) caption：all 与 cap 开；其它关
            caption_embs = torch.cat([
                caption_embs,                      # all
                torch.zeros_like(caption_embs),    # txt
                caption_embs,                      # cap
                torch.zeros_like(caption_embs),    # ref
                torch.zeros_like(caption_embs),    # spk
                torch.zeros_like(caption_embs)     # uncond
            ], dim=0)
            caption_lens = torch.cat([caption_lens, caption_lens,
                                    caption_lens, caption_lens.new_zeros(caption_lens.shape),
                                    caption_lens.new_zeros(caption_lens.shape),
                                    caption_lens.new_zeros(caption_lens.shape)], dim=0).to(torch.long)

            # 3) reference(lat_ctx)：all 与 ref 开；其它关
            lat = torch.cat([
                lat,                               # all
                torch.zeros_like(lat),             # txt
                torch.zeros_like(lat),             # cap
                lat,                               # ref
                torch.zeros_like(lat),             # spk
                torch.zeros_like(lat)              # uncond
            ], dim=0)
            ctx_mask = torch.cat([ctx_mask] * 6, dim=0)

            # 4) spk_ids（1-based；0 关）：all 与 spk 开；其它关
            spk_base = torch.ones_like(ph_seq, dtype=torch.long, device=self.device)  # 单说话人：全 1
            spk_ids = torch.cat([
                spk_base,                          # all
                torch.zeros_like(spk_base),        # txt
                torch.zeros_like(spk_base),        # cap
                torch.zeros_like(spk_base),        # ref
                spk_base,                          # spk
                torch.zeros_like(spk_base)         # uncond
            ], dim=0)

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
                # === NEW: 传入 phone-level 说话人 id（1..K；0 为 PAD/无条件）
                "spk_ids": spk_ids,
            }
            if hparams.get('use_sparse_dur', False):
                inputs['mel2ph_sparse'] = mel2ph_sparse


            with model_lock(self.lock):
                with torch.autocast(device_type='cuda', dtype=self.precision):
                    x = self.dit.inference(
                        inputs, timesteps=time_step, 
                        seq_cfg_w=[w_all, w_txt, w_cap, w_ref, w_spk],
                        timestep_annealing_w=timestep_annealing_w
                    )

                    print(f"DiT output latent: {x.shape}, min={x.min().item():.3f}, max={x.max().item():.3f}, mean={x.mean().item():.3f}")

        with Timer('WavVAE decode', enable=profile):
            x[:, :vae_latent.size(1)] = vae_latent
            with model_lock(self.lock):
                with torch.autocast(device_type='cuda', dtype=self.precision):
                    wav_pred = self.vae.decode(x)[0,0].to(torch.float32)
        with Timer('Post-processing', enable=profile):
            hop_size = self.hp_vae['hop_size']
            vae_stride = self.hp_vae['vae_stride']
            wav_pred = wav_pred[vae_latent.size(1)*vae_stride*hop_size:]
            if wav_pred.abs().max() > 1:
                print('Wav amplitude exceed 1, clip it.')
                wav_pred = wav_pred / (wav_pred.abs().max())
            wav_pred = wav_pred.cpu().numpy()
            wav_pred_[t_i] = wav_pred

    def _print_dit_inputs_debug(self, tag: str, caption_str: str, prompt_text: str, text_inputs=None, max_ids: int = 64):
        """
        打印送入 DiT 的 caption 与 text：
        - 原始字符串
        - 可见 token 数（去掉 pad）
        - 前 max_ids 个 token id
        - 反解码文本（不跳过 special tokens，便于观察 <SPK> 等）
        通过设置环境变量 MEGA_PRINT_DIT_TEXT=0 可关闭打印。
        """

        # --- Caption ---
        try:
            cap_inputs = self.caption_tokenizer([caption_str], padding=True, return_tensors="pt")
            cap_ids  = cap_inputs.input_ids[0]
            cap_mask = cap_inputs.attention_mask[0].bool()
            cap_vis  = cap_ids[cap_mask]
            cap_head = cap_vis[:max_ids].tolist()
            print(f"[DiT INPUT][{tag}] caption='{caption_str}'")
            print(f"[DiT INPUT][{tag}] caption_token_len={cap_vis.numel()}  ids(head {len(cap_head)}): {cap_head}{'...' if cap_vis.numel()>max_ids else ''}")
            try:
                cap_dec = self.caption_tokenizer.decode(cap_vis, skip_special_tokens=False)
                print(f"[DiT INPUT][{tag}] caption_decoded='{cap_dec}'")
            except Exception as e:
                print(f"[DiT INPUT][{tag}] caption_decode_error: {e}")
        except Exception as e:
            print(f"[DiT INPUT][{tag}] caption_tokenize_error: {e}")

        # --- Text / Prompt ---
        try:
            if text_inputs is None:
                txt_inputs = self.dit_text_tokenizer(prompt_text, padding=True, return_tensors="pt")
            else:
                txt_inputs = text_inputs  # 来自 GPU，也能安全搬到 CPU 打印
            ids  = txt_inputs["input_ids"][0].detach().to("cpu")
            mask = txt_inputs["attention_mask"][0].detach().to("cpu").bool()
            vis  = ids[mask]
            head = vis[:max_ids].tolist()

            print(f"[DiT INPUT][{tag}] text='{prompt_text}'")
            print(f"[DiT INPUT][{tag}] text_token_len={vis.numel()}  ids(head {len(head)}): {head}{'...' if vis.numel()>max_ids else ''}")
            try:
                dec = self.dit_text_tokenizer.decode(vis, skip_special_tokens=False)
                print(f"[DiT INPUT][{tag}] text_decoded='{dec}'")
            except Exception as e:
                print(f"[DiT INPUT][{tag}] text_decode_error: {e}")
        except Exception as e:
            print(f"[DiT INPUT][{tag}] text_tokenize_error: {e}")


    def _print_ph_dur_debug(self, tag, ph_pred: torch.Tensor, dur_pred: torch.Tensor, timestep: float = 0.01):
        """
        打印 ph 与其对应的预测 dur（单位：centiseconds=0.01s），并给出每个音素的开始时间。
        """
        # 解码成可读的音素列表
        ph_list = self.ling_dict['phone'].decode(ph_pred.squeeze().detach().cpu().numpy()).split(' ')
        # dur_pred 是整数 “百份之一秒” 计数
        dur_arr = dur_pred.squeeze().detach().cpu().numpy().astype(np.int64).tolist()

        if len(ph_list) != len(dur_arr):
            print(f"[PH/DUR DEBUG][{tag}] ⚠️ 长度不一致: ph={len(ph_list)}, dur={len(dur_arr)}")
            L = min(len(ph_list), len(dur_arr))
            ph_list = ph_list[:L]
            dur_arr = dur_arr[:L]

        # 逐个音素的起始时间（同样以 0.01s 为步长）
        starts_cs = np.cumsum([0] + dur_arr[:-1]).tolist()
        total_cs = sum(dur_arr)

        print(f"[PH/DUR DEBUG][{tag}] tokens={len(ph_list)}, total_dur={total_cs} cs ≈ {total_cs * timestep:.3f}s")
        for i, (p, d, st) in enumerate(zip(ph_list, dur_arr, starts_cs)):
            print(f"  #{i:03d}  ph={p:>8}  dur={d:4d} cs  ({d * timestep:7.3f}s)  start={st:6d} cs  ({st * timestep:7.3f}s)")


    def forward(self, resource_context, input_text, time_step, w_all, w_txt, w_cap, w_ref, w_spk,
                speech_rate=1, timestep_annealing_w=(1.0, 0.0, 1.0), return_timestamp=True, timestamp_postprocess=False, 
                return_format='wav', custom_ph_table=None, dur_disturb=0.1, dur_alpha=1.0, 
                num_parallel_workers=5, use_sa_frontend=True, **kwargs):
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
                text_segs = self.preprocess_text(input_text, ph_replace_table, use_sa_frontend)

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
                                    "return_timestamp": return_timestamp,
                                    "timestamp_postprocess": timestamp_postprocess,
                                    "use_sa_frontend": use_sa_frontend,
                                    "time_step": time_step,
                                    "w_all": w_all,
                                    "w_txt": w_txt,
                                    "w_cap": w_cap,
                                    "w_ref": w_ref,
                                    "w_spk": w_spk,
                                    "timestep_annealing_w": timestep_annealing_w,
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

import os
import glob
from pathlib import Path

import torch
from tqdm import tqdm

from utils.commons.os_utils import kill_void
from utils.text.ssml_utils import SSML
from utils.audio.io import save_wav_bytes


if __name__ == '__main__':
    if os.path.isfile('.env.local'):
        from dotenv import load_dotenv
        load_dotenv('.env.local')

    # dit_exp_name = 'checkpoints/250818_megatts3_dit_v2'
    # dit_exp_name = 'checkpoints/250826_megatts3_dit_v2_32layer'
    dit_exp_name = '/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/checkpoints/250921_megatts3_test'
    # dit_exp_name = 'checkpoints/250827_megatts3_dit_v2_32layer_16ca_sparsedur'

    infer_ins = MegaTTS3DiTInfer(
        device=f'cuda:0',
        dit_exp_name=dit_exp_name,
        dur_exp_name='checkpoints/250826_dur_lm',
        frontend_exp_name='checkpoints/250823_lm_mfa_seq2seq_small_wavlmlarge',
        use_old_aligner=False,
        use_old_dur=False,
        max_ref_duration=40,
    )
    kill_void()

    # ===== NaN/Inf 参数体检（最小侵入）=====
    def _check_nan_inf(module, tag):
        bad = False
        for n, p in module.named_parameters():
            if p is not None and p.is_floating_point():
                if torch.isnan(p).any() or torch.isinf(p).any():
                    print(f"[NaN/Inf] {tag}.{n}")
                    bad = True
                    break
        if not bad:
            print(f"[OK] {tag} 无 NaN/Inf")
        return bad

    _bad = False
    for _tag, _m in [
        ("DiT", getattr(infer_ins, "dit", None)),
        ("VAE", getattr(infer_ins, "vae", None)),
        ("DurationLM", getattr(infer_ins, "dur_model", None)),
        ("Aligner", getattr(infer_ins, "aligner_lm", None)),
        ("CaptionEncoder", getattr(infer_ins, "caption_encoder", None)),
    ]:
        if _m is not None:
            _bad = _check_nan_inf(_m, _tag) or _bad

    if _bad:
        print("| 检测到 NaN/Inf 参数，已中止推理。请检查对应 checkpoint。")
        sys.exit(1)
    # ===== 体检结束 =====


    # time_step = 100
    time_step = 32

    w_all = 3.0
    w_txt = -1.0
    w_cap = 0.0
    w_ref = 0.0
    w_spk = 0.0  

    if True:
        # wav_path = '/mnt/bn/sa-ag-data/liruiqi/code/MegaTTS3-inference-vc/infer_out/timbre_prompt/情绪音色供给4.17/活力姐姐-MEGA_87x6TAJSqzN.wav'
        wav_path = 'user/prompts/mega_eval_prompt0731/0_vocal.wav'
        # wav_path = 'user/prompts/mega_eval_prompt0731/9_vocal.wav'
        # wav_path = 'prompts/2025-07-27_211109_509_vocal.wav'
        with open(wav_path, 'rb') as file:
            file_content = file.read()
        print(f"| Start processing {wav_path}")
        resource_context = infer_ins.preprocess(file_content, wav_path)
        # torch.save(resource_context, 'infer_out/resource_fanxian.pt')
    else:
        resource_context = torch.load('/mnt/bn/sa-ag-data/liruiqi/code/MegaTTS3-inference/infer_out/resource_fanxian.pt')


    test_text = [
        # '老妹儿还是那句话，我说你行，你就行，不行也得行。就老妹儿手里拿这个断货王升级版第二代小蜜丸，你一定要去体验体验一次，让你回味无穷。体验一次，你肯定会回来找我回购的，肯定会回来感谢我的。咱经常出门出差，交朋会友的这个老妹儿，真心推荐给你啊。你别担心老妹已经替你试过了。视频左下角安排了同款。抓紧去拼手速'
        # '你好，请问您为交友软件充过钱吗？当然了，想和美女聊天不就是要充钱吗？但是我现在是真没钱充会员了。那你可以试试这个软件啊，上线就送七天会员。真的吗？还有这种好事？当然是真的了，只要你是通过本条视频下方链接下载的，就会给你匹配同城的单身美女，可以语音。可以，视频还能看到电话号码和位置呢，你要是不信的话，就点击下方链接下载试试吧！赶紧点击下方链接下载吧！'
        # '都官方直発',
        # '梅花（Titoni）瑞士手表男士机械表经典金色腕表生日礼物 宇宙系列 钢带表盘40MM 797 G-DB-306',
        # '这个抹布你千万别嫌它便宜，我怕你用完舍不得丢，因为实在是太好用了&#34;不沾油厨房抹布 &#34;厨房清洁 ...更多好用好用好用好用好用好用',
        # '入手闪铸AD5X太值了！它打印速度超快，600mm/s的速度，加速度达20000mm/s²，大大提升效率。支持四色打印，多色TPU柔性线材也能驾驭，色彩超绚丽。自带全自动调平校准，一键启动超省心。全金属CoreXY框架，稳定又精准，新手也能轻松上手，太好用啦！ '
        # '三分靠长相，七分靠打扮，女人的气质是打扮出来的。衣服要是穿的对，显得年轻好几岁。就这款超好看的蚂蚁腰防晒衬衣，穿在身上既优雅又有气质，还特别的舒服得劲，走到哪都是焦点。新品上市，厂家为了快速出一波店铺的销量，给咱姐妹们炸一波福利。今天下单还包邮送到家。姐妹们放心带回家试穿，喜欢满意就留下，不喜欢不满意，您直接退回来。精选轻薄透气面料，触感丝滑，穿上自带清凉感，仿佛给肌肤敷上冰膜，拒绝夏日闷热黏腻。版型设计独具匠心，巧妙运用公主线剪裁，勾勒出盈盈一握的蚂蚁腰，轻松展现女性曼妙身姿，谁穿谁是 “腰精”。短款衣长适配多种下装，搭配高腰裤或裙子，瞬间优化身材比例，大长腿既视感轻松拿捏。宽松的版型，胖瘦姐妹都能穿，简约的同时又不失精致。无论是日常出街、办公室久坐，还是户外游玩，这件防晒衬衣都能让你在防晒的同时，时尚感拉满，成为夏日里的靓丽风景线！不挑年龄不挑身材，能从30岁穿到五六十岁。关键价格不贵，还包邮送到家。但是福利数量不多了，喜欢的姐妹下方小黄车赶紧拼手速了。',
        # '这可是国际救援中心认证的"神器"！不用插电，晒两三个小时就能亮40个小时，露营、停电应急都能用，角度亮度随便调，还有爆闪求救模式！想省钱又图安心的，赶紧点我头像进橱窗，多备几个准没错！'
        # '家人们，来给评评理！上门女婿被老婆当众扇巴掌赶出门，人家愣是没吵没闹，抽根烟就给500公里外的亲弟弟打电话：&#34;弟，快来接我回家！&#34;这事儿，搁谁身上能忍？\n \n挨打这位李哥，打完电话扭头回屋，一声不吭就收拾行李。衣柜里的衣服哗啦往包里塞，摸着脸上火辣辣的巴掌印，对着镜子苦笑——这段婚姻，算是彻底凉透了。\n \n说起来，李哥和老婆可是实打实的校园爱情。大学四年腻歪得不行，毕业却卡在了&#34;去留&#34;难题上。老婆哭着说：&#34;我家就我一个闺女没嫁，总不能留爸妈孤苦伶仃吧？&#34;李哥纠结得不行，给亲弟弟打电话倒苦水。弟弟一拍胸脯：&#34;哥你放心！家里有我呢，爸妈绝对照顾得妥妥当当！&#34;\n \n就这么着，李哥留在老婆的城市，成了上门女婿。刚开始小两口日子过得挺甜蜜，李哥脾气温和又顾家，妥妥的&#34;模范丈夫&#34;，家里渐渐形成女主外男主内的模式。可谁能想到，岳父越看他越不顺眼！\n \n尤其是李哥事业走下坡路那阵儿，岳父逢年过节就冷嘲热讽：&#34;天天在家待着，吃我女儿软饭！&#34;起初李哥还能忍，后来连曾经心疼他的老婆，也跟着岳父阴阳怪气。有次岳父当着全家面羞辱他，李哥刚回怼两句，老婆抬手就是一巴掌！\n \n这巴掌下去，李哥彻底清醒了。任凭老婆哭着道歉、拽着衣角求原谅，他头也不回拉着行李箱就走。关上门那刻，后视镜里老婆的身影越来越小，李哥心里五味杂陈——十几年感情，终究抵不过一巴掌的伤害。\n \n话说回来，家人们可得留个心眼！现在电费贵，天气还总出幺蛾子，强烈建议囤几个太阳能足球灯！这可是国际救援中心认证的&#34;神器&#34;！不用插电，晒两三个小时就能亮40个小时，露营、停电应急都能用，角度亮度随便调，还有爆闪求救模式！想省钱又图安心的，赶紧点我头像进橱窗，多备几个准没错！',
        # '我的老朋友，你真的很不简单,这段时间悄悄看着你，发现你跟人相处时，总是带着春风般的亲切劲儿，让人打心眼里觉得温暖。你对生活那股子热乎劲儿，谁看了都受感染。每天乐呵呵的，好像心里揣着小太阳，跟你待在一起，连心情都跟着亮堂起来。这种积极的日子态度，可不就是咱们身边的正能量担嘛.你这人的特点就是实在，有啥说啥，从不藏着掖着。就像一块干干净净的璞玉，看着朴实，却透着真诚的光。现在这世道多复杂啊，但你始终活得明白 ，该咋过就咋过，这份清醒劲太难得啦，有时候我忍不住琢磨。你平时没少读书吧，不然咋活得这么通透，肯定也经历过不少风风雨雨，不然面对事儿时，咋能这么不慌不忙呢，你这人缘儿为啥这么好？看看你做的事儿就知道了，别人对你好一分你对别人好五分；宁可自己吃点亏，也不愿欠人情。遇到困难从不喊苦喊累，咬咬牙就挺过去了；日子过得不攀比、不将就，独立又有骨气，这些优点搁谁身上不招人夸，跟你说句心里话，像你这么重情重义、踏实实过日子的人，福气肯定在后头呢。相信以后你的日子只会越来越顺溜，最后还想跟你念叨两句：咱们这辈子就像跑马拉松，如今已经跑了大半程啦，别总闷头往前冲，偶尔放慢脚步，看看路边的花花草草，感受感受生活里的小确幸，才不辜负这一路的风景呀\n',
        # '梅花（Titoni）瑞士手表男士机械表经典金色腕表生日礼物 宇宙系列 钢带表盘40MM 797 G-DB-306',
        # '浪琴（LONGINES）瑞士手表 名匠系列月相腕表 机械皮带男表L29194783 银色麦粒饰纹42.0 mm',
        # '浪琴（LONGINES）瑞士手表 嘉岚系列 石英钢带女表 L42091917 白色哑光24.0 mm',
        # '百丽（Belle）简约一字拖女商场同款编织休闲拖鞋A8Z1DBT4预售 米白-平跟款 37 (235mm)',
        # 'Keep动感单车 家用智能健身器材AI调阻自发电款C2 Lite 20斤飞轮组 白',
        # '蓝立哆 （Elydo）电动升降桌电脑桌办公书桌双电机站立式工作台学习桌H2 H2白色桌腿+苏丹象牙白色桌面 25mm加厚1.2*0.6m',
        # '追剧到凌晨眼睛干涩模糊？叶黄素冰敷眼罩 yyds！叶黄素深入养护，冰感缓解疲劳，360° 透气设计，敷完眼睛明亮清晰，速抢！',
        # '蓝立哆电动升降桌双电机电脑电竞桌书桌学习桌站立式工作台写字桌 H2e Pro平椭圆白腿+白色桌面 25mm加厚环保生态板1.2*0.6m',
        # '家里有猫的注意啦\n麦富迪猫罐头来咯\n它成幼猫通用\n是猫咪都爱的零食湿粮\n这款罐头采用浓汤三种肉配方\n还添加了鱼油\n营养丰富又美味\n能满足猫咪日常营养所需\n让猫咪吃得开心 主人更放心\n如果你还在为给猫咪选零食发愁\n那就别犹豫了\n赶紧给你家宝贝安排上吧',
        # '还在为视频播放烦恼不已吗？\n别再纠结\n开通我们的升级服务\n专业团队技术加持\n就像一场及时雨\n提升你的视频品质\n让你的内容无处不在\n解锁自由发布\n点击即可拥有',
        # '蔻驰（COACH）奢侈品TOTE女包单肩拉链托特包4455【品牌授权】520送女友礼物',
        # '我真庆幸买了西门子极净魔盒Plus洗碗机！16套大容量，再多餐具都能装下。微米蒸汽洗，55°C蒸汽提前软化顽渍，搭配46000pa超强水压，啥污渍都能洗净。超五星消杀认证，除菌率高达 99.9999%。智能抽湿烘干3.0，餐具干干爽爽。操作简单，还有智能语音控制，用着超省心，强烈推荐！ ',
        # '今天给大家分享一款超赞的耐克NIKE缓震篮球鞋，型号为乔丹AJ1 JORDAN1运动鞋553558 - 145紫白42.5。这双鞋的第一个卖点，是经典设计，怎么搭配都很潮。鞋面采用皮革和合成材质组合，结实又耐用。后跟内嵌式Air缓震技术，缓震效果一流。硬质橡胶外底，抓地力强，多种地面都能轻松驾驭。我穿着它打球、日常出行都很舒服，长时间走路脚也不累，真心推荐给大家！ ',
        # '家人们！手速快的先赚翻了，金刚网绣花防蚊门帘直接炸到8块9毛9还包邮！这价格我都怀疑厂家贴钱做活动！\n \n瞅瞅这配置——中间加宽磁条&#34;啪嗒&#34;一下自动吸合，比普通门帘关得严实，魔术贴往门框一贴就能用，租房搬家拆下来还能接着用！网眼细到蚊子苍蝇钻不进，牛津布包边使劲扯都不变形，木门铁门都能装！\n \n两百单库存眼瞅着要见底，这价格平时连半片网都买不到！刷到这条就是大数据在喊你薅羊毛！想要夏天不被蚊子追着咬的，赶紧戳小黄车，手慢真的要拍大腿！',
        # '结合全网对【宝路华（BULOVA）熊猫盘手表男石英表计时码表阿波罗飞行员登月同款赠原装带98K112 98K112】这款商品做一个70字左右的种草分享。呈现出来的只要文案部分，不要乱七八糟给我发一堆，不需要AI写作助手来提供，你直接给我生成出来就可以',
        # '<speak>假期"还不知道"带孩子去哪儿的，一定要来青山上遇东坡画剧展，馆内几十个裸眼3D打卡点，直接把课本里的内容演活了，历史知识秒变视觉盛宴，让孩子感受宋式风韵，体验点茶、挂画、制香、插画，还有桌游和趣味拼图可以玩哦，关键是亲子票一大一小仅需59.9，<phoneme ph="zhen3 de3" alphabet="py">真的</phoneme>超划算，链接就在左下角，刷到赶紧囤~</speak>',
        # '都现在了，谁还傻乎乎的刷题啊，高分通过的考生早把知识点浓缩成了这300个必考点，近5年法考考来考去，翻不出这几个命门，你花几百小时刷题，不如直接啃透这套学霸笔记，别看它只有这薄薄的一套，里面是大咖讲师根据官方教材总结成的核心考点，近五年卷子90%考点对标，零基础也能懂得底层破题思路，案例分析一眼看穿命题陷阱，搭配4节直播课，老师手把手拆解考点逻辑，不到一杯奶茶钱，印刷成本都不够，下单激活还有3000＋智能题库，赶紧点进直播间看看吧，晚了可真没有了',
        # '<speak>给所有花大钱买熊胆粉的朋友们道个歉对不起我们的活动来的太迟了为了广大干友们都能用得起这次啊我们厂家亲自下场送福利免费咨询福利带回家我们这款熊胆粉呢成分就三个字熊胆粉不加辅料不混杂其他成分现在点开视频下方链接不下载不分享不转发客服直接对接<phoneme alphabet="py" ph="gong1 chang3 zhi2 fa1">工厂直发</phoneme>让中老年朋友都能用得上机会可一定要把握住一大盒熊胆粉不是07的也不是体验装是一大盒正装还不赶紧点击视频下方链接去看看无论你有多忙啊我都建议你现在就打开下方获取这个优惠</speak>',
        # '<speak>我们要搬去香港了————————这种苹果床就30一套了————————户外帐篷18————————阳台茶桌就15了————————花架6块————————垃圾桶3块————————风扇就全部10块了————————这边还有些衣帽架全身镜啥的————————都是全新的————————有需要的进我直播间<phoneme alphabet="py" ph="ba5">吧</phoneme>————————</speak>',
        # '<speak>假期"还不知道"带孩子去哪儿的，一定要来青山上遇东坡画剧展<sub alias="兄弟们">xdm</sub>，<break time="5s"></break>馆内几十个裸眼3D打卡点，直接把课本里的内容演活了，历史知识秒变视觉盛宴，让孩子感受宋式风韵，体验点茶、挂画、制香、插画，还有桌游和趣味拼图可以玩哦，关键是亲子票一大一小仅需59.9，<phoneme ph="zhen3 de3" alphabet="py">真的</phoneme>超划算，链接就在左下角，刷到赶紧囤~</speak>',
        # '<speak>这一粒小钙片富含330mg 高含量钙,<break time="5s"></break></speak>',
        # '<speak> <break time="1s"></break></speak>',
        # '<speak><break time="5s"></break>算了吧，哪有这么好的事！</speak>',
        # '<speak><break time="1s"></break> <break time="1s"></break>两种<break time="1s"></break>方式来帮你训练</speak>',
        # '九十九元不是发两袋也不是发四袋是真的六袋家好这次活动真的是拼了买过的朋友也趁着活动多囤点没买过的朋友也要趁这个活动入手简直不要太划算了拿回家不管怎么做都不心疼链接就放在直播间千万不要错过啦'
        '1955-1980年出生的您注意了！您的2025新款超声波洗菜盆专属优惠，已送达！\r\n别划走！这份福利正等着您签收！如果这次您再错过，优惠名额将转给他人——但现在，它依然属于您！点击下方链接，立即锁定您的优惠！名额已发送至您的手机。今天不参与，资格即失效！无论多忙，只需几秒就能抢到手！无需转发！无需下载！看到即得！所见即所得！现在参与，最快明天到货！名额告急，错过后悔莫及！赶紧点击下方链接，抢占最后名额！',
        '你这首独奏弹得不错呦，我以为年轻人都在搞电子乐呢',
        # '想赚钱吗，点击下方链接，这款真的能赚钱的软件，你快来试试吧',
        # '现在点击下方链接，从源头厂家直接发货到您手中，没有任何的中间环节加价。厂家开启了绿色通道，点击下方链接给自己牙齿一次调理的机会吧。',
        # '你的手机是不是也是这样。刚买没多久就提示内存不足，手机卡到用不了，动不动就发烫死机，这是内存占用过大导致的，还有很多人不知道怎么清理手机垃圾，打开这个开关，就能一键清理，方法放到左下方了。需要的赶去去看看吧！'
    ]

    out_path = f'infer_out/tts/infer_once/{Path(dit_exp_name).stem}'

    pbar = tqdm(total=len(test_text))
    for i, input_text in enumerate(test_text):
        print('='*100)
        print(f'| Generating text [{i}]: {input_text}')
        output = infer_ins.forward(
            resource_context, input_text, time_step=time_step, 
            w_all=w_all, w_txt=w_txt, w_cap=w_cap, w_ref=w_ref, w_spk=w_spk
        )


        wav_bytes = output.wav_bytes
        input_text = SSML(input_text).text_str
        os.makedirs(out_path, exist_ok=True)
        save_wav_bytes(wav_bytes, f'{out_path}/{input_text[:20]}.wav')

        print(f'| Generation of text [{i}] succeed')
        pbar.update(1)


# CUDA_VISIBLE_DEVICES=5 python user/inference/megatts3_dit_v2_infer.py
