import os
import collections
import collections.abc
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))

import sys
import io
import re
import json
import glob
import math
import time
import random
import argparse
import threading
import traceback
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import List, Union, Dict, Any, Tuple, Optional
from utils.commons.tensor_utils import move_to_cpu, move_to_cuda

import numpy as np
import librosa
import torch
import torch.nn.functional as F
import soundfile as sf
import pyloudnorm as pyln
import whisper
from tqdm import tqdm
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from langdetect import detect as classify_language, LangDetectException
from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnNormalizer
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
from utils.commons.os_utils import handle_exacption, kill_void
from utils.commons.io import print_once
from utils.commons.import_utils import import_module_bystr, get_class_from_module
from utils.nn.ema import restore_ema

from modules.tts.ar_dur.commons.nar_tts_modules import LengthRegulator
from modules.tts.ar_dur.commons.align_ops import compute_mel2aug_from_dur
from modules.tts.scriptspeech.build_model_utils import DiTBuildModelMixinV2
import inspect  # 新增：用于安全探测 dur_model.inference 的签名


# =========================================
# Utils
# =========================================

def convert_to_wav_bytes(audio_binary: bytes) -> io.BytesIO:
    """任意格式→WAV（内存字节流）"""
    audio = AudioSegment.from_file(io.BytesIO(audio_binary))
    wav_bytes = io.BytesIO()
    audio.export(wav_bytes, format="wav")
    wav_bytes.seek(0)
    return wav_bytes

@contextmanager
def model_lock(lock: threading.Lock):
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


# =========================================
# Inference Class
# =========================================

class MegaTTS3DiTInfer(DiTBuildModelMixinV2):
    """双说话人参考 + phone-level spk 条件（按 mel2ph 扩展）；两份文本：raw(带 <S{sid}>...</S{sid}>)→DiT；clean(去标签)→SA 与 Dur。"""
    def __init__(
        self,
        device=None,
        dit_exp_name=None,
        dur_exp_name=None,            
        g2p_exp_name=None,
        frontend_exp_name=None,
        use_old_aligner=False,
        use_old_dur=False,
        max_ref_duration=40,
        use_tqdm=True,
        **kwargs
    ):
        self.sr = 24000
        self.fm = 8
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_tqdm = use_tqdm

        self.dit_exp_name = dit_exp_name
        self.use_old_dur = use_old_dur
        self.dur_exp_name = 'checkpoints/megatts3_wavdit/duration_lm' if use_old_dur else dur_exp_name
        self.use_old_aligner = use_old_aligner
        self.frontend_exp_name = 'checkpoints/megatts3_wavdit/aligner_lm' if use_old_aligner else frontend_exp_name
        self.wavvae_exp_name = 'checkpoints/1231_megatts3_wavvae_v3_25hz_kl001_fix4'

        self.build_model(self.device)

        # VAD/段落拼接相关
        self.max_silence_alive = 1.28
        self.max_ref_duration = max_ref_duration
        self.lock = threading.Lock()

    # ---------------- Models ----------------

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
            self.dur_model.precision = self.precision
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
            elif self.dur_model_type == 'dit':
                from modules.tts.scriptspeech.dit_dur import build_dur_model
                hp_dur_model = self.hp_dur_model = set_hparams(f'{self.dur_exp_name}/config.yaml', global_hparams=False)
                self.dur_model = build_dur_model(hp_dur_model)
                self.dur_model.hparams = {}
                self.dur_model.eval()
                load_ckpt(self.dur_model, self.dur_exp_name, 'model', strict=True, mmap=True)
                self.dur_model.to(self.device)
                
            if hasattr(self.dur_model, 'module'):  # 兼容 DDP/FSDP
                setattr(self.dur_model.module, 'precision', self.precision)
            setattr(self.dur_model, 'precision', self.precision)

    def _dur_predict_onepass_with_spk(
        self,
        chunk_items,
        resource_context,
        dur_disturb: float = 0.1,
        normalize_dur: bool = True
    ):
        """
        一步推完整目标段（2spk 兼容）：
        - 按出现顺序拼接所有 chunk 的 ph/tone
        - 构造与之等长的 spk_ids（phone-level）
        - 先用全局 ref 段 prefill，再一次性 inference
        - （可选）按“说话人 × {静音, 非静音}”在 log 域对齐到对应 ref 统计
        返回：{chunk_idx: Tensor[1, L_chunk]}，与逐段接口一致
        """
        import torch
        device = self.device
        compute_dtype = self.precision

        # ---------- 1) 汇总目标段 ph/tone + 对应 spk_ids ----------
        ph_all, tone_all, spk_ids_all = [], [], []
        lens = []
        for it in chunk_items:
            sid = int(it['sid'])
            ph  = it['ph'].to(device)
            tone= it['tone'].to(device)
            L   = ph.shape[1]
            lens.append(L)
            ph_all.append(ph)
            tone_all.append(tone)
            spk_ids_all.append(torch.full((1, L), sid, dtype=torch.long, device=device))

        ph_all        = torch.cat(ph_all,        dim=1)  # [1, T_all]
        tone_all      = torch.cat(tone_all,      dim=1)  # [1, T_all]
        spk_ids_all   = torch.cat(spk_ids_all,   dim=1)  # [1, T_all]

        # ---------- 2) 参考前缀 prefill（用全局 ref 段） ----------
        ref_tokens = map_phone_to_tokendict(
            {'txt_token': resource_context['ph_ref'].to(device),
            'tone':      resource_context['tone_ref'].to(device)},
            pad_bos_eos=False
        )
        with model_lock(self.lock), torch.autocast(device_type='cuda', dtype=compute_dtype):
            start_pos = self.dur_model.prefill(ref_tokens, resource_context['dur_ref'].to(device))

        # ---------- 3) 文本条件（若是 seq2seq） ----------
        needs_caption = (getattr(self, "dur_model_type", "lm") == "lm_seq2seq")
        caption_embs = None
        if needs_caption:
            ref_raw = resource_context.get('text_ref_raw', '')
            tgt_raw = ''.join([f"<S{int(it['sid'])}>{it['ch'].text_str}</S{int(it['sid'])}>" for it in chunk_items])
            caption_text = ref_raw + tgt_raw
            cap_inputs = self.caption_tokenizer([caption_text], padding=True, return_tensors="pt")
            cap_ids, cap_am = cap_inputs.input_ids.to(device), cap_inputs.attention_mask.to(device)
            with model_lock(self.lock), torch.autocast(device_type='cuda', dtype=compute_dtype):
                caption_embs = self.caption_encoder(cap_ids, return_dict=False, attention_mask=cap_am)[0]
            caption_embs = caption_embs * cap_am[..., None]

        # ---------- 4) 一次性 inference ----------
        modeling_type = getattr(getattr(self.dur_model, "config", None), "modeling_type", None)
        merged = map_phone_to_tokendict({'txt_token': ph_all, 'tone': tone_all}, pad_bos_eos=False)

        infer_kwargs = dict(
            txt_tokens=merged,
            start_pos=start_pos,
            condition=caption_embs if needs_caption else None,
            temperature=dur_disturb,
            topk=5,
            use_tqdm=self.use_tqdm
        )
        # ★ 关键：AR/AR+dur 模型需要参考 dur_tokens
        if modeling_type in ['ar', 'ar_cond_durtok', 'ar_dur']:
            infer_kwargs['dur_tokens'] = resource_context['dur_ref'].to(device)

        try:
            with model_lock(self.lock), torch.autocast(device_type='cuda', dtype=compute_dtype):
                dur_pred_all = self.dur_model.inference(**infer_kwargs, spk_ids=spk_ids_all)
        except TypeError:
            # 某些实现把 spk_ids 放进 txt_tokens 字典里
            merged['spk_ids'] = spk_ids_all
            infer_kwargs['txt_tokens'] = merged
            with model_lock(self.lock), torch.autocast(device_type='cuda', dtype=compute_dtype):
                dur_pred_all = self.dur_model.inference(**infer_kwargs)

        dur_pred_all = dur_pred_all.to(torch.int)  # [1, T_all], 离散时长(0..K-1)

        # ---------- 5) （可选）按 spk 分组做 log-域静/非静音对齐 ----------
        if normalize_dur and dur_pred_all.shape[1] > 10:
            # 准备静音 id
            try:
                sil_ph_list = self.ling_dict['phone'].sil_phonemes()
            except Exception:
                sil_ph_list = []
            sil_ids = []
            for p in sil_ph_list:
                try:
                    sil_ids.append(self.ling_dict['phone'].encode(p)[0])
                except Exception:
                    pass
            sil_ids = list(set(sil_ids))

            # 预测侧静音掩码（phone-level）
            sil_mask_pred_all = torch.zeros_like(ph_all, dtype=torch.bool)
            for sid_ in sil_ids:
                sil_mask_pred_all |= (ph_all == sid_)

            z_pred_all = torch.log1p(dur_pred_all.float())

            # 参考统计按说话人
            ref_by_spk = resource_context.get('ref_by_spk', {}) or {}
            present_sids = torch.unique(spk_ids_all[spk_ids_all > 0]).tolist()

            for sid in present_sids:
                ref_pack = ref_by_spk.get(int(sid))
                if not ref_pack:
                    continue
                ph_ref_spk  = ref_pack['ph_ref'].to(device)                   # [1, Lr]
                dur_ref_spk = ref_pack['dur_ref']
                if dur_ref_spk.dim() == 1:
                    dur_ref_spk = dur_ref_spk[None]
                dur_ref_spk = dur_ref_spk.to(device)                          # [1, Lr]

                sil_mask_ref = torch.zeros_like(ph_ref_spk, dtype=torch.bool)
                for sid_ in sil_ids:
                    sil_mask_ref |= (ph_ref_spk == sid_)

                z_ref_spk = torch.log1p(dur_ref_spk.float())

                # 预测侧（该说话人的位置）
                sid_mask = (spk_ids_all == sid)                               # [1, T_all]
                if sid_mask.sum() == 0:
                    continue
                pred_sil_sid  = sil_mask_pred_all & sid_mask
                pred_non_sid  = (~sil_mask_pred_all) & sid_mask
                ref_sil_sid   = sil_mask_ref
                ref_non_sid   = ~sil_mask_ref

                # 静音对齐
                if pred_sil_sid.any() and ref_sil_sid.any():
                    diff_sil = z_ref_spk[ref_sil_sid].mean() - z_pred_all[pred_sil_sid].mean()
                    z_pred_all[pred_sil_sid] += diff_sil

                # 非静音对齐
                if pred_non_sid.any() and ref_non_sid.any():
                    diff_non = z_ref_spk[ref_non_sid].mean() - z_pred_all[pred_non_sid].mean()
                    z_pred_all[pred_non_sid] += diff_non

            # 还原为整数并保持非负
            dur_pred_all = torch.expm1(z_pred_all).clamp_min(0.0)
            d_floor = torch.floor(dur_pred_all)
            frac    = (dur_pred_all - d_floor).clamp(0, 1)
            dur_pred_all = (d_floor + torch.bernoulli(frac)).to(torch.int)

        # ---------- 6) 切分回各 chunk ----------
        results = {}
        off = 0
        for ci, L in enumerate(lens):
            results[ci] = dur_pred_all[:, off:off + L]
            off += L
        return results


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

    def build_model(self, device):
        self.device = device
        self.precision = torch.bfloat16

        set_hparams(f'{self.dit_exp_name}/config.yaml', print_hparams=False)
        hparams['use_fsdp'] = False

        # 字典
        ling_dict = json.load(open('egs/tts/megatts3_dict.json'))
        self.ling_dict = {k: TokenTextEncoder(None, vocab_list=ling_dict[k], replace_oov='<UNK>') for k in ['phone', 'tone']}
        self.token_encoder = token_encoder = self.ling_dict['phone']

        # NEW: 与单人版保持一致 —— 初始化发音替换表
        self.ph_replace_table = {'en': {}, 'zh': {}}

        # Dur
        self.build_dur_model()

        # DiT/WavVAE
        self._build_model()
        load_ckpt(self.dit, f'{self.dit_exp_name}', 'dit', strict=False, mmap=True)
        self.vae.eval(); self.vae.to(self.device, dtype=self.precision)
        self.dit.eval(); self.dit.to(self.device, dtype=self.precision)
        self.cfg_mask_token_phone = 302 - 1
        self.cfg_mask_token_tone  = 32  - 1
        self.caption_encoder.to(self.device, dtype=self.precision)

        # Frontend & ASR
        self.build_frontend_model()
        from modules.asr.sensevoice.sensevoice_api import build_asr_model
        self.asr_model = build_asr_model(self.device)

        # VAD
        from silero_vad import load_silero_vad
        self.vad_model = load_silero_vad()

    # ---------------- Text normalize & helpers ----------------

    def preprocess_text(self, input_text: SSML, ph_replace_table=None, use_sa_frontend=False):
        def batch_replace(text: str, src: Union[str, List], tgt: str = ','):
            for p in src:
                text = text.replace(p, tgt)
            return text

        def _normalize_text_en(text: str):
            text_norm = common_preprocess(text)
            if not use_sa_frontend:
                text_norm = self.en_normalizer.normalize(text_norm)
            if ph_replace_table is not None:
                for src, tgt in ph_replace_table['en'].items():
                    text_norm = text_norm.replace(src, tgt)
            text_norm = common_process(text_norm)
            return text_norm

        def _normalize_text_zh(text):
            text_norm = common_preprocess(text)
            if not use_sa_frontend:
                from opencc import OpenCC
                jp2t_converter = OpenCC('jp2t')
                t2s_converter = OpenCC('t2s')
                text_norm = t2s_converter.convert(jp2t_converter.convert(text_norm))
                # text_norm = self.zh_normalizer.normalize(text_norm)
            if ph_replace_table is not None:
                for src, tgt in ph_replace_table['zh'].items():
                    text_norm = text_norm.replace(src, tgt)
            text_norm = common_process(text_norm)
            return text_norm

        def common_process(text: str):
            text_norm = text
            if not use_sa_frontend:
                pause_punc = [
                    '~', '～', ':', '$', '¥', '&', '#', '@', '^', '・', '·', '‘', '’', '“', '”', "'", "'", '"', '"',
                    '（', '）', '(', ')', '【', '】', '{', '}', '「', '」', '[', ']', '<', '>', '《', '》',
                    '%', '*', '|', '｜', '\\', '/', '-', '+', '_', '=',
                    '²',
                ]
                text_norm = batch_replace(text_norm, pause_punc, tgt='')
            return text_norm

        def common_preprocess(text: str):
            special_symbols = ['&#34;']
            if use_sa_frontend:
                special_symbols.extend(['"'])
            text_norm = batch_replace(text, special_symbols, tgt='')
            text_norm = batch_replace(text_norm, ['\n'], tgt=' ')
            return text_norm

        input_text.apply_sub()
        try:
            language_type = classify_language(input_text.text_str)
        except LangDetectException as err:
            handle_Ex = err
            print_once('无法检测语言，默认选择中文')
            language_type = 'zh'

        if language_type == 'en':
            input_text.normalize(_normalize_text_en)
            text_segs = SSML.chunk_text_with_breaks(input_text, limit=130, language_type='en', debug=False)
        else:
            input_text.normalize(_normalize_text_zh)
            text_segs = SSML.chunk_text_with_breaks(input_text, limit=60, language_type='zh', debug=False)

        return text_segs

    def refine_ph_tone(self, text: SSML, ph_pred: torch.Tensor, tone_pred: torch.Tensor):
        ph_tokens = ph_pred.squeeze().cpu().numpy()
        tone_tokens = tone_pred.squeeze().cpu().numpy()
        ph_tokens = self.ling_dict['phone'].decode(ph_tokens).split(' ')
        tone_tokens = self.ling_dict['tone'].decode(tone_tokens).split(' ')

        # 处理“儿化”等
        ph_tokens_, tone_tokens_ = [], []
        for p_i, p in enumerate(ph_tokens):
            if (p_i > 0 and p == "C0er" and ph_tokens[p_i - 1] in SHENGMU) or (p in YUNMU_ERHUA):
                ph_tokens_.append(p[:-1]); tone_tokens_.append(tone_tokens[p_i])
                ph_tokens_.append("C0er"); tone_tokens_.append('5')
            else:
                ph_tokens_.append(p); tone_tokens_.append(tone_tokens[p_i])
        ph_tokens, tone_tokens = ph_tokens_, tone_tokens_

        text_, ph_tokens, ph2word = align_word_phone(text.text_str, ph_tokens)
        ph2word = [p-1 for p in ph2word]

        ph_tokens, tone_tokens, ph2word = SSML.replace_ph_tone(text, ph_tokens, tone_tokens, ph2word)

        ph_tokens = self.ling_dict['phone'].encode(' '.join(ph_tokens))
        ph_pred = torch.LongTensor(ph_tokens)[None].to(ph_pred)
        tone_tokens = self.ling_dict['tone'].encode(' '.join(tone_tokens))
        tone_pred = torch.LongTensor(tone_tokens)[None].to(tone_pred)
        return ph_pred, tone_pred, ph2word

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

    def make_word_timestamps(self, text: SSML, dur_pred: np.ndarray, ph2word: List):
        dur_timestep = 0.01
        offsets = [0] + np.cumsum(dur_pred).tolist()
        words_to_get = get_word_list(text.text_str)
        ph2word = ph2word + [-3]
        words, timestamps = [], []
        ph_start_idx = 0
        for ph_end_idx in range(1, len(ph2word)):
            if ph2word[ph_end_idx] != ph2word[ph_start_idx]:
                if ph2word[ph_start_idx] >= 0:
                    words.append(words_to_get[ph2word[ph_start_idx]])
                    timestamps.append([offsets[ph_start_idx] * dur_timestep, offsets[ph_end_idx] * dur_timestep])
                ph_start_idx = ph_end_idx

        text_merged, text_norm_merged, text_idx_merged, text_norm_idx_merged = merge_norm_alignment(
            text.origin.text_str, words, debug=False
        )

        words_merged, timestamps_merged = [], []
        word_idx = 0
        for merge_idx in range(len(text_merged)):
            if isinstance(text_merged[merge_idx], list):
                word_merged, timestamp_merged = [], []
                for i in range(len(text_merged[merge_idx])):
                    if len(word_merged) > 0 and is_english(word_merged[-1]) and is_english(text_merged[merge_idx][i]):
                        word_merged.append(' ')
                    word_merged.append(text_merged[merge_idx][i])
                for i in range(len(text_norm_merged[merge_idx])):
                    timestamp_merged.append(timestamps[word_idx]); word_idx += 1
                words_merged.append(''.join(word_merged))
                if len(timestamp_merged) > 0:
                    timestamps_merged.append([timestamp_merged[0][0], timestamp_merged[-1][-1]])
                else:
                    if len(timestamps_merged) <= 0:
                        timestamps_merged.append([0.0, 0.0])
                    else:
                        timestamps_merged.append([timestamps_merged[-1][-1], timestamps_merged[-1][-1]])
            else:
                words_merged.append(text_merged[merge_idx])
                timestamps_merged.append(timestamps[word_idx]); word_idx += 1

        return {'words': words_merged, 'timestamps': timestamps_merged}

    def combine_audio_segments(self, segments, words_timestamps=(), sil_pad_lst=(), crossfade_duration=0.32):
        window_length = int(self.sr * crossfade_duration)
        hanning_window = np.hanning(2 * window_length)
        return_timestamps = len(words_timestamps) > 0
        combined_words_timestamps = {'words': [], 'timestamps': []}
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
                seg = segment
                if sil_pad_start > 0:
                    seg = np.concatenate([np.zeros((int(sil_pad_start * self.sr))), seg])
                if sil_pad_end > 0:
                    seg = np.concatenate([seg, np.zeros((int(sil_pad_end * self.sr)))])
                overlap = combined_audio[-window_length:] * hanning_window[window_length:] + seg[:window_length] * hanning_window[:window_length]
                offset = combined_audio[:-window_length].shape[0] + sil_pad_start * self.sr
                combined_audio = np.concatenate([combined_audio[:-window_length], overlap, seg[window_length:]])
                if return_timestamps:
                    combined_words_timestamps['words'] += words_timestamps[i]['words']
                    timestamps = words_timestamps[i]['timestamps']
                    offset = offset / self.sr
                    timestamps = [[s[0] + offset, s[1] + offset] for s in timestamps]
                    combined_words_timestamps['timestamps'] += timestamps
        return combined_audio, combined_words_timestamps

    def chunk_wavs_vad(self, wav_16k=None, speech_timestamps=None,
                        chunk_duration=10, max_duration=60,
                        vad_thresholds=(0.50, 0.35, 0.25),
                        min_speech_ms=150, min_silence_ms=100):
        """
        并行安全版：
        - Silero VAD 是有状态的，同一实例跨线程使用会发生竞态；
        - 这里对 get_speech_timestamps 加锁，并在每次调用前 reset_states()。
        """
        if speech_timestamps is None and wav_16k is not None:
            from silero_vad import get_speech_timestamps
            # 限制输入长度，和原逻辑一致
            wav_16k = wav_16k[: int(16000 * max_duration * 1.2)]

            for thr in vad_thresholds:
                try:
                    # 关键：加锁 + reset，避免多线程破坏内部 RNN 状态
                    with self.lock:
                        if hasattr(self.vad_model, "reset_states"):
                            self.vad_model.reset_states()
                        st = get_speech_timestamps(
                            wav_16k, self.vad_model, return_seconds=True,
                            threshold=thr,
                            min_speech_duration_ms=min_speech_ms,
                            min_silence_duration_ms=min_silence_ms
                        )
                except Exception as e:
                    # 极端情况下兜底为整段（不影响后续逻辑）
                    print_once(f'| VAD failed with threshold={thr}: {e}; fallback to full clip.')
                    total_sec = (len(wav_16k) / 16000.0)
                    st = [{'start': 0.0, 'end': min(total_sec, float(max_duration))}]

                if len(st) > 0:
                    speech_timestamps = st
                    print_once(f'| VAD detected {len(st)} segments with threshold={thr}')
                    break
            else:
                speech_timestamps = []

        # 后处理与切块逻辑保持不变
        start = max(0.0, float(speech_timestamps[0]['start'])) if len(speech_timestamps) else 0.0
        end   = min(float(speech_timestamps[-1]['end']), float(max_duration)) if len(speech_timestamps) else float(max_duration)
        if end <= start:
            total_dur = (len(wav_16k) / 16000.0) if wav_16k is not None else float(max_duration)
            end = max(start + 0.1, min(total_dur, max_duration))

        offs, cur = [], start
        while cur < end - 1e-6:
            nxt = min(end, cur + float(chunk_duration))
            offs.append((cur, nxt))
            if nxt == cur:
                break
            cur = nxt
        return offs


    def _print_target_ph_durations_seconds(self,
                                        ph_seq_target: torch.Tensor,   # [1, N_tgt]
                                        tone_seq_target: torch.Tensor, # [1, N_tgt]
                                        dur_seq_target: torch.Tensor,  # [1, N_tgt]（单位：0.01s）
                                        tag: str = "TARGET"):
        """
        仅打印 target 段的逐 phone 时长（单位：秒），包含 start/end（相对 target 起点，0s 开始），并同步打印 tone。
        """
        ph_seq = ph_seq_target.squeeze(0).detach().cpu()
        tone_seq = tone_seq_target.squeeze(0).detach().cpu()
        dur_seq = dur_seq_target.squeeze(0).detach().cpu()

        L = int(min(ph_seq.numel(), tone_seq.numel(), dur_seq.numel()))
        ph_seq = ph_seq[:L]
        tone_seq = tone_seq[:L]
        dur_seq = dur_seq[:L]

        ph_list = self.ling_dict['phone'].decode(ph_seq.numpy()).split(' ')
        tone_list = self.ling_dict['tone'].decode(tone_seq.numpy()).split(' ')
        if len(ph_list) != L:
            ph_list = ph_list[:L]
        if len(tone_list) != L:
            tone_list = tone_list[:L]

        starts_cs = np.cumsum([0] + dur_seq.numpy().tolist()[:-1]).tolist()
        total_sec = float(dur_seq.sum().item()) / 100.0
        print(f"[PH/TIME][{tag}] total_phones={L}, total_dur={total_sec:.3f}s")
        print(f"[PH/TIME][{tag}] {'idx':>4} | {'ph':>10} | {'tone':>4} | {'dur(s)':>8} | {'start(s)':>10} | {'end(s)':>10}")

        for i in range(L):
            dur_s = float(dur_seq[i].item()) / 100.0
            start_s = float(starts_cs[i]) / 100.0
            end_s = start_s + dur_s
            print(f"[PH/TIME][{tag}] {i:4d} | {ph_list[i]:>10} | {tone_list[i]:>4} | {dur_s:8.3f} | {start_s:10.3f} | {end_s:10.3f}")


    def _print_dit_inputs_debug(self, tag: str, caption_str: str, prompt_text: str, text_inputs=None, max_ids: int = 64):
        """
        打印送入 DiT 的 caption 与 text（统一为 <S{sid}>...</S{sid}> 串）：
        - 原始字符串
        - 可见 token 数（去掉 pad）
        - 前 max_ids 个 token id
        - 反解码文本（不跳过 special tokens）
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
                txt_inputs = text_inputs  # 允许直接传 GPU 张量
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

    # -------- NEW: 统一解析两类标注并返回 [(sid, content), ...] --------
    def _parse_dialogue_segments(self, raw_text: str) -> List[Tuple[int, str]]:
        """
        优先解析新格式：<S{sid}>...</S{sid}>
        若找不到，则回退解析旧格式：<SPK>{sid}</SPK>content...
        解析失败则视为单说话人 sid=1。
        """
        segs = []
        # 新格式：<S1>text</S1>
        pat_new = re.compile(r"<\s*S\s*(\d+)\s*>\s*(.*?)\s*<\s*/\s*S\s*\1\s*>", re.IGNORECASE | re.DOTALL)
        for m in pat_new.finditer(raw_text):
            sid = int(m.group(1))
            content = m.group(2)
            if content and content.strip():
                segs.append((sid, content))
        if segs:
            return segs

        # 旧格式：<SPK>1</SPK>text<SPK>2</SPK>text2...
        pat_old = re.compile(r"<\s*SPK\s*>\s*(\d+)\s*<\s*/\s*SPK\s*>", re.IGNORECASE)
        parts = pat_old.split(raw_text)
        if len(parts) > 1:
            for i in range(1, len(parts), 2):
                try:
                    sid = int(parts[i])
                except:
                    sid = 1
                text_i = parts[i+1] if (i+1) < len(parts) else ""
                if text_i and text_i.strip():
                    segs.append((sid, text_i))
        if not segs:
            segs.append((1, raw_text))
        return segs

    # -------- NEW: 将任意输入串规范化为训练侧文本：<S{sid}>...</S{sid}> --------
    def _to_train_style_text(self, s: str) -> str:
        segs = self._parse_dialogue_segments(s)
        parts = []
        for sid, content in segs:
            c = (content or '').strip()
            if c:
                parts.append(f'<S{sid}>{c}</S{sid}>')
        return ''.join(parts)

    def preprocess(self, audio_bytes: Union[bytes, List[bytes], Tuple[bytes, bytes]],
                wav_path: Optional[str]=None, topk_dur=1, ref_texts: Optional[List[str]]=None, **kwargs):
        """
        audio_bytes: bytes 或 [bytes, bytes]（两位说话人的参考音频）
        ref_texts:   None 或 [str, str]（两人的 clean 文本，不含标签；若不给则用 ASR）
        返回 resource_context 增加：
        - text_ref_clean: 无 <S{sid}> 的参考文本（拼接）
        - text_ref_raw:   '<S1>ref1</S1><S2>ref2</S2>'（单人则 '<S1>ref</S1>'）
        - text_ref_by_spk:{sid: clean_ref_text_for_that_speaker}  ✅ 新增
        - spk_ids_ref:    [1, Tph_ref] 1-based phone-level 说话人 id
        - ref_by_spk:     {sid: {'ph_ref','tone_ref','dur_ref'}}
        - （当 use_old_dur=True 时）dur_prefill_by_spk: {sid: {'incremental_state','ctx_dur_tokens','last_pos'}}
        """
        def _convert_to_wav_and_16k(ab):
            wav_bytes = convert_to_wav_bytes(ab)
            wav_24k, _ = librosa.core.load(wav_bytes, sr=self.sr)
            ws = hparams['win_size']
            if len(wav_24k) % ws < ws - 1:
                wav_24k = np.pad(
                    wav_24k,
                    (0, ws - 1 - (len(wav_24k) % ws)),
                    mode='constant',
                    constant_values=0.0
                ).astype(np.float32)
            wav_24k = np.pad(wav_24k, (0, 12000), mode='constant', constant_values=0.0).astype(np.float32)
            wav_16k = librosa.resample(wav_24k, orig_sr=self.sr, target_sr=16000)
            return wav_24k, wav_16k

        @torch.no_grad()
        def _process_alignment(alignment_tokens, prompt_max_frame):
            ph_ref, tone_ref, dur_ref, _ = split_ph_timestamp(deepcopy(alignment_tokens))
            ph_ref = torch.Tensor(ph_ref)[None].to(self.device)
            tone_ref = torch.Tensor(tone_ref)[None].to(self.device)

            # 帧数对齐
            if dur_ref.sum() < prompt_max_frame:
                dur_ref[-1] += prompt_max_frame - dur_ref.sum()
            elif dur_ref.sum() > prompt_max_frame:
                len_diff = dur_ref.sum() - prompt_max_frame
                while len_diff > 0:
                    for i in range(len(dur_ref)):
                        dur_ref[i] -= 1; len_diff -= 1
                        if len_diff == 0: break
                    if len_diff == 0: break

            # dur -> mel2ph -> 干净 dur
            mel2ph_ref = self.length_regulator(torch.LongTensor(dur_ref)[None].to(self.device)).to(self.device)
            mel2ph_ref = mel2ph_ref[:, :mel2ph_ref.size(1)//self.fm*self.fm]
            dur_ref = mel2token_to_dur(mel2ph_ref)
            return ph_ref, tone_ref, dur_ref, mel2ph_ref

        @torch.no_grad()
        def _align_one_ref(wav_24k, wav_16k):
            # VAD 截取参考语音有效片段
            chunk_wav_offsets = self.chunk_wavs_vad(wav_16k, chunk_duration=10, max_duration=self.max_ref_duration)
            print(f"Detected {len(chunk_wav_offsets)} speech segments in one reference audio.")
            s0, s1 = int(chunk_wav_offsets[0][0]*self.sr), int(chunk_wav_offsets[-1][-1]*self.sr)
            wav_24k_ = wav_24k[s0:s1]
            s0k, s1k = int(chunk_wav_offsets[0][0]*16000), int(chunk_wav_offsets[-1][-1]*16000)
            wav_16k_ = wav_16k[s0k:s1k]

            if self.use_old_aligner:
                # Whisper-small：分块对齐
                fm = 160 * 8
                ph_lst, tone_lst, dur_lst, m2p_lst = [], [], [], []
                for (chunk_start, chunk_end) in chunk_wav_offsets:
                    c0 = int((chunk_start * 16000) // fm * fm)
                    c1 = int((chunk_end   * 16000) // fm * fm)
                    wav_16k_chunk = wav_16k_[c0:c1]

                    with model_lock(self.lock):
                        mel = torch.tensor(whisper.log_mel_spectrogram(wav_16k_chunk).T, dtype=self.precision).to(self.device)[None].transpose(1,2)
                        prompt_max_frame = mel.size(2) // self.fm * self.fm
                        mel = mel[:, :, :prompt_max_frame]
                        token = torch.LongTensor([[798]]).to(self.device)
                        audio_features = self.aligner_lm.embed_audio(mel)
                        for _ in tqdm(range(1024)) if self.use_tqdm else range(1024):
                            logits = self.aligner_lm.logits(token, audio_features, None)
                            token_pred = torch.argmax(F.softmax(logits[:, -1], dim=-1), 1)[None]
                            token = torch.cat([token, token_pred], dim=1)
                            if token_pred[0] == 799: break
                        alignment_tokens = token[0, 1:-1].detach().to("cpu")

                    ph_i, tone_i, dur_i, m2p_i = _process_alignment(alignment_tokens, prompt_max_frame)
                    ph_lst.append(ph_i); tone_lst.append(tone_i); dur_lst.append(dur_i); m2p_lst.append(m2p_i)

                ph_ref = torch.cat(ph_lst, dim=1)
                tone_ref = torch.cat(tone_lst, dim=1)
                dur_ref  = torch.cat(dur_lst, dim=1) if dur_lst[0].dim()==2 else torch.cat([d if d.dim()==2 else d[None] for d in dur_lst], dim=1)
                mel2ph_ref = torch.cat(m2p_lst, dim=1)

            else:
                # 自研对齐器：整段
                with model_lock(self.lock):
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        whisper_wav = torch.from_numpy(wav_16k_)[None].to(self.device, non_blocking=True)
                        whisper_wav = whisper_wav[:, :whisper_wav.shape[-1] // 1280 * 1280]
                        prompt_max_frame = whisper_wav.shape[-1] // 160 // 8 * 8
                        token = torch.LongTensor([798])[None, :].to(self.device)
                        token = self.aligner_lm.inference(
                            whisper_wav, token, topk=1, temperature=0.7,
                            max_new_tokens=16384, eos_idx=799, use_tqdm=False
                        )
                    alignment_tokens = token[0].detach().to("cpu")
                ph_ref, tone_ref, dur_ref, mel2ph_ref = _process_alignment(alignment_tokens, prompt_max_frame)

            # ASR 文本
            from modules.asr.sensevoice.sensevoice_api import run_asr_model
            with model_lock(self.lock):
                text_ref = run_asr_model([wav_16k_], self.asr_model, with_segments=False)[0]['text_normed']

            return {
                'wav_24k': wav_24k_,
                'ph_ref': ph_ref, 'tone_ref': tone_ref,
                'dur_ref': dur_ref, 'mel2ph_ref': mel2ph_ref,
                'text_ref': text_ref,
            }

        # ========= 单/双参考入口 =========
        if isinstance(audio_bytes, (list, tuple)):
            assert len(audio_bytes) == 2, "双说话人请传入两个 bytes。"
            w24_1, w16_1 = _convert_to_wav_and_16k(audio_bytes[0])
            w24_2, w16_2 = _convert_to_wav_and_16k(audio_bytes[1])

            ret1 = _align_one_ref(w24_1, w16_1)
            ret2 = _align_one_ref(w24_2, w16_2)

            # 标准化形状
            ph1, tn1, dr1, m2p1 = ret1['ph_ref'], ret1['tone_ref'], ret1['dur_ref'], ret1['mel2ph_ref']
            ph2, tn2, dr2, m2p2 = ret2['ph_ref'], ret2['tone_ref'], ret2['dur_ref'], ret2['mel2ph_ref']
            if dr1.dim() == 1: dr1 = dr1[None]
            if dr2.dim() == 1: dr2 = dr2[None]

            # mel2ph 偏移
            T1 = ph1.shape[1]
            m2p2_off = m2p2 + (m2p2 > 0).long() * T1

            ph_ref   = torch.cat([ph1,   ph2],   dim=1)
            tone_ref = torch.cat([tn1,   tn2],   dim=1)
            dur_ref  = torch.cat([dr1,   dr2],   dim=1)
            mel2ph_ref = torch.cat([m2p1, m2p2_off], dim=1)

            # 文本（拼接版 + 逐说话人版）
            if isinstance(ref_texts, (list, tuple)) and len(ref_texts) == 2:
                text_ref_by_spk = {1: ref_texts[0], 2: ref_texts[1]}        # ✅ 新增
                text_ref_clean  = ref_texts[0] + ref_texts[1]
                text_ref_raw    = f"<S1>{ref_texts[0]}</S1><S2>{ref_texts[1]}</S2>"
            else:
                text_ref_by_spk = {1: ret1['text_ref'], 2: ret2['text_ref']} # ✅ 新增
                text_ref_clean  = ret1['text_ref'] + ret2['text_ref']
                text_ref_raw    = f"<S1>{ret1['text_ref']}</S1><S2>{ret2['text_ref']}</S2>"

            # VAE latent
            wav_24k_full = np.concatenate([ret1['wav_24k'], ret2['wav_24k']])
            with model_lock(self.lock):
                wav = torch.tensor(wav_24k_full, dtype=self.precision, device=self.device)[None]
                with torch.autocast(device_type='cuda', dtype=self.precision):
                    vae_latent = self.vae.encode_latent(wav)
            vae_latent = vae_latent[:, :mel2ph_ref.size(1)//4]

            # phone-level 说话人 id（1-based）
            T2 = ph2.shape[1]
            spk_ids_ref = torch.cat([
                torch.full((1, T1), 1, dtype=torch.long),
                torch.full((1, T2), 2, dtype=torch.long)
            ], dim=1)

            # 按说话人切片（供 per-speaker prefill）
            ref_by_spk = {
                1: {'ph_ref': ph1.cpu(), 'tone_ref': tn1.cpu(), 'dur_ref': (dr1 if dr1.dim()==2 else dr1[None]).cpu()},
                2: {'ph_ref': ph2.cpu(), 'tone_ref': tn2.cpu(), 'dur_ref': (dr2 if dr2.dim()==2 else dr2[None]).cpu()},
            }

        else:
            # 单参考
            wav_bytes = convert_to_wav_bytes(audio_bytes)
            w24, _ = librosa.core.load(wav_bytes, sr=self.sr)
            ws = hparams['win_size']
            if len(w24) % ws < ws - 1:
                w24 = np.pad(w24, (0, ws - 1 - (len(w24) % ws)), mode='constant', constant_values=0.0).astype(np.float32)
            w24 = np.pad(w24, (0, 12000), mode='constant', constant_values=0.0).astype(np.float32)
            w16 = librosa.resample(w24, orig_sr=self.sr, target_sr=16000)

            ret = _align_one_ref(w24, w16)
            ph_ref, tone_ref, dur_ref, mel2ph_ref = ret['ph_ref'], ret['tone_ref'], ret['dur_ref'], ret['mel2ph_ref']
            if dur_ref.dim() == 1: dur_ref = dur_ref[None]
            text_ref_clean  = ret['text_ref']
            text_ref_raw    = f"<S1>{ret['text_ref']}</S1>"
            text_ref_by_spk = {1: ret['text_ref']}   # ✅ 新增

            if topk_dur > 1: self.dur_model.hparams["infer_top_k"] = topk_dur
            else:            self.dur_model.hparams["infer_top_k"] = None

            with model_lock(self.lock):
                wav = torch.tensor(w24, dtype=self.precision, device=self.device)[None]
                with torch.autocast(device_type='cuda', dtype=self.precision):
                    vae_latent = self.vae.encode_latent(wav)
            vae_latent = vae_latent[:, :mel2ph_ref.size(1)//4]
            spk_ids_ref = torch.ones((1, ph_ref.shape[1]), dtype=torch.long)

            ref_by_spk = {
                1: {'ph_ref': ph_ref.cpu(), 'tone_ref': tone_ref.cpu(), 'dur_ref': (dur_ref if dur_ref.dim()==2 else dur_ref[None]).cpu()}
            }

        # ========= Duration Prompting =========
        if self.use_old_dur:
            # 全局旧 dur prefill（保留作回退）
            dur_tokens_2d_ = mel2token_to_dur(mel2ph_ref, ph_ref.shape[1]).clamp(
                max=self.hp_dur_model.get('dur_code_size', self.hp_dur_model.get('dur_max_value', 128)) - 1) + 1
            ctx_dur_tokens = dur_tokens_2d_.clone().flatten(0, 1).to(self.device)
            txt_tokens_flat_ = ph_ref.flatten(0, 1)
            ctx_dur_tokens = ctx_dur_tokens[txt_tokens_flat_ > 0][None]
            last_dur_pos_prompt = ctx_dur_tokens.shape[1]
            dur_spk_pos_ids_flat = torch.arange(0, last_dur_pos_prompt, device=mel2ph_ref.device)[None, :].long()
            with model_lock(self.lock):
                _, incremental_state_dur_prompt = self.dur_model.infer(
                    ph_ref, {'tone': tone_ref}, None, None, None,
                    ctx_vqcodes=ctx_dur_tokens, spk_pos_ids_flat=dur_spk_pos_ids_flat, return_state=True)

            # ✨ 按说话人 prefill（供 old dur 分组推理使用）
            dur_prefill_by_spk = self._build_old_dur_prefill_by_speaker(ref_by_spk)

            ret_dur = {
                'incremental_state_dur_prompt': incremental_state_dur_prompt,
                'ctx_dur_tokens': ctx_dur_tokens,
                'last_dur_pos_prompt': last_dur_pos_prompt,
                'dur_prefill_by_spk': dur_prefill_by_spk,
            }
        else:
            # 新 dur_lm：prefill 参考（phone+tone，duration 为参考 dur_ref）
            merged_ph_tokens = map_phone_to_tokendict({'txt_token': ph_ref, 'tone': tone_ref}, pad_bos_eos=False)
            with model_lock(self.lock):
                with torch.autocast(device_type='cuda', dtype=self.precision):
                    dur_start_pos = self.dur_model.prefill(merged_ph_tokens, dur_ref.to(self.device))
            ret_dur = {'dur_start_pos': dur_start_pos}

        return {
            'text_ref_clean': text_ref_clean,
            'text_ref_raw':   text_ref_raw,
            'text_ref_by_spk': text_ref_by_spk,   # ✅ 新增，供时长 LM 逐 chunk caption 使用
            'ph_ref': ph_ref.cpu(),
            'tone_ref': tone_ref.cpu(),
            'dur_ref': dur_ref.cpu(),
            'mel2ph_ref': mel2ph_ref.cpu(),
            'vae_latent': vae_latent.cpu(),
            'spk_ids_ref': spk_ids_ref.cpu(),
            'ref_by_spk': ref_by_spk,
            **ret_dur
        }



    def _dur_predict_per_chunk_by_speaker(self, chunk_items, resource_context, dur_disturb=0.1, normalize_dur=True):
        """
        多说话人 & 逐段时长预测（for 新 dur_lm / dur_lm_seq2seq）：
        - 先按说话人 prefill（仅用该说话人参考 ph/tone/dur）；
        - 再对该说话人的所有 chunk 逐个 decode；
        - 关键修复：每个 chunk 的 caption 使用“该说话人的参考文本 + 该 chunk 文本”，
        而不是全局拼接的 text_ref_clean。

        Args:
            chunk_items: List[dict]，来自 forward 前的切分结果，每项含：
                {'sid': int, 'ch': SSML片段, 'ph': Tensor[1,L], 'tone': Tensor[1,L], 'ph2word': Optional}
            resource_context: dict，来自 preprocess(...)
                需要包含：
                - 'ref_by_spk': {sid: {'ph_ref','tone_ref','dur_ref'}}
                - 'text_ref_by_spk': {sid: str}   # 本函数将优先使用
                - （兼容兜底）'text_ref_clean' 或 'text_ref'
            dur_disturb: float
            normalize_dur: bool
        Returns:
            {chunk_idx: Tensor[1, L_chunk]}  # 每个 chunk 的离散时长序列（单位：code 0..K-1）
        """
        import torch
        from collections import defaultdict

        device = self.device
        compute_dtype = self.precision

        ref_by_spk = resource_context.get('ref_by_spk')
        if not ref_by_spk:
            raise RuntimeError("严格模式：未找到任何参考（ref_by_spk 为空）。")

        # 该说话人的参考文本（优先）
        text_ref_by_spk = resource_context.get('text_ref_by_spk', {}) or {}
        # 兼容老资源：若缺失则退化到全局文本
        global_text_ref = (
            resource_context.get('text_ref_clean')
            or resource_context.get('text_ref')
            or ""
        )

        # 按出现顺序分组
        groups = defaultdict(list)  # sid -> [(ci, item)]
        for ci, it in enumerate(chunk_items):
            groups[int(it['sid'])].append((ci, it))

        dur_model_type = getattr(self, "dur_model_type", "lm")
        modeling_type  = getattr(getattr(self.dur_model, "config", None), "modeling_type", None)

        results = {}

        def _encode_caption(text_str: str) -> torch.Tensor:
            """编码单个 caption（逐 chunk 独立 caption）"""
            inputs = self.caption_tokenizer([text_str], padding=True, return_tensors="pt")
            ids = inputs.input_ids.to(device)
            am  = inputs.attention_mask.to(device)
            with torch.autocast(device_type='cuda', dtype=compute_dtype):
                embs = self.caption_encoder(ids, return_dict=False, attention_mask=am)[0]
            embs = embs * am[..., None]
            return embs.to(dtype=compute_dtype)

        # 构建静音掩码（给 normalize_dur 用）
        def _build_sil_masks(ph_pred: torch.Tensor, ph_ref: torch.Tensor):
            try:
                sil_ph_list = self.ling_dict['phone'].sil_phonemes()
            except Exception:
                sil_ph_list = []
            sil_ids = []
            for sp in sil_ph_list:
                try:
                    sil_ids.append(self.ling_dict['phone'].encode(sp)[0])
                except Exception:
                    pass
            sil_ids = list(set(sil_ids))

            sil_mask_pred = torch.zeros_like(ph_pred, dtype=torch.long)
            sil_mask_ref  = torch.zeros_like(ph_ref,  dtype=torch.long)
            for sid_ in sil_ids:
                sil_mask_pred[ph_pred == sid_] = 1
                sil_mask_ref[ ph_ref  == sid_] = 1
            return sil_mask_pred, sil_mask_ref

        # === 关键：按说话人循环，prefill 后立刻解该说话人的所有 chunk ===
        for sid, items in groups.items():
            sid = int(sid)
            ref_pack = ref_by_spk[sid]
            ph_ref_spk   = ref_pack['ph_ref'].to(device)
            tone_ref_spk = ref_pack['tone_ref'].to(device)
            dur_ref_spk  = ref_pack['dur_ref']
            if dur_ref_spk.dim() == 1:
                dur_ref_spk = dur_ref_spk[None]
            dur_ref_spk = dur_ref_spk.to(device)

            # 仅用该说话人的参考做 prefill
            ref_tokens = map_phone_to_tokendict(
                {'txt_token': ph_ref_spk, 'tone': tone_ref_spk},
                pad_bos_eos=False
            )
            with model_lock(self.lock), torch.autocast(device_type='cuda', dtype=compute_dtype):
                start_pos = self.dur_model.prefill(ref_tokens, dur_ref_spk)

            # 逐段推理该说话人的每个 chunk
            for (ci, it) in items:
                ph   = it['ph'].to(device)
                tone = it['tone'].to(device)
                merged = map_phone_to_tokendict({'txt_token': ph, 'tone': tone}, pad_bos_eos=False)

                if dur_model_type == 'lm':
                    # 纯 LM 无 caption
                    with model_lock(self.lock), torch.autocast(device_type='cuda', dtype=compute_dtype):
                        dur_pred = self.dur_model.inference(
                            txt_tokens=merged,
                            start_pos=start_pos,
                            temperature=dur_disturb,
                            use_tqdm=self.use_tqdm
                        )

                elif dur_model_type == 'lm_seq2seq':
                    # —— 修复点：逐 chunk caption = 该说话人的参考文本 + 当前 chunk 文本 ——
                    local_ref_txt = text_ref_by_spk.get(sid, global_text_ref)
                    caption_text  = f"{local_ref_txt}{it['ch'].text_str}"
                    caption_embs  = _encode_caption(caption_text)

                    infer_kwargs = dict(
                        txt_tokens=merged,
                        condition=caption_embs,
                        start_pos=start_pos,
                        temperature=dur_disturb,
                        topk=5,
                        use_tqdm=self.use_tqdm
                    )
                    if modeling_type in ['ar', 'ar_cond_durtok', 'ar_dur']:
                        infer_kwargs['dur_tokens'] = dur_ref_spk

                    with model_lock(self.lock), torch.autocast(device_type='cuda', dtype=compute_dtype):
                        dur_pred = self.dur_model.inference(**infer_kwargs)

                else:
                    raise NotImplementedError(f"dur_model_type={dur_model_type} 暂不支持该逐段推理。")

                # ===== 可选：normalize_dur —— 用参考段的静/非静音均值，拉回语速 =====
                if normalize_dur and dur_pred.shape[1] > 10:
                    sil_mask_pred, sil_mask_ref = _build_sil_masks(ph, ph_ref_spk)
                    z_dur_pred = torch.log1p(dur_pred.float())
                    z_dur_ref  = torch.log1p(dur_ref_spk.float())

                    # 静音部分对齐
                    if sil_mask_pred.sum() > 0 and sil_mask_ref.sum() > 0:
                        diff_sil = z_dur_ref[sil_mask_ref == 1].mean() - z_dur_pred[sil_mask_pred == 1].mean()
                        z_dur_pred[sil_mask_pred == 1] += diff_sil

                    # 非静音部分对齐
                    non_pred = (sil_mask_pred != 1)
                    non_ref  = (sil_mask_ref  != 1)
                    if non_pred.sum() > 0 and non_ref.sum() > 0:
                        diff_non = z_dur_ref[non_ref].mean() - z_dur_pred[non_pred].mean()
                        z_dur_pred[non_pred] += diff_non

                    # 还原为整数
                    dur_pred = torch.expm1(z_dur_pred).clamp_min(0)
                    d_floor  = torch.floor(dur_pred)
                    frac     = (dur_pred - d_floor).clamp(0, 1)
                    dur_pred = (d_floor + torch.bernoulli(frac)).to(torch.int)

                results[ci] = dur_pred  # Tensor[1, L_chunk]

        return results


    def _dur_predict_old_ar_grouped_by_speaker(
        self,
        chunk_items,
        resource_context,
        dur_disturb: float = 0.1,
        mode: str = "reset",            # "reset"（默认）：单人版风格；"rolling"：原先滚动KV风格
        normalize_dur: bool = False     # 可选：时长均值归一到参考段（静/非静音分开对齐）
    ):
        """
        ARDurPredictor(old dur) 时长预测（双模式）：
        - mode="reset"（默认，推荐）：按说话人分组，每个 chunk 都“重置”到该说话人的参考 prefill 增量状态，
        不沿用上一个 chunk 的增量状态；仅 spk_pos_ids 连续累加；first_decoder_inp 固定用参考 dur 的最后一个 token。
        —— 这与“单人版”行为一致，语速与稳定性更可控。
        - mode="rolling"：原先实现；同一说话人的 chunk 之间滚动地传递增量状态与最后 token，更强的跨句耦合。

        返回:
            {chunk_idx: Tensor[1, L_chunk]}  —— 每个 chunk 的离散时长序列（0..K-1）
        """
        from collections import defaultdict
        import torch

        device = self.device
        hp = getattr(self, "hp_dur_model", {})
        dur_max = hp.get('dur_code_size', hp.get('dur_max_value', 128)) - 1

        # 按说话人分组（保持出现顺序）
        groups = defaultdict(list)  # sid -> [(ci, item_dict), ...]
        for ci, it in enumerate(chunk_items):
            groups[int(it['sid'])].append((ci, it))

        # 说话人级参考 prefill（在 preprocess 阶段构建）
        prefill_by_spk = resource_context.get('dur_prefill_by_spk', {}) or {}

        # 取目标 device（避免跨设备错误）
        try:
            dev_idx = int(str(device).split(':')[1])
            target_device = torch.device(f'cuda:{dev_idx}')
        except Exception:
            target_device = torch.device(device if isinstance(device, str) else device)

        # 准备静音音素 id（给 normalize 用）
        try:
            sil_ph_list = self.ling_dict['phone'].sil_phonemes()
        except Exception:
            sil_ph_list = []
        sil_ids = []
        for sp in sil_ph_list:
            try:
                sil_ids.append(self.ling_dict['phone'].encode(sp)[0])
            except Exception:
                pass
        sil_ids = list(set(sil_ids))

        results = {}

        for sid, items in groups.items():
            # —— 该说话人的参考 prefill（三件套） ——
            pref = prefill_by_spk.get(int(sid))
            if pref is None:
                # 兜底到全局 prefill（兼容单参考）
                pref = {
                    'incremental_state': resource_context['incremental_state_dur_prompt'],
                    'ctx_dur_tokens': resource_context['ctx_dur_tokens'],
                    'last_pos': resource_context.get('last_dur_pos_prompt', resource_context['ctx_dur_tokens'].shape[1]),
                }

            # 参考 prefill：在 CPU 保存的“安全树”
            inc_state_cpu = pref['incremental_state']                 # CPU 端状态树
            ctx_dur_tokens_cpu = pref['ctx_dur_tokens']               # [1, Lctx]（值域 1..K）
            last_pos = int(pref['last_pos'])                          # 仅用于位置编码累加
            last_token_1k_pref = ctx_dur_tokens_cpu[:, -1:]           # 参考 dur 的最后一个 token（1..K）

            # 参考段（用于 normalize_dur）
            ref_pack = resource_context.get('ref_by_spk', {}).get(int(sid))
            ph_ref_spk = ref_pack['ph_ref'].to(target_device) if ref_pack else None
            dur_ref_spk = ref_pack['dur_ref'] if ref_pack else None
            if dur_ref_spk is not None and dur_ref_spk.dim() == 1:
                dur_ref_spk = dur_ref_spk[None]
            dur_ref_spk = dur_ref_spk.to(target_device) if dur_ref_spk is not None else None

            # 用于 rolling 模式的“滚动状态”
            inc_state_cpu_rolling = inc_state_cpu
            last_token_1k_rolling = last_token_1k_pref

            for (ci, it) in items:
                ph_pred   = it['ph'].to(target_device)
                tone_pred = it['tone'].to(target_device)
                txt_len   = int(ph_pred.shape[1])

                # 位置编码：两种模式都要连续累加
                spk_pos_ids_flat = torch.arange(last_pos, last_pos + txt_len, device=target_device)[None, :].long()
                last_pos += txt_len

                if mode.lower() == "rolling":
                    # —— 原先滚动式：上一句的 inc_state + 最后 token 作为下一句起点 ——
                    inc_state_in = self._clone_tensor_tree_to(inc_state_cpu_rolling, target_device)
                    first_decoder_inp = last_token_1k_rolling.to(target_device)

                    with model_lock(self.lock):
                        ret = self.dur_model.infer(
                            ph_pred, {'tone': tone_pred},
                            None, None, None,
                            incremental_state=inc_state_in,
                            first_decoder_inp=first_decoder_inp,      # 上一句最后一个 1..K
                            spk_pos_ids_flat=spk_pos_ids_flat,
                            use_tqdm=False,
                            return_state=True
                        )
                    # 兼容不同返回
                    if isinstance(ret, tuple) and len(ret) == 2:
                        dur_pred_1k, inc_state_out = ret
                    else:
                        raise RuntimeError("ARDurPredictor.infer(return_state=True) 应返回 (pred, state)")

                    # 更新滚动起点
                    last_token_1k_rolling = dur_pred_1k[:, -1:].detach()
                    inc_state_cpu_rolling = self._clone_tensor_tree_to(inc_state_out, torch.device('cpu'))

                else:
                    # —— reset（单人版风格）：每个 chunk 都回到“参考 prefill”后的状态 ——
                    inc_state_in = self._clone_tensor_tree_to(inc_state_cpu, target_device)
                    first_decoder_inp = last_token_1k_pref.to(target_device)

                    with model_lock(self.lock):
                        ret = self.dur_model.infer(
                            ph_pred, {'tone': tone_pred},
                            None, None, None,
                            incremental_state=inc_state_in,
                            first_decoder_inp=first_decoder_inp,      # 始终用参考段最后一个 1..K
                            spk_pos_ids_flat=spk_pos_ids_flat,
                            use_tqdm=False,
                            return_state=False                         # 不需要返回滚动状态
                        )
                    dur_pred_1k = ret[0] if isinstance(ret, tuple) else ret

                # 1..K -> 0..K-1
                dur_pred = (dur_pred_1k - 1).to(torch.int)

                # ===== 可选：normalize_dur —— 用参考段的静/非静音均值，拉回语速 =====
                if normalize_dur and dur_ref_spk is not None and dur_pred.shape[1] > 10:
                    sil_mask_pred = torch.zeros_like(ph_pred, dtype=torch.long)
                    sil_mask_ref  = torch.zeros_like(ph_ref_spk, dtype=torch.long)
                    for sid_ in sil_ids:
                        sil_mask_pred[ph_pred == sid_] = 1
                        sil_mask_ref[ ph_ref_spk == sid_] = 1

                    z_pred = torch.log1p(dur_pred.float())
                    z_ref  = torch.log1p(dur_ref_spk.float())

                    # 静音对齐
                    if sil_mask_pred.sum() > 0 and sil_mask_ref.sum() > 0:
                        diff_sil = z_ref[sil_mask_ref == 1].mean() - z_pred[sil_mask_pred == 1].mean()
                        z_pred[sil_mask_pred == 1] += diff_sil

                    # 非静音对齐
                    non_pred = (sil_mask_pred != 1)
                    non_ref  = (sil_mask_ref  != 1)
                    if non_pred.sum() > 0 and non_ref.sum() > 0:
                        diff_non = z_ref[non_ref].mean() - z_pred[non_pred].mean()
                        z_pred[non_pred] += diff_non

                    # 还原为整数
                    dur_pred = torch.expm1(z_pred).clamp_min(0)
                    d_floor  = torch.floor(dur_pred)
                    frac     = (dur_pred - d_floor).clamp(0, 1)
                    dur_pred = (d_floor + torch.bernoulli(frac)).to(torch.int)
                # ===== normalize_dur 结束 =====

                # 扰动 + 裁剪
                if dur_disturb and dur_disturb > 0:
                    disturb_choice = (torch.rand_like(dur_pred.float()) > 0.5).float()
                    disturb_r = 1 + torch.rand_like(dur_pred.float()) * dur_disturb
                    dur_pred = torch.round(
                        dur_pred.float() * disturb_r * disturb_choice +
                        dur_pred.float() / disturb_r * (1 - disturb_choice)
                    ).to(torch.int)

                dur_pred = dur_pred.clamp(0, dur_max)
                results[ci] = dur_pred

        return results


    def forward(self, resource_context, input_text, time_step,
                w_all=None, w_txt=None, w_cap=None, w_ref=None, w_spk=None,
                seq_cfg_w: Optional[List[float]] = None,
                speech_rate=1, timestep_annealing_w=(1.0, 0.0, 1.0),
                return_timestamp=True, timestamp_postprocess=False,
                return_format='wav', custom_ph_table=None, dur_disturb=0.1, dur_alpha=1.0,
                num_parallel_workers=5, use_sa_frontend=True,
                normalize_dur: bool = True,            # NEW: 语速归一
                use_amo_sampler: bool = False,         # NEW: AMO 采样开关
                prefer_onepass_dur: bool = True,   # ← 新增：优先尝试“一步推完”
                **kwargs):
        """
        当 seq_cfg_w 长度为：
        - 2：走 3 路（[all, txt, uncond]），递进式（推荐）
        - 4：走 5 路（[all, txt, cap, ref, uncond]）
        - 其他/None：请传入合法权重（建议 [1.5, 3.0]）
        采样相关新增：
        - normalize_dur: 用参考段的静/非静音 log 时长均值对齐当前段，拉回语速（非 old dur 时生效）
        - use_amo_sampler: 透传到 DiT 的 inference，启用 AMO 采样
        """
        device = self.device
        profile = os.environ.get('MEGA_PROFILE', 'false').strip().lower() == 'true'

        with torch.inference_mode():
            # 0) 清洗输入
            input_text = ''.join(c for c in input_text if c.isprintable())
            if not input_text.strip():
                raise RuntimeError('输入为空，输入不合法')

            # 1) 解析（支持 <S{sid}> 与 <SPK>sid</SPK>）
            raw_text_total = input_text
            clean_text_total = re.sub(r"</?\s*S\s*\d+\s*>", "", raw_text_total)
            clean_text_total = re.sub(r"<\s*SPK\s*>\s*\d+\s*<\s*/\s*SPK\s*>", "", clean_text_total)
            spk_segs = self._parse_dialogue_segments(raw_text_total)  # [(sid, content)]
            if len(spk_segs) == 0:
                raise RuntimeError("未解析到任何说话人片段，请检查输入（支持 <S1>...</S1> 或 <SPK>1</SPK>...）。")

            # 2) normalize + chunk
            ssml_root = SSML(clean_text_total); ssml_root.rate = float(speech_rate)
            ph_replace_table = deepcopy(self.ph_replace_table)
            custom_ph_table = kwargs.get('custom_ph_table', None)
            if custom_ph_table is not None:
                ph_replace_table.update(custom_ph_table)

            text_chunks, chunk_spk_ids, chunk_raw_for_dit = [], [], []
            for sid, seg_content in spk_segs:
                sub_ssml = SSML(seg_content); sub_ssml.rate = ssml_root.rate
                sub_chunks = self.preprocess_text(sub_ssml, ph_replace_table, use_sa_frontend)
                for ch in sub_chunks:
                    text_chunks.append(ch)
                    chunk_spk_ids.append(int(sid))
                    chunk_raw_for_dit.append(f"<S{sid}>{ch.text_str}</S{sid}>")
            if len(text_chunks) == 0:
                raise RuntimeError("文本经 normalize/切分后为空。")

            # 3) 逐 chunk 做 G2P（phone/tone）
            chunk_items = []
            spk_mask_ph_list, words_ts_list = [], []
            for ci, (ch, sid) in enumerate(zip(text_chunks, chunk_spk_ids)):
                if not use_sa_frontend:
                    with model_lock(self.lock):
                        ph_pred, tone_pred = self.g2p(ch.text_str)
                    ph_pred, tone_pred, ph2word = self.refine_ph_tone(ch, ph_pred, tone_pred)
                else:
                    from modules.tts.frontend_lm.sa_frontend import call_sa_frontend
                    sa_ret = call_sa_frontend(ch.sa_ssml_str, debug=0)
                    if sa_ret is None:
                        print(f'| 跳过非法片段 #{ci}')
                        continue
                    text_sa, ph_tokens, tone_tokens, _ = sa_ret
                    new_text = SSML(text_sa); new_text.rate = ch.rate
                    new_text.pause_at_start = ch.pause_at_start; new_text.pause_at_end = ch.pause_at_end
                    ch = new_text
                    ph_pred  = torch.LongTensor(self.ling_dict['phone'].encode(' '.join(ph_tokens)))[None].to(device)
                    tone_pred= torch.LongTensor(self.ling_dict['tone'].encode(' '.join(tone_tokens)))[None].to(device)
                    ph2word = None

                chunk_items.append({'sid': int(sid), 'ch': ch, 'ph': ph_pred, 'tone': tone_pred, 'ph2word': ph2word})
                words_ts_list.append({'words': [], 'timestamps': []})
                spk_mask_ph_list.append(torch.full((1, ph_pred.shape[1]), int(sid), dtype=torch.long, device=device))

            if len(chunk_items) == 0:
                raise RuntimeError("所有 chunk 都被跳过，无法生成。")

            # 4) 时长预测
            if self.use_old_dur:
                dur_pred_by_chunk = self._dur_predict_old_ar_grouped_by_speaker(
                    chunk_items, resource_context, dur_disturb=dur_disturb
                )
            else:
                dur_pred_by_chunk = None
                # —— 一步推完整段 + spk_ids（2spk 兼容）——
                if prefer_onepass_dur:
                    dur_pred_by_chunk = self._dur_predict_onepass_with_spk(
                        chunk_items, resource_context,
                        dur_disturb=dur_disturb, normalize_dur=normalize_dur
                    )

                # —— 逐段（你原先的新 dur_lm 路径）——
                if dur_pred_by_chunk is None:
                    try:
                        dur_pred_by_chunk = self._dur_predict_per_chunk_by_speaker(
                            chunk_items, resource_context,
                            dur_disturb=dur_disturb, normalize_dur=normalize_dur
                        )
                    except TypeError:
                        # 兼容无 normalize_dur 的旧签名
                        dur_pred_by_chunk = self._dur_predict_per_chunk_by_speaker(
                            chunk_items, resource_context,
                            dur_disturb=dur_disturb
                        )


            # 5) 后处理 + vq 对齐 + 词级时间戳
            ph_list_all, tone_list_all, dur_list_all = [], [], []
            vqs = hparams.get('vq_stride', 8)
            for ci, it in enumerate(chunk_items):
                sid       = it['sid']
                ch        = it['ch']
                ph_pred   = it['ph']
                tone_pred = it['tone']
                ph2word   = it['ph2word']
                dur_pred  = dur_pred_by_chunk[ci].to(device)

                # 语速
                dur_pred = torch.round(dur_pred / ch.rate).int()
                dur_pred = dur_pred.clamp(0, self.hp_dur_model.get('dur_code_size', self.hp_dur_model.get('dur_max_value', 128)) - 1)
                if ci < len(chunk_items) - 1:
                    dur_pred[:, -1] = dur_pred[:, -1] + 30
                else:
                    dur_pred[:, -1] = dur_pred[:, -1].clamp(32, 80)
                for sil_token in [148, 153, 166, 145]:
                    dur_pred[ph_pred==sil_token] = dur_pred[ph_pred==sil_token].clamp_min(32)
                for sil_token in [163, 165]:
                    dur_pred[ph_pred==sil_token] = dur_pred[ph_pred==sil_token].clamp_min(16)
                dur_pred[:, 0] = max(8, int(dur_pred[:, 0]))

                if not use_sa_frontend:
                    ph_pred, tone_pred, dur_pred, ph2word = self.add_breaks(
                        ch, ph_pred, tone_pred, dur_pred, ph2word, break_token=163, break_tone=3
                    )
                    if return_timestamp and ph2word is not None:
                        try:
                            words_ts = self.make_word_timestamps(ch, dur_pred.squeeze().cpu().numpy(), ph2word)
                        except Exception:
                            words_ts = {'words': [], 'timestamps': []}
                    else:
                        words_ts = {'words': [], 'timestamps': []}
                    words_ts_list[ci] = words_ts

                dur_sum = int(dur_pred.sum().item())
                npad = vqs - dur_sum % vqs
                if npad < vqs:
                    dur_pred[:, -1] += npad

                ph_list_all.append(ph_pred)
                tone_list_all.append(tone_pred)
                dur_list_all.append(dur_pred)

            ph_pred_all   = torch.cat(ph_list_all,   dim=1)
            tone_pred_all = torch.cat(tone_list_all, dim=1)
            dur_pred_all  = torch.cat(dur_list_all,  dim=1)
            dur_pred_all[:, -1] = dur_pred_all[:, -1] + 50
            mel2ph_pred_all = self.length_regulator(dur_pred_all).to(device)
            spk_ids_pred_all = torch.cat(spk_mask_ph_list, dim=1).long()

            # 词级时间戳拼接
            if return_timestamp:
                offsets = [0.0]
                for i in range(len(dur_list_all)-1):
                    offsets.append(offsets[-1] + float(dur_list_all[i].sum().item())/100.0)
                words_all, ts_all = [], []
                for (wts, off) in zip(words_ts_list, offsets):
                    words_all.extend(wts['words'])
                    ts_all.extend([[a+off, b+off] for (a,b) in wts['timestamps']])
                words_timestamps = {'words': words_all, 'timestamps': ts_all}
                words_timestamps_post = None
            else:
                words_timestamps = words_timestamps_post = None

            # 6) Caption & Text（整段：统一 <S{sid}>...）
            ref_segs = self._parse_dialogue_segments(resource_context['text_ref_raw'])
            seq_for_all = ref_segs + [(sid, ch.text_str) for sid, ch in zip(chunk_spk_ids, text_chunks)]

            def _pack_S_markup(segs: List[Tuple[int,str]]) -> str:
                out = []
                for sid_i, content_i in segs:
                    content_i = (content_i or '').strip()
                    if not content_i:
                        continue
                    out.append(f'<S{sid_i}>{content_i}</S{sid_i}>')
                return ''.join(out)

            caption_merged_all = _pack_S_markup(seq_for_all)
            train_text_all     = caption_merged_all

            # 7) 组装 DiT 输入
            ph_ref     = resource_context['ph_ref'].to(device)
            tone_ref   = resource_context['tone_ref'].to(device)
            dur_ref    = resource_context['dur_ref'].to(device)
            mel2ph_ref = resource_context['mel2ph_ref'].to(device)
            vae_latent = resource_context['vae_latent'].to(device)
            spk_ids_ref= resource_context['spk_ids_ref'].to(device).long()

            # 在参考末尾插 0.5s 静音
            sil_token, sil_tone, gap_units = 145, 3, 10
            sil_ph = torch.full((1, 1), sil_token, dtype=ph_ref.dtype, device=device)
            sil_tn = torch.full((1, 1), sil_tone,  dtype=tone_ref.dtype, device=device)
            sil_du = torch.full((1, 1), gap_units, dtype=dur_ref.dtype, device=dur_ref.device)
            ph_ref_ext   = torch.cat([ph_ref,   sil_ph],   dim=1)
            tone_ref_ext = torch.cat([tone_ref, sil_tn],   dim=1)
            dur_ref_ext  = torch.cat([dur_ref,  sil_du],   dim=1)
            new_idx = int(ph_ref.shape[1] + 1)
            mel2ph_gap = torch.full((mel2ph_ref.shape[0], gap_units),
                                    new_idx, dtype=mel2ph_ref.dtype, device=device)
            mel2ph_ref_ext = torch.cat([mel2ph_ref, mel2ph_gap], dim=1)
            last_sid = int(spk_ids_ref[0, -1].item()) if spk_ids_ref.numel() > 0 else 1
            spk_ids_ref_ext = torch.cat([
                spk_ids_ref,
                torch.full((1, 1), last_sid, dtype=spk_ids_ref.dtype, device=device)
            ], dim=1)

            # 目标侧拼接
            ph_seq   = torch.cat([ph_ref_ext,   ph_pred_all],   dim=1)
            tone_seq = torch.cat([tone_ref_ext, tone_pred_all], dim=1)
            en_tone_idx = ~((tone_seq == 4) | ((11 <= tone_seq) & (tone_seq <= 15)) | (tone_seq == 0))
            tone_seq[en_tone_idx] = 3
            spk_seq_base = torch.cat([spk_ids_ref_ext, spk_ids_pred_all], dim=1).long()
            if spk_seq_base.shape[1] != ph_seq.shape[1]:
                raise RuntimeError("spk_seq_base 与 ph_seq 长度不一致！")
            mel2ph_pred_full = torch.cat([mel2ph_ref_ext, mel2ph_pred_all + ph_ref_ext.shape[1]], dim=1)
            mel2ph_pred_full = mel2ph_pred_full[:, :mel2ph_pred_full.shape[1] // self.fm * self.fm]
            target_size = mel2ph_pred_full.shape[1] // 4

            # 编码 caption/text
            def _run_caption(caps, device_):
                inputs = self.caption_tokenizer(caps, padding=True, return_tensors="pt")
                ids, am = inputs.input_ids.to(device_), inputs.attention_mask.to(device_)
                embs = self.caption_encoder(ids, return_dict=False, attention_mask=am)[0]
                return embs * am[..., None], am
            with model_lock(self.lock):
                caption_embs_all, caption_mask_all = _run_caption([caption_merged_all], device)
            caption_lens_all = caption_mask_all.sum(-1)
            text_inputs_all = self.dit_text_tokenizer(train_text_all, padding=True, return_tensors='pt').to(device)
            txt_tokens_all = text_inputs_all['input_ids']; txt_mask_all = text_inputs_all['attention_mask'].bool()
            txt_tokens_all[~txt_mask_all] = self.cfg_mask_text_token

            # VAE ctx pad
            ctx_mask = torch.ones_like(vae_latent[:, :, 0:1])
            lat = F.pad(vae_latent, (0,0,0, target_size - vae_latent.size(1)), mode='constant', value=0)
            ctx_mask = F.pad(ctx_mask, (0,0,0, target_size - ctx_mask.size(1)), mode='constant', value=0)

            # 调试打印（可通过环境变量关闭）
            self._print_dit_inputs_debug(tag="FULL", caption_str=caption_merged_all,
                                        prompt_text=train_text_all, text_inputs=text_inputs_all)

            self._print_target_ph_durations_seconds(ph_ref, tone_ref, dur_ref, tag="MFA_REF")

            # 目标侧（生成）时长明细
            self._print_target_ph_durations_seconds(ph_pred_all, tone_pred_all, dur_pred_all, tag="TARGET")

            zeros_phone = torch.full_like(ph_seq,  self.cfg_mask_token_phone)
            zeros_tone  = torch.full_like(tone_seq, self.cfg_mask_token_tone)
            zeros_txt   = torch.full_like(txt_tokens_all, self.cfg_mask_text_token)
            zeros_cap   = torch.zeros_like(caption_embs_all)
            zeros_lat   = torch.zeros_like(lat)

            # 稀疏 mel2ph（若开启）
            mel2ph_sparse_1d = None
            if hparams.get('use_sparse_dur', False):
                dur_concat = torch.cat([dur_ref_ext.to('cpu').long().squeeze(0),
                                        dur_pred_all.to('cpu').long().squeeze(0)], dim=0)
                dur_list = dur_concat.numpy().tolist()
                mel2ph_sparse_1d = compute_mel2aug_from_dur(
                    dur_list,
                    gap_mode=hparams.get('sparse_dur_mode', 'proportional'),
                    gap_frames=hparams.get('sparse_dur_frames', 4),
                    gap_alpha=hparams.get('sparse_dur_alpha', 0.2),
                    min_keep=hparams.get('sparse_dur_min_keep', 1),
                    keep_ratio=hparams.get('sparse_dur_keep_ratio'),
                    symmetric=hparams.get('sparse_dur_symmetric', True),
                )

            # 3/5 路打包
            def _pack_3way():
                phone   = torch.cat([ph_seq,          ph_seq,          zeros_phone], dim=0)
                tone    = torch.cat([tone_seq,        tone_seq,        zeros_tone ], dim=0)
                txt_tok = torch.cat([txt_tokens_all,  txt_tokens_all,  zeros_txt  ], dim=0)
                txt_msk = torch.cat([txt_mask_all] * 3, dim=0)
                cap     = torch.cat([caption_embs_all, caption_embs_all, zeros_cap], dim=0)
                cap_len = torch.cat([caption_lens_all] * 3, dim=0).long()
                lat_ctx = torch.cat([lat, zeros_lat, zeros_lat], dim=0)
                ctx_msk = torch.cat([ctx_mask] * 3, dim=0)
                m2p     = mel2ph_pred_full.repeat(3, 1)
                spk_ids = torch.cat([spk_seq_base,
                                    torch.zeros_like(spk_seq_base),
                                    torch.zeros_like(spk_seq_base)], dim=0).long()
                if mel2ph_sparse_1d is not None:
                    m2p_sparse = torch.stack([mel2ph_sparse_1d]*3).to(device)
                    m2p_sparse = m2p_sparse[:, :m2p.shape[1]]
                else:
                    m2p_sparse = None
                return phone, tone, txt_tok, txt_msk, cap, cap_len, lat_ctx, ctx_msk, m2p, m2p_sparse, spk_ids

            def _pack_5way():
                phone   = torch.cat([ph_seq, ph_seq, zeros_phone, zeros_phone, zeros_phone], dim=0)
                tone    = torch.cat([tone_seq, tone_seq, zeros_tone,  zeros_tone,  zeros_tone ], dim=0)
                txt_tok = torch.cat([txt_tokens_all, txt_tokens_all, zeros_txt, zeros_txt, zeros_txt], dim=0)
                txt_msk = torch.cat([txt_mask_all] * 5, dim=0)
                cap     = torch.cat([caption_embs_all, zeros_cap, caption_embs_all, zeros_cap, zeros_cap], dim=0)
                cap_len = torch.cat([caption_lens_all] * 5, dim=0).long()
                lat_ctx = torch.cat([lat, zeros_lat, zeros_lat, lat, zeros_lat], dim=0)
                ctx_msk = torch.cat([ctx_mask] * 5, dim=0)
                m2p     = mel2ph_pred_full.repeat(5, 1)
                spk_ids = torch.cat([spk_seq_base,
                                    torch.zeros_like(spk_seq_base),
                                    torch.zeros_like(spk_seq_base),
                                    spk_seq_base,
                                    torch.zeros_like(spk_seq_base)], dim=0).long()
                if mel2ph_sparse_1d is not None:
                    m2p_sparse = torch.stack([mel2ph_sparse_1d]*5).to(device)
                    m2p_sparse = m2p_sparse[:, :m2p.shape[1]]
                else:
                    m2p_sparse = None
                return phone, tone, txt_tok, txt_msk, cap, cap_len, lat_ctx, ctx_msk, m2p, m2p_sparse, spk_ids

            use_2step = (seq_cfg_w is not None and len(seq_cfg_w) == 2)
            use_4step = (seq_cfg_w is not None and len(seq_cfg_w) == 4)
            if use_2step:
                ph_pack, tone_pack, txt_pack, txt_mask_pack, cap_pack, cap_lens_pack, \
                lat_pack, ctx_mask_pack, m2p_pack, m2p_sparse_pack, spk_pack = _pack_3way()
            elif use_4step:
                ph_pack, tone_pack, txt_pack, txt_mask_pack, cap_pack, cap_lens_pack, \
                lat_pack, ctx_mask_pack, m2p_pack, m2p_sparse_pack, spk_pack = _pack_5way()
            else:
                raise RuntimeError("未提供合法的 seq_cfg_w（长度需为 2 或 4），建议传入 [1.5, 3.0]。")

            inputs = {
                'phone': ph_pack, 'tone': tone_pack,
                'spk_ids': spk_pack,
                "lat_ctx": lat_pack * ctx_mask_pack, "ctx_mask": ctx_mask_pack,
                "mel2ph": m2p_pack,
                "txt_tokens": txt_pack, 'txt_mask': txt_mask_pack,
                "caption_emb": cap_pack, "caption_lens": cap_lens_pack,
            }
            if m2p_sparse_pack is not None:
                inputs["mel2ph_sparse"] = m2p_sparse_pack

            # ===== 采样：新增 use_amo_sampler 透传 =====
            with model_lock(self.lock):
                with torch.autocast(device_type='cuda', dtype=self.precision):
                    x = self.dit.inference(
                        inputs,
                        timesteps=time_step,
                        seq_cfg_w=seq_cfg_w,
                        timestep_annealing_w=timestep_annealing_w,
                        use_amo_sampler=use_amo_sampler  # NEW
                    )

            # 覆写参考前缀并解码
            x[:, :vae_latent.size(1)] = vae_latent
            with model_lock(self.lock):
                with torch.autocast(device_type='cuda', dtype=self.precision):
                    wav_pred = self.vae.decode(x)[0,0].to(torch.float32)

            hop_size = self.hp_vae['hop_size']; vae_stride = self.hp_vae['vae_stride']
            pre_samples = int(vae_latent.size(1) * vae_stride * hop_size)
            gap_samples = int(round(gap_units * 0.01 * self.sr))
            trim0 = min(pre_samples + gap_samples, int(wav_pred.shape[-1]))
            wav_pred = wav_pred[trim0:]

            if wav_pred.abs().max() > 1:
                wav_pred = wav_pred / (wav_pred.abs().max())
            wav_np = wav_pred.cpu().numpy()

            wav_bytes = to_wav_bytes(wav_np.astype(float), self.sr)
            if return_format == 'mp3':
                wav_bytes = wav_bytes_to_mp3_bytes(wav_bytes)

            ph_pred_list   = self.ling_dict['phone'].decode(ph_pred_all.squeeze().cpu().numpy()).split(' ')
            tone_pred_list = self.ling_dict['tone'].decode(tone_pred_all.squeeze().cpu().numpy()).split(' ')

            return MegaTTS3Output(
                wav_bytes=wav_bytes,
                wav=wav_np,
                words_timestamps=words_timestamps,
                words_timestamps_post=words_timestamps_post,
                duration=wav_np.shape[-1] / self.sr,
                ph_pred=ph_pred_list,
                tone_pred=tone_pred_list
            )



    def _clone_tensor_tree_to(self, obj, device, clone=True, detach=True):
        """
        递归把任意嵌套结构（dict/list/tuple/Tensor/None）里的 Tensor
        转成叶子副本并迁移到 device：
            Tensor -> (detach? t.detach(): t).(clone? clone(): t).to(device)
        其余类型原样返回（或递归处理）。
        """
        import torch
        if not isinstance(device, torch.device):
            device = torch.device(device)

        if torch.is_tensor(obj):
            t = obj
            if detach:
                t = t.detach()
            if clone:
                t = t.clone()
            return t.to(device, non_blocking=True)

        if isinstance(obj, dict):
            return {k: self._clone_tensor_tree_to(v, device, clone, detach) for k, v in obj.items()}

        if isinstance(obj, list):
            return [self._clone_tensor_tree_to(v, device, clone, detach) for v in obj]

        if isinstance(obj, tuple):
            return tuple(self._clone_tensor_tree_to(v, device, clone, detach) for v in obj)

        # 其他（None、标量等）
        return obj

    def _build_old_dur_prefill_by_speaker(self, ref_by_spk):
        """
        为每位说话人分别构建 ARDurPredictor 的 prefill：
        - 仅用该说话人的参考 ph/tone/dur 做 prefill；
        - 生成并返回 {'incremental_state','ctx_dur_tokens','last_pos'} 三件套（CPU 保存，便于并发安全）。
        返回: {sid: {'incremental_state':..., 'ctx_dur_tokens':Tensor[1,L], 'last_pos':int}}
        """
        import torch
        device = self.device
        hp = getattr(self, "hp_dur_model", {})
        dur_prefill = {}

        for sid, pack in ref_by_spk.items():
            ph_ref   = pack['ph_ref'].to(device)
            tone_ref = pack['tone_ref'].to(device)
            # 统一 2D 形状
            dur_ref  = pack['dur_ref']
            dur_ref  = dur_ref if dur_ref.dim() == 2 else dur_ref[None]
            dur_ref  = dur_ref.to(device)

            # dur_ref -> mel2ph -> ctx dur token
            mel2ph_ref = self.length_regulator(dur_ref).to(device)
            mel2ph_ref = mel2ph_ref[:, :mel2ph_ref.size(1)//self.fm*self.fm]
            dur_tokens_2d = mel2token_to_dur(
                mel2ph_ref, ph_ref.shape[1]
            ).clamp(max=hp.get('dur_code_size', hp.get('dur_max_value', 128)) - 1) + 1

            ctx_dur_tokens = dur_tokens_2d.clone().flatten(0, 1).to(device)
            txt_tokens_flat = ph_ref.flatten(0, 1)
            ctx_dur_tokens = ctx_dur_tokens[txt_tokens_flat > 0][None]  # [1, Lctx]
            last_pos = ctx_dur_tokens.shape[1]
            spk_pos_ids_flat = torch.arange(0, last_pos, device=device)[None, :].long()

            # 预填充，拿到该说话人的增量状态
            with model_lock(self.lock):
                _, inc_state = self.dur_model.infer(
                    ph_ref, {'tone': tone_ref},
                    None, None, None,
                    ctx_vqcodes=ctx_dur_tokens,
                    spk_pos_ids_flat=spk_pos_ids_flat,
                    return_state=True
                )

            # 存成 CPU 上的“叶子副本”
            safe_cpu_state = self._clone_tensor_tree_to(inc_state, torch.device('cpu'))

            dur_prefill[int(sid)] = {
                'incremental_state': safe_cpu_state,               
                'ctx_dur_tokens': ctx_dur_tokens.detach().to('cpu'),
                'last_pos': int(last_pos),
            }

        return dur_prefill


# =========================================
# Example (批量多组)
# =========================================

if __name__ == '__main__':
    if os.path.isfile('.env.local'):
        from dotenv import load_dotenv
        load_dotenv('.env.local')

    # 可按需修改
    dit_exp_name = 'checkpoints/251103_megatts3_dit_dialogue'
    # dit_exp_name = 'checkpoints/251103_megatts3_dit_sparse_dialogue'
    dur_exp_name = 'checkpoints/251104_dur_lm_multispk'
    frontend_exp_name = 'checkpoints/250923_lm_mfa_seq2seq_small_wavlmlarge_long_robust'

    # 推理超参
    time_step = 100
    seq_cfg_w = [1.5,3.0]
    # seq_cfg_w = [2.5,5.0]

    # ============ 多组输入示例 ============
    groups = [
        {
            "ref_audios": (
                "/mnt/bn/sa-ag-data/wenchenyuhao/data/megatts_infer/data/单人ref_2_normalized/背景噪声几乎没有/1_vocal.wav",
                "/mnt/bn/sa-ag-data/wenchenyuhao/data/megatts_infer/data/单人ref_2_normalized/背景噪声几乎没有/4_vocal.wav",
            ),
            "text": "<SPK>1</SPK>我家这餐馆，菜味儿正宗分量也足，怎么饭点总坐不满啊？ <SPK>2</SPK>你试过做抖音团购套餐吗？ <SPK>1</SPK>抖音团购？哎呦，那是不是得花钱推广，还怕没人买亏了本啊？ <SPK>2</SPK>不用花推广费，套餐定价灵活还能吸引新客。 <SPK>1</SPK>真不用花钱？这靠谱吗？ <SPK>2</SPK>现在点击视频下方链接，就能 0 元上架抖音团购套餐，官方还会给本地流量推荐，帮你把附近想吃的人都引过来。像双人餐、家庭餐都能做，还能设置到店核销，不怕跑单。 <SPK>1</SPK>这听着行啊，那团购套餐咋设计啊？ <SPK>2</SPK>平台有现成的套餐模板，你按自家招牌菜搭配就行，还能自动生成图文海报，就算不会做宣传，用户刷到直接就能下单，用完都说好还会带朋友来。 <SPK>1</SPK>那我现在就去弄这个团购套餐！ <SPK>2</SPK>你们做餐饮的也赶紧点下方链接，上架抖音团购，让店里天天满座！"
        },
        {
            "ref_audios": (
                "/mnt/bn/sa-ag-data/wenchenyuhao/data/megatts_infer/data/单人ref_2_normalized/背景噪声几乎没有/3_vocal.wav",
                "/mnt/bn/sa-ag-data/wenchenyuhao/data/megatts_infer/data/单人ref_2_normalized/背景噪声几乎没有/1_vocal.wav",
            ),
            "text": "<SPK>1</SPK>我家这餐馆，菜味儿正宗分量也足，怎么饭点总坐不满啊？ <SPK>2</SPK>你试过做抖音团购套餐吗？ <SPK>1</SPK>抖音团购？哎呦，那是不是得花钱推广，还怕没人买亏了本啊？ <SPK>2</SPK>不用花推广费，套餐定价灵活还能吸引新客。 <SPK>1</SPK>真不用花钱？这靠谱吗？ <SPK>2</SPK>现在点击视频下方链接，就能 0 元上架抖音团购套餐，官方还会给本地流量推荐，帮你把附近想吃的人都引过来。像双人餐、家庭餐都能做，还能设置到店核销，不怕跑单。 <SPK>1</SPK>这听着行啊，那团购套餐咋设计啊？ <SPK>2</SPK>平台有现成的套餐模板，你按自家招牌菜搭配就行，还能自动生成图文海报，就算不会做宣传，用户刷到直接就能下单，用完都说好还会带朋友来。 <SPK>1</SPK>那我现在就去弄这个团购套餐！ <SPK>2</SPK>你们做餐饮的也赶紧点下方链接，上架抖音团购，让店里天天满座！"
        },
        {
            "ref_audios": (
                "/mnt/bn/sa-ag-data/wenchenyuhao/data/megatts_infer/data/单人ref_2_normalized/背景噪声几乎没有/1_vocal.wav",
                "/mnt/bn/sa-ag-data/wenchenyuhao/data/megatts_infer/data/单人ref_2_normalized/背景噪声几乎没有/4_vocal.wav",
                ),
            "text": "<SPK>1</SPK>最近愁死了，我家那家电店线下客流越来越少，线上又没头绪，这月业绩都要完不成了。 <SPK>2</SPK>你咋不试试在抖音开直播卖家电啊？我隔壁那卖油烟机的老王，上个月开直播，单场就卖了二十多台，业绩直接翻番。 <SPK>1</SPK>老王也做了？可我连直播软件都不会用，还得买设备请人，这成本太高了吧？ <SPK>2</SPK>根本不用！他就用自己的手机播，抖音商家后台有免费的直播工具，连补光灯都不用额外买，对着家电演示功能就行，他自己一个人就能搞定。 <SPK>1</SPK>那播的时候没人看咋办啊？播半天卖不出去多尴尬。 <SPK>2</SPK>你点视频下方那链接，开通商家直播权限后，抖音会自动把你推给本地想换家电的人。老王还会在直播里搞点“限时立减 200”“下单送延保”的活动，用户看着实惠就直接下单了，还有人在评论区问安装，他当场就能对接售后，特别方便。 <SPK>1</SPK>这么简单？那我播的时候该说点啥啊，我嘴笨怕说不清楚。 <SPK>2</SPK>平台有现成的家电直播话术模板，比如讲冰箱容量、洗衣机能耗这些，照着念就行，还能提前拍好家电使用视频，直播时插播进去，用户一看就明白。 <SPK>1</SPK>那我今晚就去研究下，争取下周就开播！ <SPK>2</SPK>赶紧的，你点下方链接开通权限，有啥不懂的再问我，保准你下个月业绩也能上去！"
        },
        {
            "ref_audios": (
                "/mnt/bn/sa-ag-data/wenchenyuhao/data/megatts_infer/data/单人ref_2_normalized/背景噪声几乎没有/3_vocal.wav",
                "/mnt/bn/sa-ag-data/wenchenyuhao/data/megatts_infer/data/单人ref_2_normalized/背景噪声几乎没有/1_vocal.wav",
                ),
            "text": "<SPK>1</SPK>最近愁死了，我家那家电店线下客流越来越少，线上又没头绪，这月业绩都要完不成了。 <SPK>2</SPK>你咋不试试在抖音开直播卖家电啊？我隔壁那卖油烟机的老王，上个月开直播，单场就卖了二十多台，业绩直接翻番。 <SPK>1</SPK>老王也做了？可我连直播软件都不会用，还得买设备请人，这成本太高了吧？ <SPK>2</SPK>根本不用！他就用自己的手机播，抖音商家后台有免费的直播工具，连补光灯都不用额外买，对着家电演示功能就行，他自己一个人就能搞定。 <SPK>1</SPK>那播的时候没人看咋办啊？播半天卖不出去多尴尬。 <SPK>2</SPK>你点视频下方那链接，开通商家直播权限后，抖音会自动把你推给本地想换家电的人。老王还会在直播里搞点“限时立减 200”“下单送延保”的活动，用户看着实惠就直接下单了，还有人在评论区问安装，他当场就能对接售后，特别方便。 <SPK>1</SPK>这么简单？那我播的时候该说点啥啊，我嘴笨怕说不清楚。 <SPK>2</SPK>平台有现成的家电直播话术模板，比如讲冰箱容量、洗衣机能耗这些，照着念就行，还能提前拍好家电使用视频，直播时插播进去，用户一看就明白。 <SPK>1</SPK>那我今晚就去研究下，争取下周就开播！ <SPK>2</SPK>赶紧的，你点下方链接开通权限，有啥不懂的再问我，保准你下个月业绩也能上去！"
        },
        {
            "ref_audios": (
                "/mnt/bn/sa-ag-data/wenchenyuhao/data/megatts_infer/data/单人ref_2_normalized/背景噪声几乎没有/1_vocal.wav",
                "/mnt/bn/sa-ag-data/wenchenyuhao/data/megatts_infer/data/单人ref_2_normalized/背景噪声几乎没有/4_vocal.wav",
                ),
            "text": "<SPK>1</SPK>老板，我刷到你家实体店的衣服挺好看的，但我现在不在本地，想线上买又怕尺码不合适，退换货麻烦。 <SPK>2</SPK>你可以直接在我家抖音店铺橱窗下单啊，专门解决异地购物的问题，特别方便。 <SPK>1</SPK>抖音店铺橱窗？我之前没试过，能看清衣服面料和上身效果吗？万一买回来和图片不一样咋办？ <SPK>2</SPK>放心，橱窗里每件衣服都有 360 度细节视频，面料纹理、领口袖口都拍得清清楚楚，还附了真实顾客的穿搭反馈，和你在店里看的一模一样。 <SPK>1</SPK>那尺码咋选啊？我平时穿 M 码，怕不同款式版型不一样。 <SPK>2</SPK>每个商品下面都有详细的尺码表，还标了“适合体重和身高”，你要是拿不准，直接在橱窗里点“客服”，发你的身材数据，我马上给你推荐精准尺码，退换货还能免运费。 <SPK>1</SPK>那下单后多久能到啊？我着急穿。 <SPK>2</SPK>你点视频下方链接进橱窗下单，当天就能发货，默认发顺丰，大部分地区 3 天内就能到，收到货不满意，7 天内随时能退，不用你跑实体店。 <SPK>1</SPK>那我现在就点链接去选，选好直接下单！ <SPK>2</SPK>其他想线上买衣服的朋友，也赶紧点下方链接进橱窗，选款、选码、售后都省心，和逛实体店一样放心！"
        },
        {
            "ref_audios": (
                "/mnt/bn/sa-ag-data/wenchenyuhao/data/megatts_infer/data/单人ref_2_normalized/背景噪声几乎没有/3_vocal.wav",
                "/mnt/bn/sa-ag-data/wenchenyuhao/data/megatts_infer/data/单人ref_2_normalized/背景噪声几乎没有/1_vocal.wav",
                ),
            "text": "<SPK>1</SPK>老板，我刷到你家实体店的衣服挺好看的，但我现在不在本地，想线上买又怕尺码不合适，退换货麻烦。 <SPK>2</SPK>你可以直接在我家抖音店铺橱窗下单啊，专门解决异地购物的问题，特别方便。 <SPK>1</SPK>抖音店铺橱窗？我之前没试过，能看清衣服面料和上身效果吗？万一买回来和图片不一样咋办？ <SPK>2</SPK>放心，橱窗里每件衣服都有 360 度细节视频，面料纹理、领口袖口都拍得清清楚楚，还附了真实顾客的穿搭反馈，和你在店里看的一模一样。 <SPK>1</SPK>那尺码咋选啊？我平时穿 M 码，怕不同款式版型不一样。 <SPK>2</SPK>每个商品下面都有详细的尺码表，还标了“适合体重和身高”，你要是拿不准，直接在橱窗里点“客服”，发你的身材数据，我马上给你推荐精准尺码，退换货还能免运费。 <SPK>1</SPK>那下单后多久能到啊？我着急穿。 <SPK>2</SPK>你点视频下方链接进橱窗下单，当天就能发货，默认发顺丰，大部分地区 3 天内就能到，收到货不满意，7 天内随时能退，不用你跑实体店。 <SPK>1</SPK>那我现在就点链接去选，选好直接下单！ <SPK>2</SPK>其他想线上买衣服的朋友，也赶紧点下方链接进橱窗，选款、选码、售后都省心，和逛实体店一样放心！"
        },
    ]

    # ============ 初始化推理器（仅传“建模/加载”参数） ============
    infer_ins = MegaTTS3DiTInfer(
        device='cuda:0',
        dit_exp_name=dit_exp_name,
        dur_exp_name=dur_exp_name,
        frontend_exp_name=frontend_exp_name,
        use_old_aligner=False,
        use_old_dur=True,
        max_ref_duration=60,
        use_tqdm=False  # 并行时关闭 tqdm，避免多线程输出干扰
    )

    kill_void()

    # 输出目录
    out_dir = f'/mnt/bn/sa-ag-data/wenchenyuhao/data/2spk_infer/infer_out/{Path(dit_exp_name).stem}/20251201'
    os.makedirs(out_dir, exist_ok=True)

    # 小工具：安全的保存名
    def _safe_name(*parts, maxlen=80):
        base = "+".join(parts)
        base = re.sub(r'[^0-9a-zA-Z\u4e00-\u9fa5\-\+\._\[\]\(\)]+', '_', base)
        return base[:maxlen]

    # 单个样本的处理函数（在子线程中运行）
    def _process_one(args):
        idx, g = args
        try:
            ref_paths = g["ref_audios"]
            assert isinstance(ref_paths, (list, tuple)) and len(ref_paths) == 2, "ref_audios 必须是长度为2的 (path_a, path_b)"

            # 以“字节流”读入，符合 preprocess 的接口
            with open(ref_paths[0], 'rb') as fa, open(ref_paths[1], 'rb') as fb:
                ref_bytes = [fa.read(), fb.read()]

            # 可选：传入两段 clean 文本（不含标签），否则内部会用 ASR 自动生成
            ref_texts_pair = g.get("ref_texts", None)
            if ref_texts_pair is not None:
                assert isinstance(ref_texts_pair, (list, tuple)) and len(ref_texts_pair) == 2, "ref_texts 必须是长度为2的 (txt_a, txt_b)"

            print(f'| [{idx+1}/{len(groups)}] Start preprocess -> {ref_paths[0]} & {ref_paths[1]}')
            resource_context = infer_ins.preprocess(ref_bytes, ref_texts=ref_texts_pair)

            # 目标文本（支持 <S{sid}> 或 <SPK>sid</SPK>）
            text = g["text"]

            print(f'| [{idx+1}/{len(groups)}] Start generation')
            output = infer_ins.forward(
                resource_context, text,
                time_step=time_step,
                seq_cfg_w=seq_cfg_w,
                timestep_annealing_w=(0.6, 0.6, 1.0),
                use_sa_frontend=True,
                return_format='wav',

                # ===== 生成/采样相关超参：这里传！=====
                use_amo_sampler=False,
                speech_rate=1.0,
                custom_ph_table={
                    'en': {'@': 'at', '&': 'and'},
                    'zh': {'@': '艾特', '&': '和'}
                },
                dur_disturb=0.2,
                num_parallel_workers=1,
                normalize_dur=True,
                prefer_onepass_dur=True
            )

            # 解析对话片段并生成可读标签
            def build_dialogue_label(text: str,
                                     clauses_per_seg: int = 2,
                                     per_seg_chars: int = 24,
                                     max_segments: int = 2):
                segs = infer_ins._parse_dialogue_segments(text)  # [(sid, content), ...]
                parts = []
                for sid, content in segs[:max_segments]:
                    s = re.sub(r'\s+', ' ', (content or '')).strip()
                    clauses = [c for c in re.split(r'[，。,\.！!？?\n]', s) if c]
                    summary = '_'.join(clauses[:clauses_per_seg])[:per_seg_chars]
                    parts.append(f'[{sid}]{summary}')
                return '_'.join(parts)

            label = build_dialogue_label(text,
                                         clauses_per_seg=2,
                                         per_seg_chars=24,
                                         max_segments=2)

            save_name = _safe_name(
                f"{Path(ref_paths[0]).parent.name}{Path(ref_paths[0]).stem.replace('_vocal', '')}",
                f"{Path(ref_paths[1]).parent.name}{Path(ref_paths[1]).stem.replace('_vocal', '')}",
                label
            )
            save_path = f'{out_dir}/{save_name}.wav'
            save_wav_bytes(output.wav_bytes, save_path)
            print(f'| Done -> {save_path}')
            return (idx, save_path, None)
        except Exception as e:
            print(f'| [ERROR][{idx+1}/{len(groups)}] {e}')
            traceback.print_exc()
            return (idx, None, e)

    # ============ 并行执行 ============
    from concurrent.futures import ThreadPoolExecutor

    default_workers = min(4, max(1, len(groups)))
    max_workers = int(os.getenv("MEGA_INFER_WORKERS", str(default_workers)))

    jobs = list(enumerate(groups))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_process_one, jobs))

    succ = [r for r in results if r[1] is not None]
    fail = [r for r in results if r[2] is not None]
    print(f'| ALL DONE: success={len(succ)} fail={len(fail)} out_dir={out_dir}')
