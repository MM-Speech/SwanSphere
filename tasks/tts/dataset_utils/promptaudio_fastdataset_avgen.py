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
from utils.commons.io import load_samples_from_tsv, load_samples_from_jsonl
from utils.commons.base_shm_dataset import BaseFalconReaderShmDataset, get_from_global_stores, save_samples_to_shm
from utils.commons.dataset_utils import collate_xd, SkipLogger
from utils.commons.tensor_utils import convert_to_tensor, convert_to_np
from utils.commons.tos_utils_v2 import TosClient
from utils.commons.hdfs_utils import HDFSClient
from utils.dataset.batcher import BucketBatcher
from utils.audio.vad import build_vad_model, run_vad_trim
from utils.audio.align import mel2token_to_dur
from utils.text.split_text import get_word_list, remove_spaces_between_chinese
from utils.text import is_chinese, is_english

from tasks.tts.dataset_utils.tts_datasets import MegaTTSDataset, FrontendLMDataset
# from modules.tts.ar_dur.commons.align_ops import compute_mel2aug_from_dur
from modules.tts.ar_dur.commons.nar_tts_modules import LengthRegulator
from tasks.tts.dataset_utils.tts_fastdataset_v2 import get_hdfs_file,safe_read_path,BaseTTSShmDataset

DEBUG = False


# ========= 预编译 regex pattern =========
# 空白压缩
_SPACE_RE = re.compile(r'\s+')

# <Audio>/<I> 统一成 <I>
_AUDIO_I_OPEN_TAG_RE = re.compile(
    r'<\s*(?:Audio|I)\s*>',
    flags=re.IGNORECASE
)
_AUDIO_I_CLOSE_TAG_RE = re.compile(
    r'</\s*(?:Audio|I)\s*>',
    flags=re.IGNORECASE
)

# <S1>/<S2>/<W> 统一成 <W>
_W_GROUP_OPEN_TAG_RE = re.compile(
    r'<\s*(?:S1|S2|W)\s*>',
    flags=re.IGNORECASE
)
_W_GROUP_CLOSE_TAG_RE = re.compile(
    r'</\s*(?:S1|S2|W)\s*>',
    flags=re.IGNORECASE
)

# 抽取 <W>...</W> 的内容
_W_TAG_RE = re.compile(
    r'<\s*W\s*>(.*?)</\s*W\s*>',
    flags=re.IGNORECASE | re.DOTALL
)


def _norm_spaces_caption(s: str) -> str:
    if not isinstance(s, str):
        return ''
    s = s.replace('\r\n', '\n').replace('\r', '\n')
    # 使用预编译 _SPACE_RE
    s = _SPACE_RE.sub(' ', s)
    return s.strip()


def _normalize_tags(s: str) -> str:
    """
    把 <Audio>/<I> 统一成 <I>，把 <S1>/<S2>/<W> 统一成 <W>。
    大小写和标签里的空格都不敏感。
    """
    if not isinstance(s, str):
        return s
    # <Audio> / <I> -> <I>
    s = _AUDIO_I_OPEN_TAG_RE.sub('<I>', s)
    s = _AUDIO_I_CLOSE_TAG_RE.sub('</I>', s)
    # <S1> / <S2> / <W> -> <W>
    s = _W_GROUP_OPEN_TAG_RE.sub('<W>', s)
    s = _W_GROUP_CLOSE_TAG_RE.sub('</W>', s)
    return s


def _normalize_text_field_caption(val):
    if isinstance(val, list):
        try:
            val = ''.join(map(str, val))
        except Exception:
            val = ' '.join(map(str, val))
    return _norm_spaces_caption(val) if isinstance(val, str) else ''


def _build_text_from_caption_s1s2(caption: str) -> str:
    """
    从 caption 中按顺序提取 <W>...</W> 片段，
    把相邻的 <W> 片段合并成一段纯文本（用空格拼接）。
    若没有任何 W 片段则返回 ""。
    """
    if not isinstance(caption, str) or not caption.strip():
        return ""

    # 统一标签：<Audio>/<I> -> <I>，<S1>/<S2>/<W> -> <W>
    caption = _normalize_tags(caption)

    segments = []
    # 使用预编译的 _W_TAG_RE
    for inner in _W_TAG_RE.findall(caption):
        inner_norm = _norm_spaces_caption(inner)
        if not inner_norm:
            continue
        segments.append(inner_norm)

    if not segments:
        return ""

    # 直接用空格拼起来
    return " ".join(segments)


def _build_text_from_caption_for_tts(caption: str) -> str:
    """
    从 caption 中抽取所有 <W>...</W> 内容拼接成纯文本；若没有则返回 ""。
    """
    if not isinstance(caption, str) or not caption.strip():
        return ""

    # 统一标签：<Audio>/<I> -> <I>，<S1>/<S2>/<W> -> <W>
    caption = _normalize_tags(caption)

    # 使用预编译的 _W_TAG_RE
    matches = _W_TAG_RE.findall(caption)
    chunks = []
    for m in matches:
        norm = _norm_spaces_caption(m)
        if norm:
            chunks.append(norm)

    if chunks:
        # 直接拼成一行纯文本，不再包 <W>...</W>
        return " ".join(chunks)
    else:
        return ""


def _build_caption_from_subjects_narration(meta):
    """从 new_caption / subjects / narration 构建 caption:
    形如 "Subjects:...,NARRATION:..."，并且把 <I></I> 标签替换为 <Audio></Audio>。
    """
    new_cap = meta.get('new_caption')
    subjects = ''
    narration = ''
    if isinstance(new_cap, dict):
        subjects = _normalize_text_field_caption(new_cap.get('subjects', ''))
        narration = _normalize_text_field_caption(new_cap.get('narration', ''))

    if not subjects:
        subjects = _normalize_text_field_caption(meta.get('subjects', ''))
    if not narration:
        narration = _normalize_text_field_caption(meta.get('narration', ''))

    parts = []
    if subjects:
        parts.append(f"Subjects: {subjects}")
    if narration:
        parts.append(f"Narration: {narration}")
    caption = ' '.join(parts)

    if not caption:
        return ''

    # 这里仍然用统一标签逻辑，如果你需要也可以保留纯粹 subjects/narration 文本
    caption = _normalize_tags(caption)
    return caption


def valid_item_kv(item, k):
    return k in item and item[k] is not None


def merge_A2B(A2B, B_lens):
    token_lens_cumsum = np.cumsum([0] + B_lens[:-1])
    token_lens_cumsum = torch.LongTensor(token_lens_cumsum)
    for i in range(len(B_lens)):
        A2B[i] = A2B[i] + token_lens_cumsum[i]
    A2B = torch.cat(A2B, 0)
    return A2B


def raw_text_process(txt, wav=None, wav_len=None):
    txt = txt.strip()
    if txt.startswith('sil '):
        txt = txt[4:]
    txt = txt.replace(' sil ', ' ')
    txt = txt.replace(' ,', ',').replace(',,', ',').replace(' ，', '，').replace('， ', '，')
    txt = txt.replace(' .', '.').replace(' 。', '。').replace('。 ', '。').replace('。 ', '。')
    txt = txt.replace(' ?', '?').replace(' ？', '？').replace('？ ', '？').replace('？ ', '？')
    txt = txt.replace(' !', '!').replace(' ！', '！').replace('！ ', '！').replace('！ ', '！')
    txt = txt.replace(' ;', ',').replace(' ；', '，').replace('； ', '，').replace('； ', '，').replace(';', ',').replace('；', '，')
    txt = txt.replace(' :', ',').replace(' ：', '，').replace('： ', '，').replace(':', ',').replace('：', '，')
    txt = txt.replace(' 、', '，').replace('、 ', '，').replace('、', '，')
    txt = txt.replace('"', '').replace('“', '').replace('”', '')
    txt = txt.replace('- ', ' ')
    txt = txt.replace('+', ' ')
    txt = txt.replace('，。', '。').replace('。，', '。')
    txt = txt.replace(':。', '。').replace('：。', '。')
    txt = txt.replace('……', '，')
    txt = remove_spaces_between_chinese(txt)
    if txt[-1] not in '.,?!;。，？！；、':
        if is_chinese(txt):
            txt = txt + '。'
        else:
            txt = txt + '. '
    if wav is not None:
        wav_len = wav.shape[0]
    if len(get_word_list(txt)) > wav_len // hparams['hop_size'] // 4:
        return
    return txt


class PromptAudioShmDataset(BaseTTSShmDataset):

    def _process_item(self, processer_fn, raw_item, tgt_size, hparams, global_stores, i_worker, n_worker):
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
                'promptaudio_exception',
                'promptjson_exception',
                'len_out_of_range',
                'trunc_to_frames_multiple_caused_short',
                'no_caption',
                'no_wav',
                'no_wav_len',
                'path_missing',
                'load_wav_fail',
                'bad_start_end',
                'empty_wav_after_trim',
            ], interval=1000, i_worker=i_worker, n_worker=n_worker)
        )

        items = processer_fn(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker)
        if items is None:
            if DEBUG:
                print(f'processer {i_worker}/{n_worker}: items is None')
            return

        for item_tgt in items:
            # 1) 先用原始 wav_len 做一次时长过滤
            mel_len_pre = item_tgt['wav_len'] // hop_size
            if not (hparams['max_frames'] >= mel_len_pre > hparams['min_frames']):
                skip_logger.report(1, 'len_out_of_range')
                skip_logger.update(1)
                if DEBUG:
                    print('[skip] len_out_of_range pre-trunc', item_tgt.get('item_name', ''))
                continue

            # 2) 文本处理与检测
            if item_tgt.get('use_raw_txt_as_text', False):
                txt = item_tgt.get('txt', '')
            else:
                txt = raw_text_process(item_tgt.get('txt', ''), wav_len=item_tgt['wav_len'])
            item_tgt['text'] = txt

            # 3) wav 裁剪到 frames_multiple 的整倍数，并可选增强
            if hparams.get('load_wav', True):
                before = len(item_tgt['wav'])
                item_tgt['wav'] = item_tgt['wav'][:before // fm_wav * fm_wav]
                if speech_augmentor is not None:
                    item_tgt['wav'] = speech_augmentor(item_tgt['wav'], sr)

            # 4) 裁剪后再次检查时长（避免裁剪导致过短）
            mel_len_post = len(item_tgt['wav']) // hop_size
            if not (hparams['max_frames'] >= mel_len_post > hparams['min_frames']):
                skip_logger.report(1, 'trunc_to_frames_multiple_caused_short')
                skip_logger.update(1)
                if DEBUG:
                    print('[skip] len_out_of_range post-trunc', item_tgt.get('item_name', ''), mel_len_pre, '->', mel_len_post)
                continue

            # 5) 下游上下文/掩码等保持不变
            mel_len = mel_len_post
            min_idx = max(int(mel_len * 0.1), 200)
            max_idx = min(int(mel_len * 0.9), mel_len - 200)
            if min_idx > max_idx:
                min_idx = int(mel_len * 0.4)
                max_idx = int(mel_len * 0.6)
            rand_length = random.randint(min_idx, max_idx) // fm * fm
            ctx_mask = torch.zeros((item_tgt['wav'].shape[0] // hparams['hop_size'], 1))
            ctx_mask[:rand_length] = 1.0
            item_tgt['ctx_mask'] = ctx_mask[::hparams['vae_stride']]
            item_tgt['ctx_wav'] = deepcopy(item_tgt['wav'])
            item_tgt['ctx_wav'] = item_tgt['ctx_wav'][:rand_length * hparams['hop_size']]

            if hparams.get('add_vad_mask', False):
                vad_start, vad_end = run_vad_trim(item_tgt['wav'], hparams['audio_sample_rate'], vad_model)
                vm = hparams['hop_size'] * hparams['vae_stride']
                vad_mask = np.zeros((item_tgt['wav'].shape[0] // vm))
                vad_mask[int(vad_start * hparams['audio_sample_rate'] // vm): int(
                    vad_end * hparams['audio_sample_rate'] // vm)] = 1
                item_tgt['vad_mask'] = vad_mask
            else:
                item_tgt['vad_mask'] = None

            item_tgt['len'] = mel_len // 4
            yield item_tgt
            skip_logger.step(1)

def processer_fn_promptaudio(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    """
    基于 subjects / narration 构造 caption 和 text
    最终每个 item 至少包含: wav/wav_len, txt, caption, item_name, spk_name。
    """
    items = []
    hop_size = hparams['hop_size']
    vae_stride = hparams.get('vae_stride', 4)
    sr = hparams['audio_sample_rate']

    for item_ in raw_item:
        try:
            item = {}
            if hparams.get('load_wav', True):
                wav_np = None
                org_sr = sr

                if 'vocal' in item_ and item_['vocal'] is not None:
                    wav_np = np.asarray(item_['vocal'], dtype=np.float32)
                    org_sr = int(item_.get('vocal_sr', sr))
                elif 'wav' in item_ and item_['wav'] is not None:
                    wav_np = np.asarray(item_['wav'], dtype=np.float32)
                    org_sr = int(item_.get('sr', sr))
                else:
                    skip_logger.report(1, 'no_wav')
                    skip_logger.update(1)
                    if DEBUG:
                        print('[skip@promptaudio] no_wav', item_.get('item_name', ''))
                    continue

                if wav_np.ndim > 1:
                    wav_np = np.mean(wav_np, axis=0)
                if org_sr != sr:
                    wav_np = librosa.resample(wav_np, orig_sr=org_sr, target_sr=sr)

                wav = torch.from_numpy(wav_np)
                item['wav'] = wav
                item['wav_len'] = wav.shape[0]
            else:
                if 'sec' in item_:
                    item['wav_len'] = int(float(item_['sec']) * sr)
                elif 'wav' in item_ and item_['wav'] is not None:
                    item['wav_len'] = len(item_['wav'])
                else:
                    skip_logger.report(1, 'no_wav_len')
                    skip_logger.update(1)
                    if DEBUG:
                        print('[skip@promptaudio] no_wav_len', item_.get('item_name', ''))
                    continue

            # 构造 caption 与 txt
            caption = _build_caption_from_subjects_narration(item_)
            if not caption:
                skip_logger.report(1, 'no_caption')
                skip_logger.update(1)
                if DEBUG:
                    print('[skip@promptaudio] no_caption', item_.get('item_name', ''))
                continue
            item['caption'] = caption

            txt = _build_text_from_caption_for_tts(caption)
            item['txt'] = txt

            item['item_name'] = item_.get('item_name', '')
            item['spk_name'] = item_.get('spk_name', item['item_name'])
            item['use_raw_txt_as_text'] = True

            items.append(item)
        except Exception:
            traceback.print_exc()
            skip_logger.report(1, 'promptaudio_exception')
            skip_logger.update(1)
            continue
    return items

def processer_fn_promptjson(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    """
    读取 jsonl 样本，按 subjects/narration 组合 caption，
    从 caption 中抽取 <W>...</W> 作为 text。
    若存在 start/end 字段，则对 wav 做对应时间段截取。
    """
    fm = hparams['frames_multiple']
    hop_size = hparams['hop_size']
    fm_wav = fm * hop_size
    sr = hparams['audio_sample_rate']
    vae_stride = hparams.get('vae_stride', 4)

    items = []
    for item_ in raw_item:
        try:
            # 1) 读路径
            wav_path = item_.get('wav_24k_path', item_.get('wav_path'))
            if not wav_path:
                skip_logger.report(1, 'path_missing')
                skip_logger.update(1)
                if DEBUG:
                    print('[skip@promptjson] path_missing', item_.get('item_name', ''))
                continue

            # 2) 读 wav
            try:
                wav, _ = librosa.load(wav_path, sr=sr)
            except Exception:
                traceback.print_exc()
                skip_logger.report(1, 'load_wav_fail')
                skip_logger.update(1)
                if DEBUG:
                    print('[skip@promptjson] load_wav_fail', wav_path)
                continue

            # 3) 可选截段
            if 'start' in item_ and 'end' in item_:
                start = max(float(item_['start']), 0.0)
                end = float(item_['end'])
                if end <= start:
                    skip_logger.report(1, 'bad_start_end')
                    skip_logger.update(1)
                    if DEBUG:
                        print('[skip@promptjson] bad_start_end', item_.get('item_name', ''), start, end)
                    continue
                s_idx = int(start * sr)
                e_idx = min(int(end * sr), wav.shape[0])
                if e_idx <= s_idx:
                    skip_logger.report(1, 'bad_start_end')
                    skip_logger.update(1)
                    if DEBUG:
                        print('[skip@promptjson] bad_start_end indices', item_.get('item_name', ''), s_idx, e_idx)
                    continue
                wav = wav[s_idx:e_idx]

            # 4) 裁剪到整倍数
            if wav.size > 0 and fm_wav > 0:
                wav = wav[: (len(wav) // fm_wav) * fm_wav]
            if wav.size == 0:
                skip_logger.report(1, 'empty_wav_after_trim')
                skip_logger.update(1)
                if DEBUG:
                    print('[skip@promptjson] empty_wav_after_trim', item_.get('item_name', ''))
                continue

            item = {}
            item['wav'] = torch.FloatTensor(wav)
            item['wav_len'] = item['wav'].shape[0]

            # 5) 构造 caption 与 txt
            meta_for_caption = dict(item_)
            if 'subjects' not in meta_for_caption and 'subject' in meta_for_caption:
                meta_for_caption['subjects'] = meta_for_caption['subject']

            caption = _build_caption_from_subjects_narration(meta_for_caption)
            if not caption:
                skip_logger.report(1, 'no_caption')
                skip_logger.update(1)
                if DEBUG:
                    print('[skip@promptjson] no_caption', item_.get('item_name', ''))
                continue
            item['caption'] = caption

            txt = _build_text_from_caption_for_tts(caption)
            item['txt'] = txt

            # 6) 标识信息
            item['item_name'] = item_.get(
                'item_name',
                item_.get('utt_id', item_.get('id', wav_path))
            )
            item['spk_name'] = item_.get(
                'spk_name',
                item_.get('speaker', item_.get('gender', item['item_name']))
            )
            item['use_raw_txt_as_text'] = True

            items.append(item)
            skip_logger.step(1)
        except Exception:
            traceback.print_exc()
            skip_logger.report(1, 'promptjson_exception')
            skip_logger.update(1)
            continue

    return items