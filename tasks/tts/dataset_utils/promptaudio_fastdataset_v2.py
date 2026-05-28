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

DEBUG = False

# 没有 ph/tone 时统一使用 PAD ID（与 collater 的 pad 值保持一致，默认 0）
PH_PAD_ID = int(hparams.get('ph_pad_id', 0))
TONE_PAD_ID = int(hparams.get('tone_pad_id', 0))

# ======== precompiled regex patterns ========
_SPACE_RE = re.compile(r'\s+')

_S1S2_TAG_RE = re.compile(
    r'<\s*(S[12])\s*>(.*?)</\s*S[12]\s*>',
    flags=re.IGNORECASE | re.DOTALL,
)

_S1_TAG_RE = re.compile(
    r'<\s*S1\s*>(.*?)</\s*S1\s*>',
    flags=re.IGNORECASE | re.DOTALL,
)

_I_OPEN_TAG_RE = re.compile(r'<\s*I\s*>', flags=re.IGNORECASE)
_I_CLOSE_TAG_RE = re.compile(r'</\s*I\s*>', flags=re.IGNORECASE)


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

def _norm_spaces_caption(s: str) -> str:
    if not isinstance(s, str):
        return ''
    # 行尾统一
    s = s.replace('\r\n', '\n').replace('\r', '\n')
    # 把所有连续空白压成一个空格（行为与原来 re.sub(r'\s+', ' ', s) 一致）
    s = _SPACE_RE.sub(' ', s)
    return s.strip()


def _normalize_text_field_caption(val):
    if isinstance(val, list):
        try:
            val = ''.join(map(str, val))
        except Exception:
            val = ' '.join(map(str, val))
    return _norm_spaces_caption(val) if isinstance(val, str) else ''


def _build_text_from_caption_s1s2(caption: str) -> str:
    """
    从 caption 中按顺序提取 <S1>...</S1> 和 <S2>...</S2>，
    并把相邻的 S1 片段合并成一个 <S1>... ...</S1>。
    若没有任何 S1/S2 片段则返回 ""。
    """
    if not isinstance(caption, str) or not caption.strip():
        return ""

    # 使用预编译的 _S1S2_TAG_RE
    segments = []
    for tag, inner in _S1S2_TAG_RE.findall(caption):
        inner_norm = _norm_spaces_caption(inner)
        if not inner_norm:
            continue
        segments.append([tag.upper(), inner_norm])

    if not segments:
        return ""

    # 合并相邻的 S1 段
    merged = []
    for tag, content in segments:
        if merged and tag == 'S1' and merged[-1][0] == 'S1':
            merged[-1][1] = merged[-1][1] + ' ' + content
        else:
            merged.append([tag, content])

    # 重新拼回成带标签的字符串
    out = ''.join(f"<{tag}>{content}</{tag}>" for tag, content in merged)
    return out if out.strip() else ""


def _build_caption_from_subjects_narration(meta):
    """从 new_caption / subjects / narration 构建 caption:
    形如 "Subjects:... NARRATION:..."，并且把 <I></I> 标签替换为 <Audio></Audio>。
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
        parts.append(f"Subjects:{subjects}")
    if narration:
        parts.append(f"Narration:{narration}")
    caption = ' '.join(parts)

    if not caption:
        return ''

    # 把 <I>...</I> 标签替换为 <Audio>...</Audio>（和原逻辑完全一致）
    caption = _I_OPEN_TAG_RE.sub('<Audio>', caption)
    caption = _I_CLOSE_TAG_RE.sub('</Audio>', caption)
    return caption


def _build_text_from_caption_for_tts(caption: str) -> str:
    """从 caption 中抽取所有 <S1>...</S1> 内容拼接成 text；若没有则返回 ""。"""
    if not isinstance(caption, str) or not caption.strip():
        return ""

    # 使用预编译的 _S1_TAG_RE
    matches = _S1_TAG_RE.findall(caption)

    chunks = []
    for m in matches:
        norm = _norm_spaces_caption(m)
        if norm:
            chunks.append(norm)

    if chunks:
        inner = ' '.join(chunks)
        return f"<S1>{inner}</S1>"
    else:
        return ""

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
    if txt and txt[-1] not in '.,?!;。，？！；、':
        if is_chinese(txt):
            txt = txt + '。'
        else:
            txt = txt + '. '
    if wav is not None:
        wav_len = wav.shape[0]
    # 词数超过 latent 容量时返回 None，并打印原因
    if wav_len is not None and len(get_word_list(txt)) > wav_len // hparams['hop_size'] // 4:
        _print_skip(
            reason="text_exceeds_latent_capacity",
            extra=f"words={len(get_word_list(txt))}, latent={wav_len // hparams['hop_size'] // 4}"
        )
        return None
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

def processer_fn_promptaudio(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    """基于 subjects / narration 构造 caption 和 text，且不依赖 mel2ph / dur。
    最终每个 item 至少包含: wav/wav_len, phone, tone, txt, caption, item_name, spk_name。
    如果缺失 phone / tone，则分别回退为 SIL_TOKEN_ID 和 0。

    这里额外做两件事：
      1）先把 wav 对齐到 frames_multiple * hop_size 的整数倍，再记录 wav_len；
      2）用裁剪后的 wav_len 计算 mel_len 和 latent_len = mel_len // 4，
         若 phone_encoded 长度 > latent_len，则直接丢掉该 sample。
    """
    items = []
    hop_size = hparams['hop_size']
    fm = hparams['frames_multiple']
    fm_wav = fm * hop_size
    sr = hparams['audio_sample_rate']

    for item_ in raw_item:
        try:
            item = {}

            # ========= 1. wav & 长度 =========
            wav = None
            org_sr = sr

            # 优先用 vocal（如果有的话）
            if 'vocal' in item_ and item_['vocal'] is not None:
                wav = torch.as_tensor(item_['vocal'], dtype=torch.float32)
                org_sr = int(item_.get('vocal_sr', sr))
            elif 'wav' in item_ and item_['wav'] is not None:
                wav = torch.as_tensor(item_['wav'], dtype=torch.float32)
                org_sr = int(item_.get('sr', sr))
            else:
                _print_skip("missing_wav_or_vocal", i_worker, n_worker, item_name=item_.get('item_name'))
                continue

            # 多通道 -> 单通道（假设 [C, T] 或 [T]，与原 np.mean(axis=0) 语义一致）
            if wav.dim() > 1:
                wav = wav.mean(dim=0)

            # 重采样到训练采样率（使用 torchaudio）
            if org_sr != sr:
                try:
                    wav = torchaudio.functional.resample(
                        wav.unsqueeze(0), orig_freq=org_sr, new_freq=sr
                    )[0]
                except Exception as e:
                    _print_skip(
                        "resample_failed",
                        i_worker, n_worker,
                        item_name=item_.get('item_name'),
                        extra=f"from={org_sr} to={sr}, err={str(e)}"
                    )
                    continue

            # ★ 对齐到 frames_multiple * hop_size 的整数倍
            if fm_wav > 0:
                wav = wav[: (wav.shape[0] // fm_wav) * fm_wav]
            if wav.numel() == 0:
                _print_skip("empty_wav_after_alignment", i_worker, n_worker, item_name=item_.get('item_name'))
                continue

            wav = wav.contiguous()
            item['wav'] = wav
            item['wav_len'] = wav.shape[0]

            # ========= 2. phone / tone + ph vs latent 过滤 =========
            phone_encoded = item_.get('phone_encoded')

            # 用裁剪后的 wav_len 计算 mel_len / latent_len，先过滤 ph 太长的样本
            if hparams.get('load_wav', True) and phone_encoded is not None and len(phone_encoded) > 0:
                mel_len = item['wav_len'] // hop_size
                latent_len = mel_len // 4
                if len(phone_encoded) > latent_len:
                    _print_skip("phone_longer_than_latent", i_worker, n_worker,
                                item_name=item_.get('item_name'),
                                extra=f"ph={len(phone_encoded)}, latent={latent_len}")
                    continue

            # 缺失 phone -> 用 PAD
            if phone_encoded is None or len(phone_encoded) == 0:
                phone = torch.LongTensor([PH_PAD_ID])
                _print_skip("fallback_pad_phone", i_worker, n_worker,
                            item_name=item_.get('item_name'),
                            extra=f"pad_id={PH_PAD_ID}")
            else:
                phone = torch.LongTensor(phone_encoded)

            tone_encoded = item_.get('tone_encoded')
            # 缺失 tone -> 用 PAD，长度与 phone 对齐
            if tone_encoded is None or len(tone_encoded) == 0:
                tone = torch.full((phone.shape[0],), TONE_PAD_ID, dtype=torch.long)
                _print_skip("fallback_pad_tone", i_worker, n_worker,
                            item_name=item_.get('item_name'),
                            extra=f"len={phone.shape[0]}, pad_id={TONE_PAD_ID}")
            else:
                tone = torch.LongTensor(tone_encoded)
                # 长度不一致时做对齐（pad 用 PAD）
                if tone.shape[0] != phone.shape[0]:
                    if tone.shape[0] < phone.shape[0]:
                        pad_len = phone.shape[0] - tone.shape[0]
                        tone = torch.cat(
                            [tone, torch.full((pad_len,), TONE_PAD_ID, dtype=torch.long)],
                            dim=0
                        )
                        _print_skip("tone_padded_to_match_phone", i_worker, n_worker,
                                    item_name=item_.get('item_name'),
                                    extra=f"pad={pad_len}, pad_id={TONE_PAD_ID}")
                    else:
                        _print_skip("tone_truncated_to_match_phone", i_worker, n_worker,
                                    item_name=item_.get('item_name'),
                                    extra=f"tone_len={tone.shape[0]} -> {phone.shape[0]}")
                        tone = tone[:phone.shape[0]]

            item['phone'] = phone
            item['tone'] = tone


            # ========= 3. caption & text =========
            caption = _build_caption_from_subjects_narration(item_)
            if not caption:
                _print_skip("no_caption_built", i_worker, n_worker, item_name=item_.get('item_name'))
            item['caption'] = caption
            item['txt'] = _build_text_from_caption_for_tts(caption)  # <S1>... 拼起来，没有就 ""

            # ========= 4. 其他元信息 =========
            item['item_name'] = item_.get('item_name', '')
            item['spk_name'] = item_.get('spk_name', item['item_name'])
            # 告诉 _process_item 不再跑 raw_text_process，直接用 txt
            item['use_raw_txt_as_text'] = True

            items.append(item)
        except Exception as e:
            traceback.print_exc()
            _print_skip("processer_fn_promptaudio_exception", i_worker, n_worker, item_name=item_.get('item_name'), extra=f"err={str(e)}")
            continue

    return items

def processer_fn_promptjson(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    """
    读取 jsonl 样本，按 subjects/narration 组合 caption，
    从 caption 中抽取 <W>...</W> 作为 text（若没有则用 ）。
    若存在 start/end 字段，则对 wav 做对应时间段截取。

    这里同样：
      1）先把 wav 对齐到 frames_multiple * hop_size 的整数倍，再记录 wav_len；
      2）用裁剪后的 wav_len 计算 mel_len 和 latent_len = mel_len // 4，
         若 phone_encoded 长度 > latent_len，则直接丢掉该 sample。
    """
    fm = hparams['frames_multiple']
    hop_size = hparams['hop_size']
    fm_wav = fm * hop_size
    sr = hparams['audio_sample_rate']

    items = []
    for item_ in raw_item:
        try:
            # -------- 1. 读 wav，必要时按 start/end 截取 --------
            wav_path = item_.get('wav_24k_path', item_.get('wav_path'))
            if not wav_path:
                _print_skip("missing_wav_path", i_worker, n_worker, item_name=item_.get('item_name', item_.get('id')))
                continue

            try:
                # torchaudio.load: [C, T], dtype 通常是 float32
                wav, org_sr = torchaudio.load(wav_path)
                wav = wav.to(torch.float32)
            except Exception as e:
                _print_skip("torchaudio_load_failed", i_worker, n_worker, item_name=item_.get('item_name', wav_path), extra=f"path={wav_path}, err={str(e)}")
                skip_logger.report(1, 'promptjson')
                continue

            # 多通道 -> 单通道
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

            # 重采样到训练采样率
            if org_sr != sr:
                try:
                    wav = torchaudio.functional.resample(wav, orig_freq=org_sr, new_freq=sr)
                except Exception as e:
                    _print_skip("resample_failed", i_worker, n_worker, item_name=item_.get('item_name', wav_path), extra=f"from={org_sr} to={sr}, err={str(e)}")
                    skip_logger.report(1, 'promptjson')
                    continue

            # 现在 wav: [1, T]
            wav = wav[0]  # -> [T]

            # json 中可能有 start / end（单位：秒），用目标 sr 做切
            if 'start' in item_ and 'end' in item_:
                start = max(float(item_['start']), 0.0)
                end = float(item_['end'])
                if end <= start:
                    _print_skip("invalid_time_range", i_worker, n_worker, item_name=item_.get('item_name'), extra=f"start={start}, end={end}")
                    continue
                s_idx = int(start * sr)
                e_idx = min(int(end * sr), wav.shape[0])
                if e_idx <= s_idx:
                    _print_skip("time_slice_empty", i_worker, n_worker, item_name=item_.get('item_name'), extra=f"s_idx={s_idx}, e_idx={e_idx}")
                    continue
                wav = wav[s_idx:e_idx]

            # ★ 对齐到 frames_multiple * hop_size 的整数倍（和 audioset 一致）
            if wav.numel() > 0 and fm_wav > 0:
                wav = wav[: (wav.shape[0] // fm_wav) * fm_wav]
            if wav.numel() == 0:
                _print_skip("empty_wav_after_alignment", i_worker, n_worker, item_name=item_.get('item_name', wav_path))
                continue

            wav = wav.contiguous()

            item = {}
            item['wav'] = wav
            item['wav_len'] = wav.shape[0]

            # -------- 2. phone / tone，缺失时做兜底，并根据 latent 长度过滤 --------
            phone_encoded = item_.get('phone_encoded')

            if phone_encoded is not None and len(phone_encoded) > 0:
                mel_len = item['wav_len'] // hop_size
                latent_len = mel_len // 4
                if len(phone_encoded) > latent_len:
                    _print_skip("phone_longer_than_latent", i_worker, n_worker,
                                item_name=item_.get('item_name', wav_path),
                                extra=f"ph={len(phone_encoded)}, latent={latent_len}")
                    continue

            # 缺失 phone -> 用 PAD
            if phone_encoded is None or len(phone_encoded) == 0:
                phone = torch.LongTensor([PH_PAD_ID])
                _print_skip("fallback_pad_phone", i_worker, n_worker,
                            item_name=item_.get('item_name', wav_path),
                            extra=f"pad_id={PH_PAD_ID}")
            else:
                phone = torch.LongTensor(phone_encoded)

            tone_encoded = item_.get('tone_encoded')
            # 缺失 tone -> 用 PAD，长度与 phone 对齐
            if tone_encoded is None or len(tone_encoded) == 0:
                tone = torch.full((phone.shape[0],), TONE_PAD_ID, dtype=torch.long)
                _print_skip("fallback_pad_tone", i_worker, n_worker,
                            item_name=item_.get('item_name', wav_path),
                            extra=f"len={phone.shape[0]}, pad_id={TONE_PAD_ID}")
            else:
                tone = torch.LongTensor(tone_encoded)
                if tone.shape[0] != phone.shape[0]:
                    if tone.shape[0] < phone.shape[0]:
                        pad_len = phone.shape[0] - tone.shape[0]
                        tone = torch.cat(
                            [tone, torch.full((pad_len,), TONE_PAD_ID, dtype=torch.long)],
                            dim=0
                        )
                        _print_skip("tone_padded_to_match_phone", i_worker, n_worker,
                                    item_name=item_.get('item_name', wav_path),
                                    extra=f"pad={pad_len}, pad_id={TONE_PAD_ID}")
                    else:
                        _print_skip("tone_truncated_to_match_phone", i_worker, n_worker,
                                    item_name=item_.get('item_name', wav_path),
                                    extra=f"tone_len={tone.shape[0]} -> {phone.shape[0]}")
                        tone = tone[:phone.shape[0]]

            item['phone'] = phone
            item['tone'] = tone


            # -------- 3. caption = subjects + narration（复用已有工具函数） --------
            # 部分数据只有 "subject" 没有 "subjects"，做一个兼容
            meta_for_caption = dict(item_)
            if 'subjects' not in meta_for_caption and 'subject' in meta_for_caption:
                meta_for_caption['subjects'] = meta_for_caption['subject']

            caption = _build_caption_from_subjects_narration(meta_for_caption)
            if not caption:
                _print_skip("no_caption_built", i_worker, n_worker, item_name=item_.get('item_name', wav_path))
            item['caption'] = caption

            # -------- 4. text：从 caption 的 <S1> 抽取，没有就 "" --------
            txt = _build_text_from_caption_for_tts(caption)
            if not txt:
                _print_skip("empty_text_built_from_caption", i_worker, n_worker, item_name=item_.get('item_name', wav_path))
            item['txt'] = txt

            # -------- 5. 其他元信息：item_name / spk_name / 标记使用原始 txt --------
            item['item_name'] = item_.get(
                'item_name',
                item_.get('utt_id', item_.get('id', wav_path))
            )
            item['spk_name'] = item_.get(
                'spk_name',
                item_.get('speaker', item_.get('gender', item['item_name']))
            )

            # 告诉 _process_item 不再跑 raw_text_process，直接使用 txt 作为 text
            item['use_raw_txt_as_text'] = True

            items.append(item)
            skip_logger.step(1)
        except Exception as e:
            traceback.print_exc()
            _print_skip("processer_fn_promptjson_exception", i_worker, n_worker, item_name=item_.get('item_name', item_.get('id', '')), extra=f"err={str(e)}")
            skip_logger.report(1, 'promptjson')
            continue

    return items
