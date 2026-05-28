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
import time
import hashlib
import glob
import traceback

import torch
import numpy as np
import torch.utils
import torch.utils.data
import librosa

from dataloader import FalconReader, KVReader
from utils.commons.hparams import hparams
from utils.commons.os_utils import multiprocess_glob, handle_exacption
from utils.dataset.batcher import BucketBatcher
from utils.commons.io import get_wav_duration, print_once
from utils.text.split_text import get_word_list
from utils.commons.base_shm_dataset import BaseFalconReaderShmDataset, get_from_global_stores
from utils.commons.dataset_utils import collate_xd, pad_or_cut_xd
from utils.audio.vad import build_vad_model, run_vad_trim

from modules.tts.ar_dur.commons.align_ops import compute_mel2aug_from_dur
from modules.tts.ar_dur.commons.nar_tts_modules import LengthRegulator

SIL_TOKEN_ID = int(hparams.get('sil_token_id', 301)) # audio 段对应的 <sil> token id
def _mel2token_to_dur(mel2ph: torch.LongTensor) -> np.ndarray:
    """把 mel2ph(1-based) 转为每个 token 的时长（frames）"""
    if mel2ph is None or mel2ph.numel() == 0:
        return np.zeros((0,), dtype=np.int64)
    m = int(torch.max(mel2ph).item())
    hist = np.bincount(mel2ph.cpu().numpy(), minlength=m+1)  # 0..m
    return hist[1:].astype(np.int64)

class PromptAudioShmDataset(BaseFalconReaderShmDataset):

    # ============ 原模板：reader / meta ============
    def get_dataset_meta(self):
        data_paths = hparams['datasets']
        cluster = os.environ.get('CLUSTER', '').lower()
        hdfs_root = hparams['hdfs_root']
        if cluster.lower() in ['lq', 'hl', 'sg', 'va']:
            print_once(f'| Detect cluster [{cluster.lower()}]')
            if cluster == 'lq':
                hdfs_root = hparams.get('hdfs_root_lq', hdfs_root)
            elif cluster == 'hl':
                hdfs_root = hparams.get('hdfs_root_hl', hdfs_root)
            elif cluster == 'sg':
                hdfs_root = hparams.get('hdfs_root_sg', hdfs_root)
            elif cluster == 'va':
                hdfs_root = hparams.get('hdfs_root_va', hdfs_root)
            print_once(f'| Choose hdfs_root: {hdfs_root}')
        else:
            print_once(f'| Use default hdfs_root: {hdfs_root}')
        data_paths = [os.path.join(hdfs_root, p) if not p.startswith('hdfs://') else p for p in data_paths]
        _, ds_len = self.get_reader(data_paths, 1)
        return data_paths, ds_len

    def prepare_reader(self, dataset_meta, global_stores, i_worker, n_worker):
        reader, ds_len = self.get_reader(
            dataset_meta, self.hparams.get('reader_chunk_size', 64), 
            worker_id=i_worker, worker_world_size=n_worker, reader_cache_name='reader_cache'
        )
        return reader
    
    def read_fn(self, idx, reader_pack, global_stores):
        reader = reader_pack
        try:
            items = [pickle.loads(x) for x in reader.read_many([idx])[0]]
            return items
        except:
            return

    # ============ 一次性全局资源：Batcher & 音频池 ============
    def _get_batcher(self, global_stores):
        return get_from_global_stores(
            'batcher', global_stores,
            lambda: BucketBatcher(
                buckets=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 
                         600, 650, 700, 750, 800, 850, 900, 950, 1000, 1200, 1400, 
                         1600, 1800, 2000, 2400, 2800, 3000],
                dynamic_batch=hparams.get("dynamic_batch", True),
                batch_size=hparams['max_sentences'],
                maximum_bucket_size=hparams['max_tokens'],
                length_fn=lambda x: x['len'],
            )
        )

    def _get_length_regulator(self, global_stores):
        return get_from_global_stores('length_regulator', global_stores, lambda: LengthRegulator())

    # ============ 处理主流程 ============
    def process_item(self, raw_item, hparams, global_stores, i_worker, n_worker):
        batcher = self._get_batcher(global_stores) if self.use_fast_dataloader else None
        for item in self._process_item(raw_item, hparams, global_stores, i_worker, n_worker):
            if item is None:
                continue
            if self.use_fast_dataloader:
                batch = batcher.collate_batch(item)
                if batch is not None and len(batch) > 0:
                    yield batch
            else:
                yield [item]

    def report_skip_status(self, cnt, item_cnt, name, i_worker, n_worker, step=100):
        """
        仅作为进度提示；不再称为 skipped，改为已用默认值兜底，避免误解。
        """
        if cnt > 0 and cnt % step == 0:
            print(f"| processer#{i_worker}/{n_worker}: filled defaults [{cnt}/{item_cnt}] for [{name}]")

    def _process_item(self, raw_item, hparams, global_stores, i_worker, n_worker):
        # 统计计数
        if not hasattr(self, 'item_cnt'):
            self.item_cnt = 0
            self.no_text_cnt = 0
            self.no_phone_cnt = 0
            self.drop_long_cnt = 0
        self.item_cnt += 1

        fm = hparams['frames_multiple']
        hop_size = hparams['hop_size']
        fm_wav = fm * hop_size
        sr = hparams['audio_sample_rate']

        # ---------- 选用 vocal 或 wav，并重采样、裁剪到 fm 倍数 ----------
        if hparams.get('use_vocal', True):
            wav = (raw_item['vocal']).astype(float)
            org_sr = raw_item['vocal_sr']
        else:
            wav = (raw_item['wav']).astype(float)
            org_sr = raw_item['sr']

        if sr != org_sr:
            wav = librosa.resample(wav, orig_sr=org_sr, target_sr=sr)
        wav = wav[:len(wav) // fm_wav * fm_wav]

        item = {
            'wav': wav,
            'subset': raw_item.get('subset', '')
        }

        # ---------- 文本 / Caption 工具 ----------
        def _norm_spaces(s: str) -> str:
            if not isinstance(s, str):
                return ''
            s = s.replace('\r\n', '\n').replace('\r', '\n')
            s = re.sub(r'\s+', ' ', s)
            return s.strip()

        def _normalize_text_field(val):
            # 兼容 list/str 的字段
            if isinstance(val, list):
                try:
                    val = ''.join(map(str, val))
                except Exception:
                    val = ' '.join(map(str, val))
            return _norm_spaces(val) if isinstance(val, str) else ''

        def _estimate_text_len(s: str) -> int:
            """估算文本序列长度：优先用 get_word_list，失败则用去空白后的字符数。"""
            if not s:
                return 0
            try:
                toks = get_word_list(s)
                if isinstance(toks, (list, tuple)):
                    return len(toks)
            except Exception:
                pass
            return len(re.sub(r'\s+', '', s))

        def _build_caption(raw_item):
            """
            从 new_caption 或顶层 subjects/narration 构建 caption：
            格式为 "Subjects:...,Narration:..."，并把 <I></I> 换成 <Audio></Audio>。
            """
            new_cap = raw_item.get('new_caption')
            subjects = ''
            narration = ''

            if isinstance(new_cap, dict):
                subjects = _normalize_text_field(new_cap.get('subjects', ''))
                narration = _normalize_text_field(new_cap.get('narration', ''))

            # 顶层兜底
            if not subjects:
                subjects = _normalize_text_field(raw_item.get('subjects', ''))
            if not narration:
                narration = _normalize_text_field(raw_item.get('narration', ''))

            parts = []
            if subjects:
                parts.append(f"Subjects:{subjects}")
            if narration:
                parts.append(f"Narration:{narration}")
            caption = ','.join(parts)

            if not caption:
                return ''

            # 把 <I>...</I> 标签替换为 <Audio>...</Audio>
            caption = re.sub(r'<\s*I\s*>', '<Audio>', caption, flags=re.IGNORECASE)
            caption = re.sub(r'</\s*I\s*>', '</Audio>', caption, flags=re.IGNORECASE)

            return caption

        def _build_text_from_caption(caption: str):
            """
            从处理好的 caption 中：
            - 找出所有 <S1>...</S1> 中间内容，拼成一个字符串；
            - 再用单个 <S1>...</S1> 包裹后作为 text；
            - 如果没有 S1，则 text = "<S1><Mask></S1>"，plain_text 用去标签后的 caption。
            （这样既能用 caption 控制长度过滤，又不会把没标 S1 的 caption 当成真实 text）
            返回： (text_with_S1_or_mask, plain_text_for_len)
            """
            if not isinstance(caption, str) or not caption.strip():
                return '', ''

            # 提取所有 <S1>...</S1> 的内容
            matches = re.findall(
                r'<\s*S1\s*>(.*?)</\s*S1\s*>',
                caption,
                flags=re.IGNORECASE | re.DOTALL
            )
            inner_chunks = [_norm_spaces(m) for m in matches if _norm_spaces(m)]

            if inner_chunks:
                # 正常有 S1 的情况：用真实文本
                inner = ' '.join(inner_chunks)
                if not inner:
                    return '', ''
                return f"<S1>{inner}</S1>", inner
            else:
                # 没有 S1：text 用 Mask，占位；plain_text 仍然根据 caption 估长度
                no_tags = re.sub(r'<[^>]+>', ' ', caption)
                inner_plain = _norm_spaces(no_tags)

                if not inner_plain:
                    return '', ''

                return "<Mask>", inner_plain


        # ---------- 构建 caption & text ----------
        caption_str = _build_caption(raw_item)
        text_field, text_plain = _build_text_from_caption(caption_str)

        # ---------- 对齐信息检测 ----------
        has_pe  = isinstance(raw_item.get('phone_encoded'), (list, np.ndarray)) and len(raw_item['phone_encoded']) > 0
        has_te  = isinstance(raw_item.get('tone_encoded'),  (list, np.ndarray)) and len(raw_item['tone_encoded'])  > 0
        has_m2p = isinstance(raw_item.get('mel2ph'),        (list, np.ndarray)) and len(raw_item['mel2ph'])        > 0

        m2p = np.array(raw_item['mel2ph'], dtype=int) if has_m2p else None
        if m2p is not None:
            m2p = m2p[:len(m2p) // fm * fm]

        fallback = True  # 默认兜底，除非检测到有效对齐
        ph_enc = None
        te_enc = None

        if has_pe:
            ph_enc = np.array(raw_item['phone_encoded'], dtype=int)
            te_enc = np.array(raw_item['tone_encoded'],  dtype=int) if has_te else None
            cond = (
                (m2p is not None) and
                (len(item['wav']) // hop_size - len(m2p) < 2) and
                (int(np.max(m2p)) <= len(ph_enc)) and
                (len(item['wav']) // hop_size >= len(m2p))
            )
            if cond:
                fallback = False

        # ---------- 过长样本判定 ----------
        mel_len = len(item['wav']) // hop_size
        len_limit = mel_len // 4  # 与参考代码保持一致

        # 1) phone 序列长度过长则丢弃
        if has_pe:
            ph_len_for_check = len(raw_item['phone_encoded'])
            if ph_len_for_check > len_limit:
                self.drop_long_cnt += 1
                if self.drop_long_cnt % 200 == 0:
                    print(f"| processor#{i_worker}/{n_worker}: dropped [{self.drop_long_cnt}/{self.item_cnt}] (ph>{len_limit})")
                return  # 丢弃该样本

        # 2) 文本序列长度过长则丢弃（这里用从 caption 抽出来的纯文本）
        text_len_for_check = _estimate_text_len(text_plain)
        if text_len_for_check > len_limit:
            self.drop_long_cnt += 1
            if self.drop_long_cnt % 200 == 0:
                print(f"| processor#{i_worker}/{n_worker}: dropped [{self.drop_long_cnt}/{self.item_cnt}] (text>{len_limit})")
            return  # 丢弃该样本

        # ---------- 分支：有效对齐 vs. 兜底 ----------
        if not fallback:
            # —— 有效对齐 ——
            item['mel2ph'] = m2p
            T = int(np.max(m2p))
            item['ph_token'] = ph_enc[:T]
            item['tone'] = (te_enc[:T] if has_te else np.zeros(T, dtype=int))
            item['dur'] = _mel2token_to_dur(torch.from_numpy(item['mel2ph']))
            item['wav'] = item['wav'][:len(item['mel2ph']) * hop_size]

            # text 直接来自处理好的 caption 中的 S1 串联结果
            item['text'] = text_field
            item['caption'] = caption_str

            if not text_plain:
                self.no_text_cnt += 1
                self.report_skip_status(self.no_text_cnt, self.item_cnt, 'text', i_worker, n_worker, 100)
        else:
            # —— 兜底：使用默认 phone/dur，但 text 仍然来自 caption 的 S1 抽取
            mel_frames = len(item['wav']) // hop_size
            mel_frames = mel_frames // fm * fm

            item['ph_token'] = np.array([int(SIL_TOKEN_ID)], dtype=int)
            item['tone'] = np.array([0], dtype=int)
            item['mel2ph'] = np.ones(mel_frames, dtype=int)
            item['dur'] = np.array([mel_frames], dtype=int)

            item['text'] = text_field
            item['caption'] = caption_str

            self.no_phone_cnt += 1
            self.report_skip_status(self.no_phone_cnt, self.item_cnt, 'phone_dur', i_worker, n_worker, 1000)

        # ---------- 长度字段 ----------
        if hparams.get('length_fn', 'lat') == 'lat':
            item['len'] = item['wav'].shape[0] // hparams['hop_size'] // hparams['vae_stride']
        elif hparams.get('length_fn', 'lat') == 'ph':
            item['len'] = len(item['ph_token'])

        # —— 返回样本 ——
        yield item



    def collater(self, samples):
        # 兼容 fast_dataloader 情况
        if len(samples) == 1 and isinstance(samples[0], list):
            samples = samples[0]
        if len(samples) == 0:
            if hasattr(self, 'backup_batch') and self.backup_batch is not None:
                print('use backup batch!')
                return self.backup_batch
            else:
                print('no batch to take!')
                return {}

        wavs = collate_xd([s['wav'] for s in samples], 0.0)
        wav_lengths = torch.LongTensor([s['wav'].shape[0] for s in samples])
        batch = {
            'nsamples': len(samples),
            'wavs': wavs,
            'wav_lengths': wav_lengths,
        }

        # ===== 必备：无条件汇总（_process_item 已保证字段存在） =====
        batch['mel2ph'] = collate_xd([s['mel2ph'] for s in samples], 0)
        batch['dur']    = collate_xd([s['dur']    for s in samples], 0)
        batch['dur_len'] = torch.LongTensor([s['dur'].shape[0] for s in samples])

        batch['ph_tokens'] = collate_xd([s['ph_token'] for s in samples], 0)
        batch['txt_lengths'] = torch.LongTensor([
            s['ph_token'].numel() if isinstance(s['ph_token'], torch.Tensor) else len(s['ph_token'])
            for s in samples
        ])

        batch['tone']  = collate_xd([s['tone'] for s in samples], 0)
        batch['text']  = [s.get('text', '') for s in samples]
        batch['caption'] = [s.get('caption', '') for s in samples]

        # ===== 可选：仅当所有样本都具备时再汇总，避免混合数据集冲突 =====
        def _all_have(key): return all(key in s for s in samples)

        if _all_have('mel'):
            batch['mels'] = collate_xd([s['mel'] for s in samples], -6.0)
        if _all_have('mel2ph_sparse'):
            batch['mel2ph_sparse'] = collate_xd([s['mel2ph_sparse'] for s in samples], 0)
        if _all_have('ph_timestamp'):
            batch['ph_timestamp'] = collate_xd([s['ph_timestamp'] for s in samples], 797)
            batch['ph_timestamp_len'] = torch.LongTensor([s['ph_timestamp'].shape[0] for s in samples])
        if _all_have('merged_ph_token'):
            batch['merged_ph_tokens'] = collate_xd([s['merged_ph_token'] for s in samples], 797)
            batch['merged_ph_tokens_len'] = torch.LongTensor([s['merged_ph_token'].shape[0] for s in samples])
        if _all_have('ph_dur_seq'):
            batch['ph_dur_seqs'] = collate_xd([s['ph_dur_seq'] for s in samples], 797)
            batch['ph_dur_seqs_len'] = torch.LongTensor([s['ph_dur_seq'].shape[0] for s in samples])
        if _all_have('ph_dur_seq_dur_mask'):
            batch['ph_dur_seq_dur_mask'] = collate_xd([s['ph_dur_seq_dur_mask'] for s in samples], 0)

        # prompt 相关：作为字符串列表导出，不强制所有样本具备
        batch['global_prompt'] = [s.get('global_prompt', '') for s in samples]
        batch['local_prompt']  = [s.get('local_prompt',  '') for s in samples]

        # 备份
        if not hasattr(self, 'backup_batch') or self.backup_batch is None or random.random() < 0.001:
            self.backup_batch = batch

        return batch
