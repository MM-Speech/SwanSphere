import os
import random
import json
from copy import deepcopy

import torch
import numpy as np
import torch.utils
import torch.utils.data
import librosa

from utils.commons.hparams import hparams
from utils.commons.os_utils import multiprocess_glob, handle_exacption
from utils.dataset.batcher import BucketBatcher
from utils.commons.io import get_wav_duration
from utils.text.split_text import get_word_list
from utils.commons.base_shm_dataset import BaseShmDataset, get_from_global_stores
from utils.commons.dataset_utils import collate_xd


class DialogueRandomSliceShmDataset(BaseShmDataset):
    def get_dataset_meta(self):
        hparams, prefix = self.hparams, self.prefix
        meta_dir = '/mnt/bn/lq-ads-aigc/renyi/tts_datasets_bak/InteractiveDialogue/XYZ_20w/yuzhou/metas'
        # meta_paths = multiprocess_glob(f"{meta_dir}/*/*_info.json")
        # with open('data/XYZ_20w_meta_paths.lst', 'w') as f:
        #     f.write('\n'.join(meta_paths))
        with open('data/XYZ_20w_meta_paths.lst') as f:
            meta_paths = f.read().split('\n')
        return meta_paths, len(meta_paths)

    def prepare_reader(self, dataset_meta, global_stores):
        return 1
    
    def read_fn(self, idx, reader_pack, global_stores):
        return self.dataset_meta[idx]
    
    def process_item(self, raw_item, hparams, global_stores):
        if self.use_fast_dataloader:
            buckets = [400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2400, 2800, 
                       3200, 3600, 4000, 4500, 5000, 5500, 6000, 7000, 8000, 9000, 10000, 
                       11000, 12000, 13000, 14000, 15000, 16000, 18000, 20000, 40000, 60000]
            batcher = get_from_global_stores(
                'batcher', global_stores,
                lambda: BucketBatcher(
                    buckets=buckets,
                    dynamic_batch=hparams.get("dynamic_batch", True),
                    batch_size=hparams['max_sentences'],
                    maximum_bucket_size=hparams['max_tokens'],
                    length_fn=lambda x: x['len'],
                )
            )

        try:
            item = self._process_item(raw_item, hparams)
        except Exception as err:
            handle_exacption(err, raw_item)
            return
        if item is None:
            return

        if self.use_fast_dataloader:
            batch = batcher.collate_batch(item)
            if batch is not None and len(batch) > 0:
                yield batch
        else:
            yield [item]

    def _process_item(self, raw_item, hparams):
        import re

        # ---- helpers ----
        def _mel2token_to_dur(m2p: torch.LongTensor) -> torch.LongTensor:
            if m2p.numel() == 0:
                return torch.zeros(0, dtype=torch.long)
            mx = int(m2p.max().item()) if m2p.numel() > 0 else 0
            if mx == 0:
                return torch.zeros(0, dtype=torch.long)
            cnts = torch.bincount(m2p.clamp_min(0), minlength=mx + 1)
            return cnts[1:].to(torch.long)

        def _to_int_list_maybe(x):
            """
            将多种可能形式的 tone 安全转成整型 list。
            支持：
            - list/tuple/ndarray，元素可为 str/int/float
            - 纯字符串，如 "1 2 3" / "1,2,3"
            - JSON 字符串，如 "[1, 2, 3]" 或 '["0","4","4"]'
            转换失败返回 None。
            """
            if x is None:
                return None
            if isinstance(x, str):
                s = x.strip()
                if (s.startswith('[') and s.endswith(']')) or (s.startswith('(') and s.endswith(')')):
                    try:
                        parsed = json.loads(s.replace('(', '[').replace(')', ']'))
                        return _to_int_list_maybe(parsed)
                    except Exception:
                        pass
                nums = re.findall(r'-?\d+\.?\d*', s)
                if len(nums) == 0:
                    return None
                try:
                    return [int(float(t)) for t in nums]
                except Exception:
                    return None
            if isinstance(x, (list, tuple, np.ndarray)):
                out = []
                for v in x:
                    try:
                        if isinstance(v, str):
                            v = v.strip()
                            if v == '':
                                out.append(0)
                                continue
                            out.append(int(float(v)))
                        elif isinstance(v, (int, np.integer)):
                            out.append(int(v))
                        elif isinstance(v, (float, np.floating)):
                            out.append(int(v))
                        else:
                            out.append(int(v))
                    except Exception:
                        out.append(0)
                return out
            try:
                return [int(x)]
            except Exception:
                return None

        # ---- params ----
        fm     = hparams['frames_multiple']
        hop    = hparams['hop_size']
        stride = hparams.get('vae_stride', 8)
        fm_wav = fm * hop
        sr     = hparams['audio_sample_rate']
        max_spk_num = hparams.get('max_spk_num', 8)
        use_sparse_dur = hparams.get('use_sparse_dur', False)

        meta_path = raw_item
        meta_item = json.load(open(meta_path))
        turns     = meta_item['turns']
        segments  = meta_item['segments']

        for segment in segments:
            # 临时容器（只有 turn 全部校验通过才 append）
            spk_map = {}  # 原 spk -> 连续 1-based
            wav_lst, frame_spk_mask_lst = [], []
            ph_list, tone_list, m2p_list, spk_mask_ph_list = [], [], [], []

            # === 统一 text/caption 到 <S{sid}>...</S{sid}> ===
            res_text_parts, res_cap_parts = [], []
            kept_turn_sids = []     # 只统计“真正保留”的 turn
            change_of_spk = 1
            ref_wav_start_turn_idx = 0

            for turn_idx in segment['turn_idxs']:
                turn = turns[turn_idx]

                # 原逻辑：先过滤 text 为空
                text = turn.get('text')
                if not text or not str(text).strip():
                    continue

                # 先准备说话人 sid（1-based）
                raw_spk = turn.get('spk', 'spk_unk')
                sid = spk_map.setdefault(raw_spk, len(spk_map) + 1)

                # 预取并校验：phone / mel2ph / wav（全部可用才真正累加）
                ph_enc = turn.get('phone_encoded')
                m2p_raw = turn.get('mel2ph')
                if ph_enc is None or len(ph_enc) == 0 or m2p_raw is None or len(m2p_raw) == 0:
                    continue  # 这条 turn 直接跳过

                # 读音频
                wav_rel = turn.get('wav_path').replace('wavs/16k160/', 'wavs/24k/')
                if not wav_rel:
                    continue
                full_wav = os.path.join('/mnt/bn/lq-ads-aigc/renyi/tts_datasets_bak/InteractiveDialogue', wav_rel)
                try:
                    wav_np, _ = librosa.load(full_wav, sr=sr)
                except Exception:
                    continue
                if wav_np is None or len(wav_np) == 0:
                    continue
                wav_np = np.asarray(wav_np, dtype=np.float32)

                # —— 文本与 caption：统一包裹为 <S{sid}>...</S{sid}> —— #
                seg_txt = f'<S{sid}>{text}</S{sid}>'
                res_text_parts.append(seg_txt)
                res_cap_parts.append(seg_txt)

                kept_turn_sids.append(sid)
                if len(kept_turn_sids) >= 2 and kept_turn_sids[-1] != kept_turn_sids[-2]:
                    change_of_spk += 1
                    if change_of_spk == 3:
                        ref_wav_start_turn_idx = len(wav_lst)

                # 累加音频与帧级 one-hot（旧 spk_mask → frame_spk_mask）
                one_hot = np.zeros((len(wav_np), max_spk_num), dtype=np.int16)
                col = min(max(sid - 1, 0), max_spk_num - 1)
                one_hot[:, col] = 1
                wav_lst.append(wav_np)
                frame_spk_mask_lst.append(one_hot)

                # 累加 phone/tone/m2p 与 phone 对齐的 spk_mask
                ph_i = torch.as_tensor(ph_enc, dtype=torch.long)

                # —— tone 的健壮读取与数值化 —— #
                tone_raw = turn.get('tone_encoded')
                tone_enc_clean = _to_int_list_maybe(tone_raw) if tone_raw is not None else None
                if (tone_enc_clean is None) or (len(tone_enc_clean) != len(ph_enc)):
                    tn_i = torch.zeros_like(ph_i, dtype=torch.long)
                else:
                    tn_i = torch.as_tensor(tone_enc_clean, dtype=torch.long)

                m2p_i = torch.as_tensor(m2p_raw, dtype=torch.long)  # 1-based
                ph_list.append(ph_i)
                tone_list.append(tn_i)
                m2p_list.append(m2p_i)
                spk_mask_ph_list.append(torch.full((ph_i.numel(),), int(sid), dtype=torch.long))

            # === 段级检查（按“保留下来的 turn”统计）===
            if len(wav_lst) == 0:
                continue

            # 至少 4 个有效 turn、且有来回
            if len(kept_turn_sids) < 4:
                continue
            num_conversations = 1
            for i in range(1, len(kept_turn_sids)):
                if kept_turn_sids[i] != kept_turn_sids[i-1]:
                    num_conversations += 1
            if num_conversations // 2 < 2:
                continue

            # === 拼接音频与帧级掩码 ===
            wav_cat = torch.from_numpy(np.concatenate(wav_lst))
            frame_spk_mask = torch.from_numpy(np.concatenate(frame_spk_mask_lst, axis=0))
            # 对齐到 fm_wav
            T = (wav_cat.shape[0] // fm_wav) * fm_wav
            wav_cat = wav_cat[:T]
            frame_spk_mask = frame_spk_mask[:T]

            # === 拼接 phones/tone/mel2ph（1-based offset 修正）===
            ph_offset = 0
            m2p_fixed = []
            for m2p_i, ph_i in zip(m2p_list, ph_list):
                if ph_offset > 0:
                    m2p_i = m2p_i + (m2p_i > 0).long() * ph_offset
                m2p_fixed.append(m2p_i)
                ph_offset += ph_i.numel()

            ph_token = torch.cat(ph_list, dim=0)
            tone     = torch.cat(tone_list, dim=0)
            mel2ph   = torch.cat(m2p_fixed, dim=0)
            spk_mask_ph = torch.cat(spk_mask_ph_list, dim=0)

            # 使 mel2ph 与当前音频的 mel 帧长度一致（再对齐到 fm）
            mel_len = wav_cat.shape[0] // hop
            if mel2ph.numel() < mel_len:
                pad_len = mel_len - mel2ph.numel()
                last = mel2ph[-1] if mel2ph.numel() > 0 else torch.tensor(0, dtype=torch.long)
                mel2ph = torch.cat([mel2ph, last.repeat(pad_len)], dim=0)
            mel2ph = mel2ph[:mel_len]
            mel2ph = mel2ph[: (mel_len // fm) * fm]

            # === 计算 dur，并按需在 worker 内生成 mel2ph_sparse（严格与 mel2ph 等长、对齐到 fm）===
            dur = _mel2token_to_dur(mel2ph)
            mel2ph_sparse = None
            if use_sparse_dur:
                # compute_mel2aug_from_dur 的输入是 “每个 phone 的 mel 帧时长”
                _m2a = compute_mel2aug_from_dur(
                    dur.cpu().numpy().tolist(),
                    gap_mode=hparams.get('sparse_dur_mode', 'proportional'),
                    gap_frames=hparams.get('sparse_dur_frames', 4),
                    gap_alpha=hparams.get('sparse_dur_alpha', 0.2),
                    min_keep=hparams.get('sparse_dur_min_keep', 1),
                    keep_ratio=hparams.get('sparse_dur_keep_ratio'),
                    symmetric=hparams.get('sparse_dur_symmetric', True),
                )
                mel2ph_sparse = torch.as_tensor(_m2a, dtype=torch.long)
                # 等长裁剪/补齐，并保持与 mel2ph 的 fm 对齐
                target_len = mel2ph.shape[0]  # 已对齐到 fm
                if mel2ph_sparse.numel() < target_len:
                    pad_len = target_len - mel2ph_sparse.numel()
                    last = mel2ph_sparse[-1] if mel2ph_sparse.numel() > 0 else torch.tensor(0, dtype=torch.long)
                    mel2ph_sparse = torch.cat([mel2ph_sparse, last.repeat(pad_len)], dim=0)
                mel2ph_sparse = mel2ph_sparse[:target_len]

            # === 文本与 caption（统一为 <S{sid}>...</S{sid}>）===
            text_merged    = ''.join(res_text_parts)
            caption_merged = text_merged

            # ctx 起点：按“第三次换人”的 turn 边界，再随机微调
            pre_len = 0
            if 0 < ref_wav_start_turn_idx <= len(wav_lst):
                pre_len = np.concatenate(wav_lst[:ref_wav_start_turn_idx]).shape[0]
            ref_wav_start = (pre_len // fm_wav) * fm_wav
            max_idx = min(int(wav_cat.shape[0] * 0.9), wav_cat.shape[0] - 20000)
            if max_idx > ref_wav_start:
                ref_wav_start = (random.randint(ref_wav_start, max_idx) // fm_wav) * fm_wav

            ctx_wav = wav_cat[:ref_wav_start]
            ctx_mask = torch.zeros((wav_cat.shape[0], 1), dtype=torch.float32)
            ctx_mask[:ref_wav_start] = 1.0
            ctx_mask = ctx_mask[:: hop * stride]

            # 产出
            item = {
                'wav': wav_cat,
                'text': text_merged,
                'caption': caption_merged,
                'spk_mask': spk_mask_ph.to(torch.long),   # phone-level, 1-based
                'frame_spk_mask': frame_spk_mask,         # 采样点级 one-hot（旧，改名保留）
                'ctx_wav': ctx_wav,
                'ctx_mask': ctx_mask,
                'ph_token': ph_token.to(torch.long),
                'tone': tone.to(torch.long),
                'mel2ph': mel2ph.to(torch.long),
                'dur': dur.to(torch.long),
            }
            if mel2ph_sparse is not None:
                item['mel2ph_sparse'] = mel2ph_sparse.to(torch.long)

            item['len'] = int(item['wav'].shape[0] / hop / stride)

            yield item


    def collater(self, samples):
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
        batch['text'] = [s['text'] for s in samples]
        batch['spk_mask'] = collate_xd([s['spk_mask'] for s in samples], 0.0) if 'spk_mask' in samples[0] else None

        if 'wav_w2v2' in samples[0]:
            batch['wavs_w2v2'] = collate_xd([s['wav_w2v2'] for s in samples], 0.0)
            batch['wav_w2v2_lengths'] = torch.LongTensor([s['wav_w2v2'].shape[0] for s in samples])

        if not hasattr(self, 'backup_batch') or self.backup_batch is None or random.random() < 0.001:
            self.backup_batch = batch

        return batch


class DialogueSegmentShmDataset(BaseShmDataset):
    def get_dataset_meta(self):
        meta_paths = multiprocess_glob(f'/mnt/bn/sa-ag-data/liruiqi/data/speech/XYZ_20w/metas/*/*.json', num_workers=128)
        return meta_paths, len(meta_paths)
    
    def prepare_reader(self, dataset_meta, global_stores):
        return 1
    
    def read_fn(self, idx, reader_pack, global_stores):
        return self.dataset_meta[idx]
    
    def process_item(self, raw_item, hparams, global_stores):
        if self.use_fast_dataloader:
            buckets = [400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2400, 2800, 
                       3200, 3600, 4000, 4500, 5000, 5500, 6000, 7000, 8000, 9000, 10000, 
                       11000, 12000, 13000, 14000, 15000, 16000, 18000, 20000, 40000, 60000]
            batcher = get_from_global_stores(
                'batcher', global_stores,
                lambda: BucketBatcher(
                    buckets=buckets,
                    dynamic_batch=hparams.get("dynamic_batch", True),
                    batch_size=hparams['max_sentences'],
                    maximum_bucket_size=hparams['max_tokens'],
                    length_fn=lambda x: x['len'],
                )
            )

        for item in self._process_item(raw_item, hparams):
            if item is None:
                continue
            if self.use_fast_dataloader:
                batch = batcher.collate_batch(item)
                if batch is not None and len(batch) > 0:
                    yield batch
            else:
                yield [item]

    def _process_item(self, raw_item, hparams):
        meta_path = raw_item
        meta_item = json.load(open(meta_path))
        max_spk_num = hparams.get('max_spk_num', 8)
        fm = hparams['frames_multiple']
        fm_wav = hparams['frames_multiple'] * hparams['hop_size']

        turns = meta_item['turns']
        segments = meta_item['segments']

        for segment in segments:
            spk_map = {}
            spk_lst = []
            res_text = ''
            wav_lst = []
            spk_mask_lst = []
            ref_wav_start = 0
            change_of_spk = 1
            for turn_idx in segment['turn_idxs']:
                turn = turns[turn_idx]
                text = turn['text']
                if text is None:
                    continue
                spk_name = turn['spk']
                if spk_name not in spk_map:
                    spk_map[spk_name] = len(spk_map)
                spk_id = spk_map[spk_name]
                if len(spk_lst) > 0 and spk_id == spk_lst[-1]:
                    res_text += text
                else:
                    change_of_spk += 1
                    res_text += f'<SPK>{spk_id}</SPK>' + text
                spk_lst.append(spk_id)
                if change_of_spk == 3:
                    ref_wav_start = len(wav_lst)

                wav_path = turn['wav_path']
                wav_path = os.path.join('/mnt/bn/lq-ads-aigc/renyi/tts_datasets_bak/InteractiveDialogue', wav_path)
                wav, _ = librosa.load(wav_path, sr=hparams['audio_sample_rate'])
                wav_lst.append(wav)

                spk_mask = np.zeros((len(wav), max_spk_num), dtype=int)
                spk_mask[:, spk_id] = 1
                spk_mask_lst.append(spk_mask)
            
            # check turns
            if len(spk_lst) < 4:
                continue
            num_conversations = 1
            for i in range(1, len(spk_lst)):
                if spk_lst[i] != spk_lst[i-1]:
                    num_conversations += 1
            if num_conversations // 2 < 2:
                continue

            item = {
                'wav': torch.from_numpy(np.concatenate(wav_lst)),
                'text': res_text,
                'spk_mask': torch.from_numpy(np.concatenate(spk_mask_lst, axis=0)),
            }
            item['wav'] = item['wav'][:len(item['wav']) // fm_wav * fm_wav]
            item['spk_mask'] = item['spk_mask'][:len(item['spk_mask']) // fm_wav * fm_wav]
            item['len'] = int(item['wav'].shape[0] / hparams['hop_size'] / hparams['vae_stride'])

            ref_wav_start = np.concatenate(wav_lst[:ref_wav_start]).shape[0] // fm_wav * fm_wav
            max_idx = min(int(len(item['wav']) * 0.9), len(item['wav']) - 20000)
            if max_idx > ref_wav_start:
                ref_wav_start = random.randint(ref_wav_start, max_idx) // fm_wav * fm_wav
            ctx_wav = deepcopy(item['wav'])
            ctx_wav = ctx_wav[:ref_wav_start]
            item['ctx_wav'] = ctx_wav
            ctx_mask = torch.zeros_like(item['wav'])[:, None]
            ctx_mask[:ref_wav_start] = 1.0
            ctx_mask = ctx_mask[::hparams['hop_size']*hparams['vae_stride']]
            item['ctx_mask'] = ctx_mask

            yield item

    def collater(self, samples):
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
        batch['text'] = [s['text'] for s in samples]
        batch['spk_mask'] = collate_xd([s['spk_mask'] for s in samples], 0.0) if 'spk_mask' in samples[0] else None
        batch['ctx_wavs'] = collate_xd([s['ctx_wav'] for s in samples], 0.0)
        batch['ctx_mask'] = collate_xd([s['ctx_mask'] for s in samples], 0.0)

        if 'wav_w2v2' in samples[0]:
            batch['wavs_w2v2'] = collate_xd([s['wav_w2v2'] for s in samples], 0.0)
            batch['wav_w2v2_lengths'] = torch.LongTensor([s['wav_w2v2'].shape[0] for s in samples])

        if not hasattr(self, 'backup_batch') or self.backup_batch is None or random.random() < 0.001:
            self.backup_batch = batch

        return batch

from modules.tts.ar_dur.commons.align_ops import compute_mel2aug_from_dur

class DialogueSegmentEmbDataset(BaseShmDataset):
    def get_dataset_meta(self):
        meta_paths = multiprocess_glob(
            f'/mnt/bn/sa-ag-data/zhangyu.34/data/speech/XYZ_20w/metas_with_tson_16k160_final_hard/*/*.json',
            num_workers=128
        )
        return meta_paths, len(meta_paths)
    
    def prepare_reader(self, dataset_meta, global_stores):
        return 1

    def read_fn(self, idx, reader_pack, global_stores):
        return self.dataset_meta[idx]

    # ==================== 新增/修改：支持 phone/mel2ph，并输出与 processer_fn_* 对齐的字段 ====================

    def process_item(self, raw_item, hparams, global_stores):
        """
        逐 meta 产生样本，若 use_fast_dataloader=True，则在本函数内部做 bucket 动态组 batch。
        产出的字段与 processer_fn_dit_wav_text_multispk_emb 保持一致：
            必选：wav, text, caption, ph_token, mel2ph, dur, spk_mask(按phone), ctx_wav, ctx_mask
            可选：tone（若 JSON 无 tone_encoded 则全 0）
        另保留原先按采样点的一热掩码，重命名为 frame_spk_mask，避免与新的 phone 对齐 spk_mask 混淆。
        文本/说明统一格式：将每个 turn 的文本包装为 <S{sid}>...</S{sid}>（如 <S1>... </S1>）。
        """
        # fast dataloader 下使用 BucketBatcher
        if self.use_fast_dataloader:
            buckets = [400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2400, 2800, 
                       3200, 3600, 4000, 4500, 5000, 5500, 6000, 7000, 8000, 9000, 10000, 
                       11000, 12000, 13000, 14000, 15000, 16000, 18000, 20000, 40000, 60000]
            batcher = get_from_global_stores(
                'batcher', global_stores,
                lambda: BucketBatcher(
                    buckets=buckets,
                    dynamic_batch=hparams.get("dynamic_batch", True),
                    batch_size=hparams['max_sentences'],
                    maximum_bucket_size=hparams['max_tokens'],
                    length_fn=lambda x: x['len'],
                )
            )

        for item in self._process_item(raw_item, hparams):
            if item is None:
                continue
            if self.use_fast_dataloader:
                batch = batcher.collate_batch(item)
                if batch is not None and len(batch) > 0:
                    yield batch
            else:
                yield [item]

    def _process_item(self, raw_item, hparams):
        """
        与原版逻辑一致：
        - turn 级只先按 text 过滤（None/空白跳过）
        - 只有当该 turn 的 text / wav / phone / mel2ph 都可用时，才真正累加（防止“先加后弹”把列表清空）
        - 段级再做最小轮次/来回检查；若过滤后一个 turn 都没有，直接 continue，绝不拼接空列表
        """
        import re

        # ---- helpers ----
        def _mel2token_to_dur(m2p: torch.LongTensor) -> torch.LongTensor:
            if m2p.numel() == 0:
                return torch.zeros(0, dtype=torch.long)
            mx = int(m2p.max().item()) if m2p.numel() > 0 else 0
            if mx == 0:
                return torch.zeros(0, dtype=torch.long)
            cnts = torch.bincount(m2p.clamp_min(0), minlength=mx + 1)
            return cnts[1:].to(torch.long)

        def _to_int_list_maybe(x):
            """
            将多种可能形式的 tone 安全转成整型 list。
            支持：
            - list/tuple/ndarray，元素可为 str/int/float
            - 纯字符串，如 "1 2 3" / "1,2,3"
            - JSON 字符串，如 "[1, 2, 3]" 或 '["0","4","4"]'
            转换失败返回 None。
            """
            if x is None:
                return None

            # 若是字符串，尝试按 JSON 解析；失败则用正则抽取数字
            if isinstance(x, str):
                s = x.strip()
                # JSON 风格的列表
                if (s.startswith('[') and s.endswith(']')) or (s.startswith('(') and s.endswith(')')):
                    try:
                        parsed = json.loads(s.replace('(', '[').replace(')', ']'))
                        return _to_int_list_maybe(parsed)
                    except Exception:
                        pass
                # 普通分隔字符串：先用正则提取数字（含负号/小数），再转 int
                nums = re.findall(r'-?\d+\.?\d*', s)
                if len(nums) == 0:
                    return None
                try:
                    return [int(float(t)) for t in nums]
                except Exception:
                    return None

            # 若是可迭代序列
            if isinstance(x, (list, tuple, np.ndarray)):
                out = []
                for v in x:
                    try:
                        if isinstance(v, str):
                            v = v.strip()
                            if v == '':
                                out.append(0)
                                continue
                            out.append(int(float(v)))
                        elif isinstance(v, (int, np.integer)):
                            out.append(int(v))
                        elif isinstance(v, (float, np.floating)):
                            out.append(int(v))
                        else:
                            # 兜底：尝试强转
                            out.append(int(v))
                    except Exception:
                        out.append(0)
                return out

            # 其它类型（标量等）
            try:
                return [int(x)]
            except Exception:
                return None

        # ---- params ----
        fm     = hparams['frames_multiple']
        hop    = hparams['hop_size']
        stride = hparams.get('vae_stride', 8)
        fm_wav = fm * hop
        sr     = hparams['audio_sample_rate']
        max_spk_num = hparams.get('max_spk_num', 8)

        meta_path = raw_item
        meta_item = json.load(open(meta_path))
        turns     = meta_item['turns']
        segments  = meta_item['segments']

        for segment in segments:
            # 临时容器（只有 turn 全部校验通过才 append）
            spk_map = {}  # 原 spk -> 连续 1-based
            wav_lst, frame_spk_mask_lst = [], []
            ph_list, tone_list, m2p_list, spk_mask_ph_list = [], [], [], []

            # === 修改点：统一 text/caption 到 <S{sid}>...</S{sid}> ===
            res_text_parts, res_cap_parts = [], []
            kept_turn_sids = []     # 只统计“真正保留”的 turn
            change_of_spk = 1
            ref_wav_start_turn_idx = 0

            for turn_idx in segment['turn_idxs']:
                turn = turns[turn_idx]

                # 原逻辑：先过滤 text 为空
                text = turn.get('text')
                if not text or not str(text).strip():
                    continue

                # 先准备说话人 sid（1-based）
                raw_spk = turn.get('spk', 'spk_unk')
                sid = spk_map.setdefault(raw_spk, len(spk_map) + 1)

                # 预取并校验：phone / mel2ph / wav（全部可用才真正累加）
                ph_enc = turn.get('phone_encoded')
                m2p_raw = turn.get('mel2ph')
                if ph_enc is None or len(ph_enc) == 0 or m2p_raw is None or len(m2p_raw) == 0:
                    continue  # 这条 turn 直接跳过

                # 读音频
                wav_rel = turn.get('wav_path').replace('wavs/16k160/', 'wavs/24k/')
                if not wav_rel:
                    continue
                full_wav = os.path.join('/mnt/bn/lq-ads-aigc/renyi/tts_datasets_bak/InteractiveDialogue', wav_rel)
                try:
                    wav_np, _ = librosa.load(full_wav, sr=sr)
                except Exception:
                    continue
                if wav_np is None or len(wav_np) == 0:
                    continue
                wav_np = np.asarray(wav_np, dtype=np.float32)

                # —— 文本与 caption：统一包裹为 <S{sid}>...</S{sid}> —— #
                seg_txt = f'<S{sid}>{text}</S{sid}>'
                res_text_parts.append(seg_txt)
                res_cap_parts.append(seg_txt)

                kept_turn_sids.append(sid)
                if len(kept_turn_sids) >= 2 and kept_turn_sids[-1] != kept_turn_sids[-2]:
                    change_of_spk += 1
                    if change_of_spk == 3:
                        ref_wav_start_turn_idx = len(wav_lst)

                # 累加音频与帧级 one-hot（旧 spk_mask → frame_spk_mask）
                one_hot = np.zeros((len(wav_np), max_spk_num), dtype=np.int16)
                col = min(max(sid - 1, 0), max_spk_num - 1)
                one_hot[:, col] = 1
                wav_lst.append(wav_np)
                frame_spk_mask_lst.append(one_hot)

                # 累加 phone/tone/m2p 与 phone 对齐的 spk_mask
                ph_i = torch.as_tensor(ph_enc, dtype=torch.long)

                # —— tone 的健壮读取与数值化 —— #
                tone_raw = turn.get('tone_encoded')
                tone_enc_clean = _to_int_list_maybe(tone_raw) if tone_raw is not None else None
                if (tone_enc_clean is None) or (len(tone_enc_clean) != len(ph_enc)):
                    tn_i = torch.zeros_like(ph_i, dtype=torch.long)
                else:
                    tn_i = torch.as_tensor(tone_enc_clean, dtype=torch.long)

                m2p_i = torch.as_tensor(m2p_raw, dtype=torch.long)  # 1-based
                ph_list.append(ph_i)
                tone_list.append(tn_i)
                m2p_list.append(m2p_i)
                spk_mask_ph_list.append(torch.full((ph_i.numel(),), int(sid), dtype=torch.long))

            # === 段级检查（按“保留下来的 turn”统计）===
            if len(wav_lst) == 0:
                continue

            # 至少 4 个有效 turn、且有来回
            if len(kept_turn_sids) < 4:
                continue
            num_conversations = 1
            for i in range(1, len(kept_turn_sids)):
                if kept_turn_sids[i] != kept_turn_sids[i-1]:
                    num_conversations += 1
            if num_conversations // 2 < 2:
                continue

            # === 拼接音频与帧级掩码 ===
            wav_cat = torch.from_numpy(np.concatenate(wav_lst))
            frame_spk_mask = torch.from_numpy(np.concatenate(frame_spk_mask_lst, axis=0))
            # 对齐到 fm_wav
            T = (wav_cat.shape[0] // fm_wav) * fm_wav
            wav_cat = wav_cat[:T]
            frame_spk_mask = frame_spk_mask[:T]

            # === 拼接 phones/tone/mel2ph（1-based offset 修正）===
            ph_offset = 0
            m2p_fixed = []
            for m2p_i, ph_i in zip(m2p_list, ph_list):
                if ph_offset > 0:
                    m2p_i = m2p_i + (m2p_i > 0).long() * ph_offset
                m2p_fixed.append(m2p_i)
                ph_offset += ph_i.numel()

            ph_token = torch.cat(ph_list, dim=0)
            tone     = torch.cat(tone_list, dim=0)
            mel2ph   = torch.cat(m2p_fixed, dim=0)
            spk_mask_ph = torch.cat(spk_mask_ph_list, dim=0)

            # 使 mel2ph 与当前音频的 mel 帧长度一致（再对齐到 fm）
            mel_len = wav_cat.shape[0] // hop
            if mel2ph.numel() < mel_len:
                pad_len = mel_len - mel2ph.numel()
                last = mel2ph[-1] if mel2ph.numel() > 0 else torch.tensor(0, dtype=torch.long)
                mel2ph = torch.cat([mel2ph, last.repeat(pad_len)], dim=0)
            mel2ph = mel2ph[:mel_len]
            mel2ph = mel2ph[: (mel_len // fm) * fm]

            dur = _mel2token_to_dur(mel2ph)
            mel2ph_sparse = None
            if hparams.get('use_sparse_dur', False):
                # compute_mel2aug_from_dur 的输入是“每个 phone 的 mel 帧时长”（与你的 dur 语义一致）
                _m2a = compute_mel2aug_from_dur(
                    dur.cpu().numpy().tolist(),
                    gap_mode=hparams.get('sparse_dur_mode', 'proportional'),
                    gap_frames=hparams.get('sparse_dur_frames', 4),
                    gap_alpha=hparams.get('sparse_dur_alpha', 0.2),
                    min_keep=hparams.get('sparse_dur_min_keep', 1),
                    keep_ratio=hparams.get('sparse_dur_keep_ratio'),
                    symmetric=hparams.get('sparse_dur_symmetric', True),
                )
                # 转 tensor，并与 mel2ph 保持等长和对齐到 frames_multiple
                mel2ph_sparse = torch.as_tensor(_m2a, dtype=torch.long)
                target_len = mel2ph.shape[0]  # 已对齐到 fm
                if mel2ph_sparse.numel() < target_len:
                    pad_len = target_len - mel2ph_sparse.numel()
                    last = mel2ph_sparse[-1] if mel2ph_sparse.numel() > 0 else torch.tensor(0, dtype=torch.long)
                    mel2ph_sparse = torch.cat([mel2ph_sparse, last.repeat(pad_len)], dim=0)
                mel2ph_sparse = mel2ph_sparse[:target_len]

            # === 文本与 caption（统一为 <S{sid}>...</S{sid}>）===
            text_merged    = ''.join(res_text_parts)
            caption_merged = text_merged

            # ctx 起点：按“第三次换人”的 turn 边界，再随机微调
            pre_len = 0
            if 0 < ref_wav_start_turn_idx <= len(wav_lst):
                pre_len = np.concatenate(wav_lst[:ref_wav_start_turn_idx]).shape[0]
            ref_wav_start = (pre_len // fm_wav) * fm_wav
            max_idx = min(int(wav_cat.shape[0] * 0.9), wav_cat.shape[0] - 20000)
            if max_idx > ref_wav_start:
                ref_wav_start = (random.randint(ref_wav_start, max_idx) // fm_wav) * fm_wav

            ctx_wav = wav_cat[:ref_wav_start]
            ctx_mask = torch.zeros((wav_cat.shape[0], 1), dtype=torch.float32)
            ctx_mask[:ref_wav_start] = 1.0
            ctx_mask = ctx_mask[:: hop * stride]

            # 产出
            item = {
                'wav': wav_cat,
                'text': text_merged,
                'caption': caption_merged,
                'spk_mask': spk_mask_ph.to(torch.long),
                'frame_spk_mask': frame_spk_mask,
                'ctx_wav': ctx_wav,
                'ctx_mask': ctx_mask,
                'ph_token': ph_token.to(torch.long),
                'tone': tone.to(torch.long),
                'mel2ph': mel2ph.to(torch.long),
                'dur': dur.to(torch.long),
            }
            if mel2ph_sparse is not None:
                item['mel2ph_sparse'] = mel2ph_sparse.to(torch.long)

            item['len'] = int(item['wav'].shape[0] / hop / stride)

            yield item


    def collater(self, samples):
        """
        支持 fast-dataloader（samples 由若干条样本组成）与回退备份逻辑。
        汇总字段尽量与 DiTWavTextDataset.collater 对齐，新增：
            - 'frame_spk_mask'：采样点级 one-hot（原先 spk_mask 改名后汇总）
            - 'spk_mask'     ：phone 对齐的说话人 id（1-based）
        """
        # fast path：BucketBatcher 输出的是 [list_of_samples] 包一层
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
        ctx_wavs = collate_xd([s['ctx_wav'] for s in samples], 0.0)

        batch = {
            'nsamples': len(samples),
            'wavs': wavs,
            'wav_lengths': wav_lengths,
            'ctx_wavs': ctx_wavs,
            'ctx_mask': collate_xd([s['ctx_mask'] for s in samples], 0),
            'text': [s['text'] for s in samples],
            # caption 与 text 同格式：<S{sid}>...</S{sid}>
            'caption': [s['caption'] for s in samples] if 'caption' in samples[0] else None,
        }

        # 与 DiTWavTextDataset 对齐的字段
        if 'ph_token' in samples[0]:
            batch['ph_tokens'] = collate_xd([s['ph_token'] for s in samples], 0)
            batch['txt_lengths'] = torch.LongTensor([s['ph_token'].numel() for s in samples])
        if 'tone' in samples[0]:
            batch['tone'] = collate_xd([s['tone'] for s in samples], 0)
        if 'mel2ph' in samples[0]:
            batch['mel2ph'] = collate_xd([s['mel2ph'] for s in samples], 0)
        if 'dur' in samples[0]:
            batch['dur'] = collate_xd([s['dur'] for s in samples], 0)
            batch['dur_len'] = torch.LongTensor([s['dur'].shape[0] for s in samples])
        if 'mel2ph_sparse' in samples[0]:
            batch['mel2ph_sparse'] = collate_xd([s['mel2ph_sparse'] for s in samples], 0)

        # 新旧两个 mask：phone 对齐（新）与采样点级（旧，改名）
        if 'spk_mask' in samples[0]:
            batch['spk_mask'] = collate_xd([s['spk_mask'] for s in samples], 0)         # [B, T_ph]

        if not hasattr(self, 'backup_batch') or self.backup_batch is None or random.random() < 0.001:
            self.backup_batch = batch

        return batch



if __name__ == '__main__':
    import soundfile as sf
    from utils.commons.io import json_dump
    from tqdm import tqdm
    hparams = {
        'exp_name': 'test',
        'audio_sample_rate': 24000,
        'hop_size': 240,
        'max_sentences': 200,
        'max_tokens': 2000,
        'max_spk_num': 8,
        'tgt_duration_min': 20,
        'tgt_duration_max': 60,
        'fast_ds_prefetch_steps': 8,
        'ds_workers': 4,
        'frames_multiple': 8,
        'vae_stride': 4,
    }

    dataset = DialogueSegmentShmDataset(
        prefix='train', hparams=hparams, use_fast_dataloader=True, rank_id=0, world_size=1, batch_size=1
    )
    dataloader = dataset.get_dataloader(seed=1234, num_workers=4)

    temp_dir = 'user/temp/test_dl'

    for idx, batch in tqdm(enumerate(dataloader)):
        if idx == 0:
            print(batch.keys())
        wavs = batch['wavs']

        wav = wavs[0].numpy()
        sf.write(os.path.join(temp_dir, f'{idx}.wav'), wav, hparams['audio_sample_rate'], 'PCM_16')
        ctx_wav = batch['ctx_wavs'][0].numpy()
        sf.write(os.path.join(temp_dir, f'{idx}_ctx.wav'), ctx_wav, hparams['audio_sample_rate'], 'PCM_16')

        texts = batch['text']
        json_dump({'text': texts[0]}, os.path.join(temp_dir, f'{idx}.json'))
        
        if idx > 10:
            break

