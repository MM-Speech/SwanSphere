import collections
import collections.abc
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))

import os
import pickle
import re

import torch
import numpy as np
import librosa

from utils.commons.hparams import hparams
from utils.commons.io import print_once
from utils.commons.base_shm_dataset import BaseFalconReaderShmDataset, get_from_global_stores
from utils.commons.dataset_utils import collate_xd
from utils.dataset.batcher import BucketBatcher

class VAEShmDataset(BaseFalconReaderShmDataset):
    # ============ reader / meta ============
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
        
    """
    等长 chunk + 固定 batch_size 累积：
      - 读取 vocal / wav，重采样至 audio_sample_rate
      - 对齐到 frames_multiple * hop_size 的整倍数
      - 切成等长 chunk（末尾不足一个 chunk 的部分丢弃）
      - process_item 使用 BucketBatcher 在 use_fast_dataloader=True 时按 batch_size 累积
      - collater 返回:
          {
              'nsamples': B,
              'wavs': FloatTensor[B, T],  # 同长
              'wav_lengths': LongTensor[B]
          }
    """

    # ============ 固定 batch_size 的 batcher ============
    def _get_batcher(self, global_stores):
        return get_from_global_stores(
            'batcher', global_stores,
            lambda: BucketBatcher(
                buckets=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550,
                         600, 650, 700, 750, 800, 850, 900, 950, 1000, 1200, 1400,
                         1600, 1800, 2000, 2400, 2800, 3000],
                # 关键：关闭动态 batch，仅按 batch_size 固定数量累积
                dynamic_batch=False,
                batch_size=hparams['max_sentences'],
                # 这里的 maximum_bucket_size（max_tokens）不再参与控制，可设为 None 或保留默认值
                maximum_bucket_size=hparams.get('max_tokens', None),
                length_fn=lambda x: x['len'],  # 仍提供 len，方便放入统一 bucket（长度固定）
            )
        )

    # ============ 主流程：切 chunk + 按固定 batch_size 累积 ============
    def process_item(self, raw_item, hparams, global_stores, i_worker, n_worker):
        batcher = self._get_batcher(global_stores) if self.use_fast_dataloader else None

        # 将该条 wav 切成多个 chunk，逐个送入 batcher 累积
        for item in self._process_item(raw_item, hparams, global_stores, i_worker, n_worker):
            if item is None:
                continue

            if self.use_fast_dataloader:
                # 关闭动态 batch 后，batcher 会按 batch_size 凑满再返回
                batch = batcher.collate_batch(item)
                if batch is not None and len(batch) > 0:
                    yield batch  # 这是一个 list[item]
            else:
                # 非 fast_dataloader：逐条输出，让外层 DataLoader 用 collater 统一堆叠
                yield [item]

    def _process_item(self, raw_item, hparams, global_stores=None, i_worker=None, n_worker=None):
        fm = hparams['frames_multiple']
        hop_size = hparams['hop_size']
        fm_wav = fm * hop_size
        sr = hparams['audio_sample_rate']

        # 选用 vocal 或 wav
        use_vocal = hparams.get('use_vocal', True)
        if use_vocal and isinstance(raw_item.get('vocal'), np.ndarray):
            wav = raw_item['vocal'].astype(float)
            org_sr = raw_item.get('vocal_sr', sr)
        else:
            wav = (raw_item.get('wav') or np.zeros(0, dtype=float)).astype(float)
            org_sr = raw_item.get('sr', sr)

        # 重采样
        if sr != org_sr and wav.size > 0:
            wav = librosa.resample(wav, orig_sr=org_sr, target_sr=sr)

        # 对齐到 frames_multiple * hop_size 的整倍数
        if wav.size > 0 and fm_wav > 0:
            wav = wav[: (len(wav) // fm_wav) * fm_wav]

        # 跳过空样本
        if wav.size == 0:
            return

        # ============ 计算 chunk 长度（最终对齐 fm_wav） ============
        chunk_len = int(hparams.get('chunk_size', 0))  # 单位：采样点
        if chunk_len <= 0:
            chunk_seconds = float(hparams.get('chunk_seconds', 0.0))
            if chunk_seconds > 0:
                chunk_len = int(round(sr * chunk_seconds))

        # 回退：如未配置，则至少取一个 fm_wav；若 fm_wav=0，则默认 1 秒
        if chunk_len <= 0:
            chunk_len = fm_wav if fm_wav > 0 else int(sr * 1.0)

        # 对齐到 fm_wav 的整倍数
        if fm_wav > 0:
            chunk_len = (chunk_len // fm_wav) * fm_wav
            if chunk_len <= 0:
                chunk_len = fm_wav

        # ============ 切 chunk（丢弃末尾不足一个 chunk 的部分） ============
        total = len(wav)
        n_chunks = total // chunk_len
        if n_chunks <= 0:
            return  # 该条 wav 不到一个 chunk，跳过

        # 给 batcher 的长度单位：帧（虽然关闭动态，但保留统一接口）
        if hop_size > 0:
            chunk_len_frames = chunk_len // hop_size
        else:
            chunk_len_frames = chunk_len  # 兜底

        for i in range(n_chunks):
            s = i * chunk_len
            e = s + chunk_len
            chunk = wav[s:e].astype(np.float32)

            yield {
                'wav': chunk,
                'len': int(chunk_len_frames)
            }

    # ============ collater：堆叠同长 chunk ============
    def collater(self, samples):
        # 兼容外层 yield [item] 或 yield batch(list)
        if len(samples) == 0:
            return {}
        if len(samples) == 1 and isinstance(samples[0], list):
            samples = samples[0]

        # 所有 chunk 等长，直接堆叠，无需 padding
        wavs = collate_xd([s['wav'] for s in samples], 0.0)  # FloatTensor[B, T]
        T = wavs.shape[-1]
        wav_lengths = torch.LongTensor([T] * len(samples))

        batch = {
            'nsamples': len(samples),
            'wavs': wavs,
            'wav_lengths': wav_lengths,
        }
        return batch
