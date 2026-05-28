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
from utils.commons.io import get_wav_duration, print_once, load_samples_from_tsv, load_samples_from_jsonl
from utils.commons.base_shm_dataset import BaseFalconReaderShmDataset, get_from_global_stores, save_samples_to_shm
from utils.commons.dataset_utils import collate_xd, SkipLogger
from utils.commons.tensor_utils import convert_to_tensor, convert_to_np
from utils.commons.tos_utils_v2 import TosClient
from utils.commons.hdfs_utils import HDFSClient
from utils.commons.jsonl_utils import get_jsonl_line_by_number, count_jsonl_n_lines, JsonlChunkReader, get_jsonl_lines_by_range
from utils.commons.parquet_utils import ParquetChunkReader
from utils.dataset.batcher import BucketBatcher
from utils.audio.vad import build_vad_model, run_vad_trim
from utils.audio.align import mel2token_to_dur
from utils.audio.align import mel2token_to_dur
from utils.text.split_text import get_word_list
from utils.text.ph_tone_convert import map_phone_to_tokendict
from utils.text.split_text import get_word_list, remove_spaces_between_chinese
from utils.text import is_chinese, is_english

from tasks.tts.dataset_utils.tts_datasets import MegaTTSDataset, FrontendLMDataset
from modules.tts.ar_dur.commons.align_ops import compute_mel2aug_from_dur
from modules.tts.ar_dur.commons.nar_tts_modules import LengthRegulator
import struct
from tqdm import tqdm

DEBUG = False

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

def get_hdfs_file(hdfs_path, save_path, hdfs_clients: dict = None):
    namespace = None
    if hdfs_path.startswith('hdfs://'):
        namespace = hdfs_path.split('://')[1].split('/')[0]
    if hdfs_clients is None:
        hdfs_clients = {}
    if namespace is not None:
        if namespace not in hdfs_clients:
            client = hdfs_clients[namespace] = HDFSClient(namespace=namespace)
        else:
            client = hdfs_clients[namespace]
    else:
        if 'default' not in hdfs_clients:
            client = hdfs_clients['default'] = HDFSClient()
        else:
            client = hdfs_clients['default']
    if not client.check_file_exists(hdfs_path):
        return None
    data = client.get_object(hdfs_path)
    # print(f"{client.namespace = }, {hdfs_path = }, {data is not None = }")
    if data is None:
        return
    os.makedirs(Path(save_path).parent, exist_ok=True)
    with open(save_path, 'wb') as f:
        f.write(data)
    return save_path

def safe_read_path(path, save_path, hdfs_clients: dict = None):
    if path.startswith('hdfs://'):
        return get_hdfs_file(path, save_path, hdfs_clients)
    else:
        return path

def build_jsonl_idx_with_progress(jsonl_path, idx_path):
    """
    为 JSONL 文件创建行索引 idx 文件（与 jsonl 同名加 .idx 后缀）。
    索引格式与 count_jsonl_n_lines / JsonlChunkReader 兼容：每行偏移用 8 字节无符号整数 <Q> 小端表示。
    """
    total_size = os.path.getsize(jsonl_path)
    with open(jsonl_path, 'rb') as f, open(idx_path, 'wb') as idx:
        pos = 0
        with tqdm(total=total_size,
                  unit='B',
                  unit_scale=True,
                  desc=f"Building index for {os.path.basename(jsonl_path)}") as pbar:
            while True:
                line = f.readline()
                if not line:
                    break
                idx.write(struct.pack('<Q', pos))
                line_len = len(line)
                pos += line_len
                pbar.update(line_len)

class BaseTTSShmDataset(BaseFalconReaderShmDataset):
    def controller_fn(self, ds_len, seed, q_to_pull, hparams_, max_epoch=0, n_processor=0):
        hparams.update(hparams_)
        setproctitle.setproctitle(f'data_processor:{hparams["exp_name"]}:controller_fn')
        print(f"| init controller, {ds_len =}, ")
        try:
            g = torch.Generator()  # 随机数生成器
            g.manual_seed(seed)
            indices = torch.randperm(ds_len, generator=g).tolist()
            if self.node_id is not None:
                indices = indices[self.node_id::self.node_size]
            pull_i = 0
            epoch = 0
            while max_epoch <= 0 or epoch < max_epoch:
                while not q_to_pull.full():
                    if pull_i == len(indices):
                        epoch += 1
                        indices = torch.randperm(ds_len, generator=g).tolist()
                        pull_i = 0
                        break
                    q_to_pull.put(indices[pull_i])
                    pull_i += 1
                if DEBUG:
                    print("controller: q_to_pull满了, 休息1s等等processor")
                time.sleep(1)
            for i in range(n_processor * 2):
                q_to_pull.put(None)
            print("| Controller worker finished...")
        except:
            traceback.print_exc()

    def get_binary_reader(self, data_paths, reader_chunk_size, worker_id=0, worker_world_size=1, reader_cache_name='cache'):
        fd_cache_size = 128
        io_thread_num = 64
        io_retry = 5
        reader = FalconReader(data_paths, fd_cache_size, io_thread_num, io_retry, 
                            reader_cache_name, worker_world_size, worker_id, reader_chunk_size)
        ds_len = reader.get_entry_num(list(range(len(data_paths))), False)
        return reader, ds_len

    def get_manifest_reader(self, data_paths, reader_chunk_size, worker_id=0, worker_world_size=1, reader_cache_name='cache'):
        return {}
    
    def get_dataset_meta(self):
        cluster = os.environ.get('CLUSTER', '').lower()
        dataset_meta = {
            'datasets': []
        }
        total_ds_len = 0
        idx_offset = 0
        reader_chunk_size = hparams.get('reader_chunk_size', 64)
        if not hasattr(self, 'hdfs_clients'):
            self.hdfs_clients = {}
        print(f'| training datasets:')

        with tempfile.TemporaryDirectory(dir='/dev/shm') as temp_dir:
            # print(f"if datasets in {hparams.get('datasets', None) =}")
            hp_datasets = hparams['datasets']
            if hparams.get('is_debug_run', False):
                hp_datasets = hparams.get('debug_run_dataset', hp_datasets)
                print_once('| Use debug_run_dataset')
                
            for dataset_group_name in hp_datasets:

                print(f'| - dataset group [{dataset_group_name}]')
                dataset_group = hp_datasets[dataset_group_name]
                dataset_processer_fn = import_module_bystr(dataset_group['processer_fn'])

                if dataset_group.get('binary_data_root'):
                    if f"{cluster}_hdfs" in dataset_group['binary_data_root']:
                        binary_data_root = dataset_group['binary_data_root'][f"{cluster}_hdfs"]
                    elif f"{cluster}_nas" in dataset_group['binary_data_root']:
                        binary_data_root = dataset_group['binary_data_root'][f"{cluster}_nas"]
                    else:
                        binary_data_root = dataset_group['binary_data_root']["default"]
                else:
                    binary_data_root = ''
                
                for rel_path in dataset_group['train_sets']:
                    if rel_path.endswith('.json'):
                        manifest_path = os.path.join(binary_data_root, rel_path)
                        manifest = json.load(open(safe_read_path(manifest_path, os.path.join(temp_dir, rel_path), self.hdfs_clients)))
                        ds_len = len(manifest)
                        dataset_meta_ = {
                            'data_path': manifest_path,
                            'manifest': manifest,
                            'reader_type': 'manifest'
                        }
                    elif rel_path.endswith('.jsonl'):
                        manifest_path = os.path.join(binary_data_root, rel_path)
                        read_idx = dataset_group.get('read_idx', False)
                        idx_path = manifest_path + '.idx'

                        # 非 hdfs 并且希望用 idx（read_idx=True）
                        if not manifest_path.startswith('hdfs://') and read_idx:
                            # 如果 idx 不存在，就先在线构建 idx，而不是加载整份 jsonl
                            if not os.path.isfile(idx_path):
                                print(f'| build jsonl idx: {idx_path}')
                                build_jsonl_idx_with_progress(manifest_path, idx_path)

                            ds_len = count_jsonl_n_lines(idx_path)
                            dataset_meta_ = {
                                'data_path': manifest_path,
                                'reader_type': 'jsonl_idx',
                            }
                        else:
                            # hdfs 上的 jsonl 或者 read_idx=False，保持原来的 manifest 加载逻辑
                            manifest = load_samples_from_jsonl(
                                safe_read_path(manifest_path, os.path.join(temp_dir, rel_path), self.hdfs_clients)
                            )
                            ds_len = len(manifest)
                            dataset_meta_ = {
                                'data_path': manifest_path,
                                'manifest': manifest,
                                'reader_type': 'manifest'
                            }

                    elif rel_path.endswith('.tsv'):
                        manifest_path = os.path.join(binary_data_root, rel_path)
                        manifest = load_samples_from_tsv(safe_read_path(manifest_path, os.path.join(temp_dir, rel_path), self.hdfs_clients))
                        ds_len = len(manifest)
                        dataset_meta_ = {
                            'data_path': manifest_path,
                            'manifest': manifest,
                            'reader_type': 'manifest'
                        }
                    elif rel_path.endswith('.parquet') or rel_path.endswith('.pq'):
                        manifest_path = os.path.join(binary_data_root, rel_path)
                        reader_chunk_size_ = dataset_group.get('reader_chunk_size', reader_chunk_size)
                        reader = ParquetChunkReader(manifest_path, reader_chunk_size_)
                        ds_len = len(reader)
                        dataset_meta_ = {
                            'data_path': manifest_path,
                            'reader_type': 'pq_reader'
                        }
                    else:
                        binary_data_path = os.path.join(binary_data_root, rel_path, 'data')
                        reader_chunk_size_ = dataset_group.get('reader_chunk_size', reader_chunk_size)
                        _, ds_len = self.get_binary_reader([binary_data_path], reader_chunk_size_)
                        n_chunks = math.ceil(ds_len / reader_chunk_size_)
                        dataset_meta_ = {
                            'data_path': binary_data_path,
                            'reader_type': 'binary'
                        }
                        
                    reader_chunk_size_ = dataset_group.get('reader_chunk_size', reader_chunk_size)
                    n_chunks = math.ceil(ds_len / reader_chunk_size_)
                    dataset_meta_.update({
                        'ds_len': ds_len,
                        'n_chunks': math.ceil(ds_len / reader_chunk_size_),
                        'offset': idx_offset,
                        'processer_fn': dataset_processer_fn,
                        'reader_chunk_size': reader_chunk_size_
                    })

                    dataset_meta['datasets'].append(dataset_meta_)
                    total_ds_len += ds_len
                    idx_offset += n_chunks
                    print(f'|   - {dataset_meta_["data_path"]}')
                    print(f'|     - length: {ds_len}')
                    print(f'|     - reader_chunk_size: {reader_chunk_size_}')
                    print(f'|     - n_chunks: {n_chunks}')
        print(f"| Total training data length: {total_ds_len}")
        print(f"| Total num chunks: {idx_offset}")
        return dataset_meta, idx_offset
    
    
    def process_fn(
            self, q_to_pull, q_to_push, world_size,
            shm_base, counter, hparams_, seed, i_worker, n_worker
    ):
        hparams.update(hparams_)
        setproctitle.setproctitle(f'data_processor:{hparams["exp_name"]}:processor_fn#{i_worker}/{n_worker}')
        self.seed = seed
        print(f"| Starting processor_fn_worker#{i_worker}/{n_worker}.")
        # print(f"if hparams has dac_base_path: {hparams.get('dac_base_path', None)}")

        try:
            global_stores = {}
            reader_pack = self.prepare_reader(self.dataset_meta, global_stores, i_worker, n_worker)
            print(f"| init processor (dataset_reader)#{i_worker}/{n_worker}.")
            restart_countdown = 10000
            while True:
                try:
                    idx = q_to_pull.get()
                    if idx is None:
                        return None
                    for item in self.process_item(idx, reader_pack, global_stores, hparams, i_worker, n_worker):
                        if isinstance(item, tuple):
                            item, item_meta = item
                        else:
                            item_meta = ''
                        item = convert_to_np(item)
                        with counter.get_lock():
                            cnt = counter.value
                            counter.value += 1
                        out_path = save_samples_to_shm(item, cnt, shm_base, item_meta)
                        if DEBUG:
                            print(f"processor#{i_worker}/{n_worker}: save to {out_path}")
                        q_to_push.put(out_path)
                        restart_countdown -= 1
                        if restart_countdown == 0:
                            return
                        while q_to_push.qsize() > self.shuffle_buffer * world_size * 2:
                            if DEBUG:
                                print(
                                    f"processor#{i_worker}/{n_worker}: q_to_push里面积压的太多了, 休息1s等等batch_saver")
                            time.sleep(1)
                        self.after_process_item(item, hparams, global_stores)
                except:
                    traceback.print_exc()
                    continue
        except:
            traceback.print_exc()
    
    def prepare_reader(self, dataset_meta, global_stores, i_worker, n_worker):
        reader_pack = []
        for dataset_meta_ in dataset_meta['datasets']:
            reader_pack_ = {
                'ds_len': dataset_meta_['ds_len'],
                'n_chunks': dataset_meta_['n_chunks'],
                'offset': dataset_meta_['offset'],
                'data_path': dataset_meta_['data_path'],
                'processer_fn': dataset_meta_['processer_fn'],
                'reader_chunk_size': dataset_meta_['reader_chunk_size'],
                'reader_type': dataset_meta_['reader_type'],
            }
            if reader_pack_['reader_type'] == 'binary':
                reader_pack_['reader'] = self.get_binary_reader([dataset_meta_['data_path']], dataset_meta_['reader_chunk_size'])[0]
            elif reader_pack_['reader_type'] == 'jsonl_idx':
                reader_pack_['reader'] = JsonlChunkReader(dataset_meta_['data_path'], dataset_meta_['data_path'] + '.idx')
            elif reader_pack_['reader_type'] == 'pq_reader':
                reader_pack_['reader'] = ParquetChunkReader(dataset_meta_['data_path'], dataset_meta_['reader_chunk_size'])
            reader_pack.append(reader_pack_)
        return reader_pack
    
    def read_fn(self, idx, reader_pack, global_stores):
        if idx >= reader_pack[-1]['offset']:
            reader = reader_pack[-1]
            reader_idx = len(reader_pack)
        else:
            reader_idx = 0
            while idx >= reader_pack[reader_idx]['offset']:
                reader_idx += 1
            reader = reader_pack[reader_idx - 1]
        try:
            idx = (idx - reader['offset']) * reader['reader_chunk_size']
            if reader['reader_type'] == 'binary':
                items = [pickle.loads(x) for x in reader['reader'].read_many([idx])[0]]
            elif reader['reader_type'] == 'manifest':
                items = self.dataset_meta['datasets'][reader_idx - 1]['manifest'][idx: idx + reader['reader_chunk_size']]
            elif reader['reader_type'] == 'jsonl_idx':
                items = reader['reader'].read_range(idx, min(reader['ds_len']-1, idx + reader['reader_chunk_size'] - 1))
            elif reader['reader_type'] == 'pq_reader':
                items = reader['reader'].read_chunk(idx // reader['reader_chunk_size'])
            return items, reader['processer_fn']
        except:
            return
        
    def get_batcher(self, hparams, global_stores):
        batcher = get_from_global_stores(
            'batcher', global_stores,
            lambda: BucketBatcher(
                buckets=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 
                            600, 650, 700, 750, 800, 850, 900, 950, 1000, 1200, 1400, 
                            1600, 1800, 2000, 2400, 2800, 3000, 4000],
                dynamic_batch=hparams.get("dynamic_batch", True),
                batch_size=hparams['max_sentences'],
                maximum_bucket_size=hparams['max_tokens'],
                length_fn=lambda x: x['len'],
            )
        )
        return batcher
    
    def process_item(self, index, reader_pack, global_stores, hparams, i_worker, n_worker):
        
        if DEBUG:
            print(f'processer {i_worker}/{n_worker}: {index = }')
        
        def init_new_batch():
            tgt_size = random.randint(hparams['tgt_size_min'], hparams['tgt_size_max'])
            return tgt_size
        
        read_res = self.read_fn(index, reader_pack, global_stores)
        if read_res is None:
            return
        raw_item, processer_fn = read_res
        
        if self.use_fast_dataloader:
            batcher = self.get_batcher(hparams, global_stores)
            tgt_size = init_new_batch()
        
        for item in self._process_item(processer_fn, raw_item, tgt_size, hparams, global_stores, i_worker, n_worker):
            if item is None:
                continue
            if self.use_fast_dataloader:
                batch = batcher.collate_batch(item)
                if batch is not None and len(batch) > 0:
                    # print(f"{len(batch) = } {batch[0]['inputs_embeds'].shape = } {tgt_size = }")
                    tgt_size = init_new_batch()
                    yield batch
                else:
                    if DEBUG:
                        print('batch is None or len(batch) == 0')
            else:
                yield [item]
            
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
        # 修改：统一的 skip 分类
        skip_logger: SkipLogger = get_from_global_stores(
            'skip_logger', global_stores,
            lambda: SkipLogger([
                'len_out_of_range',
                'text_invalid',
                'ph_token_too_long',
                'align_mismatch',
                'feature_build_fail',
                'dur_mismatch',
                'processer_exception',
            ], interval=1000, i_worker=i_worker, n_worker=n_worker)
        )
        items = processer_fn(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker)
        if items is None:
            if DEBUG:
                print(f'processer {i_worker}/{n_worker}: items is None')
            return
        # ------------------
        # merge same spk ...
        # ------------------
        for item_tgt in items:
            # 长度范围检查
            if not (hparams['max_frames'] >= item_tgt['wav_len'] // hop_size > hparams['min_frames']):
                skip_logger.report(1, 'len_out_of_range')
                skip_logger.update(1)
                if DEBUG:
                    print('[skip] len_out_of_range',
                          item_tgt.get('item_name', ''),
                          f"frames={item_tgt['wav_len'] // hop_size}, "
                          f"min={hparams['min_frames']}, max={hparams['max_frames']}")
                continue
            # 文本处理
            txt = raw_text_process(item_tgt['txt'], wav_len=item_tgt['wav_len'])
            if txt is None:
                skip_logger.report(1, 'text_invalid')
                skip_logger.update(1)
                if DEBUG:
                    print('[skip] text_invalid', item_tgt.get('item_name', ''))
                continue
            item_tgt['text'] = txt
            # phone/token 长度检查
            item_tgt['ph_token'] = item_tgt['phone']
            if item_tgt['ph_token'].shape[0] >= item_tgt['wav_len'] // hop_size // 4:
                skip_logger.report(1, 'ph_token_too_long')
                skip_logger.update(1)
                if DEBUG:
                    print('[skip] ph_token_too_long',
                          item_tgt.get('item_name', ''),
                          f"ph_len={item_tgt['ph_token'].shape[0]}",
                          f"mel_len={item_tgt['wav_len'] // hop_size}")
                continue
            # 裁剪与增强
            if hparams.get('load_wav', True):
                item_tgt['wav'] = item_tgt['wav'][:len(item_tgt['wav']) // fm_wav * fm_wav]
                if speech_augmentor is not None:
                    item_tgt['wav'] = speech_augmentor(item_tgt['wav'], sr)
            mel_len = len(item_tgt['wav']) // hop_size
            # mel2ph 补齐（如果 mel_len 更长）
            if mel_len > len(item_tgt['mel2ph']):
                mel2ph = torch.zeros(mel_len).long()
                mel2ph[:len(item_tgt['mel2ph'])] = item_tgt['mel2ph']
                mel2ph[len(item_tgt['mel2ph']):] = mel2ph[len(item_tgt['mel2ph'])-1]
                item_tgt['mel2ph'] = mel2ph
            # 保证 mel2ph 能被 fm 整除
            item_tgt['mel2ph'] = item_tgt['mel2ph'][:len(item_tgt['mel2ph']) // fm * fm]
            if 'dur' not in item_tgt:
                item_tgt['dur'] = mel2token_to_dur(item_tgt['mel2ph'])
            # mel2ph 与 mel 长度严格一致
            if hparams.get('load_wav', True) and len(item_tgt['mel2ph']) != len(item_tgt['wav']) // hop_size:
                skip_logger.report(1, 'align_mismatch')
                skip_logger.update(1)
                if DEBUG:
                    print('[skip] align_mismatch',
                          item_tgt.get('item_name', ''),
                          f"mel2ph_len={len(item_tgt['mel2ph'])}",
                          f"mel_len={len(item_tgt['wav']) // hop_size}")
                continue
            # 可选特征构建：ph_timestamp
            if hparams.get('use_ph_timestamp', False):
                try:
                    item_tgt['ph_timestamp'] = FrontendLMDataset.get_ph_timestamp(item_tgt)
                except:
                    skip_logger.report(1, 'feature_build_fail')
                    skip_logger.update(1)
                    if DEBUG:
                        print('[skip] feature_build_fail(ph_timestamp)', item_tgt.get('item_name', ''))
                    continue
            # 可选特征构建：merged_ph_token
            if hparams.get('use_merged_ph', False):
                try:
                    item_tgt['merged_ph_token'] = map_phone_to_tokendict({
                        'phone': item_tgt['phone'], 'tone': item_tgt['tone']
                    }, pad_bos_eos=False)
                except:
                    skip_logger.report(1, 'feature_build_fail')
                    skip_logger.update(1)
                    if DEBUG:
                        print('[skip] feature_build_fail(merged_ph_token)', item_tgt.get('item_name', ''))
                    continue
            if hparams.get('use_merged_ph', False) and 'dur' in hparams['task_cls']:
                if item_tgt['merged_ph_token'].shape[0] != item_tgt['dur'].shape[0]:
                    skip_logger.report(1, 'dur_mismatch')
                    skip_logger.update(1)
                    if DEBUG:
                        print('[skip] dur_mismatch(merged_ph vs dur)',
                              item_tgt.get('item_name', ''),
                              f"merged_ph={item_tgt['merged_ph_token'].shape[0]} dur={item_tgt['dur'].shape[0]}")
                    continue
            if hparams.get('valid_ph_dur', False):
                if item_tgt['phone'].shape[0] != item_tgt['dur'].shape[0]:
                    skip_logger.report(1, 'dur_mismatch')
                    skip_logger.update(1)
                    if DEBUG:
                        print('[skip] dur_mismatch(phone vs dur)',
                              item_tgt.get('item_name', ''),
                              f"phone={item_tgt['phone'].shape[0]} dur={item_tgt['dur'].shape[0]}")
                    continue
            # 稀疏 dur（按需）
            if hparams.get('use_sparse_dur', False):
                mel2ph_sparse = compute_mel2aug_from_dur(
                    item_tgt['dur'].numpy().tolist(),
                    gap_mode=hparams.get('sparse_dur_mode', 'proportional'),
                    gap_frames=hparams.get('sparse_dur_frames', 4),
                    gap_alpha=hparams.get('sparse_dur_alpha', 0.2),
                    min_keep=hparams.get('sparse_dur_min_keep', 1),
                    keep_ratio=hparams.get('sparse_dur_keep_ratio'),
                    symmetric=hparams.get('sparse_dur_symmetric', True),
                )
                item_tgt['mel2ph_sparse'] = mel2ph_sparse
            # 上下文 mask
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
            item_tgt['ctx_wav'] = item_tgt['ctx_wav'][:rand_length*hparams['hop_size']]
            # VAD
            if hparams.get('add_vad_mask', False):
                from utils.audio.vad import run_vad_trim
                vad_start, vad_end = run_vad_trim(item_tgt['wav'], hparams['audio_sample_rate'], vad_model)
                vm = hparams['hop_size'] * hparams['vae_stride']
                vad_mask = np.zeros((item_tgt['wav'].shape[0] // vm))
                vad_mask[int(vad_start * hparams['audio_sample_rate'] // vm): int(vad_end * hparams['audio_sample_rate'] // vm)] = 1
                item_tgt['vad_mask'] = vad_mask
            else:
                item_tgt['vad_mask'] = None
            item_tgt['len'] = mel_len // 4
            yield item_tgt
            skip_logger.step(1)
            
    def collater(self, samples):
        if len(samples) == 1 and isinstance(samples[0], list):
            samples = samples[0]
        if len(samples) == 0:
            if hasattr(self, 'backup_batch') and self.backup_batch is not None:
                print('use backup batch!')
                return self.backup_batch
            else:
                print('no batch to take!')
                return {}
        wavs = collate_xd([s['wav'] for s in samples], 0.0) if 'wav' in samples[0] and samples[0]['wav'] is not None else None
        wav_lengths = torch.LongTensor([s['wav'].shape[0] for s in samples]) if wavs is not None else None
        ctx_wavs = collate_xd([s['ctx_wav'] for s in samples], 0.0) if 'ctx_wav' in samples[0] and samples[0]['ctx_wav'] is not None else None
        if 'vad_mask' in samples[0] and samples[0]['vad_mask'] is not None:
            vad_mask = collate_xd([s['vad_mask'] for s in samples], 0.0)[..., None]
        else:
            vad_mask = None
        batch = {
            'nsamples': len(samples),
            # 'wavs': wavs,
            # 'wav_lengths': wav_lengths,
            # 'ctx_wavs': ctx_wavs,
            # 'vad_mask': vad_mask
        }
        if valid_item_kv(samples[0], 'mel'):
            batch['mels'] = collate_xd([s['mel'] for s in samples], -6.0)
        if 'mel2ph' in samples[0]:
            batch['mel2ph'] = collate_xd([s['mel2ph'] for s in samples], 0)
        if 'dur' in samples[0]:
            batch['dur'] = collate_xd([s['dur'] for s in samples], 0)
            batch['dur_len'] = torch.LongTensor([s['dur'].shape[0] for s in samples])
        if 'mel2ph_sparse' in samples[0]:
            batch['mel2ph_sparse'] = collate_xd([s['mel2ph_sparse'] for s in samples], 0)
        if valid_item_kv(samples[0], 'ctx_mask'):
            batch['ctx_mask'] = collate_xd([s['ctx_mask'] for s in samples], 0)
        if 'text' in samples[0]:
            batch['text'] = [s['text'] for s in samples]
        if 'caption' in samples[0]:
            batch['caption'] = [s['caption'] for s in samples]
        if 'caption_audio' in samples[0]:
            batch['caption_audio'] = [s['caption_audio'] for s in samples]
        if 'ph_token' in samples[0]:
            batch['ph_tokens'] = collate_xd([s['ph_token'] for s in samples], 0)
            batch['txt_lengths'] = torch.LongTensor([s['ph_token'].numel() for s in samples])
        if 'tone' in samples[0]:
            batch['tone'] = collate_xd([s['tone'] for s in samples], 0)
        if 'ph_timestamp' in samples[0]:
            batch['ph_timestamp'] = collate_xd([s['ph_timestamp'] for s in samples], 797)
            batch['ph_timestamp_len'] = torch.LongTensor([s['ph_timestamp'].shape[0] for s in samples])
        if 'merged_ph_token' in samples[0]:
            batch['merged_ph_tokens'] = collate_xd([s['merged_ph_token'] for s in samples], 797)
            batch['merged_ph_tokens_len'] = torch.LongTensor([s['merged_ph_token'].shape[0] for s in samples])
        if 'ph_dur_seq' in samples[0]:
            batch['ph_dur_seqs'] = collate_xd([s['ph_dur_seq'] for s in samples], 797)
            batch['ph_dur_seqs_len'] = torch.LongTensor([s['ph_dur_seqs'].shape[0] for s in samples])
            batch['ph_dur_seq_dur_mask'] = collate_xd([s['ph_dur_seq_dur_mask'] for s in samples], 0)
        if 'spk_mask' in samples[0]:
            batch['spk_mask'] = collate_xd([s['spk_mask'] for s in samples], 0)
        if 'audio_mask' in samples[0]:
            batch['audio_mask'] = collate_xd([s['audio_mask'] for s in samples], 0)
            
        # print(f"{type(samples) =}, {type(samples[0]) =}, {samples[0].keys() =}, {samples[0]['global_clip_embedding'].shape =}")        
        if 'inputs_embeds' in samples[0]:
            batch['inputs_embeds'] = collate_xd([s['inputs_embeds'] for s in samples])
        if 'global_clip_embedding' in samples[0]:
            batch['global_clip_embedding'] = collate_xd([s['global_clip_embedding'] for s in samples])
        if 'video_feats' in samples[0]: ## 和inputs_embeds一样
            batch['video_feats'] = collate_xd([s['video_feats'] for s in samples])
        if 'v_mask' in samples[0]:
            batch['v_mask'] = collate_xd([s['v_mask'] for s in samples], 0)
        # if 'decoder_input_ids' in samples[0]:
        #     batch['decoder_input_ids'] = collate_xd([s['decoder_input_ids'] for s in samples])
        # if 'labels' in samples[0]:
        #     batch['labels'] = collate_xd([s['labels'] for s in samples])
        if 'direction' in samples[0]:
            batch['direction'] = collate_xd([s['direction'] for s in samples])
        if 'energy_map' in samples[0]:
            batch['energy_map'] = collate_xd([s['energy_map'] for s in samples])
        if 'lat' in samples[0]:
            batch['lat'] = collate_xd([s['lat'] for s in samples])
        if 'lat_lens' in samples[0]:
            batch['lat_lens'] = torch.tensor([s['lat_lens'] for s in samples])
        if 'file_name' in samples[0]:
            batch['file_name'] = [s['file_name'] for s in samples]
        if 'vid_mae' in samples[0]:
            batch['vid_mae'] = collate_xd([s['vid_mae'] for s in samples])
        if 'rot_vid' in samples[0]:
            batch['rot_vid'] = collate_xd([s['rot_vid'] for s in samples])
        if 'foa_mae' in samples[0]:
            batch['foa_mae'] = torch.stack([s['foa_mae'] for s in samples], dim=0)
        if 'time_foa' in samples[0]:
            batch['time_foa'] = torch.stack([s['time_foa'] for s in samples], dim=0)
        if 'rot_foa' in samples[0]:
            batch['rot_foa'] = torch.stack([s['rot_foa'] for s in samples], dim=0)
        
            
            
        
        if not hasattr(self, 'backup_batch') or self.backup_batch is None or random.random() < 0.001:
            self.backup_batch = batch
        # print(f"in collator: {batch.keys()}, {samples =}")
        return batch
    
    

def processer_fn_megatts3(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    items = []
    for item_ in raw_item:
        try:
            item = {}
            if hparams.get('load_wav', True):
                item['wav'] = torch.FloatTensor(item_['wav'])
                item['wav_len'] = item['wav'].shape[0]
            else:
                item['wav_len'] = int(float(item_['sec']) * hparams['audio_sample_rate'])
            item['phone'] = torch.LongTensor(item_['phone_encoded'])
            item['tone'] = torch.LongTensor(item_['tone_encoded'])
            item['mel2ph'] = torch.LongTensor(item_['mel2ph'])
            item['item_name'] = item_['item_name']
            item['txt'] = item_['txt_raw']
            item['spk_name'] = item_['spk_name']
            items.append(item)
        except Exception:
            skip_logger.report(1, 'processer_exception')
            skip_logger.update(1)
            if DEBUG:
                print('[skip@megatts3] processer_exception', item_.get('item_name', ''))
            continue
    return items

def processer_fn_zyxc_1spk(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    tos_client: TosClient = get_from_global_stores(
        'tos_client', global_stores,
        lambda: TosClient(bucket='humanaigc-ads')
    )
    length_regulator = get_from_global_stores(
        'length_regulator', global_stores,
        lambda: LengthRegulator()
    )
    sr = hparams['audio_sample_rate']
    with tempfile.TemporaryDirectory(dir='/dev/shm') as temp_dir:
        items = []
        for item_ in raw_item:
            try:
                item_name = item_['item_name']
                feat_k = item_['feat_k']
                vocal_k = item_['vocal_k']
                subset = ['subset']  # 保持原样
                if hparams.get('load_wav', True):
                    try:
                        data = tos_client.get_object(vocal_k)
                        global_wav_path = os.path.join(temp_dir, f'global.m4a')
                        with open(global_wav_path, 'wb') as f:
                            f.write(data)
                        global_wav, sr_ = torchaudio.load(global_wav_path)
                        global_wav = global_wav.mean(dim=0).numpy()
                    except Exception:
                        skip_logger.report(1, 'processer_exception')
                        skip_logger.update(1)
                        if DEBUG:
                            print('[skip@zyxc_1spk] processer_exception(load_global_wav)', item_name, vocal_k)
                        continue
                    if len(global_wav) == 0:
                        skip_logger.report(1, 'len_out_of_range')
                        skip_logger.update(1)
                        if DEBUG:
                            print('[skip@zyxc_1spk] len_out_of_range(global_wav empty)', item_name, vocal_k)
                        continue
                for segment_idx, segment_meta in enumerate(item_['segments_1spk']):
                    item = {}
                    if hparams.get('load_wav', True):
                        wav_start, wav_end = segment_meta['start'], segment_meta['end']
                        wav = global_wav[int(wav_start * sr_): int(wav_end * sr_)]
                        if len(wav) == 0:
                            skip_logger.report(1, 'len_out_of_range')
                            skip_logger.update(1)
                            if DEBUG:
                                print('[skip@zyxc_1spk] len_out_of_range(empty segment)',
                                      item_name, f'seg#{segment_idx}', f'start={wav_start}', f'end={wav_end}')
                            continue
                        if sr_ != sr:
                            wav = librosa.resample(wav, orig_sr=sr_, target_sr=sr)
                        item['wav'] = torch.FloatTensor(wav)
                        item['wav_len'] = wav.shape[0]
                    else:
                        item['wav_len'] = int(segment_meta['sec'] * sr)
                    item['item_name'] = item_name + '#' + f'{segment_idx}'
                    if segment_meta.get('phone_encoded') is None:
                        skip_logger.report(1, 'feature_build_fail')
                        skip_logger.update(1)
                        if DEBUG:
                            print('[skip@zyxc_1spk] feature_build_fail(missing phone)',
                                  item['item_name'])
                        continue
                    item['phone'] = torch.LongTensor(segment_meta['phone_encoded'])
                    item['tone'] = torch.LongTensor(segment_meta['tone_encoded'])
                    item['dur'] = torch.LongTensor(segment_meta['dur'])
                    item['mel2ph'] = length_regulator(item['dur'][None, :])[0]
                    item['txt'] = segment_meta['txt_raw']
                    item['spk_name'] = item_name + '#' + segment_meta['spk_name']
                    items.append(item)
            except Exception:
                traceback.print_exc()
                skip_logger.report(1, 'processer_exception')
                skip_logger.update(1)
                if DEBUG:
                    print('[skip@zyxc_1spk] processer_exception(outer)', item_.get('item_name', ''))
                continue
    return items


if __name__ == '__main__':
    from utils.commons.hparams import set_hparams, hparams
    set_hparams(
        'egs/tts/megatts3_dit_v2_dataloader_v2.yaml', 
        print_hparams=False, 
    )
    exp_name = 'test_DiTT2ADataset'
    hparams.update(dict(
        exp_name=exp_name,
        sp_size=1,
        ds_workers=8,
        debug=True,
        fast_ds_shuffle_buffer=32,
        max_sentences=5,
        max_tokens=2000,
        frames_multiple=8
    ))

    ds_train = BaseTTSShmDataset('train', hparams, use_fast_dataloader=True, rank_id=0, world_size=1, batch_size=1)
    dl_train = ds_train.get_dataloader(seed=1234, num_workers=hparams['ds_workers'])
    for i, items in enumerate(dl_train):
        if 'ph_tokens' in items:
            print(items)
            break
        
    # tos_client = TosClient(bucket='humanaigc-ads')
    # vocal_k = 'tts_datasets/zhiyuexingchen/cn/podcast/apple/2CK5BNKN/apple_podcasts/cn_41/audio_01/1539659953/rssFileVip_89_features/vocal.m4a'
    # print(f"{tos_client.check_tos_file_exists(vocal_k) = }")
    
    # length_regulator = LengthRegulator()
    # dur = torch.LongTensor([ 26,  12,   9,  12,  30,  20,  35,  81,   0,  17,   6,  12,   0,  10,
    #       7,  11,   5,  39,  25,  16,  16,  15,  10,  15,   2,  17,  65,   0,
    #      63, 201,  19,   9,  12,  41,  20,  10,   6,  59,  67,   4,   8,  11,
    #      13,  13,   4,  13,  12,   9,   7,   4,  16,   7,   5,   7,  11,   6,
    #      13,  11, 114,   0,  76,  62,  10,  11,   4,   8,  27,  26,  22,  21,
    #      10,   6,  13,  10,   9,  14,  12,   4,   4,  15,   6,  15,   7, 124,
    #       8,  13,  13,  28,   9,   8,  11,   6,  16,  11,   6,  17,  19,  11,
    #      11,   6,   7,  16,   5,   6,  10,  14,   4,  10,   6,   8,   7,  14,
    #       6,  18,   8,  52])
    # mel2ph = length_regulator(dur[None, :])
    # print(mel2ph)