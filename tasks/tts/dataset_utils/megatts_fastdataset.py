import uuid
import io
import utils.commons.single_thread_env  # NOQA
# from pyarrow import hdfs
from copy import deepcopy
from pathlib import Path
import re
import math
import gc
import json
import librosa
from tqdm import tqdm
import glob
import multiprocessing
import os
import pickle
import random
import subprocess
import time
import traceback
from multiprocessing import Process, Queue
import numpy as np
from PIL import Image
# import cv2
import torch
import torchaudio
from einops import rearrange
from transformers import AutoTokenizer, WhisperFeatureExtractor
# from dataloader import FalconReader, KVReader
from tasks.tts.dataset_utils.tts_datasets import MegaTTSDataset, FrontendLMDataset, G2PLMDataset, \
        SDVAEDataset, DiTDataset, DiTWavDataset, Latent2WavDataset, DiTWavImgDataset, CodecLMWavDataset
from utils.dataset.batcher import BucketBatcher
from utils.commons.dataset_utils import pad_or_cut_xd
from utils.commons.hparams import set_hparams, hparams
import setproctitle
from utils.plot.plot import spec_to_figure
from utils.text.text_encoder import TokenTextEncoder
from utils.text.ph_tone_convert import map_phone_to_tokendict
from utils.audio import librosa_wav2spec
from utils.audio.align import mel2token_to_dur
# import modules.tts.image_encoder.processor as img_processor
from modules.tts.image_encoder.processor import ImageProcessor
from utils.text import is_chinese, is_english
from utils.text.split_text import get_word_list, remove_spaces_between_chinese
from utils.commons.io import print_once
from modules.tts.ar_dur.commons.align_ops import compute_mel2aug_from_dur
from collections import OrderedDict, defaultdict, Counter

DEBUG = False

fd_cache_size = 128
io_thread_num = 128
io_retry = 5


def controller_fn(ds_len, seed, q_to_pull, reader_chunk_size, max_epoch=0, n_processer=0):
    setproctitle.setproctitle('data_processer:controller_fn')
    try:
        g = torch.Generator()
        g.manual_seed(seed)
        indices = torch.randperm(ds_len // reader_chunk_size + 1, generator=g).tolist()
        pull_i = 0
        epoch = 0
        while max_epoch <= 0 or epoch < max_epoch:
            while not q_to_pull.full():
                if pull_i == len(indices):
                    epoch += 1
                    indices = torch.randperm(ds_len // reader_chunk_size + 1, generator=g).tolist()
                    pull_i = 0
                    break
                q_to_pull.put(indices[pull_i] * reader_chunk_size)
                pull_i += 1
            time.sleep(1)
        for i in range(n_processer * 2):
            q_to_pull.put(None)
        print("| Controller worker finished...")
    except:
        traceback.print_exc()

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

def valid_item_kv(item, k):
    return k in item and item[k] is not None

def merge_item_bytes(items_bytes, load_wav=True, load_mel=True, merge_same_spk=True, exclude_spk=None, merge_samples=None, tgt_size=None,  merge_multi_spk=False):
    merge_samples = MegaTTSDataset.merge_samples if merge_samples is None else merge_samples
    items = []
    hdfs_client = None
    for item_ in items_bytes:
        item_ = pickle.loads(item_)
        item = {}
        if ('wav' not in item_ or item_['wav'] is None) and load_wav:
            data_url = item_['data_url']
            if data_url.startswith('hdfs://'):
                if hdfs_client is None:
                    from utils.commons.hdfs_utils import HDFSClient
                    hdfs_client = HDFSClient()
                data = pickle.loads(hdfs_client.get_object(data_url))
                item_['wav'] = data['wav']
        try:
            item['txt_token'] = torch.LongTensor(item_['phone_encoded'])
        except:
            continue
        if load_wav:
            if load_mel:
                try:
                    mel, wav = MegaTTSDataset.get_mel(hparams, item_['wav'])
                    item['mel'] = torch.FloatTensor(mel)
                    item['wav'] = torch.FloatTensor(wav)
                except:
                    continue
                if len(item['mel']) < hparams['min_frames']:
                    continue
                ph_div_mel = len(item['txt_token']) / len(item['mel'])
                if ph_div_mel > 0.3 or ph_div_mel < 0.01:
                    continue
            else:
                item['wav'] = torch.FloatTensor(item_['wav'])
            wav_len = item['wav'].shape[0]
        else:
            wav_len = float(item_['sec']) * item_.get('sr', 24000)
        item['wav_len'] = wav_len
        item['tone'] = torch.LongTensor(item_['tone_encoded'])
        item['mel2ph'] = torch.LongTensor(item_['mel2ph'])
        item['ph2char'] = torch.LongTensor(item_['ph2char']) if valid_item_kv(item_, 'ph2char') else None
        item['char_token'] = torch.LongTensor(item_['char_encoded']) if valid_item_kv(item_, 'char_encoded') else None
        if item_['subset'] == 'zh_podcast':
            session_name = '/'.join(item_['item_name'].split('/')[:-1])
            item['spk_name'] = f"{session_name}/{item_['spk_name']}"
        else:
            item['spk_name'] = item_['spk_name']
        item['item_name'] = item_['item_name']
        item['txt'] = item_['txt_raw']
        if exclude_spk is not None and item['spk_name'] in exclude_spk:
            continue
        items.append(item)
    if not merge_same_spk:
        return items
    items_merged = []
    last_spk = ''
    total_frames = 0
    items_to_merge = []
    for item in items:
        # print('item length', item['wav'].shape[0] // hparams['hop_size'], 'total_frames', total_frames)
        wav_len = item['wav_len']
        if len(items_to_merge) > 0:
            if (
                    ((not merge_multi_spk) and item['spk_name'] != last_spk) or 
                    (tgt_size is not None and total_frames > 0 and total_frames + wav_len // hparams['hop_size'] > tgt_size)
                ):
                items_merged.append(merge_samples(items_to_merge))
                items_to_merge = []
                total_frames = 0
        items_to_merge.append(item)
        last_spk = item['spk_name']
        total_frames += wav_len // hparams['hop_size']
    if len(items_to_merge) > 0:
        items_merged.append(merge_samples(items_to_merge))
    return items_merged


def save_samples_to_shm(samples, cnt, shm_base):
    data_path = f'{shm_base}/{cnt}.pkl'
    with open(f'{data_path}.tmp', 'wb') as f:
        pickle.dump(samples, f)
    subprocess.check_call(f'mv {data_path}.tmp {data_path}', shell=True)


def get_reader(data_paths, reader_chunk_size, worker_id=0, worker_world_size=1, reader_cache_name='cache', hparams_=None):
    if hparams_ is None:
        hparams_ = hparams
    if hparams_.get('use_falcon', True):
        paths = []
        for data_path in data_paths:
            # if hparams_.get('use_hdfs', True):
            #     cmd = f"hdfs dfs -ls {data_path}* | grep index$ | wc -l"
            # else:
            #     cmd = f"ls {data_path}* | grep index$ | wc -l"
            # num_shard = int(subprocess.check_output(cmd, shell=True).decode().strip())
            # if num_shard > 1:
            #     paths += ['{}{}'.format(data_path, shard) for shard in range(num_shard)]
            # else:
            #     paths += [data_path]
            paths += [data_path]    # save time

        reader = FalconReader(paths,
                              fd_cache_size, io_thread_num, io_retry, reader_cache_name, worker_world_size,
                              worker_id, reader_chunk_size)
        ds_len = reader.get_entry_num(list(range(len(paths))), False)
        return reader, ds_len
    else:
        reader = [KVReader(x) for x in data_paths]  # TODO
        ds_len = [len(x.list_keys()) for x in reader]
        return reader, ds_len


def read_items(q_to_pull, reader):
    reader, _ = reader
    item_id_start = q_to_pull.get()
    if item_id_start is None:
        return None
    if hparams.get('use_falcon', True):
        items_bytes = [x for x in reader.read_many([item_id_start])[0]]
    else:
        raise NotImplementedError
    return items_bytes


def processer_fn_durlm(data_paths, q_to_pull, reader_chunk_size, world_size, shm_base, counter, hparams_,
                       seed, i_worker, n_worker, reader_cache_name='cache'):
    setproctitle.setproctitle('data_processer:processer_fn')
    hparams.update(hparams_)

    try:
        reader = get_reader(data_paths, reader_chunk_size, i_worker,
                            n_worker, reader_cache_name)

        def init_new_samples():
            while len(glob.glob(f'{shm_base}/*.pkl')) >= hparams.get('prefetch_steps', 200) * world_size:
                time.sleep(1)
            with counter.get_lock():
                cnt = counter.value
                counter.value += 1
            random.seed((cnt // world_size) % 1001 + seed)
            tgt_size = random.randint(hparams['tgt_size_min'], hparams['tgt_size_max'])
            samples = []
            spk_id = 0
            ntokens = 0
            last_spk = ''
            return samples, tgt_size, cnt, spk_id, ntokens, last_spk

        samples, tgt_size, cnt, spk_id, ntokens, last_spk = init_new_samples()
        restart_countdown = 10000
        while True:
            try:
                items_bytes = read_items(q_to_pull, reader)
            except:
                continue
            if items_bytes is None:
                break
            items = merge_item_bytes(items_bytes, merge_same_spk=False, exclude_spk=hparams.get('exclude_spk'),
                                     load_mel=False)
            for item in items:
                if last_spk != '' and last_spk != item['spk_name']:
                    spk_id += 1
                last_spk = item['spk_name']
                item['spk_id'] = spk_id
                item = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in item.items()}
                samples.append(item)
                ntokens += len(item['txt_token'])
                if ntokens >= hparams['max_tokens']:
                    save_samples_to_shm(samples, cnt, shm_base)
                    restart_countdown -= 1
                    if restart_countdown == 0:
                        return
                    samples, tgt_size, cnt, spk_id, ntokens, last_spk = init_new_samples()
    except:
        traceback.print_exc()


def processer_fn_frontendlm(data_paths, q_to_pull, reader_chunk_size, world_size, shm_base, counter, hparams_,
                       seed, i_worker, n_worker, reader_cache_name='cache'):
    setproctitle.setproctitle('data_processer:processer_fn')
    hparams.update(hparams_)

    import whisper

    try:
        reader = get_reader(data_paths, reader_chunk_size, i_worker,
                            n_worker, reader_cache_name)
        batcher = BucketBatcher(
            buckets=range(100, 5000, 100),
            dynamic_batch=True,
            maximum_bucket_size=hparams.get('max_tokens', 40000),
            length_fn=(lambda x: x['mel'].shape[0]),
            bsz_evaluator=None,
        )
        fm = hparams['frames_multiple']

        def init_new_samples():
            while len(glob.glob(f'{shm_base}/*.pkl')) >= hparams.get('prefetch_steps', 200) * world_size:
                time.sleep(1)
            with counter.get_lock():
                cnt = counter.value
                counter.value += 1
            random.seed((cnt // world_size) % 1001 + seed)
            return cnt

        def merge_samples(samples, fm=None):
            sample_merged = {
                'id': 0,
                'item_name': '|||'.join([s['item_name'] for s in samples]),
                'txt_token': torch.cat([s['txt_token'] for s in samples], 0),
                'char_token': torch.cat([s['char_token'] for s in samples], 0) if valid_item_kv(samples[0], 'char_token') else None,
                'mel': torch.cat([s['mel'] for s in samples], 0),
                'mel2ph': merge_A2B(
                    [s['mel2ph'] for s in samples], [len(s['txt_token']) for s in samples]),
                'ph2char': merge_A2B(
                    [s['ph2char'] for s in samples], [len(s['char_token']) for s in samples]) if valid_item_kv(samples[0], 'ph2char') else None,
                'tone': torch.cat([s['tone'] for s in samples], 0),
                'text': "".join([s['text'] for s in samples]),
            }
            if fm is not None:
                t = sample_merged['mel'].shape[0] // fm * fm
                sample_merged['mel'] = sample_merged['mel'][:t]
                sample_merged['mel2ph'] = sample_merged['mel2ph'][:t]
            return sample_merged

        def merge_A2B(A2B, B_lens):
            token_lens_cumsum = np.cumsum([0] + B_lens[:-1])
            token_lens_cumsum = torch.LongTensor(token_lens_cumsum)
            for i in range(len(B_lens)):
                A2B[i] = A2B[i] + token_lens_cumsum[i]
            A2B = torch.cat(A2B, 0)
            return A2B

        restart_countdown = 10000
        while True:
            try:
                items_bytes = read_items(q_to_pull, reader)
            except:
                continue
            if items_bytes is None:
                break

            items = []
            for item_ in items_bytes:
                item_ = pickle.loads(item_)
                item = {}
                try:
                    item['txt_token'] = torch.LongTensor(item_['phone_encoded'])
                except:
                    continue
                # Load mel
                try:
                    # item['mel'] = torch.FloatTensor(MegaTTSDataset.get_mel(hparams, item_['wav'])[0])
                    whisper_wav = librosa.resample(item_['wav'].astype(np.float32), orig_sr=24000, target_sr=16000)
                    item['mel'] = torch.FloatTensor(whisper.log_mel_spectrogram(whisper_wav).T)
                except:
                    continue
                if len(item['mel']) < hparams['min_frames']:
                    continue
                ph_div_mel = len(item['txt_token']) / len(item['mel'])
                if ph_div_mel > 0.3 or ph_div_mel < 0.01:
                    continue
                item['tone'] = torch.LongTensor(item_['tone_encoded'])
                item['mel2ph'] = torch.LongTensor(item_['mel2ph'])
                item['mel2ph'] = item['mel2ph']
                item['mel'] = item['mel']
                item['ph2char'] = torch.LongTensor(item_['ph2char']) if valid_item_kv(item_, 'ph2char') else None
                item['char_token'] = torch.LongTensor(item_['char_encoded']) if valid_item_kv(item_, 'char_encoded') else None
                item['spk_name'] = item_['spk_name']
                item['item_name'] = item_['item_name']
                item['text'] = item_['txt_raw']
                item['ph_timestamp'] = FrontendLMDataset.get_ph_timestamp(item)
                item['merged_ph'] = FrontendLMDataset.map_phone_to_tokendict(item)
                t = item['mel'].shape[0] // fm * fm
                item['mel'] = item['mel'][:t]
                item['mel2ph'] = item['mel2ph'][:t]

                exclude_spk = hparams.get('exclude_spk')
                if exclude_spk is not None and item['spk_name'] in exclude_spk:
                    continue
                items.append(item)

            if hparams.get('multi_sent_training', False):
                # Merge sentences for multi-sentence training
                items_merged = []
                last_spk = ''
                items_to_merge = []
                items_to_merge_len = 0
                max_item_len = random.randint(hparams['tgt_size_min'], hparams['tgt_size_max'])
                for item in items:
                    if (item['spk_name'] != last_spk and len(items_to_merge) > 0) or items_to_merge_len>=max_item_len:
                        item_tmp = merge_samples(items_to_merge, fm=None)
                        # Load ph_timestamp_seq
                        try:
                            item_tmp['ph_timestamp'] = FrontendLMDataset.get_ph_timestamp(item_tmp)
                        except:
                            print('Error occurs when get ph_timestamp!')
                            continue
                        items_merged.append(item_tmp)
                        items_to_merge = []
                        items_to_merge_len = 0
                        max_item_len = random.randint(hparams['tgt_size_min'], hparams['tgt_size_max'])

                    items_to_merge.append(item)
                    last_spk = item['spk_name']
                    items_to_merge_len += len(item['mel'])

                if len(items_to_merge) > 0:
                    item_tmp = merge_samples(items_to_merge, fm=None)
                    # Load ph_timestamp_seq
                    try:
                        item_tmp['ph_timestamp'] = FrontendLMDataset.get_ph_timestamp(item_tmp)
                    except:
                        print('Error occurs when get ph_timestamp!')
                        continue
                    items_merged.append(item_tmp)
            else:
                items_merged = items
            
            # BucketBatcher
            for item in items_merged:
                """ Save to shm """
                item = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in item.items()}
                batch = batcher.collate_batch(item)
                if batch is not None:
                    cnt = init_new_samples()
                    save_samples_to_shm(batch, cnt, shm_base)
                    restart_countdown -= 1
                    if restart_countdown == 0:
                        return
    except:
        traceback.print_exc()


def processer_fn_g2p(data_paths, q_to_pull, reader_chunk_size, world_size, shm_base, counter, hparams_,
                       seed, i_worker, n_worker, reader_cache_name='cache'):
    setproctitle.setproctitle('data_processer:processer_fn')
    hparams.update(hparams_)

    from transformers import AutoTokenizer         
    tokenizer = AutoTokenizer.from_pretrained("./pretrained/llama_tokenizer", padding_side="right")
    tokenizer.add_tokens(['[ASR_BOS]'], special_tokens=True)
    tokenizer.add_tokens(['[ASR_EOS]'], special_tokens=True)
    tokenizer.add_tokens(['[FULL]'], special_tokens=True)
    tokenizer.add_tokens(['[PARTIAL]'], special_tokens=True)

    try:
        reader = get_reader(data_paths, reader_chunk_size, i_worker,
                            n_worker, reader_cache_name)
        batcher = BucketBatcher(
            buckets=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 
                        1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300,
                        2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 5000],
            dynamic_batch=True,
            maximum_bucket_size=hparams.get('max_tokens', 40000),
            length_fn=(lambda x: x['merged_ph'].shape[0] + x['bpe_token_len']), # mel + phone + bpe
            bsz_evaluator=None,
        )

        def init_new_samples():
            while len(glob.glob(f'{shm_base}/*.pkl')) >= hparams.get('prefetch_steps', 200) * world_size:
                time.sleep(1)
            with counter.get_lock():
                cnt = counter.value
                counter.value += 1
            random.seed((cnt // world_size) % 1001 + seed)
            return cnt

        restart_countdown = 10000
        while True:
            try:
                items_bytes = read_items(q_to_pull, reader)
            except:
                continue
            if items_bytes is None:
                break

            items = []
            for item_ in items_bytes:
                item_ = pickle.loads(item_)
                item = {}
                try:
                    item['txt_token'] = torch.LongTensor(item_['phone_encoded'])
                except:
                    continue

                item['tone'] = torch.LongTensor(item_['tone_encoded'])
                item['spk_name'] = item_['spk_name']
                item['item_name'] = item_['item_name']
                item['text'] = item_['txt_raw']

                item['bpe_token_len'] = tokenizer(item['text'], return_tensors="pt", padding=True)['input_ids'].size(1)
                item['merged_ph'] = FrontendLMDataset.map_phone_to_tokendict(item)

                exclude_spk = hparams.get('exclude_spk')
                if exclude_spk is not None and item['spk_name'] in exclude_spk:
                    continue
                items.append(item)

            # BucketBatcher
            for item in items:
                """ Save to shm """
                item = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in item.items()}
                batch = batcher.collate_batch(item)
                if batch is not None:
                    cnt = init_new_samples()
                    save_samples_to_shm(batch, cnt, shm_base)
                    restart_countdown -= 1
                    if restart_countdown == 0:
                        return
    except:
        traceback.print_exc()


def processer_fn_dit(data_paths, q_to_pull, reader_chunk_size, world_size, shm_base, counter, hparams_,
                       seed, i_worker, n_worker, reader_cache_name='cache'):
    setproctitle.setproctitle('data_processer:processer_fn')
    hparams.update(hparams_)

    if hparams.get('use_multisent_training', False):
        try:
            reader = get_reader(data_paths, reader_chunk_size, i_worker,
                                n_worker, reader_cache_name)
            batcher = BucketBatcher(
                [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 
                         1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300,
                         2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 8000],
                dynamic_batch=True,
                maximum_bucket_size=hparams.get('max_tokens', 40000),
                length_fn=(lambda x: x['mel'].shape[0]//hparams.get('vae_stride', 8) + x['txt_token'].shape[0]), # mel + phone
                bsz_evaluator=None,
            )

            fm = hparams['frames_multiple']

            def init_new_samples():
                while len(glob.glob(f'{shm_base}/*.pkl')) >= hparams.get('prefetch_steps', 200) * world_size:
                    time.sleep(1)
                with counter.get_lock():
                    cnt = counter.value
                    counter.value += 1
                random.seed((cnt // world_size) % 1001 + seed)
                return cnt

            def merge_samples(samples, fm=None):
                sample_merged = {
                    'id': 0,
                    'item_name': '|||'.join([s['item_name'] for s in samples]),
                    'txt_token': torch.cat([s['txt_token'] for s in samples], 0),
                    'char_token': torch.cat([s['char_token'] for s in samples], 0) if valid_item_kv(samples[0], 'char_token') else None,
                    'mel': torch.cat([s['mel'] for s in samples], 0),
                    'mel2ph': merge_A2B(
                        [s['mel2ph'] for s in samples], [len(s['txt_token']) for s in samples]),
                    'ph2char': merge_A2B(
                        [s['ph2char'] for s in samples], [len(s['char_token']) for s in samples]) if valid_item_kv(samples[0], 'ph2char') else None,
                    'tone': torch.cat([s['tone'] for s in samples], 0),
                    'text': "".join([s['text'] for s in samples]),
                }
                sample_merged['text'] = re.sub(r'\s*([.,!?;:])\s*', r'\1 ', sample_merged['text'])
                return sample_merged
                # Get ctx_mask, splited by sentence
                # if len(samples) > 1:
                #     split_idx = random.randint(0, len(samples)-2)
                #     ctx_mask = torch.zeros_like(sample_merged['mel'][:, 0])
                #     ctx_mask[:len(samples[split_idx]['mel'])] = 1.0
                #     ctx_mask = ctx_mask[::8]
                #     sample_merged['ctx_mask'] = ctx_mask[:, None]
                #     return sample_merged
                # else:
                #     return None

            def merge_A2B(A2B, B_lens):
                token_lens_cumsum = np.cumsum([0] + B_lens[:-1])
                token_lens_cumsum = torch.LongTensor(token_lens_cumsum)
                for i in range(len(B_lens)):
                    A2B[i] = A2B[i] + token_lens_cumsum[i]
                A2B = torch.cat(A2B, 0)
                return A2B
            
            restart_countdown = 10000
            while True:
                try:
                    items_bytes = read_items(q_to_pull, reader)
                except:
                    continue
                if items_bytes is None:
                    break
                
                items = []
                for item_ in items_bytes:
                    item_ = pickle.loads(item_)

                    item = {}
                    try:
                        item['txt_token'] = torch.LongTensor(item_['phone_encoded'])
                    except:
                        continue
                    # Load mel
                    try:
                        item['mel'] = torch.FloatTensor(MegaTTSDataset.get_mel(hparams, item_['wav'])[0])
                    except:
                        continue
                    if len(item['mel']) < hparams['min_frames']:
                        continue
                    ph_div_mel = len(item['txt_token']) / len(item['mel'])
                    if ph_div_mel > 0.3 or ph_div_mel < 0.01:
                        continue
                    item['tone'] = torch.LongTensor(item_['tone_encoded'])
                    item['mel2ph'] = torch.LongTensor(item_['mel2ph'])
                    item['mel2ph'] = item['mel2ph']
                    item['mel'] = item['mel']
                    item['ph2char'] = torch.LongTensor(item_['ph2char']) if valid_item_kv(item_, 'ph2char') else None
                    item['char_token'] = torch.LongTensor(item_['char_encoded']) if valid_item_kv(item_, 'char_encoded') else None
                    item['spk_name'] = item_['spk_name']
                    item['item_name'] = item_['item_name']
                    item['text'] = item_['txt_raw']

                    exclude_spk = hparams.get('exclude_spk')
                    if exclude_spk is not None and item['spk_name'] in exclude_spk:
                        continue
                    # Restrict max length
                    max_frames = hparams['max_frames'] // fm * fm
                    item['mel'] = item['mel'][:max_frames]
                    item['mel2ph'] = item['mel2ph'][:max_frames]
                    items.append(item)
                
                # Merge sentences for multi-sentence training
                items_merged = []
                last_spk = ''
                items_to_merge = []
                items_to_merge_len = 0
                max_item_len = random.randint(500, 4000)
                    
                for item in items:
                    if (item['spk_name'] != last_spk and len(items_to_merge) > 0) or items_to_merge_len>=max_item_len:
                        item_tmp = merge_samples(items_to_merge)
                        if item_tmp != None:
                            items_merged.append(item_tmp)
                            items_to_merge = []
                            items_to_merge_len = 0
                            max_item_len = random.randint(400, 4000)
                    items_to_merge.append(item)
                    last_spk = item['spk_name']
                    items_to_merge_len += len(item['mel'])

                if len(items_to_merge) > 0:
                    item_tmp = merge_samples(items_to_merge)
                    if item_tmp != None:
                        items_merged.append(item_tmp)
                
                # BucketBatcher
                for item in items_merged:
                    # Generate full dur and sparsified dur
                    if hparams.get('use_mel_as_target', False):
                        mel2ph_ = item['mel2ph']
                    else:
                        mel2ph_ = item['mel2ph'][::hparams.get('vae_stride', 8)]
                    sparsified_dur = torch.zeros_like(mel2ph_)
                    for i in range(1, mel2ph_.max()+1):
                        indices = torch.where(sparsified_dur == i)[0]
                        if len(indices) > 0:
                            rand_idx = indices[torch.randint(len(indices), (1,)).item()]
                            sparsified_dur[rand_idx] = mel2ph_[rand_idx]
                    item['sparsified_dur'] = sparsified_dur

                    # Restrict max length
                    max_frames = hparams['max_frames'] // fm * fm
                    item['mel'] = item['mel'][:max_frames]
                    item['mel2ph'] = item['mel2ph'][:max_frames]
                    # obtain ctx mask
                    min_idx = max(int(len(item['mel']) * 0.1), 1)
                    max_idx = min(int(len(item['mel']) * 0.9), len(item['mel']) - 1)
                    rand_length = random.randint(min_idx, max_idx) // fm * fm
                    ctx_mask = torch.zeros_like(item['mel'][:, 0])
                    ctx_mask[:rand_length] = 1.0
                    item['ctx_mel'] = deepcopy(item['mel'])
                    item['ctx_mel'][rand_length:] = 0.0
                    if hparams.get('use_mel_as_target', False):
                        item['ctx_mask'] = ctx_mask[:, None]
                    else:
                        item['ctx_mask'] = ctx_mask[::hparams.get('vae_stride', 8), None]

                    """ Save to shm """
                    item = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in item.items()}
                    batch = batcher.collate_batch(item)
                    if batch is not None:
                        cnt = init_new_samples()
                        save_samples_to_shm(batch, cnt, shm_base)
                        restart_countdown -= 1
                        if restart_countdown == 0:
                            return
        except:
            traceback.print_exc()
    else:
        try:
            reader = get_reader(data_paths, reader_chunk_size, i_worker,
                                n_worker, reader_cache_name)
            batcher = BucketBatcher(
                buckets=[100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 1000, 1500, 2000],
                dynamic_batch=True,
                maximum_bucket_size=hparams.get('max_tokens', 40000),
                length_fn=(lambda x: x['mel'].shape[0]//8 + x['txt_token'].shape[0] + x['instruction'].shape[0]) if hparams.get('use_instruction', False) else (lambda x: x['mel'].shape[0]//8 + x['txt_token'].shape[0] * 2), # mel + phone + bpe
                bsz_evaluator=None,
            )
            fm = hparams['frames_multiple']

            def init_new_samples():
                while len(glob.glob(f'{shm_base}/*.pkl')) >= hparams.get('prefetch_steps', 200) * world_size:
                    time.sleep(1)
                with counter.get_lock():
                    cnt = counter.value
                    counter.value += 1
                random.seed((cnt // world_size) % 1001 + seed)
                return cnt

            restart_countdown = 10000
            while True:
                try:
                    items_bytes = read_items(q_to_pull, reader)
                except:
                    continue
                if items_bytes is None:
                    break
                
                items = []
                for item_ in items_bytes:
                    item_ = pickle.loads(item_)
                    item = {}
                    try:
                        item['txt_token'] = torch.LongTensor(item_['phone_encoded'])
                    except:
                        continue
                    # Load mel
                    try:
                        item['mel'] = torch.FloatTensor(FrontendLMDataset.get_mel(hparams, item_['wav'])[0])
                    except:
                        continue
                    if len(item['mel']) < hparams['min_frames']:
                        continue
                    ph_div_mel = len(item['txt_token']) / len(item['mel'])
                    if ph_div_mel > 0.3 or ph_div_mel < 0.01:
                        continue
                    item['tone'] = torch.LongTensor(item_['tone_encoded'])
                    item['mel2ph'] = torch.LongTensor(item_['mel2ph'])
                    item['mel2ph'] = item['mel2ph'][:len(item['mel2ph']) // fm * fm]
                    item['mel'] = item['mel'][:len(item['mel']) // fm * fm]
                    item['ph2char'] = torch.LongTensor(item_['ph2char']) if valid_item_kv(item_, 'ph2char') else None
                    item['char_token'] = torch.LongTensor(item_['char_encoded']) if valid_item_kv(item_, 'char_encoded') else None
                    item['spk_name'] = item_['spk_name']
                    item['item_name'] = item_['item_name']
                    item['text'] = item_['txt_raw']

                    # Generate full dur and sparsified dur
                    mel2ph_ = item['mel2ph'][::8]
                    sparsified_dur = torch.zeros_like(mel2ph_)
                    for i in range(1, mel2ph_.max()+1):
                        indices = torch.where(sparsified_dur == i)[0]
                        if len(indices) > 0:
                            rand_idx = indices[torch.randint(len(indices), (1,)).item()]
                            sparsified_dur[rand_idx] = mel2ph_[rand_idx]
                    item['sparsified_dur'] = sparsified_dur

                    exclude_spk = hparams.get('exclude_spk')
                    if exclude_spk is not None and item['spk_name'] in exclude_spk:
                        continue
                    # Restrict max length
                    max_frames = hparams['max_frames'] // fm * fm
                    item['mel'] = item['mel'][:max_frames]
                    item['mel2ph'] = item['mel2ph'][:max_frames]

                    ctx_mask = torch.zeros_like(item['mel'][:, 0])
                    ctx_mask[:random.randint(200, 350)] = 1.0
                    item['ctx_mask'] = ctx_mask[::8, None]

                    items.append(item)

                # BucketBatcher
                for item in items:
                    item = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in item.items()}
                    batch = batcher.collate_batch(item)
                    if batch is not None:
                        cnt = init_new_samples()
                        save_samples_to_shm(batch, cnt, shm_base)
                        restart_countdown -= 1
                        if restart_countdown == 0:
                            return
        except:
            traceback.print_exc()


def processer_fn_dit_wav(data_paths, q_to_pull, reader_chunk_size, world_size, shm_base, counter, hparams_,
                    seed, i_worker, n_worker, reader_cache_name='cache'):
    setproctitle.setproctitle('data_processer:processer_fn')
    hparams.update(hparams_)
    print(f"| Started processer_fn_dit_wav#{i_worker}/{n_worker}.")

    if hparams.get('use_glm4v_token', False):
        feature_extractor = WhisperFeatureExtractor.from_pretrained("checkpoints/glm-4-voice-tokenizer")
        resampler = torchaudio.transforms.Resample(
                        orig_freq=hparams['audio_sample_rate'],
                        new_freq=16000
                    )
    speech_augmentor = None
    if hparams.get('wav_add_noise', False) or hparams.get('wav_add_effect', False):
        from tasks.tts.dataset_utils.augment import SpeechAugment
        speech_augmentor = SpeechAugment(
            hparams.get('wav_add_noise', False), hparams.get('wav_add_effect', False),
            hparams.get('musan_dir', None), noise_prob=0.5, effect_prob=0.5, noise_snr=(6.0, 20.0)
        )
        print('| Noise mixer initialized!')
    try:
        reader = get_reader(data_paths, reader_chunk_size, i_worker,
                            n_worker, reader_cache_name)
        print(f"| init reader#{i_worker}/{n_worker}.")
        # batcher = BucketBatcher(
        #     buckets=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
        #     dynamic_batch=True,
        #     maximum_bucket_size=hparams.get('max_tokens', 40000),
        #     length_fn=lambda x: x['mel'].shape[0]//hparams['vae_stride'],
        #     bsz_evaluator=None,
        # )

        fm = hparams['frames_multiple']
        fm_wav = hparams['frames_multiple'] * hparams['hop_size']

        def init_new_samples():
            retry_num = 0
            while len(glob.glob(f'{shm_base}/*.pkl')) >= hparams.get('prefetch_steps', 200) * world_size:
                if DEBUG and retry_num % 20 == 0:
                    print(f"processer_fn_dit_wav#{i_worker}/{n_worker} waiting for shm to be released: {retry_num} seconds")
                time.sleep(1)
                retry_num += 1
            with counter.get_lock():
                cnt = counter.value
                counter.value += 1
            random.seed((cnt // world_size) % 1001 + seed)
            tgt_size = random.randint(hparams['tgt_size_min'], hparams['tgt_size_max'])
            samples = []
            spk_id = 0
            nframes = 0
            return samples, tgt_size, cnt, spk_id, nframes

        split_sample = MegaTTSDataset.split_sample
        samples, tgt_size, cnt, spk_id, nframes = init_new_samples()
        restart_countdown = 10000
        while True:
            try:
                items_bytes = read_items(q_to_pull, reader)
            except:
                continue
            if items_bytes is None:
                break
            items_merged = merge_item_bytes(items_bytes, exclude_spk=hparams.get('exclude_spk'))
            for item_merged in items_merged:
                add_spk_id = 0
                while True:
                    if len(item_merged['mel']) < tgt_size * 1.1:
                        break
                    item_tgt, item_merged = split_sample(item_merged, int(tgt_size * 1.1), force_word_bdr=True)
                    if item_merged is None:
                        break
                    if len(item_tgt['mel']) > hparams['min_frames']:
                        item_tgt['mel'] = item_tgt['mel'][:len(item_tgt['mel']) // fm * fm]
                        item_tgt['wav'] = item_tgt['wav'][:len(item_tgt['wav']) // fm_wav * fm_wav]
                        item_tgt['mel2ph'] = item_tgt['mel2ph'][:len(item_tgt['mel2ph']) // fm * fm]
                        item_tgt['spk_id'] = spk_id

                        # obtain ctx mask
                        min_idx = max(int(len(item_tgt['mel']) * 0.1), 200)
                        max_idx = min(int(len(item_tgt['mel']) * 0.9), len(item_tgt['mel']) - 200)
                        rand_length = random.randint(min_idx, max_idx) // fm * fm
                        ctx_mask = torch.zeros_like(item_tgt['mel'][:, 0:1])
                        ctx_mask[:rand_length] = 1.0
                        item_tgt['ctx_mask'] = ctx_mask[::hparams['vae_stride']]
                        item_tgt['ctx_wav'] = deepcopy(item_tgt['wav'])
                        item_tgt['ctx_wav'] = item_tgt['ctx_wav'][:rand_length*hparams['hop_size']]

                        if hparams.get('use_glm4v_token', False):
                            max_sec = 30
                            item_tgt['wav'] = item_tgt['wav'][:max_sec*24000]
                            item_tgt['mel'] = item_tgt['mel'][:max_sec*100]
                            item_tgt['mel2ph'] = item_tgt['mel2ph'][:max_sec*100]

                            # obtain ctx mask
                            min_idx = max(int(len(item_tgt['mel']) * 0.1), 200)
                            max_idx = min(int(len(item_tgt['mel']) * 0.9), len(item_tgt['mel']) - 200)
                            rand_length = random.randint(min_idx, max_idx) // fm * fm
                            ctx_mask = torch.zeros_like(item_tgt['mel'][:, 0:1])
                            ctx_mask[:rand_length] = 1.0
                            item_tgt['ctx_mask'] = ctx_mask[::hparams['vae_stride']]
                            item_tgt['ctx_wav'] = deepcopy(item_tgt['wav'])
                            item_tgt['ctx_wav'] = item_tgt['ctx_wav'][:rand_length*hparams['hop_size']]

                            # tgt wav for voice conversion
                            item_tgt['tgt_wav'] = deepcopy(item_tgt['wav'])
                            item_tgt['tgt_wav'][:rand_length*hparams['hop_size']] = 0.0
                            wav_16k = resampler(item_tgt['tgt_wav'])
                            if speech_augmentor is not None:
                                wav_16k = speech_augmentor(wav_16k, 16000)
                            features = feature_extractor(wav_16k.numpy()[:max_sec*16000], sampling_rate=16000,
                                         return_attention_mask=True, return_tensors="np", device='cpu',
                                         padding="longest", pad_to_multiple_of=1280)
                            item_tgt['glm4v_feature'] = features['input_features'][0].T
                            item_tgt['glm4v_attention_mask'] = features['attention_mask'][0]

                        item_tgt_ = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in item_tgt.items()}
                        item_tgt_ = deepcopy(item_tgt_)

                        samples.append(item_tgt_)
                        nframes += len(item_tgt_['mel'])
                        add_spk_id = 1
                    if nframes >= hparams['max_tokens']:
                        save_samples_to_shm(samples, cnt, shm_base)
                        if DEBUG:
                            print(f'processer_fn_dit_wav#{i_worker}/{n_worker} saved {shm_base}/{cnt}.pkl')
                        add_spk_id = 0
                        restart_countdown -= 1
                        if restart_countdown == 0:
                            return
                        samples, tgt_size, cnt, spk_id, nframes = init_new_samples()

                        # add_spk_id = 1
                        # batch = batcher.collate_batch(item_tgt_)
                        # if batch is not None:
                        #     save_samples_to_shm(batch, cnt, shm_base)
                        #     if DEBUG:
                        #         print(f'processer_fn_dit_wav#{i_worker}/{n_worker} saved {shm_base}/{cnt}.pkl')
                        #     add_spk_id = 0
                        #     restart_countdown -= 1
                        #     if restart_countdown == 0:
                        #         return
                        #     samples, tgt_size, cnt, spk_id, nframes = init_new_samples()

                spk_id += add_spk_id
    except:
        traceback.print_exc()

def processer_fn_dit_wav_text(data_paths, q_to_pull, reader_chunk_size, world_size, shm_base, counter, hparams_,
                    seed, i_worker, n_worker, reader_cache_name='cache'):
    setproctitle.setproctitle(f'data_processer:processer_fn_dit_wav_text ({i_worker}/{n_worker})')
    hparams.update(hparams_)
    print(f"| Started processer_fn_dit_wav_text#{i_worker}/{n_worker}.")

    try:
        reader = get_reader(data_paths, reader_chunk_size, i_worker,
                            n_worker, reader_cache_name)
        print(f"| init reader#{i_worker}/{n_worker}.")

        def length_fn(x):
            if hparams.get('length_fn', 'lat') == 'lat':
                return x['wav'].shape[0]//hparams['hop_size']//hparams['vae_stride']
            elif hparams.get('length_fn', 'lat') == 'ph':
                return len(x['ph_token'])

        batcher = BucketBatcher(
            buckets=[100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1100, 
                     1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
                     11000, 12000, 13000, 14000, 15000, 16000, 18000, 20000, 40000, 60000],
            dynamic_batch=True,
            maximum_bucket_size=hparams.get('max_tokens', 40000),
            length_fn=length_fn,
            bsz_evaluator=None,
        )
        
        if hparams.get('add_vad_mask', False):
            from utils.audio.vad import get_vad_model
            vad_model = get_vad_model()
        
        speech_augmentor = None
        if hparams.get('wav_add_noise', False) or hparams.get('wav_add_effect', False):
            from tasks.tts.dataset_utils.augment import SpeechAugment
            speech_augmentor = SpeechAugment(
                hparams.get('wav_add_noise', False), hparams.get('wav_add_effect', False), hparams.get('musan_dir', None), 
                noise_prob=hparams.get('wav_add_noise_prob', 0.5), effect_prob=hparams.get('wav_add_effect_prob', 0.5), 
                noise_snr=(6.0, 20.0), with_speech=hparams.get('musan_with_speech', False)
            )

        fm = hparams['frames_multiple']
        fm_wav = hparams['frames_multiple'] * hparams['hop_size']

        def init_new_samples():
            retry_num = 0
            while len(glob.glob(f'{shm_base}/*.pkl')) >= hparams.get('prefetch_steps', 200) * world_size:
                if DEBUG and retry_num % 20 == 0:
                    print(f"processer_fn_dit_wav_text#{i_worker}/{n_worker} waiting for shm to be released: {retry_num} seconds")
                time.sleep(1)
                retry_num += 1
            with counter.get_lock():
                cnt = counter.value
                counter.value += 1
            random.seed((cnt // world_size) % 1001 + seed)
            tgt_size = random.randint(hparams['tgt_size_min'], hparams['tgt_size_max'])
            samples = []
            spk_id = 0
            nframes = 0
            return samples, tgt_size, cnt, spk_id, nframes

        n_skip = 0
        split_sample = MegaTTSDataset.split_sample
        samples, tgt_size, cnt, spk_id, nframes = init_new_samples()
        restart_countdown = 10000
        while True:
            try:
                items_bytes = read_items(q_to_pull, reader)
            except:
                continue
            if items_bytes is None:
                break
            items_merged = merge_item_bytes(
                items_bytes, exclude_spk=hparams.get('exclude_spk'), 
                tgt_size=tgt_size, merge_multi_spk=hparams.get('merge_multi_spk', False)
            )
            for item_merged in items_merged:
                add_spk_id = 0
                # while True:
                #     if len(item_merged['mel']) < tgt_size * 1.1:
                #         break
                #     item_tgt, item_merged = split_sample(item_merged, int(tgt_size * 1.1), force_word_bdr=True)
                #     if item_merged is None:
                #         break
                item_tgt = item_merged
                # print("len(item_tgt['mel'])", len(item_tgt['mel']))
                if hparams['max_frames'] >= len(item_tgt['mel']) > hparams['min_frames']:
                    item_tgt['wav'] = item_tgt['wav'][:len(item_tgt['wav']) // fm_wav * fm_wav]
                    if speech_augmentor is not None:
                        item_tgt['wav'] = speech_augmentor(item_tgt['wav'], hparams['audio_sample_rate'])
                    mel_len = len(item_tgt['wav']) // hparams['hop_size']
                    if mel_len > len(item_tgt['mel2ph']):
                        mel2ph = torch.zeros(mel_len).long()
                        mel2ph[:len(item_tgt['mel2ph'])] = item_tgt['mel2ph']
                        mel2ph[len(item_tgt['mel2ph']):] = mel2ph[len(item_tgt['mel2ph'])-1]
                        item_tgt['mel2ph'] = mel2ph

                    item_tgt['mel'] = item_tgt['mel'][:len(item_tgt['mel']) // fm * fm]
                    item_tgt['mel2ph'] = item_tgt['mel2ph'][:len(item_tgt['mel2ph']) // fm * fm]
                    item_tgt['dur'] = mel2token_to_dur(item_tgt['mel2ph'])
                    item_tgt['spk_id'] = spk_id
                    
                    if len(item_tgt['mel2ph']) != len(item_tgt['wav']) // hparams['hop_size']:
                        n_skip += 1; continue

                    if hparams.get('use_ph_timestamp', False):
                        try:
                            item_tgt['ph_timestamp'] = FrontendLMDataset.get_ph_timestamp(item_tgt)
                        except:
                            n_skip += 1; continue
                    if hparams.get('use_merged_ph', False):
                        try:
                            item_tgt['merged_ph_token'] = map_phone_to_tokendict({
                                'txt_token': item_tgt['txt_token'], 'tone': item_tgt['tone']
                            }, pad_bos_eos=False)
                        except:
                            n_skip += 1; continue
                    if hparams.get('use_paraformer_dur_label', False):
                        from modules.asr.mfa.nar_mfa_utils import dur_to_paraformer_label
                        dur_paraformer_label = dur_to_paraformer_label(item_tgt['dur'])
                        item_tgt['dur_paraformer_label'] = dur_paraformer_label
                    if hparams.get('use_merged_ph', False) and 'dur' in hparams['task_cls']:
                        if item_tgt['merged_ph_token'].shape[0] != item_tgt['dur'].shape[0]:
                            n_skip += 1; continue
                    if hparams.get('valid_ph_dur', False):
                        if item_tgt['txt_token'].shape[0] != item_tgt['dur'].shape[0]:
                            n_skip += 1; continue

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

                    txt = raw_text_process(item_tgt['txt'], wav_len=item_tgt['wav'].shape[0])
                    if txt is None:
                        n_skip += 1; continue
                    item_tgt['text'] = txt
                    item_tgt['ph_token'] = item_tgt['txt_token']
                    if item_tgt['ph_token'].shape[0] > item_tgt['wav'].shape[0] // hparams['hop_size'] // 4:
                        n_skip += 1; continue

                    # obtain ctx mask
                    min_idx = max(int(len(item_tgt['mel']) * 0.1), 200)
                    max_idx = min(int(len(item_tgt['mel']) * 0.9), len(item_tgt['mel']) - 200)
                    if min_idx > max_idx:
                        min_idx = int(len(item_tgt['mel']) * 0.4)
                        max_idx = int(len(item_tgt['mel']) * 0.6)
                    rand_length = random.randint(min_idx, max_idx) // fm * fm
                    ctx_mask = torch.zeros((item_tgt['wav'].shape[0] // hparams['hop_size'], 1))
                    ctx_mask[:rand_length] = 1.0
                    item_tgt['ctx_mask'] = ctx_mask[::hparams['vae_stride']]
                    item_tgt['ctx_wav'] = deepcopy(item_tgt['wav'])
                    item_tgt['ctx_wav'] = item_tgt['ctx_wav'][:rand_length*hparams['hop_size']]
                    if hparams.get('add_vad_mask', False):
                        from utils.audio.vad import run_vad_trim
                        vad_start, vad_end = run_vad_trim(item_tgt['wav'], hparams['audio_sample_rate'], vad_model)
                        vm = hparams['hop_size'] * hparams['vae_stride']
                        vad_mask = np.zeros((item_tgt['wav'].shape[0] // vm))
                        vad_mask[int(vad_start * hparams['audio_sample_rate'] // vm): int(vad_end * hparams['audio_sample_rate'] // vm)] = 1
                        item_tgt['vad_mask'] = vad_mask # 直接是lat的shape
                    else:
                        item_tgt['vad_mask'] = None

                    item_tgt_ = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in item_tgt.items()}
                    item_tgt_ = deepcopy(item_tgt_)

                #     samples.append(item_tgt_)
                #     nframes += len(item_tgt_['mel'])
                #     add_spk_id = 1
                # if nframes >= hparams['max_tokens']:
                #     save_samples_to_shm(samples, cnt, shm_base)
                #     if DEBUG:
                #         print(f'processer_fn_dit_wav_text#{i_worker}/{n_worker} saved {shm_base}/{cnt}.pkl')
                #     add_spk_id = 0
                #     restart_countdown -= 1
                #     if restart_countdown == 0:
                #         return
                #     samples, tgt_size, cnt, spk_id, nframes = init_new_samples()

                    add_spk_id = 1
                    batch = batcher.collate_batch(item_tgt_)
                    if batch is not None and len(batch) > 0:
                        save_samples_to_shm(batch, cnt, shm_base)
                        if DEBUG:
                            print(f'processer_fn_dit_wav_text#{i_worker}/{n_worker} saved {shm_base}/{cnt}.pkl')
                        add_spk_id = 0
                        restart_countdown -= 1
                        if restart_countdown == 0:
                            return
                        samples, tgt_size, cnt, spk_id, nframes = init_new_samples()
                        
                    spk_id += add_spk_id

                else:
                    n_skip += 1
                    
            if n_skip % 1000 == 0 and n_skip > 0:
                print(f'processer_fn_dit_wav_text#{i_worker}/{n_worker} skipped [{n_skip}] items.')

    except:
        traceback.print_exc()


def processer_fn_dur_text_meta(data_paths, q_to_pull, reader_chunk_size, world_size, shm_base, counter, hparams_,
                    seed, i_worker, n_worker, reader_cache_name='cache'):
    setproctitle.setproctitle(f'data_processer:processer_fn_dur_text_meta ({i_worker}/{n_worker})')
    hparams.update(hparams_)
    print(f"| Started processer_fn_dur_text_meta#{i_worker}/{n_worker}.")

    try:
        reader = get_reader(data_paths, reader_chunk_size, i_worker,
                            n_worker, reader_cache_name)
        print(f"| init reader#{i_worker}/{n_worker}.")

        def length_fn(x):
            return len(x['ph_token'])

        batcher = BucketBatcher(
            buckets=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1100, 
                     1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 40000, 60000],
            dynamic_batch=True,
            maximum_bucket_size=hparams.get('max_tokens', 40000),
            length_fn=length_fn,
            bsz_evaluator=None,
        )

        fm = hparams['frames_multiple']
        fm_wav = hparams['frames_multiple'] * hparams['hop_size']
        load_wav = hparams.get('load_wav', True)

        def init_new_samples():
            retry_num = 0
            while len(glob.glob(f'{shm_base}/*.pkl')) >= hparams.get('prefetch_steps', 200) * world_size:
                if DEBUG and retry_num % 20 == 0:
                    print(f"processer_fn_dur_text_meta#{i_worker}/{n_worker} waiting for shm to be released: {retry_num} seconds")
                time.sleep(1)
                retry_num += 1
            with counter.get_lock():
                cnt = counter.value
                counter.value += 1
            random.seed((cnt // world_size) % 1001 + seed)
            tgt_size = random.randint(hparams['tgt_size_min'], hparams['tgt_size_max'])
            samples = []
            spk_id = 0
            nframes = 0
            return samples, tgt_size, cnt, spk_id, nframes

        n_skip = 0
        samples, tgt_size, cnt, spk_id, nframes = init_new_samples()
        restart_countdown = 10000
        while True:
            try:
                items_bytes = read_items(q_to_pull, reader)
            except:
                continue
            if items_bytes is None:
                break
            items_merged = merge_item_bytes(
                items_bytes, exclude_spk=hparams.get('exclude_spk'), 
                tgt_size=tgt_size, merge_multi_spk=hparams.get('merge_multi_spk', False),
                load_wav=load_wav
            )
            for item_merged in items_merged:
                add_spk_id = 0
                item_tgt = item_merged
                if hparams['max_frames'] >= len(item_tgt['mel2ph']) > hparams['min_frames']:
                    if load_wav:
                        item_tgt['wav'] = item_tgt['wav'][:len(item_tgt['wav']) // fm_wav * fm_wav]
                        mel_len = len(item_tgt['wav']) // hparams['hop_size']
                        if mel_len > len(item_tgt['mel2ph']):
                            mel2ph = torch.zeros(mel_len).long()
                            mel2ph[:len(item_tgt['mel2ph'])] = item_tgt['mel2ph']
                            mel2ph[len(item_tgt['mel2ph']):] = mel2ph[len(item_tgt['mel2ph'])-1]
                            item_tgt['mel2ph'] = mel2ph

                        item_tgt['mel'] = item_tgt['mel'][:len(item_tgt['mel']) // fm * fm]
                        item_tgt['mel2ph'] = item_tgt['mel2ph'][:len(item_tgt['mel2ph']) // fm * fm]
                        
                        if len(item_tgt['mel2ph']) != len(item_tgt['wav']) // hparams['hop_size']:
                            n_skip += 1; continue

                        wav_len = item_tgt['wav'].shape[0]
                    else:
                        wav_len = len(item_tgt['mel2ph']) * hparams['hop_size']
                            
                    item_tgt['dur'] = mel2token_to_dur(item_tgt['mel2ph'])
                    item_tgt['spk_id'] = spk_id

                    if hparams.get('use_ph_timestamp', False):
                        try:
                            item_tgt['ph_timestamp'] = FrontendLMDataset.get_ph_timestamp(item_tgt)
                        except:
                            n_skip += 1; continue
                    if hparams.get('use_merged_ph', False):
                        try:
                            item_tgt['merged_ph_token'] = map_phone_to_tokendict({
                                'txt_token': item_tgt['txt_token'], 'tone': item_tgt['tone']
                            }, pad_bos_eos=False)
                        except:
                            n_skip += 1; continue
                    if hparams.get('use_merged_ph', False) and 'dur' in hparams['task_cls']:
                        if item_tgt['merged_ph_token'].shape[0] != item_tgt['dur'].shape[0]:
                            n_skip += 1; continue

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

                    txt = raw_text_process(item_tgt['txt'], wav_len=wav_len)
                    if txt is None:
                        n_skip += 1; continue
                    item_tgt['text'] = txt
                    item_tgt['ph_token'] = item_tgt['txt_token']

                    item_tgt_ = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in item_tgt.items()}
                    item_tgt_ = deepcopy(item_tgt_)

                    add_spk_id = 1
                    batch = batcher.collate_batch(item_tgt_)
                    if batch is not None and len(batch) > 0:
                        save_samples_to_shm(batch, cnt, shm_base)
                        if DEBUG:
                            print(f'processer_fn_dur_text_meta#{i_worker}/{n_worker} saved {shm_base}/{cnt}.pkl')
                        add_spk_id = 0
                        restart_countdown -= 1
                        if restart_countdown == 0:
                            return
                        samples, tgt_size, cnt, spk_id, nframes = init_new_samples()
                    spk_id += add_spk_id

                else:
                    n_skip += 1
                    
            if n_skip % 10000 == 0 and n_skip > 0:
                print(f'processer_fn_dur_text_meta#{i_worker}/{n_worker} skipped [{n_skip}] items.')

    except:
        traceback.print_exc()
        
def processer_fn_dit_wav_audio(
    data_paths, q_to_pull, reader_chunk_size, world_size, shm_base, counter, hparams_,
    seed, i_worker, n_worker, reader_cache_name='cache'
):
    """
      规则（最新）：
      - 纯音频：text = "<Audio>"; caption = 
          * JSON 分支：直接使用 JSON 原文 caption（不包裹/不清洗/不改写）
          * 池分支：    "<Audio>{cap}</Audio>"
      - 非纯音频：
          * BGM：text = "<S1>{全文}</S1>"; caption = "<S1>{全文}</S1> <BGM>{cap}</BGM>"
          * SFX 中间：text = "<S1>{ref}</S1> <Audio> <S1>{tgt}</S1>";
                      caption = "<S1>{ref}</S1> <Audio>{cap}</Audio> <S1>{tgt}</S1>"
          * SFX 尾部：text = "<S1>{全文}</S1> <Audio>";
                      caption = "<S1>{全文}</S1> <Audio>{cap}</Audio>"
      - ph/merged_ph：audio 段一律使用 sil=145
      - 概率保持：命中哪一分支只在该分支内重试（hparams['branch_retry'] 或 8），不回退切换
      - 无 VAD；其它逻辑不变

      本版针对“读取速度与策略”的关键点：
      - 纯音频(JSON)：**整段读完整 wav**（librosa 优先），然后**仅保留时长在 [pure_json_min_sec, pure_json_max_sec]（默认 2~30s）**的条目；
        不做随机裁剪，只做 frames_multiple 对齐。
      - 纯音频(JSON) caption：**原样使用** JSON 的 caption。
      - 其余分支（BGM/SFX/非纯音频）逻辑不变。
    """
    import os, json, time, glob, traceback, pickle, re, hashlib, math
    from copy import deepcopy
    import numpy as np
    import torch
    import setproctitle

    setproctitle.setproctitle(f'data_processer:processer_fn_dit_wav_audio ({i_worker}/{n_worker})')
    hparams.update(hparams_)
    print(f"| Started processer_fn_dit_wav_audio#{i_worker}/{n_worker}.")

    # -------- 概率/阈值（可配） --------
    PURE_ONLY_RATIO = float(hparams.get('pure_ratio', 1.0))        # 纯音频总体概率
    MUSIC_RATIO     = float(hparams.get('music_ratio', 0.5))       # 非纯音频里 BGM 概率
    SFX_HEAD_RATIO  = float(hparams.get('sfx_head_ratio', 0.5))    # SFX 中间插入概率
    PURE_JSON_PROB  = float(hparams.get('pure_json_prob', 0.5))    # 纯音频里选 JSON 的概率
    PURE_BGM_RATIO  = float(hparams.get('pure_bgm_ratio', 0.5))    # 纯音频(非 JSON)里 BGM 概率

    # 纯音频(JSON) 时长过滤（默认 2~30s）；注意：仅用于**过滤**，不做随机裁剪
    PURE_JSON_MIN_SEC = float(hparams.get('pure_json_min_sec', 2.0))
    PURE_JSON_MAX_SEC = float(hparams.get('pure_json_max_sec', 30.0))
    PURE_JSON_PICK_RETRIES = int(hparams.get('pure_json_pick_retries', 16))
    LIBROSA_RES_TYPE = hparams.get('librosa_res_type', 'kaiser_fast')  # 加速重采样

    SNR_MIN_DB     = -3.0
    SNR_MAX_DB     =  6.0

    SIL_TOKEN_ID   = 145  # audio 段统一使用 sil=145

    # 仅用于清洗输入 txt 的旧标签；不会出现在 text 输出
    AUDIO_TAG      = '<Audio>'
    BGM_TAG        = '<BGM>'
    SPK_TAG        = '<SPK>'

    S1_L, S1_R     = '<S1>', '</S1>'
    BRANCH_MAX_RETRY = int(hparams.get('branch_retry', 8))

    def _wrap_s1_if_nonempty(s: str) -> str:
        s = (s or '').strip()
        return (f"{S1_L}{s}{S1_R}") if s else ""

    def _make_rng(step_hint:int):
        import random as _r
        key = f"{os.getpid()}|#{i_worker}|{step_hint}|{time.time_ns()}|{seed}".encode()
        seed64 = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), 'little', signed=False)
        return _r.Random(seed64)

    # -------- 文本辅助：按 token 比例切分字符串，并对齐就近边界 --------
    _BND_RE = re.compile(r'[ \t\n，。！？；：,.!?…]')

    def _split_text_by_token_ratio(txt_norm:str, insert_idx0:int, n_tokens:int):
        if not txt_norm or n_tokens <= 0:
            return "", txt_norm or ""
        L = len(txt_norm)
        j = int(round((max(0, min(insert_idx0, n_tokens)) / float(n_tokens)) * L))
        j = max(0, min(L, j))
        cut = j
        found = None
        for k in range(j, min(L, j+48)):
            if _BND_RE.match(txt_norm[k]):
                found = k+1
                break
        if found is None:
            for k in range(j-1, max(0, j-48)-1, -1):
                if _BND_RE.match(txt_norm[k]):
                    found = k+1
                    break
        if found is not None:
            cut = found
        ref = txt_norm[:cut].strip()
        tgt = txt_norm[cut:].strip()
        return ref, tgt

    def _ensure_trailing_punct(s: str):
        if not s: return s
        if re.search(r'[。！？.!?…]$', s): return s
        return s + ('。' if re.search(r'[\u4e00-\u9fff]', s) else '.')

    try:
        reader = get_reader(data_paths, reader_chunk_size, i_worker, n_worker, reader_cache_name)
        print(f"| init reader#{i_worker}/{n_worker}.")

        def length_fn(x):
            if hparams.get('length_fn', 'lat') == 'lat':
                return x['wav'].shape[0] // hparams['hop_size'] // hparams['vae_stride']
            elif hparams.get('length_fn', 'lat') == 'ph':
                return len(x['ph_token'])

        batcher = BucketBatcher(
            buckets=[100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1100,
                     1200,1300,1400,1500,1600,1700,1800,1900,2000,3000,4000,5000,6000,7000,8000,9000,10000,
                     11000,12000,13000,14000,15000,16000,18000,20000,40000,60000],
            dynamic_batch=True,
            maximum_bucket_size=hparams.get('max_tokens', 40000),
            length_fn=length_fn,
            bsz_evaluator=None,
        )

        fm         = hparams['frames_multiple']
        hop        = hparams['hop_size']
        fm_wav     = fm * hop
        sr_model   = hparams.get('audio_sample_rate', 24000)
        vae_stride = hparams['vae_stride']

        # ---------- JSONL 索引（池：SFX/MUSIC，.npy） ----------
        def _cache_path(jsonl_path, with_dur=False):
            return jsonl_path + (".paths_sr_cap_dur_v1.pkl" if with_dur else ".paths_sr_cap_v2.pkl")

        def _pool_is_valid_dict(pool, with_dur):
            try:
                if not isinstance(pool, list) or len(pool) == 0: return False
                e = pool[0]
                base_keys = {'npy', 'sr', 'cap', 'raw_json'}
                if not isinstance(e, dict) or not base_keys.issubset(e.keys()): return False
                if with_dur and 'dur' not in e: return False
                return True
            except Exception:
                return False

        def _atomic_dump_pickle(obj, dst):
            tmp = dst + f".tmp.{os.getpid()}"
            with open(tmp, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush(); os.fsync(f.fileno())
            os.replace(tmp, dst)

        def _acquire_lock(lock_path, timeout=600, poll=0.1):
            start = time.time()
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
            try:
                import fcntl
                while True:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        return fd
                    except BlockingIOError:
                        if time.time() - start > timeout:
                            raise TimeoutError(f"cache lock timeout: {lock_path}")
                        time.sleep(poll)
            except Exception:
                os.close(fd); raise

        def _release_lock(fd):
            try:
                import fcntl; fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

        def load_pool(jsonl_path, with_dur=False):
            if not jsonl_path: return []
            pkl = _cache_path(jsonl_path, with_dur)
            lock = pkl + ".lock"

            if os.path.exists(pkl):
                try:
                    with open(pkl, "rb") as f:
                        pool = pickle.load(f)
                    if _pool_is_valid_dict(pool, with_dur):
                        return pool
                    else:
                        print(f"| WARN: cached pool schema mismatch, rebuild: {pkl}")
                except Exception:
                    pass

            fd = _acquire_lock(lock)
            try:
                if os.path.exists(pkl):
                    try:
                        with open(pkl, "rb") as f:
                            pool = pickle.load(f)
                        if _pool_is_valid_dict(pool, with_dur):
                            return pool
                    except Exception:
                        pass

                if not os.path.exists(jsonl_path):
                    raise RuntimeError(f"jsonl not found: {jsonl_path}")

                pool = []
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for ln, line in enumerate(f, 1):
                        o = json.loads(line); raw_json = line.strip()
                        npy = o.get('npy_24k_path') or o.get('npy_path') or o.get('npy')
                        sr  = int(o.get('sr_24k', sr_model))
                        cap = o.get('caption', '')
                        if not npy or not os.path.exists(npy):
                            raise RuntimeError(f"line#{ln}: npy missing or not exists | item={raw_json}")
                        if not with_dur:
                            pool.append({'npy':npy, 'sr':sr, 'cap':cap, 'raw_json':raw_json})
                        else:
                            dur = float(o['duration_24k']) if o.get('duration_24k') is not None else None
                            pool.append({'npy':npy, 'sr':sr, 'cap':cap, 'dur':dur, 'raw_json':raw_json})
                try:
                    _atomic_dump_pickle(pool, pkl)
                except Exception:
                    pass
                return pool
            finally:
                _release_lock(fd)

        sfx_pool   = load_pool(hparams.get('sfx_jsonl'), with_dur=True)
        music_pool = load_pool(hparams.get('music_jsonl'), with_dur=False)

        # ---------- 纯音频 JSON 池（caption 原样使用；整段读取；秒级过滤） ----------
        def _normalize_extra_item(o):
            wav = o.get('wav_path') or o.get('wav') or o.get('path')
            cap_raw = o.get('caption') or ''
            if not wav or not os.path.exists(wav):
                return None
            return {'wav': wav, 'cap_raw': cap_raw}

        def _load_extra_pure_pool(paths):
            pool = []
            if not paths: return pool
            if isinstance(paths, str): paths = [paths]
            for p in paths:
                if not os.path.exists(p):
                    print(f"| WARN: pure_audio_jsons not found: {p}")
                    continue
                try:
                    if p.endswith('.jsonl'):
                        with open(p, 'r', encoding='utf-8') as f:
                            for ln, line in enumerate(f, 1):
                                try:
                                    o = json.loads(line)
                                    e = _normalize_extra_item(o)
                                    if e: pool.append(e)
                                except Exception as e:
                                    print(f"| WARN: JSONL parse error {p}#{ln}: {e}")
                    else:
                        with open(p, 'r', encoding='utf-8') as f:
                            obj = json.load(f)
                        if isinstance(obj, list):
                            for o in obj:
                                e = _normalize_extra_item(o)
                                if e: pool.append(e)
                        elif isinstance(obj, dict):
                            cand = obj.get('items') or obj.get('data') or obj.get('audios')
                            if isinstance(cand, list):
                                for o in cand:
                                    e = _normalize_extra_item(o)
                                    if e: pool.append(e)
                            else:
                                e = _normalize_extra_item(obj)
                                if e: pool.append(e)
                        else:
                            print(f"| WARN: unknown JSON root for {p}")
                except Exception as e:
                    print(f"| WARN: failed to load {p}: {e}")
            return pool

        extra_pure_pool = _load_extra_pure_pool(hparams.get('pure_audio_jsons'))

        if i_worker == 0:
            print(f"| SFX pool size: {len(sfx_pool)}; MUSIC pool size: {len(music_pool)}; EXTRA-PURE pool size: {len(extra_pure_pool)}")

        # ---------- 音频与重采样小工具 ----------
        def _fade_edges(x, sr, ms=5):
            n = int(sr * ms / 1000.0)
            if n <= 0 or len(x) < 2*n: return x
            ramp = np.linspace(0, 1, n, dtype=x.dtype)
            x[:n] *= ramp
            x[-n:] *= ramp[::-1]
            return x

        def _rms(x): return float(np.sqrt(np.mean(x.astype(np.float64)**2) + 1e-12))

        def _mix_at_snr(s, n, snr_db):
            Ps = _rms(s)**2
            Pn = _rms(n)**2 + 1e-12
            a = np.sqrt(Ps/(Pn*(10.0**(snr_db/10.0))))
            y = s + a*n
            m = float(np.max(np.abs(y)) + 1e-12)
            return (y/m if m>1.0 else y).astype(np.float32)

        def _load_wav_full(path, sr_out):
            """整段读取并重采样到 sr_out；优先 librosa（快），失败再退回 soundfile/wave。"""
            x = None; sr = None
            # 1) librosa（与示例保持一致；用 kaiser_fast 提速）
            try:
                import librosa
                x, sr = librosa.load(path, sr=sr_out, mono=True, res_type=LIBROSA_RES_TYPE)
                x = x.astype(np.float32)
            except Exception:
                x = None
            # 2) soundfile 读全 + resample（很少走到）
            if x is None:
                try:
                    import soundfile as sf
                    x, sr = sf.read(path, dtype='float32', always_2d=False)
                    if x.ndim > 1: x = x.mean(axis=1)
                    if int(sr) != int(sr_out):
                        # 退回到 librosa 重采样（若存在），否则用线性插值
                        try:
                            import librosa
                            x = librosa.resample(x, orig_sr=sr, target_sr=sr_out, res_type=LIBROSA_RES_TYPE).astype(np.float32)
                        except Exception:
                            t_old = np.arange(len(x), dtype=np.float64) / float(sr)
                            t_new = np.arange(int(round(len(x) * float(sr_out) / float(sr))), dtype=np.float64) / float(sr_out)
                            x = np.interp(t_new, t_old, x).astype(np.float32)
                        sr = sr_out
                except Exception:
                    x = None
            # 3) wave（PCM16）
            if x is None:
                try:
                    import wave
                    with wave.open(path, 'rb') as wf:
                        sr = wf.getframerate()
                        n = wf.getnframes()
                        ch = wf.getnchannels()
                        buf = wf.readframes(n)
                        dt = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
                        if ch > 1:
                            dt = dt.reshape(-1, ch).mean(axis=1)
                        # 线性插值重采样
                        t_old = np.arange(len(dt), dtype=np.float64) / float(sr)
                        t_new = np.arange(int(round(len(dt) * float(sr_out) / float(sr))), dtype=np.float64) / float(sr_out)
                        x = np.interp(t_new, t_old, dt).astype(np.float32)
                        sr = sr_out
                except Exception:
                    pass
            if x is None:
                raise RuntimeError(f"failed to load wav: {path}")
            return x.astype(np.float32), sr_out

        def _pick_valid_music(pool, sr_req, rng):
            if not pool: return None
            L = len(pool)
            for _ in range(min(10, L)):
                idx = rng.randrange(L)
                e = pool[idx]
                npy, sr = e['npy'], e['sr']
                if int(sr)!=int(sr_req) or not os.path.exists(npy): continue
                try:
                    wav = np.load(npy, mmap_mode='r').astype(np.float32)
                except Exception:
                    continue
                return wav, e['cap']
            return None

        def _pick_valid_sfx(pool, sr_req, rng):
            if not pool: return None
            L = len(pool)
            for _ in range(min(10, L)):
                idx = rng.randrange(L)
                e = pool[idx]
                npy, sr = e['npy'], e['sr']
                if int(sr)!=int(sr_req) or not os.path.exists(npy): continue
                try:
                    wav = np.load(npy, mmap_mode='r').astype(np.float32)
                except Exception:
                    continue
                return wav, e['cap'], e.get('raw_json', f'{{"npy":"{npy}"}}')
            return None

        def _bgm_match_length(sr_model, target_len, pool, rng):
            ret = _pick_valid_music(pool, sr_model, rng)
            if ret is None:
                raise RuntimeError("Music pool pick failed.")
            wav, cap = ret
            if target_len <= 0:
                raise RuntimeError("Invalid target_len for BGM.")
            if len(wav) >= target_len:
                st = rng.randint(0, len(wav)-target_len)
                seg = np.array(wav[st:st+target_len], dtype=np.float32)
            else:
                reps = target_len // max(1,len(wav)) + 1
                seg = np.tile(np.asarray(wav, dtype=np.float32), reps)[:target_len]
            return _fade_edges(seg, sr_model), cap

        def _pick_bgm_full(sr_model, pool, rng):
            ret = _pick_valid_music(pool, sr_model, rng)
            if ret is None:
                raise RuntimeError("Music pool pick failed (pure).")
            wav, cap = ret
            if len(wav) < hop:
                raise RuntimeError("Picked BGM too short.")
            return _fade_edges(np.asarray(wav, dtype=np.float32), sr_model), cap

        def _pick_sfx_full(sr_model, pool, rng):
            ret = _pick_valid_sfx(pool, sr_model, rng)
            if ret is None:
                raise RuntimeError("SFX pool pick failed (pure).")
            wav, cap, _raw_json = ret
            if len(wav) < hop:
                raise RuntimeError("Picked SFX too short.")
            return _fade_edges(np.asarray(wav, dtype=np.float32), sr_model), cap

        # 纯音频(JSON)：整段读取 -> 秒级过滤 -> 对齐
        def _pick_extra_pure_full(sr_model, pool, rng, fm, hop, min_sec, max_sec, max_tries):
            if not pool: return None
            L = len(pool)
            tries = min(max_tries, max(1, L))
            for _ in range(tries):
                e = pool[rng.randrange(L)]
                wav_path = e['wav']
                if not os.path.exists(wav_path):
                    continue
                try:
                    wav, _ = _load_wav_full(wav_path, sr_model)  # 整段读取
                except Exception:
                    continue
                if wav is None or len(wav) < hop:
                    continue
                dur_sec = float(len(wav)) / float(sr_model)
                # 仅保留 2~30s（可配），否则丢弃
                if not (min_sec <= dur_sec <= max_sec):
                    continue
                # 对齐到 frames_multiple
                wav = wav[:len(wav)//(hop)*hop]
                mel_len = len(wav) // hop
                mel_len = (mel_len // fm) * fm
                if mel_len < fm:
                    continue
                wav = wav[:mel_len*hop]
                return _fade_edges(wav.astype(np.float32), sr_model), e['cap_raw']
            return None

        # ---------- 其它流程 ----------
        def init_new_samples():
            retry = 0
            while len(glob.glob(f'{shm_base}/*.pkl')) >= hparams.get('prefetch_steps',200)*world_size:
                time.sleep(1); retry += 1
            with counter.get_lock():
                cnt = counter.value; counter.value += 1
            tgt_rng = _make_rng(cnt)
            tgt_size = tgt_rng.randint(hparams['tgt_size_min'], hparams['tgt_size_max'])
            return [], tgt_size, cnt, 0, 0, tgt_rng

        n_skip = 0
        samples, tgt_size, cnt, spk_id, nframes, rng_main = init_new_samples()
        restart_countdown = 10000

        while True:
            try:
                items_bytes = read_items(q_to_pull, reader)
            except Exception:
                continue
            if items_bytes is None: break

            items_merged = merge_item_bytes(
                items_bytes, exclude_spk=hparams.get('exclude_spk'),
                tgt_size=tgt_size, merge_multi_spk=hparams.get('merge_multi_spk', False)
            )

            for item_merged in items_merged:
                rng = _make_rng(cnt ^ (spk_id<<16) ^ len(items_merged))
                add_spk_id = 0
                item_tgt = item_merged

                try:
                    # 是否走纯音频；纯音频不读取 reference
                    pure_mode = (rng.random() < PURE_ONLY_RATIO)

                    if pure_mode:
                        # ========== 纯音频 ==========
                        use_extra = bool(extra_pure_pool) and (rng.random() < PURE_JSON_PROB)
                        wav_pure = None; cap = ""

                        for _try in range(BRANCH_MAX_RETRY):
                            try:
                                if use_extra:
                                    ret = _pick_extra_pure_full(sr_model, extra_pure_pool, rng,
                                                                fm=fm, hop=hop,
                                                                min_sec=PURE_JSON_MIN_SEC,
                                                                max_sec=PURE_JSON_MAX_SEC,
                                                                max_tries=PURE_JSON_PICK_RETRIES)
                                    if ret is None:
                                        raise RuntimeError("extra pure pick failed")
                                    wav_pure, cap = ret
                                else:
                                    # 若两池皆空但 JSON 不空，则回退到 JSON
                                    if (not sfx_pool) and (not music_pool) and extra_pure_pool:
                                        use_extra = True
                                        continue
                                    if (not sfx_pool) and (not music_pool):
                                        raise RuntimeError("no pools for pure-mode")
                                    pick_bgm = (rng.random() < PURE_BGM_RATIO)
                                    if pick_bgm and music_pool:
                                        wav_pure, cap = _pick_bgm_full(sr_model, music_pool, rng)
                                    elif sfx_pool:
                                        wav_pure, cap = _pick_sfx_full(sr_model, sfx_pool, rng)
                                    else:
                                        wav_pure, cap = _pick_bgm_full(sr_model, music_pool, rng)
                                break
                            except Exception:
                                wav_pure = None
                                continue
                        if wav_pure is None:
                            # 本条失败，跳过
                            n_skip += 1
                            if (n_skip % 200 == 1) and i_worker == 0:
                                print(f"| SKIP(pure-json) failed to pick (cnt={cnt})")
                            continue

                        # 对齐到 frames_multiple（再兜底一次）
                        wav_pure = wav_pure[:len(wav_pure)//fm_wav*fm_wav]
                        mel_len_all = len(wav_pure)//hop

                        # 池分支仍做全局 min/max_frames 校验；JSON 分支只做 2~30s 校验（上面已做）
                        if not use_extra:
                            if mel_len_all <= hparams['min_frames']:
                                n_skip += 1
                                if (n_skip % 200 == 1) and i_worker == 0:
                                    print(f"| SKIP(pure) short clip: frames={mel_len_all} <= min_frames={hparams['min_frames']} (cnt={cnt})")
                                continue
                            if not (hparams['min_frames'] < mel_len_all <= hparams['max_frames']):
                                n_skip += 1
                                if (n_skip % 200 == 1) and i_worker == 0:
                                    print(f"| SKIP(pure) mel_len out of range: {mel_len_all} (cnt={cnt})")
                                continue

                        item_tgt['wav'] = wav_pure

                        mel2ph = torch.ones((mel_len_all,), dtype=torch.long)
                        mel2ph = mel2ph[:len(mel2ph)//hparams['frames_multiple']*hparams['frames_multiple']]
                        item_tgt['mel2ph'] = mel2ph
                        item_tgt['dur'] = mel2token_to_dur(item_tgt['mel2ph'])

                        # 纯音频：ph 为 [sil]
                        item_tgt['ph_token'] = np.array([SIL_TOKEN_ID], dtype=np.int64)
                        if hparams.get('use_merged_ph', False):
                            item_tgt['merged_ph_token'] = np.array([SIL_TOKEN_ID], dtype=np.int64)
                            if item_tgt['merged_ph_token'].shape[0] != item_tgt['dur'].shape[0]:
                                raise RuntimeError("merged_ph_token len mismatch with dur in pure-mode")

                        if 'tone' in item_tgt:
                            tone0 = item_tgt['tone']
                            zero = (torch.zeros((1,), dtype=tone0.dtype, device=tone0.device)
                                    if isinstance(tone0, torch.Tensor)
                                    else np.zeros((1,), dtype=tone0.dtype))
                            item_tgt['tone'] = zero

                        # 纯音频 text 与 caption
                        item_tgt['text'] = "<Audio>"
                        item_tgt['txt']  = item_tgt['text']
                        if use_extra:
                            item_tgt['caption'] = cap            # 原样使用 JSON caption
                            item_tgt['caption_audio'] = cap
                        else:
                            item_tgt['caption'] = _ensure_trailing_punct(f"<Audio>{cap}</Audio>")
                            item_tgt['caption_audio'] = cap

                        mask_mel_zeros = torch.zeros((mel_len_all, 1))
                        mask_mel_ones  = torch.ones((mel_len_all, 1))
                        item_tgt['ctx_mask']   = mask_mel_zeros[::vae_stride]
                        item_tgt['audio_mask'] = mask_mel_ones[::vae_stride]
                        item_tgt['ctx_wav']    = np.zeros((fm_wav,), dtype=np.float32)
                        item_tgt['spk_id']     = spk_id
                        item_tgt.pop('mel', None)
                        item_tgt['vad_mask'] = None

                    else:
                        # ========== 非纯音频：此处读取 reference ==========
                        if 'wav' not in item_tgt:
                            raise RuntimeError("missing wav in input item")
                        wav_ref = np.asarray(item_tgt['wav'], dtype=np.float32)
                        wav_ref = wav_ref[:len(wav_ref)//hop*hop]
                        item_tgt['wav'] = wav_ref.copy()
                        mel_len_ref = len(wav_ref)//hop

                        if 'mel2ph' not in item_tgt:
                            raise RuntimeError("missing mel2ph in input item")
                        old_mel2ph = item_tgt['mel2ph'].long()
                        if len(old_mel2ph) != mel_len_ref:
                            if mel_len_ref > len(old_mel2ph):
                                pad = torch.full((mel_len_ref-len(old_mel2ph),),
                                                 int(old_mel2ph[-1]) if len(old_mel2ph)>0 else 1, dtype=torch.long)
                                old_mel2ph = torch.cat([old_mel2ph, pad], dim=0)
                            else:
                                old_mel2ph = old_mel2ph[:mel_len_ref]

                        if not (hparams['max_frames'] >= mel_len_ref > hparams['min_frames']):
                            raise RuntimeError(f"ref mel_len out of range: {mel_len_ref}")

                        # ref→tgt 切分点
                        mel_len_ctx = mel_len_ref
                        min_idx = max(int(mel_len_ctx*0.1), 200)
                        max_idx = min(int(mel_len_ctx*0.9), mel_len_ctx-200)
                        if min_idx > max_idx:
                            min_idx = int(mel_len_ctx*0.4); max_idx = int(mel_len_ctx*0.6)
                        rand_length = max(1, rng.randint(min_idx, max_idx)) // hparams['frames_multiple'] * hparams['frames_multiple']

                        did_sfx = did_bgm = False
                        sfx_cap = bgm_cap = None
                        sfx_at_tgt_head = False
                        sfx_frames_added = 0
                        insert_tok_idx = None  # 1-based token 索引

                        def do_bgm():
                            nonlocal did_bgm, bgm_cap
                            try:
                                bgm, cap = _bgm_match_length(sr_model, target_len=len(item_tgt['wav']), pool=music_pool, rng=rng)
                                snr = rng.uniform(SNR_MIN_DB, SNR_MAX_DB)
                                item_tgt['wav'] = _mix_at_snr(item_tgt['wav'], bgm, snr)
                                bgm_cap = cap; did_bgm = True; return True
                            except Exception:
                                return False

                        def do_sfx(place='tgt_head'):
                            nonlocal did_sfx, sfx_cap, old_mel2ph, sfx_frames_added, sfx_at_tgt_head, insert_tok_idx
                            try:
                                seg, cap = _pick_sfx_full(sr_model, sfx_pool, rng)
                                sfx_frames = len(seg)//hop
                                sfx_frames_added = sfx_frames
                                if place == 'tgt_head':
                                    pre = item_tgt['wav'][:rand_length*hop]
                                    suf = item_tgt['wav'][rand_length*hop:]
                                    item_tgt['wav'] = np.concatenate([pre, seg, suf], axis=0)
                                    if rand_length < len(old_mel2ph):
                                        insert_tok_idx = int(old_mel2ph[rand_length])
                                    else:
                                        insert_tok_idx = int(old_mel2ph[-1]) + 1
                                    prefix = old_mel2ph[:rand_length]
                                    suffix = old_mel2ph[rand_length:]
                                    if suffix.numel() > 0:
                                        shift_mask = (suffix >= insert_tok_idx).long()
                                        suffix = suffix + shift_mask
                                    audio_vec = torch.full((sfx_frames,), insert_tok_idx, dtype=torch.long)
                                    old_mel2ph = torch.cat([prefix, audio_vec, suffix], dim=0)
                                    sfx_at_tgt_head = True
                                else:
                                    item_tgt['wav'] = np.concatenate([item_tgt['wav'], seg], axis=0)
                                    T_tokens = int(item_tgt['txt_token'].shape[0] if hasattr(item_tgt['txt_token'], 'shape')
                                                   else len(item_tgt['txt_token']))
                                    new_tail = torch.full((sfx_frames,), T_tokens+1, dtype=torch.long)
                                    old_mel2ph = torch.cat([old_mel2ph, new_tail], dim=0)
                                    sfx_at_tgt_head = False
                                sfx_cap = cap; did_sfx = True; return True
                            except Exception:
                                return False

                        if sfx_pool and music_pool:
                            if rng.random() < MUSIC_RATIO:
                                ok = False
                                for _try in range(BRANCH_MAX_RETRY):
                                    if do_bgm(): ok = True; break
                                if not ok: raise RuntimeError("do_bgm failed after retries (no fallback).")
                            else:
                                place = 'tgt_head' if (rng.random() < SFX_HEAD_RATIO) else 'tail'
                                ok = False
                                for _try in range(BRANCH_MAX_RETRY):
                                    if do_sfx(place=place): ok = True; break
                                if not ok: raise RuntimeError("do_sfx failed after retries (no fallback).")
                        elif music_pool:
                            ok = False
                            for _try in range(BRANCH_MAX_RETRY):
                                if do_bgm(): ok = True; break
                            if not ok: raise RuntimeError("do_bgm failed after retries (music only).")
                        elif sfx_pool:
                            place = 'tgt_head' if (rng.random() < SFX_HEAD_RATIO) else 'tail'
                            ok = False
                            for _try in range(BRANCH_MAX_RETRY):
                                if do_sfx(place=place): ok = True; break
                            if not ok: raise RuntimeError("do_sfx failed after retries (sfx only).")
                        else:
                            raise RuntimeError("no pools for non-pure mode")

                        item_tgt['wav'] = item_tgt['wav'][:len(item_tgt['wav'])//fm_wav*fm_wav]
                        mel_len = len(item_tgt['wav'])//hop

                        mel2ph = old_mel2ph
                        if did_bgm and mel_len > len(mel2ph):
                            tail = int(mel2ph[-1]) if len(mel2ph)>0 else 1
                            mel2ph = torch.cat([mel2ph, torch.full((mel_len-len(mel2ph),), max(tail,1), dtype=torch.long)], dim=0)
                        mel2ph = mel2ph[:mel_len]
                        mel2ph = mel2ph[:len(mel2ph)//hparams['frames_multiple']*hparams['frames_multiple']]
                        item_tgt['mel2ph'] = mel2ph

                        item_tgt.pop('mel', None)

                        if 'txt' in item_tgt and item_tgt['txt'] is not None:
                            body_txt = item_tgt['txt']
                        else:
                            body_txt = ''
                        _body_for_norm = body_txt.replace(SPK_TAG, '').replace(AUDIO_TAG, '').replace(BGM_TAG, '').strip()
                        if not _body_for_norm:
                            n_skip += 1
                            print(f'| WARN: empty text after stripping tags; skip cnt={cnt}')
                            continue

                        txt_norm = raw_text_process(_body_for_norm, wav_ref)
                        if not txt_norm:
                            n_skip += 1
                            print(f'| WARN: text normalization returned None; body={_body_for_norm!r}; skip cnt={cnt}')
                            continue

                        txt_norm = re.sub(r'\s+([，。！？；：,.!?…])', r'\1', txt_norm)
                        txt_norm = re.sub(r'([ 　\s]*[，。！？；：,.!?…])\1+$', r'\1', txt_norm)

                        ref_text_part, tgt_text_part = "", ""
                        base_txt_token = item_tgt['txt_token']
                        base_n_tokens = int(base_txt_token.shape[0] if hasattr(base_txt_token, 'shape') else len(base_txt_token))
                        if did_sfx and sfx_at_tgt_head:
                            if insert_tok_idx is None:
                                insert_idx0 = base_n_tokens
                            else:
                                insert_idx0 = max(0, min(base_n_tokens, int(insert_tok_idx)-1))
                            ref_text_part, tgt_text_part = _split_text_by_token_ratio(txt_norm, insert_idx0, base_n_tokens)

                        if did_sfx:
                            if sfx_at_tgt_head:
                                parts = []
                                p_ref = _wrap_s1_if_nonempty(ref_text_part)
                                p_tgt = _wrap_s1_if_nonempty(tgt_text_part)
                                if p_ref: parts.append(p_ref)
                                parts.append("<Audio>")
                                if p_tgt: parts.append(p_tgt)
                                item_tgt['text'] = " ".join(parts).strip()
                            else:
                                base = _wrap_s1_if_nonempty(txt_norm)
                                item_tgt['text'] = (base + (" " if base else "") + "<Audio>").strip()
                        elif did_bgm:
                            item_tgt['text'] = _wrap_s1_if_nonempty(txt_norm)
                        else:
                            item_tgt['text'] = _wrap_s1_if_nonempty(txt_norm)

                        item_tgt['txt']  = item_tgt['text']
                        item_tgt['dur'] = mel2token_to_dur(item_tgt['mel2ph'])
                        item_tgt['spk_id'] = spk_id

                        def _insert_val(arr, idx0, val):
                            if isinstance(arr, torch.Tensor):
                                return torch.cat([arr[:idx0].long(),
                                                  torch.tensor([val], dtype=arr.dtype, device=arr.device).long(),
                                                  arr[idx0:].long()], dim=0)
                            else:
                                arr_np = np.asarray(arr, dtype=np.int64)
                                return np.concatenate([arr_np[:idx0],
                                                       np.array([val], dtype=arr_np.dtype),
                                                       arr_np[idx0:]], axis=0)

                        base_ph = item_tgt['txt_token']
                        if did_sfx and sfx_at_tgt_head:
                            if insert_tok_idx is None:
                                raise RuntimeError("insert_tok_idx missing for SFX at target head")
                            insert_idx0 = max(0, int(insert_tok_idx) - 1)
                            if isinstance(base_ph, torch.Tensor):
                                item_tgt['ph_token'] = _insert_val(base_ph.long(), insert_idx0, SIL_TOKEN_ID)
                            else:
                                item_tgt['ph_token'] = _insert_val(base_ph, insert_idx0, SIL_TOKEN_ID)
                            if 'tone' in item_tgt and item_tgt['tone'] is not None:
                                tone0 = item_tgt['tone']
                                if isinstance(tone0, torch.Tensor):
                                    zero = torch.zeros_like(tone0[:1])
                                    item_tgt['tone'] = torch.cat([tone0[:insert_idx0], zero, tone0[insert_idx0:]], dim=0)
                                else:
                                    tone_np = np.asarray(tone0)
                                    zero = np.zeros((1,), dtype=tone_np.dtype)
                                    item_tgt['tone'] = np.concatenate([tone_np[:insert_idx0], zero, tone_np[insert_idx0:]], axis=0)
                        elif did_sfx and (not sfx_at_tgt_head):
                            if isinstance(base_ph, torch.Tensor):
                                base_ph = base_ph.long()
                                last_id = int(base_ph[-1]) if base_ph.numel()>0 else -1
                                item_tgt['ph_token'] = base_ph if last_id == SIL_TOKEN_ID \
                                    else torch.cat([base_ph, torch.tensor([SIL_TOKEN_ID], dtype=base_ph.dtype, device=base_ph.device)], dim=0)
                                if 'tone' in item_tgt and item_tgt['tone'] is not None:
                                    tone0 = item_tgt['tone']
                                    zero = torch.zeros_like(tone0[:1])
                                    item_tgt['tone'] = torch.cat([tone0, zero], dim=0)
                            else:
                                base_np = np.array(base_ph, dtype=np.int64)
                                last_id = int(base_np[-1]) if base_np.size>0 else -1
                                item_tgt['ph_token'] = base_np if last_id == SIL_TOKEN_ID \
                                    else np.concatenate([base_np, np.array([SIL_TOKEN_ID], dtype=np.int64)], axis=0)
                                if 'tone' in item_tgt and item_tgt['tone'] is not None:
                                    tone0 = np.asarray(item_tgt['tone'])
                                    zero = np.zeros((1,), dtype=tone0.dtype)
                                    item_tgt['tone'] = np.concatenate([tone0, zero], axis=0)
                        else:
                            item_tgt['ph_token'] = base_ph

                        if hparams.get('use_merged_ph', False):
                            try:
                                mpt = map_phone_to_tokendict(
                                    {'txt_token': item_tgt['txt_token'], 'tone': item_tgt.get('tone', None)},
                                    pad_bos_eos=False
                                )
                                if did_sfx and sfx_at_tgt_head:
                                    insert_idx0 = max(0, int(insert_tok_idx) - 1)
                                    if isinstance(mpt, torch.Tensor):
                                        if int(mpt[insert_idx0]) != SIL_TOKEN_ID:
                                            item_tgt['merged_ph_token'] = torch.cat(
                                                [mpt[:insert_idx0],
                                                 torch.tensor([SIL_TOKEN_ID], dtype=mpt.dtype, device=mpt.device),
                                                 mpt[insert_idx0:]], dim=0)
                                        else:
                                            item_tgt['merged_ph_token'] = mpt
                                    else:
                                        mpt_np = np.asarray(mpt)
                                        item_tgt['merged_ph_token'] = np.concatenate(
                                            [mpt_np[:insert_idx0], np.array([SIL_TOKEN_ID], dtype=np.int64), mpt_np[insert_idx0:]],
                                            axis=0
                                        )
                                elif did_sfx and (not sfx_at_tgt_head):
                                    if isinstance(mpt, torch.Tensor):
                                        if int(mpt[-1]) != SIL_TOKEN_ID:
                                            item_tgt['merged_ph_token'] = torch.cat(
                                                [mpt, torch.tensor([SIL_TOKEN_ID], dtype=mpt.dtype, device=mpt.device)], dim=0)
                                        else:
                                            item_tgt['merged_ph_token'] = mpt
                                    else:
                                        mpt_np = np.asarray(mpt)
                                        if int(mpt_np[-1]) != SIL_TOKEN_ID:
                                            item_tgt['merged_ph_token'] = np.concatenate(
                                                [mpt_np, np.array([SIL_TOKEN_ID], dtype=np.int64)], axis=0)
                                        else:
                                            item_tgt['merged_ph_token'] = mpt_np
                                else:
                                    item_tgt['merged_ph_token'] = mpt

                                if item_tgt['merged_ph_token'].shape[0] != item_tgt['dur'].shape[0]:
                                    raise RuntimeError("merged_ph_token len mismatch with dur")
                            except Exception as e:
                                raise RuntimeError(f"map_phone_to_tokendict failed: {e}")

                        def _mk_caption_for_bgm(text_body, cap):
                            base = _wrap_s1_if_nonempty(text_body)
                            return (base + (" " + _ensure_trailing_punct(f"<BGM>{cap}</BGM>") if cap else "")).strip()

                        if did_sfx:
                            if sfx_at_tgt_head:
                                if not ref_text_part and not tgt_text_part:
                                    item_tgt['caption'] = (
                                        _ensure_trailing_punct(f"<Audio>{sfx_cap}</Audio>")
                                        + " " + _wrap_s1_if_nonempty(txt_norm)
                                    ).strip()
                                else:
                                    item_tgt['caption'] = " ".join([
                                        _wrap_s1_if_nonempty(ref_text_part),
                                        _ensure_trailing_punct(f"<Audio>{sfx_cap}</Audio>"),
                                        _wrap_s1_if_nonempty(tgt_text_part),
                                    ]).strip()
                            else:
                                item_tgt['caption'] = (
                                    _wrap_s1_if_nonempty(txt_norm) + " "
                                    + _ensure_trailing_punct(f"<Audio>{sfx_cap}</Audio>")
                                ).strip()
                            item_tgt['caption_audio'] = sfx_cap
                        elif did_bgm:
                            item_tgt['caption'] = _mk_caption_for_bgm(txt_norm, bgm_cap)
                            item_tgt['caption_audio'] = bgm_cap
                        else:
                            item_tgt['caption'] = _wrap_s1_if_nonempty(txt_norm)
                            item_tgt['caption_audio'] = None

                        ctx_mask = torch.zeros((len(item_tgt['wav'])//hop, 1))
                        ctx_mask[:rand_length] = 1.0
                        item_tgt['ctx_mask'] = ctx_mask[::vae_stride]
                        item_tgt['ctx_wav']  = deepcopy(wav_ref)[:rand_length*hop]

                        mel_len_all2 = len(item_tgt['wav']) // hop
                        mask_mel = torch.zeros((mel_len_all2, 1))
                        if did_bgm:
                            mask_mel[rand_length:mel_len_all2] = 1.0
                        elif did_sfx:
                            if sfx_at_tgt_head:
                                sfx_st = min(rand_length, mel_len_all2)
                                sfx_ed = min(rand_length + sfx_frames_added, mel_len_all2)
                                if sfx_ed > sfx_st:
                                    mask_mel[sfx_st:sfx_ed] = 1.0
                            else:
                                sfx_start = min(mel_len_ref, mel_len_all2)
                                sfx_end   = mel_len_all2
                                if sfx_end > sfx_start:
                                    mask_mel[sfx_start:sfx_end] = 1.0
                        item_tgt['audio_mask'] = mask_mel[::vae_stride]
                        item_tgt['vad_mask'] = None

                    # ---------- 写出 ----------
                    item_tgt_ = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in item_tgt.items()}
                    item_tgt_ = deepcopy(item_tgt_)

                    add_spk_id = 1
                    batch = batcher.collate_batch(item_tgt_)
                    if batch is not None and len(batch) > 0:
                        save_samples_to_shm(batch, cnt, shm_base)
                        add_spk_id = 0
                        restart_countdown -= 1
                        if restart_countdown == 0: return
                        samples, tgt_size, cnt, spk_id, nframes, rng_main = init_new_samples()
                    # 否则继续累积
                except Exception as e:
                    n_skip += 1
                    body_dbg = locals().get('_body_for_norm', None)
                    print(f'processer_fn_dit_wav_audio#{i_worker}/{n_worker} ERROR: {repr(e)} | '
                          f'txt={repr(item_tgt.get("txt", None))} | body={repr(body_dbg)} | skipped [{n_skip}] items.')
                finally:
                    spk_id += add_spk_id

    except Exception:
        traceback.print_exc()



def _get_first_with_src(d, aliases, record_counter: Counter, missing_samples_list, max_collect=3):
    for k in aliases:
        if k in d and d[k] is not None:
            record_counter[k] += 1
            return d[k], k
    record_counter['<miss>'] += 1
    if len(missing_samples_list) < max_collect:
        missing_samples_list.append(sorted(list(d.keys())))
    return None, None

def _get_first_with_src(d, aliases, record_counter: Counter, missing_samples_list, max_collect=3):
    for k in aliases:
        if k in d and d[k] is not None:
            record_counter[k] += 1
            return d[k], k
    record_counter['<miss>'] += 1
    if len(missing_samples_list) < max_collect:
        missing_samples_list.append(sorted(list(d.keys())))
    return None, None

def processer_fn_dit_wav_text_multispk_emb(
    data_paths, q_to_pull, reader_chunk_size, world_size, shm_base, counter, hparams_,
    seed, i_worker, n_worker, reader_cache_name='cache'
):
    setproctitle.setproctitle('data_processer:processer_fn')
    hparams.update(hparams_)

    hop = hparams['hop_size']
    stride = hparams.get('vae_stride', 8)
    fm = hparams['frames_multiple']
    fm_wav = fm * hop
    min_frames = hparams['min_frames']
    max_tokens = hparams.get('max_tokens', 40000)
    prefetch_steps = hparams.get('prefetch_steps', 200)
    add_vad_mask = hparams.get('add_vad_mask', False)
    use_sparse_dur = hparams.get('use_sparse_dur', False)
    use_merged_ph = hparams.get('use_merged_ph', False)
    use_ph_timestamp = hparams.get('use_ph_timestamp', False)
    min_spk_num = hparams.get('min_spk_num', 2)
    max_spk_num = hparams.get('max_spk_num', 2)
    # === CHANGED: 至少做 2 次跨 chunk 混读，除非你显式给得更大
    extra_random_reads = max(int(hparams.get('extra_random_reads', 0)), 2)  # 保底 2

    reader = get_reader(data_paths, reader_chunk_size, i_worker, n_worker, reader_cache_name)
    ds_len = reader[1]
    g = torch.Generator(); g.manual_seed(i_worker % 1001 + seed)
    num_chunks = max(1, ds_len // reader_chunk_size + 1)
    indices = torch.randperm(num_chunks, generator=g).tolist()
    other_idx = 0

    batcher = BucketBatcher(
        buckets=[100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1100,
                 1200,1300,1400,1500,1600,1700,1800,1900,2000,3000,4000,5000,6000,7000,8000,9000,10000,
                 11000,12000,13000,14000,15000,16000,18000,20000,40000,60000],
        dynamic_batch=True,
        maximum_bucket_size=max_tokens,
        length_fn=lambda x: x['wav'].shape[0] // hop // stride,
        bsz_evaluator=None,
    )

    def wait_shm_budget():
        while len(glob.glob(f'{shm_base}/*.pkl')) >= prefetch_steps * world_size:
            time.sleep(1)

    def new_counter_and_tgt():
        with counter.get_lock():
            cnt = counter.value; counter.value += 1
        random.seed((cnt // world_size) % 1001 + seed)
        tgt_size = random.randint(hparams['tgt_size_min'], hparams['tgt_size_max'])
        return cnt, tgt_size

    def flush_one(item_tgt, cnt_):
        item_np = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in item_tgt.items()}
        batch = batcher.collate_batch(item_np)
        if batch is not None and len(batch) > 0:
            save_samples_to_shm(batch, cnt_, shm_base)
            return True
        return False

    restart_countdown = 10000
    wait_shm_budget()
    cnt, tgt_size = new_counter_and_tgt()

    while True:
        items_bytes = read_items(q_to_pull, reader)
        if items_bytes is None:
            return

        # === CHANGED: 无条件做多次“跨 chunk 混读”（次数由 extra_random_reads 控制，默认≥2）
        for _ in range(extra_random_reads):
            item_id_start = indices[other_idx] * reader_chunk_size
            other_idx = (other_idx + 1) % len(indices)
            try:
                items_bytes += [x for x in reader[0].read_many([item_id_start])[0]]
            except Exception:
                pass  # 容错

        # 标准化：不在此处合并，不抽 mel
        items_std = merge_item_bytes(
            items_bytes,
            load_mel=False,
            merge_same_spk=False,
            exclude_spk=hparams.get('exclude_spk'),
            tgt_size=None
        )

        # 期望字段：wav, txt_token(=phone_encoded), tone(=tone_encoded), mel2ph, spk_name, item_name, txt(=txt_raw)
        items = []
        spk2idx = {}
        for it in items_std:
            if ('wav' not in it) or ('txt_token' not in it) or ('mel2ph' not in it) or ('tone' not in it) or ('txt' not in it):
                continue
            wav = it['wav'] if isinstance(it['wav'], torch.Tensor) else torch.as_tensor(it['wav'], dtype=torch.float32)
            frames = wav.shape[0] // hop
            if frames < min_frames:
                continue
            text = raw_text_process(it['txt'], wav)
            if text is None:
                continue
            ph = it['txt_token'] if isinstance(it['txt_token'], torch.Tensor) else torch.as_tensor(it['txt_token'], dtype=torch.long)
            tn = it['tone']      if isinstance(it['tone'], torch.Tensor)      else torch.as_tensor(it['tone'], dtype=torch.long)
            m2p= it['mel2ph']    if isinstance(it['mel2ph'], torch.Tensor)    else torch.as_tensor(it['mel2ph'], dtype=torch.long)
            if tn.shape[0] != ph.shape[0]:
                tn = torch.zeros_like(ph, dtype=torch.long)

            items.append({
                'item_name': it['item_name'],
                'spk_name':  it['spk_name'],  # 如果有 subset，可在此处拼接 subset 以去歧义
                'wav': wav,
                'frames': frames,
                'text': text,
                'txt_token': ph.long(),
                'tone': tn.long(),
                'mel2ph': m2p.long(),
            })
            spk2idx.setdefault(it['spk_name'], []).append(len(items) - 1)

        if not items:
            continue

        # === CHANGED: 不要在这里“预先剔除单例说话人”！保留它们参与首轮组块
        singles = {}  # 留空，后续在组完块之后再把只剩 0/1 条的说话人下放

        blocks, rest_idxs = [], []
        while len(spk2idx) >= min_spk_num:
            cur_spk_num = min(len(spk2idx), random.randint(min_spk_num, max_spk_num))
            spk_names = random.sample(list(spk2idx.keys()), cur_spk_num)

            picks, total_f = [], 0
            # 先每个说话人拿一条，确保真正的“多说话人”
            for sn in spk_names:
                if spk2idx[sn]:
                    idx = spk2idx[sn].pop()
                    picks.append(idx)
                    total_f += items[idx]['frames']
            if len(picks) <= 1:
                for x in picks: rest_idxs.append(x)
                break

            # 轮转追加，直到接近目标长度
            turn = 0
            while True:
                sn = spk_names[turn % cur_spk_num]; turn += 1
                if not spk2idx[sn]:
                    if all(len(spk2idx[x]) == 0 for x in spk_names):
                        break
                    continue
                nxt = spk2idx[sn][-1]
                if total_f + items[nxt]['frames'] > tgt_size:
                    break
                picks.append(spk2idx[sn].pop())
                total_f += items[picks[-1]]['frames']

            blocks.append((picks, cur_spk_num))

            # === CHANGED: 现在才把只剩 0/1 条的说话人转移到 singles，避免下一轮死循环
            for sn in list(spk_names):
                if len(spk2idx.get(sn, [])) <= 1:
                    if len(spk2idx.get(sn, [])) == 1:
                        singles[sn] = spk2idx[sn]
                    spk2idx.pop(sn, None)

        # 剩余样本下放
        for sn, lst in spk2idx.items(): rest_idxs.extend(lst)
        for sn, lst in singles.items(): rest_idxs.extend(lst)

        # 合并样本（多说话人）
        for picks, cur_spk_num in blocks:
            wav = torch.cat([items[i]['wav'] for i in picks], dim=0)
            if wav.numel() < min_frames * hop:
                continue
            wav = wav[: (wav.shape[0] // fm_wav) * fm_wav]

            # === CHANGED: 统一生成 text/caption：每段使用 <S{sid}>...</S{sid}> 包裹
            seg_parts = []
            spk_map = {}      # name -> sid (1-based)
            ctx_wavs = []
            pick_sids = []    # 记录每个 pick 的 sid（供后面的 phone-level spk_mask 用）
            for j, idx in enumerate(picks):
                if j < cur_spk_num:
                    ctx_wavs.append(items[idx]['wav'])
                sn = items[idx]['spk_name']
                sid = spk_map.setdefault(sn, len(spk_map) + 1)  # 1-based，与原下游兼容
                pick_sids.append(sid)
                txt = items[idx]['text']
                seg_parts.append(f'<S{sid}>{txt}</S{sid}>')

            text_merged    = ''.join(seg_parts)
            caption_merged = text_merged  # 与 text 完全一致

            if len(picks) > cur_spk_num:
                ctx_cat = torch.cat(ctx_wavs, dim=0) if len(ctx_wavs) > 0 else wav[:0]
                ref_wav_start = (ctx_cat.shape[0] // fm_wav) * fm_wav
            else:
                if len(ctx_wavs) > 1:
                    pre = torch.cat(ctx_wavs[:-1], dim=0)
                    ref_wav_start = max(int(pre.shape[0] + ctx_wavs[-1].shape[0] * 0.1), pre.shape[0] + 20000)
                else:
                    ref_wav_start = 20000
                ref_wav_start = (ref_wav_start // fm_wav) * fm_wav
            max_idx = min(int(wav.shape[0] * 0.7), wav.shape[0] - 20000)
            if max_idx > ref_wav_start:
                ref_wav_start = (random.randint(ref_wav_start, max_idx) // fm_wav) * fm_wav

            ctx_wav = wav[:ref_wav_start]
            ctx_mask = torch.zeros((wav.shape[0], 1), dtype=torch.float32)
            ctx_mask[:ref_wav_start] = 1.0
            ctx_mask = ctx_mask[:: hop * stride]

            # phones/tone/m2p 拼接与对齐（保持原逻辑）+【新增】phone-level spk_mask
            ph_list, tone_list, m2p_list = [], [], []
            spk_mask_ph_list = []  # 新增：与 ph 对齐的 spk mask 片段
            ph_offset = 0
            for idx, sid in zip(picks, pick_sids):
                ph_i = items[idx]['txt_token'].long()
                tn_i = items[idx]['tone'].long()
                m2p_i= items[idx]['mel2ph'].long()
                if ph_offset > 0:
                    m2p_i = m2p_i + (m2p_i > 0).long() * ph_offset
                ph_list.append(ph_i); tone_list.append(tn_i); m2p_list.append(m2p_i)

                # === 新增：phone-level spk mask（1-based）
                spk_mask_ph_list.append(torch.full((ph_i.numel(),), int(sid), dtype=torch.long))

                ph_offset += ph_i.numel()

            ph_cat   = torch.cat(ph_list,   dim=0) if len(ph_list)   > 0 else torch.zeros(0, dtype=torch.long)
            tone_cat = torch.cat(tone_list, dim=0) if len(tone_list) > 0 else torch.zeros(0, dtype=torch.long)
            m2p_cat  = torch.cat(m2p_list,  dim=0) if len(m2p_list)  > 0 else torch.zeros(0, dtype=torch.long)
            spk_ph_mask = torch.cat(spk_mask_ph_list, dim=0) if len(spk_mask_ph_list) > 0 else torch.zeros(0, dtype=torch.long)

            mel_len = wav.shape[0] // hop
            if m2p_cat.numel() < mel_len:
                pad_len = mel_len - m2p_cat.numel()
                last = m2p_cat[-1] if m2p_cat.numel() > 0 else torch.tensor(0, dtype=torch.long)
                m2p_cat = torch.cat([m2p_cat, last.repeat(pad_len)], dim=0)
            m2p_cat = m2p_cat[:mel_len]
            m2p_cat = m2p_cat[: (mel_len // fm) * fm]

            sample = {
                'id': 0,
                'item_name': '|||'.join([items[i]['item_name'] for i in picks]),
                'wav': wav,
                'text': text_merged,            # 统一格式
                'caption': caption_merged,      # 统一格式
                # === CHANGED: spk_mask 改为与 ph_token 等长（phone-level, 1-based）
                'spk_mask': np.asarray(spk_ph_mask.numpy(), dtype=np.int16),
                'ctx_wav': ctx_wav,
                'ctx_mask': ctx_mask,
                'ph_token': ph_cat,
                'tone': tone_cat,
                'mel2ph': m2p_cat,
                'dur': mel2token_to_dur(m2p_cat),
                'vad_mask': None
            }

            if use_sparse_dur:
                sample['mel2ph_sparse'] = compute_mel2aug_from_dur(
                    sample['dur'].numpy().tolist(),
                    gap_mode=hparams.get('sparse_dur_mode', 'proportional'),
                    gap_frames=hparams.get('sparse_dur_frames', 4),
                    gap_alpha=hparams.get('sparse_dur_alpha', 0.2),
                    min_keep=hparams.get('sparse_dur_min_keep', 1),
                    keep_ratio=hparams.get('sparse_dur_keep_ratio'),
                    symmetric=hparams.get('sparse_dur_symmetric', True),
                )
            if use_merged_ph:
                sample['merged_ph_token'] = map_phone_to_tokendict(
                    {'txt_token': ph_cat, 'tone': tone_cat}, pad_bos_eos=False
                )
                if 'dur' in hparams['task_cls'] and sample['merged_ph_token'].shape[0] != sample['dur'].shape[0]:
                    continue
            if use_ph_timestamp:
                try:
                    sample['ph_timestamp'] = FrontendLMDataset.get_ph_timestamp(sample)
                except Exception:
                    pass
            if add_vad_mask:
                vad_start, vad_end = get_vad_mask(sample['wav'])
                vm = hop * stride
                vad_mask = np.zeros((sample['wav'].shape[0] // vm))
                vad_mask[int(vad_start * hparams['audio_sample_rate'] // vm): int(vad_end * hparams['audio_sample_rate'] // vm)] = 1
                sample['vad_mask'] = vad_mask

            if flush_one(sample, cnt):
                restart_countdown -= 1
                if restart_countdown == 0:
                    return
                wait_shm_budget()
                cnt, tgt_size = new_counter_and_tgt()

        # 单段样本（仅将 spk_mask 改为与 ph 对齐，其他不变）
        for idx in rest_idxs:
            it = items[idx]
            wav = it['wav']
            if wav.numel() < min_frames * hop:
                continue
            wav = wav[: (wav.shape[0] // fm_wav) * fm_wav]

            m2p = it['mel2ph'].long()
            mel_len = wav.shape[0] // hop
            if m2p.numel() < mel_len:
                pad_len = mel_len - m2p.numel()
                last = m2p[-1] if m2p.numel() > 0 else torch.tensor(0, dtype=torch.long)
                m2p = torch.cat([m2p, last.repeat(pad_len)], dim=0)
            m2p = m2p[:mel_len]
            m2p = m2p[: (mel_len // fm) * fm]

            text_single = it['text']
            ph_single = it['txt_token'].long()
            # === CHANGED: 单说话人样本的 spk_mask 也与 ph 等长，且全部为 1（1-based）
            spk_mask_single = torch.ones_like(ph_single, dtype=torch.long)

            # === CHANGED: 单说话人样本的 text/caption 统一为 <S1>...</S1>
            tc = f'<S1>{text_single}</S1>'

            sample = {
                'id': 0,
                'item_name': it['item_name'],
                'wav': wav,
                'text': tc,
                'caption': tc,
                'spk_mask': np.asarray(spk_mask_single.numpy(), dtype=np.int16),
                'ph_token': ph_single,
                'tone': it['tone'].long(),
                'mel2ph': m2p,
                'dur': mel2token_to_dur(m2p),
            }
            
            if use_sparse_dur:
                sample['mel2ph_sparse'] = compute_mel2aug_from_dur(
                    sample['dur'].numpy().tolist(),
                    gap_mode=hparams.get('sparse_dur_mode', 'proportional'),
                    gap_frames=hparams.get('sparse_dur_frames', 4),
                    gap_alpha=hparams.get('sparse_dur_alpha', 0.2),
                    min_keep=hparams.get('sparse_dur_min_keep', 1),
                    keep_ratio=hparams.get('sparse_dur_keep_ratio'),
                    symmetric=hparams.get('sparse_dur_symmetric', True),
                )

            # === INSERT START: 方案A - 单段样本也生成 merged_ph_token，与多说话人分支一致 ===
            if use_merged_ph:
                merged = map_phone_to_tokendict(
                    {'txt_token': ph_single, 'tone': it['tone'].long()},
                    pad_bos_eos=False
                )
                # 与多说话人分支保持一致的长度校验；不一致直接丢弃该样本
                if 'dur' in hparams['task_cls'] and merged.shape[0] != sample['dur'].shape[0]:
                    continue
                sample['merged_ph_token'] = merged
            # === INSERT END ===

            min_idx = max(int(wav.shape[0] * 0.1), 20000)
            max_idx = min(int(wav.shape[0] * 0.9), wav.shape[0] - 20000)
            if min_idx > max_idx:
                min_idx = int(wav.shape[0] * 0.4); max_idx = int(wav.shape[0] * 0.6)
            ref_wav_start = (random.randint(min_idx, max_idx) // fm_wav) * fm_wav

            ctx_wav = wav[:ref_wav_start]
            ctx_mask = torch.zeros((wav.shape[0], 1), dtype=torch.float32)
            ctx_mask[:ref_wav_start] = 1.0
            ctx_mask = ctx_mask[:: hop * stride]
            sample['ctx_wav'] = ctx_wav
            sample['ctx_mask'] = ctx_mask

            if add_vad_mask:
                vad_start, vad_end = get_vad_mask(sample['wav'])
                vm = hop * stride
                vad_mask = np.zeros((sample['wav'].shape[0] // vm))
                vad_mask[int(vad_start * hparams['audio_sample_rate'] // vm): int(vad_end * hparams['audio_sample_rate'] // vm)] = 1
                sample['vad_mask'] = vad_mask
            else:
                sample['vad_mask'] = None

            if flush_one(sample, cnt):
                restart_countdown -= 1
                if restart_countdown == 0:
                    return
                wait_shm_budget()
                cnt, tgt_size = new_counter_and_tgt()


def merge_item_bytes_asr(items_bytes, load_mel=True, exclude_spk=None, tgt_size=6000, vad_model=None):

    def merge_samples(samples):
        def _merge_samples(samples):
            sample_merged = {
                'id': 0,
                'item_name': '|||'.join([s['item_name'] for s in samples]),
                'txt_token': torch.cat([s['txt_token'] for s in samples], 0),
                'wav': torch.cat([s['wav'] for s in samples], 0),
                'mel2ph': merge_A2B(
                    [s['mel2ph'] for s in samples], [len(s['txt_token']) for s in samples]),
                'txt': ' '.join([s['txt'] for s in samples])
            }
            if load_mel:
                sample_merged['mel'] = torch.cat([s['mel'] for s in samples], 0)
            # normalize wav
            try:
                from utils.audio.transform import normalize_lufs
                sample_merged['wav'] = normalize_lufs(
                    sample_merged['wav'],
                    sr=hparams['audio_sample_rate'],
                    target_lufs=target_lufs + np.random.randn() / 10
                )
            except Exception as err:
                pass
            return sample_merged
        
        sr = hparams['audio_sample_rate']
        hop_size = hparams['hop_size']
        
        if hparams.get('transcription_form', 'plain_text') == 'plain_text':
            return _merge_samples(samples)
        elif hparams.get('transcription_form', 'plain_text') == 'dialogue':
            spk_lst = []
            spk_map = {}
            last_spk = None
            samples_merged_ = []
            samples_to_merge = []
            for sample in samples:
                if sample['spk_name'] != last_spk and len(samples_to_merge) > 0:
                    samples_merged_.append(_merge_samples(samples_to_merge))
                    samples_to_merge = []
                    if last_spk not in spk_map:
                        spk_map[last_spk] = len(spk_map)
                    spk_lst.append(spk_map[last_spk])
                samples_to_merge.append(sample)
                last_spk = sample['spk_name']
            if len(samples_to_merge) > 0:
                samples_merged_.append(_merge_samples(samples_to_merge))
                if last_spk not in spk_map:
                    spk_map[last_spk] = len(spk_map)
                spk_lst.append(spk_map[last_spk])
            
            sample_merged = {
                'id': 0,
                'item_name': samples_merged_[0]['item_name'],
                'wav': samples_merged_[0]['wav'],
                'txt': f'<SPK>{spk_lst[0]}</SPK>' + samples_merged_[0]['txt'],
            }
            if load_mel:
                sample_merged['mel'] = samples_merged_[0]['mel']
            max_spk_num = hparams.get('max_spk_num', 128)
            sample_merged['spk_mask'] = torch.zeros((samples_merged_[0]['wav'].shape[-1], max_spk_num)).int()
            if hparams.get('use_vad', False):
                from utils.audio.vad import run_vad_trim
                vad_start, vad_end = run_vad_trim(samples_merged_[0]['wav'], hparams['audio_sample_rate'], vad_model)
                if vad_start == 0 and vad_end == 0:
                    vad_start, vad_end = run_vad_trim(samples_merged_[0]['wav'], hparams['audio_sample_rate'], vad_model, threshold=0.3)
                if vad_start == 0 and vad_end == 0:
                    vad_start = 0
                    vad_end = samples_merged_[0]['wav'].shape[0] / hparams['audio_sample_rate']
                voiced_start = int(vad_start * hparams['audio_sample_rate'])
                voiced_end = int(vad_end * hparams['audio_sample_rate'])
            else:
                durs = mel2token_to_dur(samples_merged_[0]['mel2ph'])
                if samples_merged_[0]['txt_token'][0] != 145:
                    voiced_start = 0
                else:
                    voiced_start = durs[0] * hparams['hop_size']
                voiced_end = min(samples_merged_[0]['wav'].shape[0], durs.sum() * hparams['hop_size'])
            sample_merged['spk_mask'][voiced_start: voiced_end, spk_lst[0]] = 1

            for i in range(1, len(samples_merged_)):
                sample_merged['item_name'] = sample_merged['item_name'] + '|||' + samples_merged_[i]['item_name']
                if spk_lst[i] == spk_lst[i-1]:  # don't know why, but this happens
                    sample_merged['txt'] = sample_merged['txt'] + samples_merged_[i]['txt']
                    direct_concat = True
                else:
                    sample_merged['txt'] = sample_merged['txt'] + f'<SPK>{spk_lst[i]}</SPK>' + samples_merged_[i]['txt']
                    direct_concat = False
                if hparams.get('use_vad', False):
                    from utils.audio.vad import run_vad_trim
                    vad_start, vad_end = run_vad_trim(samples_merged_[i]['wav'], hparams['audio_sample_rate'], vad_model)
                    if vad_start == 0 and vad_end == 0:
                        vad_start, vad_end = run_vad_trim(samples_merged_[i]['wav'], hparams['audio_sample_rate'], vad_model, threshold=0.3)
                    if vad_start == 0 and vad_end == 0:
                        vad_start = 0
                        vad_end = samples_merged_[i]['wav'].shape[0] / hparams['audio_sample_rate']
                    voiced_start = int(vad_start * hparams['audio_sample_rate'])
                    voiced_end = int(vad_end * hparams['audio_sample_rate'])
                else:
                    durs = mel2token_to_dur(samples_merged_[i]['mel2ph'])
                    if samples_merged_[i]['txt_token'][0] != 145:
                        voiced_start = 0
                    else:
                        voiced_start = durs[0] * hparams['hop_size']
                    voiced_end = min(samples_merged_[i]['wav'].shape[0], durs.sum() * hparams['hop_size'])

                if random.random() < hparams.get('wav_concat_overlap_prob', 0.05) and not direct_concat:
                    overlap_duration = (np.random.rand() * 
                                        (hparams.get('wav_concat_overlap_duration_max', 2) - hparams.get('wav_concat_overlap_duration_min', 0.5)) + 
                                        hparams.get('wav_concat_overlap_duration_min', 0.5))
                    overlap_duration = min(sample_merged['wav'].shape[0] / sr * 0.5, samples_merged_[i]['wav'].shape[0] / sr * 0.5, overlap_duration)
                    overlap_frames = int(overlap_duration * hparams['audio_sample_rate'])
                    sample_merged['wav'][-overlap_frames:] += samples_merged_[i]['wav'][:overlap_frames]
                    sample_merged['wav'] = torch.cat([sample_merged['wav'], samples_merged_[i]['wav'][overlap_frames:]])

                    # FIXME: mel cannot be linearly added like this, but who cares
                    if load_mel:
                        overlap_frames_mel = int(min(sample_merged['mel'].shape[0] * 0.5, samples_merged_[i]['mel'].shape[0] * 0.5, overlap_frames//hparams['hop_size']))
                        sample_merged['mel'][-overlap_frames_mel:] += samples_merged_[i]['mel'][:overlap_frames_mel]
                        sample_merged['mel'] = torch.cat([sample_merged['mel'], samples_merged_[i]['mel'][overlap_frames//hparams['hop_size']:]])

                    spk_mask_start = sample_merged['spk_mask'].shape[0] - overlap_frames
                    sample_merged['spk_mask'] = torch.cat([sample_merged['spk_mask'], torch.zeros((samples_merged_[i]['wav'].shape[0] - overlap_frames, max_spk_num)).int()], dim=0)
                    sample_merged['spk_mask'][spk_mask_start + voiced_start: spk_mask_start + voiced_end, spk_lst[i]] = 1
                else:
                    if random.random() < hparams.get('wav_concat_pause_prob', 0.05) and not direct_concat:
                        pause_duration = (np.random.rand() * 
                                          (hparams.get('wav_concat_pause_duration_max', 2) - hparams.get('wav_concat_pause_duration_min', 0.5)) + 
                                          hparams.get('wav_concat_pause_duration_min', 0.5))
                        pause_frames = int(pause_duration * hparams['audio_sample_rate'])
                        sample_merged['wav'] = torch.cat([sample_merged['wav'], torch.zeros(pause_frames).to(sample_merged['wav'])])

                        if load_mel:
                            pause_frames_mel = int(min(sample_merged['mel'].shape[0] * 0.5, samples_merged_[i]['mel'].shape[0] * 0.5, pause_frames//hparams['hop_size']))
                            sample_merged['mel'] = torch.cat([sample_merged['mel'], torch.zeros((pause_frames_mel, sample_merged['mel'].shape[1])).to(sample_merged['mel'])])
                        
                        sample_merged['spk_mask'] = torch.cat([sample_merged['spk_mask'], torch.zeros((pause_frames, max_spk_num)).int()], dim=0)
                    
                    sample_merged['wav'] = torch.cat([sample_merged['wav'], samples_merged_[i]['wav']])
                    if load_mel:
                        sample_merged['mel'] = torch.cat([sample_merged['mel'], samples_merged_[i]['mel']])

                    spk_mask_start = sample_merged['spk_mask'].shape[0]
                    sample_merged['spk_mask'] = torch.cat([sample_merged['spk_mask'], torch.zeros((samples_merged_[i]['wav'].shape[0], max_spk_num)).int()], dim=0)
                    sample_merged['spk_mask'][spk_mask_start + voiced_start: spk_mask_start + voiced_end, spk_lst[i]] = 1

            return sample_merged


    def merge_A2B(A2B, B_lens):
        token_lens_cumsum = np.cumsum([0] + B_lens[:-1])
        token_lens_cumsum = torch.LongTensor(token_lens_cumsum)
        for i in range(len(B_lens)):
            A2B[i] = A2B[i] + token_lens_cumsum[i]
        A2B = torch.cat(A2B, 0)
        return A2B

    hdfs_client = None

    items = []
    target_lufs = np.random.uniform(-21, -16)
    for item_ in items_bytes:
        item_ = pickle.loads(item_)

        if 'wav' not in item_ or item_['wav'] is None:
            try:
                data_url = item_['data_url']
                if data_url.startswith('hdfs://'):
                    if hdfs_client is None:
                        from utils.commons.hdfs_utils import HDFSClient
                        hdfs_client = HDFSClient()
                    data = pickle.loads(hdfs_client.get_object(data_url))
                    item_['wav'] = data['wav']
            except:
                continue

        item = {}
        try:
            item['txt_token'] = torch.LongTensor(item_['phone_encoded'])
        except:
            continue

        if load_mel:
            try:
                mel, wav = MegaTTSDataset.get_mel(hparams, item_['wav'])
                item['mel'] = torch.FloatTensor(mel)
                item['wav'] = torch.FloatTensor(wav)
            except:
                continue
            if len(item['mel']) < hparams['min_frames']:
                continue
            ph_div_mel = len(item['txt_token']) / len(item['mel'])
            if ph_div_mel > 0.3 or ph_div_mel < 0.01:
                continue
        else:
            item['wav'] = torch.FloatTensor(item_['wav'])
        item['mel2ph'] = torch.LongTensor(item_['mel2ph'])
        item['spk_name'] = item_['spk_name']
        item['item_name'] = item_['item_name']
        item['txt'] = item_['txt_raw']

        # preprocess txt
        txt = raw_text_process(item['txt'], item['wav'])
        if txt is None:
            continue
        item['txt'] = txt

        if exclude_spk is not None and item['spk_name'] in exclude_spk:
            continue
        items.append(item)

    random.shuffle(items)
    
    items_merged = []
    total_frames = 0
    items_to_merge = []
    spk_set = set()
    for item in items:
        # print('item length', item['wav'].shape[0] // hparams['hop_size'], 'total_frames', total_frames)
        if len(items_to_merge) > 0:
            if (
                    (
                        tgt_size is not None and 
                        total_frames > 0 and 
                        total_frames + item['wav'].shape[0] // hparams['hop_size'] > tgt_size
                    ) or
                    len(spk_set) >= hparams.get('max_spk_num', 1000000)
                ):
                items_merged.append(merge_samples(items_to_merge))
                items_to_merge = []
                total_frames = 0
                spk_set = set()
        items_to_merge.append(item)
        total_frames += item['wav'].shape[0] // hparams['hop_size']
        spk_set.add(item['spk_name'])
        if random.random() < 0.01 and total_frames + item['wav'].shape[0] // hparams['hop_size'] < tgt_size:
            items_to_merge.append(item)     # repeat
            total_frames += item['wav'].shape[0] // hparams['hop_size']
    if len(items_to_merge) > 0:
        items_merged.append(merge_samples(items_to_merge))
    return items_merged


def processer_fn_causalasr(data_paths, q_to_pull, reader_chunk_size, world_size, shm_base, counter, hparams_,
                    seed, i_worker, n_worker, reader_cache_name='cache'):
    setproctitle.setproctitle(f'data_processer:processer_fn_causalasr#{i_worker}/{n_worker}')
    hparams.update(hparams_)
    print(f"| Started processer_fn_causalasr#{i_worker}/{n_worker}.")

    try:

        reader = get_reader(data_paths, reader_chunk_size, i_worker,
                            n_worker, reader_cache_name)
        ds_len = reader[1]
        g = torch.Generator()
        g.manual_seed(i_worker % 1001 + seed)
        indices = torch.randperm(ds_len // reader_chunk_size + 1, generator=g).tolist()
        other_idx = 0
        print(f"| init reader#{i_worker}/{n_worker}.")
        batcher = BucketBatcher(
            buckets=[200, 400, 600, 800, 1000, 1100, 
                        1200, 1400, 1600, 1800, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
                        11000, 12000, 13000, 14000, 15000, 16000, 18000, 20000, 40000, 60000, 80000],
            dynamic_batch=True,
            maximum_bucket_size=hparams.get('max_tokens', 40000),
            length_fn=lambda x: x['wav'].shape[0]//hparams['hop_size'],
            bsz_evaluator=None,
        )

        vad_model = None
        if hparams.get('use_vad', False):
            from utils.audio.vad import get_vad_model
            vad_model = get_vad_model()

        fm = hparams['frames_multiple']
        fm_wav = hparams['frames_multiple'] * hparams['hop_size']

        if hparams.get('audio_encoder_type', 'wavlm') == 'xlsr-53':
            from transformers import Wav2Vec2FeatureExtractor
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hparams.get('audio_encoder_ckpt', 'facebook/wav2vec2-large-xlsr-53'))

        speech_augmentor = None
        if hparams.get('wav_add_noise', False) or hparams.get('wav_add_effect', False):
            from tasks.tts.dataset_utils.augment import SpeechAugment
            speech_augmentor = SpeechAugment(
                hparams.get('wav_add_noise', False), hparams.get('wav_add_effect', False), hparams.get('musan_dir', None), 
                noise_prob=hparams.get('wav_add_noise_prob', 0.5), effect_prob=hparams.get('wav_add_effect_prob', 0.5), 
                noise_snr=(6.0, 20.0), with_speech=hparams.get('musan_with_speech', False)
            )
            # print('| Noise mixer initialized!')

        def init_new_samples():
            retry_num = 0
            while (cur_pkl_num := len(glob.glob(f'{shm_base}/*.pkl'))) >= hparams.get('prefetch_steps', 200) * world_size:
                if retry_num % 20 == 0:
                    cur_cnt_lst = [int(Path(p).stem) for p in glob.glob(f'{shm_base}/*.pkl')]
                    max_cur_cnt = max(cur_cnt_lst)
                    min_cur_cnt = min(cur_cnt_lst)
                    with counter.get_lock():
                        cnt = counter.value
                    # if cnt < (max_cur_cnt - min_cur_cnt) * 0.9 + min_cur_cnt:
                    #     break
                    if cnt < max_cur_cnt:
                        break
                    # print(f"processer_fn_causalasr#{i_worker}/{n_worker} waiting for shm to be released: {retry_num} seconds; cnt={cnt}, max_cur_cnt={max_cur_cnt}, min_cur_cnt={min_cur_cnt}, thrs={(max_cur_cnt - min_cur_cnt) * 0.3 + min_cur_cnt}")
                    if DEBUG:
                        print(f"processer_fn_causalasr#{i_worker}/{n_worker} waiting for shm to be released: {retry_num} seconds")
                time.sleep(1)
                retry_num += 1
            with counter.get_lock():
                cnt = counter.value
                counter.value += 1
            random.seed((cnt // world_size) % 1001 + seed)
            tgt_size = random.randint(hparams['tgt_size_min'], hparams['tgt_size_max'])
            samples = []
            spk_id = 0
            nframes = 0
            return samples, tgt_size, cnt, spk_id, nframes

        n_skip = 0
        samples, tgt_size, cnt, spk_id, nframes = init_new_samples()
        restart_countdown = 10000
        while True:
            try:
                # items1
                items_bytes = read_items(q_to_pull, reader)
            except:
                continue
            if items_bytes is None:
                print(f'processer_fn_causalasr#{i_worker}/{n_worker}: items_bytes is None')
                continue
            other_items_bytes = []
            # for _ in range(hparams.get('max_n_spk_diarization', 3) - 1):
            for _ in range(4):
                try:
                    # other items_bytes
                    item_id_start = indices[other_idx] * reader_chunk_size
                    other_idx = (other_idx + 1) % len(indices)
                    items_bytes2 = [x for x in reader[0].read_many([item_id_start])[0]]
                except:
                    items_bytes2 = []
                other_items_bytes.extend(items_bytes2)
            items_bytes = items_bytes + other_items_bytes

            # print(f'processer_fn_causalasr#{i_worker}/{n_worker} read {len(items_bytes)} items.')

            try:
                items_merged = merge_item_bytes_asr(items_bytes, load_mel=hparams.get('load_mel', True), exclude_spk=hparams.get('exclude_spk'), tgt_size=tgt_size, vad_model=vad_model)
            except:
                traceback.print_exc()
                print(f'processer_fn_causalasr#{i_worker}/{n_worker} merge_item_bytes_asr failed.')
                continue

            # print(f'processer_fn_causalasr#{i_worker}/{n_worker} merged into {len(items_merged)} items.')

            for item_merged in items_merged:
                add_spk_id = 0
                item_tgt = item_merged
                # print("len(item_tgt['mel'])", len(item_tgt['mel']))
                # if hparams['max_frames'] >= len(item_tgt['mel']) > hparams['min_frames']:
                if hparams['max_frames'] >= len(item_tgt['wav']) / hparams['hop_size'] > hparams['min_frames']:
                    item_tgt['wav'] = item_tgt['wav'][:len(item_tgt['wav']) // fm_wav * fm_wav]
                    if speech_augmentor is not None:
                        item_tgt['wav'] = speech_augmentor(item_tgt['wav'], hparams['audio_sample_rate'])
                    item_tgt['text'] = item_tgt['txt']
                    if 'mel' in item_tgt:
                        item_tgt['mel'] = item_tgt['mel'][:len(item_tgt['mel']) // fm * fm]
                    if 'mel2ph' in item_tgt:
                        item_tgt['mel2ph'] = item_tgt['mel2ph'][:len(item_tgt['mel2ph']) // fm * fm]
                    if 'spk_mask' in item_tgt:
                        item_tgt['spk_mask'] = item_tgt['spk_mask'][:len(item_tgt['spk_mask']) // fm_wav * fm_wav]

                    if hparams.get('audio_encoder_type', 'wavlm') in ['xlsr-53', 'wavlm-hf']:
                        wav_16k = librosa.resample(item_tgt['wav'].numpy(), orig_sr=hparams['audio_sample_rate'], target_sr=16000)
                        item_tgt['wav_w2v2'] = feature_extractor(wav_16k, return_tensors="pt", sampling_rate=16000).input_values[0]

                    item_tgt_ = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in item_tgt.items()}
                    item_tgt_ = deepcopy(item_tgt_)

                    add_spk_id = 1
                    batch = batcher.collate_batch(item_tgt_)
                    if batch is not None and len(batch) > 0:
                        save_samples_to_shm(batch, cnt, shm_base)
                        if DEBUG:
                            print(f'processer_fn_causalasr#{i_worker}/{n_worker} saved {shm_base}/{cnt}.pkl')
                        add_spk_id = 0
                        restart_countdown -= 1
                        if restart_countdown == 0:
                            print(f'processer_fn_causalasr#{i_worker}/{n_worker} restart countdown reached, exiting.')
                            return
                        samples, tgt_size, cnt, spk_id, nframes = init_new_samples()

                else:
                    n_skip += 1
                    if n_skip % 100 == 0:
                        print(f'processer_fn_causalasr#{i_worker}/{n_worker} skipped [{n_skip}] items.')

                gc.collect()

                spk_id += add_spk_id
                
    except:
        traceback.print_exc()


def processer_fn_spk_window(data_paths, q_to_pull, reader_chunk_size, world_size, shm_base, counter, hparams_,
                    seed, i_worker, n_worker, reader_cache_name='cache'):
    setproctitle.setproctitle('data_processer:processer_fn')
    hparams.update(hparams_)
    print(f"| Started processer_fn_spk_window#{i_worker}/{n_worker}.")

    try:
        reader = get_reader(data_paths, reader_chunk_size, i_worker,
                            n_worker, reader_cache_name)
        print(f"| init reader#{i_worker}/{n_worker}.")

        ds_len = reader[1]
        g = torch.Generator()
        g.manual_seed(i_worker % 1001 + seed)
        indices = torch.randperm(ds_len // reader_chunk_size + 1, generator=g).tolist()
        other_idx = 0

        speech_augmentor = None
        if hparams.get('wav_add_noise', False) or hparams.get('wav_add_effect', False):
            from tasks.tts.dataset_utils.augment import SpeechAugment
            speech_augmentor = SpeechAugment(
                hparams.get('wav_add_noise', False), hparams.get('wav_add_effect', False), hparams.get('musan_dir', None), 
                noise_prob=hparams.get('wav_add_noise_prob', 0.5), effect_prob=hparams.get('wav_add_effect_prob', 0.5), 
                noise_snr=(6.0, 20.0), with_speech=hparams.get('musan_with_speech', False)
            )

        def init_new_samples():
            retry_num = 0
            while len(glob.glob(f'{shm_base}/*.pkl')) >= hparams.get('prefetch_steps', 200) * world_size:
                if DEBUG and retry_num % 20 == 0:
                    print(f"processer_fn_spk_window#{i_worker}/{n_worker} waiting for shm to be released: {retry_num} seconds")
                time.sleep(1)
                retry_num += 1
            with counter.get_lock():
                cnt = counter.value
                counter.value += 1
            random.seed((cnt // world_size) % 1001 + seed)
            tgt_size = random.randint(hparams['tgt_size_min'], hparams['tgt_size_max'])
            samples = []
            nframes = 0
            return samples, tgt_size, cnt, nframes

        n_skip = 0
        samples, tgt_size, cnt, nframes = init_new_samples()
        restart_countdown = 10000
        while True:
            try:
                # items1
                items_bytes = read_items(q_to_pull, reader)
            except:
                n_skip += 1
                continue
            if items_bytes is None:
                break
            try:
                # items2
                item_id_start = indices[other_idx] * reader_chunk_size
                other_idx = (other_idx + 1) % len(indices)
                items_bytes2 = [x for x in reader[0].read_many([item_id_start])[0]]
            except:
                items_bytes2 = []
            try:
                # items3
                item_id_start = indices[other_idx] * reader_chunk_size
                other_idx = (other_idx + 1) % len(indices)
                items_bytes3 = [x for x in reader[0].read_many([item_id_start])[0]]
            except:
                items_bytes3 = []
            items_bytes = items_bytes + items_bytes2 + items_bytes3

            # items_merged = merge_item_bytes(items_bytes, exclude_spk=hparams.get('exclude_spk'))
            item_tgts = []
            spk_map = {}
            for item_ in items_bytes:
                item_ = pickle.loads(item_)
                try:
                    txt_token = torch.LongTensor(item_['phone_encoded'])
                except:
                    continue

                wav = torch.FloatTensor(item_['wav'])
                spk_name = item_['spk_name']
                item_name = item_['item_name']
                if 'subset' in item_:
                    spk_name = item_['subset'] + '#' + spk_name
                
                hop_size = hparams['hop_size']
                spk_win_size = hparams.get('spk_win_size', 12000)
                spk_hop_size = hparams.get('spk_hop_size', 6000)

                if speech_augmentor is not None:
                    wav = speech_augmentor(wav, hparams['audio_sample_rate'])

                if hparams.get('use_vad', False):
                    from pyvad import vad
                    wav_ = wav.numpy().astype(float)
                    wav_ = 0.95 * wav_ / max(1.0e-6, np.max(np.abs(wav_)))
                    try:
                        voiced = vad(wav_, hparams['audio_sample_rate'], fs_vad=16000, hop_length=10, vad_mode=3)
                    except ValueError:
                        print('wav_.max()', wav_.max(), 'wav_.min()', wav_.min())
                        continue
                    voiced = pad_or_cut_xd(torch.from_numpy(voiced), wav.shape[0])
                else:
                    try:
                        mel2ph = torch.LongTensor(item_['mel2ph'])
                        offsets = [0] + mel2token_to_dur(mel2ph).numpy().tolist()
                        voiced = torch.zeros_like(wav)
                        for ph_idx_, ph_idx in enumerate(txt_token):
                            if ph_idx_ < len(offsets) - 1 and ph_idx not in [2, 148, 163, 166, 153, 165, 147, 145]:
                                voiced[offsets[ph_idx_]*hop_size: offsets[ph_idx_+1]*hop_size] = 1
                    except Exception as err:
                        # print('len(txt_token)', len(txt_token), 'len(offsets)', len(offsets))
                        continue

                # item_tgt_ = {
                #     'wav': wav.numpy(),
                #     'spk_name': spk_name,
                #     'voiced': voiced.numpy()
                # }
                # item_tgts.append(item_tgt_)

                # for start_idx in range(0, wav.shape[0], spk_hop_size):
                #     if start_idx + spk_win_size >= wav.shape[0]:
                #         break
                #     item_tgt_ = {
                #         'wav': wav[start_idx: start_idx + spk_win_size].numpy(),
                #         'spk_name': spk_name,
                #         'voiced': voiced[start_idx: start_idx + spk_win_size].numpy()
                #     }
                #     item_tgts.append(item_tgt_)
                #     if spk_name not in spk_map:
                #         spk_map[spk_name] = len(spk_map)

                # wavs = torch.nn.functional.unfold(wav.unsqueeze(0).unsqueeze(1).unsqueeze(-1), (spk_win_size, 1), padding=(0, 0), stride=spk_hop_size)    # [1, t, N]
                # # assert (wavs[0, :, 0] == wav[:spk_win_size]).all()
                # voiced = torch.nn.functional.unfold(voiced.unsqueeze(0).unsqueeze(1).unsqueeze(-1), (spk_win_size, 1), padding=(0, 0), stride=spk_hop_size)    # [1, t, N]
                # selected_win_idxs = np.random.choice(wavs.shape[-1], size=min(wavs.shape[-1], hparams.get('wav_win_select_num', 3)), replace=False)
                # for win_idx in selected_win_idxs:
                #     item_tgt_ = {
                #         'wav': wavs[0, :, win_idx].numpy(),
                #         'spk_name': spk_name,
                #         'voiced': voiced[0, :, win_idx].numpy(),
                #     }
                #     item_tgts.append(item_tgt_)
                #     if spk_name not in spk_map:
                #         spk_map[spk_name] = len(spk_map)

                start_idxs = np.random.choice(
                    list(range(0, wav.shape[0], spk_hop_size)), 
                    size=max(1, min(wav.shape[0] // spk_hop_size, hparams.get('wav_win_select_num', 3))), 
                    replace=False)
                for start_idx in start_idxs:
                    if start_idx + spk_win_size >= wav.shape[0]:
                        break
                    item_tgt_ = {
                        'wav': wav[start_idx: start_idx + spk_win_size].numpy(),
                        'spk_name': spk_name,
                        'voiced': voiced[start_idx: start_idx + spk_win_size].numpy()
                    }
                    item_tgts.append(item_tgt_)
                    if spk_name not in spk_map:
                        spk_map[spk_name] = len(spk_map)

            # print('len(spk_map)', len(spk_map))
            random.shuffle(item_tgts)

            for item_tgt_ in item_tgts:
                samples.append(item_tgt_)
                nframes += len(item_tgt_['wav']) // hparams['hop_size']
                if nframes >= hparams['max_tokens']:
                    save_samples_to_shm(samples, cnt, shm_base)
                    if DEBUG:
                        print(f'processer_fn_dit_wav#{i_worker}/{n_worker} saved {shm_base}/{cnt}.pkl')
                    restart_countdown -= 1
                    if restart_countdown == 0:
                        return
                    samples, tgt_size, cnt, nframes = init_new_samples()

    except:
        traceback.print_exc()


def processer_fn_codeclm_wav(data_paths, q_to_pull, reader_chunk_size, world_size, shm_base, counter, hparams_,
                    seed, i_worker, n_worker, reader_cache_name='cache'):
    setproctitle.setproctitle('data_processer:processer_fn_codeclm_wav')
    hparams.update(hparams_)
    print(f"| Started processer_fn_codeclm_wav#{i_worker}/{n_worker}.")

    speech_augmentor = None
    if hparams.get('wav_add_noise', False) or hparams.get('wav_add_effect', False):
        from tasks.tts.dataset_utils.augment import SpeechAugment
        speech_augmentor = SpeechAugment(
            hparams.get('wav_add_noise', False), hparams.get('wav_add_effect', False),
            hparams.get('musan_dir', None), noise_prob=0.5, effect_prob=0.5, noise_snr=(6.0, 20.0)
        )
        print('| Noise mixer initialized!')
    try:
        reader = get_reader(data_paths, reader_chunk_size, i_worker,
                            n_worker, reader_cache_name)
        print(f"| init reader#{i_worker}/{n_worker}.")
        batcher = BucketBatcher(
            buckets=[100, 200, 300, 400, 500, 600, 700, 800, 1000, 1100, 1200, 1300, 1400,
                     1500, 1600, 1700, 1800, 1900, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 
                     5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 
                     12000, 14000, 16000, 18000, 20000, 25000, 30000, 50000],
            dynamic_batch=True,
            maximum_bucket_size=hparams.get('max_tokens', 40000),
            length_fn=lambda x: x['wav'].shape[0]//(24000/12.5) + len(get_word_list(x['txt'])) + 3,
            bsz_evaluator=None,
        )

        fm = hparams['frames_multiple']
        fm_wav = hparams['frames_multiple'] * hparams['hop_size']

        def init_new_samples():
            retry_num = 0
            while len(glob.glob(f'{shm_base}/*.pkl')) >= hparams.get('prefetch_steps', 200) * world_size:
                if DEBUG and retry_num % 20 == 0:
                    print(f"processer_fn_codeclm_wav#{i_worker}/{n_worker} waiting for shm to be released: {retry_num} seconds")
                time.sleep(1)
                retry_num += 1
            with counter.get_lock():
                cnt = counter.value
                counter.value += 1
            random.seed((cnt // world_size) % 1001 + seed)
            tgt_size = random.randint(hparams['tgt_size_min'], hparams['tgt_size_max'])
            samples = []
            spk_id = 0
            nframes = 0
            return samples, tgt_size, cnt, spk_id, nframes

        split_sample = MegaTTSDataset.split_sample
        samples, tgt_size, cnt, spk_id, nframes = init_new_samples()
        restart_countdown = 10000
        while True:
            try:
                items_bytes = read_items(q_to_pull, reader)
            except:
                continue
            if items_bytes is None:
                break
            items_merged = merge_item_bytes(items_bytes, exclude_spk=hparams.get('exclude_spk'), tgt_size=tgt_size)
            for item_merged in items_merged:
                add_spk_id = 0
                # while True:
                #     if len(item_merged['mel']) < tgt_size * 1.1:
                #         break
                #     item_tgt, item_merged = split_sample(item_merged, int(tgt_size * 1.1), force_word_bdr=True)
                #     if item_merged is None:
                #         break

                item_tgt = item_merged
                if hparams['max_frames'] >= len(item_tgt['mel']) > hparams['min_frames']:
                    item_tgt['mel'] = item_tgt['mel'][:len(item_tgt['mel']) // fm * fm]
                    item_tgt['wav'] = item_tgt['wav'][:len(item_tgt['wav']) // fm_wav * fm_wav]
                    item_tgt['mel2ph'] = item_tgt['mel2ph'][:len(item_tgt['mel2ph']) // fm * fm]
                    item_tgt['spk_id'] = spk_id
                    item_tgt['text'] = '<BOT>' + item_tgt['txt'] + '<BOS>'

                    item_tgt_ = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in item_tgt.items()}
                    item_tgt_ = deepcopy(item_tgt_)

                #     samples.append(item_tgt_)
                #     nframes += len(item_tgt_['mel'])
                #     add_spk_id = 1
                # if nframes >= hparams['max_tokens']:
                #     save_samples_to_shm(samples, cnt, shm_base)
                #     if DEBUG:
                #         print(f'processer_fn_codeclm_wav#{i_worker}/{n_worker} saved {shm_base}/{cnt}.pkl')
                #     add_spk_id = 0
                #     restart_countdown -= 1
                #     if restart_countdown == 0:
                #         return
                #     samples, tgt_size, cnt, spk_id, nframes = init_new_samples()

                    add_spk_id = 1
                    batch = batcher.collate_batch(item_tgt_)
                    if batch is not None and len(batch) > 0:
                        save_samples_to_shm(batch, cnt, shm_base)
                        if DEBUG:
                            print(f'processer_fn_codeclm_wav#{i_worker}/{n_worker} saved {shm_base}/{cnt}.pkl')
                        add_spk_id = 0
                        restart_countdown -= 1
                        if restart_countdown == 0:
                            return
                        samples, tgt_size, cnt, spk_id, nframes = init_new_samples()

                spk_id += add_spk_id
    except:
        traceback.print_exc()


def check_hdfs_file_existence(file_path):
    command = f"hdfs dfs -test -e {file_path}"
    try:
        subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def get_dataset_meta(hparams, reader_chunk_size):
    data_dir = hparams['binary_data_dir']
    data_paths = hparams['train_sets'] if len(hparams['train_sets']) > 0 else [data_dir]
    if hparams.get('use_falcon', True):
        data_paths_ = []
        for path in data_paths:
            if hparams.get('use_hdfs', True):
                if 'hdfs://' not in path:
                    if os.path.exists(f'{path}/data.hdfs'):
                        path = open(f'{path}/data.hdfs').readlines()[0]
                    else:
                        cluster = os.environ.get('CLUSTER')
                        hdfs_path_replace = hparams['hdfs_path_replace']
                        if cluster.lower() in ['lq', 'hl', 'sg', 'va']:
                            print_once(f'| Detect cluster [{cluster.lower()}]')
                            if cluster.lower() == 'lq':
                                hdfs_path_replace = hparams.get('hdfs_path_replace_lq', hdfs_path_replace)
                            elif cluster.lower() == 'hl':
                                hdfs_path_replace = hparams.get('hdfs_path_replace_hl', hdfs_path_replace)
                            elif cluster.lower() == 'sg':
                                hdfs_path_replace = hparams.get('hdfs_path_replace_sg', hdfs_path_replace)
                            elif cluster.lower() == 'va':
                                if hparams.get('use_hdfs_v2', False):
                                    hdfs_path_replace = hparams.get('hdfs_path_replace_va_v2', hdfs_path_replace)
                                elif hparams.get('use_metadataset', False):
                                    hparams['load_wav'] = False
                                    hdfs_path_replace = hparams.get('hdfs_path_replace_meta_va', hdfs_path_replace)
                                else:
                                    hdfs_path_replace = hparams.get('hdfs_path_replace_va', hdfs_path_replace)
                            print_once(f'| Choose hdfs_path_replace: {hdfs_path_replace}')
                        else:
                            print_once(f'| Use default hdfs_path_replace: {hdfs_path_replace}')
                        for k, v in hdfs_path_replace.items():
                            path = path.replace(k, v)
                if not path.endswith('/data') and not path.endswith('/data_sorted'):
                    path = f'{path}/data'
                if check_hdfs_file_existence(f'{path}_sorted.index'):
                    path = f'{path}_sorted'
            data_paths_.append(path)
        data_paths = data_paths_
        _, ds_len = get_reader(data_paths, 1, hparams_=hparams)
    else:
        _, ds_len = get_reader(data_paths, reader_chunk_size, hparams_=hparams)
        ds_len = sum(ds_len)
    return data_paths, ds_len


def build_fast_dataloader(shm_base, seed, world_size, hparams, n_processer=16, reader_chunk_size=64,
                          processer_fn=processer_fn_dit_wav, controller=controller_fn,
                          max_epoch=-1, auto_restart=True):
    setproctitle.setproctitle('data_processer:constructor')
    data_paths, ds_len = get_dataset_meta(hparams, reader_chunk_size)
    if hparams.get('instruction_ds_len', False):
        ds_len = hparams['instruction_ds_len']
    q_to_pull = Queue(hparams.get('prefetch_steps', 200))
    subprocess.check_call(f'rm -rf {shm_base} || exit 0', shell=True)
    os.makedirs(shm_base)
    print(f"| training dataset len: {ds_len}")
    print("| data paths: ", json.dumps(data_paths, indent=2, ensure_ascii=False))
    proc_controller = Process(
        target=controller, args=(ds_len, seed, q_to_pull, reader_chunk_size, max_epoch, n_processer),
        daemon=True)
    proc_controller.start()
    counter = multiprocessing.Value('i', 0)
    reader_cache_name = 'reader_cache'

    def create_process(worker_i):
        return Process(target=processer_fn,
                       args=(data_paths, q_to_pull, reader_chunk_size, world_size, shm_base, counter, hparams,
                             seed, worker_i, n_processer, reader_cache_name),
                       daemon=True)

    proc_processer = [create_process(i) for i in range(n_processer)]
    for p in proc_processer:
        p.start()
    time.sleep(60)
    if auto_restart:
        while True:
            if not proc_controller.is_alive():
                break
            for p_i, p in enumerate(proc_processer):
                if not p.is_alive():
                    print(f"| Restarting process {p_i}/{n_processer}")
                    i = proc_processer.index(p)
                    new_p = create_process(i)
                    new_p.start()
                    proc_processer[i] = new_p
            time.sleep(5)  # Check every 5 seconds
    proc_controller.join()
    for p in proc_processer:
        p.join()


class MegaTTSShmDataset(MegaTTSDataset):
    def __init__(self, ds_len, shm_base, worker_id, world_size, hparams):
        self.ds_len = ds_len
        self.shm_base = shm_base
        self.worker_id = worker_id
        self.world_size = world_size
        self.load_mel = True
        self.hparams = hparams
        self.last_item = None

    @staticmethod
    def init_worker(worker_id, work_dir):
        setproctitle.setproctitle(f'NATSpeech_worker ({work_dir}) dataloader')

    def __getitem__(self, index):
        data_path = f'{self.shm_base}/{index * self.world_size + self.worker_id}.pkl'
        retry_interval = 0.5
        retry_sec = 10
        retry_times = 0
        items = None
        while not os.path.exists(data_path):
            # if retry_times > retry_sec / retry_interval and index > 100:
            #     items = self.last_item
            #     print(f"| loading {data_path} failed.... use last item..")
            #     break
            retry_times += 1
            time.sleep(retry_interval)
        if items is None:
            items = pickle.load(open(data_path, 'rb'))
            os.remove(data_path)
        self.last_item = items
        items = [
            {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in item.items()}
            for item in items
        ]
        return items

    def __len__(self):
        return self.ds_len


class FrontendLMShmDataset(FrontendLMDataset):
    def __init__(self, ds_len, shm_base, worker_id, world_size, hparams):
        self.ds_len = ds_len
        self.shm_base = shm_base
        self.worker_id = worker_id
        self.world_size = world_size
        self.load_mel = True
        self.hparams = hparams
        self.last_item = None

    @staticmethod
    def init_worker(worker_id, work_dir):
        setproctitle.setproctitle(f'NATSpeech_worker ({work_dir}) dataloader')

    def __getitem__(self, index):
        data_path = f'{self.shm_base}/{index * self.world_size + self.worker_id}.pkl'
        retry_interval = 0.5
        retry_sec = 10
        retry_times = 0
        items = None
        while not os.path.exists(data_path):
            # if retry_times > retry_sec / retry_interval and index > 100:
            #     items = self.last_item
            #     print(f"| loading {data_path} failed.... use last item..")
            #     break
            retry_times += 1
            time.sleep(retry_interval)
        if items is None:
            items = pickle.load(open(data_path, 'rb'))
            os.remove(data_path)
        self.last_item = items
        items = [
            {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in item.items()}
            for item in items
        ]
        return items

    def __len__(self):
        return self.ds_len


class G2PLMShmDataset(G2PLMDataset):
    def __init__(self, ds_len, shm_base, worker_id, world_size, hparams):
        self.ds_len = ds_len
        self.shm_base = shm_base
        self.worker_id = worker_id
        self.world_size = world_size
        self.load_mel = True
        self.hparams = hparams
        self.last_item = None

    @staticmethod
    def init_worker(worker_id, work_dir):
        setproctitle.setproctitle(f'NATSpeech_worker ({work_dir}) dataloader')

    def __getitem__(self, index):
        data_path = f'{self.shm_base}/{index * self.world_size + self.worker_id}.pkl'
        retry_interval = 0.5
        retry_sec = 10
        retry_times = 0
        items = None
        while not os.path.exists(data_path):
            # if retry_times > retry_sec / retry_interval and index > 100:
            #     items = self.last_item
            #     print(f"| loading {data_path} failed.... use last item..")
            #     break
            retry_times += 1
            time.sleep(retry_interval)
        if items is None:
            items = pickle.load(open(data_path, 'rb'))
            os.remove(data_path)
        self.last_item = items
        items = [
            {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in item.items()}
            for item in items
        ]
        return items

    def __len__(self):
        return self.ds_len
    

class SDVAEShmDataset(SDVAEDataset):
    def __init__(self, ds_len, shm_base, worker_id, world_size, hparams):
        self.ds_len = ds_len
        self.shm_base = shm_base
        self.worker_id = worker_id
        self.world_size = world_size
        self.load_mel = True
        self.hparams = hparams
        self.last_item = None

    @staticmethod
    def init_worker(worker_id, work_dir):
        setproctitle.setproctitle(f'NATSpeech_worker ({work_dir}) dataloader')

    def __getitem__(self, index):
        data_path = f'{self.shm_base}/{index * self.world_size + self.worker_id}.pkl'
        retry_interval = 0.5
        retry_sec = 10
        retry_times = 0
        items = None
        while not os.path.exists(data_path):
            # if retry_times > retry_sec / retry_interval and index > 100:
            #     items = self.last_item
            #     print(f"| loading {data_path} failed.... use last item..")
            #     break
            retry_times += 1
            time.sleep(retry_interval)
        if items is None:
            items = pickle.load(open(data_path, 'rb'))
            os.remove(data_path)
        self.last_item = items
        items = [
            {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in item.items()}
            for item in items
        ]
        return items

    def __len__(self):
        return self.ds_len


class DiTShmDataset(DiTDataset):
    def __init__(self, ds_len, shm_base, worker_id, world_size, hparams):
        self.ds_len = ds_len
        self.shm_base = shm_base
        self.worker_id = worker_id
        self.world_size = world_size
        self.load_mel = True
        self.hparams = hparams
        self.last_item = None

    @staticmethod
    def init_worker(worker_id, work_dir):
        setproctitle.setproctitle(f'NATSpeech_worker ({work_dir}) dataloader')

    def __getitem__(self, index):
        data_path = f'{self.shm_base}/{index * self.world_size + self.worker_id}.pkl'
        retry_interval = 0.5
        retry_sec = 10
        retry_times = 0
        items = None
        while not os.path.exists(data_path):
            retry_times += 1
            time.sleep(retry_interval)
        if items is None:
            items = pickle.load(open(data_path, 'rb'))
            os.remove(data_path)
        self.last_item = items
        items = [
            {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in item.items()}
            for item in items
        ]
        return items

    def __len__(self):
        return self.ds_len


class DiTWavShmDataset(DiTWavDataset):
    def __init__(self, ds_len, shm_base, worker_id, world_size, hparams):
        self.ds_len = ds_len
        self.shm_base = shm_base
        self.worker_id = worker_id
        self.world_size = world_size
        self.load_mel = True
        self.hparams = hparams
        self.last_item = None

    @staticmethod
    def init_worker(worker_id, work_dir):
        setproctitle.setproctitle(f'NATSpeech_worker ({work_dir}) dataloader')

    def __getitem__(self, index):
        data_path = f'{self.shm_base}/{index * self.world_size + self.worker_id}.pkl'
        retry_interval = 0.5
        retry_sec = 10
        retry_times = 0
        items = None
        while not os.path.exists(data_path):
            retry_times += 1
            if DEBUG and retry_times % 20 == 0:
                print(f"NATSpeech_worker {self.worker_id}/{self.world_size} wating for {data_path}")
            time.sleep(retry_interval)
        if items is None:
            items = pickle.load(open(data_path, 'rb'))
            os.remove(data_path)
            if DEBUG:
                print(f"NATSpeech_worker {self.worker_id}/{self.world_size} read {data_path}")
        self.last_item = items
        items = [
            {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in item.items()}
            for item in items
        ]
        return items

    def __len__(self):
        return self.ds_len

from tasks.tts.dataset_utils.tts_datasets import DiTWavTextDataset
class DiTWavTextShmDataset(DiTWavTextDataset):
    def __init__(self, ds_len, shm_base, worker_id, world_size, hparams):
        self.ds_len = ds_len
        self.shm_base = shm_base
        self.worker_id = worker_id
        self.world_size = world_size
        self.load_mel = True
        self.hparams = hparams
        self.last_item = None

    @staticmethod
    def init_worker(worker_id, work_dir):
        setproctitle.setproctitle(f'NATSpeech_worker ({work_dir}) dataloader')

    def __getitem__(self, index):
        data_path = f'{self.shm_base}/{index * self.world_size + self.worker_id}.pkl'
        retry_interval = 0.5
        retry_sec = 10
        retry_times = 0
        items = None
        while not os.path.exists(data_path):
            retry_times += 1
            if DEBUG and retry_times % 20 == 0:
                print(f"NATSpeech_worker {self.worker_id}/{self.world_size} wating for {data_path}")
            time.sleep(retry_interval)
        if items is None:
            items = pickle.load(open(data_path, 'rb'))
            os.remove(data_path)
            if DEBUG:
                print(f"NATSpeech_worker {self.worker_id}/{self.world_size} read {data_path}")
        self.last_item = items
        items = [
            {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in item.items()}
            for item in items
        ]
        return items

    def __len__(self):
        return self.ds_len
    

from tasks.tts.dataset_utils.tts_datasets import CausalASRDataset
class CausalASRShmDataset(CausalASRDataset):
    def __init__(self, ds_len, shm_base, worker_id, world_size, hparams):
        self.ds_len = ds_len
        self.shm_base = shm_base
        self.worker_id = worker_id
        self.world_size = world_size
        self.load_mel = True
        self.hparams = hparams
        self.last_item = None

    @staticmethod
    def init_worker(worker_id, work_dir):
        setproctitle.setproctitle(f'NATSpeech_worker ({work_dir}) dataloader')

    def __getitem__(self, index):
        data_path = f'{self.shm_base}/{index * self.world_size + self.worker_id}.pkl'
        retry_interval = 0.5
        retry_sec = 10
        retry_times = 0
        items = None
        while not os.path.exists(data_path):
            retry_times += 1
            if DEBUG and retry_times % 20 == 0:
                print(f"NATSpeech_worker {self.worker_id}/{self.world_size} wating for {data_path}")
            time.sleep(retry_interval)
        if items is None:
            items = pickle.load(open(data_path, 'rb'))
            os.remove(data_path)
            if DEBUG:
                print(f"NATSpeech_worker {self.worker_id}/{self.world_size} read {data_path}")
        self.last_item = items
        items = [
            {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in item.items()}
            for item in items
        ]
        return items

    def __len__(self):
        return self.ds_len


from tasks.tts.dataset_utils.tts_datasets import SpkWindowDataset
class SpkWindowShmDataset(SpkWindowDataset):
    def __init__(self, ds_len, shm_base, worker_id, world_size, hparams):
        self.ds_len = ds_len
        self.shm_base = shm_base
        self.worker_id = worker_id
        self.world_size = world_size
        self.load_mel = True
        self.hparams = hparams
        self.last_item = None

    @staticmethod
    def init_worker(worker_id, work_dir):
        setproctitle.setproctitle(f'NATSpeech_worker ({work_dir}) dataloader')

    def __getitem__(self, index):
        data_path = f'{self.shm_base}/{index * self.world_size + self.worker_id}.pkl'
        retry_interval = 0.5
        retry_sec = 10
        retry_times = 0
        items = None
        while not os.path.exists(data_path):
            retry_times += 1
            if DEBUG and retry_times % 20 == 0:
                print(f"NATSpeech_worker {self.worker_id}/{self.world_size} wating for {data_path}")
            time.sleep(retry_interval)
        if items is None:
            items = pickle.load(open(data_path, 'rb'))
            os.remove(data_path)
            if DEBUG:
                print(f"NATSpeech_worker {self.worker_id}/{self.world_size} read {data_path}")
        self.last_item = items
        items = [
            {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in item.items()}
            for item in items
        ]
        return items

    def __len__(self):
        return self.ds_len


class CodecLMWavShmDataset(CodecLMWavDataset):
    def __init__(self, ds_len, shm_base, worker_id, world_size, hparams):
        self.ds_len = ds_len
        self.shm_base = shm_base
        self.worker_id = worker_id
        self.world_size = world_size
        self.load_mel = True
        self.hparams = hparams
        self.last_item = None

    @staticmethod
    def init_worker(worker_id, work_dir):
        setproctitle.setproctitle(f'NATSpeech_worker ({work_dir}) dataloader')

    def __getitem__(self, index):
        data_path = f'{self.shm_base}/{index * self.world_size + self.worker_id}.pkl'
        retry_interval = 0.5
        retry_sec = 10
        retry_times = 0
        items = None
        while not os.path.exists(data_path):
            retry_times += 1
            if DEBUG and retry_times % 20 == 0:
                print(f"NATSpeech_worker {self.worker_id}/{self.world_size} wating for {data_path}")
            time.sleep(retry_interval)
        if items is None:
            items = pickle.load(open(data_path, 'rb'))
            os.remove(data_path)
            if DEBUG:
                print(f"NATSpeech_worker {self.worker_id}/{self.world_size} read {data_path}")
        self.last_item = items
        items = [
            {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in item.items()}
            for item in items
        ]
        return items

    def __len__(self):
        return self.ds_len
    

class DiTWavImgShmDataset(DiTWavImgDataset):
    def __init__(self, ds_len, shm_base, worker_id, world_size, hparams):
        self.ds_len = ds_len
        self.shm_base = shm_base
        self.worker_id = worker_id
        self.world_size = world_size
        self.load_mel = True
        self.hparams = hparams
        self.last_item = None

    @staticmethod
    def init_worker(worker_id, work_dir):
        setproctitle.setproctitle(f'NATSpeech_worker ({work_dir}) dataloader')

    def __getitem__(self, index):
        data_path = f'{self.shm_base}/{index * self.world_size + self.worker_id}.pkl'
        retry_interval = 0.5
        retry_sec = 10
        retry_times = 0
        items = None
        while not os.path.exists(data_path):
            retry_times += 1
            if DEBUG and retry_times % 20 == 0:
                print(f"NATSpeech_worker {self.worker_id}/{self.world_size} wating for {data_path}")
            time.sleep(retry_interval)
        if items is None:
            items = pickle.load(open(data_path, 'rb'))
            os.remove(data_path)
            if DEBUG:
                print(f"NATSpeech_worker {self.worker_id}/{self.world_size} read {data_path}")
        self.last_item = items
        items = [
            {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in item.items()}
            for item in items
        ]
        return items

    def __len__(self):
        return self.ds_len


class Latent2WavShmDataset(Latent2WavDataset):
    def __init__(self, ds_len, shm_base, worker_id, world_size, hparams, validation=False):
        self.ds_len = ds_len
        self.shm_base = shm_base
        self.worker_id = worker_id
        self.world_size = world_size
        self.load_mel = True
        self.hparams = hparams
        self.last_item = None

        # total downsample rate is hop_size * vae_stride (8)
        if validation:
            self.batch_max_samples = 0
        else:
            self.batch_max_samples = self.hparams['max_samples']
        
        self.batch_max_frames = hparams['max_samples'] // hparams['hop_size']
        self.hop_size = hparams['hop_size']

    @staticmethod
    def init_worker(worker_id, work_dir):
        setproctitle.setproctitle(f'NATSpeech_worker ({work_dir}) dataloader')

    def __getitem__(self, index):
        data_path = f'{self.shm_base}/{index * self.world_size + self.worker_id}.pkl'
        retry_interval = 0.5
        retry_sec = 10
        retry_times = 0
        items = None
        while not os.path.exists(data_path):
            # if retry_times > retry_sec / retry_interval and index > 100:
            #     items = self.last_item
            #     print(f"| loading {data_path} failed.... use last item..")
            #     break
            retry_times += 1
            time.sleep(retry_interval)
        if items is None:
            items = pickle.load(open(data_path, 'rb'))
            os.remove(data_path)
        self.last_item = items
        items = [
            {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in item.items()}
            for item in items
        ]
        return items

    def __len__(self):
        return self.ds_len


if __name__ == '__main__':
    import soundfile as sf
    from utils.commons.io import json_dump
    from tqdm import tqdm
    from functools import partial
    from utils.commons.hparams import hparams, set_hparams
    set_hparams('egs/tts/scriptspeech_dit_dialogue.yaml', print_hparams=False, global_hparams=True)
    hparams['exp_name'] = 'test'
    hparams['max_sentences'] = 200
    hparams['max_tokens'] = 20000
    hparams['tgt_size_min'] = 20 * 100
    hparams['tgt_size_max'] = 60 * 100
    hparams['prefetch_steps'] = 8
    hparams['ds_workers'] = 4
    hparams['work_dir'] = 'checkpoints/test'
    hparams['train_sets'] = ['data/binary_v3/sa_data_v1']

    node_id = None
    node_size = None
    multiprocessing.set_start_method('spawn', force=True)
    shm_base = f'/dev/shm/data_shm_{hparams["exp_name"]}'
    l = hparams.get('max_updates', 10000000)
    Process(target=build_fast_dataloader, kwargs={
        'shm_base': shm_base,
        'seed': 0,
        'world_size': 1,
        'hparams': hparams,
        'n_processer': 1,
        'reader_chunk_size': hparams.get('reader_chunk_size', 64),
        'processer_fn': processer_fn_dit_wav_text_multispk
    }).start()
    train_dataset = DiTWavTextShmDataset(
        l, shm_base, 0, 1, hparams)
    
    dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=train_dataset.collater,
        num_workers=4,
        worker_init_fn=partial(train_dataset.init_worker, work_dir=hparams['work_dir']),
        persistent_workers=False,
        prefetch_factor=4
    )

    temp_dir = 'user/temp/test_dl2'

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

def processer_fn_am(
        data_paths, q_to_pull, reader_chunk_size, world_size,
        shm_base, counter, hparams_, seed, i_worker, n_worker, reader_cache_name='cache'):
    setproctitle.setproctitle('data_processer:processer_fn')
    hparams.update(hparams_)

    try:
        reader = get_reader(data_paths, reader_chunk_size,
                            i_worker, n_worker, reader_cache_name)
        fm = hparams['frames_multiple']
        fm_wav = hparams['frames_multiple'] * hparams['hop_size']

        def init_new_samples():
            while len(glob.glob(f'{shm_base}/*.pkl')) >= hparams.get('prefetch_steps', 200) * world_size:
                time.sleep(1)
            with counter.get_lock():
                cnt = counter.value
                counter.value += 1
            random.seed((cnt // world_size) % 1001 + seed)
            ref_size, tgt_size = 200, 200
            ref_size = ref_size // fm * fm
            tgt_size = tgt_size // fm * fm
            max_n = min(hparams['max_tokens'] // (ref_size + tgt_size), hparams['max_sentences'])
            samples = []
            return samples, ref_size, tgt_size, max_n, cnt

        split_sample = MegaTTSDataset.split_sample
        samples, ref_size, tgt_size, max_n, cnt = init_new_samples()
        restart_countdown = 10000
        while True:
            try:
                items_bytes = read_items(q_to_pull, reader)
            except:
                traceback.print_exc()
                continue
            if items_bytes is None:
                break
            items_merged = merge_item_bytes(items_bytes, exclude_spk=hparams.get('exclude_spk'))
            n_samples = 0
            for item_merged in items_merged:
                while True:
                    if len(item_merged['mel']) < ref_size + tgt_size:
                        break
                    item_cur, item_merged = split_sample(item_merged, ref_size + tgt_size, force_word_bdr=True)
                    item_ref, item_tgt = split_sample(item_cur, ref_size, force_word_bdr=True)
                    if item_tgt is None:
                        print("| item_tgt is none", len(item_cur['mel']), ref_size)
                        break
                    if len(item_tgt['mel']) > hparams['min_frames']:
                        item_tgt['mel_timbre'] = item_ref['mel'][:len(item_ref['mel']) // fm * fm]
                        item_tgt['mel'] = item_tgt['mel'][:len(item_tgt['mel']) // fm * fm]
                        item_tgt['wav_timbre'] = item_ref['wav'][:len(item_ref['wav']) // fm_wav * fm_wav]
                        item_tgt['wav'] = item_tgt['wav'][:len(item_tgt['wav']) // fm_wav * fm_wav]
                        item_tgt['mel2ph'] = item_tgt['mel2ph'][:len(item_tgt['mel2ph']) // fm * fm]
                        item_tgt_ = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in item_tgt.items()}
                        item_tgt_ = deepcopy(item_tgt_)
                        samples.append(item_tgt_)
                        n_samples += 1
                    if item_merged is None:
                        break
                    if len(samples) >= max_n:
                        l_mel_min = min([s['mel'].shape[0] for s in samples])
                        l_timbre_min = min([s['mel_timbre'].shape[0] for s in samples])
                        for s in samples:
                            s['mel'] = s['mel'][:l_mel_min]
                            s['wav'] = s['wav'][:l_mel_min * hparams['hop_size']]
                            s['mel2ph'] = s['mel2ph'][:l_mel_min]
                            s['mel_timbre'] = s['mel_timbre'][:l_timbre_min]
                            s['wav_timbre'] = s['wav_timbre'][:l_timbre_min * hparams['hop_size']]
                        save_samples_to_shm(samples, cnt, shm_base)
                        restart_countdown -= 1
                        if restart_countdown <= 0:
                            return
                        samples, ref_size, tgt_size, max_n, cnt = init_new_samples()
    except:
        traceback.print_exc()