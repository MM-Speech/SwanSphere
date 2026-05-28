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


def remove_tag_blocks(s: str, tag: str = "tag", replace=' ') -> str:
    pattern = rf"\s*<{tag}\b[^>]*>.*?</{tag}>\s*"
    s = re.sub(pattern, " ", s, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"\s+", replace, s).strip()

def valid_item_kv(item, k):
    return k in item and item[k] is not None


class PromptTTSShmDataset(BaseFalconReaderShmDataset):
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
        
    def process_item(self, raw_item, hparams, global_stores, i_worker, n_worker):
        if self.use_fast_dataloader:
            batcher = get_from_global_stores(
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
            
        # item = self._process_item(raw_item, hparams, global_stores, i_worker, n_worker)
        # if item is None:
        #     return
        # if self.use_fast_dataloader:
        #     batch = batcher.collate_batch(item)
        #     if batch is not None and len(batch) > 0:
        #         yield batch
        # else:
        #     yield [item]

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
        if cnt > 0 and cnt % step == 0:
            print(f"| processer#{i_worker}/{n_worker}: skipped [{cnt}/{item_cnt}] items for [{name}]")
                
    def _process_item(self, raw_item, hparams, global_stores, i_worker, n_worker):
        try:
            length_regulator = get_from_global_stores(
                'length_regulator', global_stores,
                lambda: LengthRegulator()
            )
            if not hasattr(self, 'item_cnt'):
                self.item_cnt = 0
                self.no_score_cnt = 0
                self.no_text_cnt = 0
                self.no_caption_cnt = 0
                self.no_phone_cnt = 0
            self.item_cnt += 1
            
            fm = hparams['frames_multiple']
            fm_wav = hparams['frames_multiple'] * hparams['hop_size']
            hop_size = hparams['hop_size']
            vae_stride = hparams['vae_stride']
        
            if hparams.get('use_vocal', True):
                wav = (raw_item['vocal']).astype(float)
                org_sr = raw_item['vocal_sr']
            else:
                wav = (raw_item['wav']).astype(float)
                org_sr = raw_item['sr']
            
            if (sr := hparams['audio_sample_rate']) != org_sr:
                wav = librosa.resample(wav, orig_sr=org_sr, target_sr=sr)
                
            wav = wav[:len(wav) // fm_wav * fm_wav]
                
            item = {
                'wav': wav,
                'subset': raw_item['subset']
            }
            
            if 'quality_score' in raw_item:
                item['stoi'] = raw_item['quality_score']['stoi']
                item['pesq'] = raw_item['quality_score']['pesq']
                item['si_sdr'] = raw_item['quality_score']['si_sdr']
                item['mos'] = raw_item['quality_score']['mos']
            else:
                item['stoi'] = -1
                item['pesq'] = -1
                item['si_sdr'] = -50
                item['mos'] = -1
                self.no_score_cnt += 1
                self.report_skip_status(self.no_score_cnt, self.item_cnt, 'score', i_worker, n_worker, 100)
                
            if 'asr_results' in raw_item:
                item['text'] = raw_item['asr_results']['text_normed']
            else:
                item['text'] = ''
                self.no_text_cnt += 1
                self.report_skip_status(self.no_text_cnt, self.item_cnt, 'text', i_worker, n_worker, 100)
                
            if (
                    'gemini_result' in raw_item and raw_item['gemini_result'] is not None and
                    isinstance(raw_item['gemini_result']['global description'], str) and
                    isinstance(raw_item['gemini_result']['fine-grained transcription'], str)
                ):
                item['global_prompt'] = raw_item['gemini_result']['global description']
                item['local_prompt'] = raw_item['gemini_result']['fine-grained transcription']
            else:
                item['global_prompt'] = ''
                item['local_prompt'] = ''
                self.no_caption_cnt += 1
                self.report_skip_status(self.no_caption_cnt, self.item_cnt, 'caption', i_worker, n_worker, 100)
                if 'gemini_result' in raw_item and raw_item['gemini_result'] is not None:
                    if not isinstance(raw_item['gemini_result']['global description'], str):
                        print(f"{raw_item['gemini_result']['global description'] = }")
                    if not isinstance(raw_item['gemini_result']['fine-grained transcription'], str):
                        print(f"{raw_item['gemini_result']['fine-grained transcription'] = }")
                        
            if 'gemini_mix_result' in raw_item:
                if random.random() < 0.5:
                    item['global_prompt'] = raw_item['gemini_mix_result']['global description']
                    item['local_prompt'] = raw_item['gemini_mix_result']['fine-grained transcription']
                    
            if item['text'] == '' or item['text'] == '.' or (random.random() < 0.7 and len(item['local_prompt']) > 0):
                item['text'] = remove_tag_blocks(item['local_prompt'])
                
            if item['text'] == '':
                return
            
            if 'phone_encoded' in raw_item:
                raw_item['dur'] = np.trim_zeros(raw_item['dur'], 'b')
                raw_item['phone_encoded'] = raw_item['phone_encoded'][:len(raw_item['dur'])]
                raw_item['tone_encoded'] = raw_item['tone_encoded'][:len(raw_item['dur'])]
                raw_item['mel2ph'] = np.array(raw_item['mel2ph'])[:len(raw_item['mel2ph']) // fm * fm]
            if (
                    'phone_encoded' in raw_item and 
                    raw_item['phone_encoded'] is not None and len(raw_item['phone_encoded']) > 0 and
                    abs(len(raw_item['phone_encoded']) - max(raw_item['mel2ph'])) <= 1 and 
                    len(raw_item['phone_encoded']) >= max(raw_item['mel2ph']) and
                    len(item['wav']) // hop_size - len(raw_item['mel2ph']) < 2 and
                    len(raw_item['phone_encoded']) == len(raw_item['dur']) and
                    len(raw_item['phone_encoded']) <= len(item['wav']) // hop_size // 4
                ):
                    item['mel2ph'] = raw_item['mel2ph']
                    item['ph_token'] = np.array(raw_item['phone_encoded'][:max(item['mel2ph'])])
                    item['tone'] = np.array(raw_item['tone_encoded'][:max(item['mel2ph'])])
                    item['dur'] = np.array(raw_item['dur'][:max(item['mel2ph'])])
                    item['wav'] = item['wav'][:len(item['mel2ph']) * hop_size]
            else:
                item['ph_token'] = np.array([301])
                item['tone'] = np.array([31])
                item['dur'] = np.array([len(wav) // hparams['hop_size'] // fm * fm])
                # item['mel2ph'] = length_regulator(torch.from_numpy(item['dur'])[None])[0].numpy()
                item['mel2ph'] = np.ones(len(wav) // hparams['hop_size'], dtype=int)
                item['mel2ph'] = item['mel2ph'][:len(item['mel2ph']) // fm * fm]
                self.no_phone_cnt += 1
                self.report_skip_status(self.no_phone_cnt, self.item_cnt, 'phone_dur', i_worker, n_worker, 1000)
                # print(f"{max(raw_item['mel2ph']) = } | {len(raw_item['phone_encoded']) = } | {len(raw_item['dur']) = } | {self.no_phone_cnt}/{self.item_cnt}")
            
            if hparams.get('use_sparse_dur', False):
                mel2ph_sparse = compute_mel2aug_from_dur(
                    item['dur'].tolist(),
                    gap_mode=hparams.get('sparse_dur_mode', 'proportional'),
                    gap_frames=hparams.get('sparse_dur_frames', 4),
                    gap_alpha=hparams.get('sparse_dur_alpha', 0.2),
                    min_keep=hparams.get('sparse_dur_min_keep', 1),
                    keep_ratio=hparams.get('sparse_dur_keep_ratio'),
                    symmetric=hparams.get('sparse_dur_symmetric', True),
                )
                item['mel2ph_sparse'] = mel2ph_sparse
                
            mel_len = int(len(item['wav']) / hop_size)
            min_idx = max(int(mel_len * 0.1), 200)
            max_idx = min(int(mel_len * 0.9), mel_len - 200)
            if min_idx > max_idx:
                min_idx = int(mel_len * 0.4)
                max_idx = int(mel_len * 0.6)
            rand_length = random.randint(min_idx, max_idx) // fm * fm
            ctx_mask = torch.zeros((item['wav'].shape[0] // hparams['hop_size'], 1))
            ctx_mask[:rand_length] = 1.0
            item['ctx_mask'] = ctx_mask[::hparams['vae_stride']]
            item['ctx_wav'] = deepcopy(item['wav'])
            item['ctx_wav'] = item['ctx_wav'][:rand_length*hparams['hop_size']]
                
            if hparams.get('add_vad_mask', False):
                vad_model = get_from_global_stores(
                    'vad_model', global_stores,
                    lambda: build_vad_model()
                )
                vad_start, vad_end = run_vad_trim(wav, sr, vad_model, 0.5)
                if vad_start == vad_end == 0:
                    vad_start, vad_end = run_vad_trim(wav, sr, vad_model, 0.3)
                vm = hparams['hop_size'] * hparams['vae_stride']
                vad_mask = np.zeros((wav.shape[0] // vm))
                vad_mask[int(vad_start * hparams['audio_sample_rate'] // vm) : int(vad_end * hparams['audio_sample_rate'] // vm)] = 1
                item['vad_mask'] = vad_mask
                
            if hparams.get('length_fn', 'lat') == 'lat':
                item['len'] = item['wav'].shape[0] // hparams['hop_size'] // hparams['vae_stride']
            elif hparams.get('length_fn', 'lat') == 'ph':
                item['len'] = len(item['ph_token'])
            
            yield item
            
        except:
            # traceback.print_exc()
            return
        
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
        wavs = collate_xd([s['wav'] for s in samples], 0.0)
        wav_lengths = torch.LongTensor([s['wav'].shape[0] for s in samples])
        batch = {
            'nsamples': len(samples),
            'wavs': wavs,
            'wav_lengths': wav_lengths,
        }
        if 'mel' in samples[0]:
            batch['mels'] = collate_xd([s['mel'] for s in samples], -6.0)
        if 'mel2ph' in samples[0]:
            batch['mel2ph'] = collate_xd([s['mel2ph'] for s in samples], 0)
        if 'dur' in samples[0]:
            batch['dur'] = collate_xd([s['dur'] for s in samples], 0)
            batch['dur_len'] = torch.LongTensor([s['dur'].shape[0] for s in samples])
        if 'ctx_wav' in samples[0]:
            batch['ctx_wavs'] = collate_xd([s['ctx_wav'] for s in samples], 0.0)
        if valid_item_kv(samples[0], 'ctx_mask'):
            batch['ctx_mask'] = collate_xd([s['ctx_mask'] for s in samples], 0)
        if 'mel2ph_sparse' in samples[0]:
            batch['mel2ph_sparse'] = collate_xd([s['mel2ph_sparse'] for s in samples], 0)
        batch['text'] = [s['text'] for s in samples]
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
        for k in ['stoi', 'pesq', 'si_sdr', 'mos']:
            if k in samples[0]:
                batch[k] = torch.Tensor([s[k] for s in samples])
        if 'global_prompt' in samples[0]:
            batch['global_prompt'] = [s['global_prompt'] for s in samples]
        if 'local_prompt' in samples[0]:
            batch['local_prompt'] = [s['local_prompt'] for s in samples]
        if 'vad_mask' in samples[0]:
            batch['vad_mask'] = collate_xd([s['vad_mask'] for s in samples], 0.0)

        if not hasattr(self, 'backup_batch') or self.backup_batch is None or random.random() < 0.001:
            self.backup_batch = batch

        return batch
        

if __name__ == '__main__':
    from utils.commons.hparams import set_hparams, hparams
    set_hparams('egs/tts/prompttts_dit_v2.yaml', print_hparams=False)
    hparams['hdfs_root'] = 'hdfs://harunava/home/byte_advertising_genai/20250808/liruiqi/data/prompttts_250916'
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

    ds_train = PromptTTSShmDataset('train', hparams, use_fast_dataloader=True, rank_id=0, world_size=1, batch_size=1)
    dl_train = ds_train.get_dataloader(seed=1234, num_workers=hparams['ds_workers'])
    for i, items in enumerate(dl_train):
        if 'ph_tokens' in items:
            print(items)
            break
        
        
