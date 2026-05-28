import glob
import itertools
import math
import traceback

import simplejson as json
import os
import pickle
import random
import re
import shelve
from copy import deepcopy
from functools import partial
from sys import getsizeof

import librosa.core
import numpy as np
import torch
import torch.distributions
import torch.optim
import torch.utils.data
from torch.utils.data import DistributedSampler

from utils.audio import librosa_wav2spec, librosa_wav2linearspec
from utils.audio.align import mel2token_to_dur
from tasks.tts.dataset_utils.base_dataset import BaseKVDataset, collate_xd
import torch.nn.functional as F
from dataloader import KVReader



def get_itemnames_shelve_path(data_dir, is_train, load_full=False):
    if is_train:
        db_name = f'{data_dir}/item_names_train.shelve'
        if not os.path.exists(f'{db_name}.db') or load_full:
            db_name = f'{data_dir}/item_names.shelve'
    else:
        db_name = f'{data_dir}/item_names_val.shelve'
    return db_name


def load_ds_len(data_dir, is_train=True, load_full=False):
    try:
        db_name = get_itemnames_shelve_path(data_dir, is_train, load_full)
        len_path = db_name.replace(".shelve", "_len.json")
        if os.path.exists(len_path):
            l = json.load(open(len_path))['len']
            print(f"| load dataset length of {db_name} from json: {l}")
        else:
            with shelve.open(db_name, 'r') as d:
                l = len(d.keys())
            print(f"| load dataset length of {db_name} from shelve: {l}")
        return l
    except:
        print(f"| Errors in load_ds_len: {data_dir}")
        traceback.print_exc()
        return 0


def load_item_names_shelve(data_dir, is_train=True):
    db_name = get_itemnames_shelve_path(data_dir, is_train, load_full=False)
    return shelve.open(db_name, 'r')


def valid_item_kv(item, k):
    return k in item and item[k] is not None


class BaseSpeechDataset(BaseKVDataset):
    def __init__(self, prefix,  shuffle=False, data_dir=None, load_size=True, chunk_size=5):
        from utils.commons.hparams import hparams
        hparams = deepcopy(hparams)
        self.hparams = hparams
        self.data_dir = hparams['binary_data_dir'] if data_dir is None else data_dir
        self.prefix = prefix
        self.load_size = load_size
        ds_path = f'{self.data_dir}/data'
        if load_size:
            name2mellen = json.load(open(f'{self.data_dir}/meta.mel_lengths.json'))
            if 'test_dataset' not in self.hparams:
                self.hparams['test_dataset'] = self.prefix != 'train'
            self.sizes = name2mellen
            id2item_names = load_item_names_shelve(self.data_dir, self.prefix == 'train')
            self.item_names = sorted(list(id2item_names.values()))
        super().__init__(ds_path, shuffle, num_parallel=4, load_size=load_size, chunk_size=chunk_size)
        if load_size:
            del name2mellen
            del self.sizes

    @classmethod
    def init_worker(cls, worker_id=None, self=None):
        if self is None:
            worker_info = torch.utils.data.get_worker_info()
            self = worker_info.dataset
        if self.hparams.get('use_falcon'):
            self.indexed_fal_ds = self.get_falcon_reader(self.num_parallel)
        self.indexed_ds = self.indexed_kv_ds = self.get_kv_reader(self.num_parallel)
        self.name2mellen = shelve.open(f'{self.data_dir}/meta.mel_lengths.shelve', 'r')
        self.name2phlen = shelve.open(f'{self.data_dir}/meta.ph_lengths.shelve', 'r')
        self.spk2name = shelve.open(f'{self.data_dir}/meta.spk2item_names.shelve', 'r')
        self.id2item_names = load_item_names_shelve(self.data_dir, self.prefix == 'train')

    def get_key_and_sizes(self):
        return [(k, self.sizes[k]) for k in self.item_names]

    def name2spk(self, item_name):
        return "/".join(item_name.split("/")[:-1])

    def get_items_by_same_spk(self, k):
        spk = self.name2spk(k)
        itemnames = self.spk2name[spk]
        itemnames = sorted(itemnames)
        return itemnames


class BaseTTSDataset(BaseSpeechDataset):
    def __init__(self, prefix, shuffle=False, data_dir=None, max_len=None):
        super().__init__(prefix, shuffle, data_dir)
        hparams = deepcopy(self.hparams)
        if prefix in ['test', 'valid'] and len(hparams['test_ids']) > 0:
            self.key_and_sizes = [x for x in self.key_and_sizes if x[0] in hparams['test_ids']]
        if prefix == 'train' and hparams['min_frames'] > 0:
            if not hparams['pad_frames']:
                self.key_and_sizes = [x for x in self.key_and_sizes if hparams['min_frames'] <= x[1]]
        if max_len is not None:
            self.key_and_sizes = self.key_and_sizes[:max_len]
        self.load_mel = True
        print(f"| {self.data_dir}.{prefix} dataset size:", len(self.key_and_sizes))
        self.spk_map = json.load(open(f'{hparams["work_dir"]}/spk_map.json'))

    def get_sample(self, id, item):
        hparams = self.hparams
        ph_token = torch.LongTensor(item['phone_encoded'][:hparams['max_input_tokens']])
        sample = {
            "id": id,
            "item_name": item['item_name'],
            "text": item['txt_raw'],
            "txt_token": ph_token,
            'wav': item['wav']
        }
        if self.load_mel:
            if item['sec'] < 0.3:
                return None
            if 'mel' not in item:
                item['mel'], item['wav'] = self.get_mel(self.hparams, item['wav'])
            max_frames = hparams['max_frames']
            spec = torch.Tensor(item['mel'])[:max_frames]
            max_frames = spec.shape[0] // hparams['frames_multiple'] * hparams['frames_multiple']
            spec = spec[:max_frames]
            wav = torch.Tensor(item['mel'])[:max_frames*self.hparams['hop_size']]
            mel_len = spec.shape[0]
            sample.update({
                "mel": spec,
                "wav": wav
            })
            sample['mel2ph'] = torch.LongTensor(item['mel2ph'])[:max_frames]
            sample['mel2ph'][-1] = sample['mel2ph'][-2]

            if hparams['ignore_begin_end_sil']:
                vmin = hparams['mel_vmin']

                begin_sil_mask = sample['mel2ph'] == 1
                n_begin_sil = begin_sil_mask.sum()
                if n_begin_sil > 0:
                    spec_begin = torch.ones(mel_len)
                    spec_begin[:n_begin_sil] = (torch.arange(0, n_begin_sil) / n_begin_sil) ** 2
                    spec_begin = spec_begin[..., None] * (sample['mel'] - vmin) + vmin
                else:
                    spec_begin = sample['mel']

                end_sil_mask = sample['mel2ph'] == ph_token.shape[0]
                n_end_sil = end_sil_mask.sum()
                if n_end_sil > 0:
                    spec_end = torch.ones(mel_len)
                    spec_end[-n_end_sil:] = (1 - torch.arange(0, n_end_sil) / n_end_sil) ** 2
                    spec_end = spec_end[..., None] * (sample['mel'] - vmin) + vmin
                else:
                    spec_end = sample['mel']

                sample['mel'] = spec * ((~begin_sil_mask) & (~end_sil_mask)).float()[:, None] + \
                                spec_begin * begin_sil_mask.float()[:, None] + \
                                spec_end * end_sil_mask.float()[:, None]

        if hparams['use_spk_embed']:
            sample["spk_embed"] = torch.Tensor(item['spk_embed'])
        if hparams['use_spk_id']:
            if hparams.get('infer_spk_name', '') != '':
                sample["spk_id"] = int(self.spk_map[hparams['infer_spk_name']])
            else:
                sample["spk_id"] = int(self.spk_map[item['spk_name']]) \
                    if 'spk_id' not in item else int(item['spk_id'])
        return sample

    def collater(self, samples):
        if len(samples) == 1 and isinstance(samples[0], list):
            samples = samples[0]
        if len(samples) == 0:
            return {}
        hparams = self.hparams
        id = torch.LongTensor([s.get('id', 0) for s in samples])
        item_names = [s['item_name'] for s in samples]
        text = [s.get('text', '') for s in samples]
        txt_tokens = collate_xd([s['txt_token'] for s in samples], 0)
        txt_lengths = torch.LongTensor([s['txt_token'].numel() for s in samples])

        batch = {
            'id': id,
            'item_name': item_names,
            'nsamples': len(samples),
            'text': text,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
        }
        if self.load_mel:
            mels = collate_xd([s['mel'] for s in samples], 0.0)
            mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples])
            batch.update({
                'mels': mels,
                'mel_lengths': mel_lengths,
            })

        if hparams['use_spk_embed']:
            spk_embed = collate_xd([s['spk_embed'] for s in samples])
            batch['spk_embed'] = spk_embed
        if hparams['use_spk_id']:
            spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
            batch['spk_ids'] = spk_ids
        return batch

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return min(self._sizes[index], self.hparams['max_frames'])

    @staticmethod
    def get_mel(hparams, wav):
        if isinstance(wav, str):
            wav, sr = librosa.core.load(wav, sr=hparams['audio_sample_rate'])
            ws = hparams['win_size']
            if len(wav) % ws < ws - 1:
                wav = np.pad(wav, (0, ws - 1 - (len(wav) % ws)), mode='constant', constant_values=0.0)
        h, w, m = hparams['acous_params'][-1]
        if hparams.get('use_stft_spec', False):
            wav2spec_dict = librosa_wav2linearspec(
                wav,
                fft_size=hparams['fft_size'],
                hop_size=h,
                win_length=hparams['win_size'],
                num_mels=m,
                fmin=hparams['fmin'],
                fmax=hparams['fmax'],
                sample_rate=hparams['audio_sample_rate'],
                center=False)
            mel = wav2spec_dict['linear']
        else:
            wav2spec_dict = librosa_wav2spec(
                wav,
                fft_size=hparams['fft_size'],
                hop_size=h,
                win_length=hparams['win_size'],
                num_mels=m,
                fmin=hparams['fmin'],
                fmax=hparams['fmax'],
                sample_rate=hparams['audio_sample_rate'],
                center=False)
            mel = wav2spec_dict['mel']
        if hparams.get('reduce_transient_noise'):
            from utils.audio.noise_reduction import reduce_transient_noise
            mel = reduce_transient_noise(mel)
        wav = wav2spec_dict['wav']
        return mel, wav


class FastSpeechDataset(BaseTTSDataset):
    def get_sample(self, id, item):
        sample = super().get_sample(id, item)
        if sample is None:
            return None
        hparams = self.hparams
        ph_token = sample['txt_token']
        sample['char_token'] = char_token = torch.LongTensor(item['char_encoded'])
        sample['ph2char'] = ph2char = torch.LongTensor(item['ph2char'])[:hparams['max_input_tokens']]
        for l in hparams['ling_labels']:
            if l in ['emo', 'srate']:
                sample[l] = torch.LongTensor([item[f'{l}_encoded']] * len(ph_token))
            else:
                sample[l] = torch.LongTensor(item[f'{l}_encoded'][:hparams['max_input_tokens']])
        if 'bert_embed' in item:
            sample['bert_embed'] = torch.FloatTensor(item['bert_embed'])
        mel2ph = torch.LongTensor(item['mel2ph'])

        max_frames = hparams['max_frames']
        mel2ph = mel2ph[:max_frames]
        max_frames = mel2ph.shape[0] // hparams['frames_multiple'] * hparams['frames_multiple']
        sample['mel2ph'] = mel2ph[:max_frames]

        if self.load_mel:
            mel = sample['mel']
            T = mel.shape[0]
            hparams = self.hparams
            sample["f0"], sample["uv"], sample['pitch'] = None, None, None
            if 'sil_mask' in item:
                sil_mask = ~item['sil_mask'][:T]
                sample['sil_mask'] = torch.LongTensor(sil_mask)
        return sample

    def get_perturb_mel(self, sample, item):
        hparams = self.hparams
        T = sample['mel'].shape[0]
        if self.perturb_aug is None:
            from tasks.tts.nansypp_utils.augment import Augment
            from tasks.tts.nansypp_utils.config import AugConfig
            self.perturb_aug = Augment(AugConfig(
                self.hparams['audio_sample_rate'],
                self.hparams['hop_size'], self.hparams['win_size']))
        h, w, m = hparams['acous_params'][-1]

        def sampler(ratio):
            shifts = torch.rand([1]) * (ratio - 1.) + 1.
            # flip
            flip = torch.rand([1]) < 0.5
            shifts[flip] = shifts[flip] ** -1
            return shifts

        s = sampler(1.4)
        wav2spec_dict = librosa_wav2spec(
            self.perturb_aug(
                torch.FloatTensor(item['wav'][None, :]),
                formant_shift=s).numpy()[0],
            fft_size=w,
            hop_size=h,
            win_length=w,
            num_mels=m,
            fmin=hparams['fmin'],
            fmax=hparams['fmax'],
            sample_rate=hparams['audio_sample_rate'],
            center=False)
        mel_psd = wav2spec_dict['mel'][:T]
        sample['mel_psd'] = torch.FloatTensor(mel_psd)

    def collater(self, samples):
        batch = super(FastSpeechDataset, self).collater(samples)
        if len(samples) == 1 and isinstance(samples[0], list):
            samples = samples[0]
        samples = [x for x in samples if x is not None]
        if len(samples) == 0:
            return {}
        hparams = self.hparams
        for l in hparams.get('ling_labels', []):
            batch[l] = collate_xd([s[l] for s in samples])
        if hparams.get('use_char'):
            batch['char_tokens'] = collate_xd([s['char_token'] for s in samples])
            batch['char_lengths'] = torch.LongTensor([s['char_token'].numel() for s in samples])
            batch['ph2char'] = collate_xd([s['ph2char'] for s in samples])
        if 'bert_embed' in samples[0]:
            batch['bert_embed'] = collate_xd([s['bert_embed'] for s in samples], 0)
        mel2ph = collate_xd([s['mel2ph'] for s in samples])
        batch['mel2ph'] = mel2ph
        # Match the length of mel2ph and mels (if the win_size changes, the length may be padded)
        if 'mels' in batch:
            T = min(mel2ph.shape[1], batch['mels'].shape[1])
            batch['mel2ph'], batch['mels'] = batch['mel2ph'][:, :T], batch['mels'][:, :T]
        if self.load_mel:
            f0, uv, pitch, f0_ph, uv_ph = None, None, None, None, None
            if hparams.get('use_pitch_embed', False) or hparams.get('ds_add_pitch_embed', False) or \
                    hparams.get('predict_aux_pitch', False):
                f0 = collate_xd([s['f0'] for s in samples])
                uv = collate_xd([s['uv'] for s in samples])
                pitch = collate_xd([s['pitch'] for s in samples])
                if hparams.get('use_ph_level_f0', False):
                    f0_ph = collate_xd([s['f0_ph'] for s in samples])
                    uv_ph = collate_xd([s['uv_ph'] for s in samples])
            batch.update({
                'f0': f0,
                'pitch': pitch,
                'uv': uv,
                'f0_ph': f0_ph,
                'uv_ph': uv_ph,
            })

            if self.hparams.get('multistage'):
                batch['mels0'] = collate_xd([s['mel0'] for s in samples])
            if 'sil_mask' in samples[0]:
                batch['sil_mask'] = collate_xd([s['sil_mask'] for s in samples])
            if "mel_env" in samples[0]:
                batch['mels_env'] = collate_xd([s['mel_env'] for s in samples], 0.0)
            if 'w2v' in samples[0]:
                batch['w2v'] = collate_xd([s['w2v'] for s in samples], 0.0)
            if 'mel_psd' in samples[0]:
                batch['mels_psd'] = collate_xd([s['mel_psd'] for s in samples], 0.0)
        return batch


class MegaTTSDataset(FastSpeechDataset):
    def __init__(self, prefix, l=None, shuffle=False, data_dir=None, chunk_size=5, hparams=None):
        super(BaseSpeechDataset, self).__init__(f'{data_dir}/data', shuffle, chunk_size)
        if hparams is None:
            from utils.commons.hparams import hparams
        hparams = deepcopy(hparams)
        self.hparams = hparams
        self.load_mel = True
        self.data_dir = hparams['binary_data_dir'] if data_dir is None else data_dir
        self.prefix = prefix
        self.id2item_names = {}
        if l is None:
            reader = self.get_reader()
            l = len(reader.list_keys())
        self.ds_len = l

    def __getitem__(self, index):
        max_timbre_len = self.hparams['max_timbre_len']
        index = [index]
        item_names = [self.id2item_names[str(i)] for i in index]
        items = self.indexed_kv_ds.read_many(item_names)
        item_spks = [self.name2spk(self.id2item_names[str(i)][0]) for i in index]
        item_names_ref, ks_ref = self.get_timbre_items(item_names, item_spks, max_timbre_len)
        items_timbre = self.indexed_kv_ds.read_many(item_names_ref)

        samples = []
        for i, (item, spk, idx) in enumerate(zip(items, item_spks, index)):
            item = pickle.loads(item)
            items_ref = [pickle.loads(x) for x in items_timbre[:ks_ref[i]]]
            items_timbre = items_timbre[ks_ref[i]:]
            assert len(items_ref) == ks_ref[i]
            items_ref = [x for x in items_ref if x['item_name'] != item['item_name']]
            if len(items_ref) == 0:
                items_ref = [item]
            sample = self.get_sample(idx, item, items_ref, max_timbre_len)
            if sample is not None:
                samples.append(sample)
        assert len(items_timbre) == 0
        return samples

    def get_timbre_items(self, item_names, item_spks, max_timbre_len, include_self=False):
        ks_ref = [0 for _ in item_spks]
        item_names_ref = []
        for i, (item_name, spk) in enumerate(zip(item_names, item_spks)):
            itemnames = self.get_items_by_same_spk(item_name)
            cur_id = itemnames.index(item_name)
            ref_id = cur_id - 1
            ref_size = 0
            item_names_ref_ = []
            while ref_id >= 0 and ref_size < max_timbre_len:
                ref_size = ref_size + self.name2mellen[itemnames[ref_id]]
                item_names_ref_.append(itemnames[ref_id])
                ks_ref[i] += 1
                ref_id -= 1
                if ref_size >= max_timbre_len and len(item_names_ref_) > 0:
                    break
            item_names_ref_ = item_names_ref_[::-1]
            if include_self:
                item_names_ref_.append(itemnames[cur_id])
                ks_ref[i] += 1
            ref_id = cur_id + 1
            while ref_id < len(itemnames) and ref_size < max_timbre_len:
                ref_size = ref_size + self.name2mellen[itemnames[ref_id]]
                item_names_ref_.append(itemnames[ref_id])
                ks_ref[i] += 1
                ref_id += 1
                if ref_size >= max_timbre_len and len(item_names_ref_) > 0:
                    break
            item_names_ref = item_names_ref + item_names_ref_
        return item_names_ref, ks_ref

    def get_sample(self, id, item, items_timbre=None, max_timbre_len=None):
        sample = super(MegaTTSDataset, self).get_sample(id, item)
        if sample is None:
            return None
        if max_timbre_len is None:
            max_timbre_len = self.hparams['max_timbre_len']
        if self.load_mel:
            hparams = self.hparams
            if items_timbre is not None:
                assert len(items_timbre) > 0, item['item_name']
                if self.hparams['use_spk_embed']:
                    sample["spk_embed"] = torch.Tensor(
                        np.stack([x['spk_embed'] for x in items_timbre], 0))
                mel_t_size = 0
                mels_timbre = []
                wavs_timbre = []
                for item_timbre in items_timbre:
                    if mel_t_size < max_timbre_len:
                        if 'mel' not in item_timbre:
                            if hparams.get('use_env_modeling'):
                                item_timbre['mel'], _ = self.get_mel(self.hparams, item_timbre['denoised_fn'])
                            else:
                                item_timbre['mel'], item_timbre['wav'] = self.get_mel(self.hparams, item_timbre['wav'])
                        mel_timbre_ = item_timbre['mel']
                        wav_timbre_ = item_timbre['wav']
                        mel_t_size += mel_timbre_.shape[0]
                        mels_timbre.append(mel_timbre_)
                        wavs_timbre.append(wav_timbre_)
                if self.hparams.get('use_intra_sent_timbre'):
                    sample['mel_timbre'] = sample['mel']
                else:
                    mels_timbre = np.concatenate(mels_timbre, 0)
                    wavs_timbre = np.concatenate(wavs_timbre, 0)
                    sample['mel_timbre'] = torch.FloatTensor(mels_timbre)
                    sample['wav_timbre'] = torch.FloatTensor(wavs_timbre)
                if max_timbre_len > 1:
                    sample['mel_timbre'] = sample['mel_timbre'][:max_timbre_len]
                else:
                    sample['mel_timbre'] = sample['mel_timbre'][:1000]
            else:
                sample['mel_timbre'] = sample['mel']
                sample['wav_timbre'] = sample['wav']
        return sample

    def collater(self, samples):
        batch = super(MegaTTSDataset, self).collater(samples)
        if len(samples) == 1 and isinstance(samples[0], list):
            samples = samples[0]
        samples = [x for x in samples if x is not None]
        if len(samples) == 0:
            return {}
        if 'spk_id' in samples[0]:
            spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
            batch['spk_ids'] = spk_ids
            spk_pos_ids_flat = []
            spk_pos_start = 0
            cur_spk_id = 0
            for idx in range(len(samples)):
                spk_id = spk_ids[idx]
                txt_token = samples[idx]['txt_token']
                l_token = txt_token.shape[0]
                if cur_spk_id != spk_id:
                    spk_pos_start = 0
                    cur_spk_id = spk_id
                spk_pos_ids_flat += range(spk_pos_start, spk_pos_start + l_token)
                if self.hparams.get('sent_level_pos'):
                    spk_pos_start = 0
                else:
                    spk_pos_start += l_token
            batch['spk_pos_ids_flat_ph'] = torch.LongTensor([spk_pos_ids_flat])

        if self.load_mel:
            if 'mel_timbre' in samples[0]:
                batch['mels_timbre'] = collate_xd([s.get('mel_timbre', s['mel']) for s in samples])
            if 'spk_id' in samples[0]:
                spk_pos_ids_flat = []
                spk_pos_start = 0
                cur_spk_id = 0
                for idx in range(len(samples)):
                    spk_id = spk_ids[idx]
                    mel2ph = samples[idx]['mel2ph']
                    if cur_spk_id != spk_id:
                        spk_pos_start = 0
                        cur_spk_id = spk_id
                    spk_pos_ids_flat += range(spk_pos_start,
                                              spk_pos_start + mel2ph[::self.hparams['vq_stride']].shape[0])
                    spk_pos_start += mel2ph[::self.hparams['vq_stride']].shape[0]
                batch['spk_pos_ids_flat'] = torch.LongTensor([spk_pos_ids_flat])
        if 'wav' in samples[0]:
            batch['wavs'] = collate_xd([s['wav'] for s in samples])
        if 'wav_timbre' in samples[0]:
            batch['wavs_timbre'] = collate_xd([s['wav_timbre'] for s in samples])
        return batch

    def get_key_and_sizes(self):
        pass

    def __len__(self):
        return self.ds_len

    @classmethod
    def merge_samples(cls, samples, mel_timbre=None, fm=None):
        sample_merged = {
            'id': 0,
            'item_name': '|||'.join([s['item_name'] for s in samples]),
            'txt_token': torch.cat([s['txt_token'] for s in samples], 0),
            'mel_timbre': mel_timbre,
            'char_token': torch.cat([s['char_token'] for s in samples], 0) if valid_item_kv(samples[0], 'char_token') else None,
            'mel': torch.cat([s['mel'] for s in samples], 0) if valid_item_kv(samples[0], 'mel') else None,
            'wav': torch.cat([s['wav'] for s in samples], 0) if valid_item_kv(samples[0], 'wav') else None,
            'mel2ph': cls.merge_A2B(
                [s['mel2ph'] for s in samples], [len(s['txt_token']) for s in samples]),
            'ph2char': cls.merge_A2B(
                [s['ph2char'] for s in samples], [len(s['char_token']) for s in samples]) if valid_item_kv(samples[0], 'ph2char') else None,
            'tone': torch.cat([s['tone'] for s in samples], 0),
            'txt': ' '.join([s['txt'] for s in samples]),
            'spk_name': samples[0]['spk_name']
        }
        if fm is not None:
            t = sample_merged['mel'].shape[0] // fm * fm
            sample_merged['mel'] = sample_merged['mel'][:t]
            sample_merged['mel2ph'] = sample_merged['mel2ph'][:t]
            sample_merged['mel_timbre'] = sample_merged['mel_timbre'][:sample_merged['mel_timbre'].shape[0] // fm * fm]
        return sample_merged

    @classmethod
    def merge_A2B(cls, A2B, B_lens):
        token_lens_cumsum = np.cumsum([0] + B_lens[:-1])
        token_lens_cumsum = torch.LongTensor(token_lens_cumsum)
        for i in range(len(B_lens)):
            A2B[i] = A2B[i] + token_lens_cumsum[i]
        A2B = torch.cat(A2B, 0)
        return A2B

    @staticmethod
    def split_sample(item, split_pos, force_word_bdr=True, split_by_ph=False, hop_size=240):
        mel2ph = item['mel2ph']
        ph2char = item['ph2char']
        if split_by_ph:
            pos_ph = split_pos
            split_pos = 0
            while split_pos < len(mel2ph) - 1 and mel2ph[split_pos] - 1 < pos_ph:
                split_pos += 1
        if split_pos >= len(item['mel2ph']):
            return item, None
        if not split_by_ph:
            pos_ph = mel2ph[split_pos] - 1

        if force_word_bdr:
            # if pos_ph is in the middle of a word
            while 1 <= pos_ph < len(ph2char) - 1 and ph2char[pos_ph] == ph2char[pos_ph - 1]:
                pos_ph += 1
            while split_pos < len(mel2ph) - 1 and mel2ph[split_pos] - 1 < pos_ph:
                split_pos += 1
        pos_char = ph2char[pos_ph] - 1

        # build item_left
        mel2ph_left = item['mel2ph'][:split_pos]
        if len(mel2ph_left) == 0:
            ph_max = 0
        else:
            ph_max = mel2ph_left[-1]
        ph2char_left = item['ph2char'][:ph_max + 1]
        char_max = ph2char_left[-1]
        item_left = {
            'id': 0,
            'item_name': item['item_name'],
            'mel2ph': mel2ph_left,
            'ph2char': ph2char_left,
            'txt_token': item['txt_token'][:ph_max + 1],
            'tone': item['tone'][:ph_max + 1],
            'char_token': item['char_token'][:char_max + 1],
            'mel': item['mel'][:split_pos] if item.get('mel') is not None else None,
            'wav': item['wav'][:split_pos * hop_size] if item.get('mel') is not None else None,
        }

        # build item_right
        item_right = {
            'id': 0,
            'item_name': item['item_name'],
            'mel2ph': mel2ph[split_pos:] - pos_ph,
            'txt_token': item['txt_token'][pos_ph:],
            'tone': item['tone'][pos_ph:],
            'ph2char': item['ph2char'][pos_ph:] - pos_char,
            'char_token': item['char_token'][pos_char:],
            'mel': item['mel'][split_pos:] if item.get('mel') is not None else None,
            'wav': item['wav'][split_pos * hop_size:] if item.get('wav') is not None else None,
        }
        return item_left, item_right


class StaticBatchDistributedSampler(DistributedSampler):
    def __init__(self, epoch_steps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.indices = []
        self.epoch_steps = epoch_steps

    def __iter__(self):
        if len(self.indices) == 0:
            from utils.commons.hparams import hparams
            l = len(self.dataset)
            self.ref_sizes = np.random.randint(hparams['ref_size_min'], hparams['ref_size_max'] + 1, l)
            self.tgt_sizes = np.random.randint(hparams['tgt_size_min'], hparams['tgt_size_max'] + 1, l)

            if self.shuffle:
                # deterministically shuffle based on epoch and seed
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
            else:
                indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

            if not self.drop_last:
                # add extra samples to make it evenly divisible
                padding_size = self.total_size - len(indices)
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
            else:
                # remove tail of data to make it evenly divisible.
                indices = indices[:self.total_size]
            assert len(indices) == self.total_size

            # subsample
            indices = indices[self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
            self.indices = indices

        it = iter(zip(self.indices[:self.epoch_steps],
                      self.ref_sizes[:self.epoch_steps],
                      self.tgt_sizes[:self.epoch_steps]))
        self.indices = self.indices[self.epoch_steps:]
        self.ref_sizes = self.ref_sizes[self.epoch_steps:]
        self.tgt_sizes = self.tgt_sizes[self.epoch_steps:]
        return it


class MegaTTSDurationPredictorDataset(MegaTTSDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_mel = self.hparams.get('test_dataset', True)

    def __getitem__(self, index):
        inames_slt = []
        spk_ids = []
        size = 0
        full = False
        spk_id = 0
        max_ph_tokens = self.hparams['max_tokens']
        while not full:
            k = self.id2item_names[str(index)]
            itemnames = self.get_items_by_same_spk(k)
            if self.hparams['shuffle_ref']:
                random.shuffle(itemnames)
            start_id = itemnames.index(k)
            itemnames = itemnames[start_id:] + itemnames[:start_id]
            for name in itemnames:
                size += self.name2phlen[name]
                inames_slt.append(name)
                spk_ids.append(spk_id)
                if size >= max_ph_tokens:
                    full = True
                    break
            index = random.randint(0, self.ds_len - 1)
            spk_id += 1
        items = self.indexed_kv_ds.read_many(inames_slt)
        samples = []
        for item, spk_id in zip(items, spk_ids):
            item = pickle.loads(item)
            item['spk_id'] = spk_id
            sample = self.get_sample(index, item)
            if sample is not None:
                samples.append(sample)
        return samples

    def get_sample(self, id, item, items_timbre=None):
        sample = super().get_sample(id, item, items_timbre)
        sample['spk_id'] = item.get('spk_id', 0)

        txt_tokens_ref = []
        tones_ref = []
        dur_ref = []
        if items_timbre is not None:
            assert len(items_timbre) > 0, item['item_name']
            for item_timbre in items_timbre:
                txt_token_ref = item_timbre['phone_encoded']
                dur = mel2token_to_dur(item_timbre['mel2ph'], len(txt_token_ref))
                dur = dur + 1
                dur_ref.append(dur)
                txt_tokens_ref.append(txt_token_ref)
                tones_ref.append(item_timbre['tone_encoded'])
            sample['dur_ref'] = torch.LongTensor(np.concatenate(dur_ref, 0))
            sample['txt_tokens_ref'] = torch.LongTensor(np.concatenate(txt_tokens_ref, 0))
            sample['tones_ref'] = torch.LongTensor(np.concatenate(tones_ref, 0))
        return sample

    def collater(self, samples):
        batch = super().collater(samples)
        samples = samples[0]
        if len(samples) == 0:
            return {}
        spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
        batch['spk_ids'] = spk_ids
        if 'dur_ref' in samples[0]:
            batch['dur_ref'] = collate_xd([s['dur_ref'] for s in samples])
            batch['txt_tokens_ref'] = collate_xd([s['txt_tokens_ref'] for s in samples])
            batch['tones_ref'] = collate_xd([s['tones_ref'] for s in samples])

        # 'spk_pos_ids_flat' is the pos id for positional embedding
        spk_pos_ids_flat = []
        spk_pos_start = 0
        cur_spk_id = 0
        for idx in range(len(samples)):
            spk_id = spk_ids[idx]
            txt_token = samples[idx]['txt_token']
            l_token = txt_token.shape[0]
            if cur_spk_id != spk_id:
                spk_pos_start = 0
                cur_spk_id = spk_id
            spk_pos_ids_flat += range(spk_pos_start, spk_pos_start + l_token)
            if self.hparams.get('sent_level_pos'):
                spk_pos_start = 0
            else:
                spk_pos_start += l_token
        batch['spk_pos_ids_flat_ph'] = torch.LongTensor([spk_pos_ids_flat])
        return batch


class FrontendLMDataset(BaseSpeechDataset):
    def __init__(self, prefix, l=None, shuffle=False, data_dir=None, chunk_size=5, hparams=None):
        super(BaseSpeechDataset, self).__init__(f'{data_dir}/data', shuffle, chunk_size)
        if hparams is None:
            from utils.commons.hparams import hparams
        hparams = deepcopy(hparams)
        self.hparams = hparams
        self.load_mel = True
        self.data_dir = hparams['binary_data_dir'] if data_dir is None else data_dir
        self.prefix = prefix
        self.id2item_names = {}
        if l is None:
            reader = self.get_reader()
            l = len(reader.list_keys())
        self.ds_len = l

    def __getitem__(self, index):
        index = [index]
        item_names = [self.id2item_names[str(i)] for i in index]
        items = self.indexed_kv_ds.read_many(item_names)

        samples = []
        for i, (item, idx) in enumerate(zip(items, index)):
            item = pickle.loads(item)
            sample = self.get_sample(idx, item)
            if sample is not None:
                samples.append(sample)
        return samples
    
    def get_sample(self, id, item):
        import whisper
        hparams = self.hparams
        ph_token = torch.LongTensor(item['phone_encoded'][:hparams['max_input_tokens']])
        tone_token = torch.LongTensor(item['tone_encoded'][:hparams['max_input_tokens']])

        sample = {
            "id": id,
            "item_name": item['item_name'],
            "text": item['txt_raw'],
            "txt_token": ph_token,
            'tone': tone_token,
        }

        max_frames = len(item['mel2ph']) // hparams['frames_multiple'] * hparams['frames_multiple']
        sample['mel2ph'] = torch.LongTensor(item['mel2ph'])[:max_frames]
        sample['mel2ph'][-1] = sample['mel2ph'][-2]

        sample['ph_timestamp'] = self.get_ph_timestamp(sample)

        whisper_wav = librosa.resample(item['wav'].astype(np.float32), orig_sr=24000, target_sr=16000)
        sample['mel'] = whisper.log_mel_spectrogram(whisper_wav).T
        return sample
    
    @staticmethod
    def get_ph_timestamp(item, pad_bos_eos=True):
        # Obtain the timestamp of each phone, we map it to 800~6800 (max_frame is 6000)
        mel2ph = item['mel2ph']
        cur_global_time = 0
        ph_timestamp = []
        for i in range(1, mel2ph.max()+1):
            cur_ph_time = len(mel2ph[mel2ph == i])
            ph_timestamp.append(cur_global_time + cur_ph_time)
            cur_global_time += cur_ph_time
        ph_timestamp = torch.Tensor(ph_timestamp) + 800
        assert ph_timestamp[-1] < 6800

        # Merge Chinese phone and tone (Original dict ends at 173, i.e., ph_dict_size=173). 146~173 is punctuations.
        phone = item['txt_token'].clone()
        merged_phone = item['txt_token'].clone()
        tone_tmp = item['tone'].clone()
        phone, merged_phone, tone_tmp = phone[:mel2ph.max()], merged_phone[:mel2ph.max()], tone_tmp[:mel2ph.max()]
        # In tone_dict, tone_1 is 4, tone_2 is 11, tone_3 is 12, tone_4 is 13, tone_5 is 14, tone_6 is 15
        tone_tmp[tone_tmp==4] = 1
        tone_tmp[tone_tmp==11] = 2
        tone_tmp[tone_tmp==12] = 3
        tone_tmp[tone_tmp==13] = 4
        tone_tmp[tone_tmp==14] = 5
        tone_tmp[tone_tmp==15] = 6
        # Chinese phones lie in 3~100 in the phone_dict, we map them to 200~788
        ch_phone_idx = (phone >= 3) & (phone <= 100)
        merged_phone[ch_phone_idx] = (merged_phone[ch_phone_idx] - 3) * 6 + 200 + tone_tmp[ch_phone_idx]
        
        merged_phone = merged_phone[:ph_timestamp.size(0)]
        assert ph_timestamp.shape == merged_phone.shape, (ph_timestamp.shape, merged_phone.shape, mel2ph.max(), len(item['txt_token']))
        ph_timestamp = torch.stack([merged_phone, ph_timestamp], dim=1)
        ph_timestamp = ph_timestamp.view(-1).contiguous().long()

        # append ph_timestamp's start and end token
        if pad_bos_eos:
            ph_timestamp = F.pad(ph_timestamp, (1, 0), mode='constant', value=798)
            ph_timestamp = F.pad(ph_timestamp, (0, 1), mode='constant', value=799)
        return ph_timestamp

    @staticmethod
    def map_phone_to_tokendict(item, pad_bos_eos=True):
        # Merge Chinese phone and tone (Original dict ends at 173, i.e., ph_dict_size=173). 146~173 is punctuations.
        phone = item['txt_token'].clone()
        merged_phone = item['txt_token'].clone()
        tone_tmp = item['tone'].clone()
        # In tone_dict, tone_1 is 4, tone_2 is 11, tone_3 is 12, tone_4 is 13, tone_5 is 14, tone_6 is 15
        tone_tmp[tone_tmp==4] = 1
        tone_tmp[tone_tmp==11] = 2
        tone_tmp[tone_tmp==12] = 3
        tone_tmp[tone_tmp==13] = 4
        tone_tmp[tone_tmp==14] = 5
        tone_tmp[tone_tmp==15] = 6
        # Chinese phones lie in 3~100 in the phone_dict, we map them to 200~788
        ch_phone_idx = (phone >= 3) & (phone <= 100)
        merged_phone[ch_phone_idx] = (merged_phone[ch_phone_idx] - 3) * 6 + 200 + tone_tmp[ch_phone_idx]

        if pad_bos_eos:
            merged_phone = F.pad(merged_phone, (1, 0), mode='constant', value=798)
            merged_phone = F.pad(merged_phone, (0, 1), mode='constant', value=799)
        return merged_phone
    
    @staticmethod
    def split_ph_timestamp(ph_timestamp):
        ''' Input: ph_timestamp, shape [T] '''

        # Map the timestamp of each phone back to its original frame-level lengths
        ph_timestamp[ph_timestamp >= 800] -= 800

        ph_list = []
        tone_list = []
        dur_list = []
        cur_timestamp = 0
        for idx, item in enumerate(ph_timestamp):
            if idx % 2 == 0:
                # Map Chinese phones back to its original phone_dict
                if (200 <= item <= 788):
                    ph = (item - 200 - 1) // 6 + 3
                    tone = (item - 200 - 1) % 6 + 1
                    if tone == 1:
                        tone = 4
                    else:
                        tone = tone + 9
                # Set English tone to '3'
                else:
                    ph = item
                    tone = 3
                ph_list.append(ph)
                tone_list.append(tone)
            else:
                dur_list.append((item - cur_timestamp))
                cur_timestamp = item
        assert len(ph_list) == len(dur_list)
        ph_seq, tone_seq, dur_seq = torch.LongTensor(ph_list), torch.LongTensor(tone_list), torch.LongTensor(dur_list)
        return ph_seq, tone_seq, dur_seq, ph_timestamp[-1]
    
    @staticmethod
    def split_ph(ph_seq):
        ''' Input: ph_timestamp, shape [T] '''
        ph_list = []
        tone_list = []
        for idx, item in enumerate(ph_seq):
            # Map Chinese phones back to its original phone_dict
            if (200 <= item <= 788):
                ph = (item - 200 - 1) // 6 + 3
                tone = (item - 200 - 1) % 6 + 1
                if tone == 1:
                    tone = 4
                else:
                    tone = tone + 9
            # Set English tone to '3'
            else:
                ph = item
                tone = 3
            ph_list.append(ph)
            tone_list.append(tone)
           
        assert len(ph_list) == len(tone_list)
        ph_seq, tone_seq = torch.LongTensor(ph_list), torch.LongTensor(tone_list)
        return ph_seq, tone_seq

    def collater(self, samples):
        if len(samples) == 1 and isinstance(samples[0], list):
            samples = samples[0]
        samples = [x for x in samples if x is not None]
        if len(samples) == 0:
            return {}

        hparams = self.hparams
        batch = {}
        batch['id'] = torch.LongTensor([s.get('id', 0) for s in samples])
        batch['item_name'] = [s['item_name'] for s in samples]
        batch['nsamples'] = len(samples)

        # Collate ph_timestamp
        batch['ph_timestamp'] = collate_xd([s['ph_timestamp'] for s in samples], 797)
        batch['ph_timestamp_len'] = torch.LongTensor([s['ph_timestamp'].shape[0] for s in samples])
        # Collate text and mel
        batch['mel'] = collate_xd([s['mel'] for s in samples])
        batch['mel_len'] = torch.LongTensor([s['mel'].shape[0] for s in samples])
        return batch
    
    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return min(self._sizes[index], self.hparams['max_frames'])
    
    def get_key_and_sizes(self):
        pass

    def __len__(self):
        return self.ds_len


class G2PLMDataset(BaseSpeechDataset):
    def __init__(self, prefix, l=None, shuffle=False, data_dir=None, chunk_size=5, hparams=None):
        super(BaseSpeechDataset, self).__init__(f'{data_dir}/data', shuffle, chunk_size)
        if hparams is None:
            from utils.commons.hparams import hparams
        hparams = deepcopy(hparams)
        self.hparams = hparams
        self.load_mel = True
        self.data_dir = hparams['binary_data_dir'] if data_dir is None else data_dir
        self.prefix = prefix
        self.id2item_names = {}
        if l is None:
            reader = self.get_reader()
            l = len(reader.list_keys())
        self.ds_len = l
    
    def __getitem__(self, index):
        index = [index]
        item_names = [self.id2item_names[str(i)] for i in index]
        items = self.indexed_kv_ds.read_many(item_names)

        samples = []
        for i, (item, idx) in enumerate(zip(items, index)):
            item = pickle.loads(item)
            sample = self.get_sample(idx, item)
            if sample is not None:
                samples.append(sample)
        return samples

    def get_sample(self, id, item):
        hparams = self.hparams
        ph_token = torch.LongTensor(item['phone_encoded'][:hparams['max_input_tokens']])
        tone_token = torch.LongTensor(item['tone_encoded'][:hparams['max_input_tokens']])

        sample = {
            "id": id,
            "item_name": item['item_name'],
            "text": item['txt_raw'],
            "txt_token": ph_token,
            'tone_token': tone_token,
        }

        sample['merged_ph'] = self.map_phone_to_tokendict(sample)
        return sample

    def collater(self, samples):
        if len(samples) == 1 and isinstance(samples[0], list):
            samples = samples[0]
        if len(samples) == 0:
            return {}
        hparams = self.hparams
        id = torch.LongTensor([s.get('id', 0) for s in samples])
        item_names = [s['item_name'] for s in samples]
        
        merged_ph = collate_xd([s['merged_ph'] for s in samples], 797)
        merged_ph_len = torch.LongTensor([s['merged_ph'].size(-1) for s in samples])

        text = ['[ASR_BOS]' + '[FULL]' + s['text'] + '[ASR_EOS]' for s in samples]

        batch = {
            'id': id,
            'item_name': item_names,
            'nsamples': len(samples),
            'text': text,
            'merged_ph': merged_ph,
            'merged_ph_len': merged_ph_len,
        }
        return batch

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return min(self._sizes[index], self.hparams['max_frames'])
    
    @staticmethod
    def map_phone_to_tokendict(item, pad_bos_eos=True):
        # Merge Chinese phone and tone (Original dict ends at 173, i.e., ph_dict_size=173). 146~173 is punctuations.
        phone = item['txt_token'].clone()
        merged_phone = item['txt_token'].clone()
        tone_tmp = item['tone_token'].clone()
        # In tone_dict, tone_1 is 4, tone_2 is 11, tone_3 is 12, tone_4 is 13, tone_5 is 14, tone_6 is 15
        tone_tmp[tone_tmp==4] = 1
        tone_tmp[tone_tmp==11] = 2
        tone_tmp[tone_tmp==12] = 3
        tone_tmp[tone_tmp==13] = 4
        tone_tmp[tone_tmp==14] = 5
        tone_tmp[tone_tmp==15] = 6
        # Chinese phones lie in 3~100 in the phone_dict, we map them to 200~788
        ch_phone_idx = (phone >= 3) & (phone <= 100)
        merged_phone[ch_phone_idx] = (merged_phone[ch_phone_idx] - 3) * 6 + 200 + tone_tmp[ch_phone_idx]

        if pad_bos_eos:
            merged_phone = F.pad(merged_phone, (1, 0), mode='constant', value=798)
            merged_phone = F.pad(merged_phone, (0, 1), mode='constant', value=799)
        return merged_phone
    
    @staticmethod
    def split_ph(ph_seq):
        ''' Input: ph_timestamp, shape [T] '''
        ph_list = []
        tone_list = []
        for idx, item in enumerate(ph_seq):
            # Map Chinese phones back to its original phone_dict
            if (200 <= item <= 788):
                ph = (item - 200 - 1) // 6 + 3
                tone = (item - 200 - 1) % 6 + 1
                if tone == 1:
                    tone = 4
                else:
                    tone = tone + 9
            # Set English tone to '3'
            else:
                ph = item
                tone = 3
            ph_list.append(ph)
            tone_list.append(tone)
           
        assert len(ph_list) == len(tone_list)
        ph_seq, tone_seq = torch.LongTensor(ph_list), torch.LongTensor(tone_list)
        return ph_seq, tone_seq
    
    def get_key_and_sizes(self):
        pass

    def __len__(self):
        return self.ds_len


class SDVAEDataset(FastSpeechDataset):
    def __init__(self, prefix, l=None, shuffle=False, data_dir=None, chunk_size=5, hparams=None):
        super(BaseSpeechDataset, self).__init__(f'{data_dir}/data', shuffle, chunk_size)
        if hparams is None:
            from utils.commons.hparams import hparams
        hparams = deepcopy(hparams)
        self.hparams = hparams
        self.load_mel = True
        self.data_dir = hparams['binary_data_dir'] if data_dir is None else data_dir
        self.prefix = prefix
        self.id2item_names = {}
        if l is None:
            reader = self.get_reader()
            l = len(reader.list_keys())
        self.ds_len = l
    
    def __getitem__(self, index):
        index = [index]
        item_names = [self.id2item_names[str(i)] for i in index]
        items = self.indexed_kv_ds.read_many(item_names)

        samples = []
        for i, (item, idx) in enumerate(zip(items, index)):
            item = pickle.loads(item)
            sample = self.get_sample(idx, item)
            if sample is not None:
                samples.append(sample)
        return samples

    def get_sample(self, id, item):
        hparams = self.hparams
        wav = item['wav']
        if isinstance(wav, str):
            wav, sr = librosa.core.load(wav, sr=hparams['audio_sample_rate'])
            ws = hparams['win_size']
            if len(wav) % ws < ws - 1:
                wav = np.pad(wav, (0, ws - 1 - (len(wav) % ws)), mode='constant', constant_values=0.0)
        wav = torch.Tensor(wav)
        sample = {
            'wav': wav
        }
        return sample

    def collater(self, samples):
        if len(samples) == 1 and isinstance(samples[0], list):
            samples = samples[0]
        if len(samples) == 0:
            return {}
        wavs = collate_xd([s['wav'] for s in samples], 0.0)
        batch = {
            'nsamples': len(samples),
            'wavs': wavs
        }
        return batch

    def get_key_and_sizes(self):
        pass

    def __len__(self):
        return self.ds_len
    

class DiTDataset(MegaTTSDataset):
    def __init__(self, prefix, l=None, shuffle=False, data_dir=None, chunk_size=5, hparams=None):
        super(BaseSpeechDataset, self).__init__(f'{data_dir}/data', shuffle, chunk_size)
        if hparams is None:
            from utils.commons.hparams import hparams
        hparams = deepcopy(hparams)
        self.hparams = hparams
        self.load_mel = True
        self.data_dir = hparams['binary_data_dir'] if data_dir is None else data_dir
        self.prefix = prefix
        self.id2item_names = {}
        if l is None:
            reader = self.get_reader()
            l = len(reader.list_keys())
        self.ds_len = l

    def get_sample(self, id, item, items_timbre=None, max_timbre_len=None):
        sample = super(MegaTTSDataset, self).get_sample(id, item)

        # Generate full dur (mel2ph) and sparsified dur
        mel2ph_ = sample['mel2ph'][::self.hparams.get('vae_stride', 8)]
        sparsified_dur = torch.zeros_like(mel2ph_)
        for i in range(1, mel2ph_.max()+1):
            indices = torch.where(sparsified_dur == i)[0]
            if len(indices) > 0:
                rand_idx = indices[torch.randint(len(indices), (1,)).item()]
                sparsified_dur[rand_idx] = mel2ph_[rand_idx]
        sample['sparsified_dur'] = sparsified_dur
        sample['ctx_mask'] = torch.ones_like(mel2ph_)[:,None]
        sample['ctx_mel'] = torch.ones_like(sample['mel'])
        return sample
    
    def collater(self, samples):
        batch = super(DiTDataset, self).collater(samples)
        if len(samples) == 1 and isinstance(samples[0], list):
            samples = samples[0]
        samples = [x for x in samples if x is not None]
        if len(samples) == 0:
            return {}
        
        batch['text'] = [s['text'] for s in samples]
        batch['ctx_mask'] = collate_xd([s['ctx_mask'] for s in samples])
        batch['ctx_mels'] = collate_xd([s['ctx_mel'] for s in samples], -6.0)
        batch['mels'] = collate_xd([s['mel'] for s in samples], -6.0)

        if 'sparsified_dur' in samples[0]:
            batch['sparsified_dur'] = collate_xd([s['sparsified_dur'] for s in samples])
        return batch


class DiTWavDataset(FastSpeechDataset):
    def __init__(self, prefix, l=None, shuffle=False, data_dir=None, chunk_size=5, hparams=None):
        super(BaseSpeechDataset, self).__init__(f'{data_dir}/data', shuffle, chunk_size)
        if hparams is None:
            from utils.commons.hparams import hparams
        hparams = deepcopy(hparams)
        self.hparams = hparams
        self.load_mel = True
        self.data_dir = hparams['binary_data_dir'] if data_dir is None else data_dir
        self.prefix = prefix
        self.id2item_names = {}
        if l is None:
            reader = self.get_reader()
            l = len(reader.list_keys())
        self.ds_len = l
    
    def __getitem__(self, index):
        index = [index]
        item_names = [self.id2item_names[str(i)] for i in index]
        items = self.indexed_kv_ds.read_many(item_names)

        samples = []
        for i, (item, idx) in enumerate(zip(items, index)):
            item = pickle.loads(item)
            sample = self.get_sample(idx, item)
            if sample is not None:
                samples.append(sample)
        return samples

    def get_sample(self, id, item):
        hparams = self.hparams
        wav = item['wav']
        if isinstance(wav, str):
            wav, sr = librosa.core.load(wav, sr=hparams['audio_sample_rate'])
            ws = hparams['win_size']
            if len(wav) % ws < ws - 1:
                wav = np.pad(wav, (0, ws - 1 - (len(wav) % ws)), mode='constant', constant_values=0.0)
        wav = torch.Tensor(wav)
        sample = {
            'wav': wav
        }
        for l in ['phone', 'tone']:
            sample[l] = torch.LongTensor(item[f'{l}_encoded'])
        sample['txt_token'] = sample['phone']
        sample['ctx_mask'] = torch.ones(5)
        sample['mel'] = torch.ones(5)
        sample['lat_len'] = [1]
        sample['ctx_wav'] = wav
        sample['tgt_wav'] = wav
        sample['mel2ph'] = torch.ones(1, 5)
        if hparams.get('use_glm4v_token', False):
            sample['glm4v_feature'] = torch.ones(5)
            sample['glm4v_attention_mask'] = torch.ones(5).long()
        return sample

    def collater(self, samples):
        if len(samples) == 1 and isinstance(samples[0], list):
            samples = samples[0]
        if len(samples) == 0:
            return {}
        wavs = collate_xd([s['wav'] for s in samples], 0.0)
        mels = collate_xd([s['mel'] for s in samples], -6.0)
        ctx_wavs = collate_xd([s['ctx_wav'] for s in samples], 0.0)
        batch = {
            'nsamples': len(samples),
            'wavs': wavs,
            'mels': mels,
            'ctx_wavs': ctx_wavs
        }

        batch['txt_tokens'] = collate_xd([s['txt_token'] for s in samples], 0)
        batch['tone'] = collate_xd([s['tone'] for s in samples], 0)
        batch['txt_lengths'] = torch.LongTensor([s['txt_token'].numel() for s in samples])
        batch['mel2ph'] = collate_xd([s['mel2ph'] for s in samples], 0)
        batch['ctx_mask'] = collate_xd([s['ctx_mask'] for s in samples], 0)

        if self.hparams.get('use_glm4v_token', False):
            batch['tgt_wavs'] = collate_xd([s['tgt_wav'] for s in samples], 0.0)
            batch['glm4v_features'] = collate_xd([s['glm4v_feature'] for s in samples], 0)
            batch['glm4v_attention_mask'] = collate_xd([s['glm4v_attention_mask'] for s in samples], 0).int()
        return batch

    def get_key_and_sizes(self):
        pass

    def __len__(self):
        return self.ds_len


class DiTWavTextDataset(DiTWavDataset):
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
            'wavs': wavs,
            'wav_lengths': wav_lengths,
            'ctx_wavs': ctx_wavs,
            'vad_mask': vad_mask
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
        if 'dur_paraformer_label' in samples[0]:
            batch['dur_paraformer_label'] = collate_xd([s['dur_paraformer_label'] for s in samples], 0)
            batch['dur_paraformer_label_len'] = torch.LongTensor([s['dur_paraformer_label'].shape[0] for s in samples])
            
        if not hasattr(self, 'backup_batch') or self.backup_batch is None or random.random() < 0.001:
            self.backup_batch = batch

        return batch
    
class SpatialDiTLatDataset(DiTWavDataset):
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
            
        batch = {
            'nsamples': len(samples),
            # 'wavs': wavs,
            # 'wav_lengths': wav_lengths,
            # 'ctx_wavs': ctx_wavs,
            # 'vad_mask': vad_mask
        }
        if 'inputs_embeds' in samples[0][0]:
            batch['inputs_embeds'] = collate_xd([s[0]['inputs_embeds'] for s in samples])
        if 'decoder_input_ids' in samples[0][0]:
            batch['decoder_input_ids'] = collate_xd([s[0]['decoder_input_ids'] for s in samples])
        if 'labels' in samples[0][0]:
            batch['labels'] = collate_xd([s[0]['labels'] for s in samples])
        if 'direction' in samples[0][0]:
            batch['direction'] = collate_xd([s[0]['direction'] for s in samples])
        if 'energy_map' in samples[0][0]:
            batch['energy_map'] = collate_xd([s[0]['energy_map'] for s in samples])
        if 'lat' in samples[0][0]:
            batch['lat'] = collate_xd([s[0]['lat'] for s in samples])
        
        if not hasattr(self, 'backup_batch') or self.backup_batch is None or random.random() < 0.001:
            self.backup_batch = batch
            
        return batch
        
    def get_sample(self, id, item):
        hparams = self.hparams
        sample = {}
        
        for k in ['inputs_embeds', 'decoder_input_ids', 'labels', 'direction', 'energy_map', 'lat']:
            sample[k] = item[k]
        
        return sample
        
        
        
    

class CausalASRDataset(DiTWavDataset):
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
        mels = collate_xd([s['mel'] for s in samples], -6.0) if 'mel' in samples[0] else None
        batch = {
            'nsamples': len(samples),
            'wavs': wavs,
            'wav_lengths': wav_lengths,
            'mels': mels,
        }

        batch['mel2ph'] = collate_xd([s['mel2ph'] for s in samples], 0) if 'mel2ph' in samples[0] else None
        batch['text'] = [s['text'] for s in samples]
        batch['spk_mask'] = collate_xd([s['spk_mask'] for s in samples], 0.0) if 'spk_mask' in samples[0] else None

        if 'wav_w2v2' in samples[0]:
            batch['wavs_w2v2'] = collate_xd([s['wav_w2v2'] for s in samples], 0.0)
            batch['wav_w2v2_lengths'] = torch.LongTensor([s['wav_w2v2'].shape[0] for s in samples])

        if not hasattr(self, 'backup_batch') or self.backup_batch is None or random.random() < 0.001:
            self.backup_batch = batch

        return batch


class SpkWindowDataset(DiTWavDataset):
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
            'spk_names': [s['spk_name'] for s in samples],
            'voiced': collate_xd([s['voiced'] for s in samples], 0.0)
        }

        spk_map = {}
        spk_lst = []
        for sample in samples:
            spk_name = sample['spk_name']
            if spk_name not in spk_map:
                spk_map[spk_name] = len(spk_map)
            spk_lst.append(spk_map[spk_name])
        batch['spk_ids'] = torch.LongTensor(spk_lst)

        if not hasattr(self, 'backup_batch') or self.backup_batch is None or random.random() < 0.001:
            self.backup_batch = batch

        return batch


class CodecLMWavDataset(FastSpeechDataset):
    def __init__(self, prefix, l=None, shuffle=False, data_dir=None, chunk_size=5, hparams=None):
        super(BaseSpeechDataset, self).__init__(f'{data_dir}/data', shuffle, chunk_size)
        if hparams is None:
            from utils.commons.hparams import hparams
        hparams = deepcopy(hparams)
        self.hparams = hparams
        self.load_mel = True
        self.data_dir = hparams['binary_data_dir'] if data_dir is None else data_dir
        self.prefix = prefix
        self.id2item_names = {}
        if l is None:
            reader = self.get_reader()
            l = len(reader.list_keys())
        self.ds_len = l
    
    def __getitem__(self, index):
        index = [index]
        item_names = [self.id2item_names[str(i)] for i in index]
        items = self.indexed_kv_ds.read_many(item_names)

        samples = []
        for i, (item, idx) in enumerate(zip(items, index)):
            item = pickle.loads(item)
            sample = self.get_sample(idx, item)
            if sample is not None:
                samples.append(sample)
        return samples

    def get_sample(self, id, item):
        hparams = self.hparams
        wav = item['wav']
        if isinstance(wav, str):
            wav, sr = librosa.core.load(wav, sr=hparams['audio_sample_rate'])
            ws = hparams['win_size']
            if len(wav) % ws < ws - 1:
                wav = np.pad(wav, (0, ws - 1 - (len(wav) % ws)), mode='constant', constant_values=0.0)
        wav = torch.Tensor(wav)
        sample = {
            'wav': wav
        }
        for l in ['phone', 'tone']:
            sample[l] = torch.LongTensor(item[f'{l}_encoded'])
        sample['txt_token'] = sample['phone']
        sample['mel'] = torch.ones(5)
        sample['lat_len'] = [1]
        sample['mel2ph'] = torch.ones(1, 5)
        return sample

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
        mels = collate_xd([s['mel'] for s in samples], -6.0)
        batch = {
            'nsamples': len(samples),
            'wavs': wavs,
            'mels': mels,
            'wav_lengths': wav_lengths,
        }

        batch['txt_tokens'] = collate_xd([s['txt_token'] for s in samples], 0)
        batch['tone'] = collate_xd([s['tone'] for s in samples], 0)
        batch['txt_lengths'] = torch.LongTensor([s['txt_token'].numel() for s in samples])
        batch['mel2ph'] = collate_xd([s['mel2ph'] for s in samples], 0)
        batch['text'] = [s['text'] for s in samples]

        if not hasattr(self, 'backup_batch') or self.backup_batch is None or random.random() < 0.001:
            self.backup_batch = batch

        return batch

    def get_key_and_sizes(self):
        pass

    def __len__(self):
        return self.ds_len


class DiTWavImgDataset(DiTWavDataset):
    def __getitem__(self, index):
        return [{
            'wav': torch.zeros((24000)),
            'phone': torch.zeros((10), dtype=int),
            'tone': torch.zeros((10), dtype=int),
            'txt_token': torch.zeros((10), dtype=int),
            'lat_len': [1],
            'mel2ph': torch.ones(1, 10),
            'image': torch.zeros((840, 840, 3)),
            'mel': torch.zeros((1, 5, 80)),
            'glm4v_feature': torch.ones(5),
            'glm4v_attention_mask': torch.ones(5).long()
        }]

    def collater(self, samples):
        if len(samples) == 1 and isinstance(samples[0], list):
            samples = samples[0]
        if len(samples) == 0:
            return {}
        wavs = collate_xd([s['wav'] for s in samples], 0.0)
        batch = {
            'nsamples': len(samples),
            'wavs': wavs,
        }

        batch['txt_tokens'] = collate_xd([s['txt_token'] for s in samples], 0)
        batch['tone'] = collate_xd([s['tone'] for s in samples], 0)
        batch['txt_lengths'] = torch.LongTensor([s['txt_token'].numel() for s in samples])
        batch['mel2ph'] = collate_xd([s['mel2ph'] for s in samples], 0)
        # batch['mel'] = collate_xd([s['mel'] for s in samples], -6.0)
        if 'image' in samples[0] and samples[0]['image'] is not None:
            batch['image'] = collate_xd([s['image'] for s in samples], 0)
        if 'img_ids' in samples[0] and samples[0]['img_ids'] is not None:
            batch['img_ids'] = collate_xd([s['img_ids'] for s in samples], 0)
        
        if self.hparams.get('use_glm4v_token', False):
            batch['glm4v_features'] = collate_xd([s['glm4v_feature'] for s in samples], 0)
            batch['glm4v_attention_mask'] = collate_xd([s['glm4v_attention_mask'] for s in samples], 0).int()
        return batch


class Latent2WavDataset(BaseSpeechDataset):
    def __init__(self, prefix, l, shuffle=False, data_dir=None, training=False):
        self.ds_len = l
        super(Latent2WavDataset, self).__init__(prefix, shuffle, data_dir, load_size=False)
        hparams = self.hparams
        self.is_infer = not training
        self.batch_max_frames = 0 if self.is_infer else hparams['max_samples'] // hparams['hop_size']
        self.hop_size = hparams['hop_size']
        self.use_pitch = self.hparams['nsf_type'] != 'none'
        print(f"| {data_dir}@{prefix} dataset lengths: {l}")

    def __len__(self):
        return self.ds_len

    def __getitem__(self, index):
        index = [index]
        items = self.indexed_ds.read_many([self.id2item_names[str(i)] for i in index])
        samples = []
        for item, i in zip(items, index):
            item = pickle.loads(item)
            sample = self.get_sample(i, item)
            if sample is not None:
                samples.append(sample)
        return samples

    def get_sample(self, id, item):
        hparams = self.hparams
        # if len(item['wav']) <= hparams['max_samples'] + 2000:
        #     item['wav'] = np.pad(
        #         item['wav'], [0, hparams['max_samples'] - len(item['wav']) + 2000], constant_values=0)
        h, w, m = hparams['acous_params'][-1]
        wav2spec_dict = librosa_wav2spec(
            item['wav'],
            fft_size=w,
            hop_size=h,
            win_length=w,
            num_mels=m,
            fmin=hparams['fmin'],
            fmax=hparams['fmax'],
            sample_rate=hparams['audio_sample_rate'],
            center=False)
        item['mel'] = wav2spec_dict['mel']
        wav = item['wav'][:hparams['hop_size'] * item['mel'].shape[0]]
        sample = {
            "id": id,
            "mel": torch.FloatTensor(item['mel']),
            "wav": torch.FloatTensor(wav.astype(np.float32)),
        }
        return sample

    def collater(self, samples):
        hparams = self.hparams

        samples = samples[0]
        if len(samples) == 0:
            return {}

        y_batch, c_batch, p_batch, f0_batch = [], [], [], []
        for idx in range(len(samples)):
            x, c = samples[idx]['wav'], samples[idx]['mel']
            self._assert_ready_for_upsampling(x, c, hparams['hop_size'])
            if len(c) > self.batch_max_frames:
                # randomly pickup with the batch_max_steps length of the part
                batch_max_frames = self.batch_max_frames if self.batch_max_frames != 0 else len(c) - 1
                batch_max_steps = batch_max_frames * hparams['hop_size']
                interval_start = 0
                interval_end = len(c) - batch_max_frames
                start_frame = np.random.randint(interval_start, interval_end)
                start_step = start_frame * hparams['hop_size']
                y = x[start_step: start_step + batch_max_steps]
                c = c[start_frame: start_frame + batch_max_frames]
                self._assert_ready_for_upsampling(y, c, hparams['hop_size'])
            else:
                print(f"Removed short sample from batch (length={len(x)}).")
                continue
            y_batch += [y.reshape(-1, 1)]  # [(T, 1), (T, 1), ...]
            c_batch += [c]  # [(T' C), (T' C), ...]

        # convert each batch to tensor, asuume that each item in batch has the same length
        y_batch = collate_xd(y_batch, 0).transpose(2, 1)  # (B, 1, T)
        c_batch = collate_xd(c_batch, 0).transpose(2, 1)  # (B, C, T')
        p_batch = None
        f0_batch = None
        return {
            'mels': c_batch,
            'wavs': y_batch,
            'pitches': p_batch,
            'f0': f0_batch,
        }

    @staticmethod
    def _assert_ready_for_upsampling(x, c, hop_size):
        """Assert the audio and feature lengths are correctly adjusted for upsamping."""
        assert len(x) == (len(c)) * hop_size
        