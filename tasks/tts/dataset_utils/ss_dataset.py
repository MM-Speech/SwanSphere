import bisect
import collections
import uuid

import librosa
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor

from utils.commons.hdfs_utils import HDFSClient
from utils.commons.tos_utils_v2 import TosClient
import traceback

for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))
import subprocess
import time

import attrdict
import glob
import json
import random
import pickle
import cv2
import os
import torch
from utils.commons.base_shm_dataset import BaseShmDataset
from dataloader import KVReader
import numpy as np
import warnings
import re
from .megatts_fastdataset import raw_text_process
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

def convert_prompt(text):
    # 找出所有 <TAGx>...</TAGx>，记录其位置和内容
    tag_pattern = re.compile(r'<TAG>(.*?)</TAG>')
    result = []
    last_end = 0

    for match in tag_pattern.finditer(text):
        start, end = match.span()
        # 添加前面非 TAG 包裹的文本（加 <W>）
        if start > last_end:
            raw_text = text[last_end:start]
            if raw_text.strip():
                result.append(f'<W>{raw_text}</W>')
        # 添加 TAG 包裹的内容（不加标签）
        result.append(match.group(1))
        last_end = end

    # 处理最后一段非 TAG 包裹的文本
    if last_end < len(text):
        raw_text = text[last_end:]
        if raw_text.strip():
            result.append(f'<W>{raw_text}</W>')

    return ''.join(result)

class DiTText2AudioDataset(BaseShmDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = attrdict.AttrDict(self.hparams)
        if self.prefix != 'train':
            self.ds_len = len(self.hparams['test_idxs'])
        self.wav_feature_extractor = None
        self.client = None

    def get_dataset_meta(self):
        hparams, prefix = self.hparams, self.prefix
        datasets_meta = hparams['datasets']
        if prefix == 'test' and 'datasets_test' in hparams:
            datasets_meta = hparams['datasets_test']
        if isinstance(datasets_meta, str):
            datasets_meta = [{'name': 'default', 'video_pattern': datasets_meta}]
        if isinstance(datasets_meta[0], list):
            datasets_meta = [
                {'name': 'default', 'video_pattern': m[0], 'weight': m[1]}
                for m in datasets_meta
            ]
        total_len = 0
        for ds_meta in datasets_meta:
            p = ds_meta['video_pattern']
            if not os.path.exists(p): raise FileNotFoundError(f"{p} is not existed")
            ds_meta['kvreader_path'] = p[:-6]
            assert os.path.exists(f'{p[:-6]}.index'), p
            reader = KVReader(p[:-6])
            size = len(reader.list_keys())
            ds_meta['size'] = size
            print(f"| {p} dataset size: ", size)
            total_len += size
            assert size != 0, f'{p} is empty'
        print(f"| {self.prefix} dataset size: ", total_len)
        return datasets_meta, total_len

    def prepare_reader(self, dataset_meta, global_stores):
        for i in range(len(dataset_meta)):
            if 'kvreader_path' in dataset_meta[i]:
                dataset_meta[i]['kvreader'] = KVReader(dataset_meta[i]['kvreader_path'])
                dataset_meta[i]['keys'] = dataset_meta[i]['kvreader'].list_keys()
        return dataset_meta

    def read_fn(self, idx, reader_pack, global_stores):
        dataset_meta = reader_pack
        is_train = self.prefix == 'train'
        sizes = [d['size'] for d in dataset_meta]
        sizes_cumsum = np.cumsum(sizes).tolist()
        sizes_cumsum_shift = [0] + sizes_cumsum[:-1]
        if is_train:
            if self.hparams.get('use_dataset_weight'):
                weights = [d.get('weight', 1) for d in dataset_meta]
                weights = [w if isinstance(w, (float, int)) else eval(w) for w in weights]
                ds_idx = random.choices(list(range(len(weights))), weights)[0]
                meta_ds = dataset_meta[ds_idx]
                item_j = random.choice(range(meta_ds['size']))
            else:
                ds_idx = bisect.bisect_right(sizes_cumsum, idx)
                meta_ds = dataset_meta[ds_idx]
                item_j = idx - sizes_cumsum_shift[ds_idx]
        else:
            item_j = self.hparams['test_idxs'][idx % len(self.hparams['test_idxs'])]
            ds_idx = bisect.bisect_right(sizes_cumsum, item_j)
            meta_ds = dataset_meta[ds_idx]
            item_j = item_j - sizes_cumsum_shift[ds_idx]

        def unpickle_meta(data_b):
            return pickle.loads(data_b)

        if is_train:
            meta_datas = []
            keys = []
            if sizes_cumsum_shift[ds_idx] + item_j not in self.hparams['test_idxs']:
                keys.append(meta_ds['keys'][item_j])
            elif item_j + 1 < len(meta_ds['keys']):
                keys.append(meta_ds['keys'][item_j + 1])
            else:
                keys.append(meta_ds['keys'][item_j - 1])
            for data_b in meta_ds['kvreader'].read_many(keys):
                meta_datas.append(unpickle_meta(data_b))
            return meta_datas
        else:
            return unpickle_meta(meta_ds['kvreader'].read_many([meta_ds['keys'][item_j]])[0])

    def download_item_from_tos(self, video_dir, stride):
        self.client_hdfs = HDFSClient()
        self.client = TosClient()
        tmpdir = f'/tmp/item_from_tos/{uuid.uuid4()}'
        # self.client_hdfs.unzip(f"files_storage/megaavatar/{video_dir}/results_v1.zip", tmpdir)
        self.client_hdfs.unzip(f"files_storage/megaavatar/{video_dir}/audio_caption.zip", tmpdir)
        if self.hparams.get('use_image_caption'):
            self.client_hdfs.unzip(f"files_storage/megaavatar/{video_dir}/qwen2vl_caption.zip", tmpdir)
        return tmpdir

    def get_vad_mask(self, wav_path):
        wav = read_audio(wav_path)
        if getattr(self, 'vad_model', None) is None:
            self.vad_model = load_silero_vad()
        speech_timestamps = get_speech_timestamps(
            wav,
            self.vad_model,
            return_seconds=True,  # Return speech timestamps in seconds (default is samples)
        )
        if len(speech_timestamps) != 0:
            return speech_timestamps[0]['start'], speech_timestamps[-1]['end']
        else:
            return 0, 0

    def process_item(self, raw_items, hparams, global_stores):
        ### ret: wav, local prompt, video caption
        stride = 1
        if isinstance(raw_items, list):
            raw_items = raw_items[0]
        try:
            video_dir = self.download_item_from_tos(raw_items['video_dir'], stride)
        except Exception as e:
            return
        self.tmp_data_dir = video_dir

        for raw_item in raw_items['items']:
            video_name = raw_item['video_path']
            video_path_ori = f'data/{raw_items["video_dir"]}/{video_name}'
            video_path = f'{video_dir}/{video_name}'
            feature_dir = video_path[:-4] + '_features'
            if hparams.get('use_vocal_wav', True) or hparams.get('use_random_wav', 0) > random.random():
                audio_path = video_path[:-4] + '_vocal.wav'
                gemini_key = 'gemini_result'
            else:
                audio_path = video_path[:-4] + '_oriaudio.wav'
                gemini_key = 'gemini_ori_result'
            audiocaption_path = video_path[:-4] + '_gemini_results.json'

            if not os.path.exists(audio_path) or not os.path.exists(audiocaption_path):
                return
            # audio_inputs = self.process_wav(audio_path, sr=16000)
            # audio_inputs = audio_inputs.astype(np.float16)
            wav, _ = librosa.core.load(audio_path, sr=hparams['sample_rate'])
            fm = hparams['frames_multiple'] * hparams['hop_size']
            wav = wav[:wav.shape[0] // fm * fm]

            item = {}
            item['audio_path'] = audio_path
            # item['audio_cond'] = audio_inputs
            item['wavs'] = wav.astype(np.float16)
            if self.hparams['use_image_caption']:
                try:
                    caption_file = feature_dir + "/qwen2vl_caption.txt"
                    with open(caption_file, "r") as f:
                        image_caption = f.readline().strip()
                except Exception as e:
                    print(f'Read image caption error for caption_file {caption_file}')
                    # traceback.print_exc()
                    return
            else:
                image_caption = raw_item['tarsier2_caption']  # use video caption as image caption
            video_caption = raw_item['tarsier2_caption']

            try:
                with open(audiocaption_path, "r") as f:
                    audio_caption = json.load(f)
                    asr_text = audio_caption[0]['asr_result']['text']
                    global_description = audio_caption[0][gemini_key]['global description']
                    local_description = audio_caption[0][gemini_key]['fine-grained transcription']
                    if isinstance(local_description, list):
                        print(f'| find {local_description} is list, skip')
                        return
                    try:
                        gemini_version = audio_caption[0][gemini_key]['model_name']
                    except Exception as e:
                        gemini_version = 'unknown'
            except Exception as e:
                print(f'Read audio caption error for caption_file {audiocaption_path}')
                # traceback.print_exc()
                return

            if self.is_repetitive(local_description):
                return

            # clean asr txt
            if len(asr_text) > 0:
                asr_text = raw_text_process(asr_text)

            item['ori_caption'] = [image_caption, video_caption]
            item['local_video_path'] = video_path
            item['video_path'] = video_path_ori
            item['item_name'] = video_path_ori
            item["wav_lengths"] = item['seq_len'] = wav.shape[0]
            item['num_frames'] = int(wav.shape[0] / hparams['sample_rate'] * raw_item['fps'])
            item['text'] = item['asr_text'] = asr_text
            item['global_description'] = '<GPROMPT>' + global_description + '</GPROMPT>'
            item['local_description'] = local_description.replace('<tag>', '<TAG>').replace('</tag>', '</TAG>')
            caption = ''
            if hparams.get('use_global', True):
                caption = caption + global_description
            if hparams.get('use_local', True):
                caption = caption + convert_prompt(item['local_description'])
            else:
                caption = caption + '<W>' + asr_text + '</W>'
            item['caption'] = caption
            item['gemini_version'] = gemini_version

            if hparams.get('add_vad_mask', False):
                vad_start, vad_end = self.get_vad_mask(audio_path)
                vm = hparams['hop_size'] * hparams['vae_stride']
                vad_mask = np.zeros((wav.shape[0] // vm))
                vad_mask[int(vad_start * hparams['sample_rate'] // vm) : int(vad_end * hparams['sample_rate'] // vm)] = 1
                item['vad_mask'] = vad_mask
            else:
                item['vad_mask'] = None


            ctx_mask = torch.zeros((wav.shape[0] // hparams['hop_size']))
            ctx_len = random.randint(0, int(ctx_mask.shape[0] * 0.8))
            ctx_mask[:ctx_len] = 1
            ctx_mask = ctx_mask[::hparams['vae_stride']]
            ctx_wav = wav[:ctx_len * hparams['hop_size']]

            item['ctx_wavs'] = ctx_wav.astype(np.float16)
            item['ctx_mask'] = ctx_mask

            token_num = ctx_mask.shape[0] # semantic token
            if item["num_frames"] // 2 + len(item['local_description']) // 4 > 1000:
                return
            yield item, f'seqlen{token_num:07d}'

    def process_wav(self, audio_file, sr=24000):
        if self.wav_feature_extractor is None:
            self.wav_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                "pretrained_models/wav2vec2-base-960h"
            )
        speech_array, _ = librosa.core.load(audio_file, sr=sr)
        if len(speech_array.shape) > 1:
            speech_array = np.mean(speech_array, axis=1)

        # 计算RMS值 (too heavy)
        rms = librosa.feature.rms(y=speech_array)
        # 输出全局平均RMS（可选）
        average_rms = np.mean(rms)
        # print(f'Average RMS: {average_rms} {audio_path}')
        if average_rms < 0.001:
            # print("low audio", audio_path)
            return None

        # 使用 librosa 进行重采样到 sr Hz
        input_values = np.squeeze(
            self.wav_feature_extractor(speech_array, sampling_rate=sr).input_values
        )
        if input_values.shape[0] < 10000:
            return None
        return input_values

    def after_process_item(self, raw_item, hparams, global_stores):
        subprocess.check_call(f'rm -rf {self.tmp_data_dir}', shell=True)

    def collater(self, samples):
        samples = [sample for sample in samples if sample is not None]
        if len(samples) == 0:
            return {}

        batch = {}

        # 收集所有 key
        keys = samples[0].keys()

        for key in keys:
            values = [s[key] for s in samples]

            # 处理音频 wav
            if key in ["wavs", 'ctx_wavs', 'ctx_mask', 'vad_mask']:
                max_len = max([len(w) for w in values])
                padded_wavs = []

                for w in values:
                    pad_width = max_len - len(w)
                    padded = np.pad(w, (0, pad_width), mode='constant')
                    padded_wavs.append(padded)
                batch[key] = torch.tensor(padded_wavs, dtype=torch.float16)

            # 处理 int 型字段
            elif isinstance(values[0], int):
                batch[key] = torch.tensor(values, dtype=torch.long)

            # 处理 string（或其它不可 tensor 化的）字段
            else:
                batch[key] = values  # list of str 或其他

        return batch

    def is_repetitive(self, text, max_char_repeat=25, max_phrase_repeat=10):
        """
        检测文本中是否存在重复过多的字或短语。

        返回：
            True 表示重复过多
            False 表示正常
        """

        # 1. 检查单个字符重复，如：的的的的的的
        if re.search(r'(.)\1{%d,}' % max_char_repeat, text):
            return True

        # 2. 检查短语重复（长度为2~6）
        for n in range(2, 7):
            pattern = r'((.{%d})[\s，。、“”‘’！!？?]*)\1{%d,}' % (n, max_phrase_repeat)
            if re.search(pattern, text):
                return True

        return False

    def getitem_fast(self, index):
        sp_size = self.hparams.get("sp_size", 1)
        data_path = f'{self.shm_base}/{index * (self.world_size // sp_size) + (self.rank_id // sp_size)}.json'
        retry_interval = 1
        retry_cnt = 0
        sp_rank = self.rank_id % sp_size
        while not os.path.exists(data_path):
            time.sleep(retry_interval)
            retry_cnt += retry_interval
            if retry_cnt % 30 == 0:
                print(f"| waiting for data {data_path} for {retry_cnt}s @ rank{self.rank_id}")
        while True:
            try:
                fnames = json.load(open(data_path))
                break
            except:
                time.sleep(1)
        items = []
        for fname in fnames:
            item = pickle.load(open(fname, 'rb'))
            items.append(item)
        loaded_flag_path = f'{data_path}.READ_FLAGS.{sp_rank}'
        with open(loaded_flag_path, 'w') as f:
            f.write('loaded')
        if sp_rank == 0:
            while True:
                all_loaded = True
                for sp_rank_ in range(0, sp_size):
                    if not os.path.exists(f'{data_path}.READ_FLAGS.{sp_rank_}'):
                        all_loaded = False
                        time.sleep(0.5)
                        break
                if all_loaded:
                    for fname in fnames:
                        os.remove(fname)
                    os.remove(data_path)
                    for sp_rank_ in range(0, sp_size):
                        os.remove(f'{data_path}.READ_FLAGS.{sp_rank_}')
                    break
        return items

    def get_seq_len(self, path):
        return int(path.split('#')[-1].split('.')[0].replace('seqlen', ''))

    def create_batch(self, items_buffer):  # get_dataloader -> build_fast_dataloader -> batch_saver_fn -> create_batch
        ntokens = 0
        batch = []
        items_buffer_ = []
        if len(items_buffer) == 0:
            print(f"| ERROR: {items_buffer} is empty while creating batch.")
        for item in items_buffer:  # items_buffer是一个list，里面是pkl文件路径
            seq_len = self.get_seq_len(item)  # 文件路径里面计算了seq_length
            max_training_tokens = self.hparams.get('max_training_tokens', self.hparams['max_tokens'])
            if ntokens + seq_len <= max_training_tokens:
                batch.append(item)
                ntokens += seq_len
            elif seq_len > max_training_tokens:  # 如果单个item的seq_len大于max_training_tokens，直接丢弃
                os.remove(item)  # 由于没放到getitem_fast里面进行删除，因此要在这里删除，否则会越积越多
            else:
                items_buffer_.append(item)
        return batch, items_buffer_

if __name__ == '__main__':
    from utils.commons.hparams import set_hparams, hparams
    global DEBUG
    DEBUG = True
    set_hparams('egs/datasets/ss_dataset.yaml')
    exp_name = 'test_DiTT2ADataset'
    hparams['exp_name'] = exp_name
    hparams['sp_size'] = 1
    hparams['num_workers'] = 8
    hparams['debug'] = True
    hparams['fast_ds_shuffle_buffer'] = 32
    # ds_val = DiTBodyA2VDataset('test', hparams, False, 0, 1, 1)
    # dl_val = ds_val.get_dataloader(seed=1234, num_workers=int(os.environ.get('dl_workers', 4)))
    # for i, item in enumerate(dl_val):
    #     if i == 5:
    #         break
    #     print(item['tgt_lat'][0].shape, item['video_path'])

    ds_train = DiTText2AudioDataset('train', hparams, use_fast_dataloader=True, rank_id=0, world_size=hparams['sp_size'], batch_size=1)
    dl_train = ds_train.get_dataloader(seed=1234, num_workers=hparams['num_workers'])
    for i, items in enumerate(dl_train):
        # if i == 5:
        #     break
        print(
            [x['item_name'] for x in items],
            [x['num_frames'] for x in items],
            "audio_cond",[x['audio_cond'].shape for x in items],
            "text",[x['text'] for x in items],
            'asr_text', [x['asr_text'] for x in items],
            'global_description', [x['global_description'] for x in items],
            'local_description', [x['local_description'] for x in items],
        )