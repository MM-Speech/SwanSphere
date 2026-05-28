import utils.commons.single_thread_env  # NOQA
import attrdict
from copy import deepcopy
from functools import partial
import json
import math
import glob
import multiprocessing
import os
import pickle
import random
import subprocess
import time
import traceback
import torch
import bisect
# from dataloader import KVReader, FalconReader
import setproctitle
from utils.commons.hparams import hparams
from utils.commons.os_utils import multiprocess_glob
from utils.commons.tensor_utils import convert_to_tensor, convert_to_np
import numpy as np
from utils.commons.ckpt_utils import load_ckpt


DEBUG = False  # 打印各类worker的更新状态


def get_reader(data_paths, hp=None):
    reader = [KVReader(x) for x in data_paths]
    ds_len = [len(x.list_keys()) for x in reader]
    return reader, ds_len


def check_hdfs_file_existence(file_path):
    command = f"hdfs dfs -test -e {file_path}"
    try:
        subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def save_fnames_to_shm(fnames, batch_cnt, shm_base):
    data_path = f'{shm_base}/{batch_cnt}.json'
    with open(f'{data_path}', 'w') as f:
        json.dump(fnames, f)
    return data_path


def save_samples_to_shm(samples, cnt, shm_base, item_meta):
    data_path = f'{shm_base}/{cnt:08d}#{item_meta}.pkl'
    with open(f'{data_path}', 'wb') as f:
        pickle.dump(samples, f)
    return data_path


def get_from_global_stores(k, gs, fn):
    if k not in gs:
        gs[k] = fn()
    return gs[k]


class BaseShmDataset(torch.utils.data.Dataset):
    def __init__(self, prefix, hparams,
                 use_fast_dataloader=True, rank_id=None, world_size=None,
                 batch_size=1, random_frame=True, node_id=None, node_size=None):
        self.prefix = prefix
        self.use_fast_dataloader = use_fast_dataloader
        self.batch_size = batch_size
        self.hparams = deepcopy(hparams)

        self.rank_id = rank_id
        self.world_size = world_size
        self.node_id = node_id
        self.node_size = node_size
        if self.use_fast_dataloader:
            self.shm_base = f'/dev/shm/data_shm_{hparams["exp_name"]}'
        else:
            self.readers = None
            self.global_stores = {}
        
        if rank_id == 0 or not use_fast_dataloader:
            self.random_frame = self.hparams['random_frame'] = random_frame
            self.dataset_meta, self.ds_len = self.get_dataset_meta()
            self.prefetch_steps = hparams.get('fast_ds_prefetch_steps', 16)
            self.shuffle_buffer = max(hparams.get('fast_ds_shuffle_buffer', 16), batch_size * 4)
            print("| shuffle_buffer: ", self.shuffle_buffer)

    @staticmethod
    def init_worker(worker_id, exp_name):
        setproctitle.setproctitle(f'dataloader#{worker_id} ({exp_name})')

    def get_dataset_meta(self):
        raise NotImplementedError

    def prepare_reader(self, dataset_meta, global_stores):
        raise NotImplementedError

    def read_fn(self, idx, reader_pack, global_stores):
        raise NotImplementedError

    def controller_fn(self, ds_len, seed, q_to_pull, hparams_, max_epoch=0, n_processor=0):
        hparams.update(hparams_)
        setproctitle.setproctitle(f'data_processor:{hparams["exp_name"]}:controller_fn')
        print(f"| init controller")
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

    def process_fn(
            self, q_to_pull, q_to_push, world_size,
            shm_base, counter, hparams_, seed, i_worker, n_worker
    ):
        hparams.update(hparams_)
        setproctitle.setproctitle(f'data_processor:{hparams["exp_name"]}:processor_fn#{i_worker}/{n_worker}')
        self.seed = seed
        print(f"| Starting processor_fn_worker#{i_worker}/{n_worker}.")

        try:
            global_stores = {}
            reader_pack = self.prepare_reader(self.dataset_meta, global_stores)
            print(f"| init processor (dataset_reader)#{i_worker}/{n_worker}.")
            restart_countdown = 10000
            while True:
                try:
                    idx = q_to_pull.get()
                    if idx is None:
                        return None

                    raw_item = self.read_fn(idx, reader_pack, global_stores)
                    if raw_item is None:
                        if DEBUG:
                            print("processor: skip becasuse the item_bytes is None")
                        continue
                    if not isinstance(raw_item, list):
                        raw_item = [raw_item]
                    for raw_item_ in raw_item:
                        for item in self.process_item(raw_item_, hparams, global_stores):
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
                        self.after_process_item(raw_item_, hparams, global_stores)
                except:
                    traceback.print_exc()
                    continue
        except:
            traceback.print_exc()

    def after_process_item(self, raw_item, hparams, global_stores):
        pass

    def batch_saver_fn(self, q_to_push, hparams_, seed, shm_base, world_size, rank_i, batch_counter):
        hparams.update(hparams_)
        setproctitle.setproctitle(f'data_processor:{hparams["exp_name"]}:batch_saver#{rank_i}/{world_size}')
        print(f"| init batch_saver#{rank_i}/{world_size}")

        g = torch.Generator()
        g.manual_seed(seed)
        random.seed(seed)
        shuffle_buffer = self.shuffle_buffer
        items_buffer = []

        local_batch_cnt = 0
        while True:
            try:
                while len(glob.glob(f'{shm_base}/*.json')) >= self.prefetch_steps * world_size:
                    if DEBUG:
                        print(f"batch_saver#{rank_i}/{world_size}: json 文件太多({len(glob.glob(f'{shm_base}/*.json'))} >= {self.prefetch_steps} x {world_size}), 休息1s等等getitem")
                    time.sleep(1)
                if q_to_push.empty():
                    if len(items_buffer) < shuffle_buffer:
                        if DEBUG:
                            print(f"batch_saver#{rank_i}/{world_size}: q_to_push 为空, 休息1s等等processor")
                        time.sleep(1)
                else:
                    items_buffer.append(q_to_push.get())
                if DEBUG:
                    print(f"| length of items_buffer: {len(items_buffer)}")
                while len(items_buffer) > shuffle_buffer:
                    with batch_counter.get_lock():
                        batch_cnt = batch_counter.value
                        batch_counter.value += 1
                        local_batch_cnt += 1

                    if local_batch_cnt % 16 == 0:
                        random.shuffle(items_buffer)

                    batch, items_buffer = self.create_batch(items_buffer)
                    out_path = save_fnames_to_shm(batch, batch_cnt, shm_base)
                    if DEBUG:
                        print(f"batch saver#{rank_i}/{world_size}: saved to {out_path}")
                    batch_cnt += 1
            except:
                traceback.print_exc()

    def create_batch(self, items_buffer):
        batch = items_buffer[:self.batch_size]
        items_buffer_new = items_buffer[self.batch_size:]
        return batch, items_buffer_new

    def build_fast_dataloader(self, shm_base, seed, world_size,
                              hparams, n_processor=16, max_epoch=-1, auto_restart=True, ds_len=None):
        if hparams.get('dataloader_mp_type', 'fork') == 'spawn':
            ctx = multiprocessing.get_context('spawn')
        else:
            ctx = multiprocessing.get_context('fork')
        setproctitle.setproctitle(
            f'data_processor:{hparams["exp_name"]}:constructor')
        sp_size = hparams.get('sp_size', 1)
        q_to_pull = ctx.Queue(2048 * world_size // sp_size)
        q_to_push = ctx.Queue(10000)
        os.makedirs(shm_base, exist_ok=True)
        print(f"| training dataset len: {ds_len}")
        self.seed = seed
        proc_controller = ctx.Process(
            target=self.controller_fn, args=(ds_len, seed, q_to_pull, hparams, max_epoch, n_processor), daemon=True
        )
        proc_controller.start()
        batch_counter = ctx.Value('i', 0)

        def create_batch_saver(rank_i):
            return ctx.Process(target=self.batch_saver_fn,
                               args=(q_to_push, hparams, seed, shm_base, world_size // sp_size, rank_i, batch_counter),
                               daemon=True)

        proc_batch_saver = [create_batch_saver(i) for i in range(world_size // sp_size)]
        for p in proc_batch_saver:
            p.start()

        counter = ctx.Value('i', 0)

        def create_processor(worker_i):
            return ctx.Process(target=self.process_fn,
                               args=(q_to_pull, q_to_push, world_size // sp_size, shm_base, counter, hparams,
                                     seed, worker_i, n_processor),
                               daemon=True)

        proc_processor = [create_processor(i) for i in range(n_processor)]
        for p in proc_processor:
            p.start()
        time.sleep(60)
        if auto_restart:
            while True:
                if not proc_controller.is_alive():
                    break
                for p in proc_processor:
                    if not p.is_alive():
                        print("| Restarting process")
                        i = proc_processor.index(p)
                        new_p = create_processor(i)
                        new_p.start()
                        proc_processor[i] = new_p

                time.sleep(5)
        proc_controller.join()
        for p in proc_processor:
            p.join()

    def get_dataloader(self, num_workers=None, shuffle=False, seed=1234, exp_name=None):
        if exp_name is None:
            exp_name = self.hparams['exp_name']
        else:
            self.hparams['exp_name'] = exp_name
        if self.use_fast_dataloader:
            if num_workers is None:
                num_workers = 4
            data_loader = torch.utils.data.DataLoader(
                dataset=self,
                collate_fn=self.collater_fast,
                worker_init_fn=partial(self.init_worker, exp_name=exp_name),
                shuffle=False,
                batch_size=1,
                num_workers=num_workers,
                prefetch_factor=2 if num_workers >= 1 else None
            )
            if self.rank_id == 0:
                subprocess.check_call(f'rm -rf {self.shm_base}; ', shell=True)
                time.sleep(10)
                ctx = multiprocessing.get_context('spawn')
                ctx.Process(target=self.build_fast_dataloader, kwargs={
                    'shm_base': self.shm_base,
                    'seed': seed,
                    'world_size': self.world_size,
                    'hparams': self.hparams,
                    'n_processor': self.world_size * self.hparams['ds_workers'],
                    'ds_len': self.ds_len,
                }).start()
        else:
            if num_workers is None:
                num_workers = 0
            self.seed = seed
            data_loader = torch.utils.data.DataLoader(
                dataset=self,
                collate_fn=self.collater,
                worker_init_fn=partial(self.init_worker, exp_name=exp_name),
                shuffle=shuffle,
                batch_size=self.batch_size,
                num_workers=num_workers,
            )
        return data_loader

    def getitem_fast(self, index):
        data_path = f'{self.shm_base}/{index * self.world_size + self.rank_id}.json'
        retry_interval = 1
        retry_cnt = 0
        while not os.path.exists(data_path):
            time.sleep(retry_interval)
            retry_cnt += retry_interval
            if retry_cnt % 30 == 0 and DEBUG:
                print(f"| waiting for data {data_path} for {retry_cnt}s")
        while True:
            try:
                fnames = json.load(open(data_path))
                break
            except:
                time.sleep(1)
        if DEBUG:
            print(f"dataset: rm {data_path}")
        os.remove(data_path)
        items = []
        for fname in fnames:
            item = pickle.load(open(fname, 'rb'))
            items.append(item)
            if DEBUG:
                print(f"dataset: rm {fname}")
            os.remove(fname)
        return items

    def __getitem__(self, index):
        if self.use_fast_dataloader:
            items = self.getitem_fast(index)
            items = convert_to_tensor(items)
            return items
        else:
            item = self.getitem_slow(index)
            item = convert_to_tensor(item)
            return item

    def __len__(self):
        if self.use_fast_dataloader:
            return 100000000
        return self.ds_len

    def getitem_slow(self, index):
        if self.readers is None:
            self.readers = self.prepare_reader(self.dataset_meta, self.global_stores)
        while True:
            raw_item = self.read_fn(index, self.readers, self.global_stores)
            if raw_item is not None:
                for item in self.process_item(raw_item, self.hparams, self.global_stores):
                    if isinstance(item, tuple):
                        item, item_meta = item
                    return item
            index += 1

    def process_item(self, raw_item, hparams, global_stores):
        raise NotImplementedError

    def collater(self, samples):
        raise NotImplementedError

    def collater_fast(self, samples):
        samples = samples[0]
        if len(samples) == 0:
            return {}
        return self.collater(samples)
    
    
class BaseFalconReaderShmDataset(BaseShmDataset):
    def get_reader(self, data_paths, reader_chunk_size, worker_id=0, worker_world_size=1, reader_cache_name='cache'):
        fd_cache_size = 128
        io_thread_num = 1
        io_retry = 5
        reader = FalconReader(data_paths, fd_cache_size, io_thread_num, io_retry, 
                            reader_cache_name, worker_world_size, worker_id, reader_chunk_size)
        ds_len = reader.get_entry_num(list(range(len(data_paths))), False)
        return reader, ds_len
    
    def prepare_reader(self, dataset_meta, global_stores, i_worker, n_worker):
        raise NotImplementedError
    
    def controller_fn(self, ds_len, seed, q_to_pull, hparams_, max_epoch=0, n_processor=0):
        hparams.update(hparams_)
        setproctitle.setproctitle(f'data_processor:{hparams["exp_name"]}:controller_fn')
        print(f"| init controller")
        try:
            g = torch.Generator()  # 随机数生成器
            g.manual_seed(seed)
            reader_chunk_size = hparams.get('reader_chunk_size', 64)
            indices = torch.randperm(math.ceil(ds_len / reader_chunk_size), generator=g).tolist()
            if self.node_id is not None:
                indices = indices[self.node_id::self.node_size]
            pull_i = 0
            epoch = 0
            while max_epoch <= 0 or epoch < max_epoch:
                while not q_to_pull.full():
                    if pull_i == len(indices):
                        epoch += 1
                        indices = torch.randperm(math.ceil(ds_len / reader_chunk_size), generator=g).tolist()
                        pull_i = 0
                        break
                    q_to_pull.put(indices[pull_i] * reader_chunk_size)
                    pull_i += 1
                if DEBUG:
                    print("controller: q_to_pull满了, 休息1s等等processor")
                time.sleep(1)
            for i in range(n_processor * 2):
                q_to_pull.put(None)
            print("| Controller worker finished...")
        except:
            traceback.print_exc()
    
    def process_fn(
            self, q_to_pull, q_to_push, world_size,
            shm_base, counter, hparams_, seed, i_worker, n_worker
    ):
        hparams.update(hparams_)
        setproctitle.setproctitle(f'data_processor:{hparams["exp_name"]}:processor_fn#{i_worker}/{n_worker}')
        self.seed = seed
        print(f"| Starting processor_fn_worker#{i_worker}/{n_worker}.")

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

                    raw_item = self.read_fn(idx, reader_pack, global_stores)
                    if raw_item is None:
                        if DEBUG:
                            print("processor: skip becasuse the item_bytes is None")
                        continue
                    if not isinstance(raw_item, list):
                        raw_item = [raw_item]
                    for raw_item_ in raw_item:
                        for item in self.process_item(raw_item_, hparams, global_stores, i_worker, n_worker):
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
                        self.after_process_item(raw_item_, hparams, global_stores)
                except:
                    traceback.print_exc()
                    continue
        except:
            traceback.print_exc()
            
    def getitem_slow(self, index):
        if self.readers is None:
            self.readers = self.prepare_reader(self.dataset_meta, self.global_stores, 0, 1)
        while True:
            raw_item = self.read_fn(index, self.readers, self.global_stores)
            if raw_item is not None:
                for item in self.process_item(raw_item, self.hparams, self.global_stores, 0, 1):
                    if isinstance(item, tuple):
                        item, item_meta = item
                    return item
            index += 1
            
    def process_item(self, raw_item, hparams, global_stores, i_worker, n_worker):
        raise NotImplementedError


class BaseKVReaderShmDataset(BaseShmDataset):
    def get_dataset_meta(self):
        hparams, prefix = self.hparams, self.prefix
        data_dir = hparams['binary_data_dir']
        data_paths = hparams[f'{prefix}_sets'] if len(hparams[f'{prefix}_sets']) > 0 else [data_dir]
        data_paths = [f'{hparams["binary_data_dir"]}/{x}/{prefix}' for x in data_paths]
        _, ds_len = get_reader(data_paths, hp=hparams)
        ds_len = sum(ds_len)  # 返回的ds_len是所有数据集之和
        data_meta = data_paths
        print("| data paths: ", json.dumps(data_meta, indent=2, ensure_ascii=False))
        return data_meta, ds_len

    def prepare_reader(self, dataset_meta, global_stores):
        readers, ds_len_lst = get_reader(dataset_meta)
        accu = 0
        accu_ds_length = []
        for cur_len in ds_len_lst:
            accu_ds_length.append(cur_len + accu)
            accu += cur_len

        reader_idx2keys = []
        for reader_i in readers:
            reader_idx2keys.extend(reader_i.list_keys())
        return readers, accu_ds_length, reader_idx2keys

    def read_fn(self, idx, reader_pack, global_stores):
        """
            从binary data里面读取 bytes
        """
        readers, accu_ds_length, reader_idx2keys = reader_pack
        assert accu_ds_length is not None and reader_idx2keys is not None
        key = reader_idx2keys[idx]
        ds_idx = bisect.bisect_right(accu_ds_length, idx)
        try:
            items_bytes = readers[ds_idx].read_many([key])
        except:
            traceback.print_exc()
            return None
        item_bytes = items_bytes[0]
        if item_bytes is None:
            return None
        raw_item = pickle.loads(item_bytes)
        return raw_item


def get_filelist_cache(p):
    cache_path = f'data/train_filelists/{p.replace("/", "#").replace("*", "%")}.json'
    if os.path.exists(cache_path):
        return json.load(open(cache_path, 'r'))


class BaseGlobDatasetShmDataset(BaseShmDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = attrdict.AttrDict(self.hparams)
        if self.prefix != 'train':
            self.ds_len = len(self.hparams['test_idxs'])

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
            if p.endswith('.index'):
                ds_meta['kvreader_path'] = p[:-6]
                reader = KVReader(p[:-6])
                size = len(reader.list_keys())
            elif p.endswith('.jsonl'):
                if os.path.exists(p[:-6] + '.index'):
                    ds_meta['kvreader_path'] = p[:-6]
                    reader = KVReader(p[:-6])
                    size = len(reader.list_keys())
                else:
                    import orjsonl
                    ds_meta['data'] = orjsonl.load(path=p)
                    paths = [m['video_path'] for m in ds_meta['data']]
                    ds_meta['paths'] = paths
                    size = len(paths)
            else:
                c = get_filelist_cache(p)
                paths = c if c is not None else multiprocess_glob(p)
                size = len(paths)
                ds_meta['paths'] = paths
                ds_meta['data'] = [{'video_path': p} for p in paths]
            ds_meta['size'] = size
            print(f"| {p} dataset size: ", size)
            total_len += size
            if size == 0:
                print(f"!!!!!WARNING: {p} is empty!!!!!")
        print(f"| {self.prefix} dataset size: ", total_len)
        return datasets_meta, total_len

    def prepare_reader(self, dataset_meta, global_stores):
        if self.prefix != 'train' or not hparams.get('use_falconreader', False):
            for i in range(len(dataset_meta)):
                if 'kvreader_path' in dataset_meta[i]:
                    dataset_meta[i]['kvreader'] = KVReader(dataset_meta[i]['kvreader_path'])
                    dataset_meta[i]['keys'] = dataset_meta[i]['kvreader'].list_keys()
            return dataset_meta
        else:
            chunk_size = 32
            fd_cache_size = 128
            io_thread_num = 1
            io_retry = 5
            cache_name = 'test'
            world_size = 1
            rank = 0
            for i in range(len(dataset_meta)):
                if 'kvreader_path' in dataset_meta[i]:
                    dataset_meta[i]['falconreader'] = reader = FalconReader(
                        [dataset_meta[i]['kvreader_path']], fd_cache_size, io_thread_num, io_retry, cache_name,
                        world_size, rank, chunk_size)
                    entry_num = reader.get_entry_num([0], False)
                    chunk_idxs = [i * chunk_size for i in range(entry_num // chunk_size)]
                    dataset_meta[i]['keys'] = chunk_idxs
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
            meta_i = pickle.loads(data_b)
            if 'video_path' in meta_i:
                meta_i['video_name'] = os.path.dirname(meta_ds['video_pattern']) + meta_i['video_path']
            elif 'image_path' in meta_i:
                meta_i['video_name'] = os.path.dirname(meta_ds['video_pattern']) + meta_i['image_path']
            if not hparams.get('download_data_from_tos') and \
                    not os.path.exists(meta_i['video_name']):  # video_pattern is not the prefix of video_path
                meta_i['video_name'] = meta_i['video_path']
            return meta_i

        if is_train:
            meta_datas = []
            if 'falconreader' in meta_ds:
                raw_datas = meta_ds['falconreader'].read_many([meta_ds['keys'][item_j % len(meta_ds['keys'])]])[0]
                for raw_data in raw_datas:
                    meta_datas.append(unpickle_meta(raw_data))
                return meta_datas
            else:
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
