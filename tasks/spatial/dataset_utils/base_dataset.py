import bisect
import os
import sys
import traceback
import types
from functools import wraps
from itertools import chain
import numpy as np
import torch.utils.data
# from pyarrow import hdfs
from torch.utils.data import ConcatDataset
import torch.nn.functional as F
from utils.commons.hparams import hparams
# from dataloader import KVReader, FalconReader
import pickle
import subprocess


def chunk(iterable, chunk_size):
    ret = []
    for record in iterable:
        ret.append(record)
        if len(ret) == chunk_size:
            yield ret
            ret = []
    if ret:
        yield ret


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def collate_xd(values, pad_idx=0, left_pad=False, shift_right=False, max_len=None, shift_id=1):
    if len(values[0].shape) == 1:
        return collate_1d(values, pad_idx, left_pad, shift_right, max_len, shift_id)
    elif len(values[0].shape) == 2:
        return collate_2d(values, pad_idx, left_pad, shift_right, max_len)
    elif len(values[0].shape) == 3:
        return collate_3d(values, pad_idx, left_pad, shift_right, max_len)


def collate_1d(values, pad_idx=0, left_pad=False, shift_right=False, max_len=None, shift_id=1):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values) if max_len is None else max_len
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if shift_right:
            dst[1:] = src[:-1]
            dst[0] = shift_id
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def collate_2d(values, pad_idx=0, left_pad=False, shift_right=False, max_len=None):
    """Convert a list of 2d tensors into a padded 3d tensor."""
    size = max(v.size(0) for v in values) if max_len is None else max_len
    res = values[0].new(len(values), size, values[0].shape[1]).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel(), f"{dst.numel()} {src.numel()}"
        if shift_right:
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def collate_3d(values, pad_idx=0, left_pad=False, shift_right=False, max_len=None):
    """Convert a list of 2d tensors into a padded 3d tensor."""
    size = max(v.size(0) for v in values) if max_len is None else max_len
    res = values[0].new(len(values), size, values[0].shape[1], values[0].shape[2]).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if shift_right:
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def pad_or_cut_1d(values: torch.tensor, tgt_len, pad_value=0):
    src_len = values.shape[0]
    if src_len < tgt_len:
        res = F.pad(values, [0, tgt_len - src_len], value=pad_value)
        if res.device != values.device:
            res = res.to(values.device)
    else:
        res = values[:tgt_len]
    return res

def pad_or_cut_2d(values: torch.tensor, tgt_len, dim=-1, pad_value=0):
    if dim == 0 or dim == -2:
        src_len = values.shape[0]
        if src_len < tgt_len:
            res = F.pad(values, [0, 0, 0, tgt_len - src_len], value=pad_value)
        else:
            res = values[:tgt_len]
    elif dim == 1 or dim == -1:
        src_len = values.shape[1]
        if src_len < tgt_len:
            res = F.pad(values, [0, tgt_len - src_len], value=pad_value)
        else:
            res = values[:, :tgt_len]
    else:
        raise RuntimeError(f"Wrong dim number {dim} while the tensor only has {len(values.shape)} dimensions.")
    if res.device != values.device:
        res = res.to(values.device)
    return res

def pad_or_cut_3d(values: torch.tensor, tgt_len, dim=-1, pad_value=0):
    if dim == 0 or dim == -3:
        src_len = values.shape[0]
        if src_len < tgt_len:
            res = F.pad(values, [0, 0, 0, 0, 0, tgt_len - src_len], value=pad_value)
        else:
            res = values[:tgt_len]
    elif dim == 1 or dim == -2:
        src_len = values.shape[1]
        if src_len < tgt_len:
            res = F.pad(values, [0, 0, 0, tgt_len - src_len], value=pad_value)
        else:
            res = values[:, :tgt_len]
    elif dim == 2 or dim == -1:
        src_len = values.shape[2]
        if src_len < tgt_len:
            res = F.pad(values, [0, tgt_len - src_len], value=pad_value)
        else:
            res = values[:, :, :tgt_len]
    else:
        raise RuntimeError(f"Wrong dim number {dim} while the tensor only has {len(values.shape)} dimensions.")
    if res.device != values.device:
        res = res.to(values.device)
    return res

def pad_or_cut_xd(values, tgt_len, dim=-1, pad_value=0):
    if len(values.shape) == 1:
        return pad_or_cut_1d(values, tgt_len, pad_value)
    elif len(values.shape) == 2:
        return pad_or_cut_2d(values, tgt_len, dim, pad_value)
    elif len(values.shape) == 3:
        return pad_or_cut_3d(values, tgt_len, dim, pad_value)
    else:
        raise NotImplementedError


def _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
    if len(batch) == 0:
        return 0
    if len(batch) == max_sentences:
        return 1
    if num_tokens > max_tokens:
        return 1
    return 0


def batch_by_size(
        indices, num_tokens_fn, max_tokens=None, max_sentences=None,
        required_batch_size_multiple=1, distributed=False
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N (default: 1).
    """
    max_tokens = max_tokens if max_tokens is not None else sys.maxsize
    max_sentences = max_sentences if max_sentences is not None else sys.maxsize
    bsz_mult = required_batch_size_multiple

    if isinstance(indices, types.GeneratorType):
        indices = np.fromiter(indices, dtype=np.int64, count=-1)

    sample_len = 0
    sample_lens = []
    batch = []
    batches = []
    for i in range(len(indices)):
        idx = indices[i]
        num_tokens = num_tokens_fn(idx)
        sample_lens.append(num_tokens)
        sample_len = max(sample_len, num_tokens)

        assert sample_len <= max_tokens, (
            "sentence at index {} of size {} exceeds max_tokens "
            "limit of {}!".format(idx, sample_len, max_tokens)
        )
        num_tokens = (len(batch) + 1) * sample_len

        if _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )
            batches.append(batch[:mod_len])
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
        batch.append(idx)
    if len(batch) > 0:
        batches.append(batch)
    return batches


def unpack_dict_to_list(samples):
    samples_ = []
    bsz = samples.get('outputs').size(0)
    for i in range(bsz):
        res = {}
        for k, v in samples.items():
            try:
                res[k] = v[i]
            except:
                pass
        samples_.append(res)
    return samples_


def remove_padding(x, padding_idx=0):
    if x is None:
        return None
    assert len(x.shape) in [1, 2]
    if len(x.shape) == 2:  # [T, H]
        return x[np.abs(x).sum(-1) != padding_idx]
    elif len(x.shape) == 1:  # [T]
        return x[x != padding_idx]


def data_loader(fn):
    """
    Decorator to make any fx with this use the lazy property
    :param fn:
    :return:
    """

    wraps(fn)
    attr_name = '_lazy_' + fn.__name__

    def _get_data_loader(self):
        try:
            value = getattr(self, attr_name)
        except AttributeError:
            try:
                value = fn(self)  # Lazy evaluation, done only once.
            except AttributeError as e:
                # Guard against AttributeError suppression. (Issue #142)
                traceback.print_exc()
                error = f'{fn.__name__}: An AttributeError was encountered: ' + str(e)
                raise RuntimeError(error) from e
            setattr(self, attr_name, value)  # Memoize evaluation.
        return value

    return _get_data_loader


def build_dataloader(dataset, shuffle, use_ddp, max_tokens=None, max_sentences=None,
                     required_batch_size_multiple=-1, endless=False, is_batch_by_size=True,
                     chunked_read=True, num_workers=None, training=False, drop_last=True,
                     max_batches=-1, prefetch_factor=2):
    if training:
        devices_cnt = torch.cuda.device_count()
        if devices_cnt == 0:
            devices_cnt = 1
        if required_batch_size_multiple == -1:
            required_batch_size_multiple = devices_cnt
    else:
        devices_cnt = 1
        required_batch_size_multiple = 1

    def shuffle_batches(batches):
        np.random.shuffle(batches)
        return batches

    if max_tokens is not None:
        max_tokens *= devices_cnt
    if max_sentences is not None:
        max_sentences *= devices_cnt
    if shuffle:
        indices = dataset.ordered_indices()
    else:
        indices = list(range(len(dataset.ordered_indices())))
    if is_batch_by_size:
        batch_sampler = batch_by_size(
            indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )
    else:
        batch_sampler = []
        for i in range(0, len(indices), max_sentences):
            batch_sampler.append(indices[i:i + max_sentences])

    if shuffle:
        batches = shuffle_batches(list(batch_sampler))
    else:
        batches = batch_sampler
    num_workers = dataset.num_workers if num_workers is None else num_workers
    if use_ddp and training:
        import torch.distributed as dist
        num_replicas = dist.get_world_size()
        rank = dist.get_rank()
        batches = [
            x[rank::num_replicas] for x in batches
            if len(x) % num_replicas == 0 or not drop_last
        ]

    if chunked_read:
        batches = [[x] for x in batches]

    class Sampler(torch.utils.data.Sampler):
        def __init__(self):
            super(Sampler, self).__init__(None)
            self.rest_batches = []
            self.epoch = 0

        def __iter__(self):
            if len(self.rest_batches) == 0 or max_batches == -1:
                self.rest_batches = shuffle_batches(batches)
                self.epoch += 1
            if max_batches > 0:
                it = self.rest_batches[:max_batches]
            else:
                it = self.rest_batches
            yield from it
            if max_batches > 0:
                self.rest_batches = self.rest_batches[max_batches:]

        def __len__(self):
            return max_batches

    return torch.utils.data.DataLoader(
        dataset,
        collate_fn=dataset.collater,
        batch_sampler=Sampler() if training else batches,
        num_workers=num_workers,
        pin_memory=False,
        worker_init_fn=dataset.init_worker,
        prefetch_factor=prefetch_factor)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, path, shuffle, load_size=True):
        super().__init__()
        self.shuffle = shuffle
        self.path = path
        self.load_size = load_size
        if load_size:
            self.sizes_ = None
            self.key_and_sizes = self.get_key_and_sizes()

    def get_key_and_sizes(self):
        raise NotImplementedError

    @property
    def _sizes(self):
        if self.load_size:
            if self.sizes_ is None:
                self.sizes_ = [x[1] for x in self.key_and_sizes]
            return self.sizes_
        else:
            raise Exception("Size of dataset is not loaded...")

    def __getitem__(self, index):
        raise NotImplementedError

    def collater(self, samples):
        raise NotImplementedError

    def __len__(self):
        if self.load_size:
            return len(self._sizes)
        else:
            raise Exception("Size of dataset is not loaded...")

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self._sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        indices = np.random.permutation(len(self))
        indices = indices[np.argsort(np.array(self._sizes)[indices], kind='mergesort')]
        return indices

    @property
    def num_workers(self):
        return int(os.getenv('NUM_WORKERS', hparams['ds_workers']))

class BaseKVDataset(BaseDataset):
    def __init__(self, path,  shuffle, num_parallel=4, load_size=True, chunk_size=5):
        super().__init__(path, shuffle, load_size)
        self.num_parallel = num_parallel
        self.chunk_size = chunk_size
        self.indexed_ds = None

    def get_falcon_reader(self, num_parallel=None):
        path = self.path
        if os.path.exists(f'{self.path}.hdfs') and hparams.get('use_hdfs_dataset'):
            path = open(f'{self.path}.hdfs').readlines()[0]
            print("| use hdfs dataset: ", path)
        cmd = f"hdfs dfs -ls {path}* | grep index$ | wc -l"
        num_shard = int(subprocess.check_output(cmd, shell=True).decode().strip())
        path_list = []
        if num_shard > 1:
            path_list = ['{}{}'.format(self.path, shard) for shard in range(num_shard)]
        else:
            path_list = [self.path]
        reader = FalconReader(
            path_list,
            128,  # fd_cache_size
            128,  # io_thread_num
            5,  # io_retry
            "tts_data",  # unused
            1,
            0,
            self.chunk_size)
        entry_num = reader.get_entry_num(list(range(num_shard)), False)
        print("entry num: ", entry_num)
        return reader
        
    def get_kv_reader(self, num_parallel=None):
        path = self.path
        if os.path.exists(f'{self.path}.hdfs') and self.hparams.get('use_hdfs_dataset'):
            path = open(f'{self.path}.hdfs').readlines()[0]
            with HiddenPrints():
                fs = hdfs.connect()
            if not path.endswith('/data') and not path.endswith('/data_sorted'):
                path = f'{path}/data'
            if fs.exists(f'{path}_sorted.index'):
                path = f'{path}_sorted'
            print("| use hdfs dataset: ", path)
        return KVReader(path)


    def init_worker(self):
        pass

    def __getitem__(self, index):
        if isinstance(index, int):
            index = [index]
        items = self.indexed_ds.read_many([str(self.key_and_sizes[i][0]) for i in index])
        samples = []
        for item, i in zip(items, index):
            item = pickle.loads(item)
            sample = self.get_sample(i, item)
            if sample is not None:
                samples.append(sample)
        return samples

    def get_sample(self, id, item):
        raise NotImplementedError

    def collater(self, samples):
        raise NotImplementedError


class BaseConcatDatasetNoGroup(ConcatDataset):
    def __init__(self, datasets, n_workers=None):
        super().__init__(datasets)
        self.n_workers = n_workers
        self.sizes = None

    def __getitem__(self, idx):
        rest = None
        if isinstance(idx, (tuple, list)):
            rest = idx[1:]
            idx = idx[0]
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        if rest is not None:
            return self.datasets[dataset_idx][(sample_idx, *rest)]
        else:
            return self.datasets[dataset_idx][sample_idx]

    def collater(self, samples):
        return self.datasets[0].collater(samples)

    @property
    def _sizes(self):
        if self.sizes is None:
            self.sizes = list(chain.from_iterable([d._sizes for d in self.datasets]))
        return self.sizes

    def size(self, index):
        return min(self._sizes[index], hparams['max_frames'])

    def num_tokens(self, index):
        return self.size(index)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        indices = np.random.permutation(len(self))
        indices = indices[np.argsort(np.array(self._sizes)[indices], kind='mergesort')]
        return indices

    @property
    def num_workers(self):
        if self.n_workers is None:
            return self.datasets[0].num_workers
        else:
            return self.n_workers

    @classmethod
    def init_worker(cls, worker_id):
        worker_info = torch.utils.data.get_worker_info()
        self_: BaseConcatDatasetNoGroup = worker_info.dataset
        for self in self_.datasets:
            self.init_worker(worker_id, self)


class BaseConcatDataset(BaseConcatDatasetNoGroup):
    def __getitem__(self, idxs):
        samples = []
        ds_idxs = [[] for _ in self.cumulative_sizes]
        for idx in idxs:
            if idx < 0:
                if -idx > len(self):
                    raise ValueError("absolute value of index should not exceed dataset length")
                idx = len(self) + idx
            dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
            if dataset_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            ds_idxs[dataset_idx].append(sample_idx)
        for ds, idxs in zip(self.datasets, ds_idxs):
            samples += ds[idxs]
        return samples


class KVSamplerDist(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None,
                 shuffle=True, drop_last=False, endless=False):
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle,
            drop_last=drop_last)
        self.batch_size = batch_size
        self.endless = endless

    def __iter__(self):
        while True:
            iterable = super().__iter__()
            yield from chunk(iterable, self.batch_size)
            if not self.endless:
                break

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size


class KVSampler(torch.utils.data.BatchSampler):
    def __init__(self, dataset, batch_size, drop_last=False, endless=False):
        super().__init__(dataset, batch_size, drop_last=drop_last)
        self.batch_size = batch_size
        self.endless = endless

    def __iter__(self):
        while True:
            iterable = super().__iter__()
            yield from chunk(iterable, self.batch_size)
            if not self.endless:
                break

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size
