import os
import sys
import math
import traceback
import types
from functools import wraps
from itertools import chain
import numpy as np
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from utils.commons.hparams import hparams


def sequence_mask(seq_lens, max_len=None, device='cpu'):
        b = seq_lens.shape[0]
        if max_len is None:
            max_len = seq_lens.max()
        mask = torch.arange(max_len).unsqueeze(0).to(device) # [1, t]
        mask = mask < (seq_lens.unsqueeze(1)) # [1, t] + [b, 1] = [b, t]
        mask = mask.float()
        return mask


def collate_xd(values, pad_idx=0, left_pad=False, shift_right=False, max_len=None, shift_id=1):
    if len(values[0].shape) == 1:
        return collate_1d(values, pad_idx, left_pad, shift_right, max_len, shift_id)
    elif len(values[0].shape) == 2:
        return collate_2d(values, pad_idx, left_pad, shift_right, max_len)
    elif len(values[0].shape) == 3:
        return collate_3d(values, pad_idx, left_pad, shift_right, max_len)

def collate_1d_or_2d(values, pad_idx=0, left_pad=False, shift_right=False, max_len=None, shift_id=1):
    if len(values[0].shape) == 1:
        return collate_1d(values, pad_idx, left_pad, shift_right, max_len, shift_id)
    else:
        return collate_2d(values, pad_idx, left_pad, shift_right, max_len)


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
        assert dst.numel() == src.numel()
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


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, shuffle):
        super().__init__()
        self.hparams = hparams
        self.shuffle = shuffle
        self.sort_by_len = hparams['sort_by_len']
        self.sizes = None

    @property
    def _sizes(self):
        return self.sizes

    def __getitem__(self, index):
        raise NotImplementedError

    def collater(self, samples):
        raise NotImplementedError

    def __len__(self):
        return len(self._sizes)

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return min(self._sizes[index], hparams['max_frames'])

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
            if self.sort_by_len:
                indices = indices[np.argsort(np.array(self._sizes)[indices], kind='mergesort')]
        else:
            indices = np.arange(len(self))
        return indices

    @property
    def num_workers(self):
        return int(os.getenv('NUM_WORKERS', hparams['num_workers']))


class BaseConcatDataset(ConcatDataset):
    def collater(self, samples):
        return self.datasets[0].collater(samples)

    @property
    def _sizes(self):
        if not hasattr(self, 'sizes'):
            self.sizes = list(chain.from_iterable([d._sizes for d in self.datasets]))
        return self.sizes

    def size(self, index):
        return min(self._sizes[index], hparams['max_frames'])

    def num_tokens(self, index):
        return self.size(index)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.datasets[0].shuffle:
            indices = np.random.permutation(len(self))
            if self.datasets[0].sort_by_len:
                indices = indices[np.argsort(np.array(self._sizes)[indices], kind='mergesort')]
        else:
            indices = np.arange(len(self))
        return indices

    @property
    def num_workers(self):
        return self.datasets[0].num_workers


def build_dataloader(dataset, shuffle=False, use_ddp=False, max_tokens=None, max_sentences=1,
                     required_batch_size_multiple=-1, endless=False, is_batch_by_size=True,
                     chunked_read=False, num_workers=None, training=False, drop_last=True,
                     max_batches=-1, prefetch_factor=2, sample_by_weight=False):
    if sample_by_weight:
        assert is_batch_by_size is False
        assert chunked_read is False
        assert training is True
        assert hasattr(dataset, 'get_weight')
        indices = dataset.ordered_indices()
        if use_ddp:
            import torch.distributed as dist
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
            chunk_size = math.ceil(len(indices) / num_replicas)
            indices = indices[rank::num_replicas] 
            if len(indices) < chunk_size:
                indices.extend(np.random.choice(dataset.ordered_indices(), (chunk_size - len(indices,)), replace=False).tolist())
        weights = [dataset.get_weight(i) for i in indices]
        weights = torch.DoubleTensor(weights)
        sampler = WeightedDatasetSampler(indices, weights, endless=endless)
        return torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collater,
            batch_size=max_sentences,
            sampler=sampler,
            num_workers=dataset.num_workers if num_workers is None else num_workers,
            pin_memory=False,
            # worker_init_fn=dataset.init_worker,
            prefetch_factor=prefetch_factor)


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
        if endless:
            batches = [b for _ in range(10000) for b in shuffle_batches(list(batch_sampler))]
    else:
        batches = batch_sampler
        if endless:
            batches = [b for _ in range(10000) for b in batches]
    num_workers = dataset.num_workers if num_workers is None else num_workers
    if num_workers == 0:
        prefetch_factor = None
    # if use_ddp:
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
            if max_batches > 0:
                return max_batches
            else:
                return len(batches)

    return torch.utils.data.DataLoader(
        dataset,
        collate_fn=dataset.collater,
        batch_sampler=Sampler() if training else batches,
        num_workers=num_workers,
        pin_memory=False,
        # worker_init_fn=dataset.init_worker,
        prefetch_factor=prefetch_factor)


class WeightedDatasetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, indices, weights, endless=False):
        self.indices = indices
        self.weights = weights
        self.num_samples = len(self.indices)
        if endless:
            self.num_samples *= 10000

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class BaseManifestDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False, manifest_paths=None):
        super().__init__(shuffle)
        self.prefix = prefix
        if manifest_paths is None:
            manifest_paths = hparams[f'{prefix}_manifests']
        self.manifest_paths = manifest_paths


class SkipLogger:
    def __init__(self, keys: list = (), interval=1000, i_worker=0, n_worker=1):
        self.logger = {
            'item_cnt': 0,
            'just_skip': 0,
        }
        for k in keys:
            self.logger[k] = 0
        self.interval = interval
        self.i_worker = i_worker
        self.n_worker = n_worker
    
    def update(self, cnt=1, k=None):
        k = 'just_skip' if k is None else k
        if k not in self.logger:
            self.logger[k] = 0
        self.logger[k] += cnt
        self.logger['item_cnt'] += cnt
        if (
            self.logger[k] > 0 and (
                self.logger[k] % self.interval == 0 or
                self.logger[k] // self.interval != (self.logger[k] - cnt) // self.interval
            )
        ):
            if k == 'just_skip':
                print(f"| processer#{self.i_worker}/{self.n_worker}: skipped [{self.logger[k]}/{self.logger['item_cnt']}] items")
            else:
                print(f"| processer#{self.i_worker}/{self.n_worker}: skipped [{self.logger[k]}/{self.logger['item_cnt']}] items for [{k}]")

    def report(self, cnt=1, k=None):
        return self.update(cnt, k)

    def step(self, cnt=1):
        self.logger['item_cnt'] += cnt
