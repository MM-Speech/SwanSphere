from collections import defaultdict
import torch
import torch.nn.functional as F
from typing import Optional


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (
                   torch.cumsum(mask, dim=1).type_as(mask) * mask
           ).long() + padding_idx


def softmax(x, dim):
    return F.softmax(x, dim=dim, dtype=torch.float32)


# def sequence_mask(lengths, maxlen=None):
#     if maxlen is None:
#         maxlen = int(lengths.max().detach().to(torch.int64))
#     else:
#         assert maxlen >= int(lengths.max().detach().to(torch.int64)), (f'{maxlen} vs {lengths.max()}')
#     device = lengths.device
#     seq = torch.arange(maxlen, device=device)        # [T]
#     mask = seq.unsqueeze(0) < lengths.unsqueeze(1)   # [B, T], bool
#     return mask


def sequence_mask(lengths: torch.Tensor, maxlen: Optional[int] = None) -> torch.Tensor:
    assert lengths.ndim == 1, f"lengths must be 1D, got {lengths.shape}"
    assert lengths.dtype in (torch.int32, torch.int64), f"lengths dtype must be int, got {lengths.dtype}"

    lmax_cpu = int(lengths.detach().cpu().max().item())
    if maxlen is None:
        maxlen = lmax_cpu
    else:
        assert maxlen >= lmax_cpu, f"maxlen({maxlen}) < max(lengths)({lmax_cpu})"

    lengths = lengths.clamp(min=0, max=maxlen)

    device = lengths.device
    seq = torch.arange(maxlen, device=device)
    mask = seq.unsqueeze(0) < lengths.unsqueeze(1)
    return mask  # bool


def weights_nonzero_speech(target):
    # target : B x T x mel
    # Assign weight 1.0 to all labels except for padding (id=0).
    dim = target.size(-1)
    return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)


INCREMENTAL_STATE_INSTANCE_ID = defaultdict(lambda: 0)


def _get_full_incremental_state_key(module_instance, key):
    module_name = module_instance.__class__.__name__

    # assign a unique ID to each module instance, so that incremental state is
    # not shared across module instances
    if not hasattr(module_instance, '_instance_id'):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._instance_id = INCREMENTAL_STATE_INSTANCE_ID[module_name]

    return '{}.{}.{}'.format(module_name, module_instance._instance_id, key)


def get_incremental_state(module, incremental_state, key):
    """Helper for getting incremental state for an nn.Module."""
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    return incremental_state[full_key]


def set_incremental_state(module, incremental_state, key, value):
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def fill_with_neg_inf2(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(-1e8).type_as(t)


def select_attn(attn_logits, type='best'):
    """

    :param attn_logits: [n_layers, B, n_head, T_sp, T_txt]
    :return:
    """
    encdec_attn = torch.stack(attn_logits, 0).transpose(1, 2)
    # [n_layers * n_head, B, T_sp, T_txt]
    encdec_attn = (encdec_attn.reshape([-1, *encdec_attn.shape[2:]])).softmax(-1)
    if type == 'best':
        indices = encdec_attn.max(-1).values.sum(-1).argmax(0)
        encdec_attn = encdec_attn.gather(
            0, indices[None, :, None, None].repeat(1, 1, encdec_attn.size(-2), encdec_attn.size(-1)))[0]
        return encdec_attn
    elif type == 'mean':
        return encdec_attn.mean(0)


def make_pad_mask(lengths, xs=None, length_dim=-1):
    """Make mask tensor containing indices of padded part.
    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.
    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)
    Examples:
        With only lengths.
        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
        With the reference tensor.
        >>> xs = torch.zeros((3, 2, 4))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 1],
                 [0, 0, 0, 1]],
                [[0, 0, 1, 1],
                 [0, 0, 1, 1]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)
        With the reference tensor and dimension indicator.
        >>> xs = torch.zeros((3, 6, 6))
        >>> make_pad_mask(lengths, xs, 1)
        tensor([[[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]]], dtype=torch.uint8)
        >>> make_pad_mask(lengths, xs, 2)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)
    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if xs is None:
        maxlen = int(max(lengths))
    else:
        maxlen = xs.size(length_dim)

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


def make_non_pad_mask(lengths, xs=None, length_dim=-1):
    """Make mask tensor containing indices of non-padded part.
    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.
    Returns:
        ByteTensor: mask tensor containing indices of padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)
    Examples:
        With only lengths.
        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1 ,1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
        With the reference tensor.
        >>> xs = torch.zeros((3, 2, 4))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1],
                 [1, 1, 1, 1]],
                [[1, 1, 1, 0],
                 [1, 1, 1, 0]],
                [[1, 1, 0, 0],
                 [1, 1, 0, 0]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=torch.uint8)
        With the reference tensor and dimension indicator.
        >>> xs = torch.zeros((3, 6, 6))
        >>> make_non_pad_mask(lengths, xs, 1)
        tensor([[[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]]], dtype=torch.uint8)
        >>> make_non_pad_mask(lengths, xs, 2)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=torch.uint8)
    """
    return ~make_pad_mask(lengths, xs, length_dim)


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len).to(lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def group_hidden_by_segs(h, seg_ids, max_len):
    """

    :param h: [B, T, H]
    :param seg_ids: [B, T]
    :return: h_ph: [B, T_ph, H]
    """
    B, T, H = h.shape
    h_gby_segs = h.new_zeros([B, max_len + 1, H]).scatter_add_(1, seg_ids[:, :, None].repeat([1, 1, H]), h)
    all_ones = h.new_ones(h.shape[:2])
    cnt_gby_segs = h.new_zeros([B, max_len + 1]).scatter_add_(1, seg_ids, all_ones).contiguous()
    h_gby_segs = h_gby_segs[:, 1:]
    cnt_gby_segs = cnt_gby_segs[:, 1:]
    h_gby_segs = h_gby_segs / torch.clamp(cnt_gby_segs[:, :, None], min=1)
    return h_gby_segs, cnt_gby_segs

def expand_by_repeat_times(source_encoding, lengths):
    """
    source_encoding: [T, C]
    lengths, list of int, [T,], how many times each token should repeat
    return:
        expanded_encoding: [T_expand, C]
    """
    hid_dim = source_encoding.shape[1]
    out2source = []
    for i, length in enumerate(lengths):
        out2source += [i for _ in range(length)]
    out2source = torch.LongTensor(out2source).to(source_encoding.device)
    out2source_ = out2source[:, None].repeat([1, hid_dim])
    expanded_encoding = torch.gather(source_encoding, 0, out2source_)  # [B, T, H]
    return expanded_encoding


def expand_word2ph(word_encoding, ph2word):
    word_encoding = F.pad(word_encoding,[0,0,1,0])
    ph2word_ = ph2word[:, :, None].repeat([1, 1, word_encoding.shape[-1]])
    out = torch.gather(word_encoding, 1, ph2word_)  # [B, T, H]
    return out


def add_prefix(seq_1, seq_1_len, seq_2, seq_2_len, padding_value=0):
    B, T, device = seq_1.size(0), (seq_1_len+seq_2_len).max(), seq_1.device
    seq_merged = torch.full([B, T], padding_value, device=device, dtype=seq_2.dtype)

    """ Assign seq 1"""
    T_text = seq_1.shape[1]
    T_feat = seq_2.shape[1]
    indics_x = torch.arange(T, device=device)[None, :]
    mask_x = (indics_x < seq_1_len[:, None]) & (indics_x < T_text)
    mask_text = (
        torch.arange(seq_1.shape[1], device=device)[None, :]
        < seq_1_len[:, None]
    )
    seq_merged[mask_x] = seq_1[mask_text].to(dtype=seq_merged.dtype)

    """ Assign seq 2"""
    mask_x = (
        (seq_1_len[:, None] <= indics_x)
        & (indics_x < (seq_1_len + seq_2_len)[:, None])
        & (indics_x - seq_1_len[:, None] < T_feat)
    )
    mask_noisy = (
        torch.arange(T_feat, device=device)[None, :] < seq_2_len[:, None]
    )
    seq_merged[mask_x] = seq_2[mask_noisy].to(dtype=seq_merged.dtype)
    return seq_merged


def add_prefix_2d(seq_1, seq_1_len, seq_2, seq_2_len, padding_value=0):
    from einops import rearrange
    bsz, _, dim = seq_1.shape
    seq_1 = rearrange(seq_1, 'b t c -> (b c) t')
    seq_2 = rearrange(seq_2, 'b t c -> (b c) t')
    seq_1_len = seq_1_len.repeat_interleave(dim)
    seq_2_len = seq_2_len.repeat_interleave(dim)
    seq_merged = add_prefix(seq_1, seq_1_len, seq_2, seq_2_len, padding_value)
    seq_merged = rearrange(seq_merged, '(b c) t -> b t c', b=bsz)
    return seq_merged


def add_prefix_nd(seq_1, seq_1_len, seq_2, seq_2_len, padding_value=0):
    T_max = (seq_1_len + seq_2_len).max().item()
    
    B = seq_1.size(0)
    extra_dims = seq_1.shape[2:]
    
    seq_merged = torch.full(
        [B, T_max] + list(extra_dims),
        padding_value,
        device=seq_1.device,
        dtype=seq_1.dtype
    )
    
    indices = torch.arange(T_max, device=seq_1.device).repeat(B, 1)
    
    mask_seq1 = (indices < seq_1_len.unsqueeze(1))
    
    batch_idx = torch.arange(B, device=seq_1.device).unsqueeze(1).expand(-1, T_max)
    
    if mask_seq1.any():
        valid_batch, valid_time = torch.where(mask_seq1)
        valid_time_source = torch.minimum(valid_time, torch.tensor(seq_1.size(1) - 1, device=seq_1.device))
        seq_merged[valid_batch, valid_time] = seq_1[valid_batch, valid_time_source]
    
    start_seq2 = seq_1_len.unsqueeze(1)
    rel_indices = indices - start_seq2
    
    mask_seq2 = (rel_indices >= 0) & (rel_indices < seq_2_len.unsqueeze(1))
    
    if mask_seq2.any():
        valid_batch, valid_time = torch.where(mask_seq2)
        source_time = rel_indices[valid_batch, valid_time].long()
        source_time = torch.minimum(source_time, torch.tensor(seq_2.size(1) - 1, device=seq_2.device))
        seq_merged[valid_batch, valid_time] = seq_2[valid_batch, source_time]
    
    return seq_merged


def remove_prefix(merged_out, prefix_lens, output_lens):
    B, device = merged_out.size(0), merged_out.device

    seq_output = torch.full([B, output_lens.max(), merged_out.size(-1)], 0, device=device, dtype=merged_out.dtype)

    T0 = output_lens.max()
    T1 = merged_out.size(1)
    indics0 = torch.arange(T0, device=device)[None, :]
    indics1 = torch.arange(T1, device=device)[None, :]

    mask0 = (indics0 < output_lens[:, None]) & (
        (prefix_lens[:, None] + indics0) < T1
    )
    mask1 = (prefix_lens[:, None] <= indics1) & (
        indics1 < (prefix_lens[:, None] + output_lens[:, None])
    )

    seq_output[mask0] = merged_out[mask1]
    return seq_output


def remove_suffix(merged_out, prefix_lens):
    B, device = merged_out.size(0), merged_out.device
    max_prefix = prefix_lens.max().item()
    
    modal_out = torch.zeros([B, max_prefix, merged_out.size(-1)], 
                          device=device, dtype=merged_out.dtype)
    
    indices_modal = torch.arange(max_prefix, device=device).unsqueeze(0)  # [1, max_prefix]
    valid_modal = indices_modal < prefix_lens.unsqueeze(1)  # [B, max_prefix]: 前缀有效区域
    
    indices_merged = torch.arange(merged_out.size(1), device=device).unsqueeze(0)  # [1, T]
    valid_merged = indices_merged < prefix_lens.unsqueeze(1)  # [B, T]
    
    modal_out[valid_modal] = merged_out[valid_merged]
    return modal_out


def last_token_mask(attention_mask: torch.Tensor, right_pad=True) -> torch.Tensor:
    B, S = attention_mask.shape
    am = attention_mask.to(torch.int64)
    lengths = am.sum(dim=-1) 
    last_idx = (lengths - 1).clamp(min=0)
    label_mask = torch.zeros_like(am)
    valid = lengths > 0
    label_mask[torch.arange(B, device=attention_mask.device)[valid], last_idx[valid]] = 1
    return label_mask


def build_last_k_soft_labels(h_mask, K: int = 3, gamma: float = 2.0):
    device = h_mask.device
    B, T = h_mask.shape

    lengths = h_mask.long().sum(dim=1)            # [B]
    positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)  # [B, T]
    last_idx = lengths - 1                        # [B]
    dist_to_end = last_idx.unsqueeze(1) - positions    # [B, T]

    base = (K - dist_to_end.float()) / float(K)
    base = base.clamp(min=0.0, max=1.0)

    # 非线性：p -> p^gamma（gamma>1 时靠近1的地方更突出）
    labels = base ** gamma

    labels = labels * h_mask.float()
    labels.scatter_(1, last_idx.unsqueeze(1), 1.0)
    return labels


