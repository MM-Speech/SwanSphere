import os
import importlib
import sys

import torch
import torch.nn.functional as F

from utils.commons.io import print_once

DISABLE_FA = os.getenv("DISABLE_FLASH_ATTENTION", os.getenv("DISABLE_FLASH_ATTN", "0")) in ("1", "true", "True")
FORCE_FA_BACKEND = os.getenv("FORCE_FLASH_ATTN_BACKEND")  # "fa3", "fa2", "none"

fa2_available = importlib.util.find_spec("flash_attn") is not None
fa3_available = importlib.util.find_spec("flash_attn_interface") is not None

flash_attn_installed = False
flash_attn_backend = "none"  # "fa3_hopper" | "fa2" | "none"

fa_padding_utils_installed = False
fa_attn_bias_supported = False

flash_attn_varlen_func = None
flash_attn_func = None
flash_attn_with_kvcache = None

index_first_axis = None
pad_input = None
unpad_input = None

layer_norm_fn = None
RMSNorm = None

def get_unpad_data(*args, **kwargs):
    try:
        from transformers.modeling_flash_attention_utils import _get_unpad_data
    except Exception:
        from transformers.models.llama.modeling_llama import _get_unpad_data
    return _get_unpad_data(*args, **kwargs)

def _select_backend():
    global flash_attn_installed, flash_attn_backend
    global fa_padding_utils_installed, fa_attn_bias_supported
    global flash_attn_varlen_func, flash_attn_func, flash_attn_with_kvcache
    global index_first_axis, pad_input, unpad_input
    global layer_norm_fn, RMSNorm

    if DISABLE_FA or FORCE_FA_BACKEND == "none":
        print_once("| FlashAttention disabled by env")
        flash_attn_installed = False
        flash_attn_backend = "none"
        from utils.nn.fa_fallbacks import layer_norm_fn as _layer_norm_fn, RMSNorm as _RMSNorm
        layer_norm_fn = _layer_norm_fn
        RMSNorm = _RMSNorm
        return

    target = FORCE_FA_BACKEND
    if target == "fa3" and fa3_available:
        use_fa3 = True
    elif target == "fa2" and fa2_available:
        use_fa3 = False
    elif fa3_available:
        use_fa3 = True
    elif fa2_available:
        use_fa3 = False
    else:
        use_fa3 = None

    if use_fa3 is True:
        from flash_attn_interface import flash_attn_varlen_func as _varlen
        from flash_attn_interface import flash_attn_func as _fa
        from flash_attn_interface import flash_attn_with_kvcache as _kvcache
        flash_attn_varlen_func = _varlen
        flash_attn_func = _fa
        flash_attn_with_kvcache = _kvcache
        flash_attn_installed = True
        flash_attn_backend = "fa3_hopper"
        print_once("| use fa3_hopper")
    elif use_fa3 is False:
        from flash_attn import flash_attn_varlen_func as _varlen
        from flash_attn import flash_attn_func as _fa
        from flash_attn import flash_attn_with_kvcache as _kvcache
        flash_attn_varlen_func = _varlen
        flash_attn_func = _fa
        flash_attn_with_kvcache = _kvcache
        flash_attn_installed = True
        flash_attn_backend = "fa2"
        print_once("| use fa2")
    else:
        flash_attn_installed = False
        flash_attn_backend = "none"
        print_once("| flash-attn not available")

    try:
        from flash_attn.bert_padding import index_first_axis as _index_first_axis
        from flash_attn.bert_padding import pad_input as _pad_input
        from flash_attn.bert_padding import unpad_input as _unpad_input
        index_first_axis = _index_first_axis
        pad_input = _pad_input
        unpad_input = _unpad_input
        fa_padding_utils_installed = True
    except Exception:
        index_first_axis = None
        pad_input = None
        unpad_input = None
        fa_padding_utils_installed = False
        
    if flash_attn_installed:
        try:
            import inspect
            sig = inspect.signature(flash_attn_func)
            fa_attn_bias_supported = 'attn_bias' in sig.parameters
        except (ValueError, TypeError):
            fa_attn_bias_supported = fa3_available

    try:
        from flash_attn.ops.triton.layer_norm import layer_norm_fn as _layer_norm_fn, RMSNorm as _RMSNorm
        layer_norm_fn = _layer_norm_fn
        RMSNorm = _RMSNorm
    except Exception:
        from utils.nn.fa_fallbacks import layer_norm_fn as _layer_norm_fn, RMSNorm as _RMSNorm
        layer_norm_fn = _layer_norm_fn
        RMSNorm = _RMSNorm
        
        
def cross_attn_varlen(q, k, v, query_mask, context_mask):
    global flash_attn_installed
    if flash_attn_installed and flash_attn_backend != 'none':
        indices_q, cu_seqlens_q, max_seqlen_q = get_unpad_data(query_mask)
        indices_k, cu_seqlens_k, max_seqlen_k = get_unpad_data(context_mask)

        B, Lq, N, D = q.shape
        _, Lk, _, _ = k.shape

        q_unpad = index_first_axis(q.reshape(B*Lq, N, D), indices_q)
        k_unpad = index_first_axis(k.reshape(B*Lk, N, D), indices_k)
        v_unpad = index_first_axis(v.reshape(B*Lk, N, D), indices_k)

        out_unpad = flash_attn_varlen_func(
            q_unpad, k_unpad, v_unpad,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=False,
        )
        if isinstance(out_unpad, (tuple, list)):
            out_unpad = out_unpad[0]

        out = pad_input(out_unpad, indices_q, B, Lq)  # [B, Lq, N, D]
        return out
    else:
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "q/k/v must be 4D [B, L, N, D]"
        B, Lq, N, D = q.shape
        Bk, Lk, Nk, Dk = k.shape
        assert B == Bk and N == Nk and D == Dk, "q/k/v batch, heads, and head_dim must match"
        
        q_bhld = q.permute(0, 2, 1, 3).contiguous()  # [B, N, Lq, D]
        k_bhld = k.permute(0, 2, 1, 3).contiguous()  # [B, N, Lk, D]
        v_bhld = v.permute(0, 2, 1, 3).contiguous()  # [B, N, Lk, D]
        
        qm = query_mask.to(dtype=torch.bool, device=q.device)      # [B, Lq]
        km = context_mask.to(dtype=torch.bool, device=q.device)    # [B, Lk]
        
        attn_mask = (~km).view(B, 1, 1, Lk)  # True at invalid keys

        # from torch.backends.cuda import sdp_kernel
        # with sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        out_bhld = F.scaled_dot_product_attention(
            q_bhld, k_bhld, v_bhld,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False
        )  # [B, N, Lq, D]
        
        # 若某个 batch 的所有 key 都无效，SDPA 可能返回 NaN；用 where 全量置零
        has_valid_k = km.any(dim=1).view(B, 1, 1, 1)          # [B,1,1,1]
        out_bhld = torch.where(has_valid_k, out_bhld, torch.zeros_like(out_bhld))
        
        # 将无效的 query 位置输出置零
        out_bhld = out_bhld * qm.view(B, 1, Lq, 1)
        
        # 转回 [B, Lq, N, D]
        out = out_bhld.permute(0, 2, 1, 3).contiguous()
        return out


def make_additive_mask_from_padding(kv_padding_mask: torch.Tensor, q_len: int, device=None, dtype=None):
    """
    kv_padding_mask: [B, S_kv] with 1/True for valid, 0/False for pad
    Return additive mask for SDPA: shape [B, 1, q_len, S_kv], 0 for valid, -inf for pad
    """
    if kv_padding_mask is None:
        return None
    if kv_padding_mask.dtype != torch.bool:
        kv_padding_mask = kv_padding_mask.bool()
    B, S_kv = kv_padding_mask.shape
    mask = (~kv_padding_mask).unsqueeze(1).unsqueeze(1).expand(B, 1, q_len, S_kv)  # True where pad
    add_mask = torch.zeros((B, 1, q_len, S_kv), device=kv_padding_mask.device if device is None else device, dtype=torch.float32 if dtype is None else dtype)
    add_mask.masked_fill_(mask, float("-inf"))
    return add_mask


_select_backend()

__all__ = [
    "flash_attn_installed", "flash_attn_backend",
    "fa_padding_utils_installed", "fa_attn_bias_supported",
    "flash_attn_varlen_func", "flash_attn_func", "flash_attn_with_kvcache",
    "index_first_axis", "pad_input", "unpad_input",
    "layer_norm_fn", "RMSNorm",
    "get_unpad_data",
]