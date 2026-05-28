import math
import traceback
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import importlib

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import scaled_dot_product_attention

from utils.nn.fa import (
    flash_attn_installed, flash_attn_backend, fa3_available,
    flash_attn_varlen_func, flash_attn_func, flash_attn_with_kvcache,
    index_first_axis, pad_input, unpad_input,
    layer_norm_fn, RMSNorm,
    get_unpad_data as _get_unpad_data,
)

@dataclass
class ModelArgs:
    dim: int = 1024
    n_heads: int = 16
    n_kv_heads: Optional[int] = None
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_seq_len: int = 16384
    # 使用 SDPA 或 flash-attn
    use_sdpa: bool = False
    # Q/K norm
    use_qk_norm: bool = True
    # 层数：若只给 n_layers，则 encoder/decoder 都用它；否则分别使用 enc_n_layers/dec_n_layers
    n_layers: Optional[int] = 24
    enc_n_layers: Optional[int] = None
    dec_n_layers: Optional[int] = None


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat K/V heads to match Q heads"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def merge_sequence(seq_1, seq_1_len, seq_2, seq_2_len, dtype=torch.float16):
    B, C, device, T_seq_1, T_seq_2, T_merged = seq_1.size(0), seq_1.size(-1), seq_1.device, seq_1.size(1), seq_2.size(1), (seq_1_len+seq_2_len).max()
    x_merged = torch.zeros((B, T_merged, C), device=device, dtype=dtype)
    x_indices = torch.arange(T_merged, device=device)[None, :]
    mask_x_1 = (x_indices < seq_1_len[:, None]) & (x_indices < T_seq_1)
    mask_seq_1 = torch.arange(seq_1.size(1), device=device)[None, :] < seq_1_len[:, None]
    x_merged[mask_x_1] = seq_1[mask_seq_1]
    mask_for_loss = mask_x_2 = (x_indices >= seq_1_len[:, None]) & (x_indices < (seq_1_len+seq_2_len)[:, None]) & (x_indices - seq_1_len[:, None] < T_seq_2)
    mask_seq_2 = torch.arange(T_seq_2, device=device)[None, :] < seq_2_len[:, None]
    x_merged[mask_x_2] = seq_2[mask_seq_2]
    return x_merged, mask_for_loss, mask_seq_2


class RMSNormFp32(RMSNorm):
    def forward(self, x):
        dtype = x.dtype
        x_fp32 = x.float()
        return super().forward(x_fp32).to(dtype)


def make_additive_mask_from_padding(kv_padding_mask: Optional[torch.Tensor], q_len: int, device=None, dtype=None):
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


class SelfAttention(nn.Module):
    """Multi-Head Self-Attention with optional KV cache and FlashAttention/SDPA."""

    def __init__(self, args: ModelArgs, causal: bool):
        super().__init__()
        self.args = args
        self.causal = causal
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        try:
            use_qk_norm = self.use_qk_norm = args.use_qk_norm
            if use_qk_norm:
                self.q_norm = RMSNormFp32(self.head_dim, eps=args.norm_eps)
                self.k_norm = RMSNormFp32(self.head_dim, eps=args.norm_eps)
        except:
            traceback.print_exc()
            self.use_qk_norm = False

        self.use_sdpa = (args.use_sdpa if hasattr(args, 'use_sdpa') else False) or not flash_attn_installed

        # KV cache buffers (for decoder self-attn, causal=True typically)
        self.cache_k = None
        self.cache_v = None
        self.cached_bsz = None
        self.cached_max_len = None

    def _maybe_init_kv_cache(self, bsz: int, max_len: int, dtype: torch.dtype, device: torch.device):
        if (self.cache_k is None) or (self.cached_bsz != bsz) or (self.cached_max_len != max_len):
            self.cache_k = torch.zeros(bsz, max_len, self.n_kv_heads, self.head_dim, device=device, dtype=dtype)
            self.cache_v = torch.zeros(bsz, max_len, self.n_kv_heads, self.head_dim, device=device, dtype=dtype)
            self.cached_bsz = bsz
            self.cached_max_len = max_len

    def forward(
        self,
        x: torch.Tensor,                 # [B, T_q, D]
        start_pos: int,
        freqs_cis: torch.Tensor,         # RoPE for self-attn only
        padding_mask: Optional[torch.Tensor],  # [B, T_kv]; for training w/ padding; ignored in cached flash path
        use_cache: bool = False,
        max_decode_len: Optional[int] = None,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        if self.use_qk_norm:
            xq = self.q_norm(xq.float()).to(xq.dtype)
            xk = self.k_norm(xk.float()).to(xk.dtype)

        # Apply RoPE for self-attention
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # [B, H, T, D]
        xq = xq.transpose(1, 2)
        keys = xk.transpose(1, 2)
        values = xv.transpose(1, 2)

        if use_cache:
            max_len = max_decode_len if max_decode_len is not None else self.args.max_seq_len
            self._maybe_init_kv_cache(bsz, max_len, keys.dtype, keys.device)
            
            query_states = xq.transpose(1, 2)     # [B, T_q, H, D]
            key_states   = keys.transpose(1, 2)   # [B, T_q, H_kv, D]
            value_states = values.transpose(1, 2) # [B, T_q, H_kv, D]

            if flash_attn_installed and (not self.use_sdpa):
                out = flash_attn_with_kvcache(query_states, self.cache_k, self.cache_v,
                                              key_states, value_states,
                                              cache_seqlens=start_pos, causal=self.causal)
                if isinstance(out, tuple):  # fa3 returns (out, ...)
                    out = out[0]
                out = out.contiguous().view(bsz, seqlen, -1)
            else:
                # SDPA fallback: build absolute K/V by writing into cache manually
                # Append current keys/values to cache
                self.cache_k[:, start_pos:start_pos+seqlen] = key_states
                self.cache_v[:, start_pos:start_pos+seqlen] = value_states
                K_full = self.cache_k[:, :start_pos+seqlen].transpose(1, 2)  # [B, H_kv, T_kv, D]
                V_full = self.cache_v[:, :start_pos+seqlen].transpose(1, 2)
                K_full_rep = repeat_kv(K_full.transpose(1, 2), self.n_rep).transpose(1, 2)
                V_full_rep = repeat_kv(V_full.transpose(1, 2), self.n_rep).transpose(1, 2)
                
                out = F.scaled_dot_product_attention(
                    # xq, K_full_rep, V_full_rep, attn_mask=None, is_causal=self.causal
                    xq, K_full_rep, V_full_rep, attn_mask=None, is_causal=False # Fuck this! SDPA only do causal in cached mode in inference
                )
                out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
                
        else:
            # No cache (training / full context)
            if self.use_sdpa:
                # SDPA expects K/V heads match Q; repeat K/V
                K_rep = repeat_kv(keys.transpose(1,2), self.n_rep).transpose(1,2)  # [B, H, T, D]
                V_rep = repeat_kv(values.transpose(1,2), self.n_rep).transpose(1,2)
                add_mask = make_additive_mask_from_padding(padding_mask, q_len=seqlen, device=x.device, dtype=torch.float32)
                out = F.scaled_dot_product_attention(
                    xq, K_rep, V_rep,
                    attn_mask=add_mask, is_causal=self.causal
                )
                out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
            else:
                # FlashAttention varlen path with padding
                query_states = xq.transpose(1, 2)      # [B, T, H, D]
                key_states   = keys.transpose(1, 2)    # [B, T, H_kv, D]
                value_states = values.transpose(1, 2)  # [B, T, H_kv, D]
                query_length = query_states.shape[1]
                batch_size = query_states.shape[0]
                query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                    query_states, key_states, value_states, padding_mask, query_length
                )
                cu_seqlens_q, cu_seqlens_k = cu_seq_lens
                max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    causal=self.causal,
                )
                if isinstance(attn_output_unpad, (tuple, list)):
                    attn_output_unpad = attn_output_unpad[0]
                out = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
                out = out.contiguous().view(bsz, seqlen, -1)

        return self.wo(out)

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.n_kv_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q, *rest = unpad_input(query_layer, attention_mask)
            used_seqlen_q = rest[0] if rest else None

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class CrossAttention(nn.Module):
    """Decoder cross-attention: Q from decoder states, K/V from encoder states.
       支持静态 encoder K/V cache；支持 SDPA 与 FlashAttention(varlen)。"""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        try:
            use_qk_norm = self.use_qk_norm = args.use_qk_norm
            if use_qk_norm:
                self.q_norm = RMSNormFp32(self.head_dim, eps=args.norm_eps)
                self.k_norm = RMSNormFp32(self.head_dim, eps=args.norm_eps)
        except:
            traceback.print_exc()
            self.use_qk_norm = False

        self.use_sdpa = args.use_sdpa if hasattr(args, 'use_sdpa') else False or not flash_attn_installed

        # Static encoder KV cache
        self.enc_k_cache = None  # [B, S_enc, n_kv_heads, D]
        self.enc_v_cache = None
        self.enc_cache_valid = False
        self.enc_cached_shape = None  # (B, S_enc)

    def _maybe_build_enc_kv_cache(self, encoder_hidden_states: torch.Tensor):
        B, S_enc, _ = encoder_hidden_states.shape
        if (self.enc_k_cache is None) or (self.enc_cached_shape != (B, S_enc)):
            # Project once
            k = self.wk(encoder_hidden_states).view(B, S_enc, self.n_local_kv_heads, self.head_dim)
            v = self.wv(encoder_hidden_states).view(B, S_enc, self.n_local_kv_heads, self.head_dim)
            if self.use_qk_norm:
                k = self.k_norm(k.float()).to(k.dtype)
            self.enc_k_cache = k
            self.enc_v_cache = v
            self.enc_cached_shape = (B, S_enc)
            self.enc_cache_valid = True

    def _unpad_q_kv_for_flash(
        self,
        q: torch.Tensor,     # [B, T_q, H_q, D]
        k: torch.Tensor,     # [B, T_k, H_kv, D]
        v: torch.Tensor,     # [B, T_k, H_kv, D]
        q_mask: Optional[torch.Tensor],  # [B, T_q] 1/0
        k_mask: Optional[torch.Tensor],  # [B, T_k] 1/0
    ):
        B, T_q, Hq, D = q.shape
        _, T_k, Hkv, _ = k.shape

        # Unpad K/V using k_mask
        if k_mask is None:
            # Treat all valid
            cu_seqlens_k = torch.arange(0, (B+1)*T_k, T_k, device=k.device, dtype=torch.int32)
            max_seqlen_k = T_k
            k_unp = k.reshape(B*T_k, Hkv, D)
            v_unp = v.reshape(B*T_k, Hkv, D)
            indices_k = torch.arange(B*T_k, device=k.device, dtype=torch.int32)
        else:
            k_unp, indices_k, cu_seqlens_k, max_seqlen_k, *rest = unpad_input(k, k_mask)
            used_seqlen_k = rest[0] if rest else None
            v_unp, *rest = unpad_input(v, k_mask)

        # Unpad Q using q_mask
        if q_mask is None:
            cu_seqlens_q = torch.arange(0, (B+1)*T_q, T_q, device=q.device, dtype=torch.int32)
            max_seqlen_q = T_q
            q_unp = q.reshape(B*T_q, Hq, D)
            indices_q = torch.arange(B*T_q, device=q.device, dtype=torch.int32)
        else:
            q_unp, indices_q, cu_seqlens_q, max_seqlen_q, *rest = unpad_input(q, q_mask)
            used_seqlen_q = rest[0] if rest else None

        return q_unp, k_unp, v_unp, indices_q, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k

    def forward(
        self,
        decoder_x: torch.Tensor,                       # [B, T_q, D]
        encoder_hidden_states: torch.Tensor,           # [B, T_k, D]
        encoder_padding_mask: Optional[torch.Tensor],  # [B, T_k] 1/0
        decoder_padding_mask: Optional[torch.Tensor],  # [B, T_q] 1/0, for training padding
        use_cache: bool = False,                       # if True: reuse static enc K/V projections
    ):
        B, T_q, _ = decoder_x.shape
        B2, T_k, _ = encoder_hidden_states.shape
        assert B == B2, "Batch size mismatch between encoder and decoder."

        # Q from decoder
        q = self.wq(decoder_x).view(B, T_q, self.n_local_heads, self.head_dim)
        if self.use_qk_norm:
            q = self.q_norm(q.float()).to(q.dtype)
        q_t = q.transpose(1, 2)  # [B, H, T_q, D]

        # K/V from encoder, with optional static cache
        if use_cache:
            self._maybe_build_enc_kv_cache(encoder_hidden_states)
            k = self.enc_k_cache  # [B, T_k, H_kv, D]
            v = self.enc_v_cache
        else:
            k = self.wk(encoder_hidden_states).view(B, T_k, self.n_local_kv_heads, self.head_dim)
            v = self.wv(encoder_hidden_states).view(B, T_k, self.n_local_kv_heads, self.head_dim)
            if self.use_qk_norm:
                k = self.k_norm(k.float()).to(k.dtype)

        k_t = k.transpose(1, 2)  # [B, H_kv, T_k, D]
        v_t = v.transpose(1, 2)  # [B, H_kv, T_k, D]

        if self.use_sdpa or (not flash_attn_installed):
            # SDPA路径：需要把 K/V 头数重复到与 Q 一样
            K_rep = repeat_kv(k, self.n_rep)      # [B, T_k, H, D]
            V_rep = repeat_kv(v, self.n_rep)      # [B, T_k, H, D]
            K_rep = K_rep.transpose(1, 2)         # [B, H, T_k, D]
            V_rep = V_rep.transpose(1, 2)         # [B, H, T_k, D]
            add_mask = make_additive_mask_from_padding(encoder_padding_mask, q_len=T_q, device=decoder_x.device, dtype=torch.float32)
            out = F.scaled_dot_product_attention(
                q_t, K_rep, V_rep, attn_mask=add_mask, is_causal=False
            )
            out = out.transpose(1, 2).contiguous().view(B, T_q, -1)
        else:
            # FlashAttention varlen 路径（支持不同长度与 padding）
            # 将 K/V 扩展头数到与 Q 相同，然后进行 varlen 计算
            K_rep = repeat_kv(k, self.n_rep)      # [B, T_k, H, D]
            V_rep = repeat_kv(v, self.n_rep)      # [B, T_k, H, D]
            q_states = q_t.transpose(1, 2)        # [B, T_q, H, D]
            k_states = K_rep                      # [B, T_k, H, D]
            v_states = V_rep

            q_unp, k_unp, v_unp, indices_q, cu_seqlens_q, cu_seqlens_k, max_q, max_k = \
                self._unpad_q_kv_for_flash(q_states, k_states, v_states, decoder_padding_mask, encoder_padding_mask)

            attn_output_unpad = flash_attn_varlen_func(
                q_unp, k_unp, v_unp,
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_q, max_seqlen_k=max_k,
                causal=False
            )
            if isinstance(attn_output_unpad, (tuple, list)):
                attn_output_unpad = attn_output_unpad[0]
            out = pad_input(attn_output_unpad, indices_q, B, T_q)  # [B, T_q, H, D]
            out = out.contiguous().view(B, T_q, -1)

        return self.wo(out)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attn = SelfAttention(args, causal=False)
        self.ffn = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attn_norm = RMSNormFp32(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNormFp32(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,                 # [B, T, D]
        freqs_cis: torch.Tensor,         # encoder用的RoPE
        padding_mask: Optional[torch.Tensor],  # [B, T]
        do_cache: bool = False,          # 对encoder一般不用cache
    ):
        # 按原风格：pre-norm + 残差 FP32 累加
        h_dtype = x.dtype
        h = x.to(torch.float32) + self.attn(
            self.attn_norm(x),
            start_pos=0,
            freqs_cis=freqs_cis,
            padding_mask=padding_mask,
            use_cache=False,  # 编码器不使用KV cache
        )
        out = h.to(torch.float32) + self.ffn(self.ffn_norm(h))
        out = out.to(h_dtype)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = SelfAttention(args, causal=True)
        self.cross_attn = CrossAttention(args)
        self.ffn = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.self_attn_norm = RMSNormFp32(args.dim, eps=args.norm_eps)
        self.cross_attn_norm = RMSNormFp32(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNormFp32(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,                          # [B, T_dec, D]
        start_pos: int,
        freqs_cis: torch.Tensor,                  # decoder用RoPE: indices [start_pos : start_pos+T_dec]
        dec_padding_mask: Optional[torch.Tensor], # [B, T_dec]（训练时的padding）
        enc_hidden_states: torch.Tensor,          # [B, T_enc, D]
        enc_padding_mask: Optional[torch.Tensor], # [B, T_enc]
        use_cache: bool = False,
        max_decode_len: Optional[int] = None,
    ):
        h_dtype = x.dtype

        # 1) Decoder 自注意力（支持KV cache）
        h = x.to(torch.float32) + self.self_attn(
            self.self_attn_norm(x),
            start_pos=start_pos,
            freqs_cis=freqs_cis,
            padding_mask=dec_padding_mask,
            use_cache=use_cache,
            max_decode_len=max_decode_len,
        )

        # 2) Cross-Attention（静态encoder K/V cache）
        h = h.to(torch.float32) + self.cross_attn(
            self.cross_attn_norm(h),
            encoder_hidden_states=enc_hidden_states,
            encoder_padding_mask=enc_padding_mask,
            decoder_padding_mask=dec_padding_mask,
            use_cache=use_cache,  # True 时：每层仅第一次建立 enc K/V cache
        )

        # 3) FFN
        out = h.to(torch.float32) + self.ffn(self.ffn_norm(h))
        out = out.to(h_dtype)
        return out


class Encoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        n_layers = args.enc_n_layers if (args.enc_n_layers is not None) else (args.n_layers or 24)
        self.layers = nn.ModuleList([EncoderBlock(args) for _ in range(n_layers)])
        self.norm = RMSNormFp32(args.dim, eps=args.norm_eps)
        self.out_proj = nn.Linear(args.dim, args.dim, bias=False)

    def create_custom_forward(self, module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    def forward(
        self,
        x: torch.Tensor,                        # [B, T_enc, D]
        freqs_cis: torch.Tensor,                # [T_enc, head_dim] (complex)
        padding_mask: Optional[torch.Tensor],   # [B, T_enc]
        do_checkpoint: bool = False,
    ):
        for layer in self.layers:
            if do_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    self.create_custom_forward(layer),
                    x, freqs_cis, padding_mask, False
                )
            else:
                x = layer(x, freqs_cis, padding_mask, do_cache=False)
        x = self.norm(x)
        x = self.out_proj(x)
        return x


class Decoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        n_layers = args.dec_n_layers if (args.dec_n_layers is not None) else (args.n_layers or 24)
        self.layers = nn.ModuleList([DecoderBlock(args) for _ in range(n_layers)])
        self.norm = RMSNormFp32(args.dim, eps=args.norm_eps)
        self.out_proj = nn.Linear(args.dim, args.dim, bias=False)

    def create_custom_forward(self, module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    def reset_kv_cache(self):
        # 清空所有层的自注意力与交叉注意力缓存
        for m in self.layers:
            # 自注意力缓存
            m.self_attn.cache_k = None
            m.self_attn.cache_v = None
            m.self_attn.cached_bsz = None
            m.self_attn.cached_max_len = None
            # 交叉注意力静态缓存
            m.cross_attn.enc_k_cache = None
            m.cross_attn.enc_v_cache = None
            m.cross_attn.enc_cache_valid = False
            m.cross_attn.enc_cached_shape = None

    def forward(
        self,
        x: torch.Tensor,                            # [B, T_dec, D]
        start_pos: int,
        freqs_cis: torch.Tensor,                    # [T_dec, head_dim] (complex)
        dec_padding_mask: Optional[torch.Tensor],   # [B, T_dec]
        enc_hidden_states: torch.Tensor,            # [B, T_enc, D]
        enc_padding_mask: Optional[torch.Tensor],   # [B, T_enc]
        use_cache: bool = False,
        do_checkpoint: bool = False,
        max_decode_len: Optional[int] = None,
    ):
        for layer in self.layers:
            if do_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    self.create_custom_forward(layer),
                    x, start_pos, freqs_cis, dec_padding_mask,
                    enc_hidden_states, enc_padding_mask, use_cache, max_decode_len
                )
            else:
                x = layer(
                    x, start_pos, freqs_cis, dec_padding_mask,
                    enc_hidden_states, enc_padding_mask,
                    use_cache=use_cache, max_decode_len=max_decode_len
                )
        x = self.norm(x)
        x = self.out_proj(x)
        return x


class Seq2SeqLLaMA(nn.Module):
    """
    Encoder-Decoder（Seq2Seq）版本：
    - Encoder/Decoder 都是 LLaMA 风格块（RMSNorm + SwiGLU FFN + 可选 QK norm + RoPE）。
    - Decoder 自注意力支持 KV cache（flash_attn_with_kvcache 优先）。
    - Cross-Attention 缓存 encoder K/V（按层静态缓存）。
    - 训练时支持 FlashAttention varlen（要求 self-attn 下 n_kv_heads==n_heads），否则回退 SDPA。
    """

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

        # RoPE embedding 预计算（与原实现一致）
        freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads, params.max_seq_len
        )
        # buffer 存为 real，前向时再 view_as_complex 使用
        self.register_buffer("freqs_cis", torch.view_as_real(freqs_cis), persistent=False)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            n_layers_total = (self.params.enc_n_layers if self.params.enc_n_layers is not None else (self.params.n_layers or 24)) + \
                             (self.params.dec_n_layers if self.params.dec_n_layers is not None else (self.params.n_layers or 24))
            n_layers_total = max(1, n_layers_total)
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layers_total)
            )
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2)
            )

    def reset_kv_cache(self):
        self.decoder.reset_kv_cache()

    def forward(
        self,
        encoder_x: torch.Tensor,                     # [B, T_enc, D]
        encoder_padding_mask: Optional[torch.Tensor],# [B, T_enc] 1/0 (1为有效)
        decoder_x: torch.Tensor,                     # [B, T_dec, D]
        decoder_padding_mask: Optional[torch.Tensor],# [B, T_dec] 1/0
        start_pos: int = 0,
        use_cache: bool = False,
        do_checkpoint: bool = False,
        max_decode_len: Optional[int] = None,
    ):
        enc_out = self.encode(encoder_x, encoder_padding_mask, do_checkpoint)

        dec_out = self.decode(
            decoder_x, 
            decoder_padding_mask, 
            enc_out, 
            encoder_padding_mask, 
            start_pos,
            use_cache,
            do_checkpoint,
            max_decode_len
        )

        return dec_out

    def encode(
        self,
        encoder_x: torch.Tensor,                     # [B, T_enc, D]
        encoder_padding_mask: Optional[torch.Tensor],# [B, T_enc] 1/0 (1为有效)
        do_checkpoint: bool = False
    ):
        # 1) Encoder 前向
        T_enc = encoder_x.size(1)
        enc_freqs_cis = torch.view_as_complex(self.freqs_cis.float())[:T_enc]
        enc_out = self.encoder(
            encoder_x, enc_freqs_cis, encoder_padding_mask, do_checkpoint=do_checkpoint
        )
        return enc_out


    def decode(
        self,
        decoder_x: torch.Tensor,                     # [B, T_dec, D]
        decoder_padding_mask: Optional[torch.Tensor],# [B, T_dec] 1/0
        enc_out: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor],# [B, T_enc] 1/0 (1为有效)
        start_pos: int = 0,
        use_cache: bool = False,
        do_checkpoint: bool = False,
        max_decode_len: Optional[int] = None,
    ):
        # 2) Decoder 前向
        T_dec = decoder_x.size(1)
        dec_freqs_cis = torch.view_as_complex(self.freqs_cis.float())[start_pos : start_pos + T_dec]
        dec_out = self.decoder(
            decoder_x, start_pos, dec_freqs_cis, decoder_padding_mask,
            enc_out, encoder_padding_mask,
            use_cache=use_cache, do_checkpoint=do_checkpoint, max_decode_len=max_decode_len
        )

        return dec_out

