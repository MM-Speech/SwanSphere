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

from utils.nn.seq_utils import sequence_mask
from utils.nn.fa import (
    flash_attn_installed, flash_attn_backend, fa3_available,
    flash_attn_varlen_func, flash_attn_func, flash_attn_with_kvcache,
    index_first_axis, pad_input, unpad_input,
    layer_norm_fn, RMSNorm,
    get_unpad_data as _get_unpad_data,
    cross_attn_varlen,
    make_additive_mask_from_padding
)

@dataclass
class ModelArgs:
    dim: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_seq_len: int = 16384
    use_sdpa: bool = False
    use_qk_norm: bool = True
    use_causal_attn: bool = True
    crossattn_n_layers: int = 0
    use_dynamic_cross_gate: bool = False


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.




    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
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
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.



    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def merge_sequence(seq_1, seq_1_len, seq_2, seq_2_len, dtype=torch.float16):
    """ Gather batch information for removing pad tokens """
    B, C, device, T_seq_1, T_seq_2, T_merged = seq_1.size(0), seq_1.size(-1), seq_1.device, seq_1.size(1), seq_2.size(1), (seq_1_len+seq_2_len).max()
    x_merged = torch.zeros((B, T_merged, C), device=device, dtype=dtype)
    x_indices = torch.arange(T_merged, device=device)[None, :]

    """ Assign 1st seq to x_merged """
    mask_x_1 = (x_indices < seq_1_len[:, None]) & (x_indices < T_seq_1)
    mask_seq_1 = torch.arange(seq_1.size(1), device=device)[None, :] < seq_1_len[:, None]
    x_merged[mask_x_1] = seq_1[mask_seq_1]

    """ Assign 2nd seq to x_merged """
    mask_for_loss = mask_x_2 = (x_indices >= seq_1_len[:, None]) & (x_indices < (seq_1_len+seq_2_len)[:, None]) & (x_indices - seq_1_len[:, None] < T_seq_2)
    mask_seq_2 = torch.arange(T_seq_2, device=device)[None, :] < seq_2_len[:, None]
    x_merged[mask_x_2] = seq_2[mask_seq_2]
    return x_merged, mask_for_loss, mask_seq_2


class RMSNormFp32(RMSNorm):
    def forward(self, x):
        dtype = x.dtype
        x_fp32 = x.float()
        return super().forward(x_fp32).to(dtype)


class Attention(nn.Module):
    """Multi-head attention module."""

    def __init__(self, args: ModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.args = args
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )

        self.use_causal_attn = args.use_causal_attn

        try:
            use_qk_norm = self.use_qk_norm = args.use_qk_norm
            if use_qk_norm:
                self.q_norm = RMSNormFp32(self.head_dim, eps=args.norm_eps)
                self.k_norm = RMSNormFp32(self.head_dim, eps=args.norm_eps)
        except:
            traceback.print_exc()
            self.use_qk_norm = False
        
        self.use_sdpa = (args.use_sdpa if hasattr(args, 'use_sdpa') else False) or not flash_attn_installed

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
            use_cache: bool = False,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        if self.use_qk_norm:
            xq = self.q_norm(xq.float()).to(xq.dtype)
            xk = self.k_norm(xk.float()).to(xk.dtype)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = xk.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = xv.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)

        if use_cache:
            if (not hasattr(self, 'cache_k')) or (not hasattr(self, 'cache_v')) or (start_pos==0):
                self.cache_k = torch.zeros(1, self.args.max_seq_len, self.n_kv_heads, self.head_dim)
                self.cache_v = torch.zeros(1, self.args.max_seq_len, self.n_kv_heads, self.head_dim)
                self.cache_k = self.cache_k.to(xq)
                self.cache_v = self.cache_v.to(xq)

            query_states = xq.transpose(1, 2)
            attention_mask = mask
            key_states = keys.transpose(1, 2)
            value_states = values.transpose(1, 2)

            output = flash_attn_with_kvcache(query_states, self.cache_k, self.cache_v, key_states, value_states, cache_seqlens=start_pos, causal=self.use_causal_attn)
            output = output.contiguous().view(bsz, seqlen, -1)

        else:
            if self.use_sdpa:
                ### pytorch sdpa
                output = F.scaled_dot_product_attention(xq, keys, values, mask[:, None, None, :], is_causal=self.use_causal_attn)
                output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

                ## pytorch vanilla attn
                # scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
                # if mask is not None:
                #     scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
                # scores = F.softmax(scores.float(), dim=-1).type_as(xq)
                # output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
                # output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
            else:
                ### flash-attn 2
                query_states = xq.transpose(1, 2)
                attention_mask = mask
                key_states = keys.transpose(1, 2)
                value_states = values.transpose(1, 2)
                query_length = query_states.shape[1]
                batch_size = query_states.shape[0]
                query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                    query_states, key_states, value_states, attention_mask, query_length
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
                    causal=self.use_causal_attn,
                )
                if isinstance(attn_output_unpad, (tuple, list)):
                    attn_output_unpad = attn_output_unpad[0]
                output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
                output = output.contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

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
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
        
        
class CrossAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        use_sdpa=False,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        
        self.use_sdpa = use_sdpa or not flash_attn_installed
        
    def reset_parameters(self) -> None:
        self.q.reset_parameters()
        self.k.reset_parameters()
        self.v.reset_parameters()
        self.o.reset_parameters()
        self.norm_q.reset_parameters()
        self.norm_k.reset_parameters()

    def forward(self, x, context, context_lens, query_lens=None, query_mask=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        
        b, n, d = x.size(0), self.num_heads, self.head_dim
        dtype, device = x.dtype, x.device

        # compute query, key, value
        q = self.norm_q(self.q(x).float()).to(dtype).view(b, -1, n, d)
        k = self.norm_k(self.k(context).float()).to(dtype).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        
        if query_mask is None:
            if query_lens is not None:
                query_mask = sequence_mask(query_lens.long(), maxlen=x.shape[1])
            else:
                query_mask = torch.ones_like(q[..., 0, 0], dtype=torch.bool)
        
        if self.use_sdpa:
            K_rep = k.transpose(1, 2)         # [B, H, T_k, D]
            V_rep = v.transpose(1, 2)         # [B, H, T_k, D]
            q_t = q.transpose(1, 2)
            add_mask = make_additive_mask_from_padding(sequence_mask(context_lens), q_len=x.shape[1], device=device, dtype=torch.float32)
            out = F.scaled_dot_product_attention(
                q_t, K_rep, V_rep, attn_mask=add_mask, is_causal=False
            )
            x = out.transpose(1, 2).contiguous().view(b, x.shape[1], -1)
        else:
            x = cross_attn_varlen(q, k, v, query_mask=query_mask, context_mask=sequence_mask(context_lens, maxlen=context.shape[1]))

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            multiple_of: int,
            ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, args, enable_crossattn=False):
        """
        Initialize a TransformerBlock.

        Args:
            args: Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNormFp32(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNormFp32(args.dim, eps=args.norm_eps)
        self.enable_crossattn = enable_crossattn
        if enable_crossattn:
            self.cross_attention = CrossAttention(
                self.dim,
                self.n_heads,
            )
            self.cross_attention_norm = nn.LayerNorm(args.dim, eps=1e-6)
            if args.use_dynamic_cross_gate:
                self.cross_gating_proj = nn.Linear(self.dim, self.dim)
                nn.init.zeros_(self.cross_gating_proj.weight)
                nn.init.constant_(self.cross_gating_proj.bias, -4.0)
            else:
                self.cross_gate = nn.Parameter(torch.zeros(args.dim))

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
            context: Optional[torch.Tensor] = None,
            context_lens: Optional[torch.Tensor] = None,
            use_cache: bool = False,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h_dtype = x.dtype
        h = x.to(torch.float32) + self.attention(
            self.attention_norm(x),
            start_pos, freqs_cis, mask, use_cache=use_cache
        )
        
        # --- cross attention ---
        if self.enable_crossattn and context is not None:
            norm = self.cross_attention_norm(h.float()).to(h_dtype)
            cross_attn_output = self.cross_attention(
                norm,  # query
                context,  # key & value
                context_lens.to(torch.long),
                query_mask=mask
            )
            if self.args.use_dynamic_cross_gate:
                cross_gate = torch.sigmoid(self.cross_gating_proj(norm))
                h = h + cross_gate.to(h_dtype) * cross_attn_output 
            else:
                h = h + self.cross_gate.to(h_dtype) * cross_attn_output  # gated residual
        
        out = h.to(torch.float32) + self.feed_forward(
            self.ffn_norm(h)
        )
        out = out.to(h_dtype)
        return out


class LLaMa(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.n_layers = params.n_layers

        # Decoder
        self.layers = torch.nn.ModuleList()
        for idx in range(params.n_layers):
            if idx < params.crossattn_n_layers:
                self.layers.append(TransformerBlock(params, enable_crossattn=True))
            else:
                self.layers.append(TransformerBlock(params, enable_crossattn=False))

        self.norm = RMSNormFp32(params.dim, eps=params.norm_eps)
        self.out_proj = nn.Linear(params.dim, params.dim, bias=False)

        # Rope embedding
        freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len
        )
        self.register_buffer("freqs_cis", torch.view_as_real(freqs_cis), persistent=False)
        
        # init all weights
        self.apply(self._init_weights)
       
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers)
            )
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers)
            )
    
    def create_custom_forward(self, module):
        def custom_forward(*inputs):
            inputs = module(*inputs)
            return inputs
        return custom_forward

    def forward(self, x, attn_mask, context=None, context_lens=None, 
                start_pos=0, use_cache=False, do_checkpoint=False):
        freqs_cis = torch.view_as_complex(self.freqs_cis)[start_pos: start_pos + x.size(1)]
        for i, layer in enumerate(self.layers):
            if do_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    self.create_custom_forward(layer), x, start_pos, freqs_cis, attn_mask, 
                    context, context_lens, use_cache
                )
            else:
                x = layer(x, start_pos, freqs_cis, attn_mask, context, context_lens, use_cache)
        x = self.norm(x)
        x = self.out_proj(x)
        return x
