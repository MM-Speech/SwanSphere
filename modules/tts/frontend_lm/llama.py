import math
import traceback
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from torch.nn.functional import scaled_dot_product_attention
from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
try:
    from transformers.modeling_flash_attention_utils import _get_unpad_data
except:
    from transformers.models.llama.modeling_llama import _get_unpad_data
try:
    from flash_attn.ops.triton.layer_norm import layer_norm_fn, RMSNorm
except ImportError:
    layer_norm_fn, RMSNorm = None, None


@dataclass
class ModelArgs:
    dim: int = 1536
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_seq_len: int = 16384


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
        self.encoder_n_kv_heads = args.encoder_n_heads if args.encoder_n_kv_heads is None else args.encoder_n_kv_heads
        model_parallel_size = 1
        self.n_local_heads = args.encoder_n_heads // model_parallel_size
        self.n_local_kv_heads = self.encoder_n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.encoder_dim // args.encoder_n_heads

        self.wq = nn.Linear(
            args.encoder_dim,
            args.encoder_n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            args.encoder_dim,
            self.encoder_n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            args.encoder_dim,
            self.encoder_n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            args.encoder_n_heads * self.head_dim,
            args.encoder_dim,
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
        
        self.use_normal_attn = args.use_normal_attn if hasattr(args, 'use_normal_attn') else False

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
                self.cache_k = torch.zeros(
                    1, 4096, self.encoder_n_kv_heads, self.head_dim
                ).cuda()
                self.cache_v = torch.zeros(
                    1, 4096, self.encoder_n_kv_heads, self.head_dim
                ).cuda()
                self.cache_k = self.cache_k.to(xq)
                self.cache_v = self.cache_v.to(xq)

            query_states = xq.transpose(1, 2)
            attention_mask = mask
            key_states = keys.transpose(1, 2)
            value_states = values.transpose(1, 2)

            output = flash_attn_with_kvcache(query_states, self.cache_k, self.cache_v, key_states, value_states, cache_seqlens=start_pos, causal=self.use_causal_attn)
            output = output.contiguous().view(bsz, seqlen, -1)

        else:
            if self.use_normal_attn:
                ### pytorch flash_attn 2
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
                    dropout_p=0,
                    causal=self.use_causal_attn,
                )
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
                query_layer.reshape(batch_size * kv_seq_len, self.encoder_n_kv_heads, head_dim), indices_k
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
    def __init__(self, args):
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
        self.encoder_n_heads = args.encoder_n_heads
        self.encoder_dim = args.encoder_dim
        self.head_dim = args.encoder_dim // args.encoder_n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.encoder_dim,
            hidden_dim=4 * args.encoder_dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNormFp32(args.encoder_dim, eps=args.norm_eps)
        self.ffn_norm = RMSNormFp32(args.encoder_dim, eps=args.norm_eps)

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
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
        out = h.to(torch.float32) + self.feed_forward(
            self.ffn_norm(h)
        )
        out = out.to(h_dtype)
        return out


class LLaMa(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.encoder_n_layers = params.encoder_n_layers

        # Decoder
        self.layers = torch.nn.ModuleList()
        for _ in range(params.encoder_n_layers):
            self.layers.append(TransformerBlock(params))

        self.norm = RMSNormFp32(params.encoder_dim, eps=params.norm_eps)
        self.out_proj = nn.Linear(params.encoder_dim, params.encoder_dim, bias=False)

        # Rope embedding
        freqs_cis = precompute_freqs_cis(
            self.params.encoder_dim // self.params.encoder_n_heads, self.params.max_seq_len
        )
        self.register_buffer("freqs_cis", torch.view_as_real(freqs_cis), persistent=False)
        
        # init all weights
        self.apply(self._init_weights)
       
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.encoder_n_layers)
            )
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.encoder_n_layers)
            )
    
    def create_custom_forward(self, module):
        def custom_forward(*inputs):
            inputs = module(*inputs)
            return inputs
        return custom_forward

    def forward(self, x, attn_mask, start_pos=0, use_cache=False, do_checkpoint=False):
        freqs_cis = torch.view_as_complex(self.freqs_cis)[start_pos: start_pos + x.size(1)]
        for i, layer in enumerate(self.layers):
            if do_checkpoint:
                x = torch.utils.checkpoint.checkpoint(self.create_custom_forward(layer), x, start_pos, freqs_cis, attn_mask, use_cache)
            else:
                x = layer(x, start_pos, freqs_cis, attn_mask, use_cache=use_cache)
        x = self.norm(x)
        x = self.out_proj(x)
        return x