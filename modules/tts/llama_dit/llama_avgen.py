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
    from utils.nn.fa_fallbacks import layer_norm_fn, RMSNorm
from modules.tts.scriptspeech.attention import flash_attention
from modules.tts.f5_dit.f5_modules import AdaLayerNormZero, AdaLayerNormZero_Final


@dataclass
class ModelArgs:
    encoder_dim: int = 1024
    encoder_n_layers: int = 24
    encoder_n_heads: int = 16
    encoder_n_kv_heads: int = None
    mlp_extend: float = None
    max_seq_len: int = 16384
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = 2
    use_causal_attn: bool = False


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
        )
        self.wk = nn.Linear(
            args.encoder_dim,
            self.encoder_n_kv_heads * self.head_dim,
        )
        self.wv = nn.Linear(
            args.encoder_dim,
            self.encoder_n_kv_heads * self.head_dim,
        )
        self.wo = nn.Linear(
            args.encoder_n_heads * self.head_dim,
            args.encoder_dim,
        )

        self.use_causal_attn = args.use_causal_attn
        self.use_normal_attn = args.use_normal_attn if hasattr(args, 'use_normal_attn') else False
        
        self.norm_q = RMSNorm(args.encoder_dim, eps=1e-6) if args.use_qk_norm else nn.Identity()
        self.norm_k = RMSNorm(args.encoder_dim, eps=1e-6) if args.use_qk_norm else nn.Identity()

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
        h_dtype = x.dtype
        xq, xk, xv = self.norm_q(self.wq(x).float()).to(h_dtype), self.norm_k(self.wk(x).float()).to(h_dtype), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = xk.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = xv.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        xq, keys = xq.to(xv.dtype), keys.to(xv.dtype)
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

class CrossAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, x_lens, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        # compute attention
        # x = flash_attention(q, k, v, q_lens=x_lens, k_lens=context_lens)
        x = flash_attention(q, k, v, k_lens=context_lens)

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
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim, hidden_dim
        )
        self.w2 = nn.Linear(
            hidden_dim, dim
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))



class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
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
        self.encoder_n_heads = args.encoder_n_heads
        self.encoder_dim = args.encoder_dim
        self.head_dim = args.encoder_dim // args.encoder_n_heads
        self.attention = Attention(args)
        self.cross_attention = CrossAttention(
            self.encoder_dim,
            self.encoder_n_heads,
        )
        self.feed_forward = FeedForward(
            dim=args.encoder_dim,
            hidden_dim=args.encoder_dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm = AdaLayerNormZero(args.encoder_dim)
        self.cross_attention_norm = nn.LayerNorm(args.encoder_dim, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(args.encoder_dim, elementwise_affine=False, eps=1e-6)
        self.cross_gate = nn.Parameter(torch.zeros(args.encoder_dim))

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
            context: Optional[torch.Tensor] = None,
            context_lens: Optional[torch.Tensor] = None,
            caption_mark: Optional[torch.Tensor] = None,
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
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attention_norm(x, emb=t)

        # attention
        attn_output = self.attention(norm, start_pos, freqs_cis, mask=mask, use_cache=use_cache)
        # process attention output for input x
        h = x + gate_msa.unsqueeze(1) * attn_output
        # --- cross attention ---
        if context is not None:
            norm = self.cross_attention_norm(h)
            cross_attn_output = self.cross_attention(
                norm,  # query
                mask.sum(1).to(torch.long),
                context,  # key & value
                context_lens.to(torch.long)
            )
            if caption_mark is not None:
                cross_attn_output = cross_attn_output * caption_mark[:, None, None]
            h = h + self.cross_gate * cross_attn_output  # gated residual

        norm = self.ffn_norm(h) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.feed_forward(norm)
        out = h + gate_mlp.unsqueeze(1) * ff_output

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

        self.norm = AdaLayerNormZero_Final(params.encoder_dim)
        self.out_proj = nn.Linear(params.encoder_dim, params.encoder_dim)

        # 仅当启用 caption 池化注入时，创建 pool 后的投影层（与第二版保持一致：无 norm/门控）
        self._use_cap_mod = bool(getattr(self.params, "use_caption_pool_in_adaln", False))
        if self._use_cap_mod:
            # 维度与时间步嵌入一致：encoder_dim -> encoder_dim
            self.cap_embedder = nn.Linear(params.encoder_dim, params.encoder_dim)
            self.cap_gate = nn.Linear(params.encoder_dim, params.encoder_dim)
            nn.init.zeros_(self.cap_embedder.weight)
            nn.init.zeros_(self.cap_embedder.bias)
            nn.init.zeros_(self.cap_gate.weight)
            nn.init.zeros_(self.cap_gate.bias)

        # Rope embedding
        freqs_cis = precompute_freqs_cis(
            self.params.encoder_dim // self.params.encoder_n_heads, self.params.max_seq_len
        )
        self.register_buffer("freqs_cis", torch.view_as_real(freqs_cis), persistent=False)

    def forward(self, x, t, attn_mask, context=None, context_lens=None, caption_mark=None, start_pos=0, use_cache=False, do_checkpoint=False):
        freqs_cis = torch.view_as_complex(self.freqs_cis.float())[start_pos: start_pos + x.size(1)]
        
        # 可选：caption 全局池化 + 线性投影 + 融合到 AdaLN 调制
        if self._use_cap_mod and (context is not None) and (context_lens is not None):
            # context: [B, Lc, C] ; context_lens: [B]
            B, Lc = context.size(0), context.size(1)
            device = context.device

            # 有效 token mask（True=有效）
            lengths = context_lens.to(device=device).long().view(-1)               # [B]
            idxs = torch.arange(Lc, device=device).unsqueeze(0).expand(B, Lc)      # [B, Lc]
            cap_mask = (idxs < lengths.unsqueeze(1))                               # [B, Lc], bool

            # mean-pool（按有效长度），避免除 0
            cap_mask_f = cap_mask.to(context.dtype).unsqueeze(-1)                  # [B, Lc, 1]
            cap_sum = (context * cap_mask_f).sum(dim=1)                            # [B, C]
            denom = cap_mask_f.sum(dim=1).clamp(min=1.0)                           # [B, 1]
            cap_pool = cap_sum / denom                                             # [B, C]

            # pool 后线性投影，再与 t 相加（无 norm/门控，和第二版一致）
            # 注意：只有在 __init__ 创建了 cap_embedder 才会调用
            cap_emb = self.cap_embedder(cap_pool)   
            cap_gate = torch.tanh(self.cap_gate(t + cap_emb.to(t.dtype)))
            t = t + cap_emb.to(t.dtype) * cap_gate

        for i, layer in enumerate(self.layers):
            if torch.is_grad_enabled() and do_checkpoint:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs = {"use_reentrant": False}

                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    x, t, start_pos, freqs_cis, attn_mask, context, context_lens, caption_mark, use_cache, **ckpt_kwargs,
                )

            else:
                x = layer(x, t, start_pos, freqs_cis, attn_mask, context=context, context_lens=context_lens, caption_mark=caption_mark, use_cache=use_cache)

        x = self.norm(x, t)
        x = self.out_proj(x)
        return x