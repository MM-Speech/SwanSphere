# --------------------------------------------------------
# WavLM: Large-Scale Self-Supervised  Pre-training  for Full Stack Speech Processing (https://arxiv.org/abs/2110.13900.pdf)
# Github source: https://github.com/microsoft/unilm/tree/master/wavlm
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------

import math
import warnings
from typing import Dict, Optional, Tuple
import torch
from torch import Tensor, nn
from torch.nn import Parameter
import torch.nn.functional as F

from utils.nn.fa import (
    flash_attn_installed, flash_attn_backend,
    fa_padding_utils_installed, fa_attn_bias_supported,
    flash_attn_varlen_func, flash_attn_func, flash_attn_with_kvcache,
    index_first_axis, pad_input, unpad_input,
    layer_norm_fn, RMSNorm,
    get_unpad_data as _get_unpad_data,
)
    

class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)


class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class SamePad(nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class Swish(nn.Module):
    """Swish function
    """

    def __init__(self):
        """Construct an MultiHeadedAttention object."""
        super(Swish, self).__init__()
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        return x * self.act(x)


class GLU_Linear(nn.Module):
    def __init__(self, input_dim, output_dim, glu_type="sigmoid", bias_in_glu=True):
        super(GLU_Linear, self).__init__()

        self.glu_type = glu_type
        self.output_dim = output_dim

        if glu_type == "sigmoid":
            self.glu_act = torch.nn.Sigmoid()
        elif glu_type == "swish":
            self.glu_act = Swish()
        elif glu_type == "relu":
            self.glu_act = torch.nn.ReLU()
        elif glu_type == "gelu":
            self.glu_act = torch.nn.GELU()

        if bias_in_glu:
            self.linear = nn.Linear(input_dim, output_dim * 2, True)
        else:
            self.linear = nn.Linear(input_dim, output_dim * 2, False)

    def forward(self, x):
        # to be consistent with GLU_Linear, we assume the input always has the #channel (#dim) in the last dimension of the tensor, so need to switch the dimension first for 1D-Conv case
        x = self.linear(x)

        if self.glu_type == "bilinear":
            x = (x[:, :, 0:self.output_dim] * x[:, :, self.output_dim:self.output_dim * 2])
        else:
            x = (x[:, :, 0:self.output_dim] * self.glu_act(x[:, :, self.output_dim:self.output_dim * 2]))

        return x


def gelu_accurate(x):
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return (
        0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
    )


def gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x.float()).type_as(x)


def get_activation_fn(activation: str):
    """Returns the activation function corresponding to `activation`"""

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_fast":
        warnings.warn(
            "--activation-fn=gelu_fast has been renamed to gelu_accurate"
        )
        return gelu_accurate
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "glu":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(
            data.cpu().normal_(mean=0.0, std=0.02).to(data.device)
        )

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


def quant_noise(module, p, block_size):
    """
    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """

    # if no quantization noise, don't register hook
    if p <= 0:
        return module

    # supported modules
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))

    # test whether module.weight has the right sizes wrt block_size
    is_conv = module.weight.ndim == 4

    # 2D matrix
    if not is_conv:
        assert (
            module.weight.size(1) % block_size == 0
        ), "Input features must be a multiple of block sizes"

    # 4D matrix
    else:
        # 1x1 convolutions
        if module.kernel_size == (1, 1):
            assert (
                module.in_channels % block_size == 0
            ), "Input channels must be a multiple of block sizes"
        # regular convolutions
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            assert k % block_size == 0, "Kernel size must be a multiple of block size"

    def _forward_pre_hook(mod, input):
        # no noise for evaluation
        if mod.training:
            if not is_conv:
                # gather weight and sizes
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)

                # split weight matrix into blocks and randomly drop selected blocks
                mask = torch.zeros(
                    in_features // block_size * out_features, device=weight.device
                )
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)

            else:
                # gather weight and sizes
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # split weight matrix into blocks and randomly drop selected blocks
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(
                        int(in_channels // block_size * out_channels),
                        device=weight.device,
                    )
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(
                        weight.size(0), weight.size(1), device=weight.device
                    )
                    mask.bernoulli_(p)
                    mask = (
                        mask.unsqueeze(2)
                        .unsqueeze(3)
                        .repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
                    )

            # scale weights and apply mask
            mask = mask.to(
                torch.bool
            )  # x.bool() is not currently supported in TorchScript
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
            self,
            embed_dim,
            num_heads,
            kdim=None,
            vdim=None,
            dropout=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            self_attention=False,
            encoder_decoder_attention=False,
            q_noise=0.0,
            qn_block_size=8,
            has_relative_attention_bias=False,
            num_buckets=32,
            max_distance=128,
            gru_rel_pos=False,
            rescale_init=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)

        self.has_relative_attention_bias = has_relative_attention_bias
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

        self.head_dim = embed_dim // num_heads
        self.q_head_dim = self.head_dim
        self.k_head_dim = self.head_dim
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        k_bias = True
        if rescale_init:
            k_bias = False

        k_embed_dim = embed_dim
        q_embed_dim = embed_dim

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, k_embed_dim, bias=k_bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, q_embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.gru_rel_pos = gru_rel_pos
        if self.gru_rel_pos:
            self.grep_linear = nn.Linear(self.q_head_dim, 8)
            self.grep_a = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.reset_parameters()
        
        self.flash_attn_available = flash_attn_installed
        self.fa_padding_utils_available = fa_padding_utils_installed
        self.fa_attn_bias_supported = fa_attn_bias_supported

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
        if self.has_relative_attention_bias:
            nn.init.xavier_normal_(self.relative_attention_bias.weight)

    def _can_use_flash_attn(self,
                            query: Tensor,
                            key: Optional[Tensor],
                            value: Optional[Tensor],
                            key_padding_mask: Optional[Tensor],
                            attn_mask: Optional[Tensor],
                            position_bias: Optional[Tensor],
                            incremental_state: Optional[Dict]) -> bool:
        if not self.flash_attn_available:
            return False
        if query.device.type != "cuda":
            return False
        if query.dtype not in (torch.float16, torch.bfloat16):
            return False
        if incremental_state is not None:
            return False  # 如需增量可改造为 flash_attn_with_kvcache
        if self.bias_k is not None or self.bias_v is not None:
            return False
        if self.add_zero_attn:
            return False
        if self.q_head_dim != self.head_dim:
            return False
        if attn_mask is not None:
            # 通用 attn_mask（非因果）暂不接入，回退
            return False
        
        # 有相对位置/GRU gating 时，必须要求 FA 支持 attn_bias
        need_bias = (position_bias is not None) or self.has_relative_attention_bias or self.gru_rel_pos
        if need_bias and not self.fa_attn_bias_supported:
            return False
        
        return True

    def _relative_positions_bucket(self, relative_positions, bidirectional=True):
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        relative_buckets = 0

        if bidirectional:
            num_buckets = num_buckets // 2
            relative_buckets += (relative_positions > 0).to(torch.long) * num_buckets
            relative_positions = torch.abs(relative_positions)
        else:
            relative_positions = -torch.min(relative_positions, torch.zeros_like(relative_positions))

        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact

        relative_postion_if_large = max_exact + (
                torch.log(relative_positions.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_positions, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_positions_bucket(
            relative_position,
            bidirectional=True
        )
        relative_position_bucket = relative_position_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1])
        return values
    
    def _build_attn_bias(self,
                         q_BTHD: torch.Tensor,  # [B, T, H, Dh]，已缩放后的 q
                         bsz: int, tgt_len: int, src_len: int,
                         position_bias: Optional[torch.Tensor],
                         key_padding_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        返回 [B, H, T, S] 的 bias，融合三部分：
          1) 绝对/相对位置偏置（可来自 self.compute_bias 或外部传入的 position_bias）
          2) gru_rel_pos 的门控缩放（对第 1) 的逐 query 位缩放）
          3) key_padding_mask 转换成 -inf（或大负）对无效 key 的屏蔽
        """
        device, dtype = q_BTHD.device, q_BTHD.dtype
        bias_BHTS = None
        # 1) base 相对位置偏置
        base = None
        if position_bias is not None:
            # 可能是 [B*H, T, S] / [B,H,T,S] / [H,T,S]
            if position_bias.dim() == 3 and position_bias.size(0) == bsz * self.num_heads:
                base = position_bias.view(bsz, self.num_heads, tgt_len, src_len)
            elif position_bias.dim() == 4:
                base = position_bias
            elif position_bias.dim() == 3 and position_bias.size(0) == self.num_heads:
                base = position_bias.unsqueeze(0).expand(bsz, -1, -1, -1)
            else:
                raise RuntimeError("Unexpected position_bias shape: {}".format(position_bias.shape))
        elif self.has_relative_attention_bias:
            # 生成每层共享的相对偏置 [H,T,S] -> [B,H,T,S]
            base_h = self.compute_bias(tgt_len, src_len).to(device=device, dtype=dtype)
            base = base_h.unsqueeze(0).expand(bsz, -1, -1, -1)
        if base is not None:
            bias_BHTS = base.to(dtype=dtype, device=device)
        # 2) gru_rel_pos 门控（gate 对每个 query 位置缩放位置偏置）
        if self.gru_rel_pos:
            # q_BTHD: [B,T,H,D] -> [B,H,T,D]
            query_layer = q_BTHD.permute(0, 2, 1, 3).contiguous()
            _B, _H, _L, _D = query_layer.shape
            gate_a, gate_b = torch.sigmoid(
                self.grep_linear(query_layer).view(_B, _H, _L, 2, 4).sum(-1, keepdim=False)
            ).chunk(2, dim=-1)  # [B,H,T,1] each
            grep_a = self.grep_a.to(device=device, dtype=dtype)
            gate_a_1 = gate_a * (gate_b * grep_a - 1.0) + 2.0  # [B,H,T,1]
            if bias_BHTS is None:
                # 没有 base，则相当于只有门控时不改变 bias（门控需要乘以某个 bias 基准）
                # 按原论文/实现，gru_rel_pos 是对 position_bias 的缩放；如果没有 base，这里不生效
                pass
            else:
                bias_BHTS = gate_a_1 * bias_BHTS  # 广播到 S
        # 3) padding -> -inf 屏蔽无效 key
        if key_padding_mask is not None:
            # key_padding_mask: [B,S]，1 表示 pad
            valid = (~key_padding_mask.to(torch.bool)).to(dtype=dtype)
            # 选择合适的“大负数”，避免半精度的 -inf 造成 NaN
            neg_large = -1e4 if dtype == torch.float16 else -1e9
            pad_bias = (1.0 - valid) * neg_large  # [B,S]
            pad_bias = pad_bias.view(bsz, 1, 1, src_len)  # [B,1,1,S]
            bias_BHTS = pad_bias if bias_BHTS is None else (bias_BHTS + pad_bias)
        return bias_BHTS  # 可能为 None

    def forward(
            self,
            query,
            key: Optional[Tensor],
            value: Optional[Tensor],
            key_padding_mask: Optional[Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            need_weights: bool = True,
            static_kv: bool = False,
            attn_mask: Optional[Tensor] = None,
            before_softmax: bool = False,
            need_head_weights: bool = False,
            position_bias: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        if self.has_relative_attention_bias and position_bias is None:
            position_bias = self.compute_bias(tgt_len, src_len)
            position_bias = position_bias.unsqueeze(0).repeat(bsz, 1, 1, 1).view(bsz * self.num_heads, tgt_len, src_len)

        if (
                not is_tpu  # don't use PyTorch version on TPUs
                and incremental_state is None
                and not static_kv
                # A workaround for quantization to work. Otherwise JIT compilation
                # treats bias in linear module as method.
                and not torch.jit.is_scripting()
                and self.q_head_dim == self.head_dim
        ):
            assert key is not None and value is not None
            
            # 若可用 FA，则优先走 FA
            if self._can_use_flash_attn(query, key, value, key_padding_mask, attn_mask, position_bias, incremental_state):
                # 线性 + 缩放
                q = self.q_proj(query)  # [T,B,C]
                k = self.k_proj(key)
                v = self.v_proj(value)
                q *= self.scaling
                
                tgt_len, bsz, embed_dim = query.size()
                # 重排到 [B,T,H,Dh]
                q_BTHD = q.transpose(0, 1).contiguous().view(bsz, tgt_len, self.num_heads, self.q_head_dim)
                k_BTHD = k.transpose(0, 1).contiguous().view(bsz, k.size(0), self.num_heads, self.k_head_dim)
                v_BTHD = v.transpose(0, 1).contiguous().view(bsz, v.size(0), self.num_heads, self.head_dim)
                src_len = k_BTHD.size(1)
                
                dropout_p = self.dropout_module.p if self.training else 0.0
                
                # 是否需要 bias（相对位置/门控/padding）
                need_bias = (position_bias is not None) or self.has_relative_attention_bias or self.gru_rel_pos or (key_padding_mask is not None)
                attn_bias = None
                if need_bias and self.fa_attn_bias_supported:
                    attn_bias = self._build_attn_bias(q_BTHD, bsz, tgt_len, src_len, position_bias, key_padding_mask)
                    
                if attn_bias is not None:
                    # 使用全长 FA + attn_bias（可同时涵盖相对位置、gru 门控与 padding）
                    attn_out = flash_attn_func(
                        q_BTHD, k_BTHD, v_BTHD,
                        dropout_p=dropout_p,
                        softmax_scale=self.scaling,
                        causal=False,
                        attn_bias=attn_bias
                    )  # [B,T,H,Dh]
                    attn_out = attn_out.contiguous().view(bsz, tgt_len, embed_dim).transpose(0, 1)
                else:
                    # 不需要 bias 时：优先 varlen（有 padding）或普通 FA（无 padding）
                    if key_padding_mask is not None and self.fa_padding_utils_available:
                        # varlen self-attn：unpad -> varlen kernel -> pad 回来
                        valid_mask = ~key_padding_mask.to(torch.bool)  # [B,T]
                        q_unpad, indices, cu_q, max_q = unpad_input(q_BTHD, valid_mask)
                        k_unpad, _, cu_k, max_k = unpad_input(k_BTHD, valid_mask)
                        v_unpad, _, _, _ = unpad_input(v_BTHD, valid_mask)
                        attn_unpad = flash_attn_varlen_func(
                            q_unpad, k_unpad, v_unpad,
                            cu_q, cu_k,  # 自注意：同一个 cu
                            max_q, max_k,
                            dropout_p=dropout_p,
                            softmax_scale=self.scaling,
                            causal=False
                        )  # [Nnz, H, Dh]
                        attn_padded = pad_input(attn_unpad, indices, bsz, tgt_len)  # [B,T,H,Dh]
                        attn_out = attn_padded.contiguous().view(bsz, tgt_len, embed_dim).transpose(0, 1)
                    else:
                        # 无 padding：普通 FA
                        attn_out = flash_attn_func(
                            q_BTHD, k_BTHD, v_BTHD,
                            dropout_p=dropout_p,
                            softmax_scale=self.scaling,
                            causal=False
                        )  # [B,T,H,Dh]
                        attn_out = attn_out.contiguous().view(bsz, tgt_len, embed_dim).transpose(0, 1)
                        
                x = self.out_proj(attn_out)
                attn = None  # FA 不返回全量注意力图；need_weights=True 时应回退
                return x, attn, position_bias
            
            # 否则回退到原来的 PyTorch fused MHA（保持相对位置与 GRU 门控的行为）
            attn_mask_rel_pos = None
            if position_bias is not None:
                attn_mask_rel_pos = position_bias
                if self.gru_rel_pos:
                    query_layer = query.transpose(0, 1)
                    new_x_shape = query_layer.size()[:-1] + (self.num_heads, -1)
                    query_layer = query_layer.view(*new_x_shape)
                    query_layer = query_layer.permute(0, 2, 1, 3)
                    _B, _H, _L, __ = query_layer.size()

                    gate_a, gate_b = torch.sigmoid(self.grep_linear(query_layer).view(
                        _B, _H, _L, 2, 4).sum(-1, keepdim=False)).chunk(2, dim=-1)
                    gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
                    attn_mask_rel_pos = gate_a_1.view(bsz * self.num_heads, -1, 1) * position_bias

                attn_mask_rel_pos = attn_mask_rel_pos.view((-1, tgt_len, tgt_len))
            x, attn = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                torch.empty([0]),
                torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout_module.p,
                self.out_proj.weight,
                self.out_proj.bias,
                self.training,
                key_padding_mask,
                need_weights,
                attn_mask_rel_pos,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
            )
            return x, attn, position_bias

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = (
            q.contiguous()
                .view(tgt_len, bsz * self.num_heads, self.q_head_dim)
                .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                    .view(-1, bsz * self.num_heads, self.k_head_dim)
                    .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                    .view(-1, bsz * self.num_heads, self.head_dim)
                    .transpose(0, 1)
            )

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
                src_len = k.size(1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v, position_bias

        if position_bias is not None:
            if self.gru_rel_pos == 1:
                query_layer = q.view(bsz, self.num_heads, tgt_len, self.q_head_dim)
                _B, _H, _L, __ = query_layer.size()
                gate_a, gate_b = torch.sigmoid(self.grep_linear(query_layer).view(
                    _B, _H, _L, 2, 4).sum(-1, keepdim=False)).chunk(2, dim=-1)
                gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
                position_bias = gate_a_1.view(bsz * self.num_heads, -1, 1) * position_bias

            position_bias = position_bias.view(attn_weights.size())

            attn_weights = attn_weights + position_bias

        attn_weights_float = F.softmax(
            attn_weights, dim=-1
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights, position_bias

    @staticmethod
    def _append_prev_key_padding_mask(
            key_padding_mask: Optional[Tensor],
            prev_key_padding_mask: Optional[Tensor],
            batch_size: int,
            src_len: int,
            static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            if src_len > prev_key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - prev_key_padding_mask.size(1)),
                    device=prev_key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [prev_key_padding_mask.float(), filler.float()], dim=1
                )
            else:
                new_key_padding_mask = prev_key_padding_mask.float()
        elif key_padding_mask is not None:
            if src_len > key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - key_padding_mask.size(1)),
                    device=key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [filler.float(), key_padding_mask.float()], dim=1
                )
            else:
                new_key_padding_mask = key_padding_mask.float()
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    def _get_input_buffer(
            self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
            self,
            incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
            buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights