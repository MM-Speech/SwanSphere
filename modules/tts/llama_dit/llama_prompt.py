
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn

from utils.nn.fa import (
    flash_attn_installed, 
    flash_attn_varlen_func, flash_attn_with_kvcache,
    index_first_axis, pad_input, unpad_input,
    RMSNorm,
    get_unpad_data as _get_unpad_data,
    make_additive_mask_from_padding
)
    
from modules.tts.f5_dit.f5_modules import AdaLayerNormZero, AdaLayerNormZero_Final
from utils.nn.seq_utils import sequence_mask
from modules.tts.llama_dit.llama_ca import precompute_freqs_cis, Attention, CrossAttention, FeedForward
from typing import Optional

@dataclass
class ModelArgs:
    encoder_dim: int = 1024
    encoder_n_layers: int = 24
    crossattn_n_layers: int = 24
    encoder_n_heads: int = 16
    encoder_n_kv_heads: int = None
    mlp_extend: float = None
    max_seq_len: int = 16384
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = 2
    use_causal_attn: bool = False
    use_caption_pool_in_adaln: bool = False
    use_caption_pool_gate_in_adaln: bool = False
    use_qk_norm: bool = False
    use_dynamic_cross_gate: bool = False
    use_gated_attention: bool = False
    use_moe_ffn: bool = False
    moe_p: float = 0.7                # Top-P routing threshold
    moe_num_routed: int = 8           # routed experts count
    moe_num_shared: int = 2           # shared experts count (always-on)
    moe_num_null: int = 1             # null experts count (no compute)
    moe_aux_loss_weight: float = 0.01 # training-time weight (you can anneal in loop)
    moe_use_gumbel: bool = False
    moe_gumbel_tau_start: float = 1.0       # 初始温度（训练早期）
    moe_gumbel_tau_end: float = 0.3         # 最终温度（训练后期）
    moe_gumbel_tau_anneal_steps: int = 200_000  # 退火步数
    moe_expert_dropout: float = 0.0   # 每个 step 随机屏蔽部分专家（0~1）
    moe_max_experts_per_token: int = 4  # 每个 token 最多用几个 routed experts

class DynamicTopPMoEFFN(nn.Module):
    """
    UniMoE-Audio 风格的 Top-P MoE-FFN，带一些增强：
      - routed experts: 按 token 动态路由
      - shared experts: 始终计算
      - null experts: 只在 gate 里占位，不计算
      - Top-P routing: 每个 token 动态选多少个专家
      - Gumbel-softmax（可选）: 训练时用噪声让路由更离散
      - 自动退火温度 τ: 从 tau_start 线性退火到 tau_end
      - expert dropout（可选）
      - time-aware routing: gate 输入显式融合 diffusion 时间 embedding t
    """
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int,
                 ffn_dim_multiplier: float,
                 num_routed: int, num_shared: int, num_null: int,
                 p: float = 0.7,
                 use_gumbel: bool = False,
                 gumbel_tau_start: float = 1.0,
                 gumbel_tau_end: float = 0.3,
                 gumbel_tau_anneal_steps: int = 200_000,
                 expert_dropout: float = 0.0,
                 max_experts_per_token: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.num_routed = num_routed
        self.num_shared = num_shared
        self.num_null = num_null
        self.p = p

        self.use_gumbel = use_gumbel
        self.gumbel_tau_start = gumbel_tau_start
        self.gumbel_tau_end = gumbel_tau_end
        self.gumbel_tau_anneal_steps = gumbel_tau_anneal_steps
        self.expert_dropout = expert_dropout
        self.max_experts_per_token = max_experts_per_token  # 新增

        # routed experts: 与原 FeedForward 结构一致
        self.routed_experts = nn.ModuleList([
            FeedForward(dim, hidden_dim, multiple_of, ffn_dim_multiplier)
            for _ in range(num_routed)
        ])

        # shared experts: 始终计算
        self.shared_experts = nn.ModuleList([
            FeedForward(dim, hidden_dim, multiple_of, ffn_dim_multiplier)
            for _ in range(num_shared)
        ])

        # gate 只在 (routed + null) 上做路由
        self.total_gate_experts = num_routed + num_null
        self.gate = nn.Linear(dim, self.total_gate_experts, bias=False)

        # time-aware: 让 gate 显式看到时间 embedding
        self.t_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.t_proj.weight)
        nn.init.zeros_(self.t_proj.bias)

        # 用一个 buffer 记录 forward 次数，做 τ 的自动退火
        self.register_buffer("gumbel_step", torch.zeros(1), persistent=False)

    def _current_tau(self):
        """根据当前 step 计算退火之后的 τ"""
        if (not self.use_gumbel) or (self.gumbel_tau_anneal_steps <= 0):
            return self.gumbel_tau_start

        # 防止多卡乱搞，这里就当 approximate contador
        step = float(self.gumbel_step.item())
        ratio = min(1.0, step / float(self.gumbel_tau_anneal_steps))
        tau = self.gumbel_tau_start + (self.gumbel_tau_end - self.gumbel_tau_start) * ratio
        return max(tau, 1e-4)

    def _top_p_select(self, probs: torch.Tensor, p: float, max_k: Optional[int] = None):
        """
        probs: [N, E]
        返回 selected_mask: [N, E] bool，Top-p + 最多 max_k 个专家。
        """
        # 排序
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)  # [N,E]
        cumsum = torch.cumsum(sorted_probs, dim=-1)

        # 这里的 trick:
        #   cumulative - prob = 前一个位置的累积概率
        #   我们保留所有满足 cumulative_prev < p 的位置
        #   => 前缀累积 >= p 时刚好包含当前这个
        prev_cum = cumsum - sorted_probs
        keep_sorted = prev_cum < p             # [N,E]
        keep_sorted[..., 0] = True             # 至少 1 个

        if max_k is not None and max_k > 0:
            keep_sorted[..., max_k:] = False   # 限制最多专家数

        selected_mask = torch.zeros_like(keep_sorted, dtype=torch.bool)
        selected_mask.scatter_(1, sorted_idx, keep_sorted)
        return selected_mask

    def _load_balance_loss(self, probs, selected_mask):
        E = probs.shape[1]  # R + Nn
        importance = probs.mean(dim=0)          # [E]
        load = selected_mask.float().mean(dim=0)  # [E]
        return (importance * load).sum() * E

    def forward(self, x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                t: Optional[torch.Tensor] = None):
        """
        x: [B,T,C]
        padding_mask: [B,T] bool, True=有效
        t: [B,C] diffusion time embedding (或 [B,T,C])
        """
        B, T, C = x.shape
        device = x.device
        dtype = x.dtype

        # --- mask ---
        if padding_mask is None:
            padding_mask = torch.ones((B, T), device=device, dtype=torch.bool)
        else:
            padding_mask = padding_mask.to(torch.bool)

        x_flat = x.reshape(B * T, C)
        mask_flat = padding_mask.reshape(B * T)

        idx = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)  # [N]
        if idx.numel() == 0:
            return x, x.new_tensor(0.0)

        x_valid = x_flat.index_select(0, idx)  # [N,C]
        N = x_valid.shape[0]

        # ===== gate 输入：token 表征 + time embedding =====
        gate_in = x_valid
        if t is not None:
            if t.dim() == 2:
                # [B,C] -> [B,T,C]
                t_expanded = t.unsqueeze(1).expand(B, T, t.shape[-1])
            else:
                # [B,T,C] (或更长)
                t_expanded = t
                if t_expanded.shape[1] != T:
                    t_expanded = t_expanded[:, :T, :]
            t_flat = t_expanded.reshape(B * T, -1)
            t_valid = t_flat.index_select(0, idx)  # [N,C]
            gate_in = gate_in + self.t_proj(t_valid)

        gate_logits = self.gate(gate_in)  # [N,E], E=num_routed+num_null

        # # ===== null experts logit bias =====
        NULL_LOGIT_BIAS = -1.0
        if self.num_null > 0 and NULL_LOGIT_BIAS != 0.0:
            gate_logits[:, self.num_routed:] = gate_logits[:, self.num_routed:] + NULL_LOGIT_BIAS

        # ===== expert dropout（可选）=====
        if self.expert_dropout > 0.0 and self.training:
            drop_mask = (torch.rand(self.total_gate_experts, device=device) < self.expert_dropout)  # [E]
            gate_logits = gate_logits.masked_fill(drop_mask.unsqueeze(0), float('-inf'))

        # ===== probs：Gumbel-softmax + τ 退火（仅训练）=====
        if self.use_gumbel and self.training:
            tau = self._current_tau()
            with torch.no_grad():
                self.gumbel_step += 1
            u = torch.rand_like(gate_logits).clamp_(1e-9, 1 - 1e-9)
            gumbel = -torch.log(-torch.log(u))
            logits = (gate_logits + gumbel) / tau
            probs = F.softmax(logits, dim=-1, dtype=torch.float32).to(dtype)
        else:
            probs = F.softmax(gate_logits, dim=-1, dtype=torch.float32).to(dtype)

        selected_mask = self._top_p_select(probs, self.p, max_k=self.max_experts_per_token)

        # 兜底：每个 token 至少 1 个 routed
        R = self.num_routed
        no_routed = ~selected_mask[:, :R].any(dim=-1)
        if no_routed.any():
            top_r = torch.argmax(probs[no_routed, :R], dim=-1)
            selected_mask[no_routed, top_r] = True

        # 在 (routed + null) 上一起归一化
        weights_all = probs * selected_mask.to(dtype)
        weights_all = weights_all / weights_all.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        weights = weights_all[:, :R]  # routed 权重，和可能 < 1（null 吃掉的部分）

        if self.training and torch.rand((), device=device) < 1e-4:
            with torch.no_grad():
                routed_sel = selected_mask[:, :self.num_routed]
                k_per_tok = routed_sel.sum(dim=-1).to(torch.int64)
                mean_k = k_per_tok.float().mean().item()
                min_k = k_per_tok.min().item()
                max_k = k_per_tok.max().item()
                Kcap = int(self.max_experts_per_token or self.num_routed)
                hist = torch.bincount(k_per_tok.clamp(0, Kcap), minlength=Kcap + 1).cpu().tolist()
                print(f"[MoE debug] N_valid={N} routed_k mean={mean_k:.3f} min={min_k} max={max_k} hist(0..{Kcap})={hist}")

                routed_weight_sum = weights.sum(dim=-1).mean().item()
                print(f"[MoE debug] mean routed weight sum = {routed_weight_sum:.3f}")

                null_prob = probs[:, -1].mean().item()
                top1 = probs.max(dim=-1).values
                top1_mean = top1.mean().item()
                top1_gt_p = (top1 > self.p).float().mean().item()
                print(f"[MoE debug] null_prob_mean={null_prob:.3f} top1_mean={top1_mean:.3f} top1>p={top1_gt_p:.3f}")

        # ===== shared experts（始终算）=====
        shared_out = torch.zeros_like(x_valid)
        if self.num_shared > 0:
            # shared 很少（通常 1~2），这里 loop 影响不大
            for se in self.shared_experts:
                shared_out = shared_out + se(x_valid).to(shared_out.dtype)
            shared_out = shared_out / float(self.num_shared)

        # ======================================================================
        # 关键提速点 2：一次 nonzero + 分组 dispatch，避免每个 expert 做 nonzero/index_select
        # ======================================================================
        routed_out = torch.zeros_like(x_valid)

        if self.num_routed > 0:
            routed_sel = selected_mask[:, :self.num_routed]  # [N, R]
            # 拉平成 (token_id, expert_id) 对，大小约 N*k（k<=4）
            token_ids, expert_ids = torch.nonzero(routed_sel, as_tuple=True)  # [M], [M]
            M = token_ids.numel()

            if M > 0:
                # 对应权重
                w = weights[token_ids, expert_ids].unsqueeze(-1).to(dtype)  # [M,1]

                # 按 expert 分组：只做一次排序
                order = torch.argsort(expert_ids)
                token_ids = token_ids[order]
                expert_ids = expert_ids[order]
                w = w[order]

                # 一次性 gather 所有 pair 的输入（避免 per-expert index_select）
                x_pairs = x_valid.index_select(0, token_ids)  # [M,C]

                # 统计每个 expert 需要处理多少 token
                counts = torch.bincount(expert_ids, minlength=self.num_routed)  # [R]
                # 预分配输出
                y_pairs = torch.empty_like(x_pairs)

                # 顺序跑每个 expert（这里仍有 R 次前向，但不再有 R 次 gather/scatter 的高开销）
                start = 0
                # counts.tolist() 会触发一次很小的同步，但 R 很小（比如 8），总体收益仍然很大
                for e_id, cnt in enumerate(counts.tolist()):
                    if cnt == 0:
                        continue
                    end = start + cnt
                    y = self.routed_experts[e_id](x_pairs[start:end])
                    y_pairs[start:end] = y.to(dtype)
                    start = end

                # 乘权重后，一次性 index_add 聚合回 token 维度（token 可能重复，天然支持累加）
                y_pairs = y_pairs * w
                routed_out.index_add_(0, token_ids, y_pairs)

        out_valid = routed_out + shared_out

        # ===== scatter 回 [B,T,C] =====
        out_flat = torch.zeros_like(x_flat)
        out_flat.index_copy_(0, idx, out_valid)
        out = out_flat.reshape(B, T, C)

        aux = self._load_balance_loss(probs, selected_mask)
        return out, aux


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
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

        if getattr(args, "use_moe_ffn", False):
            self.feed_forward = DynamicTopPMoEFFN(
                dim=args.encoder_dim,
                hidden_dim=args.encoder_dim,
                multiple_of=args.multiple_of,
                ffn_dim_multiplier=args.ffn_dim_multiplier,
                num_routed=args.moe_num_routed,
                num_shared=args.moe_num_shared,
                num_null=args.moe_num_null,
                p=args.moe_p,
                use_gumbel=args.moe_use_gumbel,
                gumbel_tau_start=args.moe_gumbel_tau_start,
                gumbel_tau_end=args.moe_gumbel_tau_end,
                gumbel_tau_anneal_steps=args.moe_gumbel_tau_anneal_steps,
                expert_dropout=args.moe_expert_dropout,
                max_experts_per_token=args.moe_max_experts_per_token,
            )
        else:
            self.feed_forward = FeedForward(
                dim=args.encoder_dim,
                hidden_dim=args.encoder_dim,
                multiple_of=args.multiple_of,
                ffn_dim_multiplier=args.ffn_dim_multiplier,
            )

        self.attention_norm = AdaLayerNormZero(args.encoder_dim)
        self.cross_attention_norm = nn.LayerNorm(args.encoder_dim, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(args.encoder_dim, elementwise_affine=False, eps=1e-6)

        if args.use_dynamic_cross_gate:
            self.cross_gating_proj = nn.Linear(self.encoder_dim, self.encoder_dim)
            nn.init.zeros_(self.cross_gating_proj.weight)
            nn.init.constant_(self.cross_gating_proj.bias, -2.0)  # init ~ 0.12
        else:
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
            use_cache: bool = False,
    ):
        """
        x:    [B, T, C]
        mask: [B, T] bool
        t:    [B, C] time embedding
        """
        dtype = x.dtype

        # === Self-Attn ===
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attention_norm(x, emb=t)
        attn_output = self.attention(norm, start_pos, freqs_cis, mask=mask, use_cache=use_cache)
        h = x + gate_msa.unsqueeze(1) * attn_output

        # === Cross-Attn ===
        if context is not None:
            norm = self.cross_attention_norm(h.float()).to(dtype)
            cross_attn_output = self.cross_attention(
                norm,
                context,
                context_lens.to(torch.long),
                query_mask=mask
            )

            if self.args.use_dynamic_cross_gate:
                cross_gate = torch.sigmoid(self.cross_gating_proj(norm))
                h = h + cross_gate.to(dtype) * cross_attn_output
            else:
                h = h + self.cross_gate.to(dtype) * cross_attn_output

        # === FFN / MoE-FFN ===
        norm = self.ffn_norm(h.float()).to(dtype) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        moe_aux = None
        if getattr(self.args, "use_moe_ffn", False):
            ff_output, moe_aux = self.feed_forward(norm, padding_mask=mask, t=t)
        else:
            ff_output = self.feed_forward(norm)

        out = h + gate_mlp.unsqueeze(1) * ff_output

        if getattr(self.args, "use_moe_ffn", False):
            return out, moe_aux
        else:
            return out

class LLaMa(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.encoder_n_layers = params.encoder_n_layers

        self.layers = nn.ModuleList()
        for _ in range(params.encoder_n_layers):
            self.layers.append(TransformerBlock(params))

        self.norm = AdaLayerNormZero_Final(params.encoder_dim)
        self.out_proj = nn.Linear(params.encoder_dim, params.encoder_dim)

        self._use_cap_mod = bool(getattr(self.params, "use_caption_pool_in_adaln", False))
        if self._use_cap_mod:
            self.cap_embedder = nn.Linear(params.encoder_dim, params.encoder_dim)
            self.cap_gate = nn.Linear(params.encoder_dim, params.encoder_dim)
            nn.init.zeros_(self.cap_embedder.weight)
            nn.init.zeros_(self.cap_embedder.bias)
            nn.init.zeros_(self.cap_gate.weight)
            nn.init.zeros_(self.cap_gate.bias)

        freqs_cis = precompute_freqs_cis(
            self.params.encoder_dim // self.params.encoder_n_heads,
            self.params.max_seq_len
        )
        self.register_buffer("freqs_cis", torch.view_as_real(freqs_cis), persistent=False)

    def forward(self, x, t, attn_mask,
                context=None, context_lens=None,
                start_pos=0, use_cache=False, do_checkpoint=False):
        """
        x: [B,T,C]
        t: [B,C] time embedding
        attn_mask: [B,T] bool
        context/context_lens: caption / text encoder 输出（可选）
        """
        freqs_cis = torch.view_as_complex(self.freqs_cis.float())[start_pos: start_pos + x.size(1)]

        # caption pooling（如有） -> 可选注入到 t
        if (context is not None) and (context_lens is not None):
            B, Lc, C = context.size()
            device = context.device

            lengths = context_lens.to(device=device).long().view(-1)         # [B]
            idxs = torch.arange(Lc, device=device).unsqueeze(0).expand(B, Lc)
            cap_mask = (idxs < lengths.unsqueeze(1))                         # [B,Lc]

            cap_mask_f = cap_mask.to(context.dtype).unsqueeze(-1)            # [B,Lc,1]
            cap_sum = (context * cap_mask_f).sum(dim=1)                      # [B,C]
            denom = cap_mask_f.sum(dim=1).clamp(min=1.0)                     # [B,1]
            cap_pool = cap_sum / denom                                       # [B,C]

            if self._use_cap_mod:
                cap_emb = self.cap_embedder(cap_pool)
                cap_gate = torch.tanh(self.cap_gate(t + cap_emb.to(t.dtype)))
                t = t + cap_emb.to(t.dtype) * cap_gate
        else:
            cap_pool = None  # 目前 MoE 没显式用到 caption pool，hidden 里已经有信息

        use_moe = bool(getattr(self.params, "use_moe_ffn", False))
        total_moe_aux = None

        for i, layer in enumerate(self.layers):
            # 有 MoE 时，为简单起见仍然关闭 checkpoint（否则需要自定义 ckpt 返回 aux）
            if torch.is_grad_enabled() and do_checkpoint and (not use_moe):
                def create_custom_forward(module):
                    def custom_forward(x_, t_, start_pos_, freqs_cis_, attn_mask_,
                                       context_, context_lens_, use_cache_):
                        return module(x_, t_, start_pos_, freqs_cis_, attn_mask_,
                                      context=context_, context_lens=context_lens_,
                                      use_cache=use_cache_)
                    return custom_forward

                ckpt_kwargs = {"use_reentrant": False}
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    x, t, start_pos, freqs_cis, attn_mask, context, context_lens, use_cache,
                    **ckpt_kwargs,
                )
            else:
                out = layer(
                    x, t, start_pos, freqs_cis, attn_mask,
                    context=context, context_lens=context_lens,
                    use_cache=use_cache,
                )
                if use_moe:
                    x, moe_aux = out
                    if moe_aux is not None:
                        total_moe_aux = moe_aux if total_moe_aux is None else (total_moe_aux + moe_aux)
                else:
                    x = out

        x = self.norm(x, t)
        x = self.out_proj(x)

        if use_moe:
            if total_moe_aux is not None:
                total_moe_aux = total_moe_aux / float(self.encoder_n_layers)
            else:
                total_moe_aux = x.new_tensor(0.0)
            return x, total_moe_aux

        return x