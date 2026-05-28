import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

class MaskFlowMatching(nn.Module):
    """
    Mask Flow Matching for discrete tokens with padding and optional conditioning.
    - Backbone transformer: forward(tokens: [B,T], t: [B], cond: Optional[Any], attn_mask: Optional[Bool[B,T]]) -> logits [B,T,V]
    - Focus: mask-flow computations (schedule, masking, CFM weights), padding handling, conditional support.
    """

    def __init__(
        self,
        vocab_size: int,
        mask_id: int,
        backbone: nn.Module = None,
        schedule: str = "cosine",            # 'linear' | 'cosine' | 'quadratic'
        num_steps: int = 12,
        use_cfm_weight: bool = True,         # w(t) = -alpha'(t)/(alpha(t)+eps)
        pad_id: Optional[int] = None,        # enable padding support if not None
        t_eps: float = 1e-3,                 # sample t in [0, 1 - t_eps]
        enforce_one_mask: bool = True,       # ensure at least one masked non-pad per sample at train-time
        eps: float = 1e-8,
        w_clip: Optional[float] = None,      # optional clip for w(t)
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_id = mask_id
        self.pad_id = pad_id

        self.backbone = backbone
        self.schedule = schedule
        self.num_steps = num_steps
        self.use_cfm_weight = use_cfm_weight

        self.t_eps = float(t_eps)
        self.enforce_one_mask = enforce_one_mask
        self.eps = eps
        self.w_clip = w_clip

        self.device = device

    # ========= Schedule =========
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        # α(0)=1, α(1)=0
        if self.schedule == "linear":
            return 1.0 - t
        elif self.schedule == "cosine":
            return 0.5 * (1.0 + torch.cos(torch.pi * t))
        elif self.schedule == "quadratic":
            return (1.0 - t) ** 2
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def dalpha(self, t: torch.Tensor) -> torch.Tensor:
        if self.schedule == "linear":
            return -torch.ones_like(t)
        elif self.schedule == "cosine":
            return -0.5 * torch.pi * torch.sin(torch.pi * t)
        elif self.schedule == "quadratic":
            return -2.0 * (1.0 - t)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    # ========= Utilities =========
    def _sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # sample t in [0, 1 - t_eps]
        high = 1.0 - max(self.t_eps, 0.0)
        return torch.rand(batch_size, device=device) * high

    def _build_valid_mask(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        valid positions = not padding. If pad_id is None and padding_mask is None, all positions valid.
        Returns bool[B,T]
        """
        if padding_mask is not None:
            # assume padding_mask: True for pad positions
            valid = ~padding_mask.bool()
        elif self.pad_id is not None:
            valid = (x != self.pad_id)
        else:
            valid = torch.ones_like(x, dtype=torch.bool)
        return valid

    def _build_xt(
        self,
        x0: torch.Tensor,
        alpha_t: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build x_t by masking only valid (non-pad) positions.
        Returns:
            x_t: [B, T] with mask_id inserted
            mask: [B, T] boolean WHERE newly masked (only non-pad can be True)
            valid: [B, T] boolean non-pad positions
        """
        B, T = x0.shape
        device = x0.device
        valid = self._build_valid_mask(x0, padding_mask)  # [B,T]
        p = alpha_t.view(B, 1).expand(B, T)
        # Only sample on valid positions
        bern = torch.bernoulli(p.clamp(0.0, 1.0)).bool()
        mask = bern & valid

        if self.enforce_one_mask:
            # Ensure at least one mask on valid positions when any valid exists
            valid_counts = valid.sum(dim=1)  # [B]
            mask_counts = mask.sum(dim=1)    # [B]
            need = (valid_counts > 0) & (mask_counts == 0)
            if need.any():
                for b in torch.nonzero(need, as_tuple=False).squeeze(-1).tolist():
                    idx_valid = torch.nonzero(valid[b], as_tuple=False).squeeze(-1)
                    # pick one uniformly
                    j = torch.randint(low=0, high=idx_valid.numel(), size=(1,), device=device).item()
                    mask[b, idx_valid[j]] = True

        x_t = torch.where(mask, torch.full_like(x0, self.mask_id), x0)
        return x_t, mask, valid

    def _loss_weight(self, t: torch.Tensor, alpha_t: torch.Tensor) -> torch.Tensor:
        w = -self.dalpha(t)  # >= 0
        if self.use_cfm_weight:
            w = w / (alpha_t + self.eps)
        if self.w_clip is not None:
            w = torch.clamp(w, max=self.w_clip)
        return w  # [B]

    def _attn_mask_from_valid(self, valid: torch.Tensor) -> torch.Tensor:
        """
        Produce an attention mask for transformer if needed.
        We pass 'valid' (True for keep) directly; the backbone must know how to consume it.
        """
        return valid  # [B,T] bool

    # ========= Training (loss computation) =========
    def compute_loss(
        self,
        x0: torch.Tensor,                     # [B, T]
        t: Optional[torch.Tensor] = None,     # [B]
        padding_mask: Optional[torch.Tensor] = None,  # [B,T] True at pad
        cond: Optional[object] = None,        # forwarded to transformer
    ) -> Dict[str, torch.Tensor]:
        """
        Steps:
          1) Sample t ~ U(0, 1 - t_eps).
          2) α(t)、按 valid 位置采样 mask 得到 x_t。
          3) 前向：logits = net(x_t, t, cond, attn_mask).
          4) 仅在 (masked & valid) 位置计算交叉熵。
          5) 权重：w(t) = -α'(t) 或 -α'(t)/(α(t)+ε)。
          6) 归一化：
              - use_cfm_weight=True: 按非pad长度归一化（保持期望尺度稳定）。
              - use_cfm_weight=False: 可按“被mask数量”归一化得到每位置平均CE。
        """
        device = x0.device if self.device is None else self.device
        B, T = x0.shape
        if t is None:
            t = self._sample_t(B, device)
        alpha_t = self.alpha(t)  # [B]
        x_t, mask, valid = self._build_xt(x0, alpha_t, padding_mask)  # [B,T] x_t, [B,T] mask, [B,T] valid

        attn_mask = self._attn_mask_from_valid(valid)  # [B,T] bool
        logits = self.backbone(x_t, t, cond=cond, attn_mask=attn_mask)  # [B,T,V]

        # Cross-entropy on masked & valid positions        
        pos = (mask & valid)  # [B,T] bool
        if pos.any():
            # gather masked positions
            idx = pos.view(-1).nonzero(as_tuple=False).squeeze(-1)  # [M]
            logits_masked = logits.view(B * T, self.vocab_size).index_select(0, idx)  # [M,V]
            targets_masked = x0.view(-1).index_select(0, idx)                          # [M]
            ce_masked = F.cross_entropy(logits_masked, targets_masked, reduction="none")  # [M]
            # reduce per sample
            ce_sum = torch.zeros(B, device=logits.device)
            # scatter-add back to samples
            sample_ids = (torch.arange(B, device=logits.device).unsqueeze(1).expand(B, T)[pos]).long()
            ce_sum.scatter_add_(0, sample_ids, ce_masked)
        else:
            ce_sum = torch.zeros(B, device=logits.device)

        # Weights
        w = self._loss_weight(t, alpha_t)  # [B]

        # Normalization
        nonpad_counts = valid.sum(dim=1).clamp(min=1)  # [B]
        if self.use_cfm_weight:
            # Do NOT divide by masked count; divide by non-pad length to keep scale length-invariant
            loss_per_sample = w * ce_sum / nonpad_counts.float()
        else:
            masked_counts = pos.sum(dim=1).clamp(min=1)
            loss_per_sample = w * ce_sum / masked_counts.float()

        loss = loss_per_sample.mean()

        return {
            "loss": loss,
            "t": t.detach(),
            "alpha_t": alpha_t.detach(),
            "masked_ratio_nonpad": (pos.sum(dim=1).float() / nonpad_counts.float()).detach(),
        }

    # ========= Inference (iterative unmask) =========
    # @torch.no_grad()
    # def infer(
    #     self,
    #     x_init: torch.Tensor,                 # [B, T], may contain mask_id and/or pad_id
    #     steps: Optional[int] = None,
    #     temperature: float = 1.0,
    #     topk_per_step: Optional[int] = None,  # fixed number per step per sample (on valid masked positions)
    #     token_topk: Optional[int] = None,     # number of top tokens per position to sample from
    #     cond: Optional[object] = None,
    #     padding_mask: Optional[torch.Tensor] = None,  # [B,T] True at pad
    #     return_all_steps: bool = False,
    # ) -> torch.Tensor:
    #     """
    #     Iterative unmasking on non-pad positions only.
    #     """
    #     if x_init is None:
    #         raise ValueError("x_init must be provided.")
    #     x = x_init.clone()
    #     B, T = x.shape
    #     device = x.device
    #     V = self.vocab_size
        
    #     if steps is None:
    #         steps = self.num_steps

    #     # valid (non-pad) positions
    #     valid = self._build_valid_mask(x, padding_mask)  # [B,T]
    #     attn_mask = self._attn_mask_from_valid(valid)

    #     # time grid [0, 1]
    #     t_grid = torch.linspace(0.0, 1.0, steps + 1, device=x.device)
    #     traj = [x.clone()] if return_all_steps else None
        
    #     # small epsilon for numerical stability
    #     eps = max(self.eps, 1e-8)
    #     temp = max(temperature, 1e-6)  # guard temperature

    #     for k in range(steps):
    #         t_k = t_grid[k].expand(B)                # [B]
    #         alpha_k = self.alpha(t_k)                # [B]
    #         t_next = t_grid[k + 1].expand(B)
    #         alpha_next = self.alpha(t_next)          # [B]

    #         masked = (x == self.mask_id) & valid     # [B,T]
    #         masked_count = masked.sum(dim=1)         # [B]
    #         if (masked_count == 0).all():
    #             break

    #         # decide how many to unmask this step
    #         if topk_per_step is not None:
    #             n_unmask = torch.minimum(masked_count, torch.full_like(masked_count, topk_per_step))
    #         else:
    #             delta = (alpha_k - alpha_next).clamp(min=0.0)  # [B]
    #             # normalize by current alpha_k to match target masked ratio schedule
    #             denom = torch.clamp(alpha_k, min=1e-8)         # [B]
    #             frac = delta / denom                            # [B]
    #             n_unmask = torch.floor(frac * masked_count.float()).to(torch.long)
    #             # at least 1 if any masked exists
    #             n_unmask = torch.where(masked_count > 0, torch.maximum(n_unmask, torch.ones_like(n_unmask)), n_unmask)
    #             # cap by available masked positions
    #             n_unmask = torch.minimum(n_unmask, masked_count)

    #         # forward
    #         logits = self.backbone(x, t_k, cond=cond, attn_mask=attn_mask)  # [B,T,V]
            
    #         # filter out invalid tokens (mask_id, pad_id)
    #         logits_f = logits.clone()
    #         neg_inf = torch.finfo(logits_f.dtype).min
    #         logits_f[..., self.mask_id] = neg_inf
    #         if self.pad_id is not None:
    #             logits_f[..., self.pad_id] = neg_inf
                
    #         # compute position selection weights (confidence per position)
    #         # use softmax on filtered logits with temperature, then take max prob
    #         probs_all = F.softmax(logits_f / temp, dim=-1)     # [B,T,V]
    #         conf = probs_all.max(dim=-1).values                # [B,T]
    #         weights = conf.masked_fill(~masked, 0.0)           # [B,T]
            
    #         # Gumbel-Top-k over positions (weighted sampling without replacement)
    #         logw = torch.log(weights + 1e-20)                  # [B,T]
    #         U = torch.rand_like(logw).clamp_min(1e-6)
    #         g = -torch.log(-torch.log(U))                      # [B,T]
    #         y = logw + g
    #         order = torch.argsort(y, dim=1, descending=True)   # [B,T]
            
    #         Kmax = int(n_unmask.max().item())
    #         if Kmax > 0:
    #             top_pos = order[:, :Kmax]                                      # [B,Kmax]
    #             active = (torch.arange(Kmax, device=device).unsqueeze(0)
    #                     < n_unmask.unsqueeze(1))                              # [B,Kmax]
    #             fill_mask = torch.zeros(B, T, dtype=torch.bool, device=device) # [B,T]
    #             fill_mask.scatter_(1, top_pos, active)                          # mark selected positions
    #             fill_mask = fill_mask & masked                                  # keep only masked&valid
                
    #             # per-position token sampling (Top-k optional)
    #             if token_topk is None:
    #                 # sample from full (filtered) vocab
    #                 pred = torch.multinomial(probs_all.view(B * T, V), num_samples=1, replacement=True).view(B, T)
    #             else:
    #                 k_tok = int(token_topk)
    #                 # ensure k_tok does not exceed valid vocabulary size
    #                 # If k_tok >= V, this degenerates to full sampling.
    #                 k_tok = max(1, min(k_tok, V))
    #                 # take per-position top-k on filtered logits
    #                 topk_vals, topk_idx = torch.topk(logits_f, k=k_tok, dim=-1)         # [B,T,k_tok], [B,T,k_tok]
    #                 probs_topk = F.softmax(topk_vals / temp, dim=-1)                    # [B,T,k_tok]
    #                 # multinomial expects 2D [N, K]
    #                 sel = torch.multinomial(probs_topk.view(B * T, k_tok), num_samples=1, replacement=True).view(B, T)  # [B,T]
    #                 # map back to vocab indices
    #                 pred = topk_idx.gather(dim=-1, index=sel.unsqueeze(-1)).squeeze(-1) # [B,T]
                    
    #             # apply predictions only on selected positions
    #             x = torch.where(fill_mask, pred, x)
                
    #         # else: nothing to fill (shouldn't happen because n_unmask>=1 when masked_count>0)

    #         if return_all_steps:
    #             traj.append(x.clone())

    #     return torch.stack(traj, dim=0) if return_all_steps else x
    
    def backbone_inference(self, x, t_k, cond, attn_mask, guidance_kwargs=None):
        # maybe override for cfg
        return self.backbone(x, t_k, cond=cond, attn_mask=attn_mask)
    
    def _decode_step(
        self,
        x: torch.Tensor,                    # [B,T] 当前序列（包含 mask）
        valid: torch.Tensor,                # [B,T] 非 pad 位置 bool
        t_k: torch.Tensor,                  # [B]
        t_next: torch.Tensor,               # [B]
        alpha_k: torch.Tensor,              # [B]
        alpha_next: torch.Tensor,           # [B]
        attn_mask: torch.Tensor,            # [B,T] 传给 backbone
        cond=None,
        temperature: float = 1.0,
        token_topk: int = None,
        token_topp: float = None,           # 若非 None 优先用 top-p
        confidence_greedy: float = 0.9,     # 置信度阈值，超过则贪心
        use_margin: bool = True,            # 位置打分指标
        guidance_kwargs: dict = None,        
        topk_per_step: int = None,          # 固定每步位置数，优先级低于 override
        n_unmask_override: torch.Tensor = None,  # [B] 若提供则直接用
    ):
        """
        返回：
        x_new: [B,T]
        filled_any: bool 是否有位置被填
        """
        B, T = x.shape
        device = x.device
        V = self.vocab_size

        # 1) 当前被 mask 的有效位置
        masked = (x == self.mask_id) & valid               # [B,T]
        masked_count = masked.sum(dim=1)                   # [B]
        if (masked_count == 0).all():
            return x, False

        # 2) 计算本步要填的位置数 n_unmask
        if n_unmask_override is not None:
            n_unmask = torch.minimum(masked_count, n_unmask_override)
        elif topk_per_step is not None:
            n_unmask = torch.minimum(masked_count, torch.full_like(masked_count, topk_per_step))
        else:
            # 按 α 网格等比衰减的目标
            nonpad = valid.sum(dim=1)                                          # [B]
            target_k = torch.round(alpha_k * nonpad).long()
            target_n = torch.round(alpha_next * nonpad).long()
            n_unmask = (target_k - target_n).clamp_min(1)
            n_unmask = torch.minimum(n_unmask, masked_count)

        Kmax = int(n_unmask.max().item())
        if Kmax == 0:
            return x, False

        # 3) 前向（可选 CFG）
        logits = self.backbone_inference(x, t_k, cond=cond, attn_mask=attn_mask, guidance_kwargs=guidance_kwargs)     # [B,T,V]

        # 4) 过滤无效 token（mask/pad）
        logits_f = logits.clone()
        neg_inf = torch.finfo(logits_f.dtype).min
        logits_f[..., self.mask_id] = neg_inf
        if self.pad_id is not None:
            logits_f[..., self.pad_id] = neg_inf

        # 5) 位置选择权重：margin 或 max prob
        temp = _safe_temperature(temperature)
        probs_all = F.softmax(logits_f / temp, dim=-1)                         # [B,T,V]
        if use_margin:
            top2_vals, _ = torch.topk(probs_all, k=2, dim=-1)
            score = (top2_vals[..., 0] - top2_vals[..., 1])                    # [B,T]
        else:
            score = probs_all.max(dim=-1).values
        weights = score.masked_fill(~masked, 0.0)

        # Gumbel-Top-k 采样位置
        logw = torch.log(weights + 1e-20)
        U = torch.rand_like(logw).clamp_min(1e-6)
        g = -torch.log(-torch.log(U))
        y = logw + g
        order = torch.argsort(y, dim=1, descending=True)                       # [B,T]
        top_pos = order[:, :Kmax]                                              # [B,Kmax]
        active = (torch.arange(Kmax, device=device).unsqueeze(0) < n_unmask.unsqueeze(1))  # [B,Kmax]
        fill_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        fill_mask.scatter_(1, top_pos, active)
        fill_mask &= masked

        if not fill_mask.any():
            return x, False

        # 6) 在选中位置进行 token 采样：高置信度贪心 + top-p/top-k/全采样
        x_new = x.clone()
        idx = fill_mask.view(-1).nonzero(as_tuple=False).squeeze(-1)           # [M]
        logits_sel = logits_f.view(B*T, V).index_select(0, idx)                # [M,V]
        probs_sel  = probs_all.view(B*T, V).index_select(0, idx)               # [M,V]
        conf_sel   = probs_sel.max(dim=-1).values                               # [M]

        out = torch.empty(idx.numel(), dtype=x.dtype, device=device)
        # 贪心
        greedy_idx = (conf_sel >= confidence_greedy).nonzero(as_tuple=False).squeeze(-1)
        if greedy_idx.numel() > 0:
            out[greedy_idx] = probs_sel[greedy_idx].argmax(dim=-1)

        # 采样
        sample_idx = (conf_sel < confidence_greedy).nonzero(as_tuple=False).squeeze(-1)
        if sample_idx.numel() > 0:
            if token_topp is not None:
                out[sample_idx] = _top_p_sample(logits_sel[sample_idx], p=float(token_topp), temperature=temp)
            elif token_topk is not None:
                k_tok = min(int(token_topk), V)
                topk_vals, topk_idx = torch.topk(logits_sel[sample_idx], k=k_tok, dim=-1)
                probs_topk = F.softmax(topk_vals / temp, dim=-1)
                sel = torch.multinomial(probs_topk, num_samples=1).squeeze(-1)
                out[sample_idx] = topk_idx.gather(-1, sel.unsqueeze(-1)).squeeze(-1)
            else:
                out[sample_idx] = torch.multinomial(probs_sel[sample_idx], num_samples=1).squeeze(-1)

        # 回填
        x_new.view(B*T)[idx] = out
        return x_new, True
    
    @torch.no_grad()
    def infer(
        self,
        x_init: torch.Tensor,
        steps: int = None,
        temperature: float = 1.0,
        topk_per_step: int = None,
        token_topk: int = None,
        token_topp: float = None,          # 新增：top-p 采样开关
        cond=None,
        padding_mask: torch.Tensor = None,
        return_all_steps: bool = False,
        schedule_mode: str = "uniform_alpha",  # "uniform_alpha" | "linear_t"
        alpha_min: float = 1e-3,
        confidence_greedy: float = 0.9,
        use_margin: bool = True,
        guidance_kwargs: dict = None,
        remask_last: bool = True,         # 尾声重掩码一轮
        remask_frac: float = 0.05,
        remask_thresh: float = 0.5,
    ):
        x = x_init.clone()
        B, T = x.shape
        steps = steps or self.num_steps

        valid = self._build_valid_mask(x, padding_mask)
        attn_mask = self._attn_mask_from_valid(valid)
        traj = [x.clone()] if return_all_steps else None

        # 构造时间/alpha 网格
        if schedule_mode == "uniform_alpha":
            t_grid, a_grid = _build_uniform_alpha_grid(steps, schedule=self.schedule, alpha_min=alpha_min, device=x.device)
        elif schedule_mode == "linear_t":
            t_grid = torch.linspace(0.0, 1.0, steps + 1, device=x.device)
            a_grid = self.alpha(t_grid)
        else:
            raise ValueError(f"Unknown schedule_mode: {schedule_mode}")

        # 主循环
        for k in range(steps):
            t_k     = t_grid[k].expand(B)
            t_next  = t_grid[k + 1].expand(B)
            alpha_k = a_grid[k].expand(B)
            alpha_n = a_grid[k + 1].expand(B)

            x_new, filled = self._decode_step(
                x=x,
                valid=valid,
                t_k=t_k,
                t_next=t_next,
                alpha_k=alpha_k,
                alpha_next=alpha_n,
                attn_mask=attn_mask,
                cond=cond,
                temperature=temperature,
                token_topk=token_topk,
                token_topp=token_topp,
                confidence_greedy=confidence_greedy,
                use_margin=use_margin,
                guidance_kwargs=guidance_kwargs,
                topk_per_step=topk_per_step,
                n_unmask_override=None,       # 正常分配
            )
            x = x_new
            if return_all_steps:
                traj.append(x.clone())
            # 若已经无可填位置，提前结束
            if not ((x == self.mask_id) & valid).any():
                break

        # 尾声：重掩码 + 再填一轮（这里复用 _decode_step）
        if remask_last:
            # 计算置信度
            t_final = t_grid[-1].expand(B)
            logits = self.backbone_inference(x, t_final, cond=cond, attn_mask=attn_mask, guidance_kwargs=guidance_kwargs)

            logits_f = logits.clone()
            neg_inf = torch.finfo(logits_f.dtype).min
            logits_f[..., self.mask_id] = neg_inf
            if self.pad_id is not None:
                logits_f[..., self.pad_id] = neg_inf
            probs = F.softmax(logits_f, dim=-1)
            conf = probs.max(dim=-1).values                                    # [B,T]

            # 选低置信度位置进行小比例重掩码
            filled = (x != self.mask_id) & valid
            lowconf = filled & (conf < remask_thresh)

            # 各样本最多重掩多少
            nonpad = valid.sum(dim=1)
            Kmax = torch.ceil(remask_frac * nonpad.float()).long()             # [B]

            # 为每个样本挑选 Kmax 个最低置信度位置
            score = (-conf).masked_fill(~lowconf, 1e9)                         # 越小越低
            order = torch.argsort(score, dim=1)                                # 从最小开始
            mask_new = torch.zeros_like(filled)
            for b in range(B):
                k = int(Kmax[b].item())
                if k > 0:
                    pos = order[b, :k]
                    mask_new[b, pos] = True

            # 执行重掩
            x = torch.where(mask_new, torch.full_like(x, self.mask_id), x)

            # 再填一轮（一次性填完全部新掩码，override 数量）
            masked = (x == self.mask_id) & valid
            masked_count = masked.sum(dim=1)
            if masked.any():
                x, _ = self._decode_step(
                    x=x,
                    valid=valid,
                    t_k=t_final,           # 用最终 t（等效于不再依赖 α 计划）
                    t_next=t_final,
                    alpha_k=self.alpha(t_final),
                    alpha_next=self.alpha(t_final),
                    attn_mask=attn_mask,
                    cond=cond,
                    temperature=temperature,
                    token_topk=token_topk,
                    token_topp=token_topp,
                    confidence_greedy=confidence_greedy,
                    use_margin=use_margin,
                    guidance_kwargs=guidance_kwargs,
                    topk_per_step=None,
                    n_unmask_override=masked_count,    # 关键：一次填完
                )
                if return_all_steps:
                    traj.append(x.clone())

        return torch.stack(traj, dim=0) if return_all_steps else x


def _safe_temperature(temp: float) -> float:
    return max(temp, 1e-6)

def _top_p_sample(logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0) -> torch.Tensor:
    """
    logits: [N, V]
    返回：采样得到的 token 索引 [N]
    """
    temperature = _safe_temperature(temperature)
    probs = F.softmax(logits / temperature, dim=-1)  # [N,V]
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    # 保留累计概率 <= p 的集合，并确保至少保留一个元素
    keep = (cumsum <= p) | (torch.arange(probs.size(-1), device=probs.device) == 0)
    # 重新归一化
    masked = sorted_probs * keep
    denom = masked.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    masked = masked / denom
    sel = torch.multinomial(masked, num_samples=1)             # [N,1]
    return sorted_idx.gather(-1, sel).squeeze(-1)              # [N]

def _inv_alpha(alpha: torch.Tensor, schedule: str) -> torch.Tensor:
    """
    α -> t 的反函数，逐元素。
    """
    if schedule == "linear":
        return 1.0 - alpha
    elif schedule == "cosine":
        z = (2 * alpha - 1).clamp(-1 + 1e-7, 1 - 1e-7)
        return torch.arccos(z) / torch.pi
    elif schedule == "quadratic":
        return 1.0 - torch.sqrt(alpha.clamp_min(0.0))
    else:
        raise ValueError(schedule)

def _build_uniform_alpha_grid(steps: int, schedule: str, alpha_min: float, device: torch.device):
    """
    几何衰减：alpha_{k+1} = r * alpha_k, r = (alpha_min/alpha_0)^(1/steps).
    返回 t_grid, a_grid: [steps+1]
    """
    a0 = 1.0
    r = (alpha_min / a0) ** (1.0 / steps)
    a = [a0 * (r ** k) for k in range(steps + 1)]
    a = torch.tensor(a, device=device)
    t = _inv_alpha(a, schedule)
    return t, a

