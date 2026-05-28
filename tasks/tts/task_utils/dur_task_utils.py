import torch
from typing import Optional, Tuple, Dict

@torch.no_grad()
def build_dur_ctx_mask(
    dur: torch.LongTensor,                # [B, T_ph]
    ctx_mask: torch.Tensor,               # [B, T_lat] 或 [B, T_lat, 1]
    vae_stride: int,
    ph_lens: Optional[torch.LongTensor] = None,   # [B]
    mel2ph: Optional[torch.LongTensor] = None,     # [B, T_mel], 1-based, 0 表示无效
    mode: str = "ratio",                  # "any" | "all" | "ratio"
    threshold: float = 0.5,
    return_stats: bool = False,
) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    """
    并行构建音素级 dur_ctx_mask（1=上文，0=下文）。

    说明：
    - 若提供 mel2ph，会在 mel 帧上聚合（更精确）。
    - 否则用 dur+stride 将 latent 的 ctx_mask 映射到 mel 帧，并用前缀和计算每个音素区间的 context 帧计数。
    """
    assert mode in {"any", "all", "ratio"}
    device = dur.device
    B, T_ph = dur.shape

    # 规范 ctx_mask -> [B, T_lat], float
    if ctx_mask.dim() == 3 and ctx_mask.size(-1) == 1:
        ctx_mask = ctx_mask.squeeze(-1)
    assert ctx_mask.dim() == 2, "ctx_mask must be [B, T_lat] or [B, T_lat, 1]"
    ctx_mask = ctx_mask.to(device=device, dtype=torch.float32)
    T_lat = ctx_mask.size(1)

    # 有效音素 mask
    if ph_lens is not None:
        ph_lens = ph_lens.to(device)
        ph_valid = torch.arange(T_ph, device=device)[None, :] < ph_lens[:, None]
    else:
        ph_valid = dur > 0

    eps = 1e-6

    if mel2ph is not None:
        # 路径 A：使用 mel2ph 精确聚合
        mel2ph = mel2ph.to(device)
        Bm, T_mel = mel2ph.shape
        assert Bm == B, "mel2ph batch size must match"
        mel_valid = (mel2ph > 0)

        # 计算每个 mel 帧对应的 latent 帧索引，并 gather 得到 mel 级 context 标记
        pos = torch.arange(T_mel, device=device)[None, :].expand(B, -1)         # [B, T_mel]
        lat_idx = (pos // vae_stride).clamp(max=T_lat - 1)                      # [B, T_mel]
        mel_ctx = torch.gather(ctx_mask, dim=1, index=lat_idx).float()          # [B, T_mel]
        mel_ctx = mel_ctx * mel_valid.float()                                   # 无效帧清零

        # 将 mel 级 context 帧计数聚合到音素（1-based -> 0-based）
        p_idx = (mel2ph - 1).clamp(min=0, max=T_ph - 1)                         # [B, T_mel]
        ctx_counts = torch.zeros(B, T_ph, device=device).scatter_add(1, p_idx, mel_ctx)
        dur_counts = torch.zeros(B, T_ph, device=device).scatter_add(
            1, p_idx, mel_valid.float()
        )
    else:
        # 路径 B：无 mel2ph，仅基于 dur + stride，使用前缀和窗口统计
        # 为每条样本构建 mel 位置范围（对齐到 batch 最大 mel 长度）
        mel_len = dur.sum(dim=1)                                                # [B]
        T_mel_max = int(mel_len.max().item()) if B > 0 else 0
        if T_mel_max == 0:
            dur_ctx_mask = torch.zeros_like(dur, dtype=torch.float32, device=device)
            stats = None
            if return_stats:
                stats = {
                    "ctx_counts": torch.zeros_like(dur_ctx_mask),
                    "dur_counts": dur.float(),
                    "ratios": torch.zeros_like(dur_ctx_mask),
                }
            return dur_ctx_mask, stats

        pos = torch.arange(T_mel_max, device=device)[None, :]                   # [1, T_mel_max]
        valid_mel = pos < mel_len[:, None]                                      # [B, T_mel_max]

        # 映射到 latent 索引并得到 mel 级 context 标记
        lat_idx_base = (pos // vae_stride).clamp(max=T_lat - 1)                 # [1, T_mel_max]
        lat_idx = lat_idx_base.expand(B, -1)                                    # [B, T_mel_max]
        mel_ctx = torch.gather(ctx_mask, dim=1, index=lat_idx).float()          # [B, T_mel_max]
        mel_ctx = mel_ctx * valid_mel.float()

        # 计算每个音素的 mel 区间 [start, end)
        csum = torch.cumsum(dur, dim=1)                                         # [B, T_ph]
        end_idx = csum                                                          # 右开边界的右端点索引
        start_idx = csum - dur                                                  # 右开边界的左端点索引
        # 前缀和：prefix[:, j] = sum_{t < j} mel_ctx[:, t]
        prefix = torch.cumsum(
            torch.cat([torch.zeros(B, 1, device=device, dtype=mel_ctx.dtype), mel_ctx], dim=1), dim=1
        )                                                                       # [B, T_mel_max+1]

        # 防止溢出：end/start 在 [0, T_mel_max]
        end_idx = end_idx.clamp(min=0, max=T_mel_max)
        start_idx = start_idx.clamp(min=0, max=T_mel_max)

        # 用前缀和快速区间和：ctx_counts = prefix[end] - prefix[start]
        # 注：gather 需要 long 索引
        ctx_counts = prefix.gather(1, end_idx.long()) - prefix.gather(1, start_idx.long())  # [B, T_ph]
        dur_counts = dur.float()                                                             # [B, T_ph]

    # 比例与判定
    ratios = ctx_counts / (dur_counts.clamp(min=eps))
    if mode == "any":
        dur_ctx_mask = (ctx_counts > 0).float()
    elif mode == "all":
        dur_ctx_mask = (ctx_counts >= (dur_counts - eps)).float()
    else:  # "ratio"
        dur_ctx_mask = (ratios >= threshold).float()

    # 清除无效音素位置
    dur_ctx_mask = dur_ctx_mask * ph_valid.float()

    stats = None
    if return_stats:
        # 统一填充到 [B, T_ph]（已是该形状）
        stats = {
            "ctx_counts": ctx_counts * ph_valid.float(),
            "dur_counts": dur_counts * ph_valid.float(),
            "ratios": ratios * ph_valid.float(),
        }

    return dur_ctx_mask, stats


if __name__ == '__main__':
    vae_stride = 4
    dur = torch.LongTensor([[2, 4, 8, 4]]) * vae_stride
    dur += 1
    ctx_mask = torch.zeros((1, 20))
    ctx_mask[:, :11] = 1
    dur_ctx_mask, _ = build_dur_ctx_mask(
        dur=dur,                              # [B, T_ph]
        ctx_mask=ctx_mask,                    # [B, T_lat] 或 [B, T_lat, 1]
        vae_stride=vae_stride,
        ph_lens=None,  # 若有更准；否则内部用 dur>0
        mel2ph=None,    # 若提供会走更精确路径
        mode='ratio',                         # 'any' / 'all' / 'ratio'
        threshold=0.5,
        return_stats=False,
    )
    print(f"{dur = }")
    print(f"{ctx_mask = }")
    print(f"{dur_ctx_mask = }")
