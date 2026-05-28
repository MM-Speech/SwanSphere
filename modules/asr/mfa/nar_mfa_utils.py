import numpy as np
import torch
import torch.nn.functional as F
from utils.nn.generation_utils import stochastic_round

def dur_to_paraformer_label(dur):
    if isinstance(dur, np.ndarray) or isinstance(dur, list):
        if isinstance(dur, list):
            dur = np.array(dur)
        assert dur.ndim == 1
        d = np.clip(dur.astype(np.int64), 0, None)
        weights = 1.0 / np.clip(d, 1, None).astype(np.float32)
        labels = np.repeat(weights, d).astype(np.float32)
        return labels
    elif isinstance(dur, torch.Tensor):
        assert dur.dim() == 1
        d = dur.to(torch.long).clamp(min=0)
        weights = 1.0 / d.clamp(min=1).float()
        labels = torch.repeat_interleave(weights, d, dim=0)
        return labels

def aggregate_frame_by_dur(x: torch.Tensor, dur: torch.Tensor):
    """
    将帧级 x [B, T, C] 按 dur 聚合为音素级和 [B, T_dur, C]（分段求和）。
    参数:
      x: FloatTensor [B, T, C]，帧级特征（已 padding）
      dur:    Long/Int Tensor [B, T_dur]，每个音素的时长(>=0)，padding 为 0
    返回:
      agg:            FloatTensor [B, T_dur, C]，音素段内特征求和
      phoneme_mask:   BoolTensor   [B, T_dur]，True 表示 dur>0 的有效音素
      T_per_sample:   LongTensor   [B]，每个样本的有效帧总数
    """
    is_2d = False
    if x.dim() == 2:
        x = x[..., None]
        is_2d = True
    assert x.dim() == 3 and dur.dim() == 2 and x.size(0) == dur.size(0)
    device = x.device
    B, T, C = x.shape
    _, T_dur = dur.shape

    # 1) 预处理与帧级有效 mask
    d = dur.to(torch.long).clamp(min=0)            # [B, T_dur]
    phoneme_mask = d > 0
    T_per_sample = d.sum(dim=1)                    # [B]
    arange_T = torch.arange(T, device=device).unsqueeze(0)       # [1, T]
    frame_mask = arange_T < T_per_sample.unsqueeze(1)            # [B, T] bool

    # 2) 选出所有有效帧 x（按 batch 内时间顺序串接）
    valid_x = x.reshape(B * T, C)[frame_mask.reshape(B * T)]  # [sum(T_b), C]

    # 3) 构造 (b, phoneme) 的索引，长度与有效帧数一致（与 valid_x 对齐）
    base_phoneme_ids = torch.arange(T_dur, device=device).expand(B, T_dur).reshape(-1)  # [(B*T_dur)]
    base_batch_ids   = torch.arange(B, device=device).unsqueeze(1).expand(B, T_dur).reshape(-1)  # [(B*T_dur)]
    repeats = d.reshape(-1)                                                            # [(B*T_dur)]
    p_ids = torch.repeat_interleave(base_phoneme_ids, repeats)  # [sum(T_b)]
    b_ids = torch.repeat_interleave(base_batch_ids, repeats)    # [sum(T_b)]

    # 4A) 用 scatter_add_（二维目标，沿 dim=0 累加）
    # agg_flat = torch.zeros(B * T_dur, C, dtype=x.dtype, device=device)            # [B*T_dur, C]
    # flat_ids = (b_ids * T_dur + p_ids).unsqueeze(1).expand(-1, C)                      # [sum(T_b), C]
    # agg_flat.scatter_add_(dim=0, index=flat_ids, src=valid_x)                     # 累加到行
    # agg = agg_flat.view(B, T_dur, C)                                                   # [B, T_dur, C]

    # 4B) 等价写法：index_put_（有时更直观，性能相近）
    agg = torch.zeros((B, T_dur, C), dtype=x.dtype, device=device)
    agg.index_put_((b_ids, p_ids), valid_x, accumulate=True)
    
    if is_2d:
        agg = agg[..., 0]

    # return agg, phoneme_mask, T_per_sample
    return agg, phoneme_mask

def cif_durations_frames(alpha: torch.Tensor) -> torch.Tensor:
    """
    alpha: [B, T] in (0,1), 通常为sigmoid后的CIF权重
    返回:
      dur: [B, T_dur] 每个样本的段时长(帧数)，按顺序排列，超出真实段数处填0
           其中 T_dur = max(K_i) 为该batch的最大段数
    说明:
      - 只统计已“发射”的完整段（Σalpha跨过整数产生的K个段）。
      - 尾部未满的残段不计入（若需要可另行处理）。
    """
    assert alpha.dim() == 2, "alpha must be [B, T]"
    B, T = alpha.shape
    device = alpha.device

    # 累计和与“发射事件”
    S = torch.cumsum(alpha, dim=1)                 # [B, T]
    S_prev = S - alpha
    S_floor      = torch.floor(S)
    S_prev_floor = torch.floor(S_prev)
    emit = (S_floor > S_prev_floor)                # [B, T] bool

    # 每条样本的段数K，以及本batch的最大段数T_dur
    K = emit.sum(dim=1)                            # [B]
    T_dur = int(K.max().item()) if B > 0 else 0

    if T_dur == 0:
        return torch.zeros((B, 0), device=device, dtype=torch.long)

    # 事件帧索引（并行获取K个事件，升序）
    t_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # [B, T]
    pos = (t_idx + 1) * emit.int()             # 事件处为1..T，非事件为0
    topk_vals, _ = torch.topk(pos, k=T_dur, dim=1, largest=True, sorted=True)
    t_close = torch.sort(topk_vals, dim=1).values - 1                 # [B, T_dur], -1为无效填充

    # 起始帧与结束帧（离散）
    t_close_safe = t_close.clamp_min(0)
    starts = torch.zeros_like(t_close_safe, dtype=torch.long)         # [B, T_dur]
    starts[:, 1:] = t_close_safe[:, :-1] + 1
    ends = t_close                                                    # [B, T_dur]

    # 有效mask
    arange_k = torch.arange(T_dur, device=device).unsqueeze(0)        # [1, T_dur]
    valid = arange_k < K.unsqueeze(1)                                 # [B, T_dur]

    # 段时长（帧）
    dur = (ends - starts + 1).clamp_min(0)                            # [B, T_dur]
    dur = dur * valid.long()                                          # 无效位置置0
    dur_mask = dur > 0
    return dur, dur_mask


def build_alignment_from_durations(dur: torch.Tensor, dur_mask: torch.Tensor, ignore_index: int = -100):
    """
    输入:
      - dur: [B, T_txt] 每个 token 的时长（帧数），可为 0
      - dur_mask: [B, T_txt] True 表示该 token 有效（非 padding）
    输出:
      - align: [B, Tm_max, T_txt] float，one-hot 对齐矩阵（无效帧/无效列为 0）
      - frame_labels: [B, Tm_max] long，每帧的 token 索引（无效帧为 ignore_index）
      - mel_mask: [B, Tm_max] bool，True 表示有效帧
      - mel_len: [B] long，每条样本的总帧数（sum of dur over valid tokens）
    说明:
      - 完全并行，无 for 循环；复杂度 O(B * Tm_max + B * T_txt + B * Tm_max * log T_txt)。
      - 如果 T_txt 很大且需要更低复杂度，可切换到方法2。
    """
    device = dur.device
    B, Tt = dur.shape

    dur_valid = (dur * dur_mask.float())  # [B, Tt]
    # ends 有效部分为累加和；对无效列填总长度，保证非递减
    ends_valid = torch.cumsum(dur_valid, dim=1)               # [B, Tt]
    total_len = ends_valid[:, -1]                             # [B]
    ends = torch.where(dur_mask, ends_valid, total_len[:, None])  # [B, Tt]

    # 最大帧长度用于对齐到统一长度
    Tm_max = int(total_len.max().item())
    frame_idx = torch.arange(Tm_max, device=device)[None, :].expand(B, Tm_max)  # [B, Tm_max]
    mel_mask = frame_idx < total_len[:, None]  # [B, Tm_max] True=有效帧

    # searchsorted: 对每个 batch，返回每帧对应的 token 索引 i
    # right=True => 插入点在右侧，刚好对应 [start_i, end_i) 的分段
    token_idx = torch.searchsorted(ends, frame_idx.contiguous(), right=True)  # [B, Tm_max]
    # 无效帧标记为 ignore_index
    frame_labels = torch.where(mel_mask, token_idx, torch.full_like(token_idx, ignore_index))

    # one-hot -> [B, Tm_max, Tt]
    align = F.one_hot(torch.clamp(token_idx, 0, Tt - 1), num_classes=Tt).float()  # [B, Tm_max, Tt]
    # 屏蔽无效帧与无效 token 列
    align = align * mel_mask[:, :, None].float() * dur_mask[:, None, :].float()

    return {
        "align": align,                 # [B, Tm_max, Tt] one-hot
        "frame_labels": frame_labels,   # [B, Tm_max] 帧级标签
        "mel_mask": mel_mask,           # [B, Tm_max]
        "mel_len": total_len.long(),    # [B]
        "token_idx": token_idx          # [B, Tm_max]
    }


def build_alignment_broadcast(dur: torch.Tensor, dur_mask: torch.Tensor):
    """
    返回:
      - align: [B, Tm_max, T_txt] float (0/1)
      - frame_labels: [B, Tm_max] long
      - mel_mask: [B, Tm_max] bool
      - mel_len: [B] long
    复杂度 O(B * Tm_max * T_txt)，显存开销较大，但完全并行。
    """
    device = dur.device
    B, Tt = dur.shape

    dur_valid = dur * dur_mask.float()
    starts = torch.cumsum(dur_valid, dim=1) - dur_valid      # [B, Tt]
    ends = starts + dur_valid                                # [B, Tt]
    total_len = dur_valid.sum(dim=1).long()                  # [B]
    Tm_max = int(total_len.max().item())

    frame_idx = torch.arange(Tm_max, device=device)[None, :, None]    # [B?, Tm_max, 1]
    frame_idx = frame_idx.expand(B, Tm_max, 1)                         # [B, Tm_max, 1]
    starts_b = starts[:, None, :]                                      # [B, 1, Tt]
    ends_b = ends[:, None, :]                                          # [B, 1, Tt]
    mask_b = dur_mask[:, None, :].bool()                               # [B, 1, Tt]

    align_bool = (frame_idx >= starts_b) & (frame_idx < ends_b) & mask_b  # [B, Tm_max, Tt]
    align = align_bool.float()

    mel_mask = (torch.arange(Tm_max, device=device)[None, :].expand(B, Tm_max) < total_len[:, None])
    # 帧级标签（one-hot 的 argmax），无效帧设为 -100
    frame_labels = torch.where(
        mel_mask,
        align.argmax(dim=-1),
        torch.full((B, Tm_max), -100, dtype=torch.long, device=device)
    )
    # 对不合法的 (全 0 行) 保持为 0，若需严格 1-of-K 可用 mel_mask 约束保证 sum 列=dur
    return {
        "align": align,
        "frame_labels": frame_labels,
        "mel_mask": mel_mask,
        "mel_len": total_len,
    }


def expected_index_one_hot(probs: torch.Tensor, dim: int = -1, rounding: str = "stochastic", hard=False):
    """
    从按 dim 做 softmax 的概率张量 probs 生成在 dim 上为 one-hot 的张量，
    其中 one-hot 的 1 落在按分布的期望索引处。

    参数:
      - probs: 形状 [..., T, ...]，在 dim 上为概率（已 softmax）
      - dim: 计算期望与生成 one-hot 的轴，默认为最后轴
      - rounding: "nearest" | "floor" | "ceil"，期望位置到整数索引的映射方式

    返回:
      - out: 与 probs 同形状的 one-hot（dtype 同 probs）
      - idx: 期望位置取整后的整数索引（去掉 dim 后的形状）
      - mu: 期望位置（浮点，去掉 dim 后的形状）
    """
    # T 是该轴长度
    T = probs.shape[dim]
    dtype = probs.dtype
    device = probs.device
    
    if hard:
        # 直接取 argmax 索引（ties 时返回第一个最大值的索引）
        idx = torch.argmax(probs, dim=dim)
        out = torch.zeros_like(probs)
        out.scatter_(dim, idx.unsqueeze(dim), 1.0)
        mu = None
        # return out.to(dtype), idx, mu
        return out.to(dtype)

    # 构造位置向量并广播到 probs 的形状以便逐元素乘
    pos = torch.arange(T, device=device, dtype=dtype)
    # 视图到形状 [1, 1, ..., T, ..., 1]
    view_shape = [1] * probs.ndim
    view_shape[dim] = T
    pos = pos.view(*view_shape)

    # 期望位置 μ = sum(p * t)
    mu = (probs * pos).sum(dim=dim)

    # 将期望位置映射到索引
    if rounding == 'stochastic':
        idx = stochastic_round(mu)
    elif rounding == "nearest":
        idx = mu.round()
    elif rounding == "floor":
        idx = torch.floor(mu)
    elif rounding == "ceil":
        idx = torch.ceil(mu)
    else:
        raise ValueError(f"Unknown rounding: {rounding}")

    # 边界裁剪
    idx = idx.clamp(0, T - 1)
    
    # 对于原始 mu 为 NaN 的位置，回退到 argmax，以避免系统性落到 0
    nan_mask = ~torch.isfinite(mu)
    if nan_mask.any():
        fallback_idx = torch.argmax(probs, dim=dim)
        idx = torch.where(nan_mask, fallback_idx, idx.to(torch.long))
    else:
        idx = idx.to(torch.long)

    # 生成 one-hot（使用 scatter 保持维度位置不变）
    out = torch.zeros_like(probs)
    out.scatter_(dim, idx.unsqueeze(dim), 1.0)

    # return out.to(dtype), idx, mu
    return out.to(dtype)




if __name__ == '__main__':
    B, Tt = 2, 6
    dur = torch.tensor([
        [3, 0, 2, 1, 0, 0],  # 总计 6 帧
        [0, 2, 0, 0, 4, 1],  # 总计 7 帧
    ], dtype=torch.long)
    dur_mask = torch.tensor([
        [1, 1, 1, 1, 0, 0],  # 后两列 padding
        [1, 1, 1, 1, 1, 1],
    ], dtype=torch.bool)

    out1 = build_alignment_from_durations(dur, dur_mask)
    out2 = build_alignment_broadcast(dur, dur_mask)

    print("Tm_max:", out1["align"].shape[1], out2["align"].shape[1])
    print("sum over tokens equals total frames (sample 0):", out1["align"][0].sum().item(), out1["mel_len"][0].item())
    print("frame_labels[0][:10]:", out1["frame_labels"][0][:10])
