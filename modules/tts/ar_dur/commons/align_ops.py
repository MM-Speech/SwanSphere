from typing import List, Tuple, Dict, Optional
import math

import torch
import torch.nn.functional as F


def build_word_mask(x2word, y2word):
    return (x2word[:, :, None] == y2word[:, None, :]).long()


def mel2ph_to_mel2word(mel2ph, ph2word):
    mel2word = (ph2word - 1).gather(1, (mel2ph - 1).clamp(min=0)) + 1
    mel2word = mel2word * (mel2ph > 0).long()
    return mel2word


def clip_seq_to_multiple(seq, frames_multiple):
    if seq.shape[1] % frames_multiple > 0:
        max_frames = seq.shape[1] // frames_multiple * frames_multiple
        seq = seq[:, :max_frames]
    return seq


def extend_to_multiple(seq, multiple):
    remain = seq.shape[1] % multiple
    if remain > 0:
        remain = multiple - remain
        seq = torch.cat([seq] + [seq[:, -1:].detach()] * remain, 1)
    return seq


def expand_states(h, mel2token):
    h = F.pad(h, [0, 0, 1, 0])
    mel2token_ = mel2token[..., None].repeat([1, 1, h.shape[-1]])
    h = torch.gather(h, 1, mel2token_)  # [B, T, H]
    return h


def _eff_min_keep(d: int, min_keep: int, keep_ratio: Optional[float]) -> int:
    """计算某 token 的有效最小保留帧数。"""
    k = min_keep
    if keep_ratio is not None:
        k = max(k, math.ceil(d * keep_ratio))
    return min(d, max(0, k))

def compute_mel2aug_from_dur(
    dur: List[int],
    gap_mode: str = "fixed",         # "fixed" | "proportional"
    gap_frames: int = 2,             # fixed 模式下的每边界目标空隙帧数
    gap_alpha: float = 0.15,         # proportional 模式下 g_i = round(alpha * (d_i + d_{i+1}))
    min_keep: int = 1,               # 每个 token 训练后至少保留帧数
    keep_ratio: Optional[float] = None, # 也可设为如 0.1 表示至少保留 10%
    symmetric: bool = True           # True: 尽量对称从左右两侧挪帧
) -> torch.LongTensor:
    """
    基于 token 时长 dur，在相邻 token 之间插入“空隙帧”，并返回 mel2aug。
    语义：0=gap，i=第 i 个 token（1..L）
    """
    L = len(dur)
    # 2) 计算每个边界的目标空隙帧数 g[i]，i in [0..L-2]
    g = [0] * max(0, L - 1)
    for i in range(L - 1):
        if gap_mode == "fixed":
            g[i] = int(max(0, gap_frames))
        elif gap_mode == "proportional":
            g[i] = int(max(0, round(gap_alpha * (dur[i] + dur[i + 1]))))
        else:
            raise ValueError(f"Unknown gap_mode: {gap_mode}")
    # 3) 为每个 token 分配“可挪出的预算”
    # avail[i] 表示 token i 还能让出的帧数（两侧边界合计），保证最终至少留下 eff_min_keep[i]
    eff_min_keeps = [_eff_min_keep(d, min_keep, keep_ratio) for d in dur]
    avail = [max(0, d - k) for d, k in zip(dur, eff_min_keeps)]
    # 4) 给每个边界 i 分配左右挪帧 l_take[i], r_take[i]
    # l_take[i]: 从左侧 token i 的末尾挪走的帧数
    # r_take[i]: 从右侧 token i+1 的开头挪走的帧数
    l_take = [0] * max(0, L - 1)
    r_take = [0] * max(0, L - 1)
    for i in range(L - 1):
        need = g[i]
        if need <= 0:
            continue
        # 初始按对称分配
        if symmetric:
            l = min((need + 1) // 2, avail[i])
            r = min(need - l, avail[i + 1])
            # 若右侧不够，再尝试从左侧补
            deficit = need - (l + r)
            if deficit > 0:
                extra_l = min(deficit, avail[i] - l)
                l += extra_l
                deficit -= extra_l
            # 若还缺，再尽量从右侧补（通常已经为 0）
            if deficit > 0:
                extra_r = min(deficit, avail[i + 1] - r)
                r += extra_r
        else:
            # 示例：优先从右侧挪
            r = min(need, avail[i + 1])
            l = min(need - r, avail[i])
        # 最终可挪出的总量
        total = l + r
        if total <= 0:
            continue
        l_take[i] = l
        r_take[i] = r
        avail[i] -= l
        avail[i + 1] -= r
    # 5) 构造基础 mel2ph（1..L 的位置索引重复 dur[i] 次）
    T = sum(dur)
    mel2aug = torch.empty(T, dtype=torch.long)
    # 起止位置
    starts = [0] * L
    acc = 0
    for i in range(L):
        starts[i] = acc
        mel2aug[acc:acc + dur[i]] = i + 1  # 1-based
        acc += dur[i]
    # ends
    ends = [starts[i] + dur[i] - 1 for i in range(L)]
    # 6) 将边界处需挪出的帧置为 0（gap）
    # 对 token i：
    # - 左边界（i-1,i）的 r_take[i-1] 个帧从开头挪走 → s_i ... s_i + r_take[i-1]-1 = 0
    # - 右边界（i,i+1）的 l_take[i]   个帧从末尾挪走 → e_i - l_take[i] +1 ... e_i = 0
    for i in range(L):
        # 左侧
        left_remove = r_take[i - 1] if i - 1 >= 0 else 0
        if left_remove > 0:
            s = starts[i]
            mel2aug[s:s + left_remove] = 0
        # 右侧
        right_remove = l_take[i] if i < L - 1 else 0
        if right_remove > 0:
            e = ends[i]
            mel2aug[e - right_remove + 1: e + 1] = 0
    return mel2aug


if __name__ == '__main__':
    from modules.tts.ar_dur.commons.nar_tts_modules import LengthRegulator
    length_regulator = LengthRegulator()

    dur = [5, 8, 10,  9, 11,  4, 14]
    token = torch.Tensor([10, 20, 30, 40, 50, 60, 70])[None, :, None]

    mel2ph = length_regulator(torch.LongTensor(dur)[None])

    mel2aug = compute_mel2aug_from_dur(
        dur,
        gap_mode='fixed',
        # gap_mode='proportional',
        gap_frames=4,
        gap_alpha=0.2,
        min_keep=1,
        keep_ratio=None,
        symmetric=True
    )

    print(f"{mel2ph = }")
    print(f"{mel2aug = }")

    # during training

    dur = [5, 8, 10,  9, 11,  4, 14, 0]
    token = torch.Tensor([10, 20, 30, 40, 50, 60, 70, -1])[None, :, None]
    mel2aug = torch.cat([mel2aug, torch.zeros(5).long()])[None, :]

    mel2aug[:, :-5] += 1
    token = torch.Tensor([-10, 10, 20, 30, 40, 50, 60, 70, -1])[None, :, None]

    print(f"{mel2aug = }")

    token_expand = expand_states(token, mel2aug)
    print(f"{token_expand.shape = }")

    print(f"{token_expand.squeeze() = }")
    
