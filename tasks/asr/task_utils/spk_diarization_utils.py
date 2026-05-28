import itertools
import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment  # pip install scipy

def pit_bce_loss_bruteforce(
    logits: torch.Tensor,     # [B, T, K]
    targets: torch.Tensor,    # [B, T, K], 0/1 multi-label
    lengths: torch.Tensor = None,  # [B], 有效帧数，可选
    reduction: str = "mean",  # "mean" | "sum" | "none"
    pos_weight: float = None,      # 正类权重（标量），可选
    eps: float = 1e-8
):
    B, T, K = logits.shape
    device = logits.device

    # 有效帧 mask（可选，用于变长/补零的样本）
    if lengths is not None:
        frame_mask = (torch.arange(T, device=device)[None, :] < lengths[:, None]).float()  # [B, T]
        frame_mask = frame_mask.unsqueeze(-1)  # [B, T, 1]
    else:
        frame_mask = None

    perms = list(itertools.permutations(range(K)))  # 全部置换
    perm_losses = []
    
    # 遍历所有置换：对 targets 的 K 维做重排后计算 BCE
    for p in perms:
        tgt_perm = targets[:, :, list(p)]  # [B, T, K]，按置换重排参考
        loss_p = F.binary_cross_entropy_with_logits(
            logits, tgt_perm.float(),
            reduction="none",
            pos_weight=None if pos_weight is None else torch.tensor(pos_weight, device=device)
        )  # [B, T, K]

        if frame_mask is not None:
            loss_p = loss_p * frame_mask  # 广播到 [B, T, K]
            denom = frame_mask.sum(dim=(1, 2)) * K + eps  # 每个样本的有效项个数（T_valid * K）
        else:
            denom = torch.tensor(T * K, device=device, dtype=logits.dtype)

        # 归一化到每个样本
        loss_p = loss_p.sum(dim=(1, 2)) / denom  # [B]
        perm_losses.append(loss_p)

    # [B, num_perm]
    perm_losses = torch.stack(perm_losses, dim=-1)

    # 选取每个样本的最佳置换（最小损失）
    min_loss, best_idx = perm_losses.min(dim=-1)  # [B]
    
    if reduction == "mean":
        return min_loss.mean()
    elif reduction == "sum":
        return min_loss.sum()
    else:  # "none"
        return min_loss, best_idx


def pit_bce_loss_hungarian(
    logits: torch.Tensor,    # [B, T, K]
    targets: torch.Tensor,   # [B, T, K]
    lengths: torch.Tensor = None,  # [B]
    reduction: str = "mean",
    pos_weight: float = None,
    eps: float = 1e-8,
):
    B, T, K = logits.shape
    device = logits.device

    # 准备 mask
    if lengths is not None:
        frame_mask = (torch.arange(T, device=device)[None, :] < lengths[:, None]).float()  # [B, T]
        frame_mask_bt1 = frame_mask.unsqueeze(-1)  # [B, T, 1]
        valid_T = frame_mask.sum(dim=1).clamp_min(1.0)  # [B]
    else:
        frame_mask = None
        valid_T = torch.full((B,), T, device=device, dtype=logits.dtype)

    # 重排维度，便于两两组合
    # logits_bkt: [B, K, T], targets_bkt: [B, K, T]
    logits_bkt = logits.transpose(1, 2).contiguous()
    targets_bkt = targets.transpose(1, 2).contiguous()

    # 两两组合以计算成对代价：expand 到 [B, K_pred, K_ref, T]
    logits_pair = logits_bkt.unsqueeze(2).expand(B, K, K, T)
    targets_pair = targets_bkt.unsqueeze(1).expand(B, K, K, T)

    pair_loss = F.binary_cross_entropy_with_logits(
        logits_pair, targets_pair.float(),
        reduction='none',
        pos_weight=None if pos_weight is None else torch.tensor(pos_weight, device=device)
    )  # [B, K, K, T]

    if lengths is not None:
        mask_b11t = frame_mask[:, None, None, :]  # [B, 1, 1, T]
        pair_loss = pair_loss * mask_b11t

    # 聚合时间维，得到代价矩阵 C: [B, K, K]
    # 使用“每样本有效帧数”的平均，避免不同长度尺度不同
    C = pair_loss.sum(dim=-1) / (valid_T[:, None, None] + eps)  # [B, K, K]

    # 匹配：对每个 batch 样本做一次匈牙利
    assignments = []
    C_cpu = C.detach().cpu().numpy()
    for b in range(B):
        row_ind, col_ind = linear_sum_assignment(C_cpu[b])  # row_ind = [0..K-1]
        # col_ind[k] 给出 预测轨道 k 对应的 参考轨道 索引
        assignments.append(torch.tensor(col_ind, device=device, dtype=torch.long))
    assignments = torch.stack(assignments, dim=0)  # [B, K]

    # 将 targets 按匹配重排： [B, T, K]
    # index: [B, T, K]
    index = assignments[:, None, :].expand(B, T, K)
    targets_perm = torch.gather(targets, dim=2, index=index)

    # 最终逐帧 BCE（带可选 mask）
    final_loss = F.binary_cross_entropy_with_logits(
        logits, targets_perm.float(),
        reduction="none",
        pos_weight=None if pos_weight is None else torch.tensor(pos_weight, device=device)
    )  # [B, T, K]

    if lengths is not None:
        final_loss = final_loss * frame_mask_bt1  # [B, T, K]
        denom = frame_mask_bt1.sum(dim=(1, 2)) * 1.0 * K  # 实际是 T_valid*K
    else:
        denom = torch.full((B,), T * K, device=device, dtype=final_loss.dtype)

    per_sample = final_loss.sum(dim=(1, 2)) / (denom + eps)  # [B]

    if reduction == "mean":
        return per_sample.mean(), assignments
    elif reduction == "sum":
        return per_sample.sum(), assignments
    else:
        return per_sample, assignments

