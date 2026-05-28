import numpy as np
from typing import List, Tuple, Sequence, Optional, Union

def apply_gain_db(x: np.ndarray, gain_db: float) -> np.ndarray:
    """
    以 dB 应用线性增益（不改变 dtype）。
    """
    return x * (10.0 ** (gain_db / 20.0))

def true_peak_dbfs(x: np.ndarray, sr: int, oversample: int = 4) -> float:
    """
    近似 True Peak（dBTP）：按通道上采样 resample_poly 检测插值峰值。
    oversample=4~8 通常够用。
    """
    from scipy.signal import resample_poly
    if x.size == 0:
        return -np.inf
    if x.ndim == 1:
        y = resample_poly(x, up=oversample, down=1, axis=0)
        m = float(np.max(np.abs(y)))
    else:
        m = 0.0
        for ch in range(x.shape[1]):
            y = resample_poly(x[:, ch], up=oversample, down=1, axis=0)
            m = max(m, float(np.max(np.abs(y))))
    return -np.inf if m == 0.0 else 20.0 * np.log10(m)

def normalize_lufs(
    x: np.ndarray,
    sr: int,
    target_lufs: float = -16.0,
    tp_limit_db: Optional[float] = -1.0,
    oversample: int = 4,
    meter=None,
) -> np.ndarray:
    """
    LUFS/EBU R128 归一化（不启用压缩，仅线性增益 + True Peak 安全限制）。
    - target_lufs: 立体声常用 -16 LUFS；单声道 -19 LUFS。
    - tp_limit_db: True Peak 上限（dBTP），默认 -1 dBTP；设为 None 则不限制。
    策略：计算达到目标 LUFS 所需增益；若会超过 True Peak 上限，则以 TP 为准。
    """
    import pyloudnorm as pyln
    if meter is None:
        meter = pyln.Meter(sr)  # EBU R128, K-weighting + gating
    loud = meter.integrated_loudness(x)
    if not np.isfinite(loud):
        # 全静音或极短片段时，LUFS 不稳定：原样返回
        return x.copy()
    # 到达目标 LUFS 所需增益
    gain_lufs_db = target_lufs - loud
    if tp_limit_db is None:
        return apply_gain_db(x, gain_lufs_db)
    # 基于原信号的 True Peak 计算“最大允许增益”
    orig_tp_db = true_peak_dbfs(x, sr, oversample=oversample)
    max_allowed_gain_db = tp_limit_db - orig_tp_db  # 正值表示还能抬这么多
    # 实际增益 = 受限于 TP 的较小值（更保守）
    gain_db = min(gain_lufs_db, max_allowed_gain_db)
    y = apply_gain_db(x, gain_db)
    return y

def batch_resample(
    wavs: List[np.ndarray],
    sample_rates: Union[List[int], int],
    tgt_sr: int,
    resamplers=None,
    batch_size=16,
    batch_duration=300,
    use_batch_by_size=True,
    device='cpu'
):
    import torch, torchaudio, math
    from utils.commons.dataset_utils import collate_xd, batch_by_size
    if sample_rates == tgt_sr:
        return wavs
    if resamplers is None:
        resamplers = {}
    if isinstance(sample_rates, int) or isinstance(sample_rates, float):
        if len(wavs) > batch_size:
            if use_batch_by_size:
                wav_lengths = [len(wav) for wav in wavs]
                ordered_idxs = np.argsort(wav_lengths)
                batches = batch_by_size(
                    ordered_idxs, lambda idx: wav_lengths[idx],
                    max_tokens=batch_duration * sample_rates,
                    max_sentences=batch_size,
                )
                wavs_ = [None] * len(wavs)
                for batch in batches:
                    res = batch_resample([wavs[idx] for idx in batch], sample_rates, tgt_sr, resamplers, 
                                         batch_size, batch_duration, use_batch_by_size, device)
                    for idx, wav in zip(batch, res):
                        wavs_[idx] = wav
            else:
                n_batch = math.ceil(len(wavs) / batch_size)
                wavs_ = []
                for batch_i in range(n_batch):
                    res = batch_resample(wavs[batch_i * batch_size: (batch_i + 1) * batch_size], 
                                        sample_rates, tgt_sr, resamplers, batch_size, device)
                    wavs_.extend(res)
            wavs = wavs_
        else:
            wav_lengths = torch.LongTensor([len(w) for w in wavs])
            wavs = collate_xd([torch.from_numpy(w).float() for w in wavs]).to(device)
            if sample_rates != tgt_sr:
                if sample_rates not in resamplers:
                    resamplers[sample_rates] = torchaudio.transforms.Resample(orig_freq=sample_rates, new_freq=tgt_sr).to(device)
                with torch.no_grad():
                    wavs = resamplers[sample_rates](wavs)
                wav_lengths = (wav_lengths / sample_rates * tgt_sr).long()
            wavs = [wavs[i, :wav_lengths[i]].cpu().numpy() for i in range(len(wav_lengths))]
    elif isinstance(sample_rates, list):
        sr2wav_idx = {}
        for wav_idx, sr in enumerate(sample_rates):
            if sr in sr2wav_idx:
                sr2wav_idx[sr].append(wav_idx)
            else:
                sr2wav_idx[sr] = [wav_idx]
        wavs_ = [None] * len(wavs)
        for sr in sr2wav_idx:
            wav_idxs = sr2wav_idx[sr]
            res = batch_resample([wavs[idx] for idx in wav_idxs], sr, tgt_sr, resamplers, 
                                 batch_size, batch_duration, use_batch_by_size, device)
            for idx, wav in zip(wav_idxs, res):
                wavs_[idx] = wav
        wavs = wavs_

    return wavs


