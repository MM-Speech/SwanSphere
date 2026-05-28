import os
import tempfile
from pathlib import Path

import torch
import soundfile as sf
import torchaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt

from utils.commons.hparams import hparams, set_hparams
from utils.commons.os_utils import kill_void
from utils.commons.ckpt_utils import load_ckpt
from utils.audio.mel import MelNet

from modules.asr.diarization.e2e_model import build_diarization_model

class SpkDiarizationE2EInfer:
    def __init__(self, device, ckpt):
        self.device = device
        self.build_model(ckpt)

    def build_model(self, ckpt):
        if ckpt.endswith('.ckpt'):
            set_hparams(config=os.path.join(Path(ckpt).parent, 'config.yaml'), print_hparams=False)
        else:
            set_hparams(config=os.path.join(ckpt, 'config.yaml'), print_hparams=False)
        self.mel_net = MelNet(hparams)
        self.mel_net.to(self.device)

        self.model = build_diarization_model(hparams)
        load_ckpt(self.model, ckpt, 'model', strict=True, mmap=True)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def forward_model(self, wav):
        wav = torch.from_numpy(wav)[None, ...].to(self.device)
        mel = self.mel_net(wav)
        mel = mel[:, :mel.shape[1] // hparams['frames_multiple'] * hparams['frames_multiple']]
        mel_mask = torch.ones_like(mel[..., 0]).bool()

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = self.model(mel, mel_mask)  # [B, T, K]
        probs = torch.sigmoid(logits)[0].cpu().numpy()

        return probs
    
    def probs_to_segments(
        self,
        probs: np.ndarray,
        threshold: float = 0.5,
        wav_len: int = None,
        sr: int = None,
        renumber_by_first_appearance: bool = True,
        min_duration: float = 0.0,
        min_gap: float = 0.0,
        return_frame_index: bool = False,
    ):
        """
        将 [T, K] 的概率矩阵转为说话人日志（可重叠）。
        - threshold: 判定为活跃的概率阈值
        - renumber_by_first_appearance: 按首次出现时间对通道重新编号，先出现的 spk_id 小
        - min_duration: 丢弃短于该时长的片段（秒）
        - min_gap: 合并片段时允许的最小空隙（秒）
        - return_frame_index: 是否同时返回帧索引（start_frame/end_frame）
        返回:
        - segments: List[dict]，每条包含 {spk_id, start, end}，单位秒
        - spk_map: dict，新 spk_id -> 原始通道索引
        """
        assert probs.ndim == 2, "probs 必须是 [T, K]"
        T, K = probs.shape
        sec_per_frame = hparams['hop_size'] / hparams['audio_sample_rate']
        # 1) 二值化
        mask = probs >= float(threshold)  # [T, K], bool
        # 2) 提取每个通道的连续片段（基于 RLE 边界）
        def frames_to_intervals(frames_bool: np.ndarray):
            # frames_bool: [T]
            x = frames_bool.astype(np.int8)
            # prepend/append 0, 找到 0->1 为 start, 1->0 为 end
            changes = np.diff(x, prepend=0, append=0)
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]  # [start, end) 半开区间
            return starts, ends
        # 暂存原始通道的片段（以帧为单位）
        channel_segments_frames = []
        first_onset = []
        for k in range(K):
            starts, ends = frames_to_intervals(mask[:, k])
            # 可选：按 min_gap 合并相邻片段
            if min_gap > 0:
                merged_starts = []
                merged_ends = []
                if len(starts) > 0:
                    merged_starts.append(starts[0])
                    merged_ends.append(ends[0])
                    gap_thr_frames = int(round(min_gap / sec_per_frame))
                    for i in range(1, len(starts)):
                        # 若相邻片段之间的空隙小于等于阈值，则合并
                        if starts[i] - merged_ends[-1] <= max(1, gap_thr_frames):
                            merged_ends[-1] = ends[i]
                        else:
                            merged_starts.append(starts[i])
                            merged_ends.append(ends[i])
                starts, ends = np.array(merged_starts), np.array(merged_ends)
            # 可选：丢弃时长过短的片段
            if min_duration > 0 and len(starts) > 0:
                dur_thr_frames = int(round(min_duration / sec_per_frame))
                kept = (ends - starts) >= max(1, dur_thr_frames)
                starts, ends = starts[kept], ends[kept]
            channel_segments_frames.append((starts, ends))
            if len(starts) > 0:
                first_onset.append(starts[0])
            else:
                first_onset.append(np.inf)
        # 3) 根据首次出现时间为通道重新编号（先出现的分配更小 spk_id）
        order = np.argsort(np.array(first_onset))
        valid_order = [k for k in order if np.isfinite(first_onset[k])]
        # 对未出现过语音的通道放在后面（通常不需要，但保持一致性）
        rest_order = [k for k in order if not np.isfinite(first_onset[k])]
        full_order = valid_order + rest_order
        spk_map = {}
        for new_id, old_k in enumerate(full_order):
            spk_map[new_id] = int(old_k)
        # 4) 生成最终 segments（单位秒），支持重叠
        segments = []
        for new_id, old_k in spk_map.items():
            starts, ends = channel_segments_frames[old_k]
            if len(starts) == 0:
                continue
            for s, e in zip(starts, ends):
                seg = {
                    "spk_id": int(new_id) if renumber_by_first_appearance else int(old_k),
                    "start": float(s * sec_per_frame),
                    "end": float(e * sec_per_frame),
                }
                if return_frame_index:
                    seg["start_frame"] = int(s)
                    seg["end_frame"] = int(e)
                segments.append(seg)
        # 5) 统一按开始时间排序（若开始相同则按 spk_id）
        segments.sort(key=lambda x: (x["start"], x["spk_id"]))
        return segments, spk_map
    @torch.no_grad()
    def infer(
        self,
        wav: np.ndarray,
        sr: int = None,
        threshold: float = 0.5,
        min_duration: float = 0.0,
        min_gap: float = 0.0,
        return_frame_index: bool = False,
    ):
        """
        端到端接口：输入 wav，返回 probs 与 diarization 段落。
        - sr: 采样率（可选，若 hparams 已提供则可不传）
        """
        probs = self.forward_model(wav)  # [T, K]
        segments, spk_map = self.probs_to_segments(
            probs,
            threshold=threshold,
            wav_len=len(wav),
            sr=sr,
            renumber_by_first_appearance=True,
            min_duration=min_duration,
            min_gap=min_gap,
            return_frame_index=return_frame_index,
        )
        return probs, segments, spk_map

    
if __name__ == '__main__':
    if os.path.isfile('.env.local'):
        from dotenv import load_dotenv
        load_dotenv('.env.local')
    kill_void()

    # ckpt = 'checkpoints/250911_spkdiarization_e2e'
    ckpt = 'checkpoints/250916_spkdiarization_e2e'

    infer_ins = SpkDiarizationE2EInfer('cuda', ckpt)

    # audio, sr = librosa.load('/mnt/bn/lq-ads-aigc/renyi/tts_datasets_bak/InteractiveDialogue/XYZ_20w/chunfeng_download/shard_00000/xyz_00000112_000.bin.wav', sr=24000)
    # audio = audio[:int(sr * 30)]
    
    audio, sr = librosa.load('/mnt/bn/sa-ag-data/liruiqi/code/ScriptSpeech/infer_out/tts_dialogue/demo/250809_scriptspeech_semanticlm_dialogue#250809_scriptspeech_dit_dialogue_textconcat/demo_samples_concat/dzq_enhanced+jay_promptvn+[0]嗨杰伦哥！好久.wav', sr=24000)

    audio = audio[:audio.shape[0] // (32 * 240) * (32 * 240)]

    # probs = infer_ins.forward_model(audio)  # [T, K]
    # 将波形和probs 画到同一个图上


    probs = infer_ins.forward_model(audio)  # [T, K]
    print('probs.shape', probs.shape)
    print('probs', probs)

    time = np.linspace(0, len(audio)/sr, len(audio))  # time axis
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 绘制波形
    ax1.plot(time, audio, 'gray', alpha=0.5, label='speech waveform')
    ax1.set_xlabel("TIME [s]")
    ax1.set_ylabel("Amplitude")
    
    # 创建第二个y轴用于绘制说话人概率
    ax2 = ax1.twinx()
    
    # 将probs插值到与音频相同的时间长度
    probs_interp = np.zeros((time.shape[0], probs.shape[1]))
    for spk in range(probs.shape[1]):
        prob = np.repeat(probs[:, spk], time.shape[0]//probs.shape[0], axis=0)
        prob = np.concatenate([prob, np.zeros(time.shape[0]-prob.shape[0])])
        probs_interp[:, spk] = prob
    
    # 为每个说话人使用不同的颜色绘制概率曲线
    colors = ['r', 'b', 'g', 'c', 'm', 'y']  # 不同说话人使用不同颜色
    for spk in range(probs.shape[1]):
        ax2.plot(time, probs_interp[:, spk], 
                color=colors[spk % len(colors)], 
                label=f'Speaker {spk+1}',
                alpha=0.8)
    
    ax2.set_ylabel("Speaker Probability")
    ax2.set_ylim([-0.01, 1.01])
    
    # 合并两个坐标轴的图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title("Speech Waveform and Speaker Probabilities")
    plt.tight_layout()
    plt.savefig('infer_out/asr/figs/spk_diarization.png')


    segments, spk_map = infer_ins.probs_to_segments(
        probs,
        threshold=0.6,
        wav_len=len(audio),
        sr=sr,
        renumber_by_first_appearance=True,
        min_duration=0.08,
        min_gap=0.08,
        return_frame_index=False,
    )

    for seg in segments:
        print(seg)
        

    # CUDA_VISIBLE_DEVICES=0 python inference/asr/spk_diarization_e2e_infer.py
