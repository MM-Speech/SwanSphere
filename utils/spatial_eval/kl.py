import os
import torch
import librosa
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torchaudio
# 假设代理运行在本地 localhost (127.0.0.1)
# os.environ['http_proxy'] = 'http://127.0.0.1:7990'
# os.environ['https_proxy'] = 'http://127.0.0.1:7990'
# 引入 PaSST
try:
    from hear21passt.base import get_basic_model
except ImportError:
    print("请先安装 PaSST: pip install hear21passt")
    exit()

def load_opus(path):
    """
    读取 Opus 文件，返回 Tensor [Channels, Time]
    """
    try:
        waveform, sr = librosa.load(path, sr=None, mono=False)
        waveform = torch.from_numpy(waveform)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        return waveform, sr
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None, None

class KLEvaluatorPaSST:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"KL Evaluator initialized on {self.device} using PaSST")
        
        # PaSST 强制要求 32kHz
        self.target_sr = 32000
        
        # 1. 下载并加载 PaSST 模型
        # get_basic_model 会自动下载 passt_s_pw_m_mAP_295.pt
        print("Loading PaSST model...")
        self.model = get_basic_model(mode="logits")
        self.model.eval()
        self.model.to(self.device)

    def get_label_distribution(self, waveform, sr):
        """
        输入音频，输出 AudioSet 527 个类别的概率分布
        """
        # 1. 重采样到 32k (PaSST 必须)
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr).to(self.device)
            # 确保 waveform 也在 device 上
            if waveform.device != self.device:
                waveform = waveform.to(self.device)
            waveform = resampler(waveform)
        else:
             if waveform.device != self.device:
                waveform = waveform.to(self.device)

        # 2. Mixdown 到单声道
        # PaSST 也是在单声道 AudioSet 上训练的
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # 3. 维度调整
        # PaSST 期望输入 shape: [Batch, Time]
        # 目前是 [1, Time]，不需要 squeeze，直接传即可，因为它会自动处理 batch 维度
        
        with torch.no_grad():
            # PaSST 推理
            # mode="logits" 时，返回 (Batch, 527)
            logits = self.model(waveform)
            
            # PaSST 输出的是 logits，需要 Softmax 转为概率分布
            probs = F.softmax(logits, dim=1)
        
        return probs

    def compute_kl_divergence(self, gen_dir, gt_dir):
        """
        计算整个数据集的平均 KL 散度
        """
        valid_exts = ('.opus', '.wav', '.flac')
        gen_map = {os.path.splitext(f)[0]: f for f in os.listdir(gen_dir) if f.endswith(valid_exts)}
        gt_map = {os.path.splitext(f)[0]: f for f in os.listdir(gt_dir) if f.endswith(valid_exts)}
        common_stems = sorted(list(set(gen_map.keys()) & set(gt_map.keys())))
        
        if not common_stems:
            print("No common files found.")
            return None

        print(f"Evaluating KL Divergence (PaSST) on {len(common_stems)} files...")
        
        total_kl = 0.0
        count = 0
        eps = 1e-8
        
        for stem in tqdm(common_stems):
            gt_path = os.path.join(gt_dir, gt_map[stem])
            gen_path = os.path.join(gen_dir, gen_map[stem])
            
            # 加载音频 (CPU)
            wav_gt, sr_gt = load_opus(gt_path)
            wav_gen, sr_gen = load_opus(gen_path)
            
            if wav_gt is None or wav_gen is None:
                continue
            
            # --- [新增] 强制长度对齐 (截断长尾巴) ---
            # load_opus 返回的是 [Channels, Time] 或 [1, Time]
            min_len = min(wav_gt.shape[-1], wav_gen.shape[-1])
            wav_gt = wav_gt[..., :min_len]
            wav_gen = wav_gen[..., :min_len]
            
            try:
                # 获取概率分布
                P = self.get_label_distribution(wav_gt, sr_gt)   # GT
                Q = self.get_label_distribution(wav_gen, sr_gen) # Gen
                
                # 计算 KL: sum( P * (log P - log Q) )
                log_P = torch.log(P + eps)
                log_Q = torch.log(Q + eps)
                
                # batch size = 1, 取 item
                kl_value = torch.sum(P * (log_P - log_Q)).item()
                
                total_kl += kl_value
                count += 1
                
            except Exception as e:
                print(f"Error calculating KL for {stem}: {e}")
                # 可能是显存爆了或者音频过短
                if "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()
                
        if count == 0:
            return None
            
        avg_kl = total_kl / count
        
        print(f"\n========================================")
        print(f"PaSST KL Divergence: {avg_kl:.4f}")
        print(f"========================================")
        
        return avg_kl

# --- 使用示例 ---
if __name__ == "__main__":
    GEN_DIR = "/home/leike/spatial/ScriptSpeech/test/gene"
    GT_DIR = "/home/leike/spatial/ScriptSpeech/test/gt"
    
    # 第一次运行会自动下载模型
    evaluator = KLEvaluatorPaSST(device='cuda')
    kl_score = evaluator.compute_kl_divergence(GEN_DIR, GT_DIR)