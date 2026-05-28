import os
import torch
import librosa
import torchaudio
import numpy as np
import scipy.linalg
import torchopenl3
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

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

class FDEvaluator:
    def __init__(self, target_sr=48000, device='cuda'):
        self.target_sr = target_sr
        # 1. 强制检查 GPU
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Evaluator initialized on {self.device}. Target SR: {target_sr}Hz")
        
        # 2. [关键优化] 预先加载模型并扔到 GPU 上
        # 这样就不用对每个文件都重新加载模型了，速度提升巨大
        print("Loading OpenL3 Model to GPU...")
        self.model = torchopenl3.models.load_audio_embedding_model(
            input_repr="mel256", 
            content_type="env", 
            embedding_size=512
        )
        self.model.to(self.device)
        self.model.eval() # 开启评估模式

    def get_dir_stats(self, dir_path, file_list):
        all_embeddings = []
        print(f"Processing {len(file_list)} files in {dir_path}...")
        
        for fname in tqdm(file_list):
            path = os.path.join(dir_path, fname)
            
            waveform, sr = load_opus(path)
            if waveform is None:
                continue
            
            # 3. [关键步骤] 把数据搬运到 GPU
            waveform = waveform.to(self.device)

            # 重采样 (Resampling)
            if sr != self.target_sr:
                # Resample 也是 nn.Module，需要放到 GPU 上运行
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr).to(self.device)
                waveform = resampler(waveform)
            
            assert waveform.size(0) == 4, "wrong shape"
            # Mixdown
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            try:
                # 4. 传入预加载的 model
                # 此时 waveform 在 GPU，model 在 GPU，速度全开
                emb, ts = torchopenl3.get_audio_embedding(
                    waveform, 
                    self.target_sr,
                    model=self.model, # <--- 传入 GPU 模型
                    batch_size=32,
                    verbose=False
                )
                
                # 结果转回 CPU 存成 numpy
                if isinstance(emb, torch.Tensor):
                    emb = emb.cpu().detach().numpy()
                
                if emb.ndim == 3:
                    emb = emb.reshape(-1, emb.shape[-1])
                
                all_embeddings.append(emb)
                
            except Exception as e:
                print(f"Feature extraction failed for {fname}: {e}")

        if not all_embeddings:
            return None, None
            
        all_embeddings = np.vstack(all_embeddings)
        mu = np.mean(all_embeddings, axis=0)
        sigma = np.cov(all_embeddings, rowvar=False)
        return mu, sigma

    # ... (calculate_frechet_distance 函数保持不变) ...
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        diff = mu1 - mu2
        covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                pass 
            covmean = covmean.real
        tr_covmean = np.trace(covmean)
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

# ... (calculate_fd_score 和 main 部分保持不变) ...
def calculate_fd_score(gen_dir, gt_dir):
    # (复制之前的逻辑，不用变)
    valid_exts = ('.opus', '.wav', '.flac', '.mp3')
    gen_map = {os.path.splitext(f)[0]: f for f in os.listdir(gen_dir) if f.endswith(valid_exts)}
    gt_map = {os.path.splitext(f)[0]: f for f in os.listdir(gt_dir) if f.endswith(valid_exts)}
    common_stems = sorted(list(set(gen_map.keys()) & set(gt_map.keys())))
    
    if len(common_stems) == 0:
        print("Error: No common file stems found.")
        return None
    
    print(f"Found {len(common_stems)} common pairs. Starting GPU evaluation...")
    gen_files_list = [gen_map[stem] for stem in common_stems]
    gt_files_list = [gt_map[stem] for stem in common_stems]
    
    evaluator = FDEvaluator() # 默认使用 CUDA
    
    print(">>> Computing Ground Truth Stats...")
    mu_gt, sigma_gt = evaluator.get_dir_stats(gt_dir, gt_files_list)
    
    print(">>> Computing Generated Stats...")
    mu_gen, sigma_gen = evaluator.get_dir_stats(gen_dir, gen_files_list)
    
    if mu_gt is None or mu_gen is None:
        return None
        
    print(">>> Calculating FD...")
    fd_score = evaluator.calculate_frechet_distance(mu_gt, sigma_gt, mu_gen, sigma_gen)
    print(f"\n========================================")
    print(f"Fréchet Distance (FD): {fd_score:.4f}")
    print(f"========================================")
    return fd_score

if __name__ == "__main__":
    GEN_DIR = "/home/leike/spatial/ScriptSpeech/test/gene"
    GT_DIR = "/home/leike/spatial/ScriptSpeech/test/gt"
    fd = calculate_fd_score(GEN_DIR, GT_DIR)