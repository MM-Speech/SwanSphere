import os
import numpy as np
import librosa
import torch
from tqdm import tqdm

# ==========================================
# Part 1: 修改后的核心计算函数 (Channel First)
# ==========================================

def align_length_ch_first(audio_a, audio_b):
    """
    长度对齐函数 (适配 Channel-First 格式)
    
    Args:
        audio_a (np.ndarray): 形状 (Channels, N_samples_a)
        audio_b (np.ndarray): 形状 (Channels, N_samples_b)
        
    Returns:
        tuple: (aligned_a, aligned_b) 形状均为 (Channels, min_len)
    """
    # 长度维度现在是 index 1
    len_a = audio_a.shape[1]
    len_b = audio_b.shape[1]
    
    if len_a == len_b:
        return audio_a, audio_b
    
    min_len = min(len_a, len_b)
    
    # 在第二个维度 (时间维度) 上截断
    aligned_a = audio_a[:, :min_len]
    aligned_b = audio_b[:, :min_len]
    
    return aligned_a, aligned_b

def extract_angular_features(audio_foa):
    """
    从4声道FOA音频中计算方向角。
    
    Args:
        audio_foa (np.ndarray): 形状为 (4, N_samples) 的numpy数组。
                                *** 注意这里改成了 (4, N) ***
                                假设声道顺序为 AmbiX (ACN): W, Y, Z, X
                                Row 0: W (Omni)
                                Row 1: Y (Left-Right)
                                Row 2: Z (Up-Down)
                                Row 3: X (Front-Back)

    Returns:
        tuple: (theta, phi) 单位为弧度。
    """
    # [修改点] 现在的索引方式改为行索引
    # audio_foa[channel_index, :]
    w = audio_foa[0, :]
    y = audio_foa[1, :]
    z = audio_foa[2, :]
    x = audio_foa[3, :]

    # 1. 计算强度向量 (Intensities)
    # Ix = mean(W * X)
    # 注意：这里的乘法是 element-wise 乘法，然后对时间轴取平均
    I_x = np.mean(w * x)
    I_y = np.mean(w * y)
    I_z = np.mean(w * z)
    
    # 极小值防止除零
    eps = 1e-8

    # 2. 计算 Azimuth (Theta)
    # theta = arctan(Iy / Ix)
    theta = np.arctan2(I_y, I_x + eps)

    # 3. 计算 Elevation (Phi)
    # phi = arctan(Iz / sqrt(Ix^2 + Iy^2))
    hypotenuse_xy = np.sqrt(I_x**2 + I_y**2)
    phi = np.arctan2(I_z, hypotenuse_xy + eps)

    return theta, phi

def calculate_spatial_metrics(audio_gt, audio_pred):
    """
    计算单条样本的空间误差。
    
    Args:
        audio_gt (np.ndarray): (4, N)
        audio_pred (np.ndarray): (4, N)
    """
    # 1. 长度对齐
    audio_gt, audio_pred = align_length_ch_first(audio_gt, audio_pred)
    
    # 2. 获取角度特征 (返回的是标量弧度值)
    theta_gt, phi_gt = extract_angular_features(audio_gt)
    theta_pred, phi_pred = extract_angular_features(audio_pred)

    # --- Theta Error ---
    diff_theta = np.abs(theta_gt - theta_pred)
    theta_error = np.minimum(diff_theta, 2 * np.pi - diff_theta)

    # --- Phi Error ---
    phi_error = np.abs(phi_gt - phi_pred)

    # --- Spatial-Angle Error ---
    delta_phi = phi_gt - phi_pred
    delta_theta = theta_gt - theta_pred 

    term1 = np.sin(delta_phi / 2.0)**2
    term2 = np.cos(phi_gt) * np.cos(phi_pred) * (np.sin(delta_theta / 2.0)**2)
    
    a = np.clip(term1 + term2, 0.0, 1.0)
    spatial_angle_error = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return {
        "theta_error": theta_error,
        "phi_error": phi_error,
        "spatial_angle_error": spatial_angle_error
    }

# ==========================================
# Part 2: 数据加载与批量评测
# ==========================================

def load_opus_as_numpy(path):
    """
    读取 Opus/Wav 文件，返回 Numpy 数组 [Channels, Time]
    此函数专为上述数学计算设计，不返回 Tensor。
    """
    try:
        # librosa 默认返回 (Channels, Time)，这正是我们要的
        waveform, sr = librosa.load(path, sr=None, mono=False)
        
        # 确保是 2D 数组 (即使是单声道也要变成 [1, N])
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]
            
        return waveform
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def evaluate_spatial_dirs(gen_dir, gt_dir):
    """
    批量评测函数
    """
    valid_exts = ('.opus', '.wav', '.flac', '.mp3')
    
    # 1. 匹配文件名
    gen_map = {os.path.splitext(f)[0]: f for f in os.listdir(gen_dir) if f.endswith(valid_exts)}
    gt_map = {os.path.splitext(f)[0]: f for f in os.listdir(gt_dir) if f.endswith(valid_exts)}
    common_stems = sorted(list(set(gen_map.keys()) & set(gt_map.keys())))
    
    if not common_stems:
        print("Error: No common files found.")
        return None
        
    print(f"Starting Spatial Evaluation on {len(common_stems)} pairs...")
    
    total_theta = 0.0
    total_phi = 0.0
    total_spatial = 0.0
    valid_count = 0
    
    for stem in tqdm(common_stems):
        gt_path = os.path.join(gt_dir, gt_map[stem])
        gen_path = os.path.join(gen_dir, gen_map[stem])
        
        # 加载数据 (Numpy, [4, N])
        wav_gt = load_opus_as_numpy(gt_path)
        wav_gen = load_opus_as_numpy(gen_path)
        
        if wav_gt is None or wav_gen is None:
            continue
            
        # [安全检查] 必须是 4 声道才能计算 FOA 空间角
        if wav_gt.shape[0] != 4 or wav_gen.shape[0] != 4:
            # print(f"Skipping {stem}: Channels mismatch (GT:{wav_gt.shape[0]}, Gen:{wav_gen.shape[0]})")
            continue
            
        try:
            # 计算指标
            metrics = calculate_spatial_metrics(wav_gt, wav_gen)
            
            total_theta += metrics['theta_error']
            total_phi += metrics['phi_error']
            total_spatial += metrics['spatial_angle_error']
            valid_count += 1
            
        except Exception as e:
            print(f"Calculation failed for {stem}: {e}")
            
    if valid_count == 0:
        print("No valid 4-channel pairs evaluated.")
        return None
        
    # 计算平均值
    avg_theta = total_theta / valid_count
    avg_phi = total_phi / valid_count
    avg_spatial = total_spatial / valid_count
    
    print("\n" + "="*40)
    print(" >>> Spatial Metrics Summary <<<")
    print(f" Processed: {valid_count} files")
    print("-" * 40)
    print(f" Theta Error (Azimuth):   {avg_theta:.4f} rad")
    print(f" Phi Error (Elevation):   {avg_phi:.4f} rad")
    print(f" Spatial Angle Error:     {avg_spatial:.4f} rad")
    print("="*40 + "\n")
    
    return {
        "avg_theta": avg_theta,
        "avg_phi": avg_phi,
        "avg_spatial": avg_spatial
    }

# --- 测试脚本 ---
if __name__ == "__main__":
    # 替换为你的真实路径
    GEN_DIR = "/home/leike/spatial/ScriptSpeech/test/gene"
    GT_DIR = "/home/leike/spatial/ScriptSpeech/test/gt"
    
    # 运行评测
    if os.path.exists(GEN_DIR) and os.path.exists(GT_DIR):
        evaluate_spatial_dirs(GEN_DIR, GT_DIR)
    else:
        # 冒烟测试：生成假数据测试逻辑是否跑通
        print("Paths not found, running smoke test with dummy data...")
        # 模拟 (4, 48000) 数据
        dummy_gt = np.random.randn(4, 48000)
        dummy_pred = np.random.randn(4, 48000)
        # 增加一点相关性
        dummy_pred[3, :] = dummy_gt[3, :] * 0.9 
        
        m = calculate_spatial_metrics(dummy_gt, dummy_pred)
        print(f"Smoke Test Result: {m}")