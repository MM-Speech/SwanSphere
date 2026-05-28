import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F


DEBUG = True

# ---------------------------------------------------------
# 1. 配置类 (调整 Hidden Dim 以符合 ~20MB)
# ---------------------------------------------------------
@dataclass
class ModelArgs:
    # 最终输出特征维度
    embed_dim: int = 512
    
    # Video配置
    vid_input_dim: int = 1408
    vid_hidden_dim: int = 384    # 降维后的通道数 (控制在 ~20MB)
    vid_out_ch: int = 32         # 最终卷积输出通道: 4*4*32 = 512
    
    # Audio配置
    foa_channels: int = 4
    foa_embed_dim: int = 768
    foa_seq_len: int = 8
    foa_hidden_dim: int = 512    # 降维后的通道数 (控制在 ~20MB)
    
    # 对比学习配置
    temperature: float = 0.07  # InfoNCE 温度系数

    # 通用配置
    norm_eps: float = 1e-6
    
    # 是否对原始数据ln
    pre_ln: bool = True

# ---------------------------------------------------------
# 2. 修复后的基础组件 (LayerNorm 广播修复)
# ---------------------------------------------------------

class LayerNorm2d(nn.Module):
    """
    Input: (N, C, H, W)
    Normalize over C dimension.
    """
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        # 修复：广播维度为 (1, C, 1, 1) 以匹配 (N, C, H, W)
        x = self.weight[None, :, None, None] * x + self.bias[None, :, None, None]
        return x

class LayerNorm1d(nn.Module):
    """
    Input: (N, C, L)
    Normalize over C dimension.
    """
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        # 修复：广播维度为 (1, C, 1) 以匹配 (N, C, L)
        x = self.weight[None, :, None] * x + self.bias[None, :, None]
        return x

class ResBlock2d(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=False)
        self.ln1 = LayerNorm2d(out_dim)
        self.act = nn.GELU()
        
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size, stride=1, padding=padding, bias=False)
        self.ln2 = LayerNorm2d(out_dim)
        
        self.downsample = None
        if stride != 1 or in_dim != out_dim:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False),
                LayerNorm2d(out_dim)
            )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.ln2(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return x + residual

class ResBlock1d(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=False)
        self.ln1 = LayerNorm1d(out_dim)
        self.act = nn.GELU()
        
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size, stride=1, padding=padding, bias=False)
        self.ln2 = LayerNorm1d(out_dim)
        
        self.downsample = None
        if stride != 1 or in_dim != out_dim:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False),
                LayerNorm1d(out_dim)
            )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.ln2(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return x + residual

# ---------------------------------------------------------
# 3. 优化后的 Projector (使用 1x1 Conv 先降维)
# ---------------------------------------------------------

class VideoProjector(nn.Module):
    def __init__(self, hp: ModelArgs):
        super().__init__()
        self.hp = hp
        if hp.pre_ln:
            self.input_ln = nn.LayerNorm(hp.vid_input_dim)
        # 1. 先用 1x1 卷积大幅降维，节省参数
        # 1408 -> 384 (1x1 Conv, Param ~2MB)
        self.pre_compress = nn.Sequential(
            nn.Conv2d(hp.vid_input_dim, hp.vid_hidden_dim, kernel_size=1, bias=False),
            LayerNorm2d(hp.vid_hidden_dim),
            nn.GELU()
        )
        
        # 2. ResBlocks 处理特征
        # 384 -> 384 (Stride 2: 16x16 -> 8x8)
        self.stage1 = ResBlock2d(hp.vid_hidden_dim, hp.vid_hidden_dim, stride=2)
        # 384 -> 384 (Stride 1)
        self.stage2 = ResBlock2d(hp.vid_hidden_dim, hp.vid_hidden_dim, stride=1)
        
        # 3. 输出层
        # 384 -> 32 (Stride 2: 8x8 -> 4x4)
        self.final_conv = ResBlock2d(hp.vid_hidden_dim, hp.vid_out_ch, stride=2)
        
        self.final_ln = nn.LayerNorm(hp.embed_dim)

    def forward(self, x):
        if self.hp.pre_ln:
            x = self.input_ln(x)
        # x: (B, T1, 256, 1408)
        B, T1, N, C = x.shape
        H = W = int(N ** 0.5) 
        
        x = x.view(B * T1, N, C)
        x = x.permute(0, 2, 1).view(B * T1, C, H, W)
        
        x = self.pre_compress(x) # 1x1 conv
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.final_conv(x) # -> (B*T1, 32, 4, 4)
        
        x = x.flatten(1) # -> (B*T1, 512)
        x = self.final_ln(x)
        x = x.view(B, T1, -1)
        return x

class AudioProjector(nn.Module):
    def __init__(self, hp: ModelArgs):
        super().__init__()
        self.hp = hp
        if hp.pre_ln:
            self.input_ln = nn.LayerNorm(hp.foa_embed_dim)
        input_dim = hp.foa_channels * hp.foa_embed_dim # 3072
        
        # 1. 先用 1x1 卷积大幅降维
        # 3072 -> 512 (1x1 Conv, Param ~6MB)
        self.pre_compress = nn.Sequential(
            nn.Conv1d(input_dim, hp.foa_hidden_dim, kernel_size=1, bias=False),
            LayerNorm1d(hp.foa_hidden_dim),
            nn.GELU()
        )
        
        # 2. ResBlocks
        self.stage1 = ResBlock1d(hp.foa_hidden_dim, hp.foa_hidden_dim, stride=1)
        self.stage2 = ResBlock1d(hp.foa_hidden_dim, hp.foa_hidden_dim, stride=1)
        
        # 3. 压缩时间维度 (T=8 -> T=1) 并变换到 512
        # 这里的 output dim 和 hidden dim 可以不同，这里都设为 512
        self.compress_time = nn.Conv1d(hp.foa_hidden_dim, hp.embed_dim, kernel_size=hp.foa_seq_len)
        
        self.final_ln = nn.LayerNorm(hp.embed_dim)

    def forward(self, x):
        if self.hp.pre_ln:
            x = self.input_ln(x)
        # x: (B, T2, 4, 8, 768)
        B, T2, Ch, L, D = x.shape
        
        x = x.permute(0, 1, 3, 2, 4).reshape(B, T2, L, Ch * D) # (B, T2, 8, 3072)
        x = x.view(B * T2, L, -1).permute(0, 2, 1) # (B*T2, 3072, 8)
        
        x = self.pre_compress(x) # -> (B*T2, 512, 8)
        x = self.stage1(x)
        x = self.stage2(x)
        
        x = self.compress_time(x) # -> (B*T2, 512, 1)
        x = x.flatten(1)
        
        x = self.final_ln(x)
        x = x.view(B, T2, -1)
        return x

# ---------------------------------------------------------
# 4. 主模型
# ---------------------------------------------------------

class SVAC(nn.Module):
    def __init__(self, hp: ModelArgs):
        super().__init__()
        self.hp = hp
        self.vid_projector = VideoProjector(hp)
        self.foa_projector = AudioProjector(hp)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, LayerNorm2d, LayerNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, inputs):
        # vid_mae: (2*N, T1, 256, 1408)
        # foa_mae: (3*N, T2, 4, 8, 768)
        
        vid_mae = inputs['vid_mae'] 
        foa_mae = inputs['foa_mae'] 
        
        # 1. 获取序列特征 (保留时间维度)
        # vid_seq: (2N, T1, 512)
        # foa_seq: (3N, T2, 512)
        vid_seq = self.vid_projector(vid_mae)
        foa_seq = self.foa_projector(foa_mae)
        
        # 2. 如果在训练模式，计算对比损失
        # 注意：不再在这里做 mean(dim=1)，而是把序列传进去
        loss = None
        if self.training or DEBUG:
            loss = self.compute_contrastive_loss(vid_seq, foa_seq)
            
        # 3. 推理时，根据你的需求，可能仍然输出序列，或者池化后的特征
        # 这里为了保持一致性，输出序列，使用者可以自己决定如何处理
        return vid_seq, foa_seq, loss

    def compute_contrastive_loss(self, vid_seq, foa_seq):
        """
        计算基于时序对齐的对比损失
        vid_seq: (2N, T_v, D)
        foa_seq: (3N, T_a, D)
        """
        # 1. 基础参数
        N = vid_seq.shape[0] // 2
        T_a = foa_seq.shape[1] # Audio 的时间长度 (如 64)
        
        # 2. 帧级归一化 (L2 Norm over Last Dim)
        # 这样点积直接就是 Cosine Similarity
        vid_norm = F.normalize(vid_seq, p=2, dim=2) # (2N, T_v, 512)
        foa_norm = F.normalize(foa_seq, p=2, dim=2) # (3N, T_a, 512)
        
        # 3. 视频对齐 (Alignment)
        # 目标：将视频的时间轴拉伸/缩放到与音频一致 (T_v -> T_a)
        # 使用最近邻插值 (Nearest) 对应 "找中心时间最近的一帧"
        # Input need (B, C, L), so permute
        vid_norm_permuted = vid_norm.permute(0, 2, 1) # (2N, 512, T_v)
        
        # interpolate 到 T_a
        vid_aligned = F.interpolate(
            vid_norm_permuted, 
            size=T_a, 
            mode='nearest'
        ).permute(0, 2, 1) # -> (2N, T_a, 512)
        
        # 4. 准备 Anchors
        # 正样本对只有前 N 个
        vid_anchors_aligned = vid_aligned[:N] # (N, T_a, 512)
        foa_anchors = foa_norm[:N]            # (N, T_a, 512)
        
        # -------------------------------------------------------
        # Loss 1: Audio-to-Video (Anchor: Audio)
        # 我们需要计算 N 个 Audio Anchor 和 所有 2N 个 Video 的相似度
        # -------------------------------------------------------
        
        # foa_anchors: (N, T_a, D)
        # vid_aligned: (2N, T_a, D)
        # 目标: (N, 2N) 的矩阵，其中 [i, j] 是 sequence i 和 sequence j 的相似度
        # einsum 公式: 
        #   n: audio batch (N)
        #   m: video batch (2N)
        #   t: time (T_a)
        #   d: dim (512)
        #   result[n, m] = sum(A[n,t,d] * V[m,t,d]) over t, d
        
        sim_sum_a2v = torch.einsum('ntd,mtd->nm', foa_anchors, vid_aligned)
        logits_a2v = (sim_sum_a2v / T_a) / self.hp.temperature
        
        # 选取负样本逻辑 (与之前相同)
        # indices: [0..N-1] + [N+i]
        base_indices = torch.arange(N, device=logits_a2v.device).expand(N, N)
        hard_neg_indices = (torch.arange(N, device=logits_a2v.device) + N).unsqueeze(1)
        a2v_indices = torch.cat([base_indices, hard_neg_indices], dim=1) # (N, N+1)
        
        final_logits_a2v = torch.gather(logits_a2v, 1, a2v_indices)
        labels = torch.arange(N, device=logits_a2v.device)
        loss_a2v = F.cross_entropy(final_logits_a2v, labels)
        
        # -------------------------------------------------------
        # Loss 2: Video-to-Audio (Anchor: Video)
        # 我们需要计算 N 个 Video Anchor 和 所有 3N 个 Audio 的相似度
        # -------------------------------------------------------
        
        # vid_anchors_aligned: (N, T_a, D)
        # foa_norm: (3N, T_a, D)
        
        sim_sum_v2a = torch.einsum('ntd,mtd->nm', vid_anchors_aligned, foa_norm)
        logits_v2a = (sim_sum_v2a / T_a) / self.hp.temperature
        
        # 选取负样本逻辑
        # indices: [0..N-1] + [N+i] + [2N+i]
        neg1_indices = (torch.arange(N, device=logits_v2a.device) + N).unsqueeze(1)
        neg2_indices = (torch.arange(N, device=logits_v2a.device) + 2*N).unsqueeze(1)
        v2a_indices = torch.cat([base_indices, neg1_indices, neg2_indices], dim=1)
        
        final_logits_v2a = torch.gather(logits_v2a, 1, v2a_indices)
        loss_v2a = F.cross_entropy(final_logits_v2a, labels)
        
        return (loss_a2v + loss_v2a) / 2

# ---------------------------------------------------------
# 测试
# ---------------------------------------------------------
def count_parameters_in_MB(model):
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return param_count * 4 / (1024 ** 2)

if __name__ == "__main__":
    args = ModelArgs()
    model = SVAC(args)
    print(f"Video Projector Params: {count_parameters_in_MB(model.vid_projector):.2f} MB")
    print(f"Audio Projector Params: {count_parameters_in_MB(model.foa_projector):.2f} MB")
    
    # 模拟输入测试 (确保无报错)
    B = 2
    inputs = {
        'vid_mae': torch.randn(2*B, 41, 256, 1408),
        'foa_mae': torch.randn(3*B, 64, 4, 8, 768)
    }
    v, f, loss = model(inputs)
    print("Forward pass successful.")
    print(v.shape, f.shape)
    import pdb; pdb.set_trace()