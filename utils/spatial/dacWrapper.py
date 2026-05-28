'''原版dac decode只能从codes或者z来，这里改为可以从latents解码出来'''
import dac
import torch
import numpy as np
import soundfile as sf

class DACDecodePipe:
    def __init__(self, device=torch.device('cuda')):
        self.device = device
        
        dac_sample_rate = 44100
        dac_frame_rate = 512
        model_path = dac.utils.download(model_type="44khz")
        self.model = dac.DAC.load(model_path)
        self.model.eval()
        self.model.to(self.device)
        
    def decode_latents(self, latents):
        if latents.ndim == 3:
            latents = latents.reshape(4, 72, -1)
            
        z_q = 0
        n_quantizers = self.model.n_codebooks
        
        for i, quantizer in enumerate(self.model.quantizer.quantizers):
            z_q_tmp = latents[:, i*8:(i+1)*8, :]
            z_q_i = quantizer.out_proj(z_q_tmp)
            
            # Create mask to apply quantizer dropout
            mask = (
                torch.full((latents.shape[0],), fill_value=i, device=latents.device) < n_quantizers
            )
            z_q = z_q + z_q_i * mask[:, None, None]
        
        return z_q

    def decode_z(self, z):
        return self.mode.decode(z)
    
if __name__ == "__main__":
    pipe = DACDecodePipe(torch.device('cuda'))
    
    ### 测试gt
    gt_latents = np.load('/mnt/bn/sa-ag-data/leike/spatial/data/YT-Ambigen/dac_latents/-Sjj4gxDIm8_104.npy')
    gt_latents = torch.tensor(gt_latents, device='cuda')
    gt_latents = gt_latents.reshape(4, 72, -1)
    z = pipe.decode_latents(gt_latents)
    
    y = pipe.model.decode(z)
    import pdb; pdb.set_trace()
    # 假设 y 是你的 tensor，形状: torch.Size([4, 1, 220672])
    # 1. 去掉中间的维度 (4, 1, N) -> (4, N)
    audio_data = y.squeeze(1) 
    # 2. 转置为 (N, 4) 因为 soundfile 需要 (samples, channels)
    audio_data = audio_data.transpose(0, 1)
    # 3. 转为 numpy (如果 tensor 在 GPU 上需要先 .cpu())
    # .detach() 是为了断开梯度计算图，是一个好习惯
    audio_numpy = audio_data.detach().cpu().numpy()
    # 4. 保存文件
    # samplerate 根据你的实际情况填写，通常 FOA 是 44100 或 48000
    sf.write('results/-Sjj4gxDIm8_104.wav', audio_numpy, samplerate=44100)
    #### 
    
    
    #### 测试过拟合的结果
    # rec_latents = np.load('/mnt/bn/sa-ag-data/leike/spatial/ScriptSpeech/results/-3X9U3VK0tM_133_rec.npy')
    # rec_latents = torch.tensor(rec_latents, device='cuda')
    # rec_z = pipe.decode_latents(rec_latents)
    
    # rec_y = pipe.model.decode(rec_z)
    # audio_data = rec_y.squeeze(1) 
    # audio_data = audio_data.transpose(0, 1)
    # audio_numpy = audio_data.detach().cpu().numpy()
    # sf.write('results/-3X9U3VK0tM_133_rec.wav', audio_numpy, samplerate=44100)