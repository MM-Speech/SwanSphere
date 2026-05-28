import os

import torch
import torchaudio
import soundfile as sf
# import librosa
import numpy as np
import subprocess

from utils.commons.os_utils import kill_void
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import set_hparams, hparams
import dac

from modules.tts.scriptspeech.build_model_utils import DiTBuildModelMixin

from tasks.spatial.dataset_utils.visage_like_dataset import prepare_ambi

def load_opus_with_ffmpeg(path: str,
                          target_sr: int = 48000,
                          target_channels: int = 4,
                          device: str = "cpu"):
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", path,
        "-ac", str(target_channels),
        "-ar", str(target_sr),
        "-f", "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size % target_channels != 0:
        raise RuntimeError(f"Decoded audio size {audio.size} not divisible by channels {target_channels}")
    num_frames = audio.size // target_channels
    audio = audio.reshape(num_frames, target_channels).transpose(1, 0)  # [C, T]
    waveform = torch.from_numpy(audio).to(device)
    return waveform, target_sr

class SpatialDiTInfer(DiTBuildModelMixin):
    def __init__(self, device, ckpt):
        self.device = device
        dac_sample_rate = 44100
        dac_frame_rate = 512
        model_path = dac.utils.download(model_type="44khz")
        self.dac = dac.DAC.load(model_path)
        self.dac.eval()
        
        self.build_model(ckpt)
        
    def build_model(self, ckpt):
        set_hparams(config=os.path.join(ckpt, 'config.yaml'), print_hparams=False)
        self._build_model()
        self.dac.to(self.device)
        # self.audio_tokenizer.to(self.device)
        load_ckpt(self.dit, ckpt, 'dit', strict=True)
        self.dit.eval()
        self.dit.to(self.device)

    @torch.no_grad()
    def forward(self, test_id):
        
        inputs_embeds = np.load(os.path.join('/data/leike/spatial/YT-Ambigen/clip', test_id))[:20]
        global_clip_embedding = np.load(os.path.join('/data/leike/spatial/YT-Ambigen/clip_360', test_id))[:20]
        direction = prepare_ambi(test_id, augment=False)
        energy_map = np.load(os.path.join('/data/leike/spatial/YT-Ambigen/energy_map', test_id))[:20]


        inputs = {
            'inputs_embeds': torch.tensor(inputs_embeds).unsqueeze(0).to(self.device),
            'global_clip_embedding': torch.tensor(global_clip_embedding).unsqueeze(0).to(self.device),
            'direction': torch.from_numpy(direction).unsqueeze(0).to(self.device),
            'energy_map': torch.tensor(energy_map).unsqueeze(0).to(self.device),
        }
        
        # import pdb; pdb.set_trace()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            x = self.dit.inference(inputs, timesteps=200)
            
        latent_mean = np.load('/data/leike/spatial/YT-Ambigen/dac_latents_stats/mean.npy')
        latent_std = np.load('/data/leike/spatial/YT-Ambigen/dac_latents_stats/std.npy')
        latent_mean = torch.tensor(latent_mean, device=self.device, dtype=x.dtype)
        latent_std = torch.tensor(latent_std, device=self.device, dtype=x.dtype)

        x = x * latent_std + latent_mean
        
        x = x.view(1, 430, 4, 72)
            # 调整维度顺序: [4, 1024, 430]
        x = x.permute(2, 3, 1, 0).squeeze(3)
        
        return x
            
        # rec_audio = self.dac.decode(x)
        # rec_audio = rec_audio.cpu().numpy()
            
        # ### raw opus recon
        # enc_z = np.load(os.path.join('/mnt/bn/sa-ag-data/leike/spatial/data/YT-Ambigen/dac_z', test_id))
        # enc_z = torch.tensor(enc_z).to(self.device)
        # enc_z = enc_z.reshape(4, 1024, -1)
        # gt_audio = self.dac.decode(enc_z)
        # gt_audio = gt_audio.cpu().numpy()
        
        # ### enc and dec
        # wav_path = "/mnt/bn/sa-ag-data/leike/spatial/data/YT-Ambigen/audio/u71dmv9-meg_35.opus"
        # waveform, sr = load_opus_with_ffmpeg(wav_path)   # waveform: [channels, time]
        # assert waveform.shape[0] == 4, f"Expected 4 channels, got {waveform.shape[0]}"
        # assert sr == 48000, f"Expected 48kHz, got {sr}"
        # waveform = waveform.to(self.device)
        # # ========== 3. 重采样到 44.1kHz（DAC 原生采样率） ==========
        # dac_sr = 44100  # model_type="44khz" 对应 44.1kHz[^2994857466]
        # if sr != dac_sr:
        #     resampler = torchaudio.transforms.Resample(sr, dac_sr).to(self.device)
        #     waveform_dac = resampler(waveform)  # [4, T_dac]
        # else:
        #     waveform_dac = waveform
        # # ========== 4. 把 4 声道当成 batch：一次 encode / decode ==========
        # # [4, T] -> [4, 1, T]，视为 batch_size=4 的 4 条单声道
        # batch = waveform_dac.unsqueeze(1)  # [B=4, C=1, T]
        # with torch.no_grad():
        #     # encode：返回 z, codes, latents, 其余两个输出可忽略[^1358313466]
        #     z, codes, latents, _, _ = self.dac.encode(batch)
        #     # decode：从 z 重建音频 y[^1358313466]
        #     print(f"拿来decode的z的形状：{z.shape=}")
        #     recon = self.dac.decode(z)  # recon: [4, 1, T_recon]
        # # 去掉 channel 维：[4, 1, T] -> [4, T]
        # recon_foa_dac = recon.squeeze(1)  # [4, T_dac_recon]
        # # ========== 5. 重采样回 48kHz，得到 recon_foa ==========
        # # 对齐长度以防轻微长度差异
        # min_len = recon_foa_dac.shape[-1]
        # recon_foa_dac = recon_foa_dac[..., :min_len]  # [4, T_dac_recon]
        # if dac_sr != sr:
        #     resampler_back = torchaudio.transforms.Resample(dac_sr, sr).to(self.device)
        #     recon_foa = resampler_back(recon_foa_dac)  # [4, T_48k]
        # else:
        #     recon_foa = recon_foa_dac
        # recon_foa = recon_foa.detach().cpu()
        # # ========== 6. 保存为 4 声道 Opus ==========
        # out_path = "results/u71dmv9-meg_35_recon.wav"
        # os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # recon_foa_np = recon_foa.detach().cpu().numpy()  # [4, T]
        # # soundfile 期望 [T, C]，这里保存为 4 声道 WAV，FOA 没问题
        # sf.write(
        #     out_path,
        #     recon_foa_np.T,   # [T, 4]
        #     samplerate=sr,
        #     subtype="PCM_16",
        # )
        

        return rec_audio


if __name__ == '__main__':
    
    infer = SpatialDiTInfer(torch.device('cuda'), '/home/leike/spatial/ScriptSpeech/checkpoints/dit_v2')
    
    infer_lst = [
        # 'BI_heWaNfro_8', 'VX_gOGFgt14_46', 'dKye1dZuECk_89', 'bhAhh3dSzHI_85', # train里的
        # 'ECFTh6UdONY_158', 'gSRVPLekBwY_14', 'xZ3KECAMpwo_73', 'kha7D_Nt3QA_15', # test里的
        # '0A4GRMrLpWI_259', '0BDCLo2pioo_43', '0BDCLo2pioo_130', '0DhWUtlWcA0_13', '03XzVqjmECw_63', '06av4szCH1s_157',   # train里的
        # '0D4rxdOI5TM_13', # valid
        # '0FB9jMXMP8A_31', '0FB9jMXMP8A_43',  # test
        'gSRVPLekBwY_21', '_D7CJg5fvsE_99', 'IQifpz8nZDA_91', '0hCGacvtyNQ_26', 
        'MVPCbI71shM_19', 'tENB2euDcB4_227', '5rrCEo7Rwv8_88', '4EgRaySMjJQ_39',
        'sLSh-etPd4o_138', 'HK-eDj5gdPk_89', '85YGH9MdjLo_52', 'nNRoC0xn1Aw_25' # test
    ]
    for item in infer_lst:
        print(f"infer {item}...")
        rec_audio = infer.forward(item + '.npy')
        # print(f"{rec_audio.shape =}")
        rec_audio = rec_audio.cpu().numpy()
        # np.save('/mnt/bn/sa-ag-data/leike/spatial/ScriptSpeech/results/-3X9U3VK0tM_133_rec.npy', rec_audio)
        # gt = np.load('/mnt/bn/sa-ag-data/leike/spatial/data/YT-Ambigen/dac_latents/u71dmv9-meg_35.npy')
        
        from utils.spatial.dacWrapper import DACDecodePipe
        pipe = DACDecodePipe(torch.device('cuda'))
        rec_latents = torch.tensor(rec_audio, device='cuda')
        rec_z = pipe.decode_latents(rec_latents)
        
        rec_y = pipe.model.decode(rec_z)
        audio_data = rec_y.squeeze(1) 
        audio_data = audio_data.transpose(0, 1)
        audio_numpy = audio_data.detach().cpu().numpy()
        sf.write(f'results/test/{item}_rec.wav', audio_numpy, samplerate=44100)
        os.system(f"cp /data/leike/spatial/YT-Ambigen/audio/{item}.opus /home/leike/spatial/ScriptSpeech/results/test")
        os.system(f"cp /data/leike/spatial/YT-Ambigen/raw/{item[:11]}.mp4 /home/leike/spatial/ScriptSpeech/results/test")
        
        # import pdb; pdb.set_trace()
        # # 形状处理 [4, 1, samples] -> [samples, 4]
        # rec_audio = rec_audio.squeeze(1).T  # 移除中间维度并转置
        # # 保存为4声道WAV
        # sample_rate = 44100
        # sf.write("results/u71dmv9-meg_35.wav", rec_audio, sample_rate, subtype='FLOAT')
        
        # gt_audio = gt_audio.squeeze(1).T
        # sample_rate = 44100
        # sf.write("results/u71dmv9-meg_35_gt.wav", gt_audio, sample_rate, subtype='FLOAT')
        