import os

import json
import tqdm
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

from modules.tts.scriptspeech.build_model_utils import DiTBuildModelMixin
from tasks.spatial.dataset_utils.visage_like_dataset import prepare_ambi

from stable_audio_tools.models.autoencoders import create_autoencoder_from_config

class SpatialDiTInfer(DiTBuildModelMixin):
    def __init__(self, device, ckpt):
        self.device = device
        self.sample_rate = 44100
        self.frame_rate = 21.5
        
        self.build_model(ckpt)
        self.sa_vae = self.build_StableAudioVAE()
        
    def build_model(self, ckpt):
        set_hparams(config=os.path.join(ckpt, 'config.yaml'), print_hparams=False)
        self._build_model()
        # self.dac.to(self.device)
        # self.audio_tokenizer.to(self.device)
        # load_ckpt(self.dit, '/home/leike/spatial/ScriptSpeech/checkpoints/dit_v3_s/model_ckpt_steps_242000_good.ckpt', 'dit', strict=True)
        load_ckpt(self.dit, ckpt, 'dit', strict=True)
        self.dit.eval()
        self.dit.to(self.device)
    
    def build_StableAudioVAE(self):
        json_path = '/home/leike/spatial/stable-audio-tools/stable_audio_tools/checkpoints/vae_model_config.json'
        vae_path = '/home/leike/spatial/stable-audio-tools/stable_audio_tools/checkpoints/vae_model.ckpt'
        with open(json_path, "r", encoding="utf-8") as f:
            cfg: dict = json.load(f)
        
        vae = create_autoencoder_from_config(cfg)
        vae_checkpoint = torch.load(vae_path, map_location='cpu')
        if "state_dict" in vae_checkpoint:
            state_dict = vae_checkpoint["state_dict"]
        else:
            state_dict = vae_checkpoint
        vae.load_state_dict(state_dict, strict=True)
        vae = vae.to(self.device)
        vae.eval()
        print('vae loaded')
        return vae

    @torch.no_grad()
    def forward(self, test_id):
        inputs_embeds = np.load(os.path.join('/data/leike/spatial/sphere360/test_front_erp_clip', test_id))[:40]
        global_clip_embedding = np.load(os.path.join('/data/leike/spatial/sphere360/test_pad_360_erp_clip', test_id))[:40]
        # direction = prepare_ambi(test_id, augment=False)
        # energy_map = np.load(os.path.join('/data/leike/spatial/YT-Ambigen/energy_map', test_id))[:20]
        
        # gt = np.load(os.path.join('/data/leike/spatial/YT-Ambigen/stable_latents_10',test_id))
        # gt = torch.tensor(gt).unsqueeze(0).to(self.device)


        inputs = {
            'inputs_embeds': torch.tensor(inputs_embeds).unsqueeze(0).to(self.device),
            'global_clip_embedding': torch.tensor(global_clip_embedding).unsqueeze(0).to(self.device),
            'direction': None,
            'energy_map': None
            # 'direction': torch.from_numpy(direction).unsqueeze(0).to(self.device),
            # 'energy_map': torch.tensor(energy_map).unsqueeze(0).to(self.device),
        }
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            x = self.dit.inference(inputs, timesteps=200)
            
        latent_mean = np.load('/data/leike/spatial/sphere360/train_stable_latents_stats/mean.npy')
        latent_std = np.load('/data/leike/spatial/sphere360/train_stable_latents_stats/std.npy')
        latent_mean = torch.tensor(latent_mean, device=self.device, dtype=x.dtype)
        latent_std = torch.tensor(latent_std, device=self.device, dtype=x.dtype)

        x = x * latent_std + latent_mean
        # x = gt
        
        x = x.permute(0, 2, 1)
        WX = x[:, :64, :]
        YZ = x[:, 64:, :]
        # import pdb; pdb.set_trace()
        rec_WX = self.sa_vae.decode(WX)
        rec_YZ = self.sa_vae.decode(YZ)
        rec_audio = torch.cat([rec_WX, rec_YZ], dim=0)
        
        rec_audio = rec_audio.detach().to('cpu') 
        rec_audio = rec_audio.reshape(-1, rec_audio.shape[-1])
        
        # torchaudio.save("test_reconstructed.wav", rec_audio[0], 44100)
        
        # import pdb; pdb.set_trace()
        return rec_audio
    
    @torch.no_grad()
    def batch_forward(self, test_ids):
        """
        Args:
            test_ids (list): 文件名列表，例如 ['video_001.npy', 'video_002.npy', ...]
        """
        # ----------------------
        # 1. 批量加载数据
        # ----------------------
        batch_inputs_embeds = []
        batch_global_clip = []
        
        # 遍历加载数据
        for tid in test_ids:
            p1 = os.path.join('/data/leike/spatial/sphere360/test_front_erp_clip', tid)
            p2 = os.path.join('/data/leike/spatial/sphere360/test_pad_360_erp_clip', tid)
            
            # 读取并截取 [:40]
            batch_inputs_embeds.append(np.load(p1)[:40])
            batch_global_clip.append(np.load(p2)[:40])
            
        # 堆叠为 Tensor: (B, 40, Embed_Dim)
        inputs_embeds = torch.tensor(np.stack(batch_inputs_embeds)).to(self.device)
        global_clip_embedding = torch.tensor(np.stack(batch_global_clip)).to(self.device)

        inputs = {
            'inputs_embeds': inputs_embeds,
            'global_clip_embedding': global_clip_embedding,
            'direction': None,
            'energy_map': None
        }

        # ----------------------
        # 2. 批量推理 DiT
        # ----------------------
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # DiT 通常原生支持 batch 输入
            x = self.dit.inference(inputs, timesteps=200)

        # ----------------------
        # 3. 反归一化 (优化：懒加载统计量)
        # ----------------------
        if not hasattr(self, 'latent_mean'):
            mean_path = '/data/leike/spatial/sphere360/train_stable_latents_stats/mean.npy'
            std_path = '/data/leike/spatial/sphere360/train_stable_latents_stats/std.npy'
            self.latent_mean = torch.tensor(np.load(mean_path), device=self.device, dtype=x.dtype)
            self.latent_std = torch.tensor(np.load(std_path), device=self.device, dtype=x.dtype)

        # 广播计算: (B, T, D) * (D,) + (D,)
        x = x * self.latent_std + self.latent_mean

        # ----------------------
        # 4. 批量解码 VAE
        # ----------------------
        x = x.permute(0, 2, 1)  # (B, Dim, T_latent)
        
        # 切分 WX 和 YZ (Channel 维度)
        WX = x[:, :64, :]
        YZ = x[:, 64:, :]

        # VAE 批量解码 -> 输出 (B, 2, Audio_Len)
        rec_WX = self.sa_vae.decode(WX)
        rec_YZ = self.sa_vae.decode(YZ)

        # 在 Channel 维度拼接: (B, 2, L) + (B, 2, L) -> (B, 4, L)
        rec_audio = torch.cat([rec_WX, rec_YZ], dim=1)
        
        # 移回 CPU 并返回
        return rec_audio.detach().cpu()
        
def read_sphere_test():
    metadata = '/data/leike/spatial/metadata/sphere360_test.jsonl'
    
    file_names = []

    with open(metadata, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            file_names.append(item['file_name'].replace('.npy', ''))
    
    return file_names

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
        
if __name__ == '__main__':
    
    infer = SpatialDiTInfer(torch.device('cuda'), '/home/leike/spatial/ScriptSpeech/checkpoints/dit_v3_s_10_all')

    # infer_lst = [
    #     'BI_heWaNfro_8', 'VX_gOGFgt14_46', 'dKye1dZuECk_89', 'bhAhh3dSzHI_85', # train里的
    #     # 'ECFTh6UdONY_158', 'gSRVPLekBwY_14', 'xZ3KECAMpwo_73', 'kha7D_Nt3QA_15', # test里的
    #     '0A4GRMrLpWI_259', '0BDCLo2pioo_43', '0BDCLo2pioo_130', '0DhWUtlWcA0_13', '03XzVqjmECw_63', '06av4szCH1s_157',   # train里的
    #     # '0D4rxdOI5TM_13', # valid
    #     '0FB9jMXMP8A_31', '0FB9jMXMP8A_43',  # test
    #     # 'gSRVPLekBwY_21', '_D7CJg5fvsE_99', 'IQifpz8nZDA_91', '0hCGacvtyNQ_26', 
    #     'MVPCbI71shM_19', 'tENB2euDcB4_227', '5rrCEo7Rwv8_88', '4EgRaySMjJQ_39',
    #     'sLSh-etPd4o_138', 'HK-eDj5gdPk_89', '85YGH9MdjLo_52', 'nNRoC0xn1Aw_25' # test
    # ]

    infer_lst = [
        # "fQukntBmFvY_40", "OWN_J9FGZ5I_55", "kMZSoni0etA_40",
        # "yhFN_xVmNsI_50", "f2kvR5I8s3c_70", "1WFJLucjK50_540", "3GSTGzYkHks_80", "2XhYnQ5QC4E_64",
        # "5WrI-nS59kA_130", "9X2wM6HD_og_933", "aCCPRvNpcYk_45", "DcMh9zgZbSg_400", "KrEHdKxlNDQ_2", "lOJBGAd0mSw_150",
        # "aieThfuvmtY_26"
        "G8pABGosD38_17",
        # "0B7ds6NmVBQ_30", "0B7ds6NmVBQ_20", "0B7ds6NmVBQ_10", "0B7ds6NmVBQ_40"
    ]
    
    # infer_lst = read_sphere_test()
    
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    SAVE_DIR = os.path.join('results', timestamp)
    os.makedirs(SAVE_DIR, exist_ok=True)
    for item in tqdm.tqdm(infer_lst):
        print('infer ', item)
        rec_audio = infer.forward(item + '.npy')
        torchaudio.save(f'{SAVE_DIR}/{item}.wav', rec_audio, 44100)
        os.system(f"cp /data/leike/spatial/sphere360/test_audio/{item}.opus {SAVE_DIR}")
        
    # import math


    # # 配置
    # BATCH_SIZE = 8  # 根据你的显存大小调整 (例如 4, 8, 16)
    # infer_lst = read_sphere_test()
    # infer_lst = infer_lst[::-1]

    # from datetime import datetime
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # SAVE_DIR = os.path.join('results', timestamp)
    # os.makedirs(SAVE_DIR, exist_ok=True)

    # # 计算总 batch 数用于 tqdm
    # total_batches = math.ceil(len(infer_lst) / BATCH_SIZE)

    # print(f"Start Inference with Batch Size: {BATCH_SIZE}")

    # for batch_items in tqdm.tqdm(chunker(infer_lst, BATCH_SIZE), total=total_batches):
    #     try:
    #         # 1. 构造文件名列表
    #         batch_ids = [item + '.npy' for item in batch_items]
            
    #         # 2. 批量推理 -> 得到 (B, 4, Audio_Len)
    #         batch_audio = infer.batch_forward(batch_ids)
            
    #         # 3. 拆分保存
    #         for i, item in enumerate(batch_items):
    #             # batch_audio[i] 是 (4, Audio_Len)
    #             save_path = f'{SAVE_DIR}/{item}.wav'
    #             torchaudio.save(save_path, batch_audio[i], 44100)
                
    #             # 复制原始文件
    #             src_opus = f"/data/leike/spatial/sphere360/test_audio/{item}.opus"
    #             if os.path.exists(src_opus):
    #                 os.system(f"cp {src_opus} {SAVE_DIR}")
    #     except Exception as e:
    #         continue