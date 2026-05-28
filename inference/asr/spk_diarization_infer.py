import os
import tempfile
from pathlib import Path

import torch
import soundfile as sf
import torchaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt

from utils.commons.os_utils import kill_void
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import set_hparams, hparams

from modules.asr.scriptasr.build_model_utils import build_asr_text_tokenizer, build_asr_model

class SpkDiarizationInfer():
    def __init__(self, device, ckpt):
        self.device = device
        self.build_model(ckpt)

    def build_model(self, ckpt):
        if ckpt.endswith('.ckpt'):
            set_hparams(config=os.path.join(Path(ckpt).parent, 'config.yaml'), print_hparams=False)
        else:
            set_hparams(config=os.path.join(ckpt, 'config.yaml'), print_hparams=False)
        from utils.audio.mel import MelNet
        self.mel_net = MelNet(hparams, self.device)

        if hparams.get('model_name', 'simple') == 'simple':
            from modules.asr.spk_embed.spk_embed_model import SpkEmbed
            self.model = SpkEmbed(
                n_mels=hparams['audio_num_mel_bins'],
                hidden_size=hparams.get('hidden_size', 1024),
                wav_dowmsamples=[6, 5, 4, 2],
            )
        elif hparams.get('model_name', 'simple') == 'campplus':
            from modules.asr.spk_embed.campplus import CAMPPlus
            self.model = CAMPPlus(
                input_size=hparams['audio_num_mel_bins'], embd_dim=hparams.get('hidden_size', 1024),
                growth_rate=32, bn_size=4, init_channels=128, config_str='batchnorm-relu'
            )
        elif hparams.get('model_name', 'simple') == 'eres2net':
            from modules.asr.spk_embed.eres2net import ERes2Net
            self.model = ERes2Net(
                input_size=hparams['audio_num_mel_bins'], embd_dim=hparams.get('hidden_size', 1024), m_channels=32
            )
        self.model.to(self.device)
        load_ckpt(self.model, ckpt, 'model', strict=True, mmap=True)

    @torch.no_grad()
    def forward_model(self, wav):
        spk_win_size = hparams.get('spk_win_size', 12000)
        spk_hop_size = 6000
        wavs = []
        print('wav.shape', wav.shape)
        print('spk_win_size', spk_win_size)
        print('spk_hop_size', spk_hop_size)
        for start_idx in range(0, wav.shape[0], spk_hop_size):
            # print('int(start_idx * spk_hop_size + spk_win_size)', int(start_idx * spk_hop_size + spk_win_size))
            if int(start_idx + spk_win_size) >= wav.shape[0]:
                break
            wavs.append(wav[start_idx: start_idx + spk_win_size])
        print('len(wavs)', len(wavs))
        wavs = torch.from_numpy(np.stack(wavs, axis=0)).to(self.device)     # [N, T]
        mels = self.mel_net(wavs)   # [N, T, C]

        print('wavs.shape', wavs.shape)
        if hparams.get('model_name', 'simple') == 'simple':
            spk_embeds = self.model.encode(wavs, mels)    # [N, C]
        elif hparams.get('model_name', 'simple') in ['campplus', 'eres2net']:
            spk_embeds = self.model.encode(mels)    # [N, C]

        print('spk_embeds.shape', spk_embeds.shape)
        print('spk_embeds', spk_embeds)
        print('self.model.logit_scale.exp()', self.model.logit_scale.exp())

        # correlation = torch.sigmoid(self.model.logit_scale.exp() * (spk_embeds[1:] * spk_embeds[:-1]).sum(1))
        correlation = (spk_embeds[1:] * spk_embeds[:-1]).sum(1)
        # correlation = (spk_embeds[1:] * spk_embeds[:-1]).sum(1) * self.model.logit_scale.exp()
        # correlation = torch.sigmoid((spk_embeds[1:] * spk_embeds[:-1]).sum(1) * self.model.logit_scale.exp())

        from sklearn.cluster import DBSCAN
        dbscan = DBSCAN(eps=0.4, min_samples=2, metric='cosine')
        clusters = dbscan.fit_predict(spk_embeds.cpu().numpy())
        print(clusters)

        return correlation.cpu().numpy()



if __name__ == '__main__':
    if os.path.isfile('.env.local'):
        from dotenv import load_dotenv
        load_dotenv('.env.local')
    kill_void()

    # ckpt = 'checkpoints/250707_spkdiarization_win12k_hop6k'
    # ckpt = 'checkpoints/250707_spkdiarization_win12k_hop6k_fix'
    # ckpt = 'checkpoints/250707_spkdiarization_win12k_hop6k_fix_balanced'
    # ckpt = 'checkpoints/250707_spkdiarization_eres2net_win24k_hop12k_fix_balanced'
    # ckpt = 'checkpoints/250711_spkdiarization_eres2net_win24k_hop12k_fix_balanced'
    ckpt = 'checkpoints/250714_spkdiarization_eres2net_win24k_hop12k_fix2_balanced'

    infer_ins = SpkDiarizationInfer('cuda', ckpt)

    # text1 = '造型方面，其实很有意思。第一次试妆的时候，很不一样。'
    # audio = '/mnt/bn/sa-ag-data/liruiqi/code/MegaHuman/inference_api/prompt_api/celebrity_liuyifei.wav'
    # audio, sr = librosa.load(audio, sr=None)

    # text2 = '现在，已经有超过六千万的用户，在土巴兔 A P P 上获得了这份报价。'
    # audio = '/mnt/bn/sa-ag-data/liruiqi/code/MegaHuman/inference_api/prompt_api/口播_王宏源.wav'
    # audio, sr = librosa.load(audio, sr=None)

    # sr = 24000
    # audio1, _ = librosa.load('/mnt/bn/sa-ag-data/liruiqi/code/MegaHuman/inference_api/prompt_api/celebrity_liuyifei.wav', sr=sr)
    # audio2, _ = librosa.load('/mnt/bn/sa-ag-data/liruiqi/code/MegaHuman/inference_api/prompt_api/口播_王宏源.wav', sr=sr)
    # audio = np.concatenate([audio1, audio2], axis=0)

    # audio, sr = librosa.load('/mnt/bn/lq-ads-aigc3/data/speech/processed_0611/duanju/20241219/01-阿刁（65集）/1.mp4/vocal.wav', sr=24000)
    # audio = audio[:int(sr * 60)]

    audio, sr = librosa.load('/mnt/bn/lq-ads-aigc/renyi/tts_datasets_bak/InteractiveDialogue/XYZ_20w/chunfeng_download/shard_00000/xyz_00000112_000.bin.wav', sr=24000)
    audio = audio[:int(sr * 30)]

    correlation = infer_ins.forward_model(audio)
    print('correlation.shape', correlation.shape)
    print('correlation', correlation)

    time = np.linspace(0, len(audio)/sr, len(audio)) # time axis
    fig, ax1 = plt.subplots()
    ax1.plot(time, audio, label='speech waveform')
    ax1.set_xlabel("TIME [s]")
    correlation = np.repeat(correlation, time.shape[0]//correlation.shape[0])
    correlation = np.concatenate([correlation, np.zeros(time.shape[0]-correlation.shape[0])])
    ax2=ax1.twinx()
    ax2.plot(time, correlation, color="r", label = 'correlation')
    plt.yticks([1] ,['voice'])
    ax2.set_ylim([-0.01, 1.01])
    plt.legend()
    plt.savefig('infer_out/asr/figs/spk_diarization.png')

    
    # CUDA_VISIBLE_DEVICES=0 python inference/asr/spk_diarization_infer.py

