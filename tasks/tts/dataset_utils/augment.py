import os
import glob
import random

from copy import deepcopy
import numpy as np
import torch
import librosa

class SpeechAugment:
    def __init__(self, wav_add_noise, wav_add_effect, musan_dir, 
                 noise_prob=0.5, effect_prob=0.5, noise_snr=(6.0, 20.0), with_speech=False, with_music=True):
        self.wav_add_noise = wav_add_noise
        self.wav_add_effect = wav_add_effect
        self.musan_dir = musan_dir
        if musan_dir is not None and os.path.isdir(musan_dir):
            self.noise_paths = glob.glob(f"{musan_dir}/noise/*/*.wav")
            if with_speech:
                self.noise_paths += glob.glob(f"{musan_dir}/speech/*/*.wav")
            if with_music:
                self.noise_paths += glob.glob(f"{musan_dir}/music/*/*.wav")
        self.noise_snr = noise_snr
        self.noise_prob = noise_prob
        self.effect_prob = effect_prob

        self.tasks = ['noise', 'gain', 'distortion', 'reverb']
    
    def __call__(self, wav: np.ndarray, sample_rate: int = 16000, return_pt=False):
        if isinstance(wav, torch.Tensor):
            wav = wav.numpy()
            return_pt = True
        wav_len = wav.shape[0]
        import pedalboard
        tasks = deepcopy(self.tasks)
        np.random.shuffle(tasks)
        for task in tasks:
            if task == 'noise' and self.wav_add_noise and random.random() < self.noise_prob:
                noise_path = np.random.choice(self.noise_paths)
                noise, _ = librosa.load(noise_path, sr=sample_rate)
                wav = SpeechAugment.add_noise(wav, noise, self.noise_snr)
            elif task == 'gain' and self.wav_add_effect and random.random() < self.effect_prob:
                board = pedalboard.Pedalboard([
                    pedalboard.Gain(gain_db=np.random.uniform(0.5, 1.5)),
                ])
                wav = board(wav, sample_rate)
            elif task == 'distortion' and self.wav_add_effect and random.random() < self.effect_prob:
                board = pedalboard.Pedalboard([
                    pedalboard.Distortion(drive_db=np.random.uniform(15, 30)),
                ])
                wav = board(wav, sample_rate)
            elif task == 'reverb' and self.wav_add_effect and random.random() < self.effect_prob:
                board = pedalboard.Pedalboard([
                    pedalboard.Reverb(
                        room_size=np.random.uniform(0.1, 0.9), 
                        damping=np.random.uniform(0.1, 0.9), 
                        wet_level=np.random.uniform(0.1, 0.9), 
                        dry_level=np.random.uniform(0.1, 0.9)
                    )
                ])
                wav = board(wav, sample_rate)
        if wav.shape[0] > wav_len:
            wav = wav[:wav_len]
        elif wav.shape[0] < wav_len:
            wav = np.concatenate([wav, np.zeros(wav_len - wav.shape[0])], axis=0)
        if return_pt:
            wav = torch.from_numpy(wav)
        return wav
            

    @staticmethod
    def add_noise(clean_wav, noise_wav, noise_snr):
        if isinstance(noise_snr, tuple):
            snr = np.random.rand() * (noise_snr[1] - noise_snr[0]) + noise_snr[0]
        elif isinstance(noise_snr, float):
            snr = noise_snr
        clean_rms = np.sqrt(np.mean(np.square(clean_wav), axis=-1))
        if len(clean_wav) > len(noise_wav):
            ratio = int(np.ceil(len(clean_wav)/len(noise_wav)))
            noise_wav = np.concatenate([noise_wav for _ in range(ratio)])
        if len(clean_wav) < len(noise_wav):
            start = np.random.choice(len(noise_wav) - len(clean_wav))
            noise_wav = noise_wav[start: start + len(clean_wav)]
        clean_start = random.randint(0, int(len(clean_wav) * 0.4))
        clean_end = random.randint(int(len(clean_wav) * 0.6), len(clean_wav))
        noise_start = random.randint(0, len(noise_wav) - clean_end + clean_start)
        noise_end = noise_start + clean_end - clean_start
        noise_rms = np.sqrt(np.mean(np.square(noise_wav), axis=-1)) + 1e-5
        adjusted_noise_rms = clean_rms / (10 ** (snr / 20) + 1e-5)
        adjusted_noise_wav = noise_wav * (adjusted_noise_rms / noise_rms)
        mixed = clean_wav
        mixed[clean_start: clean_end] += adjusted_noise_wav[noise_start: noise_end]
        # Avoid clipping noise
        max_int16 = np.iinfo(np.int16).max
        min_int16 = np.iinfo(np.int16).min
        if mixed.max(axis=0) > max_int16 or mixed.min(axis=0) < min_int16:
            if mixed.max(axis=0) >= abs(mixed.min(axis=0)):
                reduction_rate = max_int16 / mixed.max(axis=0)
            else:
                reduction_rate = min_int16 / mixed.min(axis=0)
            mixed = mixed * reduction_rate
        return mixed
