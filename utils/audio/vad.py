import torch
import numpy as np
import torchaudio

def get_vad_model():
    from silero_vad import load_silero_vad
    return load_silero_vad()

def build_vad_model(device='cpu'):
    vad_model = get_vad_model()
    vad_model.to(device)
    return vad_model

def run_vad_model(wav, sr, vad_model, threshold=0.5):
    from silero_vad import get_speech_timestamps
    if isinstance(wav, np.ndarray):
        wav = torch.from_numpy(wav).float()
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        wav = resampler(wav)
    speech_timestamps = get_speech_timestamps(
        wav,
        vad_model,
        return_seconds=True,  # Return speech timestamps in seconds (default is samples)
        threshold=threshold
    )
    return speech_timestamps
    
def run_vad_trim(wav, sr, vad_model, threshold=0.5):
    speech_timestamps = run_vad_model(wav, sr, vad_model, threshold)
    if len(speech_timestamps) != 0:
        return speech_timestamps[0]['start'], speech_timestamps[-1]['end']
    else:
        return 0, 0

