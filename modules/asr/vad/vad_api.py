import librosa

from silero_vad import load_silero_vad, get_speech_timestamps

def build_vad_model():
    vad_model = load_silero_vad()
    return vad_model

def run_vad_model(wav, vad_model, sr=16000):
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)

    speech_timestamps = get_speech_timestamps(
        wav,
        vad_model,
        return_seconds=True,  # Return speech timestamps in seconds (default is samples)
    )

    return speech_timestamps

def cut_prefix_wav(wav, duration, vad_model, sr=16000):
    speech_timestamps = run_vad_model(wav, vad_model, sr)
    timestamp_end_idx = -1   # include
    for timestamp_idx, timestamp in enumerate(speech_timestamps):
        if timestamp['end'] > duration:
            timestamp_end_idx = timestamp_idx - 1
            break
    else:
        timestamp_end_idx = len(speech_timestamps) - 1
    if timestamp_end_idx == -1:
        return wav[:int(duration * sr)], duration
    else:
        duration = speech_timestamps[timestamp_end_idx]['end']
        return wav[:int(duration * sr)], duration
        
    


