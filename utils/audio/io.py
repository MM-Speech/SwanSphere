import subprocess
import io

import numpy as np
from scipy.io import wavfile
import pyloudnorm as pyln


def save_wav(wav, path, sr, norm=False):
    wav = wav.astype(float)
    if norm:
        meter = pyln.Meter(sr)  # create BS.1770 meter
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -18.0)
        if np.abs(wav).max() >= 1:
            wav = wav / np.abs(wav).max() * 0.95
    wav = wav * 32767
    wavfile.write(path[:-4] + '.wav', sr, wav.astype(np.int16))
    if path[-4:] == '.mp3':
        to_mp3(path[:-4])


def to_mp3(out_path):
    if out_path[-4:] == '.wav':
        out_path = out_path[:-4]
    subprocess.check_call(
        f'ffmpeg -threads 1 -loglevel error -i "{out_path}.wav" -vn -b:a 192k -y -hide_banner -async 1 "{out_path}.mp3"',
        shell=True, stdin=subprocess.PIPE)
    subprocess.check_call(f'rm -f "{out_path}.wav"', shell=True)


def to_wav_bytes(wav, sr, norm=False):
    wav = wav.astype(float)
    if norm:
        meter = pyln.Meter(sr)  # create BS.1770 meter
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -18.0)
        if np.abs(wav).max() >= 1:
            wav = wav / np.abs(wav).max() * 0.95
    wav = wav * 32767
    bytes_io = io.BytesIO()
    wavfile.write(bytes_io, sr, wav.astype(np.int16))
    return bytes_io.getvalue()

def wav_bytes_to_mp3_bytes(wav_bytes):
    from pydub import AudioSegment
    wav_io = io.BytesIO(wav_bytes)
    audio = AudioSegment.from_wav(wav_io)
    mp3_io = io.BytesIO()
    audio.export(mp3_io, format="mp3")
    mp3_bytes = mp3_io.getvalue()
    return mp3_bytes


def save_mp3_bytes(wav_bytes, path):
    with open(path[:-4] + '.mp3', 'wb') as file:
        file.write(wav_bytes)


def save_wav_bytes(wav_bytes, path):
    with open(path[:-4] + '.wav', 'wb') as file:
        file.write(wav_bytes)
    if path[-4:] == '.mp3':
        to_mp3(path[:-4])
