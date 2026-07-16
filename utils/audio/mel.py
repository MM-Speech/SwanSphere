from typing import List, Optional, Sequence
import math
import numpy as np
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
from librosa.core.convert import hz_to_mel
from scipy.io.wavfile import read
import torch
import torch.nn as nn

MAX_WAV_VALUE = 32768.0

def get_mel_len(wav_len, hop_size):
    return (wav_len + hop_size - 1) // hop_size


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log10(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log10(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


class MelNet(nn.Module):
    def __init__(self, hparams, device='cpu'):
        super().__init__()
        self.n_fft = hparams['fft_size']
        self.num_mels = hparams['audio_num_mel_bins']
        self.sampling_rate = hparams['audio_sample_rate']
        self.hop_size = hparams['hop_size']
        self.win_size = hparams['win_size']
        self.fmin = hparams['fmin']
        self.fmax = hparams['fmax']
        self.device = device

        mel = librosa_mel_fn(sr=self.sampling_rate, n_fft=self.n_fft, n_mels=self.num_mels, fmin=self.fmin,
                             fmax=self.fmax)
        self.mel_basis = torch.from_numpy(mel).float().to(self.device)
        self.hann_window = torch.hann_window(self.win_size).to(self.device)

    def to(self, device, **kwagrs):
        super().to(device=device, **kwagrs)
        self.mel_basis = self.mel_basis.to(device)
        self.hann_window = self.hann_window.to(device)
        self.device = device

    def forward(self, y, center=False, return_complex=False):
        if isinstance(y, np.ndarray):
            y = torch.FloatTensor(y)
        if len(y.shape) == 1:
            y = y.unsqueeze(0)
        y = y.clamp(min=-1., max=1.)
        if y.device != self.device:
            y = y.to(self.device)

        pad_length = math.ceil(y.shape[1] / self.hop_size) * self.hop_size - y.shape[1]
        y = torch.nn.functional.pad(y.unsqueeze(1),
                                    [int((self.n_fft - self.hop_size) / 2),
                                    int((self.n_fft - self.hop_size) / 2 + pad_length)],
                                    mode='reflect')
        y = y.squeeze(1)

        spec = torch.stft(y, self.n_fft, hop_length=self.hop_size, win_length=self.win_size, window=self.hann_window,
                          center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
        if not return_complex:
            spec = torch.view_as_real(spec)
            spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))     # [B, n_fft, T]
            spec = torch.matmul(self.mel_basis, spec)
            spec = spectral_normalize_torch(spec)
            spec = spec.transpose(1, 2)     # [B, T, n_fft]
        else:
            B, C, T, _ = spec.shape
            spec = spec.transpose(1, 2)  # [B, T, n_fft, 2]
        return spec

    def __call__(self, y, center=False, return_complex=False):
        return self.forward(y, center, return_complex)


class MultiResolutionMelLoss(nn.Module):
    def __init__(self, hparams, loss_fn: nn.Module = nn.L1Loss()):
        super().__init__()
        self.mel_nets = nn.ModuleList([MelNet(hp) for hp in hparams])
        self.loss_fn = loss_fn

    def forward(self, y_pred, y_ref):
        loss = 0.0
        for mel_net in self.mel_nets:
            mel_pred = mel_net(y_pred)
            mel_ref = mel_net(y_ref)
            loss = loss + self.loss_fn(mel_pred, mel_ref)
        return loss / max(1, len(self.mel_nets))


class MultiResolutionMultiBandMelLoss(nn.Module):
    def __init__(
        self,
        hparams_list: List[dict],
        band_edges_hz: Sequence[float],
        band_weights: Optional[Sequence[float]] = None,
        loss_fn: nn.Module = nn.L1Loss(),
    ):
        super().__init__()
        if len(band_edges_hz) < 2:
            raise ValueError("band_edges_hz must contain at least two values")
        self.mel_nets = nn.ModuleList([MelNet(hp) for hp in hparams_list])
        self.loss_fn = loss_fn
        self.band_edges_hz = sorted([float(x) for x in band_edges_hz])
        self.num_bands = len(self.band_edges_hz) - 1
        if band_weights is None or len(band_weights) == 0:
            band_weights = [1.0] * self.num_bands
        if len(band_weights) != self.num_bands:
            raise ValueError("band_weights length must equal len(band_edges_hz) - 1")
        self.band_weights = [float(w) for w in band_weights]
        self.band_slices_per_melnet = [
            self._compute_band_slices_for_melnet(mel_net) for mel_net in self.mel_nets
        ]

    def _compute_band_slices_for_melnet(self, mel_net):
        fmin = float(mel_net.fmin)
        fmax = float(mel_net.fmax)
        num_mels = int(mel_net.num_mels)
        mel_fmin = hz_to_mel(fmin)
        mel_fmax = hz_to_mel(fmax)
        mel_range = max(float(mel_fmax - mel_fmin), 1.0e-9)

        def hz_to_pos(freq_hz: float) -> float:
            freq_hz = max(fmin, min(fmax, float(freq_hz)))
            return max(0.0, min(1.0, float((hz_to_mel(freq_hz) - mel_fmin) / mel_range)))

        band_slices = []
        for low_hz, high_hz in zip(self.band_edges_hz[:-1], self.band_edges_hz[1:]):
            start = int(math.floor(hz_to_pos(low_hz) * num_mels))
            end = int(math.ceil(hz_to_pos(high_hz) * num_mels))
            start = max(0, min(start, num_mels - 1))
            end = max(1, min(end, num_mels))
            if end <= start:
                end = min(start + 1, num_mels)
            band_slices.append((start, end))
        return band_slices

    def forward(self, y_pred, y_ref):
        total_loss = 0.0
        weight_sum = max(sum(self.band_weights), 1.0e-8)
        for mel_net, band_slices in zip(self.mel_nets, self.band_slices_per_melnet):
            mel_pred = mel_net(y_pred)
            mel_ref = mel_net(y_ref)
            mel_loss = 0.0
            for (start, end), weight in zip(band_slices, self.band_weights):
                mel_loss = mel_loss + float(weight) * self.loss_fn(
                    mel_pred[..., start:end],
                    mel_ref[..., start:end],
                )
            total_loss = total_loss + mel_loss / weight_sum
        return total_loss / max(1, len(self.mel_nets))


## below can be used in one gpu, but not ddp
mel_basis = {}
hann_window = {}


def mel_spectrogram(y, hparams, center=False, complex=False):  # y should be a tensor with shape (b,wav_len)
    # hop_size: 512  # For 22050Hz, 275 ~= 12.5 ms (0.0125 * sample_rate)
    # win_size: 2048  # For 22050Hz, 1100 ~= 50 ms (If None, win_size: fft_size) (0.05 * sample_rate)
    # fmin: 55  # Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
    # fmax: 10000  # To be increased/reduced depending on data.
    # fft_size: 2048  # Extra window size is filled with 0 paddings to match this parameter
    # n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax,
    n_fft = hparams['fft_size']
    num_mels = hparams['audio_num_mel_bins']
    sampling_rate = hparams['audio_sample_rate']
    hop_size = hparams['hop_size']
    win_size = hparams['win_size']
    fmin = hparams['fmin']
    fmax = hparams['fmax']
    if isinstance(y, np.ndarray):
        y = torch.FloatTensor(y)
    if len(y.shape) == 1:
        y = y.unsqueeze(0)
    y = y.clamp(min=-1., max=1.)
    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax) + '_' + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), [int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)],
                                mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=complex)

    if not complex:
        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
        spec = torch.matmul(mel_basis[str(fmax) + '_' + str(y.device)], spec)
        spec = spectral_normalize_torch(spec)
    else:
        B, C, T, _ = spec.shape
        spec = spec.transpose(1, 2)  # [B, T, n_fft, 2]
    return spec
