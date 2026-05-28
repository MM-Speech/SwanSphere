from typing import Union, List, Optional
import numpy as np
import torch
from transformers import WhisperFeatureExtractor as TransformerWhisperFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import TensorType

class WhisperFeatureExtractorV2(TransformerWhisperFeatureExtractor):
    def _torch_extract_fbank_features_cuda(self, waveform: np.array) -> np.ndarray:
        """
        Compute the log-mel spectrogram of the audio using PyTorch's GPU-accelerated STFT implementation with batching,
        yielding results similar to cpu computing with 1e-5 tolerance.
        """
        waveform = waveform.to(torch.float32)
        device = waveform.device
        window = torch.hann_window(self.n_fft, device=device)

        # Note: it would be better to dither the chunked waveform,
        # so overlapping signal does not get the same dithering.
        # But, chunking is happening inside pytorch, so it is here.
        if self.dither != 0.0:
            waveform += self.dither * torch.randn(waveform.shape, dtype=waveform.dtype, device=waveform.device)

        stft = torch.stft(waveform, self.n_fft, self.hop_length, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2

        mel_filters = torch.from_numpy(self.mel_filters).to(device, torch.float32)
        mel_spec = mel_filters.T @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        if waveform.dim() == 2:
            max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            log_spec = torch.maximum(log_spec, max_val - 8.0)
        else:
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        truncation: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = None,
        padding: Optional[str] = "max_length",
        max_length: Optional[int] = None,
        sampling_rate: Optional[int] = None,
        do_normalize: Optional[bool] = None,
        device: Optional[str] = "cpu",
        return_token_timestamps: Optional[bool] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        skip all the crap

        raw_speech: [B, T]
        """
        input_features = self._torch_extract_fbank_features_cuda(raw_speech)

        padded_inputs = BatchFeature({
            'input_features': input_features
        })

        return padded_inputs
