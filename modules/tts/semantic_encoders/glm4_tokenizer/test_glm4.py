import json
import os.path
import tempfile
import sys
import re
import uuid
import requests
from argparse import ArgumentParser

import torchaudio
from transformers import WhisperFeatureExtractor, AutoTokenizer
from modules.vc.glm4_tokenizer.modeling_whisper import WhisperVQEncoder


sys.path.insert(0, "./cosyvoice")
sys.path.insert(0, "./third_party/Matcha-TTS")

from modules.tts.semantic_encoders.glm4_tokenizer.utils import extract_speech_token

import torch

audio_token_pattern = re.compile(r"<\|audio_(\d+)\|>")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tokenizer-path", type= str, default="checkpoints/glm-4-voice-tokenizer")
    args = parser.parse_args()

    device = "cuda"
    whisper_model, feature_extractor = None, None


    def initialize_fn():
        global feature_extractor, whisper_model

        # Speech tokenizer
        whisper_model = WhisperVQEncoder.from_pretrained(args.tokenizer_path).eval().to(device)
        feature_extractor = WhisperFeatureExtractor.from_pretrained(args.tokenizer_path)


    def clear_fn():
        return [], [], '', '', '', None, None


    def inference_fn(audio_path):
        audio, sample_rate = torchaudio.load(audio_path)
        audio = audio
        resampler = torchaudio.transforms.Resample(
                        orig_freq=sample_rate,
                        new_freq=16000
                    )
        audio = resampler(audio)
        audio = torch.nn.functional.pad(audio, (0, 478500-audio.size(1)))
        audio = audio.numpy()

        print(audio.shape)

        pooling_kernel_size = whisper_model.config.pooling_kernel_size or 1
        print(pooling_kernel_size, whisper_model.conv1.stride[0], whisper_model.conv2.stride[0], feature_extractor.hop_length)
        stride = whisper_model.conv1.stride[0] * whisper_model.conv2.stride[0] * pooling_kernel_size * feature_extractor.hop_length
        print(stride)
        features = feature_extractor(audio, sampling_rate=16000,
                                         return_attention_mask=True, return_tensors="pt", device='cuda',
                                         padding="longest", pad_to_multiple_of=stride)
        features = features.to(device="cuda")
        outputs = whisper_model(**features)
        speech_tokens = outputs.quantized_token_ids
        print(speech_tokens.shape)
        return speech_tokens
    
    initialize_fn()
    speech_tokens = inference_fn('/mnt/bn/sa-ag-data/jiangziyue/MegaHuman/inference_result/prompt_api/prompt_chou.wav')
    print(speech_tokens)