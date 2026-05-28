import os
import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

def build_emotion2vec_model(device='cuda'):
    if isinstance(device, torch.device):
        device_index = device.index
        device = device.type
        if device_index is not None:
            device = f"{device}:{device_index}"
    inference_pipeline = pipeline(
        task=Tasks.emotion_recognition,
        model="iic/emotion2vec_plus_large",
        device=device
    )
    return inference_pipeline

def run_emotion2vec_model(audio_file, emotion_model):
    rec_result = emotion_model(audio_file, granularity="utterance", extract_embedding=False)
    return rec_result

if __name__ == '__main__':
    inference_pipeline = pipeline(
        task=Tasks.emotion_recognition,
        # model="iic/emotion2vec_plus_large")
        model="pretrained_models/emotion2vec_plus_large")
    
    rec_result = inference_pipeline('https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav', granularity="utterance", extract_embedding=False)
    print(rec_result)
