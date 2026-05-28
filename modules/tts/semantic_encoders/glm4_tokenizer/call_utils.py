import os
import io
import glob
import math
import tarfile
import torch
import torchaudio
import safetensors
from modules.tts.semantic_encoders.glm4_tokenizer.configuration_whisper import WhisperVQConfig
from modules.tts.semantic_encoders.glm4_tokenizer.modeling_whisper import WhisperVQEncoder, WhisperVQForConditionalGeneration
from transformers import WhisperFeatureExtractor, WhisperTokenizerFast


def load_quantize_encoder(model_path):
    config = WhisperVQConfig.from_pretrained(model_path)
    config.quantize_encoder_only = True
    model = WhisperVQEncoder(config)
    state_dict = {}
    for path in glob.glob(os.path.join(model_path, "model*.safetensors")):
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("model.encoder."):
                    new_key = key[len("model.encoder."):]
                    if new_key.startswith("layer_norm"):
                        continue
                    if new_key.startswith("layers"):
                        layer_id = int(new_key.split(".")[1])
                        if layer_id >= config.quantize_position:
                            continue
                    state_dict[new_key] = f.get_tensor(key)
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()
    return model


_resample_buffer: dict[int, torchaudio.transforms.Resample] = {}


def extract_speech_token(model: WhisperVQEncoder, feature_extractor: WhisperFeatureExtractor, utts):
    with torch.no_grad():
        audios, indices = [], []
        for idx, utt in enumerate(utts):
            if isinstance(utt, tuple):
                audio, sample_rate = utt
            else:
                audio, sample_rate = torchaudio.load(utt)
            audio = audio.cuda()
            if sample_rate != 16000:
                if sample_rate not in _resample_buffer:
                    _resample_buffer[sample_rate] = torchaudio.transforms.Resample(
                        orig_freq=sample_rate,
                        new_freq=16000
                    ).to('cuda')
                audio = _resample_buffer[sample_rate](audio)
            # if audio.shape[0] > 1:
            #     audio = audio[:1]
            audio = audio[0]
            audio = audio.cpu().numpy()
            time_step = 0
            while time_step * 16000 < audio.shape[0]:
                audio_segment = audio[time_step * 16000: (time_step + 30) * 16000]
                audios.append(audio_segment)
                indices.append(idx)
                time_step += 30
        pooling_kernel_size = model.config.pooling_kernel_size or 1
        stride = model.conv1.stride[0] * model.conv2.stride[0] * pooling_kernel_size * feature_extractor.hop_length
        all_speech_tokens = [[] for _ in range(len(utts))]
        batch_size = 128
        for start in range(0, len(audios), batch_size):
            features = feature_extractor(audios[start: start + batch_size], sampling_rate=16000,
                                         return_attention_mask=True, return_tensors="pt", device='cuda',
                                         padding="longest", pad_to_multiple_of=stride)
            features = features.to(device="cuda")
            outputs = model(**features)
            speech_tokens = outputs.quantized_token_ids
            attention_mask = features.attention_mask[:, ::model.conv1.stride[0] * model.conv2.stride[0]]
            attention_mask = attention_mask[:, ::model.config.pooling_kernel_size]
            assert attention_mask.shape == speech_tokens.shape
            for i in range(len(speech_tokens)):
                idx = indices[start + i]
                speech_token = speech_tokens[i][attention_mask[i].bool()].tolist()
                all_speech_tokens[idx].extend(speech_token)
        return all_speech_tokens
    

def sequence_mask(lengths, maxlen=None, dtype=torch.bool):
    if maxlen is None:
        maxlen = lengths.max()
    mask = ~(torch.ones((len(lengths), maxlen)).to(lengths.device).cumsum(dim=1).t() > lengths).t()
    mask.type(dtype)
    return mask
    
@torch.no_grad()
def extract_speech_token_v1(model: WhisperVQEncoder, feature_extractor: WhisperFeatureExtractor, wavs, sample_rate, wav_lens, device='cuda'):
    # wavs [B, T]
    # wav_lens [B]
    # wav_mask [B, T]
    if sample_rate != 16000:
        if sample_rate not in _resample_buffer:
            _resample_buffer[sample_rate] = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000).to(device)
        wavs = _resample_buffer[sample_rate](wavs)
        wav_lens = (wav_lens * 16000 / sample_rate).int()
    wav_mask = sequence_mask(wav_lens)

    pooling_kernel_size = model.config.pooling_kernel_size or 1
    stride = model.conv1.stride[0] * model.conv2.stride[0] * pooling_kernel_size * feature_extractor.hop_length

    bsz, orig_len = wavs.shape
    chunk_size = 16000 * 30
    if orig_len > chunk_size:
        if orig_len % chunk_size > 0:
            wavs = torch.nn.functional.pad(wavs, (0, chunk_size - orig_len % chunk_size), value=0.0)
            wav_mask = torch.nn.functional.pad(wav_mask, (0, chunk_size - orig_len % chunk_size), value=0.0)
        wavs = torch.cat(torch.chunk(wavs, wavs.shape[1] // chunk_size, dim=1), dim=0)  # [B, NT] -> [NB, T]
        wav_mask = torch.cat(torch.chunk(wav_mask, wav_mask.shape[1] // chunk_size, dim=1), dim=0)
    if orig_len % stride != 0:
        wavs = torch.nn.functional.pad(wavs, (0, stride - orig_len % stride), value=0.0)
        wav_mask = torch.nn.functional.pad(wav_mask, (0, stride - orig_len % stride), value=0.0)
        orig_len += stride - orig_len % stride
    
    wavs = wavs.cpu().numpy()   # [B, T]

    features = feature_extractor(wavs, sampling_rate=16000, return_attention_mask=True, return_tensors='pt', device='cuda', 
                                 padding='longest', pad_to_multiple_of=stride)
    features = features.to(device)
    wav_mask = wav_mask[:, ::feature_extractor.hop_length]
    outputs = model(
        input_features=features['input_features'],
        attention_mask=wav_mask.int()
    )
    speech_tokens = outputs.quantized_token_ids
    attention_mask = wav_mask[:, ::model.conv1.stride[0] * model.conv2.stride[0]]
    attention_mask = attention_mask[:, ::model.config.pooling_kernel_size]
    assert attention_mask.shape == speech_tokens.shape

    if orig_len > chunk_size:
        speech_tokens = torch.cat(torch.chunk(speech_tokens, speech_tokens.shape[0] // bsz, dim=0), dim=1)   # [NB, T] -> [B, NT]
        attention_mask = torch.cat(torch.chunk(attention_mask, attention_mask.shape[0] // bsz, dim=0), dim=1)   # [NB, T] -> [B, NT]
        speech_tokens = speech_tokens[:, :orig_len//stride]

    return speech_tokens, attention_mask


@torch.no_grad()
def extract_speech_token_v2(model: WhisperVQEncoder, feature_extractor: WhisperFeatureExtractor, wavs, sample_rate, wav_lens, device='cuda'):
    # wavs [B, T]
    # wav_lens [B]
    # wav_mask [B, T]
    if sample_rate != 16000:
        if sample_rate not in _resample_buffer:
            _resample_buffer[sample_rate] = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000).to(device)
        wavs = _resample_buffer[sample_rate](wavs)
        wav_lens = (wav_lens * 16000 / sample_rate).int()
    wav_mask = sequence_mask(wav_lens)

    pooling_kernel_size = model.config.pooling_kernel_size or 1
    stride = model.conv1.stride[0] * model.conv2.stride[0] * pooling_kernel_size * feature_extractor.hop_length

    bsz, orig_len = wavs.shape
    chunk_size = int(16000 * 30)
    if orig_len > chunk_size:
        if orig_len % chunk_size != 0:
            wavs = torch.nn.functional.pad(wavs, (0, chunk_size - orig_len % chunk_size), value=0.0)
            wav_mask = torch.nn.functional.pad(wav_mask, (0, chunk_size - orig_len % chunk_size), value=0.0)
        wavs = torch.cat(torch.chunk(wavs, wavs.shape[1] // chunk_size, dim=1), dim=0)  # [B, NT] -> [NB, T]
        wav_mask = torch.cat(torch.chunk(wav_mask, wav_mask.shape[1] // chunk_size, dim=1), dim=0)
    if orig_len % stride != 0:
        wavs = torch.nn.functional.pad(wavs, (0, stride - orig_len % stride), value=0.0)
        wav_mask = torch.nn.functional.pad(wav_mask, (0, stride - orig_len % stride), value=0.0)
        orig_len += stride - orig_len % stride
    
    features = feature_extractor(wavs)
    wav_mask = wav_mask[:, ::feature_extractor.hop_length]
    outputs = model(
        input_features=features['input_features'],
        attention_mask=wav_mask.int()
    )
    speech_tokens = outputs.quantized_token_ids
    attention_mask = wav_mask[:, ::model.conv1.stride[0] * model.conv2.stride[0]]
    attention_mask = attention_mask[:, ::model.config.pooling_kernel_size]
    assert attention_mask.shape == speech_tokens.shape

    if orig_len > chunk_size:
        speech_tokens = torch.cat(torch.chunk(speech_tokens, speech_tokens.shape[0] // bsz, dim=0), dim=1)   # [NB, T] -> [B, NT]
        attention_mask = torch.cat(torch.chunk(attention_mask, attention_mask.shape[0] // bsz, dim=0), dim=1)   # [NB, T] -> [B, NT]
        speech_tokens = speech_tokens[:, :orig_len//stride]

    return speech_tokens, attention_mask

if __name__ == '__main__':
    import librosa
    feature_extractor = WhisperFeatureExtractor.from_pretrained("checkpoints/glm-4-voice-tokenizer")
    from modules.tts.semantic_encoders.glm4_tokenizer.feature_extraction_whisper import WhisperFeatureExtractorV2
    feature_extractorv2 = WhisperFeatureExtractorV2.from_pretrained("checkpoints/glm-4-voice-tokenizer")
    model = WhisperVQEncoder.from_pretrained("checkpoints/glm-4-voice-tokenizer").eval()
    model.to('cuda')

    wav, sr = librosa.load('/mnt/bn/sa-ag-data/liruiqi/code/MegaHuman/inference_api/prompt_api/celebrity_liuyifei.wav', sr=24000)
    wav = torch.from_numpy(wav).to('cuda')
    wav = torch.cat([wav] * 20)
    wavs = torch.stack([wav] * 5)
    wavs = wavs[:, :int(sr * 60 - 1920)]
    wav_mask = torch.ones_like(wavs)

    wavs[0, -(24000*10-1920):] = 0
    wav_mask[0, -(24000*10-1920):] = 0

    wav_lens = wav_mask.sum(1)

    speech_tokens, attention_mask = extract_speech_token_v1(model, feature_extractor, wavs, sr, wav_lens, 'cuda')
    print('speech_tokens', speech_tokens)

    speech_tokens_v2, attention_mask_v2 = extract_speech_token_v2(model, feature_extractorv2, wavs, sr, wav_lens, 'cuda')
    print('speech_tokens_v2', speech_tokens_v2)

    print((speech_tokens == speech_tokens_v2).all())
    print((attention_mask == attention_mask_v2).all())
