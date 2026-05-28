from dataclasses import dataclass
from typing import Any, Optional, Tuple, List
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_tagger_model(hparams, init_pretrained=False):
    config = ModelArgs(
        init_pretrained=init_pretrained,
        audio_encoder_ckpt='checkpoints/wavlm/WavLM-Large.pt'
    )
    model = TaggerModel(config)
    return model
    

@dataclass
class ModelArgs:
    # audio
    # fft_size: int = 800
    # audio_num_mel_bins: int = 160
    # audio_sample_rate: int = 16000
    # hop_size: int = 160
    # win_size: int = 800
    # fmin: int = 0
    # fmax: int = 8000
    
    # encoder
    # dim: int = 1024
    
    audio_encoder_type: str = 'wavlm'
    audio_encoder_ckpt: str = None
    init_pretrained: bool = True
    
    
class TaggerTokenizer:
    def __init__(self):
        self.age = ['Child', 'Teenager', 'Youth-Adult', 'Middle-aged', 'Elderly']
        self.emotion = ["UNKNOWN", "NEUTRAL", "HAPPY", "ANGRY", "SAD", "ANXIOUS", "APOLOGETIC", "ASSERTIVE", "CONCERNED", "CONFUSED", "CONTEMPT", "DISGUSTED", "EMPHASIS", "ENCOURAGING", "ENUNCIATED", "EXCITED", "LAUGHING", "NO-AGREEMENT", "SARCASTIC", "SLEEPY", "SURPRISED", "WHISPER", "WORRIED"]
        self.gender = ['male', 'female']

        self.age2idx = {l:i for i, l in enumerate(self.age)}
        self.emotion2idx = {l:i for i, l in enumerate(self.emotion)}
        self.gender2idx = {l:i for i, l in enumerate(self.gender)}
        
    def encode(self, labels: List[str], task='age'):
        number_input = False
        if not isinstance(labels, list):
            number_input = True
            labels = [labels]
        encoded = []
        for l in labels:
            if task == 'age':
                idx = self.age2idx[l]
            elif task == 'emotion':
                idx = self.emotion2idx[l]
            elif task == 'gender':
                idx = self.gender2idx[l]
            encoded.append(idx)
        if number_input:
            encoded = encoded[0]
        return encoded
    
    def decode(self, idxs: List[int], task='age'):
        number_input = False
        if not isinstance(idxs, list):
            number_input = True
            idxs = [idxs]
        decoded = []
        for idx in idxs:
            if task == 'age':
                l = self.age[idx]
            elif task == 'emotion':
                l = self.emotion[idx]
            elif task == 'gender':
                l = self.gender[idx]
            decoded.append(l)
        if number_input:
            decoded = decoded[0]
        return decoded
    
    
class AttnPooling(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.attn = nn.Conv1d(hidden_size, 1, (3,), padding='same')

    def forward(self, x, attn_mask=None):
        # x [B, T, C]
        logits = self.attn(x.transpose(1, 2)).transpose(1, 2)
        if attn_mask is not None:
            if len(attn_mask.shape) == 2:
                attn_mask = attn_mask[..., None]
            logits.masked_fill_(attn_mask == 0, -torch.inf)
        x = x * torch.softmax(logits, dim=1)
        x = x.sum(dim=1)
        return x # [B, C]
    
    
class ClassificationDecoder(nn.Module):
    def __init__(self, dim, n_class):
        super().__init__()
        self.attn_pooling = AttnPooling(dim)
        self.out = nn.Linear(dim, n_class, bias=False)
    
    def forward(self, x, x_mask):
        x = self.attn_pooling(x, x_mask)
        x = self.out(x)
        return x
        
        
class RegressionDecoder(nn.Module):
    def __init__(self, dim, init_bias: float = None):
        super().__init__()
        self.attn_pooling = AttnPooling(dim)
        self.out = nn.Linear(dim, 1)
        if init_bias is not None:
            nn.init.constant_(self.out.bias, init_bias)
    
    def forward(self, x, x_mask):
        x = self.attn_pooling(x, x_mask)
        x = self.out(x)
        return x
    
    
class TaggerModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        from modules.asr.wavlm.WavLM import WavLM, WavLMConfig
        if config.init_pretrained:
            checkpoint = torch.load(config.audio_encoder_ckpt)
            cfg = WavLMConfig(checkpoint['cfg'])
            model = WavLM(cfg)
            model.load_state_dict(checkpoint['model'])
        else:
            cfg = json.load(open('checkpoints/wavlm/WavLM-Large-Config.json'))
            cfg = WavLMConfig(cfg)
            model = WavLM(cfg)
        self.audio_encoder_dim = cfg.encoder_embed_dim
        self.audio_encoder = model
        self.audio_encoder_hopsize = 320
        self.audio_encoder_sample_rate = 16000
        
        self.age_decoder = ClassificationDecoder(self.audio_encoder_dim, 5)     # Child, Teenager, Youth-Adult, Middle-aged, Elderly
        self.gender_decoder = ClassificationDecoder(self.audio_encoder_dim, 2)  # male, female
        self.emotion_decoder = ClassificationDecoder(self.audio_encoder_dim, 23)    # 
        self.pitch_decoder = RegressionDecoder(self.audio_encoder_dim, math.log1p(stats_meta['pitch']['mean']))
        self.pitch_std_decoder = RegressionDecoder(self.audio_encoder_dim, stats_meta['pitch_std']['mean'])
        self.speed_decoder = RegressionDecoder(self.audio_encoder_dim, stats_meta['speed']['mean'])
        
    def forward_encoder(self, wavs, wav_mask, do_checkpoint=False):
        feat, feat_padding_mask = self.audio_encoder.extract_features(wavs, padding_mask=~(wav_mask.bool()), do_checkpoint=do_checkpoint)    # [B, T, C]
        return feat, (~(feat_padding_mask.bool())).to(feat)
    
    def forward(self, wav, wav_mask, do_checkpoint=False):
        x, x_mask = self.forward_encoder(wav, wav_mask, do_checkpoint)
        
        age_logits = self.age_decoder(x, x_mask)
        gender_logits = self.gender_decoder(x, x_mask)
        emotion_logits = self.emotion_decoder(x, x_mask)
        pitch = self.pitch_decoder(x, x_mask)
        pitch_std = self.pitch_std_decoder(x, x_mask)
        speed = self.speed_decoder(x, x_mask)
        
        return {
            'age_logits': age_logits,
            'gender_logits': gender_logits,
            'emotion_logits': emotion_logits,
            'pitch': pitch,
            'pitch_std': pitch_std,
            'speed': speed,
        }
        
    def inference(self, wav, wav_mask):
        if not hasattr(self, 'tagger_tokenizer'):
            self.tagger_tokenizer = TaggerTokenizer()
        model_outputs = self.forward(wav, wav_mask)
        bsz = wav.shape[0]
        
        outputs = []
        for i in range(bsz):

            result = {}
            
            result['age'] = {'probs': {}}
            age_logits = model_outputs['age_logits'][i]
            age_probs = torch.softmax(age_logits).cpu().numpy()
            age_pred = torch.argmax(age_probs).item()
            for j in range(len(self.tagger_tokenizer.age)):
                result['age']['probs'][self.tagger_tokenizer.age[j]] = age_probs[j]
            result['age']['pred'] = self.tagger_tokenizer.decode(age_pred)

            result['gender'] = {'probs': {}}
            gender_logits = model_outputs['gender_logits'][i]
            gender_probs = torch.softmax(gender_logits).cpu().numpy()
            gender_pred = torch.argmax(gender_probs).item()
            for j in range(len(self.tagger_tokenizer.gender)):
                result['gender']['probs'][self.tagger_tokenizer.gender[j]] = gender_probs[j]
            result['gender']['pred'] = self.tagger_tokenizer.decode(gender_pred)
            
            result['emotion'] = {'probs': {}}
            emotion_logits = model_outputs['emotion_logits'][i]
            emotion_probs = torch.softmax(emotion_logits).cpu().numpy()
            emotion_pred = torch.argmax(emotion_probs).item()
            for j in range(len(self.tagger_tokenizer.emotion)):
                result['emotion']['probs'][self.tagger_tokenizer.emotion[j]] = emotion_probs[j]
            result['emotion']['pred'] = self.tagger_tokenizer.decode(emotion_pred)
            
            pitch_pred = model_outputs['pitch'][i]
            pitch_pred = torch.expm1(pitch_pred).item()
            result['pitch'] = pitch_pred
            
            pitch_std_pred = model_outputs['pitch_std'][i].item()
            result['pitch_std'] = pitch_std_pred
            
            speed_pred = model_outputs['speed'][i].item()
            result['speed'] = speed_pred
            
            outputs.append(result)
            
        return outputs
        
        
stats_meta = {
    "pitch": {
        "min": 0.0,
        "max": 846.642,
        "mean": 164.34912500282118,
        "median": 153.79
    },
    "pitch_std": {
        "min": 0.0,
        "max": 0.5,
        "mean": 0.18259202257758814,
        "median": 0.181
    },
    "speed": {
        "min": 0.0,
        "max": 10.0,
        "mean": 4.177214022692233,
        "median": 4.2
    },
    "duration": {
        "min": 0.06,
        "max": 25.0,
        "mean": 7.666654049531603,
        "median": 6.248
    },
    "speech_duration": {
        "min": 0.128,
        "max": 25.0,
        "mean": 7.432436383908156,
        "median": 6.09
    },
    "syllable_num": {
        "min": 0.0,
        "max": 187.0,
        "mean": 30.90022173745993,
        "median": 26.0
    },
    "emotion": {
        "ANGRY": {
            "count": 364888,
            "proportion": 0.007452068035268618
        },
        "ANXIOUS": {
            "count": 240,
            "proportion": 4.901493961063308e-06
        },
        "APOLOGETIC": {
            "count": 240,
            "proportion": 4.901493961063308e-06
        },
        "ASSERTIVE": {
            "count": 240,
            "proportion": 4.901493961063308e-06
        },
        "CONCERNED": {
            "count": 240,
            "proportion": 4.901493961063308e-06
        },
        "CONFUSED": {
            "count": 1520,
            "proportion": 3.104279508673428e-05
        },
        "CONTEMPT": {
            "count": 4606,
            "proportion": 9.406783826940665e-05
        },
        "DISGUSTED": {
            "count": 5966,
            "proportion": 0.00012184297071543207
        },
        "EMPHASIS": {
            "count": 800,
            "proportion": 1.633831320354436e-05
        },
        "ENCOURAGING": {
            "count": 240,
            "proportion": 4.901493961063308e-06
        },
        "ENUNCIATED": {
            "count": 1520,
            "proportion": 3.104279508673428e-05
        },
        "EXCITED": {
            "count": 390,
            "proportion": 7.964927686727875e-06
        },
        "HAPPY": {
            "count": 600178,
            "proportion": 0.012257370177346058
        },
        "LAUGHING": {
            "count": 1520,
            "proportion": 3.104279508673428e-05
        },
        "NEUTRAL": {
            "count": 16188107,
            "proportion": 0.3306079529231111
        },
        "NO-AGREEMENT": {
            "count": 13883,
            "proportion": 0.00028353100275600794
        },
        "SAD": {
            "count": 338142,
            "proportion": 0.006905837379091121
        },
        "SARCASTIC": {
            "count": 139,
            "proportion": 2.8387819191158326e-06
        },
        "SLEEPY": {
            "count": 1504,
            "proportion": 3.0716028822663395e-05
        },
        "SURPRISED": {
            "count": 15198,
            "proportion": 0.000310387105084334
        },
        "UNKNOWN": {
            "count": 31423082,
            "proportion": 0.6417501944208214
        },
        "WHISPER": {
            "count": 1518,
            "proportion": 3.100194930372542e-05
        },
        "WORRIED": {
            "count": 502,
            "proportion": 1.0252291535224086e-05
        }
    },
    'age': {
        "Child": {
            "count": 594,
            "proportion": 1.2131197553631687e-05
        },
        "Elderly": {
            "count": 3751625,
            "proportion": 0.07661903034030888
        },
        "Middle-aged": {
            "count": 11475341,
            "proportion": 0.23435964421934244
        },
        "Teenager": {
            "count": 2522661,
            "proportion": 0.05152003190545802
        },
        "Youth-Adult": {
            "count": 31214442,
            "proportion": 0.637489162337337
        }
    }

}
