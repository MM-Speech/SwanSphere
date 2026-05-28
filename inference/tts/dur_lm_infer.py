import os
import tempfile
from copy import deepcopy
import json
from pathlib import Path

import torch
import soundfile as sf
import torchaudio
import librosa
import numpy as np
from tqdm import tqdm

from utils.commons.os_utils import kill_void
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import set_hparams, hparams
from utils.nn.generation_utils import detect_repetition
from utils.text.ph_tone_convert import split_ph_timestamp, split_ph
from utils.text.text_encoder import TokenTextEncoder
from utils.text import TONE_VOCAB, PHONE_VOCAB
from utils.plot.plot import spec_to_figure
from utils.commons.ckpt_utils import load_ckpt
from utils.text.ph_tone_convert import map_phone_to_tokendict

from modules.tts.ar_dur.dur_lm import DurationLM, build_dur_model

class DurLMInfer():
    def __init__(self, device, ckpt):
        self.device = device
        self.build_model(ckpt)

    def build_model(self, ckpt):
        set_hparams(config=os.path.join(ckpt, 'config.yaml'), print_hparams=False)
        self.model = build_dur_model(hparams, vocab_size=810, padding_idx=797)
        self.model.eval()
        load_ckpt(self.model, ckpt, 'model', strict=True, mmap=True)
        self.model.to(self.device)

        self.ph_tokenizer = TokenTextEncoder(None, vocab_list=PHONE_VOCAB, replace_oov='<UNK>')
        self.tone_tokenizer = TokenTextEncoder(None, vocab_list=TONE_VOCAB, replace_oov='<UNK>')

    @torch.no_grad()
    def forward_model(self, ph, tone, ref_ph=None, ref_tone=None, ref_dur=None, print_candidates=True, use_tqdm=True):
        if isinstance(ph, list) and isinstance(ph[0], str):
            ph = torch.LongTensor(self.ph_tokenizer.encode(' '.join(ph)))[None]
            tone = torch.LongTensor(self.tone_tokenizer.encode(' '.join(tone)))[None]
            if ref_ph is not None:
                ref_ph = torch.LongTensor(self.ph_tokenizer.encode(' '.join(ref_ph)))[None]
                ref_tone = torch.LongTensor(self.tone_tokenizer.encode(' '.join(ref_tone)))[None]

        if ref_dur is not None and isinstance(ref_dur, list) or isinstance(ref_dur, np.ndarray):
            ref_dur = torch.LongTensor(ref_dur)[None]
        
        if ref_ph is not None:
            ph = torch.cat([ph, ref_ph], dim=1)
            tone = torch.cat([tone, ref_tone], dim=1)

        merged_ph_tokens = map_phone_to_tokendict({'txt_token': ph, 'tone': tone}, pad_bos_eos=False)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            dur_pred = self.model.inference(
                txt_tokens=merged_ph_tokens.to(self.device),
                dur_tokens=ref_dur.to(self.device),
                temperature=0.9,
            )

        return dur_pred.cpu()



if __name__ == '__main__':
    if os.path.isfile('.env.local'):
        from dotenv import load_dotenv
        load_dotenv('.env.local')
    kill_void()

    ckpt = 'checkpoints/250826_dur_lm'

    infer_ins = DurLMInfer('cuda', ckpt)

    ref_ph = ['sil', 'C0zh', 'C0e', 'C0g', 'C0e', 'C0z', 'C0uei', 'C0l', 'C0i', 'C0b', 'C0ian', 'C0f', 'C0a', 'C0k', 'C0u', 'C0＿', 'C0a', '，', 'C0t', 'C0a', 'C0sh', 'C0iii', 'C0g', 'C0an', '，', 'C0ian', 'C0j', 'C0ing', 'C0m', 'C0o', 'C0h', 'C0u', 'C0n', 'C0e', '，', 'C0sh', 'C0iii', 'C0g', 'C0an', '，', 'C0sh', 'C0uei', 'C0b', 'C0u', 'C0h', 'C0ao', 'C0j', 'C0iao', 'C0n', 'C0e', '，', 'C0h', 'C0ai', 'C0sh', 'C0iii', 'C0g', 'C0an', '，', 'C0n', 'C0a', 'C0m', 'C0e', 'C0uo', 'C0m', 'C0en', 'C0d', 'C0e', 'C0g', 'C0an', 'C0n', 'C0e', '，', 'C0b', 'C0ei', 'C0ch', 'C0eng', 'C0uei', '，', 'C0r', 'C0en', 'C0t', 'C0i', 'C0d', 'C0e', 'C0j', 'C0iang', 'C0j', 'C0vn', '。', 'sil', 'C0t', 'C0a', 'C0zh', 'C0u', 'C0iao', 'C0f', 'C0u', 'C0z', 'C0e', 'C0d', 'C0e', 'C0n', 'C0e', 'C0j', 'C0iou', 'C0sh', 'C0iii', 'C0sh', 'C0u', 'C0x', 'C0ie', 'C0q', 'C0i', 'C0j', 'C0i', '，', 'C0t', 'C0iao', 'C0j', 'C0ie', 'C0x', 'C0ve', 'C0ie', '。']
    ref_tone = ['0', '4', '4', '4', '4', '6', '6', '3', '3', '1', '1', '1', '1', '3', '3', '5', '5', '0', '1', '1', '4', '4', '1', '1', '0', '3', '5', '5', '2', '2', '5', '5', '5', '5', '0', '4', '4', '1', '1', '0', '4', '4', '4', '4', '3', '3', '4', '4', '5', '5', '0', '2', '2', '4', '4', '1', '1', '0', '4', '4', '5', '5', '3', '5', '5', '5', '5', '1', '1', '5', '5', '0', '4', '4', '1', '1', '2', '0', '2', '2', '3', '3', '5', '5', '1', '1', '1', '1', '0', '0', '1', '1', '3', '3', '4', '4', '4', '2', '2', '5', '5', '5', '5', '4', '4', '4', '4', '1', '1', '4', '4', '4', '4', '1', '1', '0', '2', '2', '2', '2', '4', '4', '4', '0']
    ref_dur = [ 1,  4,  4,  5, 30, 13, 26,  4, 14,  7, 13, 18, 12, 23, 12,  0, 26, 29, 7,  6, 14, 44, 13, 26, 63, 29,  8, 17,  6, 19,  8,  9,  6, 18, 45, 15, 5,  7, 24, 55, 16,  6,  7,  2,  9,  8,  8, 16,  6, 19, 23, 17,  9, 10, 4,  9, 20, 59,  2,  4,  6, 39, 23,  7,  5,  4,  4, 19, 12, 12, 25,  7, 5,  8, 13, 10, 27, 11, 11, 18, 13,  6,  7,  4, 13, 17,  8,  5, 10, 46, 10, 24,  7, 14, 10, 13,  7,  5,  8,  3,  6,  9, 18,  8,  5, 11, 12, 19, 9, 12,  9, 17, 12,  8, 20, 27, 10, 12,  6,  8, 13,  9, 20,  5]

    ph = ['sil', 'C0x', 'C0iang', 'C0zh', 'C0uan', 'C0q', 'C0ian', 'C0m', 'C0a', '，', 'C0d', 'C0ian', 'C0j', 'C0i', 'C0x', 'C0ia', 'C0f', 'C0ang', 'C0l', 'C0ian', 'C0j', 'C0ie', '，', 'C0zh', 'C0e', 'C0k', 'C0uan', 'C0zh', 'C0en', 'C0d', 'C0e', 'C0n', 'C0eng', 'C0zh', 'C0uan', 'C0q', 'C0ian', 'C0d', 'C0e', 'C0r', 'C0uan', 'C0j', 'C0ian', '，', 'C0n', 'C0i', 'C0k', 'C0uai', 'C0l', 'C0ai', 'C0sh', 'C0iii', 'C0sh', 'C0iii', 'C0b', 'C0a', '。']
    tone = ['0', '3', '3', '4', '4', '2', '2', '5', '5', '0', '3', '3', '1', '1', '4', '4', '1', '1', '4', '4', '1', '1', '0', '4', '4', '3', '3', '1', '1', '5', '5', '2', '2', '4', '4', '2', '2', '5', '5', '3', '3', '4', '4', '0', '3', '3', '4', '4', '2', '2', '4', '4', '5', '5', '5', '5', '0']

    dur_pred_old = [10,  9, 10, 12, 13, 13, 12, 10,  9,  9, 10, 10, 11, 12, 12, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 14, 13, 14, 14, 14, 14]

    dur_pred = infer_ins.forward_model(
        ph=ph,
        tone=tone,
        ref_ph=ref_ph,
        ref_tone=ref_tone,
        ref_dur=ref_dur
    )

    print(f'{dur_pred_old = }')
    print(f'{dur_pred = }')

# CUDA_VISIBLE_DEVICES=0 python inference/tts/dur_lm_infer.py
