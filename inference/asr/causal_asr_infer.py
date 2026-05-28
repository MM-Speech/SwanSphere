import os
import tempfile

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

from modules.asr.scriptasr.build_model_utils import build_asr_text_tokenizer, build_asr_model

class CausalASRInfer():
    def __init__(self, device, ckpt):
        self.device = device
        self.build_model(ckpt)

    def build_model(self, ckpt):
        set_hparams(config=os.path.join(ckpt, 'config.yaml'), print_hparams=False)
        self.asr_text_tokenizer, self.asr_vocab_size = build_asr_text_tokenizer()
        self.model = build_asr_model(hparams, self.asr_text_tokenizer, init_pretrained=False)
        self.model.eval()
        load_ckpt(self.model, ckpt, 'model', strict=True, mmap=True)
        self.model.to(self.device)
        self.model.text_tokenizer = self.asr_text_tokenizer
        self.eos_idx = self.asr_text_tokenizer.encode('<|endoftext|>')[0]

        self.resamplers = {}

    @torch.no_grad()
    def forward_model(self, wav, sample_rate=16000, diarization=False, print_candidates=True, use_tqdm=True):
        wav = torch.from_numpy(wav)[None].to(self.device)
        if sample_rate != 16000:
            if sample_rate not in self.resamplers:
                self.resamplers[sample_rate] = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000).to(self.device)
            wav = self.resamplers[sample_rate](wav)
        
        txt_tokens = self.asr_text_tokenizer(['<BOT>'], return_tensors="pt")['input_ids'].to(self.device)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            retry_cnt = 0
            while True:
                tokens_pred = self.model.inference(
                    wav, txt_tokens, 
                    topk=5,
                    temperature=0.7,
                    max_new_tokens=512, 
                    eos_idx=self.eos_idx,
                    print_candidates=print_candidates,
                    diarization=diarization,
                    use_tqdm=use_tqdm
                )
                if diarization:
                    tokens_pred, spk_mask_logits = tokens_pred

                text_pred = self.asr_text_tokenizer.decode(tokens_pred[0].cpu().numpy().tolist())

                if (
                    not detect_repetition(text_pred, min_repeats=8, window_size=3, max_distance=30) and 
                    not detect_repetition(text_pred, min_repeats=8, window_size=6, max_distance=50) and
                    retry_cnt > 3
                    ):
                    break
                retry_cnt += 1

        if diarization:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111)
            im = ax.imshow(spk_mask_logits[0, ::50].cpu().numpy())
            plt.colorbar(im)
            plt.savefig(f"infer_out/asr/figs/{text_pred.replace('<SPK>', '[').replace('</SPK>', ']')[:15]}.png")

        return text_pred

