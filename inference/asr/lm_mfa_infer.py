import os
import tempfile
from copy import deepcopy
import json
from pathlib import Path
import traceback

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
from utils.plot.plot import spec_to_figure
from utils.audio.diarization_utils import merge_segments
from utils.audio.vad import build_vad_model, run_vad_model
from utils.audio.transform import batch_resample
from utils.commons.dataset_utils import collate_xd, batch_by_size
from utils.nn.seq_utils import sequence_mask

from modules.asr.scriptasr.build_model_utils import build_asr_text_tokenizer, build_asr_model

# import whisper
# from whisper.audio import SAMPLE_RATE, HOP_LENGTH, N_FFT

from utils.commons.ckpt_utils import load_ckpt
from utils.text.ph_tone_convert import map_phone_to_tokendict
from utils.audio import librosa_wav2spec, librosa_wav2linearspec
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# LINE_COLORS = ['w', 'r', 'orange', 'k', 'cyan', 'm', 'b', 'lime', 'g', 'brown', 'navy']

def get_mel(hparams, wav):
    if isinstance(wav, str):
        wav, sr = librosa.core.load(wav, sr=hparams['audio_sample_rate'])
        ws = hparams['win_size']
        if len(wav) % ws < ws - 1:
            wav = np.pad(wav, (0, ws - 1 - (len(wav) % ws)), mode='constant', constant_values=0.0)
    h, w, m = hparams['acous_params'][-1]
    if hparams.get('use_stft_spec', False):
        wav2spec_dict = librosa_wav2linearspec(
            wav,
            fft_size=hparams['fft_size'],
            hop_size=h,
            win_length=hparams['win_size'],
            num_mels=m,
            fmin=hparams['fmin'],
            fmax=hparams['fmax'],
            sample_rate=hparams['audio_sample_rate'],
            center=False)
        mel = wav2spec_dict['linear']
    else:
        wav2spec_dict = librosa_wav2spec(
            wav,
            fft_size=hparams['fft_size'],
            hop_size=h,
            win_length=hparams['win_size'],
            num_mels=m,
            fmin=hparams['fmin'],
            fmax=hparams['fmax'],
            sample_rate=hparams['audio_sample_rate'],
            center=False)
        mel = wav2spec_dict['mel']
    # if hparams.get('reduce_transient_noise'):
    #     from utils.audio.noise_reduction import reduce_transient_noise
    #     mel = reduce_transient_noise(mel)
    wav = wav2spec_dict['wav']
    return mel, wav


class MFAInfer():
    def __init__(self, device, ckpt, torch_compile=False, precision=torch.bfloat16):
        self.device = device
        self.build_model(ckpt, torch_compile)
        self.precision = precision

    def build_model(self, ckpt, torch_compile=False):
        self.hparams = set_hparams(config=os.path.join(ckpt, 'config.yaml'), print_hparams=False, global_hparams=False)
        self.hparams['audio_encoder_ckpt'] = 'checkpoints/wavlm/WavLM-Large.pt'
        self.mfa_vocab_size = 6800
        self.model = build_asr_model(self.hparams, init_pretrained=False, vocab_size=self.mfa_vocab_size, padding_idx=797)
        self.model.eval()
        load_ckpt(self.model, ckpt, 'model', strict=True, mmap=True)
        self.model.to(self.device)
        if torch_compile:
            self.model = torch.compile(self.model, mode='max-autotune')
        self.bos_idx = 798
        self.eos_idx = 799

        ling_dict = json.load(open('egs/tts/megatts3_dict.json'))
        self.ling_dict = {k: TokenTextEncoder(None, vocab_list=ling_dict[k], replace_oov='<UNK>') for k in ['phone', 'tone']}

        self.resamplers = {}
        
        self.vad_model = build_vad_model(self.device)
        
    def txt_postprocess(self, tokens_pred, prompt_max_frame):
        ph, tone, dur, _ = split_ph_timestamp(tokens_pred)
        dur = dur.clamp_min(0)
        if dur.sum() < prompt_max_frame:
            dur[-1] += prompt_max_frame - dur.sum()
        elif dur.sum() > prompt_max_frame:
            len_diff = dur.sum() - prompt_max_frame
            while True:
                for i in range(len(dur)):
                    if dur[i] > 0:
                        dur[i] -= 1
                        len_diff -= 1
                    if len_diff == 0:
                        break
                if len_diff == 0:
                    break
        dur = np.trim_zeros(dur.numpy(), 'b')
        ph = ph.numpy()[:len(dur)]
        tone = tone.numpy()[:len(dur)]
        return ph, tone, dur
        
    def forward_model(self, wav, print_candidates=True, use_tqdm=True):
        fm = 8
        prompt_max_frame = wav.shape[-1] // 160 // fm * fm

        txt_tokens = torch.LongTensor([self.bos_idx])[None, :].to(self.device)

        with torch.autocast(device_type='cuda', dtype=self.precision):
            retry_cnt = 0
            while True:
                tokens_pred = self.model.inference(
                    wav, txt_tokens, 
                    topk=1,
                    temperature=0.7,
                    max_new_tokens=4096, 
                    eos_idx=self.eos_idx,
                    print_candidates=print_candidates,
                    diarization=False,
                    use_tqdm=use_tqdm
                )

                ph, tone, dur = self.txt_postprocess(deepcopy(tokens_pred)[0], prompt_max_frame)

                ph = self.ling_dict['phone'].decode(ph).split(' ')
                tone = self.ling_dict['tone'].decode(tone).split(' ')
                dur = dur.tolist()

                break

        return ph, tone, dur

    @torch.no_grad()
    def forward(self, wav, sample_rate=16000, print_candidates=True, use_tqdm=True):
        fm = 8
        fm_wav = 8 * 160
        wav = torch.from_numpy(wav)[None].to(self.device)
        if sample_rate != 16000:
            if sample_rate not in self.resamplers:
                self.resamplers[sample_rate] = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000).to(self.device)
            wav = self.resamplers[sample_rate](wav)
        
        vad_res = run_vad_model(wav[0], 16000, self.vad_model, threshold=0.3)
        wav_chunk_offsets_sec = merge_segments(vad_res, max_overlap_duration=0, max_silence=1.28, min_voiced_duration=0.16)
                
        ph_res = []
        tone_res = []
        dur_res = []
        for chunk_idx, wav_chunk_offset_sec in enumerate(wav_chunk_offsets_sec):
            start = int(wav_chunk_offset_sec['start'] * 16000) // fm_wav * fm_wav
            end = int(wav_chunk_offset_sec['end'] * 16000) // fm_wav * fm_wav
            
            ph, tone, dur = self.forward_model(wav[:, start: end], print_candidates=print_candidates, use_tqdm=use_tqdm)
            
            if chunk_idx > 0:
                last_end = int(wav_chunk_offsets_sec[chunk_idx - 1]['end'] * 16000) // fm_wav * fm_wav
                if ph[0] == 'sil':
                    dur[0] += round((start - last_end) / 160)
                else:
                    ph = ['sil'] + ph
                    tone = ['0'] + tone
                    dur = [round((start - last_end) / 160)] + dur
            else:
                if start > 0:
                    if ph[0] == 'sil':
                        dur[0] += round(start / 160)
                    else:
                        ph = ['sil'] + ph
                        tone = ['0'] + tone
                        dur = [round(start / 160)] + dur
            
            ph_res += ph
            tone_res += tone
            dur_res += dur
            
        return ph_res, tone_res, dur_res      
    
    @torch.no_grad()
    def forward_batch(self, wavs, sample_rates=16000, max_new_tokens=None, 
                      max_batch_duration=600, max_batch_size=100, use_tqdm=True):
        wavs = batch_resample(wavs, sample_rates, tgt_sr=16000, resamplers=self.resamplers, batch_size=4, device=self.device)
        
        fm = 8
        fm_wav = 8 * 160
        items = []
        wav_chunks = []
        for item_idx, wav in enumerate(wavs):
            vad_res = run_vad_model(torch.from_numpy(wav).to(self.device), 16000, self.vad_model, threshold=0.3)
            if len(vad_res) == 0:
                items.append(None)
                continue
            wav_chunk_offsets_sec = merge_segments(vad_res, max_overlap_duration=0, max_silence=1.28, min_voiced_duration=0.16, max_voiced_duration=60.0)
            wav_chunk_offsets = []
            chunk_idxs = []
            for chunk_idx, wav_chunk_offset_sec in enumerate(wav_chunk_offsets_sec):
                start = int(wav_chunk_offset_sec['start'] * 16000) // fm_wav * fm_wav
                end = int(wav_chunk_offset_sec['end'] * 16000) // fm_wav * fm_wav
                wav_chunk_offsets.append((start, end))
                wav_chunks.append(wav[start: end])
                chunk_idxs.append(len(wav_chunks) - 1)
                
            items.append({
                'wav_chunk_offsets_sec': wav_chunk_offsets_sec,
                'wav_chunk_offsets': wav_chunk_offsets,
                'chunk_idxs': chunk_idxs,
                'wav_len': wav.shape[-1],
            })
                    
        chunk_results = self.forward_model_batch(wav_chunks, max_new_tokens, max_batch_duration, max_batch_size, use_tqdm)
        
        results = []
        for item in items:
            if item is None:
                results.append(None)
                continue
            ph_res = []
            tone_res = []
            dur_res = []
            fail = False
            for i, chunk_idx in enumerate(item['chunk_idxs']):
                if chunk_results[chunk_idx] is None:
                    fail = True
                    break
                
                start, end = item['wav_chunk_offsets'][i]
                
                ph = chunk_results[chunk_idx]['ph']
                tone = chunk_results[chunk_idx]['tone']
                dur = chunk_results[chunk_idx]['dur']
                
                if i > 0:
                    last_end = item['wav_chunk_offsets'][i-1][1]
                    if ph[0] == 'sil':
                        dur[0] += round((start - last_end) / 160)
                    else:
                        ph = ['sil'] + ph
                        tone = ['0'] + tone
                        dur = [round((start - last_end) / 160)] + dur
                else:
                    if start > 0:
                        if ph[0] == 'sil':
                            dur[0] += round(start / 160)
                        else:
                            ph = ['sil'] + ph
                            tone = ['0'] + tone
                            dur = [round(start / 160)] + dur
                
                ph_res += ph
                tone_res += tone
                dur_res += dur
            
            if fail:
                results.append(None)
                continue
                
            if (np.array(dur_res) < 0).sum() > 0:
                results.append(None)
                continue
            
            # append tail silence
            if item['wav_len'] // 160 > sum(dur_res):
                diff = item['wav_len'] // 160 - sum(dur_res)
                if ph_res[-1] in self.ling_dict['phone'].sil_phonemes():
                    dur_res[-1] += diff
                else:
                    ph_res.append('.')
                    tone_res.append('0')
                    dur_res.append(diff)
            
            results.append({
                'ph': ph_res,
                'tone': tone_res,
                'dur': dur_res,
                # 'tokens_pred': chunk_results[chunk_idx]['tokens_pred']
            })
            
        return results
            
            
    def forward_model_batch(self, wavs, max_new_tokens=4096, max_batch_duration=600, max_batch_size=100, use_tqdm=True):
        fm = 8
        
        wav_lengths = [len(wav) for wav in wavs]
        ordered_idxs = np.argsort(wav_lengths).tolist()
        
        batches = batch_by_size(
            ordered_idxs,
            num_tokens_fn=lambda idx: wav_lengths[idx],
            max_tokens=max_batch_duration * 16000,
            max_sentences=max_batch_size
        )
        
        results = [None] * len(wavs)
        for wav_idxs in batches:
            bsz = len(wav_idxs)
            
            if self.model.config.backbone == 'llama':
                pass    # TODO
            elif self.model.config.backbone == 'llama_seq2seq':
                wav_batch = collate_xd([torch.from_numpy(wavs[i]) for i in wav_idxs]).to(self.device)
                wav_mask = sequence_mask(torch.LongTensor([wav_lengths[i] for i in wav_idxs])).to(self.device)
            
            txt_tokens = torch.LongTensor([self.bos_idx] * bsz)[..., None].to(self.device)
            txt_mask = torch.ones_like(txt_tokens).to(self.device)
                        
            max_new_tokens_ = max_new_tokens
            if max_new_tokens is None:
                if wav_batch.shape[1] < 20 * 16000:
                    max_new_tokens_ = 512
                elif 20 * 16000 <= wav_batch.shape[1] < 40 * 16000:
                    max_new_tokens_ = 1024
                elif 40 * 16000 <= wav_batch.shape[1] < 60 * 16000:
                    max_new_tokens_ = 1536
                elif 60 * 16000 <= wav_batch.shape[1] < 120 * 16000:
                    max_new_tokens_ = 3200
                elif 120 * 16000 <= wav_batch.shape[1]:
                    max_new_tokens_ = 4096
            
            try:
                with torch.autocast(device_type='cuda', dtype=self.precision):
                    tokens_pred_lst = self.model.inference_batch(
                        wav_batch, wav_mask,
                        txt_tokens, txt_mask,
                        topk=1,
                        temperature=0.7,
                        max_new_tokens=max_new_tokens_, 
                        eos_idx=self.eos_idx,
                        use_tqdm=use_tqdm
                    )
            except:
                traceback.print_exc()
                continue
            
            for wav_idx, tokens_pred in zip(wav_idxs, tokens_pred_lst):
                tokens_pred = tokens_pred.cpu()
                prompt_max_frame = round(wavs[wav_idx].shape[-1] / 160 / fm * fm)
                try:
                    ph, tone, dur = self.txt_postprocess(deepcopy(tokens_pred), prompt_max_frame)
                except:
                    continue
                ph = self.ling_dict['phone'].decode(ph).split(' ')
                tone = self.ling_dict['tone'].decode(tone).split(' ')
                dur = dur.tolist()
                results[wav_idx] = {
                    'ph': ph,
                    'tone': tone,
                    'dur': dur,
                    'tokens_pred': tokens_pred.squeeze().numpy().tolist()
                }
        
        return results


if __name__ == '__main__':
    if os.path.isfile('.env.local'):
        from dotenv import load_dotenv
        load_dotenv('.env.local')
    kill_void()
    
    benchmark = False
    
    if not benchmark:

        ckpt = 'checkpoints/250823_lm_mfa_seq2seq_small_wavlmlarge'
        # ckpt = 'checkpoints/250923_lm_mfa_seq2seq_small_wavlmlarge_long_robust'

        infer_ins = MFAInfer('cuda', ckpt, torch_compile=False)
        # infer_ins = MFAInfer('cpu', ckpt, torch_compile=False, precision=torch.float32)
        
        wav_path = 'user/prompts/dzq_enhanced.wav'
        # wav_path = 'user/temp/vocal.m4a'

        wav, sr = librosa.load(wav_path, sr=16000)
        # wav = wav[int(2.16 * 16000): int(100.27 * 16000)]
        
        ph, tone, dur = infer_ins.forward(wav, sr)
        
        # result = infer_ins.forward_batch([wav] * 100, [sr] * 100, max_new_tokens=None, max_batch_duration=600, max_batch_size=64, use_tqdm=True)[0]
        # ph, tone, dur = result['ph'], result['tone'], result['dur']

        print(ph)
        print(tone)
        print(dur)
        
        # save_dir = 'infer_out/asr/figure'
        # mel_vis, _ = get_mel({
        #     'audio_sample_rate': SAMPLE_RATE,
        #     'win_size': N_FFT,
        #     'acous_params': [[HOP_LENGTH, N_FFT, 80]],
        #     'fft_size': N_FFT,
        #     'fmin': 0,
        #     'fmax': 12000
        # }, wav)
        # fig = spec_to_figure(mel_vis, vmin=-6, vmax=1.5, dur_info={'txt': ph, 'dur_gt': dur}, figsize=(64, 6))
        # os.makedirs(save_dir, exist_ok=True)
        # plt.savefig(os.path.join(save_dir, f'{Path(wav_path).stem}.png'))
        
    else:
        import glob
        import time
        from utils.commons.io import json_dumps
        from utils.commons.multiprocess_utils import chunked_multiprocess_run
        
        ckpt = '/mnt/bn/sa-ag-data/liruiqi/code/ScriptSpeech/checkpoints/250823_lm_mfa_seq2seq_small_wavlmlarge'
        # infer_ins = MFAInfer('cuda', ckpt)
        infer_ins = MFAInfer('cpu', ckpt)
        
        # wav_paths = glob.glob('/mnt/bn/sa-ag-data/liruiqi/code/ScriptSpeech/user/prompts/*.wav')
        # wavs, srs = [], []
        # for wav_path in wav_paths:
        #     wav, sr = librosa.load(wav_path, sr=None)
        #     wavs.append(wav)
        #     srs.append(sr)
            
        wav_paths = glob.glob('/mnt/bn/sa-ag-data/liruiqi/code/MegaTTS3-inference-safrontend/infer_out/infer_once/*.wav')
        wav_paths = [wav_path for wav_path in wav_paths if wav_path not in (
            '/mnt/bn/sa-ag-data/liruiqi/code/MegaTTS3-inference-safrontend/infer_out/infer_once/ .wav',
            '/mnt/bn/sa-ag-data/liruiqi/code/MegaTTS3-inference-safrontend/infer_out/infer_once/0-人缺什么最容易糖尿病呢，有人说缺营养，也.wav',
            '/mnt/bn/sa-ag-data/liruiqi/code/MegaTTS3-inference-safrontend/infer_out/infer_once/三分靠长相，七分靠打扮，女人的气质是打扮.wav',
            '/mnt/bn/sa-ag-data/liruiqi/code/MegaTTS3-inference-safrontend/infer_out/infer_once/入手闪铸AD5X太值了！它打印速度超快，.wav',
            '/mnt/bn/sa-ag-data/liruiqi/code/MegaTTS3-inference-safrontend/infer_out/infer_once/家人们，来给评评理！上门女婿被老婆当众扇.wav'
            '/mnt/bn/sa-ag-data/liruiqi/code/MegaTTS3-inference-safrontend/infer_out/infer_once/我的老朋友，你真的很不简单,这段时间悄悄.wav',
            '/mnt/bn/sa-ag-data/liruiqi/code/MegaTTS3-inference-safrontend/infer_out/infer_once/这可是国际救援中心认证的\"神器\"！不用插.wav',
        )]
        wavs, srs = [], []
        for (wav, sr) in chunked_multiprocess_run(librosa.load, wav_paths):
            wavs.append(wav)
            srs.append(sr)
        
        # print(f"{len(wavs)} = ")
        # results = infer_ins.forward_batch(wavs, srs, max_new_tokens=None, max_batch_duration=600, 
        #                                   max_batch_size=32, use_tqdm=True)   
        # for i, r in enumerate(results):
        #     if results[i] is None:
        #         results[i] = {}
        #     results[i]['wav_path'] = wav_paths[i]
        # print(json_dumps(results))
        
        # wavs = [*wavs] * 10
        # srs = [*srs] * 10
        print(f"{len(wavs) = }")
        total_duration = sum([wav.shape[0] / sr for (wav, sr) in zip(wavs, srs)])
        print(f"{total_duration = }, {total_duration / 60}min")
        start = time.time()
        for i in tqdm(range(len(wavs))):
            res = infer_ins.forward(wavs[i], srs[i], False, False)
        elapsed = time.time() - start
        print(f"Single infer: elapsed = {elapsed}, RTF = {elapsed / total_duration}")
        
        start = time.time()
        results = infer_ins.forward_batch(wavs, srs, max_new_tokens=None, max_batch_duration=600, max_batch_size=64, use_tqdm=False)
        elapsed = time.time() - start
        print(f"Batch infer: elapsed = {elapsed}, RTF = {elapsed / total_duration}")
        

# CUDA_VISIBLE_DEVICES=0 python inference/asr/lm_mfa_infer.py
# CUDA_VISIBLE_DEVICES= python inference/asr/lm_mfa_infer.py
# FORCE_FLASH_ATTN_BACKEND=none CUDA_VISIBLE_DEVICES=2 python inference/asr/lm_mfa_infer.py