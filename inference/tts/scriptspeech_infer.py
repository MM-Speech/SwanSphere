import os
import tempfile
from pathlib import Path

import torch
import soundfile as sf
import librosa
from tqdm import tqdm
import numpy as np

from utils.commons.os_utils import kill_void, handle_exacption
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import set_hparams, hparams

from modules.asr.sensevoice.sensevoice_api import build_asr_model, run_asr_model
from modules.tts.scriptspeech.build_model_utils import DiTBuildModelMixin, SemanticLMBuildModelMixin, build_vae, build_audio_tokenizer

class ScriptSpeechInfer(DiTBuildModelMixin, SemanticLMBuildModelMixin):
    def __init__(self, device, dit_ckpt, lm_ckpt):
        self.device = device
        self.build_model(dit_ckpt, lm_ckpt)
        self.sr = 24000

    def build_model(self, dit_ckpt, lm_ckpt):
        self.asr_model = build_asr_model(self.device)

        set_hparams(config=os.path.join(dit_ckpt, 'config.yaml'), print_hparams=False)
        self.lm_hparams = set_hparams(config=os.path.join(lm_ckpt, 'config.yaml'), print_hparams=False, global_hparams=False)
        self.lm_hparams['gradient_checkpointing'] = False

        self.vae, self.hp_vae = build_vae(hparams.get('vae_ckpt'))
        self.vae.to(self.device)

        self.audio_token_feature_extractor, self.audio_tokenizer, self.audio_vocab_size = build_audio_tokenizer(hparams.get('audio_tokenizer', 'glm4v'))
        self.audio_tokenizer.to(self.device)

        self.dit_text_tokenizer, self.dit_vocab_size = self.build_dit_text_tokenizer()
        self.lm_text_tokenizer, self.lm_vocab_size = self.build_lm_text_tokenizer()

        self.lm = self.build_lm(self.lm_hparams)
        load_ckpt(self.lm, lm_ckpt, 'lm', strict=True, mmap=True)
        self.eos_idx = self.lm_text_tokenizer.encode('<|endoftext|>')[0]
        self.speech_start_token = self.lm_text_tokenizer.encode('<SpeechToken_0>')[0]
        self.lm.eval()
        self.lm.to(self.device)

        self.dit = self.build_dit(hparams)
        load_ckpt(self.dit, dit_ckpt, 'dit', strict=True, mmap=True)
        self.dit.eval()
        self.dit.to(self.device)

    @torch.no_grad()
    def forward(self, text, ref_audio, ref_text=None):
        if ref_text is None:
            ref_audio = ref_audio[:int(30 * self.sr)]
            if not hasattr(self, 'vad_model') or self.vad_model is None:
                from silero_vad import load_silero_vad, get_speech_timestamps
                self.vad_model = load_silero_vad()
            ref_audio_16k = librosa.resample(ref_audio, orig_sr=self.sr, target_sr=16000)
            speech_timestamps = get_speech_timestamps(
                ref_audio_16k,
                self.vad_model,
                return_seconds=True,  # Return speech timestamps in seconds (default is samples)
            )
            timestamp_end_idx = -1   # include
            for timestamp_idx, timestamp in enumerate(speech_timestamps):
                if timestamp['end'] > 15:
                    timestamp_end_idx = timestamp_idx - 1
                    break
            else:
                timestamp_end_idx = len(speech_timestamps) - 1
            if timestamp_end_idx == -1:
                ref_audio = ref_audio[:int(10 * self.sr)]
                ref_audio_16k = ref_audio_16k[:int(10 * 16000)]
            else:
                ref_audio = ref_audio[:int(speech_timestamps[timestamp_end_idx]['end'] * self.sr)]
                ref_audio_16k = ref_audio_16k[:int(speech_timestamps[timestamp_end_idx]['end'] * 16000)]

            ref_text = run_asr_model([ref_audio_16k], self.asr_model, with_segments=False)[0]['text_normed']
            # print('ref_text', ref_text)

        fm_wav = hparams['frames_multiple'] * hparams['hop_size']
        # ref_audio = ref_audio[:int(self.sr * 10)]
        ref_wav = torch.from_numpy(ref_audio)[None, :].to(self.device)
        ref_wav_lens = torch.LongTensor([ref_wav.shape[1] // fm_wav * fm_wav]).to(self.device)
        ref_wav = ref_wav[:, :ref_wav_lens[0]]

        if hparams.get('audio_tokenizer', 'glm4v') == 'glm4v':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                from modules.tts.semantic_encoders.glm4_tokenizer.call_utils import extract_speech_token_v2
                ref_semantic_tokens, ref_semantic_mask = extract_speech_token_v2(
                    self.audio_tokenizer, self.audio_token_feature_extractor, 
                    wavs=ref_wav, sample_rate=hparams['audio_sample_rate'], wav_lens=ref_wav_lens, device=self.device
                )
                ref_semantic_tokens = ref_semantic_tokens.clone().detach()  # [1, T]

        text = ''.join(c for c in text if c.isprintable())

        from langdetect import detect as classify_language, LangDetectException
        try:
            language_type = classify_language(text)
        except LangDetectException as err:
            handle_exacption(err, '无法检测语言，默认选择中文')
            language_type = 'zh'

        from modules.tts.frontend_lm.sa_frontend import call_sa_frontend
        sa_ret = call_sa_frontend(text, debug=0, lang=language_type, text_type='plain')
        # if sa_ret is None:  # 文本不合法，跳过
        #     print(f'文本段落不合法，跳过')
        #     return
        text, ph_tokens, tone_tokens, alignment_sa = sa_ret

        from utils.text.split_text import chunk_text_chinese_v2, chunk_text_english
        if language_type == 'zh':
            text_segs = chunk_text_chinese_v2(text)
        else:
            text_segs = chunk_text_english(text)

        wav_pred_lst = []
        for text_seg in text_segs:
            # print('='*20)
            # print('text_seg', text_seg)
            # print('ref_wav.shape', ref_wav.shape)
            # print('ref_text', ref_text)
            # print('ref_semantic_tokens.shape', ref_semantic_tokens.shape)
            wav_pred = self.forward_chunk(text_seg, ref_wav, ref_text, ref_semantic_tokens)
            wav_pred_lst.append(wav_pred)

        wav_pred = self.combine_audio_segments(wav_pred_lst)

        return wav_pred

    @torch.no_grad()
    def forward_chunk(self, text, ref_wav, ref_text, ref_semantic_tokens):
        lm_input_tokens = self.lm_text_tokenizer('<BOT>' + ref_text + text + '<BOS>', padding=True, return_tensors='pt')['input_ids'].to(self.device)
        lm_input_tokens = torch.cat([lm_input_tokens, ref_semantic_tokens + self.speech_start_token], dim=1)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            lm_output_tokens = self.lm.generate(lm_input_tokens, max_new_tokens=2048, do_sample=True, top_p=0.99, top_k=20, repetition_penalty=1.1, eos_token_id=self.eos_idx)
        # print('[SEMANTIC OUTPUT]', self.lm_text_tokenizer.decode(list(lm_output_tokens[:, lm_input_tokens.size(1):].cpu().numpy()[0])))
        semantic_tokens = lm_output_tokens[:, lm_input_tokens.size(1):-1] - self.speech_start_token

        semantic_tokens = torch.cat([ref_semantic_tokens, semantic_tokens], dim=1)
        tgt_len = semantic_tokens.shape[1] * 2  # includ ref
            
        with torch.inference_mode():
            lat_ctx = self.vae.encode_latent(ref_wav)
            ctx_mask = torch.ones_like(lat_ctx[:, :, 0:1])
            lat = torch.nn.functional.pad(
                lat_ctx, (0,0,0,tgt_len-lat_ctx.size(1)), mode='constant', value=0)
            ctx_mask = torch.nn.functional.pad(
                ctx_mask, (0,0,0,tgt_len-ctx_mask.size(1)), mode='constant', value=0)

        # ref_text_tokens = self.dit_text_tokenizer(ref_text, return_tensors='pt')['input_ids'].to(self.device)
        text_inputs = self.dit_text_tokenizer(ref_text + text, padding=True, return_tensors='pt').to(self.device)
        txt_tokens = text_inputs['input_ids']
        txt_mask = text_inputs['attention_mask'].bool()
        
        txt_tokens = torch.cat([
            txt_tokens,
            txt_tokens,
            torch.full(txt_tokens.size(), self.cfg_mask_text_token, device=self.device),
            torch.full(txt_tokens.size(), self.cfg_mask_text_token, device=self.device),
            torch.full(txt_tokens.size(), self.cfg_mask_text_token, device=self.device),
        ], dim=0)
        semantic_tokens = torch.cat([
            semantic_tokens,
            torch.full(semantic_tokens.size(), self.cfg_mask_audio_token, device=self.device),
            semantic_tokens,
            torch.full(semantic_tokens.size(), self.cfg_mask_audio_token, device=self.device),
            torch.full(semantic_tokens.size(), self.cfg_mask_audio_token, device=self.device),
        ], dim=0)
        lat = torch.cat([
            lat,
            torch.zeros_like(lat),
            torch.zeros_like(lat),
            lat,
            torch.zeros_like(lat)
        ], dim=0)
        ctx_mask = torch.cat([ctx_mask] * 5, dim=0)
        txt_mask = torch.cat([txt_mask] * 5, dim=0)

        inputs = {
            'txt_tokens': txt_tokens,
            'txt_mask': txt_mask,
            'ctx_mask': ctx_mask,
            'lat_ctx': lat,
            'semantic_tokens': semantic_tokens
        }

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            x = self.dit.inference(inputs, timesteps=32, seq_cfg_w=[2, 2, 0, 4])

            x[:, :lat_ctx.shape[1]] = lat_ctx
            wav_pred = self.vae.decode(x)[0,0].to(torch.float32)
            
            hop_size = self.hp_vae['hop_size']
            vae_stride = self.hp_vae['vae_stride']
            # Trim prompt wav
            wav_pred = wav_pred[lat_ctx.size(1)*vae_stride*hop_size:]
            # clamp the maximum value
            if wav_pred.abs().max() > 1:
                print('Wav amplitude exceed 1, clip it.')
                wav_pred = wav_pred / (wav_pred.abs().max())

            wav_pred = wav_pred.cpu().numpy()

        return wav_pred

    def combine_audio_segments(self, segments, crossfade_duration=0.1):
        window_length = int(self.sr * crossfade_duration)
        hanning_window = np.hanning(2 * window_length)
        for i, segment in enumerate(segments):
            if i == 0:
                combined_audio = segment
            else:
                overlap = combined_audio[-window_length:] * hanning_window[window_length:] + segment[:window_length] * hanning_window[:window_length]
                combined_audio = np.concatenate(
                    [combined_audio[:-window_length], overlap, segment[window_length:]]
                )
        return combined_audio
