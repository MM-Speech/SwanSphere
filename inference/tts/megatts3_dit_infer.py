import json
import os
from typing import List, Union, Dict
import argparse
import librosa
import numpy as np
import torch
import io
import threading
import torch.nn.functional as F
import whisper
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass

from copy import deepcopy
# from tn.chinese.normalizer import Normalizer as ZhNormalizer
# from tn.english.normalizer import Normalizer as EnNormalizer
from langdetect import detect as classify_language, LangDetectException
from pydub import AudioSegment
# from vllm import LLM, SamplingParams
import pyloudnorm as pyln

from modules.tts.ar_dur.commons.nar_tts_modules import LengthRegulator
from utils.audio.align import mel2token_to_dur
from utils.audio.io import save_wav_bytes, to_wav_bytes, wav_bytes_to_mp3_bytes
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import set_hparams, hparams
from utils.text import is_english, YUNMU_ERHUA, SHENGMU, TONE_VOCAB, PHONE_VOCAB
from utils.text.text_encoder import TokenTextEncoder
from utils.text.split_text import chunk_text_english, chunk_text_chinese, get_word_list, remove_space, remove_unprintable
from utils.text.ph_tone_convert import split_ph_timestamp, split_ph
from utils.text.ssml_utils import SSML
from utils.text.ph_alignment import align_word_phone, print_align, merge_norm_alignment
from utils.commons.hparams import hparams, set_hparams
from utils.commons.meters import Timer
from utils.commons.os_utils import handle_exacption

if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def convert_to_wav(wav_path):
    # Check if the file exists
    if not os.path.exists(wav_path):
        print(f"The file '{wav_path}' does not exist.")
        return

    # Check if the file already has a .wav extension
    if not wav_path.endswith(".wav"):
        # Define the output path with a .wav extension
        out_path = os.path.splitext(wav_path)[0] + ".wav"

        # Load the audio file using pydub and convert it to WAV
        audio = AudioSegment.from_file(wav_path)
        audio.export(out_path, format="wav")

        print(f"Converted '{wav_path}' to '{out_path}'")


def convert_to_wav_bytes(audio_binary):
    # Load the audio binary using pydub and convert it to WAV
    audio = AudioSegment.from_file(io.BytesIO(audio_binary))
    wav_bytes = io.BytesIO()
    audio.export(wav_bytes, format="wav")
    wav_bytes.seek(0)
    return wav_bytes

@contextmanager
def model_lock(lock):
    try:
        lock.acquire()
        yield
    finally:
        torch.cuda.synchronize()
        lock.release()
        
        

@dataclass
class MegaTTS3Output:
    wav_bytes: bytes = None
    wav: np.ndarray = None
    words_timestamps: Dict[str, List] = None
    words_timestamps_post: Dict[str, List] = None
    duration: float = None
    ph_pred: List[str] = None
    tone_pred: List[str] = None


class MegaTTS3DiTInfer():
    ''' 这里指定所用ckpt的路径 '''
    def __init__(
            self, 
            device=None,
            ckpt_root='checkpoints/megatts3_wavdit',
            # ckpt_root='/opt/tiger/model_files/megatts3_wavdit/model/',
            dit_exp_name='diffusion_transformer',
            frontend_exp_name='aligner_lm',
            wavvae_exp_name='wavvae',
            dur_ckpt_path='duration_lm',
            g2p_exp_name='g2p',
            tokenizer_name='llama_tokenizer',
            vllm_gpu_memory_utilization=0.5,
            **kwargs
        ):

        self.sr = 24000
        self.fm = 8
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        self.dit_exp_name = os.path.join(ckpt_root, dit_exp_name)
        self.frontend_exp_name = os.path.join(ckpt_root, frontend_exp_name)
        self.wavvae_exp_name = os.path.join(ckpt_root, wavvae_exp_name)
        self.dur_exp_name = os.path.join(ckpt_root, dur_ckpt_path)
        self.g2p_exp_name = os.path.join(ckpt_root, g2p_exp_name)
        self.tokenizer_name = os.path.join(ckpt_root, tokenizer_name)
            
        # build models
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self.build_model(self.device)

        self.change_wav_header = False
        # init text normalizer
        # self.zh_normalizer = ZhNormalizer(overwrite_cache=True, remove_erhua=False, remove_interjections=False, full_to_half=False)
        # self.en_normalizer = EnNormalizer(overwrite_cache=True)

        # break (silence)
        self.max_silence_alive = 1.28    # 1.28s

    ''' 加载模型参数 '''
    def build_model(self, device):
        self.device = device
        self.precision = torch.float16

        set_hparams(config=f"{self.dit_exp_name}/config.yaml", print_hparams=False)

        ''' Load Dict '''
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ling_dict = {'phone': PHONE_VOCAB, 'tone': TONE_VOCAB}
        self.ling_dict = {k: TokenTextEncoder(None, vocab_list=ling_dict[k], replace_oov='<UNK>') for k in ['phone', 'tone']}
        self.token_encoder = token_encoder = self.ling_dict['phone']
        ph_dict_size = len(token_encoder)
        # init phone replace table
        # self.ph_replace_table = json.load(open(f"{current_dir}/ph_replace_table.json"))
        self.ph_replace_table = {'en': {}, 'zh': {}}

        ''' Load Duration LM '''
        from modules.tts.ar_dur.ar_dur_predictor import ARDurPredictor
        hp_dur_model = self.hp_dur_model = set_hparams(f'{self.dur_exp_name}/config.yaml', global_hparams=False)
        hp_dur_model['frames_multiple'] = hparams['frames_multiple']
        self.dur_model = ARDurPredictor(
            hp_dur_model, hp_dur_model['dur_txt_hs'], hp_dur_model['dur_model_hidden_size'],
            hp_dur_model['dur_model_layers'], ph_dict_size,
            hp_dur_model['dur_code_size'],
            use_rot_embed=hp_dur_model.get('use_rot_embed', False),
            precision=self.precision
            )
        self.length_regulator = LengthRegulator()
        load_ckpt(self.dur_model, f'{self.dur_exp_name}', 'dur_model')
        self.dur_model.eval()
        self.dur_model.to(device, dtype=self.precision)

        ''' Load Diffusion Transformer '''
        from modules.tts.llama_dit.speechdit import Diffusion, ModelArgs
        config = ModelArgs()
        config.max_seq_len = 16384
        config.in_channels = config.out_channels = hparams['latent_dim']
        self.dit = Diffusion(config)
        self.vae_stride = hparams.get('vae_stride', 8)
        load_ckpt(self.dit, f'{self.dit_exp_name}', 'dit', strict=False)
        self.dit.eval()
        self.dit.to(device, dtype=self.precision)
        self.cfg_mask_token_phone = 302 - 1
        self.cfg_mask_token_tone = 32 - 1

        ''' Load Frontend LM '''
        from modules.tts.frontend_lm.whisper.whisper_small import Whisper
        self.aligner_lm = Whisper()
        load_ckpt(self.aligner_lm, f'{self.frontend_exp_name}', 'model')
        self.aligner_lm.eval()
        self.aligner_lm.to(device, dtype=self.precision)
        self.kv_cache = None
        self.hooks = None

        ''' Load G2P LM'''
        # from transformers import AutoTokenizer, AutoModelForCausalLM
        # g2p_tokenizer = AutoTokenizer.from_pretrained(self.g2p_exp_name, padding_side="right")
        # g2p_tokenizer.padding_side = "right"  # avoid overflow issue in batched inference for llama2
        # self.g2p_model = LLM(model=self.g2p_exp_name, dtype='float16', block_size=32, 
        #                      gpu_memory_utilization=self.vllm_gpu_memory_utilization, device=device)
        # self.g2p_tokenizer = g2p_tokenizer
        # self.speech_start_idx = g2p_tokenizer.encode('<Reserved_TTS_0>')[0]
        
        ''' Wav VAE '''
        self.hp_wavvae = hp_wavvae = set_hparams(f'{self.wavvae_exp_name}/config.yaml', global_hparams=False)
        from modules.tts.wavvae.decoder.wavvae_v3 import WavVAE_V3
        self.wavvae = WavVAE_V3(hparams=hp_wavvae)
        load_ckpt(self.wavvae, f'{self.wavvae_exp_name}', 'model_gen', strict=True)
        self.wavvae.eval()
        self.wavvae.to(device, dtype=self.precision)
        
        self.wavvae = torch.compile(self.wavvae, mode='max-autotune')
        self.aligner_lm = torch.compile(self.aligner_lm, mode='max-autotune')
        self.dit = torch.compile(self.dit, mode='max-autotune')
        self.dur_model = torch.compile(self.dur_model, mode='max-autotune')        
        self.lock = threading.Lock()

    ''' 文本预处理 '''
    def preprocess_text(self, input_text: SSML, ph_replace_table=None, use_sa_frontend=False):

        def _normalize_text_en(text: str):
            text_norm = common_preprocess(text)
            if not use_sa_frontend:
                text_norm = self.en_normalizer.normalize(text_norm)
            if ph_replace_table is not None:
                for src, tgt in ph_replace_table['en'].items():
                    text_norm = text_norm.replace(src, tgt)
            text_norm = common_process(text_norm)
            # text_norm = common_postprocess(text_norm)
            return text_norm
            
        def _normalize_text_zh(text):
            text_norm = common_preprocess(text)

            if not use_sa_frontend:
                from opencc import OpenCC
                jp2t_converter = OpenCC('jp2t')
                t2s_converter = OpenCC('t2s')
                text_norm = t2s_converter.convert(jp2t_converter.convert(text_norm))
                
                text_norm = self.zh_normalizer.normalize(text_norm)

            if ph_replace_table is not None:
                for src, tgt in ph_replace_table['zh'].items():
                    text_norm = text_norm.replace(src, tgt)
            text_norm = common_process(text_norm)
            # text_norm = common_postprocess(text_norm)
            return text_norm

        def common_process(text: str):
            text_norm = text
            if not use_sa_frontend:
                # 处理特殊符号
                pause_punc = [
                    '~', '～', ':', '$', '¥', '&', '#', '@', '^', '・', '·', '‘', '’', '“', '”', "'", "'", '"', '"',
                    '（', '）', '(', ')', '【', '】', '{', '}', '「', '」', '[', ']', '<', '>', '《', '》',
                    '%', '*', '|', '｜', '\\', '/', '-', '+', '_', '=',
                    '²',
                ]
                text_norm = batch_replace(text_norm, pause_punc, tgt='')
            return text_norm
        
        def common_preprocess(text: str):
            special_symbols = [
                '&#34;'
            ]
            if use_sa_frontend:
                special_symbols.extend(['"'])   # 删除引号，否则json格式问题
            text_norm = batch_replace(text, special_symbols, tgt='')
            text_norm = batch_replace(text_norm, ['\n'], tgt=' ')
            return text_norm
        
        def common_postprocess(text: str):
            import re
            max_no_punct = 77
            errors = []
            no_punct_segments = re.split(r'[，。、；：！？,.;:!?]', text)
            for seg in no_punct_segments:
                len_seg = len(get_word_list(seg))
                if len_seg > max_no_punct:
                    errors.append(f"无效文本：存在{len_seg}个连续字符无标点（允许上限{max_no_punct}字）")
                    break
            if len(errors) > 0:
                if len(errors) > 1:
                    msg = '; '.join([f'{i+1}. {errors[i]}' for i in range(len(errors))])
                else:
                    msg = errors[0]
                raise RuntimeError(msg)
            return text
        
        def batch_replace(text: str, src: Union[str, List], tgt: str = ','):
            for p in src:
                text = text.replace(p, tgt)
            return text
        
        input_text.apply_sub()

        try:
            language_type = classify_language(input_text.text_str)
        except LangDetectException as err:
            handle_exacption(err, '无法检测语言，默认选择中文')
            language_type = 'zh'
        if language_type == 'en':
            input_text.normalize(_normalize_text_en)
            text_segs = SSML.chunk_text_with_breaks(input_text, limit=130, language_type='en', debug=False)
        else:
            # print('input_text', input_text)
            input_text.normalize(_normalize_text_zh)
            # print('input_text', input_text)
            text_segs = SSML.chunk_text_with_breaks(input_text, limit=60, language_type='zh', debug=False)
            # print('text_segs', text_segs)

        return text_segs

    ''' 进行G2P '''
    def g2p(self, text_inp):
        sampling_params = SamplingParams(
            top_k=1,
            repetition_penalty=1,
            max_tokens=512,
            stop_token_ids=[int(800+1+self.speech_start_idx)],
        )
        txt_token = self.g2p_tokenizer('<BOT>' + text_inp + '<BOS>')['input_ids']
        prompt_token_ids = [txt_token+[145+self.speech_start_idx]]  # 145=sil
        outputs = self.g2p_model.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params, use_tqdm=False)
        outputs = np.array([145+self.speech_start_idx] + list(outputs[0].outputs[0].token_ids), dtype=int)
        ph_tokens = outputs[:-1]-self.speech_start_idx
        ph_pred, tone_pred = split_ph(ph_tokens)
        ph_pred, tone_pred = ph_pred[None, :].to(self.device), tone_pred[None, :].to(self.device)
        return ph_pred, tone_pred
    
    ''' 根据SSML修改发音 '''
    def refine_ph_tone(self, text: SSML, ph_pred: torch.Tensor, tone_pred: torch.Tensor):
        ph_tokens = ph_pred.squeeze().cpu().numpy()
        tone_tokens = tone_pred.squeeze().cpu().numpy()
        ph_tokens = self.ling_dict['phone'].decode(ph_tokens).split(' ')
        tone_tokens = self.ling_dict['tone'].decode(tone_tokens).split(' ')

        # FIXME：这可能导致表现力下降，但如果不这么做，ph2word可能将不再是单向的
        ph_tokens_ = []
        tone_tokens_ = []
        for p_i, p in enumerate(ph_tokens):
            # 额外考虑“这儿(zh er -> zh e er)”这种情况
            if (p_i > 0 and p == "C0er" and ph_tokens[p_i - 1] in SHENGMU) or (p in YUNMU_ERHUA):
                ph_tokens_.append(p[:-1])
                ph_tokens_.append("C0er")
                tone_tokens_.append(tone_tokens[p_i])
                tone_tokens_.append('5')    # 轻声？
            else:
                ph_tokens_.append(p)
                tone_tokens_.append(tone_tokens[p_i])
        ph_tokens = ph_tokens_
        tone_tokens = tone_tokens_

        text_, ph_tokens, ph2word = align_word_phone(text.text_str, ph_tokens)
        # print_align(text_, ph_tokens, ph2word)
        ph2word = [p-1 for p in ph2word]    # ignore "sil" at the begining

        ph_tokens, tone_tokens, ph2word = SSML.replace_ph_tone(text, ph_tokens, tone_tokens, ph2word)

        ph_tokens = self.ling_dict['phone'].encode(' '.join(ph_tokens))
        ph_pred = torch.LongTensor(ph_tokens)[None].to(ph_pred)
        tone_tokens = self.ling_dict['tone'].encode(' '.join(tone_tokens))
        tone_pred = torch.LongTensor(tone_tokens)[None].to(tone_pred)

        return ph_pred, tone_pred, ph2word
    
    ''' 根据SSML增加停顿 '''
    def add_breaks(self, text: SSML, ph_pred: torch.Tensor, tone_pred: torch.Tensor, dur_pred: torch.Tensor, 
                   ph2word: List, break_token=145, break_tone=3):
        ph_tokens = ph_pred.squeeze().cpu().numpy().tolist()
        tone_tokens = tone_pred.squeeze().cpu().numpy().tolist()
        dur_tokens = dur_pred.squeeze().cpu().numpy().tolist()
        ph_tokens, tone_tokens, ph2word, dur_tokens = SSML.add_breaks(
            text, ph_tokens, tone_tokens, ph2word, dur_tokens, break_token, break_tone, 0.01
        )
        ph_pred = torch.Tensor(ph_tokens)[None].to(ph_pred)
        tone_pred = torch.Tensor(tone_tokens)[None].to(tone_pred)
        dur_pred = torch.Tensor(dur_tokens)[None].to(dur_pred)

        return ph_pred, tone_pred, dur_pred, ph2word
    
    ''' 处理文字时间戳 '''
    def make_word_timestamps(self, text: SSML, dur_pred: np.ndarray, ph2word: List):
        dur_timestep = 0.01
        offsets = [0] + np.cumsum(dur_pred).tolist()
        words_to_get = get_word_list(text.text_str)
        ph2word = ph2word + [-3]
        words = []
        timestamps = []
        ph_start_idx = ph_end_idx = 0
        for ph_end_idx in range(1, len(ph2word)):
            if ph2word[ph_end_idx] != ph2word[ph_start_idx]:
                if ph2word[ph_start_idx] >= 0:
                    words.append(words_to_get[ph2word[ph_start_idx]])
                    timestamps.append(
                        [offsets[ph_start_idx] * dur_timestep, offsets[ph_end_idx] * dur_timestep]
                    )
                ph_start_idx = ph_end_idx


        text_merged, text_norm_merged, text_idx_merged, text_norm_idx_merged = merge_norm_alignment(
            text.origin.text_str, words, debug=False
        )

        words_merged = []
        timestamps_merged = []
        word_idx = 0
        for merge_idx in range(len(text_merged)):
            if isinstance(text_merged[merge_idx], list):
                word_merged = []
                timestamp_merged = []
                for i in range(len(text_merged[merge_idx])):
                    if len(word_merged) > 0 and is_english(word_merged[-1]) and is_english(text_merged[merge_idx][i]):
                        word_merged.append(' ')
                    word_merged.append(text_merged[merge_idx][i])
                for i in range(len(text_norm_merged[merge_idx])):
                    timestamp_merged.append(timestamps[word_idx])
                    word_idx += 1
                words_merged.append(''.join(word_merged))
                if len(timestamp_merged) > 0:
                    timestamps_merged.append([timestamp_merged[0][0], timestamp_merged[-1][-1]])
                else:
                    # 此时，raw text有符号，但norm后该符号被删除。该符号的时长暂定为0
                    if len(timestamps_merged) <= 0:
                        timestamps_merged.append([0.0, 0.0])
                    else:
                        timestamps_merged.append([timestamps_merged[-1][-1], timestamps_merged[-1][-1]])
            else:
                words_merged.append(text_merged[merge_idx])
                timestamps_merged.append(timestamps[word_idx])
                word_idx += 1

        return {
            'words': words_merged,
            'timestamps': timestamps_merged
        }
    
    ''' 拼接音频段，使用 crossfade 实现平滑过渡。 '''
    def combine_audio_segments(self, segments, words_timestamps=(), sil_pad_lst=(), crossfade_duration=0.32):
        window_length = int(self.sr * crossfade_duration)
        hanning_window = np.hanning(2 * window_length)
        return_timestamps = len(words_timestamps) > 0
        combined_words_timestamps = {'words': [], 'timestamps': []}
        # Combine
        for i, segment in enumerate(segments):
            if i == 0:
                combined_audio = segment
                if return_timestamps:
                    combined_words_timestamps['words'] = words_timestamps[i]['words']
                    combined_words_timestamps['timestamps'] = words_timestamps[i]['timestamps']
                sil_pad_start, sil_pad_end = sil_pad_lst[i]
                if sil_pad_start > 0:
                    combined_audio = np.concatenate([np.zeros((int(sil_pad_start * self.sr))), combined_audio])
                    combined_words_timestamps['timestamps'] = [[s[0] + sil_pad_start, s[1] + sil_pad_start] for s in combined_words_timestamps['timestamps']]
                if sil_pad_end > 0:
                    combined_audio = np.concatenate([combined_audio, np.zeros((int(sil_pad_end * self.sr)))])
            else:
                sil_pad_start, sil_pad_end = sil_pad_lst[i]
                if sil_pad_start > 0:
                    segment = np.concatenate([np.zeros((int(sil_pad_start * self.sr))), segment])
                if sil_pad_end > 0:
                    segment = np.concatenate([segment, np.zeros((int(sil_pad_end * self.sr)))])
                overlap = combined_audio[-window_length:] * hanning_window[window_length:] + segment[:window_length] * hanning_window[:window_length]
                offset = combined_audio[:-window_length].shape[0] + sil_pad_start * self.sr
                combined_audio = np.concatenate(
                    [combined_audio[:-window_length], overlap, segment[window_length:]]
                )
                if return_timestamps:
                    combined_words_timestamps['words'] = combined_words_timestamps['words'] + words_timestamps[i]['words']
                    timestamps = words_timestamps[i]['timestamps']
                    offset = offset / self.sr
                    timestamps = [[s[0] + offset, s[1] + offset] for s in timestamps]
                    combined_words_timestamps['timestamps'] = combined_words_timestamps['timestamps'] + timestamps
        return combined_audio, combined_words_timestamps

    def preprocess(self, audio_bytes, topk_dur=1, **kwargs):
        wav_bytes = convert_to_wav_bytes(audio_bytes)

        ''' 准备音频和各种参数 '''
        device = self.device

        # Process reference text and wav
        wav, _ = librosa.core.load(wav_bytes, sr=self.sr)
        wav = wav[:18*self.sr]
        # Pad wav if necessary
        ws = hparams['win_size']
        if len(wav) % ws < ws - 1:
            wav = np.pad(wav, (0, ws - 1 - (len(wav) % ws)), mode='constant', constant_values=0.0).astype(np.float32)
        wav = np.pad(wav, (0, 12000), mode='constant', constant_values=0.0).astype(np.float32)

        ''' 使用aligner_lm进行音频特征提取 '''
        with torch.inference_mode():
            whisper_wav = librosa.resample(wav, orig_sr=self.sr, target_sr=16000)
            mel = torch.tensor(whisper.log_mel_spectrogram(whisper_wav).T, dtype=self.precision).to(self.device)[None].transpose(1,2)
            prompt_max_frame = mel.size(2) // self.fm * self.fm
            mel = mel[:, :, :prompt_max_frame]
            token = torch.LongTensor([[798]]).to(device)
            audio_features = self.aligner_lm.embed_audio(mel)
            for i in range(800):
                logits = self.aligner_lm.logits(token, audio_features, None)
                token_pred = torch.argmax(F.softmax(logits[:, -1], dim=-1), 1)[None]
                token = torch.cat([token, token_pred], dim=1)
                if token_pred[0] == 799:
                    break
            alignment_tokens = token

        ''' 将aligner_lm的输出进行处理，得到音素、声调、持续时间 '''
        ph_ref, tone_ref, dur_ref, _ = split_ph_timestamp(deepcopy(alignment_tokens)[0, 1:-1])
        ph_ref = torch.Tensor(ph_ref)[None].to(self.device)
        tone_ref = torch.Tensor(tone_ref)[None].to(self.device)
        if dur_ref.sum() < prompt_max_frame:
            dur_ref[-1] += prompt_max_frame - dur_ref.sum()
        elif dur_ref.sum() > prompt_max_frame:
            len_diff = dur_ref.sum() - prompt_max_frame
            while True:
                for i in range(len(dur_ref)):
                    dur_ref[i] -= 1
                    len_diff -= 1
                    if len_diff == 0:
                        break
                if len_diff == 0:
                    break
        mel2ph_ref = self.length_regulator(dur_ref[None]).to(self.device)
        mel2ph_ref = mel2ph_ref[:, :mel2ph_ref.size(1)//self.fm*self.fm]

        if topk_dur > 1:
            self.dur_model.hparams["infer_top_k"] = topk_dur
        else:
            self.dur_model.hparams["infer_top_k"] = None

        with torch.inference_mode():
            ''' Forward WavVAE to obtain: prompt latent '''
            wav = torch.tensor(wav, dtype=self.precision)[None].to(device)
            vae_latent = self.wavvae.encode_latent(wav)
            vae_latent = vae_latent[:, :mel2ph_ref.size(1)//4]
        
            ''' Duration Prompting '''
            dur_tokens_2d_ = mel2token_to_dur(mel2ph_ref, ph_ref.shape[1]).clamp(
                    max=self.hp_dur_model['dur_code_size'] - 1) + 1
 
            ctx_dur_tokens = dur_tokens_2d_.clone().flatten(0, 1).to(self.device)
            txt_tokens_flat_ = ph_ref.flatten(0, 1)
            ctx_dur_tokens = ctx_dur_tokens[txt_tokens_flat_ > 0][None]

            last_dur_pos_prompt = ctx_dur_tokens.shape[1]
            dur_spk_pos_ids_flat = range(0, last_dur_pos_prompt)
            dur_spk_pos_ids_flat = torch.LongTensor([dur_spk_pos_ids_flat]).to(mel2ph_ref.device)

            _, incremental_state_dur_prompt = self.dur_model.infer(
                ph_ref, {'tone': tone_ref}, None, None, None,
                ctx_vqcodes=ctx_dur_tokens, spk_pos_ids_flat=dur_spk_pos_ids_flat, return_state=True)
            return {
                'ph_ref': ph_ref,
                'tone_ref': tone_ref,
                'mel2ph_ref': mel2ph_ref,
                'vae_latent': vae_latent,
                'incremental_state_dur_prompt': incremental_state_dur_prompt,
                'ctx_dur_tokens': ctx_dur_tokens,
            }

    @torch.inference_mode()
    def process_text_seg(self, t_i, text, len_text_segs, profile,
                         ph_ref, 
                         tone_ref,
                         mel2ph_ref,
                         vae_latent,
                         ctx_dur_tokens,
                         incremental_state_dur_prompt,
                         last_dur_pos_prompt,
                         wav_pred_,
                         sil_pad_lst,
                         ph_pred_lst,
                         tone_pred_lst,
                         words_timestamps,
                         words_timestamps_post,
                         dur_disturb,
                         dur_alpha,
                         return_timestamp,
                         timestamp_postprocess,
                         use_sa_frontend,
                         time_step,
                         p_w,
                         t_w,
                         ):
        
        if text.text_str.strip() == '':
            ph_pred_lst[t_i] = []
            tone_pred_lst[t_i] = []
            wav_pred_[t_i] = np.zeros(int(0.16 * self.sr))
            if return_timestamp:
                words_timestamps[t_i] = {'words': [], 'timestamps': []}
            sil_pad_lst[t_i] = (text.pause_at_start, text.pause_at_end)
            return

        ''' G2P '''
        with Timer('G2P', enable=profile):
            if not use_sa_frontend:
                with model_lock(self.lock):
                    ph_pred, tone_pred = self.g2p(text.text_str)
            else:
                # with model_lock(self.lock):
                from modules.tts.frontend_lm.sa_frontend import call_sa_frontend
                # print('text.sa_ssml_str', text.sa_ssml_str)
                sa_ret = call_sa_frontend(text.sa_ssml_str, debug=0)

                if sa_ret is None:  # 文本不合法，跳过
                    print(f'文本段落{t_i}/{len_text_segs}不合法，跳过')
                    return

                text_sa, ph_tokens, tone_tokens, alignment_sa = sa_ret
                
                # directly override
                new_text = SSML(text_sa)
                new_text.rate = text.rate
                new_text.pause_at_start = text.pause_at_start
                new_text.pause_at_end = text.pause_at_end
                text = new_text

                ph_pred = self.ling_dict['phone'].encode(' '.join(ph_tokens))
                ph_pred = torch.LongTensor(ph_pred)[None].to(self.device)
                tone_pred = self.ling_dict['tone'].encode(' '.join(tone_tokens))
                tone_pred = torch.LongTensor(tone_pred)[None].to(self.device)
                # print('ph_tokens', ph_tokens)
                # print('tone_tokens', tone_tokens)
                # print('text_sa', text_sa)
                # print('alignment_sa', alignment_sa)
        
        ''' Refeine Phonemes and Tones using SSML'''
        if not use_sa_frontend:
            # with Timer('Refeine Phonemes and Tones using SSML', enable=profile):
            ph_pred, tone_pred, ph2word = self.refine_ph_tone(text, ph_pred, tone_pred)
        # else:
        #     text_, ph_tokens, ph2word = align_word_phone(text.text_str, ph_tokens)
        #     print_align(text_, ph_tokens, ph2word)
        #     ph2word = [p-1 for p in ph2word]    # ignore "sil" at the begining

        ''' Duration Prediction '''
        with Timer('Duration Prediction', enable=profile):
            last_dur_token = ctx_dur_tokens[:, -1:]
            last_dur_pos = last_dur_pos_prompt
            txt_len = ph_pred.shape[1]
            dur_spk_pos_ids_flat = range(last_dur_pos, last_dur_pos + txt_len)
            dur_spk_pos_ids_flat = torch.LongTensor([dur_spk_pos_ids_flat]).to(self.device)
            last_dur_pos = last_dur_pos + txt_len
            with model_lock(self.lock):
                dur_pred = self.dur_model.infer(
                    ph_pred, {'tone': tone_pred}, None, None, None,
                    incremental_state=incremental_state_dur_prompt,
                    first_decoder_inp=last_dur_token,
                    spk_pos_ids_flat=dur_spk_pos_ids_flat, use_tqdm=False
                )
            dur_pred = dur_pred - 1

            print(f"{dur_pred = }")

            ''' Control Speach Spead '''
            dur_pred = torch.round(dur_pred / text.rate).int()

            dur_pred = dur_pred.clamp(0, self.hp_dur_model['dur_code_size'] - 1)
            if t_i < len_text_segs - 1:
                # add 0.32ms for crossfade
                dur_pred[:, -1] = dur_pred[:, -1] + 32
            else:
                dur_pred[:, -1] = dur_pred[:, -1].clamp(32, 80)

            dur_disturb_choice = (torch.rand_like(dur_pred.float()) > 0.5).float()
            dur_disturb_r = 1 + torch.rand_like(dur_pred.float()) * dur_disturb
            dur_pred = dur_pred * dur_disturb_r * dur_disturb_choice + \
                    dur_pred / dur_disturb_r * (1 - dur_disturb_choice)
            dur_pred = torch.round(dur_pred * dur_alpha).clamp(0, 127)
            # ['。', '！', '？', 'sil']
            for sil_token in [148, 153, 166, 145]:
                dur_pred[ph_pred==sil_token] = dur_pred[ph_pred==sil_token].clamp_min(32)
            # ['，', '；'] 
            for sil_token in [163, 165]:
                dur_pred[ph_pred==sil_token] = dur_pred[ph_pred==sil_token].clamp_min(16)
            dur_pred[:, 0] = 8

            if t_i == 0:
                dur_pred[:, 0] = 8
            # else:
            #     dur_pred[:, 0] = 48     # add 0.16ms for crossfade

            ''' Add Breaks '''
            if not use_sa_frontend:
                ph_pred, tone_pred, dur_pred, ph2word = self.add_breaks(
                    text, ph_pred, tone_pred, dur_pred, ph2word, break_token=163, break_tone=3
                )
            else:
                sil_pad_start, sil_pad_end = 0, 0

                if text.pause_at_start > 0:
                    if ph_pred[:, 0] == 145:
                        if dur_pred[:, 0]/100 + text.pause_at_start > self.max_silence_alive:
                            sil_pad_start = text.pause_at_start - (self.max_silence_alive - dur_pred[:, 0]/100).item()
                            dur_pred[:, 0] = round(self.max_silence_alive * 100)
                        else:
                            sil_pad_start = 0
                            dur_pred[:, 0] += round(text.pause_at_start * 100)
                    else:
                        ph_pred = torch.cat([torch.full((1, 1), 145).to(ph_pred), ph_pred], dim=1)
                        if text.pause_at_start > self.max_silence_alive:
                            sil_pad_start = text.pause_at_start - self.max_silence_alive
                            dur_pred = torch.cat([torch.full((1, 1), round(self.max_silence_alive * 100)).to(dur_pred), dur_pred], dim=1)
                        else:
                            sil_pad_start = 0
                            dur_pred = torch.cat([torch.full((1, 1), round(text.pause_at_start * 100)).to(dur_pred), dur_pred], dim=1)

                if text.pause_at_end > 0:
                    if ph_pred[:, -1].item() in [148, 163, 166, 153, 165, 147]:     # 。 ， ？ ！ ； 、
                        if dur_pred[:, -1]/100 + text.pause_at_end > self.max_silence_alive:
                            sil_pad_end = text.pause_at_end - (self.max_silence_alive - dur_pred[:, -1]/100).item()
                            dur_pred[:, -1] = round(self.max_silence_alive * 100)
                        else:
                            sil_pad_end = 0
                            dur_pred[:, -1] += round(text.pause_at_end * 100)
                    else:
                        ph_pred = torch.cat([ph_pred, torch.full((1, 1), 163).to(ph_pred)], dim=1)
                        if text.pause_at_end > self.max_silence_alive:
                            sil_pad_end = text.pause_at_end - self.max_silence_alive
                            dur_pred = torch.cat([dur_pred, torch.full((1, 1), round(self.max_silence_alive * 100)).to(dur_pred)], dim=1)
                        else:
                            sil_pad_end = 0
                            dur_pred = torch.cat([dur_pred, torch.full((1, 1), round(text.pause_at_end * 100)).to(dur_pred)], dim=1)
                
                sil_pad_lst[t_i] = (sil_pad_start, sil_pad_end)
            ''''''

            ph_pred_lst[t_i] = self.ling_dict['phone'].decode(ph_pred.squeeze().cpu().numpy()).split(' ')
            tone_pred_lst[t_i] = self.ling_dict['tone'].decode(tone_pred.squeeze().cpu().numpy()).split(' ')

            dur_sum = dur_pred.sum()
            vqs = hparams['vq_stride']
            npad = vqs - dur_sum % vqs
            if npad < vqs:
                dur_pred[:, -1] += npad

            ''' Make words&timestamps '''
            if return_timestamp:
                try:
                    if not use_sa_frontend:
                        words_timestamps_cur = self.make_word_timestamps(text, dur_pred.squeeze().cpu().numpy(), ph2word)
                    else:
                        words = []
                        timestamps = []
                        offsets = [0] + np.cumsum(dur_pred.squeeze().cpu().numpy()).tolist()
                        dur_timestep = 0.01
                        for align_item in alignment_sa:
                            words.append(align_item['word'])
                            timestamps.append([offsets[align_item['phone_idx'][0]] * dur_timestep, 
                                               offsets[align_item['phone_idx'][-1] + 1] * dur_timestep])
                        words_timestamps_cur = {
                            'words': words,
                            'timestamps': timestamps
                        }
                    words_timestamps[t_i] = (words_timestamps_cur)
                except IndexError as err:
                    handle_exacption(err, text)
            ''''''
            mel2ph_pred = self.length_regulator(dur_pred).to(self.device)

        ''' DiT target speech generation '''
        with Timer('DiT target speech generation', enable=profile):
            # Prepare duration token 
            mel2ph_pred = torch.cat((mel2ph_ref, mel2ph_pred+ph_ref.size(1)), dim=1)
            mel2ph_pred = mel2ph_pred[:, :mel2ph_pred.size(1)//self.fm*self.fm].repeat(3, 1)

            # Disable the English tone (set them to 3)"""
            ph_seq = torch.cat((ph_ref, ph_pred), dim=1)
            tone_seq = torch.cat((tone_ref, tone_pred), dim=1)
            en_tone_idx = ~((tone_seq == 4) | ( (11 <= tone_seq) & (tone_seq <= 15)) | (tone_seq == 0))
            tone_seq[en_tone_idx] = 3
            ph_seq = torch.cat([ph_seq, ph_seq, torch.full(ph_seq.size(), self.cfg_mask_token_phone, device=self.device)], 0)
            tone_seq = torch.cat([tone_seq, tone_seq, torch.full(tone_seq.size(), self.cfg_mask_token_tone, device=self.device)], 0)
            target_size = mel2ph_pred.size(1)//4
            vae_latent_ = vae_latent.repeat(3, 1, 1)
            ctx_mask = torch.ones_like(vae_latent_[:, :, 0:1])
            vae_latent_ = F.pad(vae_latent_, (0, 0, 0, target_size - vae_latent.size(1)), mode='constant', value=0)
            vae_latent_[1:] = 0.0
            ctx_mask = F.pad(ctx_mask, (0, 0, 0, target_size - vae_latent.size(1)), mode='constant', value=0)
            latent_lengths = torch.LongTensor([s.shape[0] for s in vae_latent_]).to(self.device)
            txt_lengths = torch.LongTensor([s.shape[0] for s in ph_seq]).to(self.device)

            inputs = {
                'phone': ph_seq,
                'tone': tone_seq,
                'txt_lens': txt_lengths,
                "lat": vae_latent_ * (1 - ctx_mask),
                "lat_lens": latent_lengths,
                "lat_ctx": vae_latent_ * ctx_mask,
                "ctx_mask": ctx_mask,
                "dur": mel2ph_pred,
                "golden_latent": None
            }

            # Euler ODE solver
            with model_lock(self.lock):
                with torch.autocast(device_type='cuda', dtype=self.precision):
                    x = self.dit.inference(inputs, timesteps=time_step, seq_cfg_w=[p_w, t_w], use_amo_sampler=True)
        # WavVAE decode
        with Timer('WavVAE decode', enable=profile):
            x[:, :vae_latent.size(1)] = vae_latent
            with model_lock(self.lock):
                wav_pred = self.wavvae.decode(x)[0,0].to(torch.float32)
            
            ''' Post-processing '''
        with Timer('Post-processing', enable=profile):
            hop_size = self.hp_wavvae['hop_size']
            vae_stride = self.hp_wavvae['vae_stride']
            # Trim prompt wav
            wav_pred = wav_pred[vae_latent.size(1)*vae_stride*hop_size:]
            # clamp the maximum value
            if wav_pred.abs().max() > 1:
                print('Wav amplitude exceed 1, clip it.')
                wav_pred = wav_pred / (wav_pred.abs().max())

            wav_pred = wav_pred.cpu().numpy()
            wav_pred_[t_i] = wav_pred
            if return_timestamp and timestamp_postprocess:
                from modules.tts.frontend_lm.whisper.frontend_lm_mfa import run_mfa_model
                ph_tokens = ph_pred.squeeze().cpu().numpy()
                tone_tokens = tone_pred.squeeze().cpu().numpy()
                ph_tokens = self.ling_dict['phone'].decode(ph_tokens).split(' ')
                with model_lock(self.lock):
                    mfa_result = run_mfa_model(wav_pred, 
                                               self.aligner_lm, 
                                               self.ling_dict, 
                                               sr=self.sr, 
                                               ph_input=ph_tokens, 
                                               tone_input=tone_tokens,
                                               precision=self.precision)
                dur_seq = mfa_result['dur_seq']
                ph_pred = mfa_result['ph_pred']
                words_timestamps_cur = self.make_word_timestamps(text, np.array(dur_seq), ph2word)
                words_timestamps_post[t_i] = words_timestamps_cur

    def forward(self, resource_context, input_text, time_step, p_w, t_w, 
                speech_rate=1, return_timestamp=True, timestamp_postprocess=False, 
                return_format='wav', custom_ph_table=None, dur_disturb=0.1, dur_alpha=1.0, 
                num_parallel_workers=5, use_sa_frontend=True, **kwargs):
        """
        Args:
            resource_context (dict): resource context dict generated from self.preprocess(). 由 self.preprocess() 方法返回的资源包
            input_text (str): input text, with or without SSML format. 输入的文本，可支持 SSML 格式
            time_step (int): number of time steps for generation. 生成所需要的步数
            p_w (float): weight to control pronunciation. 控制发音清晰度的权重
            t_w (float): weight to control speaker similarity. 控制音色相似度的权重
            speech_rate (float): control speech rate, from 0.01 to 2, 1 for original rate. 控制语速，最低0.01倍，最高2倍速，默认1倍速
            return_timestamp (bool): if true, return raw-text timestamps. 控制是否返回与原始文本对齐的时间戳
            timestamp_postprocess (bool): [deprecated]. 已弃用. if true, also perform post-alignment to obtain more accurate timestamps. 控制是否额外使用后处理对齐的方式获得时间戳
            return_format (str): only support 'wav' or 'mp3', 'wav' recommended. 控制返回的文件格式，支持'wav'和'mp3'，推荐使用'wav'
            custom_ph_table (dict | None): whitelist for special pronunciation. 控制特殊发音规则的白名单
            dur_disturb (float): randomly disturb phoneme duration. 控制音素时长随机扰动的权重
            dur_alpha (float): [deprecated]. 已弃用

        Returns:
            output (MegaTTS3Output): contains wav_bytes, words_timestamps, and words_timestamps_post. 
                结构体包含音频文件的二进制内容(wav_bytes)、时间戳(words_timestamps)、后处理获得的时间戳(words_timestamps_post)，时间戳默认返回 None
        """
        device = self.device
        incremental_state_dur_prompt = resource_context['incremental_state_dur_prompt']
        last_dur_pos_prompt = resource_context['ctx_dur_tokens'].shape[1]

        profile = os.environ.get('MEGA_PROFILE', 'false').strip().lower() == 'true'

        with torch.inference_mode():
            ''' Generating '''
            # input_text = remove_space(input_text)
            # remove blank special symbols
            input_text = ''.join(c for c in input_text if c.isprintable())

            input_text = SSML(input_text)
            input_text.rate = float(speech_rate)

            ''' generate pure silence '''
            if input_text.text_str.strip() == '':
                from bs4 import Tag
                sil_time = 0
                for ele in input_text:
                    if ele.name == 'break' and isinstance(ele, Tag):
                        sil_time += float(ele.get('time')[:-1])
                if sil_time == 0:
                    raise RuntimeError('输入为空，输入不合法')
                wav_bytes = to_wav_bytes(np.zeros((int(sil_time * self.sr),)).astype(float), self.sr)
                if return_format == 'mp3':
                    wav_bytes = wav_bytes_to_mp3_bytes(wav_bytes)
                output = MegaTTS3Output(
                    wav_bytes=wav_bytes,
                    words_timestamps={'words': [], 'timestamps': []},
                    words_timestamps_post=None,
                    duration=sil_time,
                    ph_pred=[],
                    tone_pred=[]
                )
                return output
                        
            ''' preprocess text '''
            with Timer('preprocess text', enable=profile):
                ph_replace_table = deepcopy(self.ph_replace_table)
                if custom_ph_table is not None:
                    ph_replace_table.update(custom_ph_table)
                text_segs = self.preprocess_text(input_text, ph_replace_table, use_sa_frontend)

            len_text_segs = len(text_segs)
            words_timestamps = [None] * len_text_segs
            words_timestamps_post = [None] * len_text_segs
            wav_pred_ = [None] * len_text_segs
            sil_pad_lst = [None] * len_text_segs
            ph_pred_lst = [None] * len_text_segs
            tone_pred_lst = [None] * len_text_segs

            with ThreadPoolExecutor(max_workers=num_parallel_workers) as executor:
                futs = []
                for t_i, text in enumerate(text_segs):
                    future = executor.submit(
                                self.process_text_seg, 
                                *(t_i, text, len_text_segs, profile), 
                                **{
                                    "ph_ref": resource_context['ph_ref'].detach().clone().to(device), 
                                    "tone_ref":resource_context['tone_ref'].detach().clone().to(device),
                                    "mel2ph_ref": resource_context['mel2ph_ref'].detach().clone().to(device),
                                    "vae_latent": resource_context['vae_latent'].detach().clone().to(device),
                                    "ctx_dur_tokens": resource_context['ctx_dur_tokens'].detach().clone().to(device),
                                    "incremental_state_dur_prompt": deepcopy(incremental_state_dur_prompt),
                                    "last_dur_pos_prompt": last_dur_pos_prompt,
                                    "wav_pred_": wav_pred_,
                                    "sil_pad_lst": sil_pad_lst,
                                    "ph_pred_lst": ph_pred_lst,
                                    "tone_pred_lst": tone_pred_lst,
                                    "words_timestamps": words_timestamps,
                                    "words_timestamps_post": words_timestamps_post,
                                    "dur_disturb": dur_disturb,
                                    "dur_alpha":dur_alpha,
                                    "return_timestamp": return_timestamp,
                                    "timestamp_postprocess": timestamp_postprocess,
                                    "use_sa_frontend": use_sa_frontend,
                                    "time_step": time_step,
                                    "p_w": p_w,
                                    "t_w": t_w,
                                })
                    futs.append(future)
                
                results = [f.result() for f in futs]

            words_timestamps = [s for s in words_timestamps if s is not None]
            wav_pred_ = [s for s in wav_pred_ if s is not None]
            ph_pred_lst = [s for s in ph_pred_lst if s is not None]
            tone_pred_lst = [s for s in tone_pred_lst if s is not None]

            # assert len(wav_pred_) == len(ph_pred_lst) == len(tone_pred_lst), f"{len(wav_pred_)}, {len(ph_pred_lst)}, {len(tone_pred_lst)}"

            for i in range(1, len(ph_pred_lst)):
                ph_pred_lst[i] = ph_pred_lst[i][1:]
            ph_pred_lst = np.concatenate(ph_pred_lst).tolist()
            for i in range(1, len(tone_pred_lst)):
                tone_pred_lst[i] = tone_pred_lst[i][1:]
            tone_pred_lst = np.concatenate(tone_pred_lst).tolist()
            
            words_timestamps_fail = (None in words_timestamps) or (len(words_timestamps) == 0) or (len(words_timestamps) != len(wav_pred_))
            if words_timestamps_fail:
                words_timestamps = []
                words_timestamps_post = []

            # normalize loudness
            if len(wav_pred_) > 1:
                silent_speech = False
                meter = pyln.Meter(self.sr)  # create BS.1770 meter
                j = 0
                while j < len(wav_pred_):
                    try:
                        loudness_1 = meter.integrated_loudness(wav_pred_[0].astype(float))
                        break
                    except:
                        pass
                    j += 1
                else:
                    silent_speech = True
                if not silent_speech:
                    for i in range(j+1, len(wav_pred_)):
                        wav_pred__ = wav_pred_[i]
                        loudness_pred = meter.integrated_loudness(wav_pred__)
                        wav_pred__ = pyln.normalize.loudness(wav_pred__, loudness_pred, loudness_1)
                        if np.abs(wav_pred__).max() >= 1:
                            wav_pred__ = wav_pred__ / np.abs(wav_pred__).max() * 0.95
                        wav_pred_[i] = wav_pred__

            if not timestamp_postprocess:
                wav_pred_, words_timestamps = self.combine_audio_segments(wav_pred_, words_timestamps, sil_pad_lst)
                words_timestamps_post = None
            else:
                _, words_timestamps = self.combine_audio_segments(wav_pred_, words_timestamps, sil_pad_lst)
                wav_pred_, words_timestamps_post = self.combine_audio_segments(wav_pred_, words_timestamps_post, sil_pad_lst)

            if not return_timestamp:
                words_timestamps = words_timestamps_post = None

            if words_timestamps_fail:
                words_timestamps = None

            wav_bytes = to_wav_bytes(wav_pred_.astype(float), self.sr)
            if return_format == 'mp3':
                wav_bytes = wav_bytes_to_mp3_bytes(wav_bytes)

            output = MegaTTS3Output(
                wav_bytes=wav_bytes,
                words_timestamps=words_timestamps,
                words_timestamps_post=words_timestamps_post,
                duration=wav_pred_.shape[-1] / self.sr,
                ph_pred=ph_pred_lst,
                tone_pred=tone_pred_lst
            )

            return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wav', type=str)
    parser.add_argument('--input_text', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--time_step', type=int, default=32, help='Diffusion Transformer推理步数')
    parser.add_argument('--p_w', type=float, default=1.4, help='控制发音清晰度的权重')
    parser.add_argument('--t_w', type=float, default=3.0, help='控制音色相似度的权重')
    args = parser.parse_args()
    wav_path, input_text, out_path, time_step, p_w, t_w = args.input_wav, args.input_text, args.output_dir, args.time_step, args.p_w, args.t_w

    infer_ins = MegaTTS3DiTInfer(device=f'cuda:0', vllm_gpu_memory_utilization=0.2)
    os.system(f'pkill -f "voidgpu0"')

    if True:
        with open(wav_path, 'rb') as file:
            file_content = file.read()
        print(f"| Start processing {wav_path}+{input_text}")
        resource_context = infer_ins.preprocess(file_content)
        # torch.save(resource_context, 'infer_out/resource_fanxian.pt')
    else:
        resource_context = torch.load('infer_out/resource_fanxian.pt')

    custom_ph_table = {
        'en': {
            '@': 'at',
            '&': 'and'
        },
        'zh': {
            '@': '艾特',
            '&': '和'
        }
    }
    # for i in range(10):
    with Timer('total', enable=True):
        output = infer_ins.forward(resource_context, input_text, time_step=time_step, 
                                p_w=p_w, t_w=t_w, return_timestamp=True,
                                custom_ph_table=custom_ph_table)

    wav_bytes = output.wav_bytes
    words_timestamps = output.words_timestamps
    input_text = SSML(input_text).text_str
    os.makedirs(out_path, exist_ok=True)
    save_wav_bytes(wav_bytes, f'{out_path}/[P]{input_text[:20]}.wav')

    print(output.ph_pred)
    print(output.tone_pred)

    # words_timestamps_post = output.words_timestamps_post
    # if words_timestamps_post is not None:
    #     for i in range(len(words_timestamps_post['words'])):
    #         print(words_timestamps_post['words'][i], words_timestamps_post['timestamps'][i])