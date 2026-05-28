import os
import collections
import collections.abc
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))

import sys
import json
from typing import List, Union, Dict
import argparse
import librosa
import numpy as np
import torch
import io
import threading
import traceback
import torch.nn.functional as F
import whisper
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import subprocess
from pathlib import Path

from copy import deepcopy
from langdetect import detect as classify_language, LangDetectException
from pydub import AudioSegment
import pyloudnorm as pyln
from tqdm import tqdm

from utils.audio.align import mel2token_to_dur
from utils.audio.io import save_wav_bytes, to_wav_bytes, wav_bytes_to_mp3_bytes
from utils.text import is_english, YUNMU_ERHUA, SHENGMU
from utils.text.text_encoder import TokenTextEncoder
from utils.text.split_text import chunk_text_english, chunk_text_chinese, get_word_list, remove_space, remove_unprintable
from utils.text.ph_tone_convert import split_ph_timestamp, split_ph, map_phone_to_tokendict
from utils.text.ssml_utils import SSML
from utils.text.ph_alignment import align_word_phone, print_align, merge_norm_alignment
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import set_hparams, hparams
from utils.commons.meters import Timer
from utils.commons.os_utils import handle_exacption
from utils.commons.io import print_once
from utils.commons.import_utils import import_module_bystr, get_class_from_module
from utils.commons.tensor_utils import move_to_cpu, move_to_cuda

from modules.tts.ar_dur.commons.nar_tts_modules import LengthRegulator
from modules.tts.ar_dur.commons.align_ops import compute_mel2aug_from_dur
from modules.tts.scriptspeech.build_model_utils import DiTBuildModelMixinV2
# 原 import 们之后补一行
from tasks.tts.task_utils.prompttts_task_utils import build_audio_mask_from_ids


DEBUG = False

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
        
        
def move_to_shm(exp_name):
    os.makedirs('/dev/shm/mega_ckpt', exist_ok=True)
    shm_exp_name = f"/dev/shm/mega_ckpt/{Path(exp_name).stem}"
    if os.path.exists(shm_exp_name):
        return shm_exp_name
    subprocess.check_call(f"cp -r {exp_name} /dev/shm/mega_ckpt/", shell=True)
    return shm_exp_name


@dataclass
class MegaTTS3Output:
    wav_bytes: bytes = None
    wav: np.ndarray = None
    words_timestamps: Dict[str, List] = None
    words_timestamps_post: Dict[str, List] = None
    duration: float = None
    ph_pred: List[str] = None
    tone_pred: List[str] = None


class MegaTTS3DiTInfer(DiTBuildModelMixinV2):
    ''' 这里指定所用ckpt的路径 '''
    def __init__(
            self, 
            device=None,
            dit_exp_name='checkpoints/250818_megatts3_dit_v2',
            dur_exp_name='checkpoints/250826_dur_lm',
            frontend_exp_name='checkpoints/250820_lm_mfa_base_wavlmlarge',
            wavvae_exp_name='checkpoints/1231_megatts3_wavvae_v3_25hz_kl001_fix4',
            use_old_aligner=True,
            use_old_dur=True,
            max_ref_duration=20,
            use_tqdm=True,
            use_ema=False,
            precision='bf16',
            **kwargs
        ):
        
        if precision == 'fp16':
            self.precision = torch.float16
        elif precision == 'bf16':
            self.precision = torch.bfloat16

        self.sr = 24000
        self.fm = 8
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.use_tqdm = use_tqdm
        self.use_ema = use_ema

        self.dit_exp_name = dit_exp_name
        self.use_old_dur = use_old_dur
        if use_old_dur:
            self.dur_exp_name = 'checkpoints/megatts3_wavdit/duration_lm'
        else:
            self.dur_exp_name = dur_exp_name
        self.use_old_aligner = use_old_aligner
        if use_old_aligner:
            self.frontend_exp_name = 'checkpoints/megatts3_wavdit/aligner_lm'
        else:
            self.frontend_exp_name = frontend_exp_name
        self.wavvae_exp_name = wavvae_exp_name

        self.build_model(self.device)

        # break (silence)
        self.max_silence_alive = 1.28    # 1.28s
        self.max_ref_duration = max_ref_duration
        self.chunk_num_words_zh = 60
        self.chunk_num_words_en = 130

    def build_dur_model(self):
        self.length_regulator = LengthRegulator()
        if self.use_old_dur:
            from modules.tts.ar_dur.ar_dur_predictor import ARDurPredictor
            hp_dur_model = self.hp_dur_model = set_hparams(f'{self.dur_exp_name}/config.yaml', global_hparams=False)
            hp_dur_model['frames_multiple'] = hparams['frames_multiple']
            self.dur_model = ARDurPredictor(
                hp_dur_model, hp_dur_model['dur_txt_hs'], hp_dur_model['dur_model_hidden_size'],
                hp_dur_model['dur_model_layers'], len(self.token_encoder),
                hp_dur_model['dur_code_size'],
                use_rot_embed=hp_dur_model.get('use_rot_embed', False), 
                precision=self.precision
                )
            load_ckpt(self.dur_model, f'{self.dur_exp_name}', 'dur_model')
            self.dur_model.eval()
            self.dur_model.to(self.device, dtype=self.precision)
        else:
            if 'lm' in (dur_exp_name_eles := [n.lower() for n in self.dur_exp_name.split('_')]):
                if 'seq2seq' in dur_exp_name_eles:
                    self.dur_model_type = 'lm_seq2seq'
                else:
                    self.dur_model_type = 'lm'
            elif 'dit' in dur_exp_name_eles:
                self.dur_model_type = 'dit'
            if self.dur_model_type in ['lm', 'lm_seq2seq']:
                from modules.tts.ar_dur.dur_lm import build_dur_model
                hp_dur_model = self.hp_dur_model = set_hparams(f'{self.dur_exp_name}/config.yaml', global_hparams=False)
                self.dur_model = build_dur_model(hp_dur_model, vocab_size=810, padding_idx=797)
                self.dur_model.hparams = {}
                self.dur_model.eval()
                load_ckpt(self.dur_model, self.dur_exp_name, 'model', mmap=True)
                self.dur_model.to(self.device)
            elif self.dur_model_type == 'dit':
                from modules.tts.scriptspeech.dit_dur import build_dur_model
                hp_dur_model = self.hp_dur_model = set_hparams(f'{self.dur_exp_name}/config.yaml', global_hparams=False)
                self.dur_model = build_dur_model(hp_dur_model)
                self.dur_model.hparams = {}
                self.dur_model.eval()
                load_ckpt(self.dur_model, self.dur_exp_name, 'model', strict=True, mmap=True)
                self.dur_model.to(self.device)
                
    def _encode_caption_for_infer(self, captions: List[str]):
        """
        与训练完全一致的 caption 编码：
        - 返回 (caption_embs, caption_mask, caption_lens, caption_text_mark[0/1])
        - mark 只有 0/1
        """
        inputs = self.caption_tokenizer(
            captions,
            padding=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_masks = inputs.attention_mask.to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=self.precision, enabled=True):
            caption_embs = self.caption_encoder(
                input_ids, return_dict=False,
                attention_mask=attention_masks,
            )[0]

        # mask 掉 padding
        caption_embs = caption_embs * attention_masks[..., None]
        caption_lens = attention_masks.sum(-1)

        caption_text_mark = None
        if hparams.get('use_caption_text_mark', False):
            # 与训练一致：只输出 0/1 mask
            caption_text_mark = build_audio_mask_from_ids(
                input_ids=input_ids,
                attention_mask=attention_masks,
                tokenizer=self.caption_tokenizer
            ).to(input_ids.device).long()

        return caption_embs, attention_masks, caption_lens, caption_text_mark


    def _ensure_caption_special_tokens(self):
        """
        缓存 caption tokenizer 的特殊 tag 的 id，供标注时使用。
        不修改词表，只做 id 查询；若未训练相应 tag，id 由 tokenizer 决定。
        """
        tok = self.caption_tokenizer
        self._caption_special_ids = {
            "S1_START":   tok.convert_tokens_to_ids("<S1>"),
            "S1_END":     tok.convert_tokens_to_ids("</S1>"),
            "AUDIO_START":tok.convert_tokens_to_ids("<Audio>"),
            "AUDIO_END":  tok.convert_tokens_to_ids("</Audio>"),
            "BGM_START":  tok.convert_tokens_to_ids("<BGM>"),
            "BGM_END":    tok.convert_tokens_to_ids("</BGM>"),
            "TAG_START":  tok.convert_tokens_to_ids("<TAG>"),
            "TAG_END":    tok.convert_tokens_to_ids("</TAG>"),
            "GP_START":   tok.convert_tokens_to_ids("<GPROMPT>"),
            "GP_END":     tok.convert_tokens_to_ids("</GPROMPT>"),
        }
    def _norm_caption_tags(self, s: str) -> str:
        """
        规范大小写，避免数据里出现 <tag>/<audio>/<bgm>/<gprompt> 等小写写法。
        """
        s = (s or "").strip()
        if not s:
            return s
        s = s.replace("<tag>", "<TAG>").replace("</tag>", "</TAG>")
        s = s.replace("<audio>", "<Audio>").replace("</audio>", "</Audio>")
        s = s.replace("<bgm>", "<BGM>").replace("</bgm>", "</BGM>")
        s = s.replace("<gprompt>", "<GPROMPT>").replace("</gprompt>", "</GPROMPT>")
        return s
    @torch.no_grad()
    def _build_caption_mark_from_ids(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        将带 tag 的 caption token 序列转为多类别标注：
        0 = 其它/填充
        1 = 文本（仅 <S1>…</S1> 内）
        2 = 音效（<Audio>…</Audio>）
        3 = BGM（<BGM>…</BGM>）

        规则：
        - <S1>…</S1> 内标 1
        - <Audio>…</Audio> 内标 2
        - <BGM>…</BGM> 内标 3
        - 进入 <GPROMPT>…</GPROMPT> 或 <TAG>…</TAG> 段落时强制置 0（不继承外层）
        - 所有起止 tag token 本身标 0
        """
        ids = getattr(self, "_caption_special_ids", None)
        if ids is None:
            self._ensure_caption_special_tokens()
            ids = self._caption_special_ids

        start2label = {
            ids["S1_START"]: 1,
            ids["AUDIO_START"]: 2,
            ids["BGM_START"]: 3,
        }
        # 这两类段落内强制置 0，不继承外层状态
        start_zero = {ids["GP_START"], ids["TAG_START"]}
        endset = {
            ids["S1_END"], ids["AUDIO_END"], ids["BGM_END"],
            ids["GP_END"], ids["TAG_END"]
        }

        B, T = input_ids.shape
        device = input_ids.device
        mark = torch.zeros((B, T), dtype=torch.long, device=device)

        for b in range(B):
            cur = 0
            stack = []
            for t in range(T):
                tok = int(input_ids[b, t])
                if tok in start2label:
                    stack.append(cur)
                    cur = start2label[tok]
                    # tag token 本身标 0
                    continue
                if tok in start_zero:
                    stack.append(cur)
                    cur = 0
                    continue
                if tok in endset:
                    cur = stack.pop() if stack else 0
                    continue
                # 普通内容：按当前段落类型打标
                mark[b, t] = cur

        # 去掉 padding
        return mark * attention_mask.long()


    def build_frontend_model(self):
        if self.use_old_aligner:
            from modules.tts.frontend_lm.whisper.whisper_small import Whisper
            self.aligner_lm = Whisper()
            load_ckpt(self.aligner_lm, f'{self.frontend_exp_name}', 'model')
            self.aligner_lm.eval()
            self.aligner_lm.to(self.device, dtype=self.precision)
            self.kv_cache = None
            self.hooks = None
        else:
            from modules.asr.scriptasr.build_model_utils import build_asr_model
            if self.frontend_exp_name.endswith('ckpt'):
                aligner_hparams = set_hparams(f'{Path(self.frontend_exp_name).parent}/config.yaml', global_hparams=False)
            else:
                aligner_hparams = set_hparams(f'{self.frontend_exp_name}/config.yaml', global_hparams=False)
            self.aligner_lm = build_asr_model(aligner_hparams, init_pretrained=False, vocab_size=6800, padding_idx=797)
            self.aligner_lm.eval()
            load_ckpt(self.aligner_lm, self.frontend_exp_name, 'model', strict=True, mmap=True)
            self.aligner_lm.to(self.device)

    ''' 加载模型参数 '''
    def build_model(self, device):
        self.device = device

        set_hparams(f'{self.dit_exp_name}/config.yaml', print_hparams=False)
        hparams['use_fsdp'] = False

        ''' Load Dict '''
        ling_dict = json.load(open('egs/tts/megatts3_dict.json'))
        self.ling_dict = {k: TokenTextEncoder(None, vocab_list=ling_dict[k], replace_oov='<UNK>') for k in ['phone', 'tone']}
        self.token_encoder = token_encoder = self.ling_dict['phone']
        # self.ph_replace_table = json.load(open(f"{current_dir}/ph_replace_table.json"))
        self.ph_replace_table = {'en': {}, 'zh': {}}
        self.sil_ph = self.ling_dict['phone'].sil_phonemes()

        ''' Load Duration LM '''
        self.build_dur_model()

        ''' Load DiT '''
        self._build_model()
        if self.use_ema and hparams.get('use_ema', False):
            load_ckpt(self.dit, f'{self.dit_exp_name}', 'ema_model', strict=False, mmap=True)
        else:
            load_ckpt(self.dit, f'{self.dit_exp_name}', 'dit', strict=False, mmap=True)
        self.vae.eval()
        self.vae.to(self.device, dtype=self.precision)
        self.dit.eval()
        self.dit.to(device, dtype=self.precision)
        self.cfg_mask_token_phone = 302 - 1
        self.cfg_mask_token_tone = 32 - 1
        self.caption_encoder.to(device, dtype=self.precision)

        ''' Load Frontend LM '''
        self.build_frontend_model()

        ''' ASR '''
        from modules.asr.sensevoice.sensevoice_api import build_asr_model
        self.asr_model = build_asr_model(device)

        ''' VAD '''
        from silero_vad import load_silero_vad
        self.vad_model = load_silero_vad()

        self.lock = threading.Lock()
        self._ensure_caption_special_tokens()

    ''' 文本预处理 '''
    def preprocess_text(self, input_text: SSML, ph_replace_table=None, use_sa_frontend=False, chunk_num_words_zh=60, chunk_num_words_en=130):

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
            text_segs = SSML.chunk_text_with_breaks(input_text, limit=chunk_num_words_en, language_type='en', debug=False)
        else:
            # print('input_text', input_text)
            input_text.normalize(_normalize_text_zh)
            # print('input_text', input_text)
            text_segs = SSML.chunk_text_with_breaks(input_text, limit=chunk_num_words_zh, language_type='zh', debug=False)
            # print('text_segs', text_segs)

        return text_segs
    
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

    def chunk_wavs_vad(self, wav_16k=None, speech_timestamps=None, chunk_duration=10, max_duration=60):
        if speech_timestamps is None and wav_16k is not None:
            from silero_vad import load_silero_vad, get_speech_timestamps
            wav_16k = wav_16k[:int(16000 * max_duration * 1.2)]
            speech_timestamps = get_speech_timestamps(
                wav_16k,
                self.vad_model,
                return_seconds=True,  # Return speech timestamps in seconds (default is samples)
            )
        timestamp_end_idx = -1   # include
        chunk_end_offsets = []
        last_end = 0
        for timestamp_idx, timestamp in enumerate(speech_timestamps):
            if timestamp['end'] > max_duration:
                timestamp_end_idx = timestamp_idx - 1
                break
            if timestamp['end'] - last_end > chunk_duration:
                chunk_end_offsets.append(timestamp['end'])
                last_end = timestamp['end']
        else:
            timestamp_end_idx = len(speech_timestamps) - 1
        if timestamp_end_idx == -1:
            chunk_end_offsets.append(max_duration)
        else:
            if len(chunk_end_offsets) > 0 and chunk_end_offsets[-1] != speech_timestamps[timestamp_end_idx]['end']:
                chunk_end_offsets.append(speech_timestamps[timestamp_end_idx]['end'])

        chunk_offsets = [speech_timestamps[0]['start']] + chunk_end_offsets
        chunk_offsets = [(chunk_offsets[i], chunk_offsets[i+1]) for i in range(len(chunk_offsets)-1)]
        
        if len(chunk_offsets) == 0:
            chunk_offsets = [(0, max_duration)]
        
        return chunk_offsets
    
    def trim_silence_vad(self, wav_16k=None, speech_timestamps=None, max_duration=60, max_silence=1.0):
        if speech_timestamps is None and wav_16k is not None:
            from silero_vad import load_silero_vad, get_speech_timestamps
            wav_16k = wav_16k[:int(16000 * max_duration * 1.2)]
            speech_timestamps = get_speech_timestamps(
                wav_16k,
                self.vad_model,
                return_seconds=True,  # Return speech timestamps in seconds (default is samples)
            )
        from utils.audio.diarization_utils import merge_segments
        segments = merge_segments(speech_timestamps, max_silence=max_silence, max_voiced_duration=max_duration)
        cur_duration = 0
        for i in range(len(segments)):
            if (cur_duration + segments[i]['end'] - segments[i]['start'] > max_duration) and cur_duration > 0:
                return segments[:i]
            cur_duration += segments[i]['end'] - segments[i]['start']
        return segments

    def preprocess(self, audio_bytes, wav_path=None, topk_dur=1, **kwargs):
        wav_bytes = convert_to_wav_bytes(audio_bytes)

        ''' 准备音频和各种参数 '''
        device = self.device

        # Process reference text and wav
        wav, _ = librosa.core.load(wav_bytes, sr=self.sr)
        # Pad wav if necessary
        ws = hparams['win_size']
        if len(wav) % ws < ws - 1:
            wav = np.pad(wav, (0, ws - 1 - (len(wav) % ws)), mode='constant', constant_values=0.0).astype(np.float32)
        wav = np.pad(wav, (0, 12000), mode='constant', constant_values=0.0).astype(np.float32)

        wav_16k = librosa.resample(wav, orig_sr=self.sr, target_sr=16000)
        max_ref_duration = self.max_ref_duration

        def _process_alignment(alignment_tokens):
            ''' 将aligner_lm的输出进行处理，得到音素、声调、持续时间 '''
            ph_ref, tone_ref, dur_ref, _ = split_ph_timestamp(deepcopy(alignment_tokens))
            dur_ref = dur_ref.clamp_min(0)
            ph_ref = torch.Tensor(ph_ref)[None].to(self.device)
            tone_ref = torch.Tensor(tone_ref)[None].to(self.device)
            if dur_ref.sum() < prompt_max_frame:
                dur_ref[-1] += prompt_max_frame - dur_ref.sum()
            elif dur_ref.sum() > prompt_max_frame:
                len_diff = dur_ref.sum() - prompt_max_frame
                while True:
                    for i in range(len(dur_ref)):
                        if dur_ref[i] > 0:
                            dur_ref[i] -= 1
                            len_diff -= 1
                        if len_diff == 0:
                            break
                    if len_diff == 0:
                        break
            return ph_ref, tone_ref, dur_ref
        
        if self.use_old_aligner:
            
            chunk_wav_offsets = self.chunk_wavs_vad(wav_16k, chunk_duration=10, max_duration=max_ref_duration)
            wav = wav[int(chunk_wav_offsets[0][0] * self.sr): int(chunk_wav_offsets[-1][-1] * self.sr)]
            wav_16k = wav_16k[int(chunk_wav_offsets[0][0] * 16000): int(chunk_wav_offsets[-1][-1] * 16000)]

            ph_ref_lst = []
            tone_ref_lst = []
            dur_ref_lst = []
            fm = 160 * 8
            for chunk_wav in chunk_wav_offsets:
                chunk_start, chunk_end = chunk_wav
                chunk_start = int((chunk_start * 16000) // fm * fm)
                chunk_end = int((chunk_end * 16000) // fm * fm)
                
                wav_16k_chunk = wav_16k[chunk_start: chunk_end]

                ''' 使用aligner_lm进行音频特征提取 '''
                with torch.inference_mode():
                    whisper_wav = wav_16k_chunk
                    mel = torch.tensor(whisper.log_mel_spectrogram(whisper_wav).T, dtype=self.precision).to(self.device)[None].transpose(1,2)
                    prompt_max_frame = mel.size(2) // self.fm * self.fm
                    mel = mel[:, :, :prompt_max_frame]
                    token = torch.LongTensor([[798]]).to(device)
                    audio_features = self.aligner_lm.embed_audio(mel)
                    # for i in tqdm(range(1024)):
                    for i in range(1024):
                        logits = self.aligner_lm.logits(token, audio_features, None)
                        token_pred = torch.argmax(F.softmax(logits[:, -1], dim=-1), 1)[None]
                        token = torch.cat([token, token_pred], dim=1)
                        if token_pred[0] == 799:
                            break
                    alignment_tokens = token[0, 1:-1]

                ph_ref, tone_ref, dur_ref = _process_alignment(alignment_tokens)
                ph_ref_lst.append(ph_ref)
                tone_ref_lst.append(tone_ref)
                dur_ref_lst.append(dur_ref)

            ph_ref = torch.cat(ph_ref_lst, dim=1)
            tone_ref = torch.cat(tone_ref_lst, dim=1)
            dur_ref = torch.cat(dur_ref_lst)
        
        else:
            segments = self.trim_silence_vad(wav_16k, max_duration=max_ref_duration)
            wav = np.concatenate([wav[int(segment['start'] * self.sr): int(segment['end'] * self.sr)] for segment in segments])
            wav_16k = np.concatenate([wav_16k[int(segment['start'] * 16000): int(segment['end'] * 16000)] for segment in segments])
            # remove gap leak
            wav = np.concatenate([wav, np.zeros(int(self.sr * 0.05))])
            wav_16k = np.concatenate([wav_16k, np.zeros(int(16000 * 0.05))])
            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=self.precision):
                # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                whisper_wav = torch.from_numpy(wav_16k)[None].to(self.device, self.precision)
                whisper_wav = whisper_wav[:, :whisper_wav.shape[-1] // 1280 * 1280]
                prompt_max_frame = whisper_wav.shape[-1] // 160 // 8 * 8
                token = torch.LongTensor([798])[None, :].to(self.device)
                token = self.aligner_lm.inference(
                    whisper_wav, token, topk=1, temperature=0.7,
                    max_new_tokens=16384, eos_idx=799, use_tqdm=self.use_tqdm
                )
            alignment_tokens = token[0]
            ph_ref, tone_ref, dur_ref = _process_alignment(alignment_tokens)

        if DEBUG:
            # print(f"{ph_ref.shape = }")
            # print(f"{dur_ref.shape = }")
            print(f"{dur_ref.max() = } {dur_ref.min() = }")
            # print(f"{dur_ref = }")

        mel2ph_ref = self.length_regulator(dur_ref[None]).to(self.device)
        mel2ph_ref = mel2ph_ref[:, :mel2ph_ref.size(1)//self.fm*self.fm]
        dur_ref = mel2token_to_dur(mel2ph_ref)

        if topk_dur > 1:
            self.dur_model.hparams["infer_top_k"] = topk_dur
        else:
            self.dur_model.hparams["infer_top_k"] = None

        with torch.inference_mode():
            ''' Forward WavVAE to obtain: prompt latent '''
            wav = torch.tensor(wav, dtype=self.precision)[None].to(device)
            with torch.autocast(device_type='cuda', dtype=self.precision):
                vae_latent = self.vae.encode_latent(wav)
            vae_latent = vae_latent[:, :mel2ph_ref.size(1)//4]
        
            ''' Duration Prompting '''
            if self.use_old_dur:
                dur_tokens_2d_ = mel2token_to_dur(mel2ph_ref, ph_ref.shape[1]).clamp(
                        max=self.hp_dur_model.get('dur_code_size', self.hp_dur_model.get('dur_max_value', 128)) - 1) + 1
    
                ctx_dur_tokens = dur_tokens_2d_.clone().flatten(0, 1).to(self.device)
                txt_tokens_flat_ = ph_ref.flatten(0, 1)
                ctx_dur_tokens = ctx_dur_tokens[txt_tokens_flat_ > 0][None]

                last_dur_pos_prompt = ctx_dur_tokens.shape[1]
                dur_spk_pos_ids_flat = range(0, last_dur_pos_prompt)
                dur_spk_pos_ids_flat = torch.LongTensor([dur_spk_pos_ids_flat]).to(mel2ph_ref.device)

                _, incremental_state_dur_prompt = self.dur_model.infer(
                    ph_ref, {'tone': tone_ref}, None, None, None,
                    ctx_vqcodes=ctx_dur_tokens, spk_pos_ids_flat=dur_spk_pos_ids_flat, return_state=True)
            else:
                if self.dur_model_type in ['lm', 'lm_seq2seq']:
                    merged_ph_tokens = map_phone_to_tokendict({'txt_token': ph_ref, 'tone': tone_ref}, pad_bos_eos=False)
                    try:
                        with torch.autocast(device_type='cuda', dtype=self.precision):
                            dur_start_pos = self.dur_model.prefill(merged_ph_tokens, dur_ref.clamp(0, self.dur_model.config.dur_max_value).to(self.device))
                    except:
                        print(f"{wav_path = }")
                        print(f"{merged_ph_tokens.shape = }")
                        print(f"{ph_ref.shape = }")
                        print(f"{tone_ref.shape = }")
                        print(f"{dur_ref.shape = }")
                        traceback.print_exc()
                        sys.exit()

            ''' ASR '''
            from modules.asr.sensevoice.sensevoice_api import run_asr_model
            text = run_asr_model([wav_16k], self.asr_model, with_segments=False)[0]['text_normed']

            # print('text_ref', text)
            # print(f'{dur_ref = }')

            ret = {
                'text_ref': text,
                'ph_ref': ph_ref.cpu(),
                'tone_ref': tone_ref.cpu(),
                'dur_ref': dur_ref.cpu(),
                'mel2ph_ref': mel2ph_ref.cpu(),
                'vae_latent': vae_latent.cpu(),
            }
            if self.use_old_dur:
                ret['incremental_state_dur_prompt'] = move_to_cpu(incremental_state_dur_prompt)
                ret['ctx_dur_tokens'] = ctx_dur_tokens.cpu()
            else:
                if self.dur_model_type in ['lm', 'lm_seq2seq']:
                    ret['dur_start_pos'] = dur_start_pos

            return ret

    def _pad_silence_head_tail(
        self,
        ph_pred: torch.Tensor,
        tone_pred: torch.Tensor,
        dur_pred: torch.Tensor,
        seconds: float = 2.0,
        sil_ph: int = 145,
        sil_tone: int = 0,
        timestep_hz: int = 100,
    ):
        """
        在首尾各插入给定秒数的静音（以 dur 的时间步为 1/100 秒计）。
        - ph_pred/tone_pred/dur_pred: 形状 [B, T]
        - 返回同形状三元组（首尾各多 1 个 token）
        """
        assert ph_pred.dim() == 2 and tone_pred.dim() == 2 and dur_pred.dim() == 2, \
            "ph_pred/tone_pred/dur_pred 必须是 [B, T] 形状"
        B = ph_pred.size(0)
        sil_frames = int(round(seconds * timestep_hz))

        ph_pad   = torch.full((B, 1), sil_ph,   device=ph_pred.device,  dtype=ph_pred.dtype)
        tone_pad = torch.full((B, 1), sil_tone, device=tone_pred.device, dtype=tone_pred.dtype)
        dur_pad  = torch.full((B, 1), sil_frames, device=dur_pred.device, dtype=dur_pred.dtype)

        ph_pred   = torch.cat([ph_pad,   ph_pred,   ph_pad],   dim=1)
        tone_pred = torch.cat([tone_pad, tone_pred, tone_pad], dim=1)
        dur_pred  = torch.cat([dur_pad,  dur_pred,  dur_pad],  dim=1)
        return ph_pred, tone_pred, dur_pred

    @torch.inference_mode()
    def process_text_seg(self, t_i, text, len_text_segs, profile,
                        text_ref,
                        ph_ref, 
                        tone_ref,
                        dur_ref,
                        dur_start_pos,
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
                        normalize_dur,
                        return_timestamp,
                        timestamp_postprocess,
                        use_sa_frontend,
                        time_step,
                        w_all, w_txt, w_cap, w_ref, seq_cfg_w, timestep_annealing_w, use_amo_sampler,
                        caption,                  # 直接传完整 caption
                        prompttts_kwargs):
        # 是否启用 PromptTTS 条件
        is_prompttts = isinstance(caption, str) and (caption.strip() != '')

        # 空文本段直接返回静音占位
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
                from modules.tts.frontend_lm.sa_frontend import call_sa_frontend
                sa_ret = call_sa_frontend(text.sa_ssml_str, debug=0)
                if sa_ret is None:  # 文本不合法，跳过
                    print(f'文本段落{t_i}/{len_text_segs}不合法，跳过')
                    return
                text_sa, ph_tokens, tone_tokens, alignment_sa = sa_ret
                # 覆盖 text（SA 前端已经做了规范化）
                new_text = SSML(text_sa)
                new_text.rate = text.rate
                new_text.pause_at_start = text.pause_at_start
                new_text.pause_at_end = text.pause_at_end
                text = new_text

                ph_pred = self.ling_dict['phone'].encode(' '.join(ph_tokens))
                ph_pred = torch.LongTensor(ph_pred)[None].to(self.device)
                tone_pred = self.ling_dict['tone'].encode(' '.join(tone_tokens))
                tone_pred = torch.LongTensor(tone_pred)[None].to(self.device)

        ''' Refeine Phonemes and Tones using SSML'''
        if not use_sa_frontend:
            ph_pred, tone_pred, ph2word = self.refine_ph_tone(text, ph_pred, tone_pred)

        ''' Caption encoding（与训练一致：直接编码完整 caption；mark 为 0/1） '''
        has_caption = isinstance(caption, str) and (caption.strip() != '')
        if has_caption:
            caption_embs, caption_mask, caption_lens, caption_text_mark = \
                self._encode_caption_for_infer([caption])
        else:
            caption_embs, caption_mask, caption_lens, caption_text_mark = \
                self._encode_caption_for_infer([''])
            caption_embs.zero_()

        ''' Duration Prediction '''
        with Timer('Duration Prediction', enable=profile):
            if self.use_old_dur:
                last_dur_token = ctx_dur_tokens[:, -1:]
                last_dur_pos = last_dur_pos_prompt
                txt_len = ph_pred.shape[1]
                dur_spk_pos_ids_flat = range(last_dur_pos, last_dur_pos + txt_len)
                dur_spk_pos_ids_flat = torch.LongTensor([dur_spk_pos_ids_flat]).to(self.device)
                last_dur_pos = last_dur_pos + txt_len
                with model_lock(self.lock):
                    dur_pred = self.dur_model.infer(
                        ph_pred, {'tone': tone_pred}, None, None, None,
                        incremental_state=move_to_cuda(incremental_state_dur_prompt, int(self.device.lstrip('cuda:'))),
                        first_decoder_inp=last_dur_token.to(self.device),
                        spk_pos_ids_flat=dur_spk_pos_ids_flat, use_tqdm=False
                    )
                dur_pred = dur_pred - 1
            else:
                if self.dur_model_type == 'lm':
                    merged_ph_tokens = map_phone_to_tokendict({'txt_token': ph_pred, 'tone': tone_pred}, pad_bos_eos=False)
                    with model_lock(self.lock), torch.autocast(device_type='cuda', dtype=self.precision):
                        dur_pred = self.dur_model.inference(
                            txt_tokens=merged_ph_tokens,
                            start_pos=dur_start_pos,
                            temperature=dur_disturb, use_tqdm=self.use_tqdm
                        )
                elif self.dur_model_type == 'lm_seq2seq':
                    merged_ph_tokens = map_phone_to_tokendict({'txt_token': ph_pred, 'tone': tone_pred}, pad_bos_eos=False)
                    with model_lock(self.lock):
                        dur_infer_kwargs = dict(
                            txt_tokens=merged_ph_tokens,
                            condition=caption_embs,  # 与训练一致：时长模型条件为 caption_embs
                            start_pos=dur_start_pos,
                            temperature=dur_disturb, 
                            topk=5,
                            use_tqdm=self.use_tqdm
                        )
                        if self.dur_model.config.modeling_type == 'ar':
                            dur_infer_kwargs['dur_tokens'] = dur_ref.clamp(0, self.dur_model.config.dur_max_value)
                        with torch.autocast(device_type='cuda', dtype=self.precision):
                            dur_pred = self.dur_model.inference(**dur_infer_kwargs)
                elif self.dur_model_type == 'dit':
                    from inference.tts.dur_model_infer import forward_dur_dit
                    with model_lock(self.lock), torch.autocast(device_type='cuda', dtype=self.precision):
                        dur_pred = forward_dur_dit(
                            self.dur_model, ph_pred, tone_pred, text.text_str,
                            ph_ref, tone_ref, text_ref, dur_ref, caption_embs, caption_lens,
                            self.cfg_mask_text_token, self.cfg_mask_token_phone, self.cfg_mask_token_tone,
                            self.device
                        )

            if dur_pred.max() < 2 and dur_pred.min() == 0:
                print(f"{text_ref = }")
                print(f"{dur_ref = }")
                print(f"{text = }")
                try:
                    print(f"{ph_tokens = }")
                except:
                    pass
                print(f"{dur_pred = }")
                print(f"{dur_pred.max() = } {dur_pred.min() = }")

            ''' Normalize Speech Speed '''
            if normalize_dur and dur_pred.shape[1] > 10:
                sil_mask_pred = torch.zeros_like(ph_pred)
                sil_mask_ref = torch.zeros_like(ph_ref)
                for p in self.sil_ph:
                    sil_mask_pred[ph_pred == self.ling_dict['phone'].encode(p)[0]] = 1
                    sil_mask_ref[ph_ref == self.ling_dict['phone'].encode(p)[0]] = 1
                z_dur_pred = torch.log1p(dur_pred.float())
                z_dur_ref  = torch.log1p(dur_ref.float())
                if sil_mask_pred.sum() > 0 and sil_mask_ref.sum() > 0:
                    diff_dur_sil = z_dur_ref[sil_mask_ref == 1].mean() - z_dur_pred[sil_mask_pred == 1].mean()
                    z_dur_pred[sil_mask_pred == 1] = z_dur_pred[sil_mask_pred == 1] + diff_dur_sil
                if (sil_mask_pred != 1).sum() > 0 and (sil_mask_ref != 1).sum() > 0:
                    diff_dur_nonsil = z_dur_ref[sil_mask_ref != 1].mean() - z_dur_pred[sil_mask_pred != 1].mean()
                    z_dur_pred[sil_mask_pred != 1] = z_dur_pred[sil_mask_pred != 1] + diff_dur_nonsil
                dur_pred = torch.expm1(z_dur_pred).clamp_min(0)
                d_floor = torch.floor(dur_pred)
                frac = (dur_pred - d_floor).clamp(0, 1)
                dur_pred = (d_floor + torch.bernoulli(frac)).long()
            ''' Control Speech Speed '''
            dur_pred = torch.round(dur_pred.float() / text.rate).long()
            dur_pred = dur_pred.clamp(0, self.hp_dur_model.get('dur_code_size', self.hp_dur_model.get('dur_max_value', 128)) - 1)
            if t_i < len_text_segs - 1:
                dur_pred[:, -1] = dur_pred[:, -1] + 32  # crossfade
            else:
                dur_pred[:, -1] = dur_pred[:, -1].clamp(32, 80)

            if self.use_old_dur:
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

            ''' Add Breaks / pauses '''
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
                    if ph_pred[:, -1].item() in [148, 163, 166, 153, 165, 147]:
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

            # 去掉首尾静音（数值为 0 秒，这里仅保持 token 对齐）
            ph_pred, tone_pred, dur_pred = self._pad_silence_head_tail(ph_pred, tone_pred, dur_pred, seconds=0)

        # 记录可视化的 ph/tone
        ph_pred_lst[t_i] = self.ling_dict['phone'].decode(ph_pred.squeeze().cpu().numpy()).split(' ')
        tone_pred_lst[t_i] = self.ling_dict['tone'].decode(tone_pred.squeeze().cpu().numpy()).split(' ')

        # 对齐 mel2ph
        dur_sum = dur_pred.sum()
        vqs = hparams.get('vq_stride', 8)
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
                    words_timestamps_cur = {'words': words, 'timestamps': timestamps}
                words_timestamps[t_i] = (words_timestamps_cur)
            except IndexError as err:
                handle_exacption(err, text)

        mel2ph_pred = self.length_regulator(dur_pred).to(self.device)

        ''' DiT target speech generation '''
        with Timer('DiT target speech generation', enable=profile):

            if prompttts_kwargs.get('ref_dur_only', False):
                # 仅用时长，不接参考音频
                mel2ph_pred = mel2ph_pred[:, :mel2ph_pred.size(1)//self.fm*self.fm].repeat(3, 1)
                target_size = mel2ph_pred.size(1)//4

                lat = torch.zeros((1, target_size, 32)).to(self.device)
                ctx_mask = torch.zeros((1, target_size, 1)).to(self.device)

                text_inputs = self.dit_text_tokenizer(text.text_str, padding=True, return_tensors='pt').to(self.device)
                txt_tokens = text_inputs['input_ids']
                txt_mask = text_inputs['attention_mask'].bool()
                txt_tokens[~txt_mask] = self.cfg_mask_text_token

                ph_seq = ph_pred
                tone_seq = tone_pred
                en_tone_idx = ~((tone_seq == 4) | ( (11 <= tone_seq) & (tone_seq <= 15)) | (tone_seq == 0))
                tone_seq[en_tone_idx] = 3

            else:
                # 拼接参考段 + 目标段
                mel2ph_pred = torch.cat((mel2ph_ref, mel2ph_pred+ph_ref.size(1)), dim=1)
                if seq_cfg_w is None:
                    mel2ph_pred = mel2ph_pred[:, :mel2ph_pred.size(1)//self.fm*self.fm].repeat(5, 1)
                else:
                    mel2ph_pred = mel2ph_pred[:, :mel2ph_pred.size(1)//self.fm*self.fm].repeat(3, 1)
                target_size = mel2ph_pred.size(1)//4

                if hparams.get('use_sparse_dur', False):
                    _dur_pred_cat = torch.cat([dur_ref, dur_pred], dim=1)
                    mel2ph_sparse = compute_mel2aug_from_dur(
                        _dur_pred_cat.int().squeeze().cpu().numpy().tolist(),
                        gap_mode=hparams.get('sparse_dur_mode', 'proportional'),
                        gap_frames=hparams.get('sparse_dur_frames', 4),
                        gap_alpha=hparams.get('sparse_dur_alpha', 0.2),
                        min_keep=hparams.get('sparse_dur_min_keep', 1),
                        keep_ratio=hparams.get('sparse_dur_keep_ratio'),
                        symmetric=hparams.get('sparse_dur_symmetric', True),
                    )
                    if seq_cfg_w is None:
                        mel2ph_sparse = torch.stack([mel2ph_sparse] * 5).to(self.device)
                    else:
                        mel2ph_sparse = torch.stack([mel2ph_sparse] * 3).to(self.device)
                    mel2ph_sparse = mel2ph_sparse[:, :mel2ph_pred.shape[1]]

                ctx_mask = torch.ones_like(vae_latent[:, :, 0:1])
                lat = F.pad(vae_latent, (0, 0, 0, target_size - vae_latent.size(1)), mode='constant', value=0)
                ctx_mask = F.pad(ctx_mask, (0, 0, 0, target_size - ctx_mask.size(1)), mode='constant', value=0)

                text_inputs = self.dit_text_tokenizer(text_ref + text.text_str, padding=True, return_tensors='pt').to(self.device)
                txt_tokens = text_inputs['input_ids']
                txt_mask = text_inputs['attention_mask'].bool()
                txt_tokens[~txt_mask] = self.cfg_mask_text_token

                # 英文声调置 3
                ph_seq = torch.cat((ph_ref, ph_pred), dim=1)
                tone_seq = torch.cat((tone_ref, tone_pred), dim=1)
                en_tone_idx = ~((tone_seq == 4) | ( (11 <= tone_seq) & (tone_seq <= 15)) | (tone_seq == 0))
                tone_seq[en_tone_idx] = 3

            # ===== CFG 复制部分 =====
            if seq_cfg_w is None:
                # 5 路：cond_all / cond_txt / cond_cap / cond_ref / uncond
                ph_seq = torch.cat([
                    ph_seq,
                    ph_seq,
                    torch.full(ph_seq.size(), self.cfg_mask_token_phone, device=self.device),
                    torch.full(ph_seq.size(), self.cfg_mask_token_phone, device=self.device),
                    torch.full(ph_seq.size(), self.cfg_mask_token_phone, device=self.device)
                ], dim=0)
                tone_seq = torch.cat([
                    tone_seq,
                    tone_seq,
                    torch.full(tone_seq.size(), self.cfg_mask_token_tone, device=self.device),
                    torch.full(tone_seq.size(), self.cfg_mask_token_tone, device=self.device),
                    torch.full(tone_seq.size(), self.cfg_mask_token_tone, device=self.device)
                ], dim=0)

                txt_tokens = torch.cat([
                    txt_tokens,
                    txt_tokens,
                    torch.full(txt_tokens.size(), self.cfg_mask_text_token, device=self.device),
                    torch.full(txt_tokens.size(), self.cfg_mask_text_token, device=self.device),
                    torch.full(txt_tokens.size(), self.cfg_mask_text_token, device=self.device),
                ], dim=0)
                txt_mask = torch.cat([txt_mask] * 5, dim=0)

                cap_zero = torch.zeros_like(caption_embs)
                caption_embs = torch.cat([
                    caption_embs,  # cond_all
                    cap_zero,      # cond_txt（不带 caption）
                    caption_embs,  # cond_cap（只看 caption）
                    cap_zero,      # cond_ref
                    cap_zero       # uncond
                ], dim=0)
                caption_lens = torch.cat([caption_lens] * 5, dim=0).long()
                if hparams.get('use_caption_text_mark', False) and (caption_text_mark is not None):
                    mark_zero = torch.zeros_like(caption_text_mark)
                    caption_text_mark = torch.cat([
                        caption_text_mark,
                        mark_zero,
                        caption_text_mark,
                        mark_zero,
                        mark_zero
                    ], dim=0)

                lat = torch.cat([
                    lat,
                    torch.zeros_like(lat),
                    torch.zeros_like(lat),
                    lat,
                    torch.zeros_like(lat)
                ], dim=0)
                ctx_mask = torch.cat([ctx_mask] * 5, dim=0)

                vad_mask = torch.ones_like(ctx_mask)[..., 0]
                vad_mask[:, :int(0.2 * 25)] = 0
                vad_mask[:, -int(0.2 * 25):] = 0

                seq_cfg_w = [w_all, w_txt, w_cap, w_ref]

            else:
                # 3 路：cond_all / cond_txt / uncond
                if is_prompttts and prompttts_kwargs.get('drop_phone', False):
                    ph_seq = torch.cat([torch.full_like(ph_seq, self.cfg_mask_token_phone)] * 3, dim=0)
                    tone_seq = torch.cat([torch.full_like(tone_seq, self.cfg_mask_token_tone)] * 3, dim=0)
                else:
                    ph_seq = torch.cat([
                        ph_seq,
                        ph_seq,
                        torch.full(ph_seq.size(), self.cfg_mask_token_phone, device=self.device)
                    ], dim=0)
                    tone_seq = torch.cat([
                        tone_seq,
                        tone_seq,
                        torch.full(tone_seq.size(), self.cfg_mask_token_tone, device=self.device)
                    ], dim=0)

                txt_tokens = torch.cat([
                    txt_tokens,
                    txt_tokens,
                    torch.full(txt_tokens.size(), self.cfg_mask_text_token, device=self.device)
                ], dim=0)
                txt_mask = torch.cat([txt_mask] * 3, dim=0)

                cap_zero = torch.zeros_like(caption_embs)
                # 与训练一致：cond_txt 不带 caption；uncond 也不带
                caption_embs = torch.cat([caption_embs, cap_zero, cap_zero], dim=0)
                caption_lens = torch.cat([caption_lens] * 3, dim=0).long()
                if hparams.get('use_caption_text_mark', False) and (caption_text_mark is not None):
                    mark_zero = torch.zeros_like(caption_text_mark)
                    caption_text_mark = torch.cat([caption_text_mark, mark_zero, mark_zero], dim=0)

                lat = torch.cat([
                    lat,
                    torch.zeros_like(lat),
                    torch.zeros_like(lat)
                ], dim=0)
                ctx_mask = torch.cat([ctx_mask] * 3, dim=0)

                vad_mask = torch.ones_like(ctx_mask)[..., 0]
                vad_mask[:, :int(0.2 * 25)] = 0
                vad_mask[:, -int(0.2 * 25):] = 0

            inputs = {
                'phone': ph_seq,
                'tone': tone_seq,
                "lat_ctx": lat * ctx_mask,
                "ctx_mask": ctx_mask,
                "mel2ph": mel2ph_pred,
                "txt_tokens": txt_tokens,
                'txt_mask': txt_mask,
                "caption_emb": caption_embs,
                "caption_lens": caption_lens,  # B
                "vad_mask": vad_mask,
            }
            if hparams.get('use_sparse_dur', False) and not prompttts_kwargs.get('ref_dur_only', False):
                inputs['mel2ph_sparse'] = mel2ph_sparse
            if hparams.get('use_caption_text_mark', False) and (caption_text_mark is not None):
                inputs['caption_text_mark'] = caption_text_mark

            with model_lock(self.lock), torch.autocast(device_type='cuda', dtype=self.precision):
                x = self.dit.inference(
                    inputs, timesteps=time_step, 
                    seq_cfg_w=seq_cfg_w, 
                    timestep_annealing_w=timestep_annealing_w,
                    use_amo_sampler=use_amo_sampler
                )

        # WavVAE decode
        with Timer('WavVAE decode', enable=profile):
            if not prompttts_kwargs.get('ref_dur_only', False):
                x[:, :vae_latent.size(1)] = vae_latent
            with model_lock(self.lock):
                with torch.autocast(device_type='cuda', dtype=self.precision):
                    wav_pred = self.vae.decode(x)[0,0].to(torch.float32)

        # Post-processing
        with Timer('Post-processing', enable=profile):
            hop_size = self.hp_vae['hop_size']
            vae_stride = self.hp_vae['vae_stride']
            if not prompttts_kwargs.get('ref_dur_only', False):
                wav_pred = wav_pred[vae_latent.size(1)*vae_stride*hop_size:]
            if wav_pred.abs().max() > 1:
                print('Wav amplitude exceed 1, clip it.')
                wav_pred = wav_pred / (wav_pred.abs().max())
            wav_pred = wav_pred.cpu().numpy()
            wav_pred_[t_i] = wav_pred


    def _forward_single(self, resource_context, input_text, time_step, w_all=1.0, w_txt=1.0, w_cap=1.0, w_ref=1.0, seq_cfg_w=None,
                speech_rate=1, timestep_annealing_w=(1.0, 0.0, 1.0), use_amo_sampler=False, return_timestamp=True, timestamp_postprocess=False, 
                return_format='wav', custom_ph_table=None, dur_disturb=0.1, dur_alpha=1.0, normalize_dur=False,
                num_parallel_workers=5, use_sa_frontend=True, chunk_num_words_zh=60, chunk_num_words_en=130, 
                caption=None,                      # 直接传完整 caption
                prompttts_kwargs=None, **kwargs):
        """
        单条输入版本，返回一个 MegaTTS3Output。
        """
        device = self.device
        incremental_state_dur_prompt = resource_context.get('incremental_state_dur_prompt')
        last_dur_pos_prompt = resource_context['ctx_dur_tokens'].shape[1] if 'ctx_dur_tokens' in resource_context else None
        prompttts_kwargs = prompttts_kwargs if prompttts_kwargs is not None else {}

        profile = os.environ.get('MEGA_PROFILE', 'false').strip().lower() == 'true'

        with torch.inference_mode():
            # 清洗输入
            input_text = ''.join(c for c in input_text if c.isprintable())
            input_text = SSML(input_text)
            input_text.rate = float(speech_rate)

            # 纯静音
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
                            
            # 预处理文本（切段）
            with Timer('preprocess text', enable=profile):
                ph_replace_table = deepcopy(self.ph_replace_table)
                if custom_ph_table is not None:
                    ph_replace_table.update(custom_ph_table)
                text_segs = self.preprocess_text(input_text, ph_replace_table, use_sa_frontend, chunk_num_words_zh, chunk_num_words_en)

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
                                    "text_ref": resource_context['text_ref'],
                                    "ph_ref": resource_context['ph_ref'].detach().clone().to(device), 
                                    "tone_ref": resource_context['tone_ref'].detach().clone().to(device),
                                    "dur_ref": resource_context['dur_ref'].detach().clone().to(device),
                                    "dur_start_pos": resource_context.get('dur_start_pos', None),
                                    "mel2ph_ref": resource_context['mel2ph_ref'].detach().clone().to(device),
                                    "vae_latent": resource_context['vae_latent'].detach().clone().to(device),
                                    "ctx_dur_tokens": resource_context['ctx_dur_tokens'].detach().clone().to(device) if 'ctx_dur_tokens' in resource_context else None,
                                    "incremental_state_dur_prompt": deepcopy(incremental_state_dur_prompt),
                                    "last_dur_pos_prompt": last_dur_pos_prompt,
                                    "wav_pred_": wav_pred_,
                                    "sil_pad_lst": sil_pad_lst,
                                    "ph_pred_lst": ph_pred_lst,
                                    "tone_pred_lst": tone_pred_lst,
                                    "words_timestamps": words_timestamps,
                                    "words_timestamps_post": words_timestamps_post,
                                    "dur_disturb": dur_disturb,
                                    "dur_alpha": dur_alpha,
                                    "normalize_dur": normalize_dur,
                                    "return_timestamp": return_timestamp,
                                    "timestamp_postprocess": timestamp_postprocess,
                                    "use_sa_frontend": use_sa_frontend,
                                    "time_step": time_step,
                                    "w_all": w_all,
                                    "w_txt": w_txt,
                                    "w_cap": w_cap,
                                    "w_ref": w_ref,
                                    "seq_cfg_w": seq_cfg_w,
                                    "timestep_annealing_w": timestep_annealing_w,
                                    "use_amo_sampler": use_amo_sampler,
                                    "caption": caption,                    # 传入完整 caption
                                    "prompttts_kwargs": prompttts_kwargs or {},
                                })
                    futs.append(future)
                
                _ = [f.result() for f in futs]

            words_timestamps = [s for s in words_timestamps if s is not None]
            wav_pred_ = [s for s in wav_pred_ if s is not None]
            ph_pred_lst = [s for s in ph_pred_lst if s is not None]
            tone_pred_lst = [s for s in tone_pred_lst if s is not None]

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

            # 响度对齐
            if len(wav_pred_) > 1:
                silent_speech = False
                meter = pyln.Meter(self.sr)
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


    def forward(self, resource_context, input_text, time_step, w_all=1.0, w_txt=1.0, w_cap=1.0, w_ref=1.0, seq_cfg_w=None,
                speech_rate=1, timestep_annealing_w=(1.0, 0.0, 1.0), use_amo_sampler=False, return_timestamp=True, timestamp_postprocess=False, 
                return_format='wav', custom_ph_table=None, dur_disturb=0.1, dur_alpha=1.0, normalize_dur=False,
                num_parallel_workers=5, use_sa_frontend=True, chunk_num_words_zh=60, chunk_num_words_en=130, 
                captions=None,                         # 直接传完整 caption（str 或与 input_text 等长的 list）
                prompttts_kwargs=None, **kwargs):
        """
        支持单条或多条输入：
        - input_text 为 str => 返回 MegaTTS3Output
        - input_text 为 List[str] => 返回 List[MegaTTS3Output]
        - captions：完整 caption，str 或与 input_text 等长的 List[str]/List[None]
        """
        def _broadcast(param, n, name):
            if isinstance(param, list):
                if len(param) != n:
                    raise ValueError(f"{name} 的长度({len(param)})必须与 input_text 列表长度({n})一致")
                return param
            else:
                return [param] * n  # 包含 None 的广播

        # 多条
        if isinstance(input_text, list):
            n = len(input_text)
            c_list = _broadcast(captions, n, "captions")

            outputs = []
            for i in range(n):
                out_i = self._forward_single(
                    resource_context=resource_context,
                    input_text=input_text[i],
                    time_step=time_step,
                    w_all=w_all, w_txt=w_txt, w_cap=w_cap, w_ref=w_ref, seq_cfg_w=seq_cfg_w,
                    speech_rate=speech_rate, timestep_annealing_w=timestep_annealing_w, use_amo_sampler=use_amo_sampler,
                    return_timestamp=return_timestamp, timestamp_postprocess=timestamp_postprocess,
                    return_format=return_format, custom_ph_table=custom_ph_table,
                    dur_disturb=dur_disturb, dur_alpha=dur_alpha, normalize_dur=normalize_dur,
                    num_parallel_workers=num_parallel_workers, use_sa_frontend=use_sa_frontend,
                    chunk_num_words_zh=chunk_num_words_zh, chunk_num_words_en=chunk_num_words_en,
                    caption=c_list[i],                        # 传入完整 caption
                    prompttts_kwargs=prompttts_kwargs, **kwargs
                )
                outputs.append(out_i)
            return outputs

        # 单条
        return self._forward_single(
            resource_context=resource_context,
            input_text=input_text,
            time_step=time_step,
            w_all=w_all, w_txt=w_txt, w_cap=w_cap, w_ref=w_ref, seq_cfg_w=seq_cfg_w,
            speech_rate=speech_rate, timestep_annealing_w=timestep_annealing_w, use_amo_sampler=use_amo_sampler,
            return_timestamp=return_timestamp, timestamp_postprocess=timestamp_postprocess,
            return_format=return_format, custom_ph_table=custom_ph_table,
            dur_disturb=dur_disturb, dur_alpha=dur_alpha, normalize_dur=normalize_dur,
            num_parallel_workers=num_parallel_workers, use_sa_frontend=use_sa_frontend,
            chunk_num_words_zh=chunk_num_words_zh, chunk_num_words_en=chunk_num_words_en,
            caption=captions,                               # 传入完整 caption
            prompttts_kwargs=prompttts_kwargs, **kwargs
        )



import os
import glob
from pathlib import Path

import torch
from tqdm import tqdm

from utils.commons.os_utils import kill_void
from utils.text.ssml_utils import SSML
from utils.audio.io import save_wav_bytes

if __name__ == '__main__':
    if os.path.isfile('.env.local'):
        from dotenv import load_dotenv
        load_dotenv('.env.local')

    # dit_exp_name = 'checkpoints/251010_prompttts_v2'
    # dit_exp_name = 'checkpoints/251017_prompttts_v2'
    dit_exp_name = 'checkpoints/251105_promptaudio_post_adaln'
    
    # dur_exp_name = 'checkpoints/250928_dur_lm_base_seq2seq_v2_txtcond_bias'
    dur_exp_name = 'checkpoints/251016_dur_lm_base_seq2seq_v4_txtcond_bias_fix'
    
    frontend_exp_name = 'checkpoints/250923_lm_mfa_seq2seq_small_wavlmlarge_long_robust'

    infer_ins = MegaTTS3DiTInfer(
        device=f'cuda:0',
        dit_exp_name=dit_exp_name,
        dur_exp_name=dur_exp_name,
        frontend_exp_name=frontend_exp_name,
        use_old_aligner=False,
        use_old_dur=True,
        max_ref_duration=60,
        use_ema=False,
    )
    kill_void()

    infer_kwargs = dict(
        time_step=100, 
        seq_cfg_w=(1.5, 3.0), 
        speech_rate=1.0, 
        dur_disturb=0.2, 
        timestep_annealing_w=(0.6, 0.6, 1.0),
        chunk_num_words_zh=80
    )

    # ========== 参考音频，保持单个 ==========
    wav_path = 'user/prompts/mega_eval_prompt0731/0_vocal.wav'
    with open(wav_path, 'rb') as file:
        file_content = file.read()
    print(f"| Start processing {wav_path}")
    resource_context = infer_ins.preprocess(file_content, wav_path)

    # ========== 多条输入 ==========
    # 你的 texts / local_prompts / global_prompts 三个列表，原样放在这里（不需要改内容）
    texts = [
        """<S1>Oh, no. No, you know what? He's not into that stuff anymore. He quit for me. Mm-hmm.</S1>""",
        """<S1>哦，不。不，你知道的？他不再喜欢那类东西了。他为了我辞职了。嗯嗯。</S1>""",

        """<S1>tracted by women with big chests. Um you know I might be looking at them or something but I I I</S1>""",
        """<S1>被漂亮的女性吸引。呃，你知道我可能正在看她们或者什么，但是我我我</S1>""",

        """<S1>You know how traditional my parents are. If they found out I got divorced and married to a black man, they'd crap in a sock.</S1>""",
        """<S1>你知道我父母有多传统。如果他们发现我离婚并嫁给了一个黑人，他们会气死的。</S1>""",

        """<S1>Okay, I'm finished with the cucumbers. I did go ahead and put some of the pickling cucumbers in.</S1>""",
        """<S1>好的，我已经处理完黄瓜了。我确实放了一些腌黄瓜进去。</S1>""",

        """<S1>I'm in a hurry. But it'll take at least a year before it makes it to the judge.</S1>""",
        """<S1>我赶时间。但至少需要一年时间才能到法官那里。</S1>""",

        """<S1>I bought my motorcycle last year because you guys recommended it. You gave it five stars.</S1>""",
        """<S1>我去年买了我的摩托车，因为你们推荐了它。你们给它评了五星级。</S1>""",

        """<S1>You said I needed a job. He gave me a job. Simple as that.</S1>""",
        """<S1>你说我需要一份工作。他给了我一份工作。就这么简单。</S1>""",

        """<S1>when we had that grudge match, over who had the shittiest life back home?</S1>""",
        """<S1>是我们进行那次关于谁在老家过得最糟糕的比赛时吗？</S1>""",

        """<S1>From me. I had a biopsy taken from the smallpox vaccination scar.</S1>""",
        """<S1>我从天花疫苗接种疤痕中做了活检。</S1>""",

        """<S1>I think our not-so-normal son might be going through some classic teenage rebellion.</S1>""",
        """<S1>我认为我们不那么正常的儿子可能正在经历一些典型的青少年叛逆。</S1>""",
    ]

    # 旧版本地 caption：<S1> + <TAG> 全量保留（我顺手补了 A_EN 最后一段缺失的 </TAG> 闭合）
    local_prompts = [
        """<S1>Oh, no. No, you know what? He's not into that stuff anymore.</S1><TAG>The overall environment sound is the clear speech of a woman. The woman, with long blond hair, starts speaking, gesturing with her arms bent at the waist, her palms facing her chest. She then stretches her arms outward and raises them up and down in an irregular manner.</TAG><S1>He quit for me.</S1><TAG>She continues, while lowering her arms and imitating holding a tennis ball in both hands.</TAG><S1>Mm-hmm.</S1><TAG>She then turns slightly to the right and looks, and finally she lowers her arms naturally and turns her head to look to the left side of the screen. The posture changes subtly, indicating a shift from explanation to relaxation.</TAG>""",
        """<S1>哦，不。不，你知道的？他不再喜欢那类东西了。</S1><TAG>The woman, with long blond hair, starts speaking, gesturing with her arms bent at the waist, her palms facing her chest. She then stretches her arms outward and raises them up and down in an irregular manner.</TAG><S1>他为了我辞职了。</S1><TAG>She continues, while lowering her arms and imitating holding a tennis ball in both hands.</TAG><S1>嗯嗯。</S1><TAG>She then turns slightly to the right and looks, and finally she lowers her arms naturally and turns her head to look to the left side of the screen. The posture changes subtly, indicating a shift from explanation to relaxation.</TAG>""",

        """<S1>tracted by women with big chests. Um you know I might be looking at them or something but I I I</S1><TAG>The overall environment sound is a quiet indoor ambiance. The man in the security uniform speaks, his facial expression changing subtly and his body moving slightly as he gestures with his hands to emphasize his words.</TAG>""",
        """<S1>被漂亮的女性吸引。呃，你知道我可能正在看她们或者什么，但是我我我</S1><TAG>The overall environment sound is a quiet indoor ambiance. The man in the security uniform speaks, his facial expression changing subtly and his body moving slightly as he gestures with his hands to emphasize his words.</TAG>""",

        """<S1>You know how traditional my parents are. If they found out I got divorced and married to a black man, they'd crap in a sock.</S1><TAG>The overall environment sound is quiet. A woman with long blonde hair speaks with an animated and concerned expression to a man in a green striped shirt. Her expression is intense as she talks.</TAG>""",
        """<S1>你知道我父母有多传统。如果他们发现我离婚并嫁给了一个黑人，他们会气死的。</S1><TAG>The overall environment sound is quiet. A woman with long blonde hair speaks with an animated and concerned expression to a man in a green striped shirt. Her expression is intense as she talks.</TAG>""",

        """<S1>Okay, I'm finished with the cucumbers. I did go ahead and put some of the pickling cucumbers in.</S1><TAG>The overall environment sound is minimal ambient sound. The woman kneels beside a row of small green plants. She explains something, gesturing with her hands, pointing at the plants or emphasizing certain points with her fingers. She then turns her head to the right of the frame to look at the plants, and uses her left hand to gently touch or adjust them. She continues to focus on the plants, using her left hand to point at different parts of the plant or area of the soil that she is discussing. She then turns her head to look at the camera and continues to explain.</TAG>""",
        """<S1>好的，我已经处理完黄瓜了。我确实放了一些腌黄瓜进去。</S1><TAG>The overall environment sound is minimal ambient sound. The woman kneels beside a row of small green plants. She explains something, gesturing with her hands, pointing at the plants or emphasizing certain points with her fingers. She then turns her head to the right of the frame to look at the plants, and uses her left hand to gently touch or adjust them. She continues to focus on the plants, using her left hand to point at different parts of the plant or area of the soil that she is discussing. She then turns her head to look at the camera and continues to explain.</TAG>""",

        """<S1>I'm in a hurry.</S1><TAG>The overall environment sound is faint, indistinct background chatter. The man on screen, with a slightly concerned expression, says.</TAG><S2>But it'll take at least a year before it makes it to the judge.</S2><TAG>An off-screen voice responds. The man on screen listens, his expression becoming more serious and contemplative.</TAG>""",
        """<S1>我赶时间。</S1><TAG>The overall environment sound is faint, indistinct background chatter. The man on screen, with a slightly concerned expression, says.</TAG><S1>但至少需要一年时间才能到法官那里。</S1><TAG>An off-screen voice responds. The man on screen listens, his expression becoming more serious and contemplative.</TAG>""",

        """<S1>I bought my motorcycle last year because you guys recommended it. You gave it five stars.</S1><TAG>The overall environment sound is indistinct background chatter and ambient room noise. The man's right hand is raised to his chest, and he gestures with his index finger, moving from time to time during the conversation. The woman maintains eye contact and listens attentively, occasionally adjusting her posture to emphasize engagement. The man's expressive gestures and focused body language indicate that this may be a crucial moment in the dialogue.</TAG>""",
        """<S1>我去年买了我的摩托车，因为你们推荐了它。你们给它评了五星级。</S1><TAG>The overall environment sound is indistinct background chatter and ambient room noise. The man's right hand is raised to his chest, and he gestures with his index finger, moving from time to time during the conversation. The woman maintains eye contact and listens attentively, occasionally adjusting her posture to emphasize engagement. The man's expressive gestures and focused body language indicate that this may be a crucial moment in the dialogue.</TAG>""",

        """<S1>You said I needed a job. He gave me a job. Simple as that.</S1><TAG>The overall environment sound is quiet. The man holds a small notepad, looking down at it as he speaks, quoting someone. The woman beside him listens intently with her arms crossed.</TAG>""",
        """<S1>你说我需要一份工作。他给了我一份工作。就这么简单。</S1><TAG>The overall environment sound is quiet. The man holds a small notepad, looking down at it as he speaks, quoting someone. The woman beside him listens intently with her arms crossed.</TAG>""",

        """<S1>when we had that grudge match, over who had the shittiest life back home?</S1><TAG>The overall environment sound is quiet with subtle ambient forest noises. The dark-haired woman speaks with an attentive posture, her mouth open and her eyes fixed on the other party. The blond woman listens intently, her head slightly tilted toward the speaker.</TAG>""",
        """<S1>是我们进行那次关于谁在老家过得最糟糕的比赛时吗？</S1><TAG>The overall environment sound is quiet with subtle ambient forest noises. The dark-haired woman speaks with an attentive posture, her mouth open and her eyes fixed on the other party. The blond woman listens intently, her head slightly tilted toward the speaker.</TAG>""",

        """<S1>From me. I had a biopsy taken from the smallpox vaccination scar.</S1><TAG>The overall environment sound is minimal. The woman initially faces the camera, then turns to her right and speaks. Her body language remains calm and focused, and she occasionally looks to her left and right as if explaining something or listening attentively.</TAG>""",
        """<S1>我从天花疫苗接种疤痕中做了活检。</S1><TAG>The overall environment sound is minimal. The woman initially faces the camera, then turns to her right and speaks. Her body language remains calm and focused, and she occasionally looks to her left and right as if explaining something or listening attentively.</TAG>""",

        """<S1>I think our not-so-normal son might be going through some classic teenage rebellion.</S1><TAG>The overall environment sound is quiet. The woman's expression changes subtly from a slight smile to a more serious look as her eyebrows raise slightly and she opens her mouth to speak. She looks at the person off-screen.</TAG>""",
        """<S1>我认为我们不那么正常的儿子可能正在经历一些典型的青少年叛逆。</S1><TAG>The overall environment sound is quiet. The woman's expression changes subtly from a slight smile to a more serious look as her eyebrows raise slightly and she opens her mouth to speak. She looks at the person off-screen.</TAG>""",
    ]

    # 每条对应的 <GPROMPT>（按 EN/CN 重复两次以与 texts/local_prompts 对齐）
    global_prompts = [
        """<GPROMPT>The video features a woman with long blond hair standing in a room, facing the camera. She wears a printed blouse with a mixture of dark and bright colors and a necklace. Her hair is neatly combed and falls naturally around her shoulders. The scene takes place in a domestic setting, possibly a living room, with a wooden cabinet with ornate metal handles and a flat-screen TV on top. The soft and warm lighting creates a relaxed atmosphere.</GPROMPT>""",
        """<GPROMPT>The video features a woman with long blond hair standing in a room, facing the camera. She wears a printed blouse with a mixture of dark and bright colors and a necklace. Her hair is neatly combed and falls naturally around her shoulders. The scene takes place in a domestic setting, possibly a living room, with a wooden cabinet with ornate metal handles and a flat-screen TV on top. The soft and warm lighting creates a relaxed atmosphere.</GPROMPT>""",

        """<GPROMPT>In the video, a man is standing in the middle of an indoor space, possibly an office or administrative area. The man wears a dark blue uniform with a badge that reads "SECURITY OFFICER" on his left arm, over a white shirt. He has short hair and a serious expression. In the background, there are several items that add character to the environment: a yellow wall is painted a bright yellow, adorned with a traditional cuckoo clock and a framed picture or poster depicting a yellow figure. There is also a dark blue object in the back left corner of the frame, which may be a box or container. The overall lighting is bright, creating a well-lit and clear environment. The camera focuses on the man and the camera does not move, and the picture is shaky.</GPROMPT>""",
        """<GPROMPT>In the video, a man is standing in the middle of an indoor space, possibly an office or administrative area. The man wears a dark blue uniform with a badge that reads "SECURITY OFFICER" on his left arm, over a white shirt. He has short hair and a serious expression. In the background, there are several items that add character to the environment: a yellow wall is painted a bright yellow, adorned with a traditional cuckoo clock and a framed picture or poster depicting a yellow figure. There is also a dark blue object in the back left corner of the frame, which may be a box or container. The overall lighting is bright, creating a well-lit and clear environment. The camera focuses on the man and the camera does not move, and the picture is shaky.</GPROMPT>""",

        """<GPROMPT>The video shows two people in a room, with the focus on a woman with long blond hair, wearing a pink floral shirt, a pink headband, and earrings. A man in a green striped shirt stands opposite her with his back mostly to the camera. The scene takes place in an indoor setting, with a wooden wall decorated with children's drawings, including an orange paper with a black line drawing and a purple paper with a drawing. A plaid shirt is hung on the wall. The lighting is warm, creating a casual and intimate atmosphere. The camera focuses on the woman's facial expressions.</GPROMPT>""",
        """<GPROMPT>The video shows two people in a room, with the focus on a woman with long blond hair, wearing a pink floral shirt, a pink headband, and earrings. A man in a green striped shirt stands opposite her with his back mostly to the camera. The scene takes place in an indoor setting, with a wooden wall decorated with children's drawings, including an orange paper with a black line drawing and a purple paper with a drawing. A plaid shirt is hung on the wall. The lighting is warm, creating a casual and intimate atmosphere. The camera focuses on the woman's facial expressions.</GPROMPT>""",

        """<GPROMPT>The video shows a woman in a greenhouse, interacting with soil and plants. She is wearing a blue and beige striped long-sleeved top and black pants. Her hair is long and dark, falling over her shoulders. She appears to be gardening, as she kneels beside a row of soil or mud, with small green plants beginning to grow. In the background is a translucent plastic sheet covering the entire space, which is tied to metal poles and supported by wooden stakes at the four corners. Through the plastic sheet, you can vaguely see the outline of an outdoor field, suggesting that this may be a contained or controlled environment for growing plants. The lighting indicates that it may be daytime, and the overall scene conveys a sense of calmness and concentration as the woman works in the greenhouse. The camera is shot on the spot and then turned to the woman. There is no change in perspective and the picture is shaky.</GPROMPT>""",
        """<GPROMPT>The video shows a woman in a greenhouse, interacting with soil and plants. She is wearing a blue and beige striped long-sleeved top and black pants. Her hair is long and dark, falling over her shoulders. She appears to be gardening, as she kneels beside a row of soil or mud, with small green plants beginning to grow. In the background is a translucent plastic sheet covering the entire space, which is tied to metal poles and supported by wooden stakes at the four corners. Through the plastic sheet, you can vaguely see the outline of an outdoor field, suggesting that this may be a contained or controlled environment for growing plants. The lighting indicates that it may be daytime, and the overall scene conveys a sense of calmness and concentration as the woman works in the greenhouse. The camera is shot on the spot and then turned to the woman. There is no change in perspective and the picture is shaky.</GPROMPT>""",

        """<GPROMPT>In the video, a bald man stands indoors, wearing a dark jacket over a button-up shirt. The scene is set against a warm, ambient background featuring horizontal wooden paneling on the walls and a modern red lamp with a round lampshade that emits a soft glow. The lighting is warm, creating a cozy atmosphere. The camera focuses steadily on the man, with no noticeable changes in perspective or movement, emphasizing his presence in the intimate setting.</GPROMPT>""",
        """<GPROMPT>In the video, a bald man stands indoors, wearing a dark jacket over a button-up shirt. The scene is set against a warm, ambient background featuring horizontal wooden paneling on the walls and a modern red lamp with a round lampshade that emits a soft glow. The lighting is warm, creating a cozy atmosphere. The camera focuses steadily on the man, with no noticeable changes in perspective or movement, emphasizing his presence in the intimate setting.</GPROMPT>""",

        """<GPROMPT>The video depicts an indoor scene set in a formal or semi-formal environment, such as a restaurant or private club, where warm lighting creates a cozy and intimate atmosphere. A man in a gray suit jacket and patterned tie faces a woman with long curly hair wearing a black outfit. In the background, other people are engaged in conversations or activities, and a chandelier and framed artwork enhance the elegant atmosphere. The camera remains fixed, capturing the interaction between the two people without any changes in perspective.</GPROMPT>""",
        """<GPROMPT>The video depicts an indoor scene set in a formal or semi-formal environment, such as a restaurant or private club, where warm lighting creates a cozy and intimate atmosphere. A man in a gray suit jacket and patterned tie faces a woman with long curly hair wearing a black outfit. In the background, other people are engaged in conversations or activities, and a chandelier and framed artwork enhance the elegant atmosphere. The camera remains fixed, capturing the interaction between the two people without any changes in perspective.</GPROMPT>""",

        """<GPROMPT>The scene takes place in a dimly lit office or interrogation room with brick walls and a grid-patterned window. A middle-aged man with graying hair, wearing a light gray suit jacket, white shirt, and yellow checkered tie, sits on the left side of the frame. Next to him sits a woman with long dark hair, wearing a dark gray suit jacket, with her arms crossed over her chest, looking intently at someone off-screen. The lighting casts shadows on the faces of the characters, enhancing the serious atmosphere of the scene. The formal setting and focused interactions suggest that this is an important discussion, perhaps related to business or law enforcement matters. The camera is fixed in position, with no noticeable changes in perspective or shaking.</GPROMPT>""",
        """<GPROMPT>The scene takes place in a dimly lit office or interrogation room with brick walls and a grid-patterned window. A middle-aged man with graying hair, wearing a light gray suit jacket, white shirt, and yellow checkered tie, sits on the left side of the frame. Next to him sits a woman with long dark hair, wearing a dark gray suit jacket, with her arms crossed over her chest, looking intently at someone off-screen. The lighting casts shadows on the faces of the characters, enhancing the serious atmosphere of the scene. The formal setting and focused interactions suggest that this is an important discussion, perhaps related to business or law enforcement matters. The camera is fixed in position, with no noticeable changes in perspective or shaking.</GPROMPT>""",

        """<GPROMPT>The video records a serene outdoor exchange between two people in a forest setting. The frame focuses on the dark-haired woman facing the left side of the frame, engaging in a conversation with the blond woman in the foreground, who is slightly out of focus. The dark-haired woman has her hair tied up and wears a black sweatshirt with blue trim. The blond woman, with her hair braided, wears a brown top. The background is softly blurred, highlighting the lush greenery of the forest, with trees and leaves creating a natural and peaceful atmosphere. The lighting is soft and diffuse, probably filtered through the tree canopy, casting gentle shadows and adding to the tranquil mood. The camera is centered in the frame with slight shaking.</GPROMPT>""",
        """<GPROMPT>The video records a serene outdoor exchange between two people in a forest setting. The frame focuses on the dark-haired woman facing the left side of the frame, engaging in a conversation with the blond woman in the foreground, who is slightly out of focus. The dark-haired woman has her hair tied up and wears a black sweatshirt with blue trim. The blond woman, with her hair braided, wears a brown top. The background is softly blurred, highlighting the lush greenery of the forest, with trees and leaves creating a natural and peaceful atmosphere. The lighting is soft and diffuse, probably filtered through the tree canopy, casting gentle shadows and adding to the tranquil mood. The camera is centered in the frame with slight shaking.</GPROMPT>""",

        """<GPROMPT>The video features a close-up of a woman with shoulder-length hair, wearing a dark blazer over a light-colored top, sitting in an indoor setting. Behind her, a colorful abstract painting hangs on a wooden wall, adding an artistic touch to the scene. The lighting in the room is stable, highlighting the details of her clothing and facial features without casting harsh shadows. The overall atmosphere seems formal or professional, perhaps a meeting or conversation, as she sits at a table. The camera perspective does not change, and the image is shaky.</GPROMPT>""",
        """<GPROMPT>The video features a close-up of a woman with shoulder-length hair, wearing a dark blazer over a light-colored top, sitting in an indoor setting. Behind her, a colorful abstract painting hangs on a wooden wall, adding an artistic touch to the scene. The lighting in the room is stable, highlighting the details of her clothing and facial features without casting harsh shadows. The overall atmosphere seems formal or professional, perhaps a meeting or conversation, as she sits at a table. The camera perspective does not change, and the image is shaky.</GPROMPT>""",

        """<GPROMPT>The video features a close-up of a woman with shoulder-length light brown hair, wearing a light-colored shirt with a subtle pattern. She is in an interior setting with warm lighting that creates a cozy atmosphere. A blurred figure appears on the right side of the frame, suggesting interaction. In the background, there are wooden shelves filled with various items, reinforcing the setting's homely and casual ambiance. The lighting remains constant, highlighting the woman's facial features and creating a soft glow on her face. The scene conveys a moment of communication between the woman and another person in a small, intimate environment.</GPROMPT>""",
        """<GPROMPT>The video features a close-up of a woman with shoulder-length light brown hair, wearing a light-colored shirt with a subtle pattern. She is in an interior setting with warm lighting that creates a cozy atmosphere. A blurred figure appears on the right side of the frame, suggesting interaction. In the background, there are wooden shelves filled with various items, reinforcing the setting's homely and casual ambiance. The lighting remains constant, highlighting the woman's facial features and creating a soft glow on her face. The scene conveys a moment of communication between the woman and another person in a small, intimate environment.</GPROMPT>""",
    ]

    # === 关键改动：组合成完整 caption 列表 ===
    captions = [gp + lp for gp, lp in zip(global_prompts, local_prompts)]
    # 如果你已经直接准备好完整 caption，也可以直接：
    # captions = your_full_caption_list

    out_path = f'infer_out/tts/prompttts/{Path(dit_exp_name).stem}'
    os.makedirs(out_path, exist_ok=True)

    # === 关键改动：改为传 captions=，并移除 global_prompt/local_prompt ===
    outputs = infer_ins.forward(
        resource_context,
        texts,
        captions=captions,                      # <== 仅传完整 caption
        prompttts_kwargs={'ref_dur_only': True},
        **infer_kwargs
    )

    # 逐条保存
    for i, (txt, out) in enumerate(zip(texts, outputs), start=1):
        input_text_norm = SSML(txt).text_str
        save_wav_bytes(out.wav_bytes, f'{out_path}/{i:02d}_{input_text_norm[:20]}.wav')
