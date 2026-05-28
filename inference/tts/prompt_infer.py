import os
import random
import re
import tempfile
from datetime import datetime, timedelta
import collections
import collections.abc
from glob import glob
import math
import numpy as np
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))
from attrdict import AttrDict
from typing import Optional, Dict
import socket
from contextlib import closing
import yaml
from argparse import ArgumentParser
import torch.distributed as dist
import torch
import soundfile as sf
import librosa
from multiprocessing import Process, set_start_method
from utils.commons.os_utils import kill_void
from utils.commons.ckpt_utils import load_ckpt, get_last_checkpoint, torch_load_dist
from utils.commons.hparams import set_hparams, hparams
from modules.asr.sensevoice.sensevoice_api import build_asr_model, run_asr_model
from modules.tts.scriptspeech.build_model_utils import DiTBuildModelMixin, SemanticLMBuildModelMixin, build_vae
from utils.commons.upload_tos_utils import send_file_to_tos
import json
from tasks.tts.task_utils.prompttts_task_utils import build_audio_mask_from_ids
from utils.text.split_text import get_word_list, remove_spaces_between_chinese
from utils.text import is_chinese
from utils.text.cosyvoice2_tokenizer import CosyVoice2Tokenizer

def _is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        # 不用 SO_REUSEADDR，避免“看起来可用但其实被占用”的假象
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False

def find_available_port(base_port: int = 10521, host: str = "127.0.0.1", max_search: int = 2000) -> int:
    """
    按 base, base+1, base-1, base+2, base-2 ... 搜索可用端口。
    max_search 表示最多尝试多少个偏移步（2000 => 最大探测到 base±2000）
    """
    if base_port < 1 or base_port > 65535:
        raise ValueError(f"Invalid base_port={base_port}")

    # offsets: 0, +1, -1, +2, -2, ...
    for k in range(0, max_search + 1):
        if k == 0:
            candidates = [base_port]
        else:
            candidates = []
            p1 = base_port + k
            p2 = base_port - k
            if 1 <= p1 <= 65535:
                candidates.append(p1)
            if 1 <= p2 <= 65535:
                candidates.append(p2)

        for p in candidates:
            if _is_port_free(p, host=host):
                return p

    raise RuntimeError(f"No free port found around {base_port} within ±{max_search}")

def _fmt_mb(x: int) -> str:
    return f"{x / 1024**2:,.1f} MB"

def _cuda_snapshot(device=None):
    if not torch.cuda.is_available():
        return None
    if device is None:
        device = torch.cuda.current_device()
    dev = torch.device(device)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    torch.cuda.synchronize(idx)
    free, total = torch.cuda.mem_get_info(idx)  # driver 视角 free/total
    alloc = torch.cuda.memory_allocated(idx)    # pytorch 实际分配
    reserved = torch.cuda.memory_reserved(idx)  # pytorch 缓存池占用
    max_alloc = torch.cuda.max_memory_allocated(idx)
    max_reserved = torch.cuda.max_memory_reserved(idx)
    return {
        "idx": idx,
        "free": free, "total": total,
        "alloc": alloc, "reserved": reserved,
        "max_alloc": max_alloc, "max_reserved": max_reserved,
    }

def log_cuda_mem(tag: str, device=None, prev=None):
    snap = _cuda_snapshot(device)
    if snap is None:
        print(f"[MEM][CPU] {tag} (cuda not available)")
        return snap

    line = (
        f"[MEM][CUDA:{snap['idx']}] {tag} | "
        f"free {_fmt_mb(snap['free'])} / total {_fmt_mb(snap['total'])} | "
        f"alloc {_fmt_mb(snap['alloc'])} | reserved {_fmt_mb(snap['reserved'])} | "
        f"max_alloc {_fmt_mb(snap['max_alloc'])} | max_reserved {_fmt_mb(snap['max_reserved'])}"
    )
    if prev is not None:
        da = snap["alloc"] - prev["alloc"]
        dr = snap["reserved"] - prev["reserved"]
        df = snap["free"] - prev["free"]
        line += f" | Δalloc {_fmt_mb(da)} Δreserved {_fmt_mb(dr)} Δfree {_fmt_mb(df)}"
    print(line)
    return snap

def module_footprint_bytes(m: torch.nn.Module):
    # 参数 + buffer 的字节数（按当前 dtype 计算）
    p_bytes = 0
    b_bytes = 0
    for p in m.parameters(recurse=True):
        p_bytes += p.numel() * p.element_size()
    for b in m.buffers(recurse=True):
        b_bytes += b.numel() * b.element_size()
    n_params = sum(p.numel() for p in m.parameters(recurse=True))
    dtype = None
    try:
        dtype = next(m.parameters()).dtype
    except StopIteration:
        dtype = None
    return p_bytes, b_bytes, n_params, dtype

def print_module_mem(name: str, m: torch.nn.Module):
    p_bytes, b_bytes, n_params, dtype = module_footprint_bytes(m)
    print(
        f"[MODULE] {name}: params={n_params/1e6:.2f}M "
        f"dtype={dtype} param={_fmt_mb(p_bytes)} buffer={_fmt_mb(b_bytes)} total={_fmt_mb(p_bytes+b_bytes)}"
    )


cfg_weight = None
infer_step = 100
extend_dur = 0
vad_len = 0

# ===== target 时长自动估计 =====
_AUDIO_TAG_RE = re.compile(r"<\s*Audio\s*>", flags=re.IGNORECASE)
_ANY_TAG_RE = re.compile(r"<[^>]+>")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")

def _count_speech_units_for_len(s: Optional[str]) -> int:
    """
    用于估时长的“单位数”：
    - 含中文：中文按“字”计数；英文/数字按“词”计数并加到一起（粗略但稳定）
    - 纯英文：按词计数
    会去掉 <S1>...</S1> / <S2>... 等标签以及其他 <...> 标签
    """
    if not isinstance(s, str):
        return 0
    s = s.strip()
    if not s:
        return 0

    # 去掉所有 <...> 标签（含 <S1> / </S1> / <Audio> 等）
    s = _ANY_TAG_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return 0

    cjk = len(_CJK_RE.findall(s))
    if cjk > 0:
        en_words = len(_WORD_RE.findall(s))
        return int(cjk + en_words)
    else:
        return int(len(_WORD_RE.findall(s)))

def _estimate_target_infer_length(
    target_text: str,
    caption: Optional[str],
    ref_audio: Optional[np.ndarray],
    ref_text: Optional[str],
    sr: int = 24000,
    default_cpm: float = 300.0, 
) -> float:
    """
    返回“新生成段”的时长（秒），不含 ref 段。
    规则：
    - 有 target_text：若有 ref，则用 ref 语速替代
    - caption 有 <Audio>：额外 +2s
    - 无 text 且有 ref_audio：5s
    """

    tgt_units = _count_speech_units_for_len(target_text)

    # 没有 text：如果有 audio(ref) => 5s；否则也给个 5s 兜底（防止生成 0 帧）
    if tgt_units <= 0:
        base = 5.0
        return float(base)

    extra = 2.0 if (isinstance(caption, str) and _AUDIO_TAG_RE.search(caption)) else 0.0

    # 有 ref：优先用 ref 语速
    if ref_audio is not None and isinstance(ref_audio, np.ndarray) and ref_audio.size > 0:
        ref_units = _count_speech_units_for_len(ref_text)
        ref_dur = float(ref_audio.shape[0]) / float(sr)
        if ref_units > 0 and ref_dur > 1e-3:
            units_per_sec = ref_units / ref_dur
            base = tgt_units / max(units_per_sec, 1e-6)
        else:
            base = (tgt_units / default_cpm) * 60.0
    else:
        base = (tgt_units / default_cpm) * 60.0

    # 稳定性保护：太短/太长都裁一下
    final = base + extra
    final = float(np.clip(final, 5.0, 120.0))
    return final


def merge_model_weights(model, new_ckpt_path, ignore_module=[], weight=0.5):
    """
    Args:
        model: 已经加载了原始权重的 torch.nn.Module
        new_ckpt_path: 新的 checkpoint 文件路径 (state_dict)
        weight: 融合比例，merged = weight * old + (1 - weight) * new
    """
    # 加载新的 ckpt
    if os.path.isfile(new_ckpt_path):
        base_dir = os.path.dirname(new_ckpt_path)
        ckpt_path = new_ckpt_path
        new_state_dict = torch_load_dist(new_ckpt_path, map_location='cpu', mmap=None)
    else:
        base_dir = new_ckpt_path
        new_state_dict, ckpt_path = get_last_checkpoint(new_ckpt_path)
    # new_state_dict = torch.load(new_ckpt_path, map_location="cpu")
    print(f'merge model from {ckpt_path} with weight {1 - weight}')
    # 拿到旧模型参数
    old_state_dict = model.state_dict()

    merged_state_dict = {}
    new_state_dict = new_state_dict['dit']
    for k, old_param in old_state_dict.items():
        if k in new_state_dict and old_param.shape == new_state_dict[k].shape and not any(ign in k for ign in ignore_module):
            new_param = new_state_dict[k]
            merged_state_dict[k] = weight * old_param + (1 - weight) * new_param
        else:
            # 如果没有对应权重，就保留旧的
            merged_state_dict[k] = old_param

    # 加载融合后的权重
    model.load_state_dict(merged_state_dict)

    return model

def gen_audio_html(infos: Dict[int, dict], output_fp: Optional[str] = None,
                   title_name=None, extra_desc=None):
    if output_fp is None:
        output_fp = tempfile.NamedTemporaryFile(suffix=".html", delete=False).name

    num_per_row = 5
    total = len(infos)
    rows = (total + num_per_row - 1) // num_per_row

    with open(output_fp, 'w') as f:
        print('<html lang="en">', file=f)
        print('<head>', file=f)
        print('<meta charset="UTF-8">', file=f)
        print('<meta name="viewport" content="width=device-width, initial-scale=1.0">', file=f)
        if title_name is not None:
            print(f'<title>{title_name}</title>', file=f)
        else:
            print('<title>Audio Samples</title>', file=f)

        print('<style>', file=f)
        print(r'''
            body { margin: 0; padding: 20px; font-family: Arial, sans-serif; }
            .container { max-width: 1280px; margin: 0 auto; }
            table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
            h1 { font-size: 2em; margin-bottom: 0.2em; }
            p.description { color: #666; margin-bottom: 1.5em; }
            td { padding: 10px; border: 2px solid DodgerBlue; vertical-align: top; text-align: center; }
            audio { width: 100%; }
            .audio-block { margin-bottom: 10px; text-align: left; }
            .audio-title { font-size: 13px; color: #333; margin: 0 0 4px 0; }
            .desc { margin-top: 10px; white-space: pre-wrap; text-align: left; font-size: 14px; background: #f8f8f8; padding: 8px; border-radius: 5px; }
        ''', file=f)
        print('</style>', file=f)
        print('</head>', file=f)
        print('<body>', file=f)
        if title_name is not None:
            print(f'  <h1>{title_name}</h1>', file=f)
        if extra_desc is not None:
            print(f'  <p class="description">{extra_desc}</p>', file=f)
        print('<div class="container">', file=f)
        print('<table>', file=f)

        keys = list(infos.keys())

        for row in range(rows):
            print('<tr>', file=f)
            for col in range(num_per_row):
                j = row * num_per_row + col
                if j >= total:
                    print('<td></td>', file=f)
                    continue

                k = keys[j]
                info = infos[k]

                # 生成音频（必选）
                tos_url = info.get('tos_url', '')

                # prompt 音频（可选）
                prompt_tos_url = info.get('prompt_tos_url', '')

                # caption（兜底）
                caption = info.get('caption', '')
                if caption is None:
                    caption = ''
                caption = str(caption)

                print('<td>', file=f)

                # Prompt audio
                if prompt_tos_url:
                    print('<div class="audio-block">', file=f)
                    print('<div class="audio-title">Prompt</div>', file=f)
                    print(f'<audio controls preload="none">', file=f)
                    print(f'  <source src="{prompt_tos_url}" type="audio/wav">', file=f)
                    print('  Your browser does not support the audio element.', file=f)
                    print('</audio>', file=f)
                    print('</div>', file=f)

                # Output audio
                print('<div class="audio-block">', file=f)
                print('<div class="audio-title">Output</div>', file=f)
                print(f'<audio controls preload="none">', file=f)
                print(f'  <source src="{tos_url}" type="audio/wav">', file=f)
                print('  Your browser does not support the audio element.', file=f)
                print('</audio>', file=f)
                print('</div>', file=f)

                print('<div class="desc">', file=f)
                safe_caption = caption.replace("<", "&lt;").replace(">", "&gt;")
                print(f'caption: {safe_caption}\n', file=f)
                print('</div>', file=f)

                print('</td>', file=f)
            print('</tr>', file=f)

        print('</table>', file=f)
        print('</div>', file=f)
        print('</body>', file=f)
        print('</html>', file=f)

    return output_fp


def upload_tos_html(yml=None, out_path=None, title_name=None, extra_desc=None):
    sub_dir = 'prompttts'

    from collections import defaultdict
    infos = defaultdict(dict)

    with open(yml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        samples = cfg['samples']

    for idx, sample in enumerate(samples):
        wav = os.path.join(out_path, f"out_{idx}.wav")
        if not os.path.exists(wav):
            continue

        # 1) 上传 output wav
        cluster = os.environ.get('CLUSTER', '').lower()
        if cluster == 'va':
            bucket='sa-ag-sg-research-sg'
        else:
            bucket='humanaigc-ads'
        tos_url = send_file_to_tos(wav, sub_dir=sub_dir, bucket=bucket)
        print("out tos_url: ", tos_url)
        infos[idx]['tos_url'] = tos_url
        infos[idx].update(sample)

        # 2) 如果有 prompt_audio，也上传
        prompt_audio = sample.get('prompt_audio', None)
        if prompt_audio:
            try:
                if os.path.exists(prompt_audio):
                    prompt_tos_url = send_file_to_tos(prompt_audio, sub_dir=sub_dir, bucket=bucket)
                    print("prompt tos_url: ", prompt_tos_url)
                    infos[idx]['prompt_tos_url'] = prompt_tos_url
                else:
                    print(f"[WARN] prompt_audio not found: {prompt_audio}")
            except Exception as e:
                print(f"[WARN] upload prompt_audio failed: {prompt_audio}, err={e}")

        # 3) 如果没有 caption，但有 global/local，则自动拼一个 caption
        if 'caption' not in infos[idx]:
            g = infos[idx].get('global', '')
            l = infos[idx].get('local', '')
            parts = []
            if isinstance(g, str) and g.strip():
                parts.append(f"Global: {g}")
            if isinstance(l, str) and l.strip():
                parts.append(f"Local: {l}")
            infos[idx]['caption'] = "\n".join(parts) if parts else ""

    html_path = gen_audio_html(infos, title_name=title_name, extra_desc=extra_desc)
    print(f"生成的HTML文件路径：{html_path}")
    html_tos = send_file_to_tos(html_path, sub_dir=sub_dir,bucket=bucket)
    print(f"生成的HTML TOS：{html_tos}")
    return html_tos
    
class ScriptSpeechInfer(DiTBuildModelMixin, SemanticLMBuildModelMixin):
    def __init__(self, device, dit_ckpt,
                 vae_ckpt=None,
                 merge_ckpt=None, merge_weight=0.5,
                 use_sa_front: bool = False,
                 g2p_model: str = 'qwen'):
        self.device = device
        self.precision = torch.bfloat16
        self.build_model(
            dit_ckpt,
            vae_ckpt=vae_ckpt,
            merge_ckpt=merge_ckpt,
            merge_weight=merge_weight,
        )


    def _tokenize_dit_text(self, text: str):
        """
        兼容 HuggingFace tokenizer 和 CosyVoice2Tokenizer：
        - HF: BatchEncoding 支持 .to(self.device)
        - Cosy: 返回 dict，没有 .to 方法，需要手动搬 tensor
        """

        if isinstance(self.dit_text_tokenizer, CosyVoice2Tokenizer):
            # CosyVoice2Tokenizer 分支（参考你给的 process_text_seg）
            text_inputs = self.dit_text_tokenizer(
                text,
                padding=True,
                return_tensors='pt',
            )
            txt_tokens = text_inputs['input_ids'].to(self.device)
            txt_mask = text_inputs['attention_mask'].bool().to(self.device)
        else:
            # 原来的 Qwen / HF 分支
            text_inputs = self.dit_text_tokenizer(
                text,
                padding=True,
                return_tensors='pt',
            ).to(self.device)
            txt_tokens = text_inputs['input_ids'].clone()
            txt_mask = text_inputs['attention_mask'].bool()

        # 把 padding 位置替换成 cfg 用的 mask token
        txt_tokens[~txt_mask] = self.cfg_mask_text_token
        txt_lens = txt_mask.long().sum(-1)

        # ====== spk_mask: [B, T], 0/1/2/3/4 ======
        B, T = txt_tokens.shape
        masks = []
        for b in range(B):
            L = int(txt_lens[b].item())
            # 只在有效 token 范围内找 tag，避免 pad 区域误匹配
            m = build_spk_mask_from_text_tokens(txt_tokens[b, :L].detach().cpu(), self._sx_patterns)
            if L < T:
                m = torch.cat([m, torch.zeros((T - L,), dtype=torch.long)], dim=0)
            masks.append(m)
        spk_mask = torch.stack(masks, dim=0).to(self.device)

        return txt_tokens, txt_mask, txt_lens, spk_mask

    def build_model(self, dit_ckpt,
                    vae_ckpt=None,
                    merge_ckpt=None, merge_weight=0.5,):
        self.asr_model = None

        # 建议：每次 build 前清一下峰值统计，方便看 max_alloc
        if torch.cuda.is_available():
            idx = torch.device(self.device).index if torch.device(self.device).index is not None else torch.cuda.current_device()
            torch.cuda.reset_peak_memory_stats(idx)

        snap = log_cuda_mem("enter build_model()", self.device)

        # ====== hparams & config ======
        set_hparams(config=os.path.join(dit_ckpt, 'config.yaml'),
                    print_hparams=False, global_hparams=True)
        hparams["exp_name"] = 'infer'
        self.config = AttrDict(hparams)

        snap = log_cuda_mem("after set_hparams()", self.device, prev=snap)

        # ====== VAE & audio tokenizer ======
        vae_ckpt_path = vae_ckpt or hparams.get('vae_ckpt')
        self.vae, self.hp_vae = build_vae(vae_ckpt_path)
        print_module_mem("VAE (on CPU)", self.vae)
        snap = log_cuda_mem("after build_vae() (still CPU)", self.device, prev=snap)

        # 搬到 GPU
        self.vae.to(self.device)
        print_module_mem("VAE (on GPU)", self.vae)
        snap = log_cuda_mem("after vae.to(device)", self.device, prev=snap)

        # ====== DiT & 文本 tokenizer ======
        self.dit_text_tokenizer, self.dit_vocab_size = self.build_dit_text_tokenizer()
        self._sx_patterns = _get_sx_token_patterns(self.dit_text_tokenizer)
        snap = log_cuda_mem("after build_dit_text_tokenizer()", self.device, prev=snap)

        self.dit = self.build_dit(hparams)
        print_module_mem("DiT (on CPU, before load_ckpt)", self.dit)
        snap = log_cuda_mem("after build_dit() (still CPU)", self.device, prev=snap)

        load_ckpt(self.dit, dit_ckpt, 'dit', strict=False)
        print_module_mem("DiT (on CPU, after load_ckpt)", self.dit)
        snap = log_cuda_mem("after load_ckpt(dit) (still CPU)", self.device, prev=snap)

        if merge_ckpt is not None:
            self.dit = merge_model_weights(self.dit, merge_ckpt,
                                        ignore_module=['cross', 'caption_proj'],
                                        weight=merge_weight)
            print_module_mem("DiT (on CPU, after merge)", self.dit)
            snap = log_cuda_mem("after merge_model_weights() (still CPU)", self.device, prev=snap)

        self.vae.eval(); self.vae.to(self.device, dtype=self.precision)
        self.dit.eval(); self.dit.to(self.device, dtype=self.precision)
        print_module_mem("DiT (on GPU)", self.dit)
        snap = log_cuda_mem("after dit.to(device)", self.device, prev=snap)

        # ====== caption 相关 encoder ======
        self.use_caption = hparams.get('use_caption', False)
        print(f"[INFO] use_caption={self.use_caption}, model_size={hparams.get('model_size', 'base')}")

        if self.use_caption:
            if hparams.get('model_size', 'base') == 'seedance_7b':
                self.build_sd_text_encoder(hparams['text'])
                print_module_mem("sd_text_encoder (CPU)", self.sd_text_encoder)
                snap = log_cuda_mem("after build_sd_text_encoder() (CPU)", self.device, prev=snap)

                self.sd_text_encoder.eval()
                self.sd_text_encoder.to(self.device, dtype=self.precision)
                print_module_mem("sd_text_encoder (GPU)", self.sd_text_encoder)
                snap = log_cuda_mem("after sd_text_encoder.to(device)", self.device, prev=snap)

            elif 'goku' in hparams.get('model_size', 'base'):
                self.build_goku_text_encoder(hparams)
                print_module_mem("goku_text_encoder (CPU)", self.goku_text_encoder)
                snap = log_cuda_mem("after build_goku_text_encoder() (CPU)", self.device, prev=snap)

                self.goku_text_encoder.eval()
                self.goku_text_encoder.to(self.device, dtype=self.precision)
                print_module_mem("goku_text_encoder (GPU)", self.goku_text_encoder)
                snap = log_cuda_mem("after goku_text_encoder.to(device)", self.device, prev=snap)
        else:
            self.sd_text_encoder = None
            self.goku_text_encoder = None

        log_cuda_mem("leave build_model()", self.device, prev=snap)


    def run_goku_text_encoder(self, captions: list):
        inputs = self.goku_tokenizer(
            captions,
            padding=True,  # Dynamic / longest
            truncation=True,
            max_length=hparams['text_max_token_length'],
            return_tensors="pt",
        )

        input_ids, attention_masks = inputs.input_ids.cuda(), inputs.attention_mask.cuda()
        encoder_hidden_states = self.goku_text_encoder(
            input_ids,
            return_dict=False,
            attention_mask=attention_masks,
        )[0]  # [B, T, C]

        # 新的多类掩码（0/1/2）
        if hparams.get('use_caption_text_mark', False):
            caption_text_mark = build_audio_mask_from_ids(
                input_ids=input_ids,
                attention_mask=attention_masks,
                tokenizer=self.goku_tokenizer,
            )  
        else:
            caption_text_mark = None

        return encoder_hidden_states, caption_text_mark, attention_masks

    @torch.no_grad()
    def forward(self,
                text,
                ref_audio=None,
                ref_text=None,
                prompt=None,          # 推理时唯一的 caption（语义上：有 ref 时是 local，无 ref 时是 full）
                cfg_w=None,
                negative_prompt=None, # 仍然保留接口，但实际不用
                infer_length=5.0,
                start_time=0.2,
                end_time=0.2,
                num_step=100):
        """
        infer_length 语义：
          - 无 ref_audio：infer_length = 纯生成时长（秒）
          - 有 ref_audio：infer_length = 目标“新生成”时长（秒），
            内部会自动做 tgt_len = ref_lat_len + gen_lat_len

        task 类型（只用于你脑子里的模式区分，不再喂进模型）：
          task1: 有 ref_audio（ctx wav） + caption（此时 caption 语义更偏 local/情感）
          task2: 无 ref_audio，仅 caption 控制内容/情感
        """
        speech = len(text) > 0
        fm_wav = hparams['frames_multiple'] * hparams['hop_size']

        # ====== 参考音频 & 参考文本 ======
        use_ref = ref_audio is not None
        if ref_audio is not None:
            # 有 prompt audio ⇒ 认为是 task1
            # 例如最多用 120 秒 prompt
            max_prompt_sec = 15.0
            if ref_audio is not None:
                max_len = int(max_prompt_sec * 24000)
                ref_audio = ref_audio[:max_len]

            ref_wav = torch.from_numpy(
                np.concatenate([ref_audio, np.zeros(0, dtype=np.float16)])
            )[None, :].to(self.device)

            ref_wav_lens = torch.LongTensor(
                [ref_wav.shape[1] // fm_wav * fm_wav]
            ).to(self.device)
            ref_wav = ref_wav[:, :ref_wav_lens[0]]

            # 自动 ASR 得到 ref_text（跟原来一样）
            if ref_text is None:
                self.asr_model = build_asr_model(self.device) 
                wav_16k = librosa.resample(
                    ref_audio.astype(np.float32),
                    orig_sr=24000,
                    target_sr=16000
                )
                asr_out = run_asr_model([wav_16k], self.asr_model, with_segments=False)
                if isinstance(asr_out, (list, tuple)) and len(asr_out) > 0:
                    asr_item = asr_out[0] or {}
                else:
                    asr_item = asr_out or {}

                print('asr_result', asr_item)
                ref_text = asr_item.get('text_normed') or asr_item.get('text') or ''
                if ref_text and not ref_text.endswith(('.', '。', '!', '！', '?', '？')):
                    ref_text = ref_text + '.'

            task_type = 1
        else:
            # 无 prompt audio ⇒ task2
            ref_text = None
            ref_wav = None
            task_type = 2

        print(f'| task_type: {task_type}, CFG: {cfg_w}')

        infer_length = _estimate_target_infer_length(
            target_text=text,
            caption=prompt,
            ref_audio=ref_audio,
            ref_text=ref_text,
            sr=24000,
            default_cpm=280.0,
        )

        print(f'| task_type: {task_type}, CFG: {cfg_w}, target_infer_length: {infer_length:.2f}s')

        # ====== 计算新生成的 latent 长度 ======
        gen_lat_len = int(
            float(infer_length) * 24000 / self.hp_vae['hop_size'] / self.hp_vae['vae_stride']
        )
        gen_lat_len = max(gen_lat_len, 0)

        # ====== 准备唯一的 caption 字符串 ======
        caption = prompt

        use_caption = getattr(self, "use_caption", False) and (caption is not None)
        caption_emb = None
        caption_lens = None
        caption_text_mark = None

        import torch.nn.functional as F

        if use_caption and 'goku' in self.config.model_size:
            if negative_prompt is None:
                negative_prompt = '<I>distorted audio</I><I>background static</I>...'

            # 一次性编码，动态 padding 会对齐到同一 T
            all_embs, all_mark, all_att = self.run_goku_text_encoder([caption, negative_prompt])
            # all_embs: [2, T, C], all_att: [2, T]

            pos_text_embs = all_embs[0:1] * all_att[0:1][..., None]
            neg_text_embs = all_embs[1:2] * all_att[1:2][..., None]
            pos_lens = all_att[0:1].sum(-1)
            neg_lens = all_att[1:2].sum(-1)

            caption_emb = torch.cat([
                pos_text_embs,
                neg_text_embs,
                neg_text_embs,
            ], dim=0)

            caption_lens = torch.cat([
                pos_lens,
                neg_lens,
                neg_lens,
            ], dim=0).long()

            if all_mark is not None:
                pos_mark = all_mark[0:1]
                neg_mark = all_mark[1:2]
                caption_text_mark = torch.cat([
                    pos_mark,
                    neg_mark,
                    neg_mark,
                ], dim=0)

        # ====== 推断 latent_dim ======
        latent_dim = None
        if hasattr(self.vae, "latent_dim"):
            latent_dim = getattr(self.vae, "latent_dim")
        if latent_dim is None and hasattr(self.dit, "hp") and hasattr(self.dit.hp, "in_channels"):
            latent_dim = self.dit.hp.in_channels
        if latent_dim is None:
            latent_dim = 32  # 兜底

        # ====== VAE 编码参考音频，构造 lat_ctx / ctx_mask ======
        if ref_wav is not None:
            with torch.inference_mode():
                with torch.autocast(device_type='cuda', dtype=self.precision):
                    lat_ctx_ref = self.vae.encode_latent(ref_wav)  # [1, L_ref, C]
                latent_dim = lat_ctx_ref.size(-1)

                ref_lat_len = lat_ctx_ref.size(1)
                tgt_len = ref_lat_len + gen_lat_len

                ctx_mask_ref = torch.ones_like(lat_ctx_ref[:, :, 0:1])

                lat = torch.nn.functional.pad(
                    lat_ctx_ref, (0, 0, 0, tgt_len - ref_lat_len),
                    mode='constant', value=0
                )
                ctx_mask = torch.nn.functional.pad(
                    ctx_mask_ref, (0, 0, 0, tgt_len - ref_lat_len),
                    mode='constant', value=0
                )
                if ref_text is not None:
                    full_text = ref_text + text
                else:
                    full_text = text
        else:
            tgt_len = gen_lat_len
            lat_ctx_ref = torch.zeros(1, 0, latent_dim).to(self.device)
            lat = torch.zeros(1, tgt_len, latent_dim).to(self.device)
            ctx_mask = torch.zeros(1, tgt_len, 1).to(self.device)
            full_text = text

        if not self.use_caption:
            full_text = strip_s1s2_tags(full_text)

        # 文本 token
        txt_tokens, txt_mask, txt_lens, spk_mask = self._tokenize_dit_text(full_text)

        # ====== VAD mask（只当作条件传进去）======
        vad_mask = torch.zeros_like(lat[:, :, :1])
        if not self.config.get('drop_vad', False) and speech:
            vad_mask[:, int(start_time * 25):-int(end_time * 25)] = 1.0

        # 3 路 CFG：在 batch 维度上复制 VAD
        vad_mask = torch.cat([vad_mask] * 3, dim=0)

        # ====== 文本 CFG======
        txt_tokens = torch.cat([
            txt_tokens,
            txt_tokens,
            torch.full_like(txt_tokens, self.cfg_mask_text_token),
        ], dim=0)
        txt_mask = torch.cat([txt_mask] * 3, dim=0)
        txt_lens = torch.cat([txt_lens] * 3, dim=0)

        if spk_mask is not None:
            spk_mask = torch.cat([
                spk_mask,                          # 路1：真实文本
                spk_mask,                          # 路2：真实文本
                torch.zeros_like(spk_mask),        # 路3：文本全 mask
            ], dim=0)

        # ====== latent / ctx_mask 也复制 3 路 ======
        lat = torch.cat([
            lat,
            torch.zeros_like(lat),
            torch.zeros_like(lat),
        ], dim=0)
        ctx_mask = torch.cat([ctx_mask] * 3, dim=0)

        batch_size = lat.shape[0]

        # ====== 组装 Diffusion.inference 所需 inputs ======
        inputs = {
            'txt_tokens': txt_tokens,
            'spk_mask': spk_mask,
            'txt_mask': txt_mask,
            'txt_lens': txt_lens,
            'ctx_mask': ctx_mask,
            'lat_ctx': lat,
            'caption_emb': caption_emb,
            'caption_lens': caption_lens,
            'caption_text_mark': caption_text_mark,
            'vad_mask': vad_mask,
            'tgt_len': torch.full(
                (batch_size,),
                tgt_len,
                dtype=torch.long,
                device=self.device,
            ),
        }

        global cfg_weight, infer_step, extend_dur, vad_len
        cfg_weight, infer_step, extend_dur, vad_len = cfg_w, num_step, infer_length, [start_time, end_time]

        # ====== 调 Diffusion.inference 生成 latent，再用 VAE decode 回波形 ======
        with torch.autocast(device_type='cuda', dtype=self.precision):
            x = self.dit.inference(inputs, timesteps=num_step, seq_cfg_w=cfg_w)

            # 把前面 ref 的部分替换为真实 prompt latent
            if lat_ctx_ref.shape[1] > 0:
                x[:, :lat_ctx_ref.shape[1]] = lat_ctx_ref

            hop_size = self.hp_vae['hop_size']
            vae_stride = self.hp_vae['vae_stride']

            ref_lat = lat_ctx_ref.size(1) if lat_ctx_ref is not None else 0
            gen_lat = gen_lat_len

            # 0.5 秒 overlap -> latent 帧数
            overlap_sec = 0.5
            overlap_lat = int(overlap_sec * 24000 / hop_size / vae_stride)

            # 只解码 [ref_lat - overlap_lat, ref_lat + gen_lat]
            start_lat = max(0, ref_lat - overlap_lat)
            end_lat = ref_lat + gen_lat
            x_dec = x[:, start_lat:end_lat]   # [1, L_dec, C]

            # 解码
            with torch.autocast(device_type='cuda', dtype=self.precision):
                wav_dec = self.vae.decode(x_dec)[0, 0].to(torch.float32)

            # 丢掉 overlap 对应的 wav（以及 ref 部分）
            drop_lat = ref_lat - start_lat               # <= overlap_lat
            drop_wav = drop_lat * vae_stride * hop_size  # samples
            wav_pred = wav_dec[drop_wav:]                # 现在基本就是 “gen wav”


            # clip 防止溢出
            if wav_pred.abs().max() > 1:
                print('Wav amplitude exceed 1, clip it.')
                wav_pred = wav_pred / (wav_pred.abs().max())

            wav_pred = wav_pred.cpu().numpy()

        return wav_pred


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ===== 文本清洗辅助正则（与数据处理侧对齐） =====
_SPACE_RE = re.compile(r"\s+")

_S1S2_TAG_RE = re.compile(
    r'<\s*(S[1-4])\s*>(.*?)</\s*S[1-4]\s*>',
    flags=re.IGNORECASE | re.DOTALL,
)

_S1S2_TEXT_RE = re.compile(
    r'<\s*(S[1-4])\s*>(.*?)</\s*\1\s*>',
    flags=re.IGNORECASE | re.DOTALL,
)


def strip_s1s2_tags(x):
    """
    去掉 <S1>...</S1> / <S2>...</S2> 标签，只保留内部文本。
    支持 str 或 list[str]。
    """
    if isinstance(x, str):
        return _S1S2_TEXT_RE.sub(lambda m: m.group(2), x)
    elif isinstance(x, (list, tuple)):
        return [
            _S1S2_TEXT_RE.sub(lambda m: m.group(2), t) if isinstance(t, str) else t
            for t in x
        ]
    return x

def _norm_spaces_caption(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\u3000", " ")
    s = _SPACE_RE.sub(" ", s)
    return s.strip()

def raw_text_process(txt, wav=None, wav_len=None, check_len=False):
    """
    训练侧同款清洗规则（简化 skip 打印），默认不做长度过滤。
    """
    if not isinstance(txt, str):
        return ""

    txt = txt.strip()
    if txt.startswith('sil '):
        txt = txt[4:]
    txt = txt.replace(' sil ', ' ')
    txt = txt.replace(' ,', ',').replace(',,', ',').replace(' ，', '，').replace('， ', '，')
    txt = txt.replace(' .', '.').replace(' 。', '。').replace('。 ', '。').replace('。 ', '。')
    txt = txt.replace(' ?', '?').replace(' ？', '？').replace('？ ', '？').replace('？ ', '？')
    txt = txt.replace(' !', '!').replace(' ！', '！').replace('！ ', '！').replace('！ ', '！')
    txt = txt.replace(' ;', ',').replace(' ；', '，').replace('； ', '，').replace('； ', '，').replace(';', ',').replace('；', '，')
    txt = txt.replace(' :', ',').replace(' ：', '，').replace('： ', '，').replace(':', ',').replace('：', '，')
    txt = txt.replace(' 、', '，').replace('、 ', '，').replace('、', '，')
    txt = txt.replace('"', '').replace('“', '').replace('”', '')
    txt = txt.replace('- ', ' ')
    txt = txt.replace('+', ' ')
    txt = txt.replace('，。', '。').replace('。，', '。')
    txt = txt.replace(':。', '。').replace('：。', '。')
    txt = txt.replace('……', '，')
    txt = remove_spaces_between_chinese(txt)

    if txt and txt[-1] not in '.,?!;。，？！；、':
        if is_chinese(txt):
            txt = txt + '。'
        else:
            txt = txt + '.'

    if wav is not None:
        wav_len = wav.shape[0]

    # 仅当你未来真的想做“推理侧长度保护”才打开
    if check_len and wav_len is not None:
        try:
            if len(get_word_list(txt)) > wav_len // hparams['hop_size'] // 4:
                return None
        except Exception:
            # hparams 未就绪/异常时不做过滤
            pass

    return txt


def raw_text_process_s1s2_tagged(txt, wav=None, wav_len=None, check_len=False):
    """
    对含 <S1>/<S2> 的 text 做 raw_text_process 等价清洗，
    只处理标签内部文本，保留标签结构与顺序。
    """
    if not isinstance(txt, str) or not txt.strip():
        return ""

    # 没标签就走普通清洗
    if _S1S2_TEXT_RE.search(txt) is None:
        return raw_text_process(txt, wav=wav, wav_len=wav_len, check_len=check_len)

    if wav is not None:
        wav_len = wav.shape[0]

    # 可选总长度检查（默认关闭）
    if check_len and wav_len is not None:
        plain_parts = []
        for m in _S1S2_TEXT_RE.finditer(txt):
            inner = _norm_spaces_caption(m.group(2))
            if inner:
                plain_parts.append(inner)
        plain = " ".join(plain_parts)

        if plain:
            plain_norm = raw_text_process(plain, wav=None, wav_len=None, check_len=False) or ""
            try:
                if len(get_word_list(plain_norm)) > wav_len // hparams['hop_size'] // 4:
                    return None
            except Exception:
                pass

    # 分段清洗并重建
    def _repl(m):
        tag = m.group(1).upper()
        inner = m.group(2)
        inner_proc = raw_text_process(inner, wav=None, wav_len=None, check_len=False)
        if inner_proc is None:
            inner_proc = ""
        return f"<{tag}>{inner_proc}</{tag}>"

    out = _S1S2_TEXT_RE.sub(_repl, txt)
    out = _SPACE_RE.sub(' ', out).strip()
    return out

def extract_s_text(caption: str) -> str:
    """
    从 caption 中按顺序提取 <S1>...</S1> 和 <S2>...</S2>，
    并把相邻的同标签片段合并成一个 <S1>... ...</S1> 或 <S2>... ...</S2>。
    若没有任何 S1/S2 片段则返回 ""。
    返回结果带有 S1/S2 包裹的合并版本。
    """
    if not isinstance(caption, str) or not caption.strip():
        return ""

    segments = []
    for tag, inner in _S1S2_TAG_RE.findall(caption):
        inner_norm = _norm_spaces_caption(inner)
        if not inner_norm:
            continue
        segments.append([tag.upper(), inner_norm])

    if not segments:
        return ""

    merged = []
    for tag, content in segments:
        if merged and tag == merged[-1][0]:
            merged[-1][1] = merged[-1][1] + ' ' + content
        else:
            merged.append([tag, content])

    out = ''.join(f"<{tag}>{content}</{tag}>" for tag, content in merged)
    return out if out.strip() else ""

def _encode_tag_pattern(tokenizer, s: str):
    ids = tokenizer.encode(s)
    if isinstance(ids, np.ndarray):
        ids = ids.tolist()
    return list(map(int, ids))

def _get_sx_token_patterns(tokenizer):
    patterns = {}
    for i in range(1, 5):
        tag = f"S{i}"
        patterns[tag] = {
            "open": _encode_tag_pattern(tokenizer, f"<{tag}>"),
            "close": _encode_tag_pattern(tokenizer, f"</{tag}>"),
            "id": i,
        }
    return patterns

def _find_pattern_starts(arr: np.ndarray, pat: np.ndarray):
    n = arr.shape[0]
    m = pat.shape[0]
    if m == 0 or n < m:
        return np.empty((0,), dtype=np.int64)
    if m == 1:
        return np.flatnonzero(arr == pat[0]).astype(np.int64)
    try:
        win = np.lib.stride_tricks.sliding_window_view(arr, m)
    except Exception:
        shape = (n - m + 1, m)
        strides = (arr.strides[0], arr.strides[0])
        win = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    eq = (win == pat)
    return np.flatnonzero(eq.all(axis=1)).astype(np.int64)

def build_spk_mask_from_text_tokens(text_tokens: torch.LongTensor, sx_patterns: dict):
    if text_tokens is None or text_tokens.numel() == 0:
        return torch.zeros((0,), dtype=torch.long)

    arr = text_tokens.detach().cpu().numpy().astype(np.int64, copy=False)
    arr = np.ascontiguousarray(arr)  # ✅ 强烈建议加，避免 stride view 出坑
    L = arr.shape[0]
    mask = np.zeros((L,), dtype=np.int16)

    for _, info in sx_patterns.items():
        open_pat = np.asarray(info["open"], dtype=np.int64)
        close_pat = np.asarray(info["close"], dtype=np.int64)
        spk_id = int(info["id"])
        if open_pat.size == 0 or close_pat.size == 0:
            continue

        open_starts = _find_pattern_starts(arr, open_pat)
        if open_starts.size == 0:
            continue
        close_starts = _find_pattern_starts(arr, close_pat)
        if close_starts.size == 0:
            continue

        inner_start_min = open_starts + open_pat.size
        idx = np.searchsorted(close_starts, inner_start_min, side="left")
        valid = idx < close_starts.size
        if not np.any(valid):
            continue

        open_starts_v = open_starts[valid]
        close_starts_v = close_starts[idx[valid]]
        close_ends_v = close_starts_v + close_pat.size - 1

        for s, e in zip(open_starts_v.tolist(), close_ends_v.tolist()):
            if s < 0: s = 0
            if e >= L: e = L - 1
            if s <= e:
                mask[s:e + 1] = spk_id

    return torch.from_numpy(mask.astype(np.int64))


def worker(rank, world_size, args, cfg, out_path, master_port):

    device = f'cuda:{rank}'
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(rank)
    print(f"[CUDA:{rank}] name={props.name}, total={_fmt_mb(props.total_memory)}")
    log_cuda_mem("after set_device()", device)


    # ====== init process group（保持你原来的逻辑）======
    if world_size > 1:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(master_port)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(rank)

        from utils.commons import trainer
        trainer.LOCAL_RANK = rank

        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size,
            device_id=torch.device(rank),
            timeout=timedelta(seconds=3000)
        )

    dit_ckpt = args.dit_ckpt
    infer_ins = ScriptSpeechInfer(
        device,
        dit_ckpt=dit_ckpt,
        vae_ckpt=args.vae_ckpt,
        merge_ckpt=args.merge_ckpt,
        merge_weight=args.merge_weight,
        use_sa_front=cfg.get('use_sa_front', False),
    )
    os.makedirs(out_path, exist_ok=True)
    negative_prompt = cfg.get('negative_prompt', None)

    samples = cfg.get('samples', []) or []
    N = len(samples)

    # 每个 rank 固定跑这么多轮：保证所有 rank forward 次数一致
    num_iters = int(math.ceil(N / float(world_size))) if N > 0 else 0

    for t in range(num_iters):
        idx = t * world_size + rank
        is_dummy = idx >= N

        if is_dummy:
            # dummy 占位：仍然调用 forward 以参与 collective
            sample = {"caption": "", "prompt_audio": None}
        else:
            sample = samples[idx]

        print(f"[Rank {rank}] Iter {t}/{num_iters-1} | idx={idx} | dummy={is_dummy}")

        # ====== 构造 text / caption ======
        caption = sample.get('caption', '')
        if not isinstance(caption, str):
            caption = str(caption)
        caption = _norm_spaces_caption(caption)

        text = extract_s_text(caption)
        text = raw_text_process_s1s2_tagged(text)
        if text is None:
            text = ""
        print(f"[Rank {rank}] text: {text}")
        print(f"[Rank {rank}] caption: {caption}")

        # dummy 的 seed 也给一个固定值即可
        set_seed((len(text) + idx) if not is_dummy else (100000 + rank * 1000 + t))

        # prompt audio（ref wav）
        if (not is_dummy) and sample.get('prompt_audio'):
            audio_path = sample['prompt_audio']
            audio, _ = librosa.load(audio_path, sr=24000)
        else:
            audio = None

        # dummy 可以把 num_step 降到 1，减少无意义计算（仍会走一次 forward 参与 collective）
        num_step = cfg.get('num_step', 100)
        if is_dummy:
            num_step = 1

        wav = infer_ins.forward(
            text,
            ref_audio=audio,
            prompt=caption,
            cfg_w=cfg.get('cfg_w', None),
            infer_length=cfg.get('infer_length', 0),
            num_step=num_step,
            start_time=cfg.get('vad_len', 0.2),
            end_time=cfg.get('vad_len', 0.2),
            negative_prompt=negative_prompt,
        )

        # dummy 不落盘
        if (not is_dummy) and wav is not None:
            print(f"[Rank {rank}] save wav at {out_path}/out_{idx}.wav")
            sf.write(f'{out_path}/out_{idx}.wav', wav, 24000, 'PCM_16')

    # ====== 确保所有 rank 一起结束，防止有人提前退出导致别人 collective 卡住 ======
    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    print(f"[Rank {rank}] Finished all samples (with dummy padding if needed).")


if __name__ == '__main__':
    if os.path.isfile('.env.local'):
        from dotenv import load_dotenv

        load_dotenv('.env.local')

    kill_void()

    try:
        set_start_method('spawn')  # 多进程启动方式，Linux/Windows 通用
    except RuntimeError:
        pass

    parser = ArgumentParser()
    parser.add_argument("--config", help="Path to YAML config")
    parser.add_argument("--dit_ckpt", help="Path to model", type=str,
                        default='checkpoints/250622_scriptspeech_dit_singlespk_01')
    parser.add_argument("--merge_ckpt", help="Path to merge model", type=str)
    parser.add_argument("--merge_weight", help="Weight to merge model", type=float)
    parser.add_argument("--vae_ckpt", help="Path to VAE ckpt", type=str)
    args = parser.parse_args()
    # 读取 config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_path = f'{cfg["out_path"]}/{os.path.basename(args.dit_ckpt)}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    base_port = int(cfg.get("master_port", 10521))  # 允许你在 YAML 里配
    master_port = find_available_port(base_port, host="127.0.0.1", max_search=2000)
    print(f"[PORT] MASTER_PORT selected: {master_port} (base={base_port})")

    # 启动多进程，每个进程绑定一张 GPU
    processes = []
    gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    for rank in range(gpus):
        p = Process(target=worker, args=(rank, gpus, args, cfg, out_path, master_port))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All ranks finished. 可以做后处理或上传结果")
    # upload_tos_html(out_path)  # 可选，上传结果
    desc = (f"Inference setting: cfg weight: {cfg.get('cfg_w', None)}, inference step: {infer_step}, "
            f"extend duration: {extend_dur}, vad length (silence duration at bugin and tail): {vad_len}")
    upload_tos_html(yml=args.config, out_path=out_path, title_name=os.path.basename(out_path), extra_desc=desc)
    # CUDA_VISIBLE_DEVICES=0 python inference/tts/scriptspeech_infer.py
