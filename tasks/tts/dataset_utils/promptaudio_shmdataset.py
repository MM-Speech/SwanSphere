import collections
import collections.abc
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))

import os
import random
import json
from copy import deepcopy
import pickle
import re
import time
import hashlib
import glob
import traceback

import torch
import numpy as np
import torch.utils
import torch.utils.data
import librosa

from dataloader import FalconReader, KVReader
from utils.commons.hparams import hparams
from utils.commons.os_utils import multiprocess_glob, handle_exacption
from utils.dataset.batcher import BucketBatcher
from utils.commons.io import get_wav_duration, print_once
from utils.text.split_text import get_word_list
from utils.commons.base_shm_dataset import BaseFalconReaderShmDataset, BaseShmDataset, get_from_global_stores
from utils.commons.dataset_utils import collate_xd, pad_or_cut_xd
from utils.audio.vad import build_vad_model, run_vad_trim

from modules.tts.ar_dur.commons.align_ops import compute_mel2aug_from_dur
from modules.tts.ar_dur.commons.nar_tts_modules import LengthRegulator

# ============= 小工具 =============
def remove_tag_blocks(s: str, tag: str = "tag", replace=' ') -> str:
    pattern = rf"\s*<{tag}\b[^>]*>.*?</{tag}>\s*"
    s = re.sub(pattern, " ", s, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"\s+", replace, s).strip()

def valid_item_kv(item, k):
    return k in item and item[k] is not None

def _ensure_trailing_punct(s: str):
    if not s: return s
    if re.search(r'[。！？.!?…]$', s):
        return s
    return s + ('。' if re.search(r'[\u4e00-\u9fff]', s) else '.')

_BND_RE = re.compile(r'[ \t\n，。！？；：,.!?…]')
def _split_text_by_token_ratio(txt_norm:str, insert_idx0:int, n_tokens:int):
    """根据 token 比例把 txt_norm 切为 (ref_text, tgt_text)。insert_idx0: 0-based"""
    if not txt_norm or n_tokens <= 0:
        return "", txt_norm or ""
    L = len(txt_norm)
    j = int(round((max(0, min(insert_idx0, n_tokens)) / float(n_tokens)) * L))
    j = max(0, min(L, j))
    cut = j
    found = None
    for k in range(j, min(L, j+48)):
        if _BND_RE.match(txt_norm[k]):
            found = k+1
            break
    if found is None:
        for k in range(j-1, max(0, j-48)-1, -1):
            if _BND_RE.match(txt_norm[k]):
                found = k+1
                break
    if found is not None:
        cut = found
    ref = txt_norm[:cut].strip()
    tgt = txt_norm[cut:].strip()
    return ref, tgt

def _wrap_s1_if_nonempty(s: str) -> str:
    s = (s or '').strip()
    return (f"<S1>{s}</S1>") if s else ""

def _fade_edges(x, sr, ms=5):
    n = int(sr * ms / 1000.0)
    if n <= 0 or len(x) < 2*n: return x
    ramp = np.linspace(0, 1, n, dtype=x.dtype)
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return x

def _rms(x): return float(np.sqrt(np.mean(x.astype(np.float64)**2) + 1e-12))

def _mix_at_snr(s, n, snr_db):
    Ps = _rms(s)**2
    Pn = _rms(n)**2 + 1e-12
    a = np.sqrt(Ps/(Pn*(10.0**(snr_db/10.0))))
    y = s + a*n
    m = float(np.max(np.abs(y)) + 1e-12)
    return (y/m if m>1.0 else y).astype(np.float32)

def _mel2token_to_dur(mel2ph: torch.LongTensor) -> np.ndarray:
    """把 mel2ph(1-based) 转为每个 token 的时长（frames）"""
    if mel2ph is None or mel2ph.numel() == 0:
        return np.zeros((0,), dtype=np.int64)
    m = int(torch.max(mel2ph).item())
    hist = np.bincount(mel2ph.cpu().numpy(), minlength=m+1)  # 0..m
    return hist[1:].astype(np.int64)

def _atomic_dump_pickle(obj, dst):
    tmp = dst + f".tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, dst)

def _acquire_lock(lock_path, timeout=600, poll=0.1):
    start = time.time()
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        import fcntl
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except BlockingIOError:
                if time.time() - start > timeout:
                    raise TimeoutError(f"cache lock timeout: {lock_path}")
                time.sleep(poll)
    except Exception:
        os.close(fd); raise

def _release_lock(fd):
    try:
        import fcntl; fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)

def _cache_path(jsonl_path, with_dur=False):
    return jsonl_path + (".paths_sr_cap_dur_v1.pkl" if with_dur else ".paths_sr_cap_v2.pkl")

def _pool_is_valid_dict(pool, with_dur):
    try:
        if not isinstance(pool, list) or len(pool) == 0:
            return False
        e = pool[0]
        base_keys = {'npy', 'sr', 'cap'}
        if not isinstance(e, dict) or not base_keys.issubset(e.keys()):
            return False
        if with_dur and 'dur' not in e:
            return False
        return True
    except Exception:
        return False

def _load_pool(jsonl_path, with_dur=False):
    """读取 jsonl 索引池：每行含 {npy(必有), sr, caption, duration(可选)}；本地缓存 pkl。"""
    if not jsonl_path:
        return []
    pkl = _cache_path(jsonl_path, with_dur)
    lock = pkl + ".lock"

    # 先尝试已有缓存
    if os.path.exists(pkl):
        try:
            with open(pkl, "rb") as f:
                pool = pickle.load(f)
            if _pool_is_valid_dict(pool, with_dur):
                return pool
        except Exception:
            pass

    fd = _acquire_lock(lock)
    try:
        if os.path.exists(pkl):
            try:
                with open(pkl, "rb") as f:
                    pool = pickle.load(f)
                if _pool_is_valid_dict(pool, with_dur):
                    return pool
            except Exception:
                pass

        if not os.path.exists(jsonl_path):
            print(f"| WARN: jsonl not found: {jsonl_path}")
            return []

        pool = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                o = json.loads(line)
                npy = o.get('npy_24k_path') or o.get('npy_path') or o.get('npy')
                sr  = int(o.get('sr_24k', hparams.get('audio_sample_rate', 24000)))
                cap = o.get('caption', '')
                if not npy or not os.path.exists(npy):
                    # 文件不存在则跳过该条
                    continue
                if not with_dur:
                    pool.append({'npy':npy, 'sr':sr, 'cap':cap})
                else:
                    dur = float(o['duration_24k']) if o.get('duration_24k') is not None else None
                    pool.append({'npy':npy, 'sr':sr, 'cap':cap, 'dur':dur})
        try:
            _atomic_dump_pickle(pool, pkl)
        except Exception:
            pass
        return pool
    finally:
        _release_lock(fd)

# ==== 预加载加速（可选 RAM/SHM） ====
_SHM_OWNERS = []  # 持有者，避免 SHM 被提前 GC；进程退出可统一释放

def _preload_audio_pool(pool, sr_model, to_shm=False, limit=None):
    """
    将 pool 中的 npy 音频预加载到 RAM 或 SHM。
    - 返回的新 pool 元素会包含 e['arr'](RAM) 或 e['shm'](SHM 元信息)，并把 sr 统一为 sr_model。
    - limit: 限制最多预加载多少条；None 表示全量。
    """
    try:
        from multiprocessing import shared_memory as _shm_mod
    except Exception:
        _shm_mod = None
    out = []
    n = 0
    for e in pool:
        if (limit is not None) and (n >= int(limit)):
            break
        npy = e.get('npy')
        try:
            x = np.load(npy, mmap_mode='r').astype(np.float32)
            sr_e = int(e.get('sr', sr_model))
            if sr_e != sr_model:
                x = librosa.resample(x, orig_sr=sr_e, target_sr=sr_model)
            x = x.astype(np.float32, copy=False).reshape(-1)

            ee = dict(e)  # 浅拷贝，避免改动原 dict
            ee['sr'] = sr_model
            if to_shm and (_shm_mod is not None):
                shm = _shm_mod.SharedMemory(create=True, size=x.nbytes)
                np.ndarray(x.shape, dtype=x.dtype, buffer=shm.buf)[:] = x
                ee['shm'] = {'name': shm.name, 'shape': (x.shape[0],), 'dtype': 'float32'}
                _SHM_OWNERS.append(shm)  # 保持 owner 活着；退出时由主进程 unlink
            else:
                ee['arr'] = x  # RAM
            out.append(ee)
            n += 1
        except Exception as ex:
            print_once(f'| WARN preload failed: {npy}: {repr(ex)}')
            continue
    return out

def _fetch_audio_from_entry(e, sr_model):
    """
    从预加载条目 e 取出音频（float32，1D），返回一个独立拷贝（不会修改缓存本体）。
    若未预加载，则回落到磁盘读取+重采样。
    """
    try:
        if 'arr' in e and isinstance(e['arr'], np.ndarray):
            return e['arr'].astype(np.float32, copy=True).reshape(-1)
        if 'shm' in e:
            from multiprocessing import shared_memory as _shm_mod
            meta = e['shm']
            shm  = _shm_mod.SharedMemory(name=meta['name'])
            view = np.ndarray((int(meta['shape'][0]),), dtype=np.float32, buffer=shm.buf)
            arr  = np.array(view, copy=True)  # 拷贝出独立副本
            shm.close()  # 关闭句柄（不 unlink，owner 持有）
            return arr
        # 回落到磁盘
        seg = np.load(e['npy'], mmap_mode='r').astype(np.float32)
        sr_e = int(e.get('sr', sr_model))
        if sr_e != sr_model:
            seg = librosa.resample(seg, orig_sr=sr_e, target_sr=sr_model)
        return seg.reshape(-1)
    except Exception:
        raise


class PromptAudioShmDataset(BaseShmDataset):
    """改版：在原模板上增加音频池读取与 BGM/SFX 拼接/叠加、pure 模式，以及 zero-shot/text-prompt 概率开关。"""

    # ============ 原模板：reader / meta ============
    def get_dataset_meta(self):
        """
        支持多 JSON/JSONL 文件（含 .gz），来源可为：
        - hparams['meta_paths']: list[str] / 逗号分隔 str / 通配符(glob) / 目录
        - hparams['meta_glob']: 仅通配符
        - hparams['meta_path']: 单文件（兼容旧写法）
        返回: (metas, len(metas))
        """
        import os, glob, json, gzip
        from utils.commons.io import print_once  # 已在文件头导入过，可再次安全引用

        def _expand_paths(spec):
            """把列表/逗号字符串/通配符/目录 统一展开为文件列表"""
            files = []
            if spec is None:
                return files
            if isinstance(spec, (list, tuple, set)):
                specs = list(spec)
            else:
                # 逗号分隔也支持
                specs = [s for s in str(spec).split(',') if s.strip()]

            for s in specs:
                s = s.strip()
                if not s:
                    continue
                if any(ch in s for ch in '*?[]'):  # 通配符
                    files.extend(sorted(glob.glob(s)))
                elif os.path.isdir(s):  # 目录：扫常见后缀
                    files.extend(sorted(
                        os.path.join(s, f)
                        for f in os.listdir(s)
                        if f.endswith(('.json', '.jsonl', '.json.gz', '.jsonl.gz'))
                    ))
                else:
                    files.append(s)
            return files

        hp = getattr(self, 'hparams', {}) or {}
        default_path = '/mnt/bn/genai-data2/renyi/entries/interns_env/leike/data/metadata/vdataset_1spk.json'

        # 优先级：meta_paths > meta_glob > meta_path > 默认
        meta_spec = hp.get('meta_paths') or hp.get('meta_glob') or hp.get('meta_path') or default_path
        meta_files = _expand_paths(meta_spec)
        if not meta_files and default_path:
            meta_files = [default_path]

        metas = []
        bad = 0

        for fp in meta_files:
            try:
                is_gz = fp.endswith('.gz')
                is_jsonl = fp.endswith('.jsonl') or fp.endswith('.jsonl.gz')
                opener = gzip.open if is_gz else open
                with opener(fp, 'rt', encoding='utf-8') as f:
                    if is_jsonl:
                        # JSON Lines
                        for ln, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue
                            metas.append(json.loads(line))
                    else:
                        # 普通 JSON
                        obj = json.load(f)
                        if isinstance(obj, list):
                            metas.extend(obj)
                        elif isinstance(obj, dict):
                            if isinstance(obj.get('data'), list):
                                metas.extend(obj['data'])
                            elif isinstance(obj.get('metas'), list):
                                metas.extend(obj['metas'])
                            else:
                                bad += 1
                                print_once(f'| WARN: {fp} 是 dict 但没有 "data"/"metas" 列表，已跳过。')
                        else:
                            bad += 1
                            print_once(f'| WARN: {fp} 既不是 list 也不是 dict，已跳过。')
            except Exception as ex:
                bad += 1
                print_once(f'| WARN: 读取失败 {fp}: {repr(ex)}')

        if not metas:
            print_once('| WARN: 没有加载到任何 meta 项。')
        print_once(f'| Loaded {len(metas)} items from {len(meta_files)} file(s); {bad} file(s) skipped.')
        return metas, len(metas)


    def prepare_reader(self, dataset_meta, global_stores):
        return 1
    
    def read_fn(self, idx, reader_pack, global_stores):
        # print(f"in read_fn, {idx = }")
        return self.dataset_meta[idx]

    # ============ 一次性全局资源：Batcher & 音频池 ============
    def _get_batcher(self, global_stores):
        return get_from_global_stores(
            'batcher', global_stores,
            lambda: BucketBatcher(
                buckets=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 
                         600, 650, 700, 750, 800, 850, 900, 950, 1000, 1200, 1400, 
                         1600, 1800, 2000, 2400, 2800, 3000],
                dynamic_batch=hparams.get("dynamic_batch", True),
                batch_size=hparams['max_sentences'],
                maximum_bucket_size=hparams['max_tokens'],
                length_fn=lambda x: x['len'],
            )
        )

    # def _get_length_regulator(self, global_stores):
    #     return get_from_global_stores('length_regulator', global_stores, lambda: LengthRegulator())

    # def _get_audio_pools(self, global_stores):
    #     """仅加载 jsonl 索引，不做 RAM/SHM 预加载，按需从磁盘读取。"""
    #     def _load():
    #         sfx_pool   = _load_pool(hparams.get('sfx_jsonl'),   with_dur=True)
    #         music_pool = _load_pool(hparams.get('music_jsonl'), with_dur=False)
    #         print_once(f"| SFX pool size: {len(sfx_pool)}; MUSIC pool size: {len(music_pool)}")
    #         # 这里不做任何预加载，保持磁盘懒加载
    #         return {'sfx': sfx_pool, 'music': music_pool}
    #     return get_from_global_stores('audio_pools', global_stores, _load)

    # ============ 处理主流程 ============
    def process_item(self, raw_item, hparams, global_stores):
        # print(f"{i_worker =}, {n_worker=}")
        # print(f"begin process_item...")
        batcher = self._get_batcher(global_stores) if self.use_fast_dataloader else None
        for item in self._process_item(raw_item, hparams, global_stores):
            # print(f"raw_item keys: {raw_item.keys()}")
            # print(f"item keys: {item.keys()}")
            if item is None:
                continue
            if self.use_fast_dataloader:
                batch = batcher.collate_batch(item)
                if batch is not None and len(batch) > 0:
                    yield batch
            else:
                yield item

    # def report_skip_status(self, cnt, item_cnt, name, i_worker, n_worker, step=100):
    #     if cnt > 0 and cnt % step == 0:
    #         print(f"| processer#{i_worker}/{n_worker}: skipped [{cnt}/{item_cnt}] items for [{name}]")
    
    def _process_item(self, raw_item, hparams, global_stores):
        """
        回退策略（更新）：
        - 无 text 或文本等价于 '<Audio>' -> text = '<Audio>'（不包裹 <S1>）；phone_encoded=[145]；tone_encoded=[0]；mel2ph=全1（长度由音频推导）
        - 其余：保持原逻辑；有 text 时正常拼成单字符串并包裹 <S1>…</S1>
        """
        # ===== 基本参数 =====
        fm         = int(hparams['frames_multiple'])
        hop        = int(hparams['hop_size'])
        fm_wav     = fm * hop
        vae_stride = int(hparams['vae_stride'])
        sr_model   = int(hparams.get('audio_sample_rate', 24000))

        meta_item = raw_item

        # ===== 读音频并裁剪到 fm_wav 的整倍数 =====
        wav_path = meta_item['wav_path']
        wav, _ = librosa.load(wav_path, sr=hparams['audio_sample_rate'])
        if wav.ndim > 1:
            wav = np.mean(wav, axis=0)
        wav = wav[:len(wav)//fm_wav*fm_wav].astype(np.float32, copy=False)
        if wav.size == 0:
            return  # 丢弃异常样本

        # 基于音频推导 mel 帧数，并对齐到 frames_multiple
        mel_len = (len(wav) // hop)
        mel_len = (mel_len // fm) * fm
        if mel_len <= 0:
            return

        # ===== 文本：纯 audio 用 '<Audio>'；有文本则包裹 <S1>…</S1> =====
        text_missing = ('text' not in meta_item) or (meta_item.get('text') is None) \
                    or (isinstance(meta_item.get('text'), str) and meta_item.get('text').strip() == '') \
                    or (isinstance(meta_item.get('text'), list) and len([t for t in meta_item['text'] if str(t).strip()]) == 0)

        if text_missing:
            text_raw = '<Audio>'
        else:
            text_raw = meta_item['text']
            if isinstance(text_raw, list):
                parts = [str(t) for t in text_raw if t is not None and str(t).strip() != '']
                text_raw = ' '.join(parts) if parts else '<Audio>'
            else:
                text_raw = str(text_raw).strip() or '<Audio>'

        # 仅在“纯 audio”（缺失文本或文本内容等价于 <Audio>）时不包裹 <S1>
        if re.fullmatch(r'\s*<\s*Audio\s*>\s*', text_raw, flags=re.IGNORECASE):
            text = '<Audio>'
        else:
            text = f'<S1>{text_raw}</S1>'

        # ===== phone/tone/mel2ph：回退兼容 =====
        needs_fallback_phone = text_missing or ('phone_encoded' not in meta_item) or (meta_item.get('phone_encoded') is None) or (len(meta_item.get('phone_encoded', [])) == 0)
        needs_fallback_tone  = text_missing or ('tone_encoded'  not in meta_item) or (meta_item.get('tone_encoded')  is None) or (len(meta_item.get('tone_encoded',  [])) == 0)
        needs_fallback_m2p   = text_missing or ('mel2ph'        not in meta_item) or (meta_item.get('mel2ph')        is None) or (len(meta_item.get('mel2ph',        [])) == 0)

        if needs_fallback_phone:
            phone_encoded = [145]  # 占位音素 id
        else:
            phone_encoded = list(meta_item['phone_encoded'])

        if needs_fallback_tone:
            tone_encoded = [0]     # 占位音调 id
        else:
            tone_encoded = list(meta_item['tone_encoded'])

        # 以音频推导的 mel_len 为准重建/对齐 mel2ph，保证长度为 fm 的整倍数
        if needs_fallback_m2p:
            mel2ph_list = [1] * mel_len  # 只有一个占位 token
        else:
            mel2ph_list = list(meta_item['mel2ph'])
            # 对齐到 mel_len（截断或补齐），并保证索引合法（1-based，且不超过 token 数）
            m2p_len = (len(mel2ph_list) // fm) * fm
            mel2ph_list = mel2ph_list[:m2p_len]
            if len(mel2ph_list) > mel_len:
                mel2ph_list = mel2ph_list[:mel_len]
            elif len(mel2ph_list) < mel_len:
                last = mel2ph_list[-1] if mel2ph_list else 1
                mel2ph_list = mel2ph_list + [last] * (mel_len - len(mel2ph_list))

        # 保证 mel2ph 的最大索引不超过 token 数
        num_tokens = max(1, len(phone_encoded))
        mel2ph_list = [1 if (p is None or int(p) < 1) else (int(p) if int(p) <= num_tokens else num_tokens) for p in mel2ph_list]

        # ===== 由 mel2ph 统计 dur =====
        raw_item_dur = []
        for idx in range(1, num_tokens + 1):
            raw_item_dur.append(mel2ph_list.count(idx))
        dur = raw_item_dur

        # ===== 转 array/tensor 并截到有效 token 范围 =====
        mel2ph_arr = torch.as_tensor(np.array(mel2ph_list, dtype=np.int64)).long()
        max_idx = int(mel2ph_arr.max().item()) if mel2ph_arr.numel() > 0 else 1

        ph_token = np.asarray(phone_encoded[:max_idx], dtype=np.int64)
        tone_arr = np.asarray(tone_encoded [:max_idx], dtype=np.int64)
        dur_arr  = np.asarray(dur           [:max_idx], dtype=np.int64)

        # ===== 其他字段（安全默认） =====
        caption = meta_item.get('caption', '')

        # ===== VAD mask（可选） =====
        vad_mask = None
        if hparams.get('add_vad_mask', False):
            vad_model = get_from_global_stores('vad_model', global_stores, lambda: build_vad_model())
            vm = hop * vae_stride
            vad_start, vad_end = run_vad_trim(wav, sr_model, vad_model, 0.5)
            if vad_start == vad_end == 0:
                vad_start, vad_end = run_vad_trim(wav, sr_model, vad_model, 0.3)
            n_lat = len(wav) // vm
            st_idx = max(0, min(n_lat, int(vad_start * sr_model // vm)))
            ed_idx = max(0, min(n_lat, int(vad_end   * sr_model // vm)))
            vad_mask = np.zeros((n_lat,), dtype=np.float32)
            if ed_idx > st_idx:
                vad_mask[st_idx:ed_idx] = 1.0

        # ===== 组装样本 =====
        item = {
            'wav': torch.from_numpy(wav),
            'text': text,
            'caption': caption,
            'ph_token': ph_token,
            'tone': tone_arr,
            'mel2ph': mel2ph_arr,
            'dur': dur_arr,
        }
        item['len'] = int(item['wav'].shape[0] / hparams['hop_size'] / hparams['vae_stride'])
        if vad_mask is not None:
            item['vad_mask'] = vad_mask

        yield item


    # ============ collater：保持向后兼容，补充新增字段 ============
    def collater(self, samples):
        if len(samples) == 1 and isinstance(samples[0], list):
            samples = samples[0]
        if len(samples) == 0:
            if hasattr(self, 'backup_batch') and self.backup_batch is not None:
                return self.backup_batch
            else:
                return {}
        # import pdb; pdb.set_trace()
        # samples = samples[0]
        wavs = collate_xd([s['wav'] for s in samples], 0.0)
        wav_lengths = torch.LongTensor([s['wav'].shape[0] for s in samples])
        batch = {
            'nsamples': len(samples),
            'wavs': wavs,
            'wav_lengths': wav_lengths,
        }
        if 'mel' in samples[0]:
            batch['mels'] = collate_xd([s['mel'] for s in samples], -6.0)
        if 'mel2ph' in samples[0]:
            batch['mel2ph'] = collate_xd([s['mel2ph'] for s in samples], 0)
        if 'mel2ph_sparse' in samples[0]:
            batch['mel2ph_sparse'] = collate_xd([s['mel2ph_sparse'] for s in samples], 0)
        if 'dur' in samples[0]:
            batch['dur'] = collate_xd([s['dur'] for s in samples], 0)
            batch['dur_len'] = torch.LongTensor([s['dur'].shape[0] for s in samples])

        if 'ctx_wav' in samples[0]:
            batch['ctx_wavs'] = collate_xd([s['ctx_wav'] for s in samples], 0.0)
        if valid_item_kv(samples[0], 'ctx_mask'):
            batch['ctx_mask'] = collate_xd([s['ctx_mask'] for s in samples], 0)

        # 文本 / token / 语气
        batch['text'] = [s.get('text','') for s in samples]
        if 'ph_token' in samples[0]:
            batch['ph_tokens'] = collate_xd([s['ph_token'] for s in samples], 0)
            batch['txt_lengths'] = torch.LongTensor([s['ph_token'].numel() if isinstance(s['ph_token'], torch.Tensor) else len(s['ph_token']) for s in samples])
        if 'tone' in samples[0]:
            batch['tone'] = collate_xd([s['tone'] for s in samples], 0)

        # 可选对齐扩展
        if 'ph_timestamp' in samples[0]:
            batch['ph_timestamp'] = collate_xd([s['ph_timestamp'] for s in samples], 797)
            batch['ph_timestamp_len'] = torch.LongTensor([s['ph_timestamp'].shape[0] for s in samples])
        if 'merged_ph_token' in samples[0]:
            batch['merged_ph_tokens'] = collate_xd([s['merged_ph_token'] for s in samples], 797)
            batch['merged_ph_tokens_len'] = torch.LongTensor([s['merged_ph_token'].shape[0] for s in samples])
        if 'ph_dur_seq' in samples[0]:
            batch['ph_dur_seqs'] = collate_xd([s['ph_dur_seq'] for s in samples], 797)
            batch['ph_dur_seqs_len'] = torch.LongTensor([s['ph_dur_seqs'].shape[0] for s in samples])
            batch['ph_dur_seq_dur_mask'] = collate_xd([s['ph_dur_seq_dur_mask'] for s in samples], 0)

        # 分数
        for k in ['stoi', 'pesq', 'si_sdr', 'mos']:
            if k in samples[0]:
                batch[k] = torch.Tensor([s[k] for s in samples])

        # prompt & caption（新增）
        if 'global_prompt' in samples[0]:
            batch['global_prompt'] = [s['global_prompt'] for s in samples]
        if 'local_prompt' in samples[0]:
            batch['local_prompt'] = [s['local_prompt'] for s in samples]
        if 'caption' in samples[0]:
            batch['caption'] = [s['caption'] for s in samples]
        if 'caption_audio' in samples[0]:
            batch['caption_audio'] = [s['caption_audio'] for s in samples]
        if 'vad_mask' in samples[0]:
            batch['vad_mask'] = collate_xd([s['vad_mask'] for s in samples], 0.0)
        # if 'len' in samples[0]:
        #     batch['len'] = collate_xd([s['len'] for s in samples]) 


        if not hasattr(self, 'backup_batch') or self.backup_batch is None or random.random() < 0.001:
            self.backup_batch = batch

        return batch



if __name__ == '__main__':
    import soundfile as sf
    from utils.commons.io import json_dump
    from tqdm import tqdm
    hparams = {
        'exp_name': 'test',
        'audio_sample_rate': 24000,
        'hop_size': 240,
        'max_sentences': 200,
        'max_tokens': 2000,
        'max_spk_num': 8,
        'tgt_duration_min': 20,
        'tgt_duration_max': 60,
        'fast_ds_prefetch_steps': 8,
        'ds_workers': 1,
        'frames_multiple': 8,
        'vae_stride': 4,
        'add_vad_mask': True
    }
    # meta_path = '/mnt/bn/genai-data2/renyi/entries/interns_env/leike/data/metadata/vdataset_1spk.json'
    # metas = json.load(open(meta_path))

    # print(len(metas))
    dataset = PromptAudioShmDataset(
        prefix='train', hparams=hparams, use_fast_dataloader=False, rank_id=0, world_size=1, batch_size=4
    )
    print('init dataset over')
    dataloader = dataset.get_dataloader(seed=1234, num_workers=4)
    print('init dataloader over')

    temp_dir = 'user/temp/test_dl'

    for idx, batch in tqdm(enumerate(dataloader)):
        if idx == 0:
            print(batch.keys())
            del batch['wavs']
            serializable_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    serializable_batch[key] = value.cpu().numpy().tolist()
                elif isinstance(value, np.ndarray):
                    serializable_batch[key] = value.tolist()
                # 可以添加对其他类型的处理
                else:
                    # 假设其他类型都是兼容的
                    serializable_batch[key] = value
            # --- 预处理结束 ---
            # 现在保存净化后的字典
            with open('my_batch.json', 'w', encoding='utf-8') as f:
                json.dump(serializable_batch, f, indent=4, ensure_ascii=False)
        break
    #     wavs = batch['wavs']

    #     wav = wavs[0].numpy()
    #     sf.write(os.path.join(temp_dir, f'{idx}.wav'), wav, hparams['audio_sample_rate'], 'PCM_16')
    #     ctx_wav = batch['ctx_wavs'][0].numpy()
    #     sf.write(os.path.join(temp_dir, f'{idx}_ctx.wav'), ctx_wav, hparams['audio_sample_rate'], 'PCM_16')

    #     texts = batch['text']
    #     json_dump({'text': texts[0]}, os.path.join(temp_dir, f'{idx}.json'))
        
    #     if idx > 10:
    #         break
