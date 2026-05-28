import os
from typing import Union, List, Dict, Any
from pathlib import Path
import csv
import json
import importlib
import struct
import traceback
np_installed = importlib.util.find_spec('numpy') is not None
if np_installed:
    import numpy as np

def save_df_to_tsv(dataframe, path: Union[str, Path], sep='\t'):
    _path = path if isinstance(path, str) else path.as_posix()
    dataframe.to_csv(
        _path,
        sep=sep,
        header=True,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )

def save_dicts_to_tsv(items: List[Dict], path: Union[str, Path], sep='\t'):
    import pandas as pd
    if len(items) <= 0:
        print('Nothing saved.')
    manifest_columns = list(items[0].keys())
    manifest = {c: [] for c in manifest_columns}
    for item in items:
        for c in manifest_columns:
            manifest[c].append(item[c])
    save_df_to_tsv(pd.DataFrame.from_dict(manifest), safe_path(path), sep)

def load_samples_from_tsv(tsv_path, sep='\t'):
    tsv_path = Path(tsv_path)
    if not tsv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {tsv_path}")
    with open(tsv_path) as f:
        reader = csv.DictReader(
            f,
            delimiter=sep,
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        samples = [dict(e) for e in reader]
    if len(samples) == 0:
        print(f"warning: empty manifest: {tsv_path}")
        return []
    return samples

def load_dict_from_tsv(tsv_path, key, sep='\t'):
    samples = load_samples_from_tsv(tsv_path, sep)
    samples = {samples[key]: sample for sample in samples}
    return samples

def load_samples_from_jsonl(jsonl_path):
    lines = []
    with open(jsonl_path, 'rb') as f:
        for line in f.readlines():
            if line.strip() != '':
                lines.append(json.loads(line))
    return lines

def safe_path(path):
    os.makedirs(Path(path).parent, exist_ok=True)
    return path

def remove_path(path):
    if os.path.isfile(path):
        os.remove(path)


def json_dumps(obj: Any, indent: int = 2, compact_list_threshold: int = float('inf')) -> str:
    """
    生成JSON时智能压缩简单列表
    
    参数：
        compact_list_threshold: 元素数量小于等于此值时尝试单行显示
    """
    def _serialize(o: Any, current_indent: int, in_list: bool = False) -> str:
        # 基础类型直接序列化
        if isinstance(o, (str, int, float, bool)):
            return json.dumps(o, ensure_ascii=False)
        if o is None:
            return 'null'
        
        # 处理列表
        if isinstance(o, list):
            # 判断是否简单列表
            is_simple = all(
                isinstance(item, (str, int, float, bool, type(None))) 
                and not isinstance(item, (list, dict))
                for item in o
            ) and len(o) <= compact_list_threshold
            
            items = []
            indent_str = ' ' * (current_indent + indent)
            
            for item in o:
                item_str = _serialize(item, current_indent + indent, in_list=True)
                items.append(f'\n{indent_str}{item_str}' if not is_simple else item_str)
            
            if is_simple:
                return f"[{', '.join(items)}]"
            else:
                return f"[{','.join(items)}\n{' ' * current_indent}]"
        
        # 处理字典
        if isinstance(o, dict):
            items = []
            indent_str = ' ' * (current_indent + indent)
            for k, v in o.items():
                key = json.dumps(k, ensure_ascii=False)
                value = _serialize(v, current_indent + indent)
                items.append(f'\n{indent_str}{key}: {value}')
            return f"{{{','.join(items)}\n{' ' * current_indent}}}"
        
        if np_installed and isinstance(o, np.ndarray):
            return _serialize(o.tolist(), current_indent, in_list)
        
        # 其他类型回退标准序列化
        return json.dumps(o, ensure_ascii=False)
    
    return _serialize(obj, 0)


def json_dump(obj: Any, path: str, indent: int = 2, compact_list_threshold: int = float('inf')) -> str:
    content = json_dumps(obj, indent, compact_list_threshold)
    with open(safe_path(path), 'w', encoding='utf-8') as f:
        f.write(content)
    return content

def get_wav_num_frames(path, sr=None, return_orig_sr=True):
    try:
        import wave
        with wave.open(path, 'rb') as f:
            sr_ = f.getframerate()
            if sr is None:
                sr = sr_
                if return_orig_sr:
                    return int(f.getnframes()), sr_
            return int(f.getnframes() / (sr_ / sr))
    except wave.Error:
        import soundfile as sf
        wav_file, sr_ = sf.read(path, dtype='float32')
        if sr is None:
            sr = sr_
            if return_orig_sr:
                return len(wav_file), sr_
        return int(len(wav_file) / (sr_ / sr))
    except:
        import librosa
        # wav_file, sr_ = librosa.core.load(path, sr=sr)
        # return len(wav_file)
        wav_file, sr_ = librosa.load(path=path, sr=None, mono=True)
        if sr is None:
            sr = sr_
            if return_orig_sr:
                return len(wav_file), sr_
        return int(len(wav_file) / (sr_ / sr))
    
def get_wav_duration(path):
    try:
        ffprobe_res = ffprobe(path)
        duration = float(ffprobe_res['format']['duration'])
        return duration
    except:
        n_frames, sr = get_wav_num_frames(path, sr=None, return_orig_sr=True)
        return n_frames / sr


def ffprobe(path):
    command = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        '-show_format',
        # '-select_streams', 'a:0',
        path
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    ffprobe_output = json.loads(result.stdout)
    return ffprobe_output


def convert_to_m4a(input_path, output_path=None):
    if output_path is None:
        output_path = Path(input_path).with_suffix('.m4a')
        assert output_path != input_path
    cmd = ' '.join([
        'ffmpeg -hide_banner',
        f'-i "{input_path}"',
        '-vn',
        '-c:a aac',
        '-map_metadata 0',
        '-movflags +faststart',
        f'"{output_path}"',
        '-loglevel quiet',
        '-y'
    ])
    subprocess.check_call(cmd, shell=True)
    return output_path


# 使用字典存储每个调用位置的计数器
_PRINT_ONCE_REGISTRY: Dict[str, int] = {}
def print_once(message: str, *, flush: bool = True, process_id: int = 0) -> None:
    import importlib
    torch_installed = False
    if importlib.util.find_spec('torch') is not None:
        import torch.distributed as dist
        torch_installed = True
        
    import inspect
    frame = inspect.currentframe().f_back  # 获取上一级调用帧
    try:
        # 使用文件名、行号和函数名组合作为唯一标识符
        caller_info = (
            frame.f_code.co_filename,
            frame.f_lineno,
            frame.f_code.co_name
        )
        call_id = f"{caller_info[0]}:{caller_info[1]}:{caller_info[2]}"
    finally:
        # 避免内存泄漏，手动删除引用
        del frame
    
    if torch_installed and dist.is_initialized() and dist.get_rank() == process_id:
        # 首次调用该位置时打印
        if _PRINT_ONCE_REGISTRY.get(call_id, 0) == 0:
            print(message, flush=flush)
            _PRINT_ONCE_REGISTRY[call_id] = 1
    else:
        # 非分布式环境处理
        if _PRINT_ONCE_REGISTRY.get(call_id, 0) == 0:
            print(message, flush=flush)
            _PRINT_ONCE_REGISTRY[call_id] = 1
            

import re
import subprocess

def get_audio_bitrate(file_path):
    try:
        # 调用ffmpeg获取文件信息
        result = subprocess.run(
            ['ffmpeg', '-i', file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 将stderr重定向到stdout
            text=True
        )
        
        # 从输出中提取比特率（正则匹配）
        output = result.stdout
        bitrate_match = re.search(r'(\d+) kb/s', output)
        
        if bitrate_match:
            return int(bitrate_match.group(1))  # 返回比特率数值（kb/s）
        else:
            return None  # 未找到比特率信息
        
    except Exception as e:
        return None

