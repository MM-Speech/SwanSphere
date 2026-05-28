import os
import struct
import numpy as np
import traceback
import mmap
import json
from pathlib import Path
import tempfile
from typing import List, Tuple, Sequence, Literal, Dict, Union

def save_dicts_to_jsonl(items: List[Dict], jsonl_path: Union[str, Path]):
    with open(jsonl_path, 'w') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def load_samples_from_jsonl(jsonl_path):
    lines = []
    with open(jsonl_path, 'rb') as f:
        for line in f.readlines():
            if line.strip() != '':
                lines.append(json.loads(line))
    return lines

def build_jsonl_index(jsonl_path, idx_path=None, use_tqdm=True):
    """
    为 JSONL 文件创建行索引，并显示处理进度的tqdm进度条。
    进度条基于文件处理的字节数。
    """
    if idx_path is None:
        idx_path = jsonl_path + '.idx'
    
    # 1. 获取文件总大小，用于tqdm计算进度
    total_size = os.path.getsize(jsonl_path)
    
    # 二进制模式，避免编码换行差异
    with open(jsonl_path, 'rb') as f, open(idx_path, 'wb') as idx:
        pos = 0
        # 2. 使用tqdm包装循环，并配置进度条
        #    - total: 进度条的总长度（这里是文件总字节数）
        #    - unit: 进度的单位（'B' 表示字节）
        #    - unit_scale: 自动将单位转换为KB, MB, GB等
        #    - desc: 进度条的描述文字
        if use_tqdm:
            from tqdm import tqdm
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Building index for {os.path.basename(jsonl_path)}") as pbar:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    idx.write(struct.pack('<Q', pos))
                    line_length = len(line)
                    pos += line_length
                    # 3. 更新进度条，前进刚刚读取的字节数
                    pbar.update(line_length)
        else:
            while True:
                line = f.readline()
                if not line:
                    break
                idx.write(struct.pack('<Q', pos))
                line_length = len(line)
                pos += line_length
                
    return idx_path

def get_jsonl_line_by_number(jsonl_path, idx_path, n, use_mmap=True, parser='orjson', verbose=True):
    try:
        offsets = np.memmap(idx_path, dtype=np.uint64, mode='r')
        if n < 0 or n >= len(offsets):
            raise IndexError("line number out of range")
        start_off = int(offsets[n])
        end_off = int(offsets[n+1]) if (n+1) < len(offsets) else os.path.getsize(jsonl_path)
        length = end_off - start_off
        if length <= 0:
            raise IndexError("line length <= 0")
        if use_mmap:
            with open(jsonl_path, 'rb') as f:
                mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
                data = mm[start_off:end_off]
                mm.close()
        else:
            with open(jsonl_path, 'rb') as f:
                data = os.pread(f.fileno(), length, start_off)
        data = data.rstrip(b'\r\n')
        if parser == 'orjson':
            import orjson
            return orjson.loads(data)
        elif parser == 'json':
            import json
            return json.loads(data)
        elif parser == 'simdjson':
            import simdjson
            return simdjson.Parser().parse(data)
    except:
        if verbose:
            print(f"{jsonl_path = } {idx_path = } {n = }")
            traceback.print_exc()

def count_jsonl_n_lines(idx_path):
    size = os.path.getsize(idx_path)
    if size % 8 != 0:
        raise ValueError("idx 文件大小不是 8 的整数倍，可能不是全量偏移索引或文件损坏")
    return size // 8


class JsonlChunkReader:
    
    def __init__(
        self,
        jsonl_path: str,
        idx_path: str = None,
        mmap_jsonl: bool = True,
        mmap_idx: bool = True,
        parser: Literal['orjson', 'simdjson', 'json'] = 'orjson',
    ):
        self.jsonl_path = jsonl_path
        self.idx_path = idx_path = idx_path if idx_path is not None else jsonl_path + '.idx'
        if not os.path.isfile(idx_path):
            self.idx_path = build_jsonl_index(jsonl_path, idx_path, use_tqdm=False)
        self.file_size = os.path.getsize(jsonl_path)
        # 索引内存映射（uint64偏移）
        self.offsets = np.memmap(idx_path, dtype=np.uint64, mode='r') if mmap_idx else None
        # JSONL 内存映射或文件描述符
        self._f = open(jsonl_path, 'rb')
        self._mm = mmap.mmap(self._f.fileno(), length=0, access=mmap.ACCESS_READ) if mmap_jsonl else None
        # 选择解析器
        self.parser_name = parser
        if parser == 'simdjson':
            import simdjson
            self._simd_parser = simdjson.Parser()
        else:
            self._simd_parser = None

    def close(self):
        try:
            if self._mm is not None:
                self._mm.close()
            if self._f is not None:
                self._f.close()
        except Exception:
            pass

    def _get_offsets(self, start: int, end: int, strict=False) -> Tuple[int, int]:
        # 获取 [start, end] 的字节范围 [start_off, end_off)
        if self.offsets is None:
            offsets = np.memmap(self.idx_path, dtype=np.uint64, mode='r')  # 退化为每次构造
        else:
            offsets = self.offsets
            
        if not strict:
            if end >= len(offsets):
                end = len(offsets) - 1

        if start < 0 or end < start or end >= len(offsets):
            raise IndexError("line range out of range")
        start_off = int(offsets[start])
        if end + 1 < len(offsets):
            end_off = int(offsets[end + 1])
        else:
            end_off = self.file_size
        return start_off, end_off

    def read_range(self, start: int, end: int, strict=False):
        """
        读取 [start, end] 连续行，返回已解析的对象列表。
        """
        start_off, end_off = self._get_offsets(start, end, strict)
        length = end_off - start_off
        if length <= 0:
            return []

        if self._mm is not None:
            # 注意：mmap 切片返回 bytes（会复制），但只做一次大拷贝，减少多次系统调用
            chunk = self._mm[start_off:end_off]
        else:
            # 使用 pread 一次性读取
            chunk = os.pread(self._f.fileno(), length, start_off)

        # 解析
        if self.parser_name == 'orjson':
            import orjson
            # 分行解析（JSON 允许空白字符，若行末有换行，orjson也能处理；也可以 rstrip）
            return [orjson.loads(line) for line in chunk.splitlines()]
        elif self.parser_name == 'json':
            import json
            return [json.loads(line) for line in chunk.splitlines()]
        elif self.parser_name == 'simdjson':
            # simdjson 的 parse_many 可以直接解析 NDJSON（推荐用于批量 chunk）
            return list(self._simd_parser.parse_many(chunk))
        else:
            raise ValueError(f"unknown parser: {self.parser_name}")

    def read_one(self, n: int):
        """
        单行读取（通过两个偏移计算长度 + mmap 切片/posix pread），避免 readline 的逐字节扫描。
        """
        start_off, end_off = self._get_offsets(n, n)
        length = end_off - start_off
        if length <= 0:
            raise IndexError("line length <= 0")

        if self._mm is not None:
            data = self._mm[start_off:end_off]
        else:
            data = os.pread(self._f.fileno(), length, start_off)

        # 去掉换行更保险（JSON允许空白，但某些解析器严格）
        data = data.rstrip(b"\r\n")
        if self.parser_name == 'orjson':
            import orjson
            return orjson.loads(data)
        elif self.parser_name == 'json':
            import json
            return json.loads(data)
        elif self.parser_name == 'simdjson':
            # 单对象用 parse；parse_many 也行但没必要
            return self._simd_parser.parse(data)


def get_jsonl_lines_by_range(jsonl_path, idx_path, start, end, use_mmap=True, parser='orjson'):
    file_size = os.path.getsize(jsonl_path)
    offsets = np.memmap(idx_path, dtype=np.uint64, mode='r')
    if start < 0 or end < start or end >= len(offsets):
        raise IndexError("line range out of range")

    start_off = int(offsets[start])
    end_off = int(offsets[end+1]) if (end+1) < len(offsets) else file_size
    length = end_off - start_off
    if length <= 0:
        return []

    if use_mmap:
        with open(jsonl_path, 'rb') as f:
            mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
            chunk = mm[start_off:end_off]
            mm.close()
    else:
        with open(jsonl_path, 'rb') as f:
            chunk = os.pread(f.fileno(), length, start_off)

    if parser == 'orjson':
        import orjson
        return [orjson.loads(line) for line in chunk.splitlines()]
    elif parser == 'json':
        import json
        return [json.loads(line) for line in chunk.splitlines()]
    elif parser == 'simdjson':
        import simdjson
        return list(simdjson.Parser().parse_many(chunk))
    else:
        raise ValueError(f"unknown parser: {parser}")

