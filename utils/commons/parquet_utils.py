from dataclasses import dataclass
from typing import List, Optional, Union
import pyarrow.parquet as pq
import pyarrow as pa
from collections import OrderedDict
import bisect

@dataclass
class ParquetChunkReader:
    path: str
    chunk_size: int
    columns: Optional[List[str]] = None
    memory_map: bool = True
    cache_max: int = 16  # 小型 LRU 缓存大小

    def __post_init__(self):
        self.pf = pq.ParquetFile(self.path, memory_map=self.memory_map)
        self.rg_rows = [self.pf.metadata.row_group(i).num_rows for i in range(self.pf.num_row_groups)]
        self.rg_cumsum = []
        s = 0
        for n in self.rg_rows:
            s += n
            self.rg_cumsum.append(s)
        self.total_rows = self.rg_cumsum[-1] if self.rg_cumsum else 0
        self.num_chunks = (self.total_rows + self.chunk_size - 1) // self.chunk_size

        # LRU 缓存：rg_idx -> pa.Table
        self._rg_cache = OrderedDict()

    def __len__(self):
        return self.num_chunks

    def _read_rg(self, rg_idx: int) -> pa.Table:
        # 命中缓存
        if rg_idx in self._rg_cache:
            self._rg_cache.move_to_end(rg_idx)
            return self._rg_cache[rg_idx]
        # 读入并放入缓存
        tbl = self.pf.read_row_group(rg_idx, columns=self.columns)
        self._rg_cache[rg_idx] = tbl
        if len(self._rg_cache) > self.cache_max:
            self._rg_cache.popitem(last=False)  # 淘汰最旧
        return tbl

    def read_chunk(self, chunk_idx: int, to_pylist: bool = True, to_pandas: bool = False):
        if chunk_idx < 0 or chunk_idx >= self.num_chunks:
            raise IndexError("chunk_idx out of range")

        start = chunk_idx * self.chunk_size
        end = min(start + self.chunk_size, self.total_rows)

        rg_start = bisect.bisect_right(self.rg_cumsum, start)
        rg_end   = bisect.bisect_right(self.rg_cumsum, end - 1)

        pieces = []
        for rg in range(rg_start, rg_end + 1):
            rg_global_begin = 0 if rg == 0 else self.rg_cumsum[rg - 1]
            rg_global_end   = self.rg_cumsum[rg]
            local_start = max(0, start - rg_global_begin)
            local_end   = min(rg_global_end - rg_global_begin, end - rg_global_begin)
            if local_start >= local_end:
                continue
            tbl = self._read_rg(rg)
            if local_start != 0 or local_end != (rg_global_end - rg_global_begin):
                tbl = tbl.slice(local_start, local_end - local_start)
            pieces.append(tbl)

        if not pieces:
            # 空 chunk，构造空表（兼容没有数据的边界情况）
            arrow_schema = getattr(self.pf, "schema_arrow", None)
            if arrow_schema is None:  # 兼容旧版 PyArrow
                # 退化：读一个空表再取 schema
                tmp = self.pf.read_row_group(0, columns=self.columns) if self.pf.num_row_groups > 0 else pa.table({})
                arrow_schema = tmp.schema
            cols = self.columns or [f.name for f in arrow_schema]
            empty_cols = {c: pa.array([], type=arrow_schema.field(c).type) for c in cols}
            out = pa.table(empty_cols)
        else:
            out = pa.concat_tables(pieces, promote=True)

        if to_pylist:
            return out.to_pylist()
        if to_pandas:
            return out.to_pandas()
        return out
