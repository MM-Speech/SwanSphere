# utils/profile.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import time
import csv
import threading
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Iterable, Tuple

try:
    import psutil
except Exception as e:
    psutil = None

import tracemalloc

# ---------------------------
# 通用工具
# ---------------------------

def _ensure_psutil():
    if psutil is None:
        raise RuntimeError("psutil is required. Please `pip install psutil`.")

def _now_ts() -> float:
    return time.time()

def format_bytes(n_bytes: float, precision: int = 1) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    x = float(n_bytes)
    while x >= 1024.0 and idx < len(units) - 1:
        x /= 1024.0
        idx += 1
    fmt = f"{{:.{precision}f}} {{}}"
    return fmt.format(x, units[idx])

def try_malloc_trim() -> bool:
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        res = libc.malloc_trim(0)
        return res == 1
    except Exception:
        return False

# ---------------------------
# MemoryWatchdog：RSS/USS 监控
# ---------------------------

@dataclass
class WatchdogConfig:
    interval: float = 5.0
    include_children: bool = True
    top_children: int = 0
    csv_path: Optional[str] = None
    logger: Optional[logging.Logger] = None
    log_level: int = logging.INFO
    tag: str = "mem"
    show_uss: bool = True
    show_threads: bool = False

class MemoryWatchdog:
    def __init__(self, config: WatchdogConfig = WatchdogConfig()):
        _ensure_psutil()
        self.cfg = config
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc = psutil.Process(os.getpid())

        self._csv_writer = None
        if self.cfg.csv_path:
            os.makedirs(os.path.dirname(self.cfg.csv_path), exist_ok=True)
            new_file = not os.path.exists(self.cfg.csv_path)
            self._csv_fp = open(self.cfg.csv_path, "a", newline="")
            self._csv_writer = csv.writer(self._csv_fp)
            if new_file:
                self._csv_writer.writerow([
                    "ts", "parent_rss", "parent_uss", "children_rss", "n_children", "n_threads"
                ])
        else:
            self._csv_fp = None

        # Logger
        if self.cfg.logger is None:
            self._logger = logging.getLogger(__name__)
            if not self._logger.handlers:
                handler = logging.StreamHandler(sys.stdout)
                fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
                handler.setFormatter(fmt)
                self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)
        else:
            self._logger = self.cfg.logger

    def start(self) -> "MemoryWatchdog":
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.cfg.interval * 2)
        if self._csv_fp:
            try:
                self._csv_fp.flush()
                self._csv_fp.close()
            except Exception:
                pass

    def _get_mem(self, proc: "psutil.Process") -> Tuple[int, Optional[int]]:
        try:
            info = proc.memory_full_info()
            rss = int(info.rss)
            uss = int(getattr(info, "uss", 0)) or None
            return rss, uss
        except psutil.AccessDenied:
            try:
                rss = proc.memory_info().rss
                return int(rss), None
            except Exception:
                return 0, None
        except Exception:
            return 0, None

    def snapshot(self) -> Dict[str, Any]:
        rss, uss = self._get_mem(self._proc)
        children = self._proc.children(recursive=True) if self.cfg.include_children else []
        child_rss_total = 0
        child_list = []
        for ch in children:
            crss, _ = self._get_mem(ch)
            child_rss_total += crss
            if self.cfg.top_children > 0:
                child_list.append((ch.pid, crss, ch.name()))
        if self.cfg.top_children > 0:
            child_list.sort(key=lambda x: x[1], reverse=True)
            child_list = child_list[: self.cfg.top_children]
        n_threads = self._proc.num_threads() if self.cfg.show_threads else None

        return {
            "ts": _now_ts(),
            "parent_rss": rss,
            "parent_uss": uss,
            "children_rss": child_rss_total,
            "n_children": len(children),
            "n_threads": n_threads,
            "top_children": child_list,
        }

    def _loop(self):
        while not self._stop.is_set():
            snap = self.snapshot()
            rss_f = format_bytes(snap["parent_rss"])
            uss_f = format_bytes(snap["parent_uss"]) if (self.cfg.show_uss and snap["parent_uss"]) else "N/A"
            crss_f = format_bytes(snap["children_rss"])
            parts = [
                f"[{self.cfg.tag}] parent_rss={rss_f}",
                f"parent_uss={uss_f}" if self.cfg.show_uss else None,
                f"child_rss={crss_f}",
                f"children={snap['n_children']}",
            ]
            if self.cfg.show_threads and snap["n_threads"] is not None:
                parts.append(f"threads={snap['n_threads']}")
            line = " ".join(p for p in parts if p)

            self._logger.log(self.cfg.log_level, line)

            if self.cfg.top_children and snap["top_children"]:
                for pid, crss, name in snap["top_children"]:
                    self._logger.log(self.cfg.log_level,
                                     f"[{self.cfg.tag}] child(pid={pid}, name={name}) rss={format_bytes(crss)}")

            if self._csv_writer:
                self._csv_writer.writerow([
                    f"{snap['ts']:.3f}",
                    snap["parent_rss"],
                    snap["parent_uss"] or 0,
                    snap["children_rss"],
                    snap["n_children"],
                    snap["n_threads"] or 0
                ])
                try:
                    self._csv_fp.flush()
                except Exception:
                    pass

            self._stop.wait(self.cfg.interval)

# ---------------------------
# TracemallocProfiler：Python 层内存分配
# ---------------------------

@dataclass
class TracemallocConfig:
    nframe: int = 25              # 调用栈深度
    top: int = 20                 # top N
    group_by: str = "lineno"      # 'lineno' | 'filename' | 'traceback'
    filter_includes: Optional[List[str]] = None  # 仅展示包含这些子串的条目
    filter_excludes: Optional[List[str]] = None  # 排除包含这些子串的条目
    logger: Optional[logging.Logger] = None
    log_level: int = logging.INFO
    tag: str = "tm"

class TracemallocProfiler:
    def __init__(self, config: TracemallocConfig = TracemallocConfig()):
        self.cfg = config
        self._started = False
        self._last_snapshot: Optional["tracemalloc.Snapshot"] = None

        if self.cfg.logger is None:
            self._logger = logging.getLogger(__name__ + ".Tracemalloc")
            if not self._logger.handlers:
                handler = logging.StreamHandler(sys.stdout)
                fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
                handler.setFormatter(fmt)
                self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)
        else:
            self._logger = self.cfg.logger

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.report(tag="final")
        except Exception:
            pass
        self.stop()

    def start(self):
        if not self._started:
            tracemalloc.start(self.cfg.nframe)
            self._started = True
            self._last_snapshot = None

    def stop(self):
        if self._started:
            tracemalloc.stop()
            self._started = False
            self._last_snapshot = None

    def _apply_filters(self, stats: List["tracemalloc.Statistic"]) -> List["tracemalloc.Statistic"]:
        inc = self.cfg.filter_includes
        exc = self.cfg.filter_excludes
        if not inc and not exc:
            return stats
        filtered = []
        for st in stats:
            s = str(st)
            if inc and not any(tok in s for tok in inc):
                continue
            if exc and any(tok in s for tok in exc):
                continue
            filtered.append(st)
        return filtered

    def take_snapshot(self) -> "tracemalloc.Snapshot":
        if not self._started:
            self.start()
        return tracemalloc.take_snapshot()

    def report(self, tag: str = ""):
        snap = self.take_snapshot()
        stats = snap.statistics(self.cfg.group_by)
        stats = self._apply_filters(stats)[: self.cfg.top]
        self._logger.log(self.cfg.log_level, f"=== tracemalloc top ({self.cfg.group_by}) {self.cfg.tag}:{tag} ===")
        for st in stats:
            # st.size 是 bytes
            self._logger.log(self.cfg.log_level, str(st))

        if self._last_snapshot is not None:
            diff = snap.compare_to(self._last_snapshot, self.cfg.group_by)
            diff = self._apply_filters(diff)[: self.cfg.top]
            self._logger.log(self.cfg.log_level, f"=== tracemalloc diff since last {self.cfg.tag}:{tag} ===")
            for d in diff:
                self._logger.log(self.cfg.log_level, str(d))
        self._last_snapshot = snap

# ---------------------------
# 装饰器与便捷函数
# ---------------------------

def tm_profiled(func=None, *, cfg: TracemallocConfig = TracemallocConfig()):
    """
    装饰器：在函数进入/退出时打印 tracemalloc top & diff
    用法：
    @tm_profiled
    def foo(...): ...
    或
    @tm_profiled(cfg=TracemallocConfig(top=10))
    """
    def decorator(f):
        def wrapper(*args, **kwargs):
            prof = TracemallocProfiler(cfg)
            prof.start()
            try:
                return f(*args, **kwargs)
            finally:
                try:
                    prof.report(tag=f.__name__)
                finally:
                    prof.stop()
        return wrapper
    return decorator(func) if func else decorator

def set_malloc_arena_max(n: int = 2):
    """
    在 glibc 平台，可以通过环境变量限制 arena，缓解多线程内存碎片。
    注意：必须在进程启动早期设置才有效。
    """
    os.environ.setdefault("MALLOC_ARENA_MAX", str(n))

def malloc_trim_periodic(every_n: int, counter: int) -> bool:
    """
    每处理 N 次调用一次 malloc_trim，返回本次是否执行了 trim。
    用法：
        if malloc_trim_periodic(50, chunk_idx // chunk_size):
            ...
    """
    if every_n <= 0:
        return False
    if counter % every_n == 0:
        return try_malloc_trim()
    return False
