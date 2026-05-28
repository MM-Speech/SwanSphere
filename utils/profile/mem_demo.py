from utils.profile.mem import (
    WatchdogConfig, MemoryWatchdog,
    TracemallocConfig, TracemallocProfiler,
    malloc_trim_periodic, set_malloc_arena_max
)

# 尽早（进程启动时）设置，缓解多线程 glibc arena 膨胀
set_malloc_arena_max(2)

# 启动内存监控（日志 + 可选 CSV）
wd = MemoryWatchdog(WatchdogConfig(
    interval=5.0,
    include_children=True,
    top_children=3,             # 打印最占内存的 3 个子进程
    csv_path=None,              # 需要落盘则给路径，比如 "/tmp/mem.csv"
    show_uss=True,
    show_threads=True,
    tag="watchdog",
)).start()

# 启动 tracemalloc（可带过滤，仅关注你的项目路径）
tm = TracemallocProfiler(TracemallocConfig(
    nframe=25,
    top=20,
    group_by="lineno",
    filter_includes=[ "/mnt/bn/sa-ag-data/liruiqi/code/megaavatar_dataprocess_lrq" ],
    # filter_excludes=[ "site-packages" ],
    tag="tm",
))
tm.start()

# 你的主循环中，周期性打印 diff + 尝试归还内存
if (chunk_idx // chunk_size) % 20 == 0:
    tm.report(tag=f"chunk={chunk_idx}")
    if malloc_trim_periodic(every_n=50, counter=(chunk_idx // chunk_size)):
        print("[malloc_trim] called")
