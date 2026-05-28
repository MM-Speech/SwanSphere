import os
import traceback
import multiprocessing
from multiprocessing import Process, Queue
from queue import Empty
import random
import time

setproctitle_installed = False
try:
    import setproctitle
    setproctitle_installed = True
except:
    setproctitle_installed = False

def chunked_worker(worker_id, map_func, args, results_queue=None, init_ctx_func=None):
    ctx = init_ctx_func(worker_id) if init_ctx_func is not None else None
    for job_idx, arg in args:
        try:
            if not isinstance(arg, tuple) and not isinstance(arg, list):
                arg = [arg]
            if ctx is not None:
                res = map_func(*arg, ctx=ctx)
            else:
                res = map_func(*arg)
            results_queue.put((job_idx, res))
        except:
            traceback.print_exc()
            results_queue.put((job_idx, None))

def chunked_multiprocess_run(
        map_func, args, num_workers=None, ordered=True,
        init_ctx_func=None, q_max_size=1000, multithread=False):
    if multithread:
        from multiprocessing.dummy import Queue, Process
    else:
        from multiprocessing import Queue, Process
    args = zip(range(len(args)), args)
    args = list(args)
    n_jobs = len(args)
    if num_workers is None:
        num_workers = int(os.getenv('N_PROC', os.cpu_count()))
    results_queues = []
    if ordered:
        for i in range(num_workers):
            results_queues.append(Queue(maxsize=q_max_size // num_workers))
    else:
        results_queue = Queue(maxsize=q_max_size)
        for i in range(num_workers):
            results_queues.append(results_queue)
    workers = []
    for i in range(num_workers):
        args_worker = args[i::num_workers]
        p = Process(target=chunked_worker, args=(
            i, map_func, args_worker, results_queues[i], init_ctx_func), daemon=True)
        workers.append(p)
        p.start()
    for n_finished in range(n_jobs):
        results_queue = results_queues[n_finished % num_workers]
        job_idx, res = results_queue.get()
        assert job_idx == n_finished or not ordered, (job_idx, n_finished)
        yield res
    for w in workers:
        w.join()


def multiprocess_glob(pattern, num_workers=None):
    from multiprocessing import dummy
    import glob
    from tqdm import tqdm
    split_pattern = pattern.split("/")
    recursive_depth = 0  # number of recursive depth
    for split in split_pattern:
        if '*' in split:
            recursive_depth += 1
    if recursive_depth == 1:
        paths = glob.glob(pattern)
        return paths
    else:
        dirs = multiprocess_glob('/'.join(split_pattern[:-1]))
        ret = []
        args = [f'{d}/{split_pattern[-1]}' for d in dirs]
        if '*' not in split_pattern[-1]:
            return args
        p = dummy.Pool(num_workers)
        for res in tqdm(p.imap_unordered(glob.glob, args), total=len(args), desc=f"globing {pattern}"):
            ret += res
        return ret


class MultiprocessDataPipe:
    def __init__(self, device='cuda'):
        raise NotImplementedError

    def process(self, *args, **kwargs):
        raise NotImplementedError
    
def data_pipe_worker(data_pipe_cls, init_kwargs, task_queue: Queue, result_queue: Queue, pbar_queue: Queue):
    if setproctitle_installed:
        setproctitle.setproctitle(f'data_pipe_worker:({init_kwargs["worker_id"]}/{init_kwargs["num_workers"]})')
    data_pipe = data_pipe_cls(**init_kwargs)
    while True:
        try:
            job_args = task_queue.get()
            if job_args is None:
                break
            job_id, (args, kwargs) = job_args
            result = data_pipe.process(*args, **kwargs)
            result_queue.put((job_id, result))
        except:
            traceback.print_exc()
            result_queue.put((job_id, None))
        pbar_queue.put(1)

def pbar_worker(pbar_queue: Queue, total=None, desc=None, timeout=None):
    from tqdm import tqdm
    pbar = tqdm(total=total, desc=desc)
    retry = 0
    cnt = 0
    while True:
        try:
            item = pbar_queue.get(timeout=timeout)
            if item is None:
                break
            pbar.update(item)
            cnt += item
            retry = 0
            if total is not None and total > 0 and cnt >= total:
                break
        except Empty:
            retry += 1
            print(f"pbar_queue is empty after {timeout * retry} seconds")

def multiprocess_data_pipe_run(
        data_pipe_cls=MultiprocessDataPipe,
        job_args=[],
        job_kwargs=[],
        total=None,
        cls_init_kwargs={},
        init_shared_data={},
        n_devices=None,
        workers_per_device=1,
        ordered=True,
        desc=None,
        q_max_size=10000,
        use_tqdm=True,
        time_window=None,   # limited rate
        max_jobs_per_time_window=None,
        start_method='spawn',
        daemon=True
    ):
    try:
        ctx = multiprocessing.get_context(start_method) if start_method else multiprocessing.get_context()
    except ValueError:
        ctx = multiprocessing.get_context()

    manager = ctx.Manager()
    shared_dict = manager.dict()
    shared_lock = manager.Lock()
    
    if init_shared_data is not None and len(init_shared_data) > 0:
        shared_dict.update(init_shared_data)

    assert len(job_args) > 0 or len(job_kwargs) > 0 or total
    if len(job_args) > 0 and not (isinstance(job_args[0], list) or isinstance(job_args[0], tuple)):
        job_args = [(job_arg,) for job_arg in job_args]
    if len(job_args) > 0 and len(job_kwargs) > 0:
        assert len(job_args) == len(job_kwargs)
        jobs = [(a, kw) for a, kw in zip(job_args, job_kwargs)]
    elif len(job_args) > 0:
        jobs = [(a, {}) for a in job_args]
    elif len(job_kwargs) > 0:
        jobs = [((), kw) for kw in job_kwargs]
    else:
        jobs = [((), {}) for _ in range(total)]
    jobs = list(enumerate(jobs))

    if n_devices is None or n_devices < 0:
        devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(",")
    elif n_devices == 0:    # cpu
        devices = []
    elif n_devices > 0:
        devices = list(range(n_devices))
    use_cuda = len(devices) > 0

    num_workers = len(devices) * workers_per_device if use_cuda else workers_per_device
    num_workers = max(1, num_workers)

    pbar_queue = ctx.Queue()
    result_queue = ctx.Queue(maxsize=q_max_size)
    task_queue = ctx.Queue(maxsize=q_max_size)

    workers = []
    for i in range(num_workers):
        init_kwargs_ = {
            **cls_init_kwargs, 
            "worker_id": i, 
            "num_workers": num_workers,
            "shared_dict": shared_dict,
            "shared_lock": shared_lock
        }
        if use_cuda:
            init_kwargs_["device"] = f"cuda:{i % len(devices)}"
        p = ctx.Process(target=data_pipe_worker, args=(data_pipe_cls, init_kwargs_, task_queue, result_queue, pbar_queue), daemon=daemon)
        p.start()
        workers.append(p)
    if use_tqdm:
        pbar = ctx.Process(target=pbar_worker, args=(pbar_queue, len(jobs), desc), daemon=daemon)
        pbar.start()
        
    apply_rate_limit = (time_window is not None and max_jobs_per_time_window is not None)
    start_time = time.time()
    job_cnt_in_window = 0
    
    in_flight_window = max(1, q_max_size)
    sent = 0
    got = 0
    yielded = 0
    next_id = 0
    buffer = {}
    
    try:
        while got < len(jobs):
            while sent - got < in_flight_window and sent < len(jobs):
                if apply_rate_limit:
                    now = time.time()
                    elapsed = now - start_time
                    if job_cnt_in_window >= max_jobs_per_time_window and elapsed < time_window:
                        time.sleep(time_window - elapsed)
                        start_time = time.time()
                        job_cnt_in_window = 0
                        
                task_queue.put(jobs[sent])
                sent += 1
                job_cnt_in_window += 1
                
            job_id, result = result_queue.get()
            got += 1
            if ordered:
                buffer[job_id] = result
                while next_id in buffer:
                    res = buffer.pop(next_id)
                    yielded += 1
                    next_id += 1
                    yield res
            else:
                yield result
            
        for i in range(num_workers):
            task_queue.put(None)

    except KeyboardInterrupt:
        for i in range(num_workers):
            try:
                task_queue.put_nowait(None)
            except Exception:
                pass
        raise
    
    finally:
        try:
            pbar_queue.put(None)
        except Exception:
            pass
        
        for p in workers:
            if p.is_alive():
                p.join()
                
        if use_tqdm and pbar.is_alive():
            pbar.terminate()
            
        if ordered:
            while not result_queue.empty():
                job_id, result = result_queue.get()
                buffer[job_id] = result
            while next_id in buffer:
                yield buffer.pop(next_id)
                next_id += 1
        else:
            while not result_queue.empty():
                job_id, result = result_queue.get()
                yield result
                
        if use_tqdm and pbar.is_alive():
            pbar.join()
        os.system('stty echo')
    


class DummyDataPipe(MultiprocessDataPipe):
    def __init__(self, device='cuda', fuck=True, **kwargs):
        pass
    
    def process(self, a, b):
        import time
        time.sleep(random.random())
        return a + b

if __name__ == '__main__':
        
    jobs = [(i, i) for i in range(1000)]

    results = []
    for result in multiprocess_data_pipe_run(
        DummyDataPipe, 
        job_args=jobs, 
        cls_init_kwargs={'fuck': False}, 
        desc='fuck', 
        ordered=False,
        workers_per_device=50,
        q_max_size=100
    ):
        results.append(result)
    
    print(results[:100])