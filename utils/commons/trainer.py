import contextlib
import gc
import random
import tempfile
import threading
import time
import subprocess
import traceback
import socket
import setproctitle
import yaml
from torch import nn
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import torch.optim
import torch.utils.data
import copy
import logging
import os
import gc
import re
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm
import datetime
import atexit
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision, ShardingStrategy,BackwardPrefetch
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.nn.parallel import DistributedDataParallel
from utils.commons.ckpt_utils import get_last_checkpoint, get_all_ckpts, torch_load_dist
from utils.commons.fsdp_utils import get_fsdp_optimizer_states, get_model_states, set_fsdp_optimizer_states
from utils.commons.hparams import hparams
from utils.commons.tensor_utils import move_to_cuda, move_to_cpu
from utils.commons.os_utils import remove_file
from utils.commons.meters import Timer
from utils.commons.io import print_once

os.environ['NCCL_DEBUG'] = 'WARN'

GLOBAL_RANK = 0
LOCAL_RANK = 0

def get_module_to_ignore_mixed_precision():
    try:
        from apex.normalization import FusedLayerNorm

        return [
            torch.nn.GroupNorm,
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            FusedLayerNorm,
        ]
    except:
        return [
            torch.nn.GroupNorm,
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
        ]

def check_port_is_occupied(host='localhost', port=10080):
    s = socket.socket()
    try:
        s.connect((host, port))
        print(f"{host}:{port} is occupied!")
        return True
    except:
        print(f"{host}:{port} is not occupied!")
        return False
    finally:
        s.close()


class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout  # 保存之前的 stdout
        self.closed = False
        sys.stdout = self
        
    def write(self, data):
        try:
            self.file.write(data)
        except Exception:
            pass
        try:
            self.stdout.write(data)
        except Exception:
            pass
        
    def flush(self):
        try:
            self.file.flush()
        except Exception:
            pass
        try:
            self.stdout.flush()
        except Exception:
            pass
        
    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if hasattr(self, 'stdout'):
                sys.stdout = self.stdout
        except Exception:
            pass
        try:
            if hasattr(self, 'file'):
                self.file.close()
        except Exception:
            pass
        
    def __del__(self):
        # 析构里只调用安全的 close
        try:
            self.close()
        except Exception:
            pass


def save_checkpoint(checkpoint, filepath, save_to_tos):
    tmp_path = f'{filepath}.ckpt.tmp'
    torch.save(checkpoint, tmp_path, _use_new_zipfile_serialization=True)
    subprocess.check_call(f'mv "{tmp_path}" "{filepath}"', shell=True)
    save_ckpt_tos_dir = hparams.get('save_ckpt_tos_dir', '')
    if save_ckpt_tos_dir != '' and save_to_tos:
        print(f"| saving local ckpt {filepath} to tos path: {save_ckpt_tos_dir}/{filepath}")
        from utils.commons.tos_utils import put_object
        put_object(f'{save_ckpt_tos_dir}/{filepath}', open(filepath, 'rb').read())


class Trainer:
    def __init__(
            self,
            work_dir,
            default_save_path=None,
            accumulate_grad_batches=1,
            max_updates=160000,
            print_nan_grads=False,
            val_check_interval=2000,
            num_sanity_val_steps=5,
            amp=False,
            # tb logger
            log_save_interval=100,
            tb_log_interval=10,
            # checkpoint
            monitor_key='val_loss',
            monitor_mode='min',
            num_ckpt_keep=5,
            save_best=True,
            resume_from_checkpoint=0,
            resume_from="",
            seed=1234,
            debug=False,
            profile=False
    ):
        os.makedirs(work_dir, exist_ok=True)
        self.work_dir = work_dir
        self.accumulate_grad_batches = accumulate_grad_batches
        self.max_updates = max_updates
        self.num_sanity_val_steps = num_sanity_val_steps
        self.print_nan_grads = print_nan_grads
        self.default_save_path = default_save_path
        self.resume_from_checkpoint = resume_from_checkpoint if resume_from_checkpoint != 0 else None
        self.resume_from = resume_from
        self.seed = seed
        self.debug = debug
        self.profiler = None
        self.profile = profile
        # model and optm
        self.task = None
        self.optimizers = []

        # trainer state
        self.testing = False
        self.global_step = 0
        self.current_epoch = 0
        self.total_batches = 0

        # configure checkpoint
        self.monitor_key = monitor_key
        self.num_ckpt_keep = num_ckpt_keep
        self.save_best = save_best

        # allow int, string and gpu list
        self.all_gpu_ids = [
            int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x != '']
        hparams['sp_size'] = min(hparams.get('sp_size', 1), len(self.all_gpu_ids))

        self.use_multi_machine_ddp = hparams['world_size'] != -1
        self.num_local_gpus = len(self.all_gpu_ids)  # if world_size is not -1, multi-machine setting
        self.num_total_gpus = len(self.all_gpu_ids) if not self.use_multi_machine_ddp else hparams[
            'world_size']  # if world_size is not -1, multi-machine setting
        # self.num_gpus = len(self.all_gpu_ids)
        self.on_gpu = self.num_local_gpus > 0
        self.root_gpu = 0
        self.device = f'cuda:{self.root_gpu}'
        logging.info(
            f'GPU available: {torch.cuda.is_available()}, GPU used: {self.all_gpu_ids}, world_size: {self.num_total_gpus}, multi-machine training: {self.use_multi_machine_ddp}')
        self.use_ddp = self.num_local_gpus > 1 or self.use_multi_machine_ddp
        self.proc_rank = 0
        self.proc_rank_local = 0
        # Tensorboard logging
        self.log_save_interval = log_save_interval
        self.val_check_interval = val_check_interval
        self.tb_log_interval = tb_log_interval
        self.precision = hparams.get('mixed_precision', 'fp16')
        self.comm_grad_bf16 = hparams.get('comm_grad_bf16', False)
        if self.precision == 'fp16':
            self.precision = torch.float16
        else:
            self.precision = torch.bfloat16
        self.amp = amp
        self.use_fsdp = hparams.get('use_fsdp', False)
        self.amp_scalar = None
        if self.amp and self.precision == torch.float16:
            if self.use_fsdp:
                self.amp_scalar = ShardedGradScaler()
            else:
                self.amp_scalar = GradScaler()
        print("| Init trainer v2.")

    def test(self, task_cls):
        self.testing = True
        self.fit(task_cls)

    def fit(self, task_cls):
        try:
            if self.use_ddp:
                assert hparams.get('sp_size', 1) <= self.num_local_gpus
                mp.start_processes(
                    self.ddp_run, nprocs=self.num_local_gpus, args=(task_cls, copy.deepcopy(hparams)),
                    start_method='spawn')
            else:
                self.task = task_cls()
                self.task.trainer = self
                setproctitle.setproctitle(f'MegaAvatar_worker ({hparams["work_dir"]})')
                self.run_single_process(self.task)
        except:
            traceback.print_exc()
            time.sleep(5)
            if self.proc_rank == 0:
                if len(hparams["exp_name"]) > 5 and not self.testing and hparams.get('pkill_at_end', True):
                    subprocess.check_call(f'pkill -f -9 "{hparams["exp_name"]}"', shell=True)
        return 1

    def ddp_run(self, gpu_idx, task_cls, hparams_):
        hparams.update(hparams_)
        setproctitle.setproctitle(f'MegaAvatar_worker ({hparams_["work_dir"]}_gpu{gpu_idx})')

        if hparams.get('use_file_system_mp', False):
            torch.multiprocessing.set_sharing_strategy('file_system')
        if hparams.get('use_fork', True):
            torch.multiprocessing.set_start_method('fork', force=True)
        self.proc_rank_local = self.root_gpu = gpu_idx
        self.proc_rank = gpu_idx
        global GLOBAL_RANK
        global LOCAL_RANK
        if self.use_multi_machine_ddp:
            start_rank = int(hparams['start_rank'])
            if start_rank == -1:
                if 'ARNOLD_WORKER_GPU' not in os.environ:
                    os.environ['ARNOLD_WORKER_GPU'] = os.environ['ARNOLD_EXECUTOR_GPU']
                gpus_per_worker = int(os.environ['ARNOLD_WORKER_GPU'])
                start_rank = int(os.environ['ARNOLD_ID']) * gpus_per_worker
            start_rank = max(start_rank, 0)
            self.proc_rank = gpu_idx + start_rank
            hparams['start_rank'] = start_rank
        GLOBAL_RANK = self.proc_rank
        os.environ['RANK'] = str(self.proc_rank)
        LOCAL_RANK = self.proc_rank_local
        os.environ['LOCAL_RANK'] = str(self.proc_rank_local)
        os.environ['WORLD_SIZE'] = str(self.num_total_gpus)
        if hparams['init_method'] == 'file':
            self.init_ddp_connection_file(self.proc_rank, self.num_total_gpus)
        elif hparams['init_method'] == 'tcp':
            self.init_ddp_connection_tcp(self.proc_rank, self.num_total_gpus)
        else:
            raise NotImplementedError()

        if gpu_idx != 0 and not self.debug:
            sys.stdout = open(os.devnull, "w")
            # sys.stderr = open(os.devnull, "w")
        dist.barrier()

        task = task_cls()
        task.trainer = self
        torch.cuda.set_device(gpu_idx)
        self.device = f'cuda:{self.root_gpu}'
        self.task = task
        self.run_single_process(task)

    def build_model_optms(self):
        task = self.task
        build_model_res = task.build_model()
        map_location = self.device if hparams.get('load_map_to_gpu', False) else 'cpu'
        mmap = hparams.get('mmap', None)
        if self.testing:
            ckpt_path = f'{self.work_dir}/model_only_last.ckpt'
            if os.path.exists(ckpt_path):
                checkpoint = torch_load_dist(ckpt_path, map_location=map_location, mmap=mmap)
                global_step_ckpt = checkpoint['global_step']
                checkpoint_model = checkpoint.pop('state_dict')
            else:
                checkpoint_model, ckpt_path, global_step_ckpt = \
                    get_last_checkpoint(
                        self.work_dir, self.resume_from_checkpoint, map_location=map_location, mmap=mmap,
                        return_step=True)
        else:
            checkpoint, ckpt_path, global_step_ckpt = \
                get_last_checkpoint(self.work_dir, self.resume_from_checkpoint, map_location=map_location, mmap=mmap,
                                    return_step=True)
            if self.resume_from != "" and checkpoint is None:
                checkpoint, ckpt_path, global_step_ckpt = \
                    get_last_checkpoint(self.resume_from, self.resume_from_checkpoint,
                                        map_location=map_location, mmap=mmap, return_step=True)
            if checkpoint is not None and 'state_dict' in checkpoint:
                checkpoint_model = checkpoint.pop('state_dict')
            else:
                checkpoint_model = checkpoint
        if checkpoint_model is not None:
            # load training state (affects trainer only)
            self.global_step = global_step_ckpt
            self.task.global_step = self.global_step

            print(f'| resume training from checkpoint {ckpt_path}')
            if "backbone" in ckpt_path:
                new_checkpoint_model = {"backbone": checkpoint_model}
                self.restore_weights(new_checkpoint_model)
            else:
                self.restore_weights(checkpoint_model)
            del checkpoint_model
            if self.on_gpu:
                task.to(self.device)
        else:
            if self.on_gpu:
                task.to(self.device)
            self.task.load_model()
        # clear cache after restore
        if self.on_gpu:
            torch.cuda.empty_cache()

        self.other_module_names = []

        def modules2names(ms):
            return [n for n, m in self.task.named_children() if m in ms]

        if isinstance(build_model_res, dict):
            if 'others' in build_model_res:
                self.other_module_names = modules2names(build_model_res['others'])
            self.training_module_names = modules2names(build_model_res['trainable'])
            print(f"| <build_model> returns dict. trainable modules: {self.training_module_names}, "
                  f"other modules: {self.other_module_names}")
        else:
            if build_model_res is not None and not isinstance(build_model_res, (list, tuple)):
                build_model_res = [build_model_res]
            if build_model_res is None or len(build_model_res) == 0:
                build_model_res = [n for n, _ in self.task.named_children()]
                print(f"| <build_model> returns None, all modules to save: {build_model_res}")
            elif isinstance(build_model_res[0], nn.Module):
                build_model_res = [n for n, m in self.task.named_children() if m in build_model_res]
                print(f"| <build_model> returns module objects. Specific modules to save: {build_model_res}")
            elif isinstance(build_model_res[0], str):
                print(f"| <build_model> returns a str list. Specific modules to save: {build_model_res}")
            self.training_module_names = build_model_res
        if self.use_ddp:
            if self.use_fsdp:
                fsdp_ignored_modules = self.task.fsdp_ignored_modules() \
                    if hasattr(self.task, 'fsdp_ignored_modules') else None
                fsdp_wrap_policy = self.task.fsdp_wrap_policy() \
                    if hasattr(self.task, 'fsdp_wrap_policy') else None
            else:
                fsdp_ignored_modules = None
                fsdp_wrap_policy = None
            for n in self.training_module_names:
                m = getattr(self.task, n)
                setattr(self.task, n, self.configure_ddp(m, fsdp_wrap_policy, fsdp_ignored_modules))
            for n in self.other_module_names:
                m = getattr(self.task, n)
                m.to(self.device)
        if dist.is_initialized():
            dist.barrier()
        if not self.testing:
            self.optim2model = {}
            self.optimizers = task.configure_optimizers()
            self.fisrt_epoch = True
            if checkpoint is not None:
                if 'optimizer_states' in checkpoint:
                    checkpoint_optm = checkpoint.pop('optimizer_states', None)
                else:
                    if os.path.exists(ckpt_path[:-5] + '_optm.ckpt'):
                        checkpoint_optm = torch_load_dist(
                            ckpt_path[:-5] + '_optm.ckpt', map_location=map_location, mmap=mmap)
                    else:
                        checkpoint_optm = None
                if checkpoint_optm is not None:
                    self.restore_opt_state(checkpoint_optm)
                    del checkpoint_optm
            del checkpoint

    def run_single_process(self, task):
        """Sanity check a few things before starting actual training.

        :param task:
        """
        # build model, optm and load checkpoint
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision(hparams.get('matmul_precision', 'high'))  # 'high'/'medium'
        except Exception:
            pass
        torch.backends.cudnn.benchmark = hparams.get('cudnn_benchmark', False)
        
        if self.proc_rank == 0:
            self.save_terminal_logs()
            if not self.testing:
                self.save_codes()
        random.seed(self.seed)
        np.random.seed(self.seed)
        try:
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        except:
            traceback.print_exc()

        self.build_model_optms()
        gc.collect()
        task.trainer = self
        task.testing = self.testing
        # link up experiment object
        if self.proc_rank == 0 and not self.testing:
            if hparams.get('wandb', False):
                import wandb
                wandb.init(project='MegaAvatar', name=hparams['exp_name'],
                           id=hparams['exp_name'],
                           dir='wandb_logs',
                           sync_tensorboard=True, config=hparams)
            task.build_tensorboard(save_dir=self.work_dir, name='tb_logs')
        else:
            os.makedirs('tmp', exist_ok=True)
            task.build_tensorboard(save_dir='tmp', name='tb_tmp')
        self.logger = task.logger
        if hparams.get('pkill_at_end', True):
            try:
                if self.testing:
                    self.run_evaluation(test=True)
                else:
                    self.train()
            except:
                traceback.print_exc()
                task.on_keyboard_interrupt()
                time.sleep(5)
                if self.proc_rank == 0:
                    if len(hparams["exp_name"]) > 5:
                        subprocess.check_call(f'pkill -f -9 "{hparams["exp_name"]}"', shell=True)
            finally:
                if hasattr(self, '_tee'):
                    self._tee.close()
        else:
            if self.testing:
                self.run_evaluation(test=True)
            else:
                self.train()

    ####################
    # valid and test
    ####################
    def run_evaluation(self, test=False):
        eval_results = self.evaluate(
            self.task, test, tqdm_desc='Valid' if not test else 'test',
            max_batches=hparams['eval_max_batches'] if not test else
            hparams.get('test_max_batches', hparams['eval_max_batches']))
        if eval_results is not None and 'tb_log' in eval_results:
            tb_log_output = eval_results['tb_log']
            self.log_metrics_to_tb(tb_log_output)
        for optm in self.optimizers:
            from torch.distributed.optim import ZeroRedundancyOptimizer
            if isinstance(optm, ZeroRedundancyOptimizer):
                optm.consolidate_state_dict(to=0)
        if (self.proc_rank == 0 or self.use_fsdp) and not test:
            self.save_checkpoint(epoch=self.current_epoch, logs=eval_results)

    def evaluate(self, task, test=False, tqdm_desc='Valid', max_batches=None):
        if max_batches == 0:
            return
        if max_batches == -1:
            max_batches = None
        # enable eval mode
        task.zero_grad()
        task.eval()
        torch.set_grad_enabled(False)

        ctx_managers = []
        if not self.use_fsdp:
            for n in self.training_module_names:
                m = getattr(self.task, n)
                if isinstance(m, DistributedDataParallel):
                    ctx_managers.append(m.no_sync())
        with contextlib.ExitStack() as stack, torch.no_grad():
            for mgr in ctx_managers:
                stack.enter_context(mgr)

            task_ref = self.get_task_ref()
            if test:
                ret = task_ref.test_start()
                if ret == 'EXIT':
                    return
            else:
                task_ref.validation_start()
            outputs = []
            dataloader = task_ref.test_dataloader() if test else task_ref.val_dataloader()
            pbar = tqdm.tqdm(dataloader, desc=tqdm_desc, total=max_batches, dynamic_ncols=True, unit='step',
                             disable=self.root_gpu > 0)
            # give model a chance to do something with the outputs (and method defined)
            for batch_idx, batch in enumerate(pbar):
                if batch is None:  # pragma: no cover
                    continue
                # stop short when on fast_dev_run (sets max_batch=1)
                if max_batches is not None and batch_idx >= max_batches:
                    break

                # make dataloader_idx arg in validation_step optional
                if self.on_gpu:
                    batch = move_to_cuda(batch, self.root_gpu)
                args = [batch, batch_idx]

                if self.use_fsdp:
                    ctx_managers = []
                    for n in self.training_module_names:
                        if n == "tea_model" or "ema": continue # tea不需要unshard,即FSDP的参数聚集
                        m = getattr(self.task, n)
                        if isinstance(m, FullyShardedDataParallel):
                            ctx_managers.append(
                                m.summon_full_params(m, rank0_only=True, writeback=False, with_grads=False,offload_to_cpu=True)) # 训练过程中临时将分片的参数聚集起来，以便进行前向传播或计算,offload_to_cpu可以减少大模型OOM风险
                    with contextlib.ExitStack() as stack:
                        for mgr in ctx_managers:
                            stack.enter_context(mgr)
                        with autocast(dtype=self.precision, enabled=self.amp):
                            output = task(*args)
                else:
                    with autocast(dtype=self.precision, enabled=self.amp):
                        output = task(*args)
                outputs.append(output)
            # give model a chance to do something with the outputs (and method defined)
            if test:
                eval_results = task_ref.test_end(outputs)
            else:
                eval_results = task_ref.validation_end(outputs)
        # enable train mode again
        task.train()
        torch.set_grad_enabled(True)
        return eval_results

    ####################
    # train
    ####################
    def train(self):
        task_ref = self.get_task_ref()
        task_ref.on_train_start()
        dataloader = task_ref.train_dataloader()
        if self.num_sanity_val_steps > 0:
            # run tiny validation (if validation defined) to make sure program won't crash during val
            num_sanity_val_steps = self.num_sanity_val_steps
            if hparams['eval_max_batches'] != -1:
                num_sanity_val_steps = min(self.num_sanity_val_steps, hparams['eval_max_batches'])
            self.evaluate(self.task, False, 'Sanity Val', max_batches=num_sanity_val_steps)
        # clear cache before training
        if self.on_gpu:
            torch.cuda.empty_cache()
        epoch = self.current_epoch
        tb_metrics_avg = {}
        # Profiler step
        if self.profile:
            self.configure_profilers()

        # Start the profiler.
        if self.profiler:
            self.profiler.start()

        # run all epochs
        while True:
            # set seed for distributed sampler (enables shuffling for each epoch)
            if self.use_ddp and hasattr(dataloader.sampler, 'set_epoch'):
                dataloader.sampler.set_epoch(epoch)
            # update training progress in trainer and model
            task_ref.current_epoch = epoch
            self.current_epoch = epoch
            # total batches includes multiple val checks
            self.batch_loss_value = 0  # accumulated grads
            # before epoch hook
            task_ref.on_epoch_start()

            # run epoch
            train_pbar = tqdm.tqdm(dataloader, initial=self.global_step, total=float('inf'),
                                   dynamic_ncols=True, unit='step', disable=self.root_gpu > 0)
            gc.disable()
            gc.collect()
            # for batch_idx, batch in enumerate(train_pbar):
            for batch_idx, batch in enumerate(train_pbar):
                if self.global_step % self.val_check_interval == 0 and not self.fisrt_epoch:
                    self.run_evaluation()
                pbar_metrics, tb_metrics = self.run_training_batch(batch_idx, batch)
                if self.profiler:
                    self.profiler.step()
                train_pbar.set_postfix(**pbar_metrics)
                self.fisrt_epoch = False
                scalar_metrics = self.metrics_to_scalars(tb_metrics)
                for k, v in scalar_metrics.items():
                    if k not in tb_metrics_avg:
                        tb_metrics_avg[k] = 0
                    tb_metrics_avg[k] += v

                # when metrics should be logged
                if self.tb_log_interval > 0 and (self.global_step + 1) % self.tb_log_interval == 0:
                    # logs user requested information to logger
                    for k, v in tb_metrics_avg.items():
                        tb_metrics_avg[k] = v / self.tb_log_interval
                    self.log_metrics_to_tb(tb_metrics_avg)
                    tb_metrics_avg = {}

                if self.global_step > self.max_updates + 1:
                    self.run_evaluation()
                    print("| Training end..")
                    break

                self.global_step += 1

                # Do garbage collection.
                # FIXME: extremely slow
                if hparams.get('gc_every_n_steps', -1) > 0:
                    if self.global_step % hparams.get('gc_every_n_steps', -1) == 0:
                        gc.collect()

                task_ref.global_step = self.global_step

            # epoch end hook
            epoch_loss_dict = task_ref.on_epoch_end()
            self.log_metrics_to_tb(epoch_loss_dict)
            epoch += 1
            if self.global_step > self.max_updates + 1:
                break
        # End the profiler.
        if self.profiler:
            self.profiler.stop()
        task_ref.on_train_end()

    def run_training_batch(self, batch_idx, batch):
        if batch is None:
            return {}
        all_progress_bar_metrics = []
        all_log_metrics = []
        task_ref = self.get_task_ref()
        for opt_idx, optimizer in enumerate(self.optimizers):
            if optimizer is None:
                continue
            # make sure only the gradients of the current optimizer's paramaters are calculated
            # in the training step to prevent dangling gradients in multiple-optimizer setup.
            if len(self.optimizers) > 1:
                for k, param in task_ref.named_parameters():
                    param.requires_grad = False
                for group in optimizer.param_groups:
                    for param in group['params']:
                        param.requires_grad = True
            should_sync = (self.global_step + 1) % self.accumulate_grad_batches == 0 or \
                          (self.use_fsdp and self.num_total_gpus > 0)
            ctx_managers = []
            if not should_sync and not self.use_fsdp:
                for n in self.training_module_names:
                    m = getattr(self.task, n)
                    if isinstance(m, DistributedDataParallel):
                        ctx_managers.append(m.no_sync())
            with (contextlib.ExitStack() if not should_sync else contextlib.nullcontext()) as stack:
                for mgr in ctx_managers:
                    stack.enter_context(mgr)
                with Timer("forward_training_step", enable=False):
                    # with Timer("forward_training_step", enable=self.debug):
                    with autocast(dtype=self.precision, enabled=self.amp):
                        if self.on_gpu:
                            if len(self.optimizers) > 1:
                                batch = move_to_cuda(copy.copy(batch), self.root_gpu)
                            else:
                                batch = move_to_cuda(batch, self.root_gpu)
                        args = [batch, batch_idx, opt_idx]
                        output = self.task(*args)
                        loss = output['loss']
                        if loss is None:
                            continue
                        progress_bar_metrics = output['progress_bar']
                        log_metrics = output['tb_log']
                        # accumulate loss
                        loss = loss / self.accumulate_grad_batches

                # backward pass
                with Timer("backward_training_step", enable=False):
                    # with Timer("backward_training_step", enable=self.debug):
                    if loss.requires_grad:
                        if self.amp and self.amp_scalar is not None:
                            self.amp_scalar.scale(loss).backward()
                        else:
                            # print(f"{loss =}. {output =}")
                            loss.backward()
                    # track progress bar metrics
                    all_log_metrics.append(log_metrics)
                    all_progress_bar_metrics.append(progress_bar_metrics)

            # nan grads
            with Timer("checkNan_training_step", enable=False):
                # with Timer("checkNan_training_step", enable=self.debug):
                has_nan_grad = False
                nan_params_names = []
                if self.print_nan_grads:
                    for name, param in task_ref.named_parameters():
                        if (param.grad is not None) and torch.isnan(param.grad.float()).any():
                            # print("| NaN grad params: ", name)
                            has_nan_grad = True
                            nan_params_names.append(name)
                    if has_nan_grad:
                        # exit(0)
                        print(f"| WARN: found nan in grad! first nan params: {nan_params_names[0]}; last nan params: {nan_params_names[-1]}.")
                        optimizer.zero_grad()
                        task_ref.on_after_optimization(self.current_epoch, batch_idx, optimizer, opt_idx)

            # gradient update with accumulated gradients
            with Timer("optimUpdate_training_step", enable=False):
                # with Timer("optimUpdate_training_step", enable=self.debug):
                if (self.global_step + 1) % self.accumulate_grad_batches == 0 and not has_nan_grad:
                    # if (self.global_step + 1) % self.accumulate_grad_batches == 0:
                    # Unscales the gradients of optimizer's assigned params in-place
                    if self.amp and self.amp_scalar is not None:
                        self.amp_scalar.unscale_(optimizer)
                    
                    grad_norm_dict = task_ref.on_before_optimization(opt_idx)
                    if grad_norm_dict is not None:
                        all_log_metrics[-1].update(grad_norm_dict)
                        # all_progress_bar_metrics[-1].update(grad_norm_dict)

                    if self.task.gradient_clip_norm > 0 or self.task.gradient_clip_val > 0:
                        for n in self.training_module_names:
                            m = getattr(self.task, n)
                            if self.task.gradient_clip_norm > 0:
                                if isinstance(m, FullyShardedDataParallel):
                                    grad_norm = m.clip_grad_norm_(self.task.gradient_clip_norm)
                                else:
                                    torch.nn.utils.clip_grad_norm_(m.parameters(), self.task.gradient_clip_norm)
                            if self.task.gradient_clip_val > 0:
                                assert not isinstance(m, FullyShardedDataParallel)
                                torch.nn.utils.clip_grad_value_(m.parameters(), self.task.gradient_clip_val)
                    
                    skip_step = False
                    if (
                            (grad_norm_skip_threshold := hparams.get('grad_norm_skip_threshold')) is not None and
                            grad_norm_dict is not None
                        ):
                        if (
                                (grad_norm_skip_init_threshold := hparams.get('grad_norm_skip_init_threshold')) is not None and
                                (grad_norm_skip_warmup_steps := hparams.get('grad_norm_skip_warmup_steps')) is not None and
                                grad_norm_skip_init_threshold > grad_norm_skip_threshold and
                                grad_norm_skip_warmup_steps > self.global_step
                            ):
                            grad_norm_skip_threshold = (
                                - (grad_norm_skip_init_threshold - grad_norm_skip_threshold) / max(1, grad_norm_skip_warmup_steps) * (self.global_step + 1) + 
                                grad_norm_skip_init_threshold
                            )
                            print(f"{grad_norm_skip_threshold = }; grad_norm@opt{opt_idx} = {list(grad_norm_dict.values())[0]}")
                        skip_step = list(grad_norm_dict.values())[0] > grad_norm_skip_threshold
                    if skip_step:
                        if self.proc_rank_local == 0:
                            print(f"| Grad Norm of optm{opt_idx} = {list(grad_norm_dict.values())[0]} > {grad_norm_skip_threshold}, skipping step.")
                    else:
                        if self.amp and self.amp_scalar is not None:
                            self.amp_scalar.step(optimizer)
                            self.amp_scalar.update()
                        else:
                            optimizer.step()
                    optimizer.zero_grad()
                    task_ref.on_after_optimization(self.current_epoch, batch_idx, optimizer, opt_idx)

        # collapse all metrics into one dict
        all_progress_bar_metrics = {k: v for d in all_progress_bar_metrics for k, v in d.items()}
        all_log_metrics = {k: v for d in all_log_metrics for k, v in d.items()}
        return all_progress_bar_metrics, all_log_metrics

    def configure_profilers(self):
        # Set scheduler.
        wait = 0
        active = 1
        schedule = torch.profiler.schedule(wait=wait, warmup=10, active=active, repeat=1)
        print("| configure self.profiler")

        # Set profiler.
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=schedule,
            with_stack=True,
            record_shapes=False,
            with_modules=True,
            profile_memory=False,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'{self.work_dir}/profiler'),
            # on_trace_ready=trace_handler,
        )

    ####################
    # load and save checkpoint
    ####################
    def restore_weights(self, checkpoint):
        # load model state
        task_ref = self.get_task_ref()
        for k, state_dict in checkpoint.items():
            if k in hparams.get('not_save_modules', []) and k != "tea_model":
                continue
            if not hasattr(task_ref, k):
                continue
            state_dict = {
                k.replace('_orig_mod.', ''): v for k, v in state_dict.items()
            }
            try:
                getattr(task_ref, k).load_state_dict({
                    k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()
                })
                print(f"| load {k} from last ckpt.")
            except:
                try:
                    getattr(task_ref, k).load_state_dict(state_dict, strict=False)
                    print(f"| load {k} from last ckpt.")
                except:
                    traceback.print_exc()
                    print(f"| WARMING: {k} parameters not match !!!")

        self.load_ema(task_ref, checkpoint)
        
        if hparams.get("use_tea_distill", False): # tea和stu load同一套权重
            self.load_tea_model(task_ref, checkpoint)

        if hparams.get("distill_strategy", "") == "gan" and hparams.get("disc_backbone") == "DiT":
            self.load_disc_model(task_ref, checkpoint)

        # wait for all models to restore weights
        if self.use_ddp:
            # wait for all processes to catch up
            dist.barrier()
    
    def load_tea_model(self, task_ref, checkpoint):
        state_dict = checkpoint['backbone']
        state_dict = {
            k.replace('_orig_mod.', ''): v for k, v in state_dict.items()
        }
        try:
            getattr(task_ref, 'tea_model').load_state_dict({
                k.replace('_orig_mod.', ''): v for k, v in state_dict.items()
            })
            print(f"| load tea_model from last ckpt.")
        except:
            traceback.print_exc()
            print(f"| WARMING: ema parameters not match !!!")

    def load_disc_model(self, task_ref, checkpoint):
        # 提取 checkpoint 中 backbone 部分的参数
        state_dict = checkpoint['backbone']
        # 去除参数名称中可能存在的 '_orig_mod.' 前缀
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        # 获取 disc_model 实例
        disc_model = getattr(task_ref, 'disc_model')
        # 获取当前模型的状态字典
        model_state = disc_model.state_dict()

        # 过滤出 checkpoint 中那些也存在于模型 state_dict 里的参数
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state}

        try:
            # 加载过滤后的参数到 disc_model，忽略模型中缺失的参数（例如disc_heads）
            disc_model.load_state_dict(filtered_state_dict, strict=False)
            print(f"| load disc_model from last ckpt, {len(filtered_state_dict)}/{len(state_dict)}.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("| WARNING: disc_model parameters do not fully match !!!")

    def load_ema(self, task_ref, checkpoint):
        if hasattr(task_ref, 'ema') and 'ema' not in checkpoint.keys():
            state_dict = checkpoint['backbone']
            state_dict = {
                k.replace('_orig_mod.', ''): v for k, v in state_dict.items()
            }
            try:
                getattr(task_ref, 'ema').load_state_dict({
                    k.replace('_orig_mod.', ''): v for k, v in state_dict.items()
                })
                print(f"| load ema from last ckpt.")
            except:
                traceback.print_exc()
                print(f"| WARMING: ema parameters not match !!!")
                
    def _fix_optimizer_state_shapes(self, optimizer):
        # fix 0-dimensional params
        cnt = 0
        for group in optimizer.param_groups:
            for p in group['params']:
                st = optimizer.state.get(p, None)
                if not st:
                    continue
                # Typical keys that need to align shapes
                for key in ('exp_avg', 'exp_avg_sq', 'maximize', 'step'):
                    t = st.get(key, None)
                    if isinstance(t, torch.Tensor):
                        # If the numel is the same but the shape is different (for example, [] vs [1]), rearrange according to the parameter shape
                        if t.numel() == p.numel() and t.shape != p.shape:
                            st[key] = t.reshape(p.shape)
                            cnt += 1
        if cnt > 0:
            print_once(f"| Rearrange [{cnt}] optimizer keys according to numels.")

    def restore_opt_state(self, checkpoint):
        if self.testing:
            return
        # restore the optimizers
        optimizer_states = checkpoint
        for i, (optimizer, opt_state) in enumerate(zip(self.optimizers, optimizer_states)):
            if optimizer is None:
                return
            try:
                if self.use_fsdp:
                    optm2model = self.task.fsdp_optm2model()
                    if i < len(optm2model):
                        set_fsdp_optimizer_states(opt_state, optimizer, optm2model[i])
                    else:
                        optimizer.load_state_dict(opt_state)
                else:
                    optimizer.load_state_dict(opt_state)
                    
                # fix 0-dimensional params
                # self._fix_optimizer_state_shapes(optimizer)
                    
                # move optimizer to GPU 1 weight at a time
                if self.on_gpu:
                    for i, state in enumerate(optimizer.state.values()):
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(self.device)
            except ValueError:
                print("| WARMING: optimizer parameters not match !!!")
        try:
            if dist.is_initialized() and dist.get_rank() > 0:
                return
        except Exception as e:
            print(e)
            return
        did_restore = True
        return did_restore

    def save_checkpoint(self, epoch, logs=None):
        ckpt_path = f'{self.work_dir}/model_ckpt_steps_{self.global_step}.ckpt'
        logging.info(f'Epoch {epoch:05d}@{self.global_step}: saving model to {ckpt_path}')
        self._atomic_save(ckpt_path)
        if self.proc_rank != 0:
            return
        print(f"saved to {ckpt_path}")
        get_ckpt_step_fn = lambda x: int(re.findall('.*steps\_(\d+)[._].*', x)[0])
        for old_ckpt in get_all_ckpts(self.work_dir)[self.num_ckpt_keep:]:
            # leave the milestone ckpts
            step_old = get_ckpt_step_fn(old_ckpt)
            if hparams.get("ckpt_milestone_interval", 10_0000) != 0 and \
                    step_old % hparams.get("ckpt_milestone_interval", 10_0000) == 0:
                pass
            else:
                subprocess.check_call(f'rm -rf {old_ckpt[:-5]}.ckpt {old_ckpt[:-5]}_*.ckpt', shell=True)
                logging.info(f'Delete ckpt: {os.path.basename(old_ckpt[:-5])}')

    def _atomic_save(self, filepath):
        # dump optm state_dict
        if self.use_fsdp:
            checkpoint = self.dump_optm_fsdp()
            dist.barrier()
        else:
            checkpoint = self.dump_optm()

        # dump model state_dict
        state_dict = {}
        dtype_save = torch.float32 if not hparams.get('dtype_save_bf16') else torch.bfloat16
        for n in self.training_module_names + self.other_module_names:
            if n in hparams.get('not_save_modules', []):
                continue
            m = getattr(self.task, n)
            state_dict_ = get_model_states(m)
            if self.proc_rank == 0:
                state_dict[n] = {
                    k.replace('module.', '') if k.startswith('module.') else k:
                        move_to_cpu(copy.deepcopy(v.to(dtype_save))) for k, v in state_dict_.items()
                }
        checkpoint['state_dict'] = state_dict
        if self.use_fsdp:
            dist.barrier()

        if self.proc_rank == 0:
            os.makedirs(hparams['work_dir'], exist_ok=True)
            ckpt_config_path = f"{hparams['work_dir']}/config.yaml"
            with open(ckpt_config_path, 'w') as f:
                yaml.safe_dump(hparams, f)
            save_thread = threading.Thread(
                target=save_checkpoint, args=[checkpoint['state_dict'], filepath, False])
            save_thread.start()
            save_thread = threading.Thread(
                target=save_checkpoint, args=[checkpoint['optimizer_states'], filepath[:-5] + '_optm.ckpt', False])
            save_thread.start()
            if len(checkpoint['state_dict'].keys()) > 1:
                for m, v in checkpoint['state_dict'].items():
                    save_thread = threading.Thread(
                        target=save_checkpoint, args=[v, filepath[:-5] + f'_{m}.ckpt', False])
                    save_thread.start()

    def dump_optm_fsdp(self):
        # # 在获取 OSD 前确保所有 FSDP 模块已初始化
        # for n in self.training_module_names:
        #     m = getattr(self.task, n)
        #     if isinstance(m, FullyShardedDataParallel):
        #         _ensure_fsdp_initialized(m)
                
        checkpoint = {'global_step': self.global_step}
        # save optimizers
        optimizer_states = []
        dist.barrier()
        for i, (optimizer, model) in enumerate(zip(self.optimizers, self.task.fsdp_optm2model())):
            if optimizer is not None:
                # if isinstance(model, FullyShardedDataParallel) and not _optimizer_params_match_model(optimizer, model):
                #     if self.proc_rank == 0:
                #         print(f"[WARN] optimizer {i} includes params not owned by provided FSDP model; "
                #             f"fall back to optimizer.state_dict.")
                #     state_dict = optimizer.state_dict()
                #     state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
                #     state_dict = move_to_cpu(copy.deepcopy(state_dict))
                #     optimizer_states.append(state_dict)
                #     continue
                state_dict = get_fsdp_optimizer_states(optimizer, model)
                if self.proc_rank == 0:
                    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
                    state_dict = move_to_cpu(copy.deepcopy(state_dict))
                    optimizer_states.append(state_dict)
        for i, optimizer in enumerate(self.optimizers[len(self.task.fsdp_optm2model()):]):
            if optimizer is not None and self.proc_rank == 0:
                state_dict = optimizer.state_dict()
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
                state_dict = move_to_cpu(copy.deepcopy(state_dict))
                optimizer_states.append(state_dict)

        checkpoint['optimizer_states'] = optimizer_states
        dist.barrier()
        return checkpoint

    def dump_optm(self):
        checkpoint = {'global_step': self.global_step}
        # save optimizers
        optimizer_states = []
        for i, optimizer in enumerate(self.optimizers):
            if optimizer is not None:
                state_dict = optimizer.state_dict()
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
                state_dict = move_to_cpu(copy.deepcopy(state_dict))
                optimizer_states.append(state_dict)

        checkpoint['optimizer_states'] = optimizer_states
        return checkpoint

    ####################
    # DDP
    ####################
    def configure_ddp(self, m, fsdp_wrap_policy=None, fsdp_ignored_modules=[]):
        if self.use_fsdp and m not in fsdp_ignored_modules:
            m = FullyShardedDataParallel(
                m,
                process_group=None,
                forward_prefetch=True,
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                limit_all_gathers=True,
                use_orig_params=True,
                sync_module_states=False,
                mixed_precision=MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.float32,
                    keep_low_precision_grads=False,
                    cast_forward_inputs=True,
                    cast_root_forward_inputs=True,
                    _module_classes_to_ignore=get_module_to_ignore_mixed_precision(),
                ),
                auto_wrap_policy=fsdp_wrap_policy,
                ignored_modules=fsdp_ignored_modules,
                sharding_strategy=ShardingStrategy.HYBRID_SHARD if hparams.get("sharding_strategy", "hybrid") == "hybrid" else ShardingStrategy.FULL_SHARD,
                device_id=self.device,
            )
            torch.cuda.empty_cache()
            gc.collect()
        else:
            n_learnable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
            if n_learnable_params > 0:
                m = DistributedDataParallel(m, device_ids=[self.root_gpu],
                                            find_unused_parameters=hparams.get('find_unused_parameters', True))
        return m

    def init_ddp_connection_file(self, proc_rank, world_size):
        """
        use a shared file in the network file system to bind all process
        if you found num_worker is larger than world_size, remove the old shard_file_name
        """
        exp_name = hparams['exp_name']
        shared_file_name = hparams.get('ddp_dir', '/mnt/bn/sa-ag-data/yezhenhui/nfs/pytorch_ddp_sharedfile')
        shared_file_name = f'file://{shared_file_name}/{exp_name}'
        os.makedirs(os.path.dirname(shared_file_name).replace("file://", ""), exist_ok=True)
        dist.init_process_group(
            backend='nccl',
            init_method=shared_file_name,
            world_size=world_size,
            rank=proc_rank,
            timeout=datetime.timedelta(minutes=60)   # ← 这里设置更长，比如 60 分钟
        )

    def init_ddp_connection_tcp(self, proc_rank, world_size):
        if hparams.get('master_addr', '') == '':
            if self.use_multi_machine_ddp:
                if 'ARNOLD_WORKER_HOSTS' not in os.environ:
                    os.environ['ARNOLD_WORKER_HOSTS'] = os.environ['ARNOLD_EXECUTOR_HOSTS']
                x = os.environ['ARNOLD_WORKER_HOSTS'].split(",")[0]
                port = x.split(":")[-1]
                root_node = x[:-len(port) - 1]
            else:
                root_node = '127.0.0.1'
            root_node = self.resolve_root_node_address(root_node)
            os.environ['MASTER_ADDR'] = root_node
        else:
            os.environ['MASTER_ADDR'] = hparams['master_addr']
        os.environ['MASTER_PORT'] = str(hparams.get('master_port','6668'))
        print("| use master addr: ", os.environ['MASTER_ADDR']," use master port: ", os.environ['MASTER_PORT'])
        dist.init_process_group(
            'nccl', timeout=datetime.timedelta(seconds=7200000),  # was 1800000
            rank=proc_rank, world_size=world_size)

    def resolve_root_node_address(self, root_node):
        if '[' in root_node:
            root_node = root_node[1:-1]
        return root_node

    ####################
    # utils
    ####################
    def get_task_ref(self):
        from utils.commons.base_task import BaseTask
        task: BaseTask = self.task
        return task

    def log_metrics_to_tb(self, metrics, step=None):
        """Logs the metric dict passed in.

        :param metrics:
        """
        # turn all tensors to scalars
        scalar_metrics = self.metrics_to_scalars(metrics)

        step = step if step is not None else self.global_step
        # log actual metrics
        if self.proc_rank == 0:
            self.log_metrics(self.logger, scalar_metrics, step=step)

    @staticmethod
    def log_metrics(logger, metrics, step=None):
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            logger.add_scalar(k, v, step)

    def metrics_to_scalars(self, metrics):
        new_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if type(v) is dict:
                v = self.metrics_to_scalars(v)

            new_metrics[k] = v

        return new_metrics

    def save_terminal_logs(self):
        t = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        os.makedirs(f'{self.work_dir}/terminal_logs', exist_ok=True)
        self._tee = Tee(f'{self.work_dir}/terminal_logs/log_{t}.txt', 'w')
        atexit.register(lambda: getattr(self, '_tee', None) and self._tee.close())

    def save_codes(self):
        t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        if len(hparams['save_codes']) > 0:
            code_dir = f'{self.work_dir}/codes/{t}'
            subprocess.check_call(f'mkdir -p "{code_dir}"', shell=True)
            for c in hparams['save_codes']:
                if os.path.exists(c):
                    subprocess.check_call(
                        f'rsync -aR '
                        f'--include="*.py" '
                        f'--include="*.yaml" '
                        f'--exclude="__pycache__" '
                        f'--include="*/" '
                        f'--exclude="*" '
                        f'"./{c}" "{code_dir}/"',
                        shell=True)
            print(f"| Copied codes to {code_dir}.")
        cmd_path = f'{self.work_dir}/run_cmd/{t}.log'
        os.makedirs(f'{self.work_dir}/run_cmd', exist_ok=True)
        with open(cmd_path, 'w') as f:
            for k in os.environ:
                f.write(f"{k}={os.environ[k]}\n")
            f.write('\n')
            f.write(' '.join(['python'] + sys.argv) + '\n')
        config_path = f'{self.work_dir}/run_configs/{t}.yaml'
        os.makedirs(f'{self.work_dir}/run_configs', exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(hparams, f, allow_unicode=True)

def _ensure_fsdp_initialized(m: FullyShardedDataParallel):
    # 通过召回参数确保 flat param 以及 shard 元数据被正确构建
    try:
        with m.summon_full_params(m, writeback=False, with_grads=False):
            pass
    except Exception as e:
        print(f"[WARN] summon_full_params failed on {m.__class__.__name__}: {e}")
        
def _optimizer_params_match_model(optimizer, model):
    model_params = {id(p) for p in model.parameters()}
    opt_params = {id(p) for g in optimizer.param_groups for p in g['params'] if p is not None}
    missing_in_model = opt_params - model_params
    return len(missing_in_model) == 0
