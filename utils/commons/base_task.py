import logging
import os
import time
import random
import sys
import yaml
import numpy as np
import torch.utils.data
from torch import nn
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from utils.commons.hparams import hparams
from utils.commons.meters import AvgrageMeter
from utils.commons.tensor_utils import tensors_to_scalars
from utils.commons.trainer import Trainer
from utils.nn.model_utils import print_arch, num_params, unwrap_model


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


class BaseTask(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BaseTask, self).__init__()
        self.current_epoch = 0
        self.global_step = 0
        self.trainer = None
        self.use_ddp = False
        self.gradient_clip_norm = hparams['clip_grad_norm']
        self.gradient_clip_val = hparams.get('clip_grad_value', 0)
        self.model = None
        self.epoch_training_losses_meter = None
        self.logger: SummaryWriter = None
        self.log_grad_every_n_steps = hparams.get('log_grad_every_n_steps', 0)

    ######################
    # build model, dataloaders, optimizer, scheduler and tensorboard
    ######################
    def build_model(self):
        raise NotImplementedError

    def train_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def build_scheduler(self, optimizer):
        return None

    def build_optimizer(self):
        raise NotImplementedError

    def fsdp_wrap_policy(self):
        return None

    def fsdp_optm2model(self):
        return []

    def fsdp_ignored_modules(self):
        return []

    def configure_optimizers(self):
        optm = self.build_optimizer()
        self.scheduler = self.build_scheduler(optm)
        if isinstance(optm, (list, tuple)):
            return optm
        return [optm]

    def build_tensorboard(self, save_dir, name, **kwargs):
        log_dir = os.path.join(save_dir, name)
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir, **kwargs)

    ######################
    # training
    ######################
    def on_train_start(self):
        if self.trainer.proc_rank_local == 0:
            devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(",")
            for d in devices:
                os.system(f'pkill -f "voidgpu{d}"')
            for n in self.trainer.training_module_names:
                if 'ema' in [n_.lower() for n_ in n.split('_')]:
                    continue
                m = getattr(self, n)
                print_arch(m, model_name=n)
                num_params(m, model_name=n)
                for n_, m_ in unwrap_model(m).named_children():
                    num_params(m_, model_name=n_)
                if self.trainer.use_fsdp:
                    print("↑ trainable params before training (possibly split by FSDP)")
            # if torch.__version__.split(".")[0] == '2' and hparams.get("torch_compile", False):
            #     self.model = torch.compile(self.model, mode='default')
                
        if torch.__version__.split(".")[0] == '2' and hparams.get("torch_compile", False):
            compile_backend = hparams.get('compile_backend', 'inductor')
            compile_mode = hparams.get('compile_mode', 'max-autotune')
            compile_dynamic = hparams.get('compile_dynamic', True)
            compile_fullgraph = hparams.get('compile_fullgraph', False)
            
            if self.trainer.proc_rank == 0:
                print(f'| Enable torch.compile for modules (except EMA shadows): {self.trainer.training_module_names}')
                print(f'| compile: backend={compile_backend}, mode={compile_mode}, '
                    f'dynamic={compile_dynamic}, fullgraph={compile_fullgraph}')
            
            for n in self.trainer.training_module_names:
                if 'ema' in [n_.lower() for n_ in n.split('_')]:
                    continue
                m = getattr(self, n)
                try:
                    compiled_m = torch.compile(
                        m,
                        backend=compile_backend,           # inductor (默认) 最快/最稳
                        mode=compile_mode,                 # 'max-autotune' 通常训练更快；也可 'reduce-overhead'
                        dynamic=compile_dynamic,           # 动态 shape 更稳当
                        fullgraph=compile_fullgraph        # True 要求整体可导出；一般先 False
                    )
                    setattr(self, n, compiled_m)
                except Exception as e:
                    if self.trainer.proc_rank == 0:
                        print(f'| WARN: torch.compile failed on module `{n}`: {e}. Fallback to eager.')

    def load_model(self):
        pass

    def on_train_end(self):
        pass

    def on_epoch_start(self):
        self.epoch_training_losses_meter = {'total_loss': AvgrageMeter()}

    def on_epoch_end(self):
        loss_outputs = {k: v.avg for k, v in self.epoch_training_losses_meter.items()}
        print(f"Epoch {self.current_epoch} ended. Steps: {self.global_step}. {loss_outputs}")
        loss_outputs = {"epoch_mean/" + k: v for k, v in loss_outputs.items()}
        return loss_outputs

    def forward(self, *args, **kwargs):
        if self.training:
            output = self.training_step(*args, **kwargs)
        elif self.testing:
            output = self.test_step(*args, **kwargs)
        else:
            output = self.validation_step(*args, **kwargs)
        return output

    def _training_step(self, sample, batch_idx, optimizer_idx):
        """

        :param sample:
        :param batch_idx:
        :return: total loss: torch.Tensor, loss_log: dict
        """
        raise NotImplementedError

    def training_step(self, sample, batch_idx, optimizer_idx=-1):
        """

        :param sample:
        :param batch_idx:
        :param optimizer_idx:
        :return: {'loss': torch.Tensor, 'progress_bar': dict, 'tb_log': dict}
        """
        # perform the main training step in a specific task
        loss_ret = self._training_step(sample, batch_idx, optimizer_idx)
        if loss_ret is None:
            return {'loss': None}
        total_loss, log_outputs = loss_ret
        log_outputs = tensors_to_scalars(log_outputs)

        # add to epoch meter
        for k, v in log_outputs.items():
            if '/' in k:
                k_split = k.split("/")
                assert len(k_split) == 2, "we only support one `/` in tag_name, i.e., `<tag>/<sub_tag>`"
                k = k.replace("/", "_")
            if k not in self.epoch_training_losses_meter:
                self.epoch_training_losses_meter[k] = AvgrageMeter()
            if not np.isnan(v):
                self.epoch_training_losses_meter[k].update(v)

        if optimizer_idx >= 0:
            for params_group_i in range(len(self.trainer.optimizers[optimizer_idx].param_groups)):
                log_outputs[f'lr/optimizer{optimizer_idx}_params_group{params_group_i}'] = self.trainer.optimizers[optimizer_idx].param_groups[params_group_i]['lr']

        # add to progress bar
        progress_bar_log = {}
        for k, v in log_outputs.items():
            if 'monitor/' in k:
                continue
            if '/' in k:
                k_split = k.split("/")
                assert len(k_split) == 2, "we only support one `/` in tag_name, i.e., `<tag>/<sub_tag>`"
                k = k.replace("/", "_")
            k = k.replace('optimizer', 'optm').replace('params_group', 'pg')
            assert k not in progress_bar_log, f"we got duplicate tags in log_outputs, check this `{k}`"
            progress_bar_log[k] = v

        # add to tensorboard
        tb_log = {}
        for k, v in log_outputs.items():
            if '/' in k:
                tb_log[k] = v
            else:
                tb_log[f'tr/{k}'] = v

        if not isinstance(total_loss, torch.Tensor):
            return {'loss': None}
        self.epoch_training_losses_meter['total_loss'].update(total_loss.item())

        return {
            'loss': total_loss,
            'progress_bar': progress_bar_log,
            'tb_log': tb_log
        }
        
    def compute_grad_norm(self, optimizer, distributed=True, norm_type=2.0):
        """
        计算当前 optimizer 全部参数的全局 L2 grad norm。
        - 要求在 AMP 的 unscale_ 之后调用，保证是真实梯度。
        - 支持 DDP/FSDP：通过 all_reduce(sum of squares) 聚合各 rank。
        """
        if norm_type != 2.0:
            norm_type = 2.0

        device = torch.device(self.trainer.device) if isinstance(self.trainer.device, str) else self.trainer.device
        local_sq_sum = torch.zeros(1, device=device, dtype=torch.float32)
        has_grad = False

        for group in optimizer.param_groups:
            for p in group['params']:
                if p is None or p.grad is None:
                    continue
                g = p.grad
                # 稀疏梯度
                if g.is_sparse:
                    g = g.coalesce().values()
                # 统一到 float32 做范数更稳
                g = g.detach().float()
                local_sq_sum += torch.sum(g * g)
                has_grad = True

        if not has_grad:
            return 0.0

        if distributed and dist.is_initialized():
            dist.all_reduce(local_sq_sum, op=dist.ReduceOp.SUM)

        total_norm = torch.sqrt(local_sq_sum)
        return float(total_norm.item())

    def on_before_optimization(self, opt_idx):
        """
        在 AMP unscale 之后、梯度裁剪之前调用（推荐调整 Trainer 调用顺序到此处）。
        返回一个 dict，以便 Trainer 统一写入 TensorBoard / pbar。
        """
        # 频率控制（按“有效 step”，即累计边界）
        if getattr(self, 'log_grad_every_n_steps', 0) <= 0:
            return
        else:
            eff_step = (self.global_step + 1) // hparams.get('accumulate_grad_batches', 1)
            if eff_step % self.log_grad_every_n_steps != 0:
                return None

        try:
            optimizer = self.trainer.optimizers[opt_idx]
            gnorm = self.compute_grad_norm(optimizer, distributed=True, norm_type=2.0)
            return {f'monitor/grad_norm_optm{opt_idx}': gnorm}
        except Exception as e:
            if self.trainer.proc_rank == 0:
                print(f'| WARN: on_before_optimization compute_grad_norm failed: {e}')
            return None

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if isinstance(self.scheduler, (list, tuple)):
            scheduler = self.scheduler[optimizer_idx] if optimizer_idx < len(self.scheduler) else None
        else:
            scheduler = self.scheduler if optimizer_idx == 0 else None
        if scheduler is not None:
            scheduler.step(self.global_step // hparams['accumulate_grad_batches'])

    ######################
    # validation
    ######################
    def validation_start(self):
        pass

    def validation_step(self, sample, batch_idx):
        """

        :param sample:
        :param batch_idx:
        :return: output: {"losses": {...}, "total_loss": float, ...} or (total loss: torch.Tensor, loss_log: dict)
        """
        raise NotImplementedError

    def validation_end(self, outputs):
        """

        :param outputs:
        :return: loss_output: dict
        """
        all_losses_meter = {'total_loss': AvgrageMeter()}
        for output in outputs:
            if output is None or len(output) == 0:
                continue
            if isinstance(output, dict):
                assert 'losses' in output, 'Key "losses" should exist in validation output.'
                n = output.pop('nsamples', 1)
                losses = tensors_to_scalars(output['losses'])
                total_loss = output.get('total_loss', sum(losses.values()))
            else:
                assert len(output) == 2, 'Validation output should only consist of two elements: (total_loss, losses)'
                n = 1
                total_loss, losses = output
                losses = tensors_to_scalars(losses)
            if isinstance(total_loss, torch.Tensor):
                total_loss = total_loss.item()
            for k, v in losses.items():
                if k not in all_losses_meter:
                    all_losses_meter[k] = AvgrageMeter()
                all_losses_meter[k].update(v, n)
            all_losses_meter['total_loss'].update(total_loss, n)
        loss_output = {k: round(v.avg, 10) for k, v in all_losses_meter.items()}
        self.validation_extra_logging(loss_output)
        print(f"| Validation results@{self.global_step}: {loss_output}")
        return {
            'tb_log': {f'val/{k}': v for k, v in loss_output.items()},
            'val_loss': loss_output['total_loss']
        }
    
    def validation_extra_logging(self, logger_dict: dict):
        pass

    ######################
    # testing
    ######################
    def test_start(self):
        pass

    def test_step(self, sample, batch_idx):
        return self.validation_step(sample, batch_idx)

    def test_end(self, outputs):
        return self.validation_end(outputs)

    ######################
    # start training/testing
    ######################
    @classmethod
    def start(cls):
        if hparams.get('use_file_system_mp'):
            torch.multiprocessing.set_sharing_strategy('file_system')

        def is_port_in_use(port: int) -> bool:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('localhost', port)) == 0

        os.environ['MASTER_PORT'] = str(random.randint(10000, 11000))
        while is_port_in_use(int(os.environ['MASTER_PORT'])):
            print(f"| Port {os.environ['MASTER_PORT']} is in use. Change another port...")
            os.environ['MASTER_PORT'] = str(random.randint(10000, 11000))
            time.sleep(1)

        random.seed(hparams['seed'])
        np.random.seed(hparams['seed'])
        work_dir = hparams['work_dir']
        trainer = Trainer(
            work_dir=work_dir,
            val_check_interval=hparams['val_check_interval'],
            tb_log_interval=hparams['tb_log_interval'],
            max_updates=hparams['max_updates'],
            num_sanity_val_steps=hparams['num_sanity_val_steps'] if not hparams['validate'] else 10000,
            accumulate_grad_batches=hparams['accumulate_grad_batches'],
            print_nan_grads=hparams['print_nan_grads'],
            resume_from_checkpoint=hparams.get('resume_from_checkpoint', 0),
            resume_from=hparams.get('resume_from', ''),
            amp=hparams['amp'],
            monitor_key=hparams['valid_monitor_key'],
            monitor_mode=hparams['valid_monitor_mode'],
            num_ckpt_keep=hparams['num_ckpt_keep'],
            save_best=hparams['save_best'],
            seed=hparams['seed'],
            debug=hparams['debug'],
            profile=hparams.get('profile', False)
        )
        if trainer.proc_rank == 0:
            with open(f"{work_dir}/hparams.yaml", "w") as file:
                yaml.dump(hparams, file, allow_unicode=True)
        if not hparams['infer']:  # train
            trainer.fit(cls)
        else:
            trainer.test(cls)

    def on_keyboard_interrupt(self):
        pass
