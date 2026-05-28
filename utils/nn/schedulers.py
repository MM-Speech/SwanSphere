import numpy as np
from utils.commons.hparams import hparams


class NoneSchedule(object):
    def __init__(self, optimizer, lr):
        self.optimizer = optimizer
        self.constant_lr = lr
        self.step(0)

    def step(self, num_updates):
        self.lr = self.constant_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def get_last_lr(self):
        return self.get_lr()


class RSQRTSchedule(NoneSchedule):
    def __init__(self, optimizer, lr, warmup_updates, hidden_size):
        self.optimizer = optimizer
        self.constant_lr = lr
        self.warmup_updates = warmup_updates
        self.hidden_size = hidden_size
        self.lr = lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr
        self.step(0)

    def step(self, num_updates):
        constant_lr = self.constant_lr
        warmup = min(num_updates / self.warmup_updates, 1.0)
        rsqrt_decay = max(self.warmup_updates, num_updates) ** -0.5
        rsqrt_hidden = self.hidden_size ** -0.5
        self.lr = max(constant_lr * warmup * rsqrt_decay * rsqrt_hidden, 1e-7)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr


class WarmupSchedule(NoneSchedule):
    def __init__(self, optimizer, lr, warmup_updates):
        self.optimizer = optimizer
        self.constant_lr = self.lr = lr
        self.warmup_updates = max(1, warmup_updates)
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr
        self.step(0)

    def step(self, num_updates):
        constant_lr = self.constant_lr
        warmup = min(num_updates / self.warmup_updates, 1.0)
        self.lr = max(constant_lr * warmup, 1e-7)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr


class ExponentialSchedule(NoneSchedule):
    def __init__(self, optimizer, lr, warmup_updates):
        self.optimizer = optimizer
        self.constant_lr = self.lr = lr
        self.warmup_updates = warmup_updates
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr
        self.step(0)

    def step(self, num_updates):
        constant_lr = self.constant_lr
        if self.warmup_updates > 0 and num_updates <= self.warmup_updates:
            warmup = min(num_updates / self.warmup_updates, 1.0)
            self.lr = max(constant_lr * warmup, 1e-7)
        else:
            new_lrate = constant_lr * (0.1 ** (num_updates / 250_000)) # decay by 0.1x for every 250k steps
            self.lr = max(new_lrate, hparams.get("min_lr", 1e-6))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr


class ExponentialScheduleWithAudattNet(NoneSchedule):
    """
    Default Scheduler in AD-NeRF
    for audatt net, since it starts at 20_0000 steps, we need to enlarge its lr
    in optimizer, we set param_groups[1] to optimize audatt net
    """
    def __init__(self, optimizer, lr, warmup_updates=0):
        self.optimizer = optimizer
        self.constant_lr = self.lr = lr
        self.warmup_updates = warmup_updates
        optimizer.param_groups[0]['lr'] = self.lr
        optimizer.param_groups[1]['lr'] = self.lr * 5
        self.step(0)

    def step(self, num_updates):
        constant_lr = self.constant_lr
        if self.warmup_updates > 0 and num_updates <= self.warmup_updates:
            warmup = min(num_updates / self.warmup_updates, 1.0)
            self.lr = max(constant_lr * warmup, 1e-7)
        else:
            new_lrate = constant_lr * (0.1 ** (num_updates / 250_000)) # decay by 0.1x for every 250k steps
            self.lr = max(new_lrate, 1e-7)

        self.optimizer.param_groups[0]['lr'] = self.lr
        self.optimizer.param_groups[1]['lr'] = self.lr * 5
        return self.lr

class ExponentialScheduleForRADNeRF(NoneSchedule):
    """
    Default Scheduler in RAD-NeRF
    RAD-NeRF has two groups of params with different lr
    for tileGrid embedding, the lr=5e-3
    for other network params, the lr=5e-4
    """
    def __init__(self, optimizer, lr, warmup_updates=0):
        self.optimizer = optimizer
        self.constant_lr = self.lr = lr # 0.0005
        self.warmup_updates = warmup_updates
        self.finetune_lips = hparams['finetune_lips']
        self.finetune_lips_start_iter = hparams['finetune_lips_start_iter']

        optimizer.param_groups[0]['lr'] = self.lr # for Net_params in RAD-NeRF, lr starts from 0.0005
        optimizer.param_groups[1]['lr'] = self.lr * 10 # for tileGrid, lr starts from 0.005
        optimizer.param_groups[2]['lr'] = self.lr * 5 # for Att Net, lr starts from 0.0025
        self.step(0)

    def step(self, num_updates):
        constant_lr = self.constant_lr
        if self.warmup_updates > 0 and num_updates <= self.warmup_updates:
            warmup = min(num_updates / self.warmup_updates, 1.0)
            self.lr = max(constant_lr * warmup, 1e-5)
        else:
            if self.finetune_lips and num_updates > self.finetune_lips_start_iter:
                new_lrate = constant_lr * (0.1 ** (num_updates / 250_000)) # decay by 0.05x for every 200k steps
            else:
                new_lrate = constant_lr * (0.1 ** (num_updates / 250_000)) # decay by 0.1x for every 200k steps

            self.lr = max(new_lrate, 1e-5)

        self.optimizer.param_groups[0]['lr'] = self.lr
        self.optimizer.param_groups[1]['lr'] = self.lr * 10
        self.optimizer.param_groups[2]['lr'] = self.lr * 5
        return self.lr
    

class ExponentialScheduleForRADNeRFTorso(NoneSchedule):
    """
    Default Scheduler in RAD-NeRF
    RAD-NeRF has two groups of params with different lr
    for tileGrid embedding, the lr=5e-3
    for other network params, the lr=5e-4
    """
    def __init__(self, optimizer, lr, warmup_updates=0):
        self.optimizer = optimizer
        self.constant_lr = self.lr = lr # 0.0005
        self.warmup_updates = warmup_updates

        optimizer.param_groups[0]['lr'] = self.lr # for Net_params in RAD-NeRF, lr starts from 0.0005
        optimizer.param_groups[1]['lr'] = self.lr * 10 # for tileGrid, lr starts from 0.005
        self.step(0)

    def step(self, num_updates):
        constant_lr = self.constant_lr
        if self.warmup_updates > 0 and num_updates <= self.warmup_updates:
            warmup = min(num_updates / self.warmup_updates, 1.0)
            self.lr = max(constant_lr * warmup, 1e-5)
        else:
            new_lrate = constant_lr * (0.1 ** (num_updates / 250_000)) # decay by 0.1x for every 200k steps
            self.lr = max(new_lrate, 1e-5)
        self.optimizer.param_groups[0]['lr'] = self.lr
        self.optimizer.param_groups[1]['lr'] = self.lr * 10
        return self.lr
    

class CosineSchedule(NoneSchedule):
    def __init__(self, optimizer, lr, warmup_updates, total_updates):
        self.optimizer = optimizer
        self.constant_lr = lr
        self.warmup_updates = warmup_updates
        self.total_updates = total_updates
        self.lr = lr
        self.assign_learning_rate(self.optimizer, self.lr)
        self.step(0)

    def assign_learning_rate(self, optimizer, new_lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

    def _warmup_lr(self, base_lr, warmup_length, step):
        return base_lr * (step + 1) / warmup_length

    def step(self, num_updates):
        if self.warmup_updates > 0 and num_updates <= self.warmup_updates:
            lr = self._warmup_lr(self.lr, self.warmup_updates, num_updates)
        elif num_updates <= self.total_updates:
            e = num_updates - self.warmup_updates
            es = self.total_updates - self.warmup_updates
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * self.lr
        else:
            lr = 1e-5
        lr = max(1e-5, lr)
        self.assign_learning_rate(self.optimizer, lr)
        return lr


class CosineAnnealingWarmRestartsWithWarmup:
    def __init__(self, optimizer, lr_max, warmup_updates, total_updates, 
                 initial_period, period_mult=1.0, lr_min=1e-5):
        """
        带周期性重启的余弦退火学习率调度器（含预热）
        
        参数:
            optimizer: 优化器对象
            lr_max: 最大学习率（预热结束后达到）
            warmup_updates: 预热步数
            total_updates: 总训练步数
            initial_period: 初始周期长度（步数）
            period_mult: 周期倍增因子（>1 会使后续周期变长，1.0 表示等长周期）
            lr_min: 最小学习率
        """
        self.optimizer = optimizer
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.warmup_updates = warmup_updates
        self.total_updates = total_updates
        self.initial_period = initial_period
        self.period_mult = period_mult
        
        # 初始化周期状态
        self.cycle_start = self.warmup_updates  # 第一个周期开始位置
        self.current_period = initial_period
        self.cycle_end = self.cycle_start + self.current_period
        self.cycle_count = 0
        
        self.assign_learning_rate(self.optimizer, self._calculate_lr(0))

    def assign_learning_rate(self, optimizer, new_lr):
        """设置优化器的学习率"""
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

    def _calculate_lr(self, num_updates):
        """计算给定步数的学习率"""
        # 预热阶段：线性增加学习率
        if num_updates < self.warmup_updates:
            ratio = num_updates / max(1, self.warmup_updates)
            return self.lr_min + (self.lr_max - self.lr_min) * ratio
        
        # 检查是否进入新周期
        if num_updates >= self.cycle_end and num_updates < self.total_updates:
            # 启动新周期
            self.cycle_count += 1
            self.cycle_start = self.cycle_end
            
            # 更新周期长度（应用倍增因子）
            self.current_period = int(self.current_period * self.period_mult)
            self.cycle_end = min(self.cycle_start + self.current_period, self.total_updates)
        
        # 训练结束后保持最小学习率
        if num_updates >= self.total_updates:
            return self.lr_min
        
        # 余弦退火阶段（在当前周期内）
        # 计算当前周期内的进度 (0.0 → 1.0)
        cycle_position = (num_updates - self.cycle_start) / max(1, (self.cycle_end - self.cycle_start))
        # 确保进度在 [0, 1] 范围内
        cycle_position = min(max(cycle_position, 0.0), 1.0)
        
        # 余弦退火公式：从 lr_max 退火到 lr_min
        cosine_decay = 0.5 * (1 + np.cos(np.pi * cycle_position))
        return self.lr_min + (self.lr_max - self.lr_min) * cosine_decay

    def step(self, num_updates):
        """更新学习率并返回当前学习率值"""
        lr = self._calculate_lr(num_updates)
        self.assign_learning_rate(self.optimizer, lr)
        return lr
    