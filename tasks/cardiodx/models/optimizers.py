from torch.optim.lr_scheduler import _LRScheduler
import torch.optim.lr_scheduler as lr_scheduler
import math
import torch.optim as optim

# 优化器
def get_optimizer(pg, optimizer_name, optimizer_args):
    if not hasattr(optim, optimizer_name):
        raise ValueError(f"Optimizer '{optimizer_name}' not found in torch.optim")
    
    optimizer_class = getattr(optim, optimizer_name)
    return optimizer_class(pg, **optimizer_args)

# 学习率调度器
def get_scheduler(optimizer, scheduler_name, args):
    if scheduler_name == 'constant':
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    elif scheduler_name == 'cosine':
        lambda_cosine = lambda x: ((1 + math.cos(x * math.pi / args.total_steps)) / 2) * (1 - args.lrf) + args.lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_cosine)
    elif scheduler_name == 'warm_cosine':
        warm_up_with_cosine_lr = lambda epoch: epoch / args.warm_up_epochs if epoch <= args.warm_up_epochs else 0.5 * (
            math.cos((epoch - args.warm_up_epochs) / (args.epochs - args.warm_up_epochs) * math.pi) + 1)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    elif scheduler_name == 'poly':
        scheduler = PolyLRScheduler(optimizer, args.lr, args.epochs)
    else:
        raise ValueError("scheduler not support")
    return scheduler

class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
