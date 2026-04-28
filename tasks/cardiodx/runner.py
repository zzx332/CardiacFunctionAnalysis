"""
tasks/cardiodx/runner.py
CardioDxRunner：ED+ES 双输入心脏疾病二分类任务
"""
import torch
import numpy as np
import json
from torchvision.transforms import v2

from .dataset import CardioDxDataLoader
from .trainer import CardioDxTrainer
from .data_process import create_transforms

class CardioDxRunner:
    def __init__(self, args):
        self.args = args

    def run(self):
        args = self.args
        from functools import partial
        transforms_dict = create_transforms(args)

        # ── DataLoaders ─────────────────────────────────────────
        train_loader = CardioDxDataLoader(
            data_path=args.train_data_path, transform=transforms_dict['train'],
            random_state=getattr(args, 'random_seed', 42),
            infinite=True,
            sampling_strategy=getattr(args, 'sampling_strategy', 'progressively_balanced'),
            debug=args.debug, batch_size=args.batch_size,
            num_workers=args.num_workers, shuffle=True,
        )
        eval_loader = None
        if getattr(args, 'val_data_path', None):
            eval_loader = CardioDxDataLoader(
                data_path=args.val_data_path, transform=transforms_dict['val'],
                debug=args.debug, batch_size=args.batch_size,
                num_workers=args.num_workers, shuffle=False,
            )

        # ── Model ────────────────────────────────────────────────
        arch = getattr(args, 'arch', 'resnet')
        num_classes = getattr(args, 'num_classes', 1)  # 二分类用 sigmoid
        in_c = getattr(args, 'in_channels', 3)
        
        if arch == 'resnet':
            from .models.resnet import TwoStreamResNet
            model = TwoStreamResNet(
                in_channels=in_c,
                num_classes=num_classes,
                backbone_type='resnet18',
                fusion_method='concat',
                include_top=False,
            )
        else:
            from .models.efficientnet import two_stream_efficientnetv2_s
            model = two_stream_efficientnetv2_s(
                in_c=in_c,
                num_classes=num_classes,
                fusion_method='concat',
            )

        # ── Optimizer / Scheduler ────────────────────────────────
        import math
        import torch.optim as optim
        import torch.optim.lr_scheduler as lr_sched
        from common.utils import PolyLRScheduler

        pg = [p for p in model.parameters() if p.requires_grad]
        opt_name = getattr(args, 'optimizer', 'AdamW')
        if opt_name == 'Adam':
            optimizer = optim.Adam(pg, lr=args.lr)
        elif opt_name == 'SGD':
            optimizer = optim.SGD(pg, args.lr, momentum=0.99, nesterov=True)
        else:
            optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=getattr(args, 'weight_decay', 1e-4))

        total_steps = getattr(args, 'total_steps', None)
        if total_steps is None:
            total_steps = args.epochs * 100
        lrf = getattr(args, 'lrf', 0.01)
        sch_name = getattr(args, 'scheduler', 'cosine')
        if sch_name == 'constant':
            scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        elif sch_name == 'poly':
            scheduler = PolyLRScheduler(optimizer, initial_lr=args.lr, max_steps=total_steps)
        else:
            scheduler = lr_sched.LambdaLR(
                optimizer,
                lr_lambda=lambda x: ((1 + math.cos(x * math.pi / total_steps)) / 2) * (1 - lrf) + lrf
            )
        args.total_steps   = total_steps
        args.current_step  = 0
        args.current_epoch = 0

        # ── Trainer ──────────────────────────────────────────────
        losses_config = getattr(args, 'losses_config', '{"batch_cross_entropy_loss":{"weight":1}}')
        if isinstance(losses_config, str):
            losses_config = json.loads(losses_config)
        trainer = CardioDxTrainer(
            model=model, args=args,
            train_loader=train_loader, eval_loader=eval_loader,
            losses_config=losses_config,
            optimizers=(optimizer, scheduler),
        )
        return trainer.train()
