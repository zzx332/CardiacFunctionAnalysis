"""
tasks/landmark/runner.py
LandmarkRunner：整合 dataset / model / trainer 的一键执行入口
"""
from functools import partial
import torch
from torch.optim import AdamW

from common.utils import PolyLRScheduler
from .dataset import LandmarkDataLoader, transforms_img_landmark
from .models import HighResolutionNet
from .trainer import LandmarkTrainer


class LandmarkRunner:
    """由 main.py 调用：runner = LandmarkRunner(args); runner.run()"""

    def __init__(self, args):
        self.args = args

    def run(self):
        args = self.args
        # ── Data ──────────────────────────────────────────────
        train_loader = LandmarkDataLoader(
            data_path=args.train_data_path,
            sce='train',
            transform=partial(transforms_img_landmark, args=args, sec='train'),
            sigma=args.sigma,
            debug=args.debug,
            img_root_path=args.img_root_path_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            repeat_factor=args.repeat_factor,
            shuffle=True,
        )
        eval_loader = None
        if args.val_data_path:
            eval_loader = LandmarkDataLoader(
                data_path=args.val_data_path,
                sce='val',
                transform=partial(transforms_img_landmark, args=args, sec='val'),
                sigma=args.sigma,
                debug=args.debug,
                img_root_path=args.img_root_path_val,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
            )

        # ── Model ─────────────────────────────────────────────
        model = HighResolutionNet(
            base_channel=args.base_channel,
            num_joints=args.num_joints,
            head=args.head,
        )

        # ── Optimizer / Scheduler ─────────────────────────────
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = PolyLRScheduler(optimizer, initial_lr=args.lr, max_steps=args.epochs)

        # ── Trainer ───────────────────────────────────────────
        trainer = LandmarkTrainer(
            model=model,
            args=args,
            train_loader=train_loader,
            eval_loader=eval_loader,
            losses_config={'mse_loss': {'weight': 1.0}},
            network_config={'head': args.head},
            optimizers=(optimizer, scheduler),
        )
        return trainer.train()
