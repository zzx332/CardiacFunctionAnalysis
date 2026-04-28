"""
common/base_trainer.py
BaseTrainer：所有任务 Trainer 的公共骨架
- train_one_epoch / eval 由子类覆写
- train() 主循环、save_model、early stop 在此统一实现
"""
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple, Union, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys

from .config import LoggerSingleton
from .losses import MyLoss, DeepSupervisionWrapper


class BaseTrainer(ABC):
    """
    公共训练框架，子类仅需覆写：
    - train_one_epoch()
    - eval()
    - is_better(new_metric, best_metric) -> bool
    """

    def __init__(
        self,
        model: nn.Module,
        args,
        train_loader: Optional[DataLoader] = None,
        eval_loader: Optional[DataLoader] = None,
        model_init: Optional[str] = None,
        losses_config: Optional[Dict] = None,
        network_config: Optional[Dict] = None,
        optimizers: Tuple = (None, None),
    ):
        self.logger = LoggerSingleton()

        # 预加载权重（断点续训）
        if model_init:
            state = torch.load(model_init, map_location='cpu')
            model.load_state_dict(state.get('model_state_dict', state))
            self.logger.info(f"Loaded weights from {model_init}")

        self.model = model
        self.device = args.device
        self.model.to(self.device)

        self.train_loader = train_loader
        self.eval_loader = eval_loader

        # 损失函数
        if losses_config:
            self.loss_fn = MyLoss(losses_config)
            self.eval_loss_fn = MyLoss(losses_config)
        self._apply_deep_supervision(network_config)

        self.optimizer = optimizers[0]
        self.scheduler = optimizers[1]
        self.args = args

    def _apply_deep_supervision(self, network_config):
        """若启用深度监督，用 DeepSupervisionWrapper 包装 loss_fn。"""
        if not network_config:
            return
        # landmark / srp head 风格
        if network_config.get('head') == 'srp':
            ws = [1 / (2 ** i) for i in range(4)]
            ws = [w / sum(ws) for w in ws]
            self.loss_fn = DeepSupervisionWrapper(self.loss_fn, ws)
        # seg 风格
        elif network_config.get('deep_supervision') and network_config.get('layers'):
            import numpy as np
            pool_kernels = [[1, 1]] + [[2, 2]] * (len(network_config['layers']) - 1)
            scales = list(1 / np.cumprod(pool_kernels, axis=0))[:-1]
            ws = [1 / (2 ** i) for i in range(len(scales))]
            ws[-1] = 0
            ws = [w / sum(ws) for w in ws]
            self.loss_fn = DeepSupervisionWrapper(self.loss_fn, ws)

    # ------------------------------------------------------------------
    # 子类必须实现
    # ------------------------------------------------------------------
    @abstractmethod
    def train_one_epoch(self, model, optimizer, data_loader, device, epoch, tb_writer):
        ...

    @abstractmethod
    def eval(self, model, data_loader, tb_writer, epoch, sce) -> float:
        """返回用于比较"最优模型"的标量指标（越大越好）。"""
        ...

    def is_better(self, new_metric: float, best_metric: float) -> bool:
        """默认：越大越好（分类acc/dice/f1等）。MSE 任务子类可覆写为越小越好。"""
        return new_metric > best_metric

    # ------------------------------------------------------------------
    # 公共方法
    # ------------------------------------------------------------------
    def save_model(self, epoch: int, num_params: int, model_name: str,
                   best_metric=None):
        os.makedirs(self.args.model_output_path, exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epoch,
            'best_metric': best_metric,
            'num_params': num_params,
        }
        path = os.path.join(self.args.model_output_path, model_name)
        torch.save(checkpoint, path)
        self.logger.info(f"Saved model: {model_name}")

    def train(self):
        self.logger.info(f"Train samples  : {len(self.train_loader.dataset)}")
        self.logger.info(f"Train batches  : {len(self.train_loader)}")
        if self.eval_loader:
            self.logger.info(f"Val samples    : {len(self.eval_loader.dataset)}")

        tb_writer = SummaryWriter(log_dir=self.args.tensorboard_output_path)
        # 初始化最优指标（子类 is_better 决定比较方向）
        best_metric = -1e9 if self.is_better(1, 0) else 1e9
        patient = 0
        num_params = sum(p.numel() for p in self.model.parameters())

        for epoch in range(self.args.epochs):
            self.logger.info(f"===== EPOCH {epoch} =====")

            # 训练
            self.train_one_epoch(self.model, self.optimizer,
                                 self.train_loader, self.device, epoch, tb_writer)
            self.scheduler.step()

            # 验证
            val_metric = best_metric
            if self.eval_loader:
                val_metric = self.eval(self.model, self.eval_loader, tb_writer, epoch, 'val')

            # 定期保存
            if (epoch + 1) % self.args.save_epoch == 0 or epoch == self.args.epochs - 1:
                self.save_model(epoch, num_params, f'checkpoint-{epoch}.pt', val_metric)

            # 最优模型
            if self.eval_loader and self.is_better(val_metric, best_metric):
                best_metric = val_metric
                self.save_model(epoch, num_params, 'best_checkpoint.pt', best_metric)
                self.logger.info(f"New best metric at epoch {epoch}: {best_metric:.4f}")
                patient = 0
            elif self.eval_loader:
                patient += 1

            # 早停
            if getattr(self.args, 'earlystop', False):
                self.logger.info(f"Early stop counter: {patient}/{self.args.earlystop_patient}")
                if patient > self.args.earlystop_patient:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break

        # 加载最优权重
        best_path = os.path.join(self.args.model_output_path, 'best_checkpoint.pt')
        if os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path)['model_state_dict'])
            self.logger.info("Loaded best checkpoint weights")

        return self.model
