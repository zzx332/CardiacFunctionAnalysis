"""
tasks/cardiodx/trainer.py
CardioDx Trainer：step-based 训练循环，二分类，渐进平衡采样
迁移自 CardioDx/trainer.py
"""
import os
import sys
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, Recall, Precision, F1Score

from common.base_trainer import BaseTrainer

class CardioDxTrainer(BaseTrainer):
    """
    Step-based 训练（而非 epoch-based），每轮自动更新采样策略。
    """

    def __init__(self, model, args, train_loader, eval_loader=None,
                 losses_config=None, network_config=None, optimizers=(None, None)):
        super().__init__(
        model=model,
        args=args,
        train_loader=train_loader,
        eval_loader=eval_loader,
        model_init=None,
        losses_config=losses_config,
        network_config=network_config,
        optimizers=optimizers,
    )

        # Metrics
        self.metrics = MetricCollection({
            'accuracy':  Accuracy('binary').to(self.device),
            'recall':    Recall('binary').to(self.device),
            'precision': Precision('binary').to(self.device),
            'f1':        F1Score('binary').to(self.device),
        })

        self.total_steps   = getattr(args, 'total_steps', args.epochs * 100)
        self.current_step  = getattr(args, 'current_step', 0)
        self.current_epoch = getattr(args, 'current_epoch', 0)
        self.best_f1       = 0.0
        self.tb_writer     = SummaryWriter(log_dir=args.tensorboard_output_path)

    def train_one_epoch(self, batch):
        self.model.train()
        ed = batch['ed'].to(self.device, non_blocking=True)
        es = batch['es'].to(self.device, non_blocking=True)
        labels = batch['label'].float().to(self.device, non_blocking=True)

        pred = self.model(ed, es).squeeze().float()
        total_loss, losses = self.loss_fn(pred, labels)
        self.optimizer.zero_grad()
        total_loss.backward()
        if getattr(self.args, 'clip_grad_norm', False):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
        self.optimizer.step()
        self.metrics.update(pred, labels.long())
        self.tb_writer.add_scalar('Train/TotalLoss', total_loss.item(), self.current_step)
        self.tb_writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], self.current_step)

    def eval(self):
        self.model.eval()
        self.metrics.reset()
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, dynamic_ncols=True, file=sys.stdout):
                ed = batch['ed'].to(self.device, non_blocking=True)
                es = batch['es'].to(self.device, non_blocking=True)
                labels = batch['label'].float().to(self.device, non_blocking=True)
                pred = self.model(ed, es).squeeze().float()
                self.metrics.update(pred, labels.long())
        val_metrics = self.metrics.compute()
        self.logger.info(f"Val F1={val_metrics['f1']:.4f}  Acc={val_metrics['accuracy']:.4f}")
        for k, v in val_metrics.items():
            self.tb_writer.add_scalar(f'Val/{k}', v, self.current_epoch)
        return val_metrics

    def save(self, path: str, is_best: bool = False):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'current_step': self.current_step,
                    'current_epoch': self.current_epoch,
                    'best_f1': self.best_f1}, path)
        self.logger.info(f"Saved checkpoint: {path}")

    def train(self):
        iter_per_epoch = getattr(self.args, 'iter_per_epoch', None)
        self.logger.info(f"Total steps: {self.total_steps}")

        while self.current_step < self.total_steps:
            self.current_epoch += 1
            self.model.train()
            self.metrics.reset()
            # 更新采样器（渐进平衡）
            self.train_loader = self.train_loader.update_sampler(
                self.current_step, self.total_steps)
            n_iters = iter_per_epoch or len(self.train_loader)
            loader_iter = iter(self.train_loader)

            with tqdm(total=n_iters, dynamic_ncols=True, file=sys.stdout) as pbar:
                for _ in range(n_iters):
                    if self.current_step >= self.total_steps:
                        break
                    try:
                        batch = next(loader_iter)
                    except StopIteration:
                        loader_iter = iter(self.train_loader)
                        batch = next(loader_iter)
                    self.train_one_epoch(batch)
                    self.scheduler.step()
                    self.current_step += 1
                    pbar.update(1)
                    pbar.set_description(f"Step {self.current_step}/{self.total_steps}")

            self.logger.info(f"Epoch {self.current_epoch} done, step={self.current_step}")

            if self.eval_loader:
                val = self.eval()
                if val['f1'] > self.best_f1:
                    self.best_f1 = val['f1']
                    self.save(os.path.join(self.args.model_output_path, 'best.pt'), is_best=True)

        self.save(os.path.join(self.args.model_output_path, 'final.pt'))
        return self.model
