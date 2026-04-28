"""
tasks/view/trainer.py
视图分类 Trainer：Acc / F1 指标，迁移自 Cardiac_View/trainer.py
"""
import sys
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from common.base_trainer import BaseTrainer


class ViewTrainer(BaseTrainer):
    """Classification trainer: metric = macro-F1 (↑)."""

    @staticmethod
    def metric_function(y_true, y_pred):
        labels = [0, 1, 2]
        return {
            'acc':     accuracy_score(y_true, y_pred),
            'recall':  recall_score(y_true, y_pred, average=None, labels=labels),
            'f1':      f1_score(y_true, y_pred, average=None, labels=labels),
            'precision': precision_score(y_true, y_pred, average=None, labels=labels),
            'macro-f1': f1_score(y_true, y_pred, average='macro', labels=labels),
        }

    def train_one_epoch(self, model, optimizer, data_loader, device, epoch, tb_writer):
        model.train()
        mean_loss = torch.zeros(1).to(device)
        optimizer.zero_grad()
        res, y_true = [], []

        for it, (batch, labels, *_) in enumerate(
                tqdm(data_loader, dynamic_ncols=True, file=sys.stdout)):
            batch, labels = batch.to(device), labels.to(device)
            pred = model(batch)
            res.extend(pred.argmax(dim=1).detach().cpu())
            y_true.extend(labels.detach().cpu())

            total_loss, losses = self.loss_fn(pred, labels)
            total_loss.backward()
            if getattr(self.args, 'clip_grad_norm', False):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            optimizer.step()
            optimizer.zero_grad()

            mean_loss = (mean_loss * it + total_loss.detach()) / (it + 1)
            tb_writer.add_scalars('Train/Loss', {'mean_loss': mean_loss.item()},
                                  it + epoch * len(data_loader))
            tb_writer.add_scalars('Train/LR',
                                  {'lr': optimizer.param_groups[0]['lr']},
                                  it + epoch * len(data_loader))

        m = self.metric_function(y_true, res)
        tb_writer.add_scalar('Train/Acc', m['acc'], epoch)
        tb_writer.add_scalar('Train/MacroF1', m['macro-f1'], epoch)
        return mean_loss.item()

    @torch.no_grad()
    def eval(self, model, data_loader, tb_writer, epoch, sce) -> float:
        self.logger.info(f'Evaluating {sce}...')
        model.eval()
        res, y_true = [], []

        for batch, labels, *_ in tqdm(data_loader, dynamic_ncols=True, file=sys.stdout):
            pred = model(batch.to(self.device)).argmax(dim=1).cpu()
            res.extend(pred)
            y_true.extend(labels)

        m = self.metric_function(y_true, res)
        self.logger.info(f'[{sce}] acc={m["acc"]:.3f} macro-f1={m["macro-f1"]:.3f}')
        tb_writer.add_scalar(f'{sce}/Acc', m['acc'], epoch)
        tb_writer.add_scalar(f'{sce}/MacroF1', m['macro-f1'], epoch)
        return m['macro-f1']
