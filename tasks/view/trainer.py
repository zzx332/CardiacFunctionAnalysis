"""
tasks/view/trainer.py
视图分类 Trainer：Acc / F1 指标，迁移自 Cardiac_View/trainer.py
"""
import sys
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from torch.utils.tensorboard import SummaryWriter
import os
import joblib
from scipy import stats
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

    @torch.no_grad()
    def predict_tta(self, data_loader):
        self.model.eval()
        self.logger.info(f'*********** predict on {self.args.test_data_path}')
        test_data_name = os.path.basename(self.args.test_data_path).split('.')[0]
        tb_writer = SummaryWriter(log_dir=self.args.tensorboard_output_path)
        # 打印验证进度
        data_loader = tqdm(data_loader, desc="predict...", dynamic_ncols=True, file=sys.stdout)
        res = []
        y_true = []
        for step, batch in enumerate(data_loader):
            inputs, labels, fp = batch
            pred = self.model(inputs.to(self.device))
            pred = pred.argmax(axis=1).cpu()
            pred = pred.reshape(-1) #reshpe(4,-1)
            pred = stats.mode(pred, axis=0, keepdims=True)[0]
            labels = labels[0:1].numpy()
            # if any(pred!=labels):
            #     print(pred,labels)
            #     print(fp)
            #     print('Failed.')
            res.extend(pred)
            y_true.extend(labels)
        import joblib
        joblib.dump( y_true,self.args.test_data_path.replace('.pkl', '_true.pkl'))
        joblib.dump( res, self.args.test_data_path.replace('.pkl', '_pred.pkl'))
        metrics = self.metric_function(y_true, res)
        # tensorboard可视化，for循环加入或者直接传入字典
        tb_writer.add_scalars(f'Test{test_data_name}/Accuracy', {'accuracy': metrics['acc']},0)
        tb_writer.add_scalars(f'Test{test_data_name}/Recall', {f'class_{class_index}': value for class_index, value in enumerate(metrics['recall'])},0)
        tb_writer.add_scalars(f'Test{test_data_name}/Precision', {f'class_{class_index}': value for class_index, value in enumerate(metrics['precision'])},0)
        tb_writer.add_scalars(f'Test{test_data_name}/F1', {f'class_{class_index}': value for class_index, value in enumerate(metrics['f1'])},0)
        tb_writer.add_scalars(f'Test{test_data_name}/', {'macro-f1': metrics['macro-f1']},0)
        return metrics