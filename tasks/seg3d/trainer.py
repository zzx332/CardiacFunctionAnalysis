"""
tasks/seg3d/trainer.py
3D SAX 分割 Trainer，接近原 SAX_Segmentation/train.py 风格，
但套入 BaseTrainer 骨架并支持 TensorBoard。
"""
import sys
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from common.base_trainer import BaseTrainer


class Seg3DTrainer(BaseTrainer):
    """
    3D SAX segmentation trainer.
    loss_fn 在 runner 里直接传入（原 config.py 的 loss_func）。
    metric: 平均 Dice（这里简化为 training loss 本身作排名依据，
            更完整的验证参见原 test.py）
    """

    def is_better(self, new_metric, best_metric):
        # loss 越小越好
        return new_metric < best_metric

    def train_one_epoch(self, model, optimizer, data_loader, device, epoch, tb_writer):
        model.train()
        epoch_loss = 0.0

        for it, (data, seg) in enumerate(tqdm(data_loader, dynamic_ncols=True, file=sys.stdout)):
            # seg shape from dataset: [B, C, d, w, h], take middle slice channel
            seg = seg[:, :, 0]          # [B, C, w, h]
            data, seg = data.to(device), seg.to(device)
            output = model(data)

            total_loss, _ = self.loss_fn(output, seg)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.detach().item()
            tb_writer.add_scalar('Train/Loss', total_loss.item(),
                                 it + epoch * len(data_loader))
            tb_writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'],
                                 it + epoch * len(data_loader))

        avg = epoch_loss / len(data_loader)
        return avg

    @torch.no_grad()
    def eval(self, model, data_loader, tb_writer, epoch, sce) -> float:
        """
        轻量 eval：以验证集 loss 作为排名依据。
        （完整 3D 滑窗推理建议用独立 test.py，与原项目保持一致）
        """
        self.logger.info(f'Evaluating {sce}...')
        model.eval()
        total_loss = 0.0

        for data, seg in tqdm(data_loader, dynamic_ncols=True, file=sys.stdout):
            seg = seg[:, :, 0]
            data, seg = data.to(self.device), seg.to(self.device)
            output = model(data)
            loss, _ = self.loss_fn(output, seg)
            total_loss += loss.item()

        avg = total_loss / max(len(data_loader), 1)
        self.logger.info(f'[{sce}] avg loss = {avg:.4f}')
        tb_writer.add_scalar(f'{sce}/Loss', avg, epoch)
        # is_better 越小越好，返回负值与 BaseTrainer 逻辑统一
        return -avg
