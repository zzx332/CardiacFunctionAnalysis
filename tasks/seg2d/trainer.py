"""
tasks/seg2d/trainer.py
2D 分割 Trainer：Dice 指标 + 滑窗推理，迁移自 CH4_Segmentation/trainer.py
"""
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics import MetricCollection
from torchmetrics.segmentation import DiceScore

from common.base_trainer import BaseTrainer
from common.utils import SlidingAverageMeter
from torch.utils.tensorboard import SummaryWriter

class Seg2DTrainer(BaseTrainer):
    """2D segmentation trainer: metric = macro Dice (↑)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nc = self.args.num_classes
        self.metrics = MetricCollection({
            'dice_per_class': DiceScore(num_classes=nc, average='none', include_background=False),
            'dice_macro':     DiceScore(num_classes=nc, average='macro', include_background=False),
        })
        self.avg_meter = SlidingAverageMeter()

    def _one_hot(self, t):
        nc = self.args.num_classes
        oh = F.one_hot(t, num_classes=nc)
        dims = list(range(len(oh.shape)))
        dims.insert(1, dims.pop())
        return oh.permute(*dims)

    def _pred_one_hot(self, pred):
        if isinstance(pred, list):
            pred = pred[0]
        return self._one_hot(pred.argmax(dim=1))

    def train_one_epoch(self, model, optimizer, data_loader, device, epoch, tb_writer):
        model.train()
        self.metrics.reset(); self.avg_meter.reset()
        mean_loss = torch.zeros(1).to(device)
        optimizer.zero_grad()

        for it, (imgs, labels, *_) in enumerate(
                tqdm(data_loader, dynamic_ncols=True, file=sys.stdout)):
            imgs = imgs.to(device, non_blocking=True)
            if isinstance(labels, list):
                labels = [l.long().to(device, non_blocking=True) for l in labels]
            else:
                labels = labels.long().to(device, non_blocking=True)

            pred = model(imgs)
            if isinstance(pred, list) and not isinstance(labels, list):
                labels = [labels] * len(pred)
            total_loss, losses = self.loss_fn(pred, labels)

            total_loss.backward()
            if getattr(self.args, 'clip_grad_norm', False):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            optimizer.step(); optimizer.zero_grad()

            mean_loss = (mean_loss * it + total_loss.detach()) / (it + 1)
            # metrics
            true_lbl = labels[0] if isinstance(labels, list) else labels
            dm = self.metrics(self._pred_one_hot(pred), self._one_hot(true_lbl))
            self.avg_meter.update(dm, weight=imgs.shape[0])

            tb_writer.add_scalars('Train/Loss', {'mean_loss': mean_loss.item()},
                                  it + epoch * len(data_loader))
            tb_writer.add_scalars('Train/LR', {'lr': optimizer.param_groups[0]['lr']},
                                  it + epoch * len(data_loader))

        m = self.avg_meter.get_average()
        tb_writer.add_scalar('Train/DiceMacro', m['dice_macro'], epoch)
        return mean_loss.item()

    @torch.no_grad()
    def eval(self, model, data_loader, tb_writer, epoch, sce) -> float:
        self.logger.info(f'Evaluating {sce}...')
        model.eval()
        self.metrics.reset(); self.avg_meter.reset()

        for imgs, labels, *_ in tqdm(data_loader, dynamic_ncols=True, file=sys.stdout):
            imgs = imgs.to(self.device)
            labels = labels.long().to(self.device)
            pred = model(imgs)
            dm = self.metrics(self._pred_one_hot(pred), self._one_hot(labels))
            self.avg_meter.update(dm, weight=imgs.shape[0])

        m = self.avg_meter.get_average()
        dice_macro = m['dice_macro'].item()
        self.logger.info(f'[{sce}] Dice macro={dice_macro:.4f} | '
                         f'per-class={[round(v,4) for v in m["dice_per_class"].tolist()]}')
        tb_writer.add_scalar(f'{sce}/DiceMacro', dice_macro, epoch)
        tb_writer.add_scalars(f'{sce}/DiceClass',
                              {f'cls{i}': v for i, v in enumerate(m['dice_per_class'].tolist())}, epoch)
        return dice_macro

    @torch.no_grad()
    def sliding_window_inference(self, img, roi_size, sw_batch_size, overlap, tta=False):
        """
        Perform sliding window inference on `img` with `roi_size` window size.
        Args:
            img (torch.Tensor): input image tensor.
            roi_size (tuple): the size of the sliding window.
            sw_batch_size (int): the batch size for sliding window inference.
            overlap (float): the overlap between windows.
            tta (bool): whether to use test-time augmentation.
        Returns:
            torch.Tensor: the output prediction.
        """
        from monai.inferers import sliding_window_inference
        from monai.transforms import Compose, Flip, Rotate90
        import itertools

        if tta:
            mirror_axes = [0,1]
            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]
            img = torch.cat([img] + [torch.flip(img, axes) for axes in axes_combinations], dim=0)

        pred = sliding_window_inference(
            inputs=img,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=self.model,
            overlap=overlap,
            mode='gaussian'
        )
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        if tta:
            for i, axes in enumerate(axes_combinations, start=1):
                pred[i:i+1] = torch.flip(pred[i:i+1], axes)
            pred = pred.mean(dim=0, keepdim=True)

        return pred

    @torch.no_grad()
    def predict_with_sliding_window(self, data_loader, inverse_transform, roi_size, 
                                    sw_batch_size, overlap, tta=False, save_path=None):
        self.logger.info(f'*********** Test with sliding window on {self.args.test_data_path}')
        tb_writer = SummaryWriter(log_dir=self.args.tensorboard_output_path)
        self.model.eval()
        # 重置评估指标
        self.metrics.reset()
        self.avg_meter.reset()
        # 初始化两个字典来存储ED和ES的指标
        ed_metrics = SlidingAverageMeter()
        es_metrics = SlidingAverageMeter()
        # 打印验证进度
        data_loader = tqdm(data_loader, desc="Test...", dynamic_ncols=True, file=sys.stdout)
        for step, batch in enumerate(data_loader):
            imgs, labels, file_name, sizes_list = batch
            if len(file_name) != 1:
                raise ValueError('Only support single image for test')
            imgs, labels = imgs.to(self.device), labels.long().to(self.device)
            pred = self.sliding_window_inference(imgs, roi_size, sw_batch_size, overlap, tta)  # [B,C,H,W]
            ### 将pred还原为原始空间
            pred = pred.reshape(-1, *pred.shape[-2:])
            pred = inverse_transform(pred, sizes_list[0])
            pred = pred.reshape(-1, self.args.num_classes, *pred.shape[-2:])
            ### one-hot + permute
            if isinstance(labels, list):
                labels_one_hot = F.one_hot(labels[0], num_classes=self.args.num_classes)
            else:
                labels_one_hot = F.one_hot(labels, num_classes=self.args.num_classes)
            dims = list(range(len(labels_one_hot.shape)))
            dims.insert(1, dims.pop())
            labels_one_hot = labels_one_hot.permute(*dims)  # [B,C,H,W]
            # 计算评估指标
            pred_argmax = pred.argmax(dim=1)  # [B,H,W]
            pred_one_hot = F.one_hot(pred_argmax, num_classes=self.args.num_classes).permute(*dims)  # [B,C,H,W]
            dice_metric = self.metrics(pred_one_hot, labels_one_hot)
            if save_path is not None:
                pred_argmax = pred_argmax.cpu().numpy().astype(np.uint8)
                suffix = dice_metric['dice_macro']
                name = file_name[0].split('/')[-1].split('.')[0]
                import nibabel as nib
                pred_argmax_nib = nib.Nifti1Image(pred_argmax.transpose(2,1,0), np.eye(4))
                nib.save(pred_argmax_nib, os.path.join(save_path, name+f'_{suffix:.3f}.nii.gz'))
                #np.save(os.path.join(save_path, name+f'_{suffix:.3f}.npy'), pred_argmax)
            self.avg_meter.update(dice_metric, weight=imgs.shape[0])

            # 根据file_name将指标分成ED和ES两组
            if 'ED' in file_name[0]:
                ed_metrics.update(dice_metric, weight=1)
            elif 'ES' in file_name[0]:
                es_metrics.update(dice_metric, weight=1)

        # 计算平均评估指标
        metrics = self.avg_meter.get_average()
        ed_avg_metrics = ed_metrics.get_average()
        es_avg_metrics = es_metrics.get_average()

        self.logger.info(f'Average metrics: {metrics}')
        self.logger.info(f'ED metrics: {ed_avg_metrics}')
        self.logger.info(f'ES metrics: {es_avg_metrics}')

        tb_writer.add_scalars('Test/Dice_macro', {'dice_macro': metrics['dice_macro']}, 0)
        tb_writer.add_scalars('Test/Dice_class', {f'class_{class_index}': value for class_index, value in enumerate(metrics['dice_per_class'].tolist())}, 0)
        tb_writer.add_scalars('Test/ED_Dice_macro', {'dice_macro': ed_avg_metrics['dice_macro']}, 0)
        tb_writer.add_scalars('Test/ED_Dice_class', {f'class_{class_index}': value for class_index, value in enumerate(ed_avg_metrics['dice_per_class'].tolist())}, 0)
        tb_writer.add_scalars('Test/ES_Dice_macro', {'dice_macro': es_avg_metrics['dice_macro']}, 0)
        tb_writer.add_scalars('Test/ES_Dice_class', {f'class_{class_index}': value for class_index, value in enumerate(es_avg_metrics['dice_per_class'].tolist())}, 0)

        return metrics, ed_avg_metrics, es_avg_metrics