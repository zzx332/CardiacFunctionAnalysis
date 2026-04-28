"""
tasks/landmark/trainer.py
关键点检测 Trainer：MSE 热图损失，欧氏距离 + SDR 评估指标
迁移自 Cardiac_Landmark/trainer.py
"""
import sys
import json
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm

from common.base_trainer import BaseTrainer


def _debug_log(hypothesis_id, location, message, data):
    # #region agent log
    try:
        with open('/home/zzx/Cardiac_Function_Analysis/.cursor/debug-aceb60.log', 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'sessionId': 'aceb60',
                'runId': 'pre-fix',
                'hypothesisId': hypothesis_id,
                'location': location,
                'message': message,
                'data': data,
                'timestamp': int(time.time() * 1000),
            }, ensure_ascii=False) + '\n')
    except Exception:
        pass
    # #endregion


class LandmarkTrainer(BaseTrainer):
    """Landmark detection trainer: loss=MSE on heatmaps; metric=Euclidean dist (↓)."""

    def is_better(self, new_metric, best_metric):
        # 欧氏距离越小越好
        return new_metric < best_metric

    def train_one_epoch(self, model, optimizer, data_loader, device, epoch, tb_writer):
        model.train()
        mean_loss = torch.zeros(1).to(device)
        optimizer.zero_grad()

        for it, (imgs, labels) in enumerate(tqdm(data_loader, dynamic_ncols=True, file=sys.stdout)):
            imgs = imgs.to(device, non_blocking=True)
            if isinstance(labels, list):
                labels = [l.to(device, non_blocking=True) for l in labels]
            else:
                labels = labels.to(device, non_blocking=True)

            pred = model(imgs)
            if isinstance(pred, list) and not isinstance(labels, list):
                labels = [labels] * len(pred)

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

        return mean_loss.item()

    @torch.no_grad()
    def eval(self, model, data_loader, tb_writer, epoch, sce) -> float:
        """返回平均欧氏距离（越小越好）。"""
        self.logger.info(f'Evaluating {sce}...')
        model.eval()
        H = W = getattr(self.args, 'target_size', (160, 160))[0]
        device = self.device

        # 期望坐标解码用的网格
        y_coords = torch.arange(H, device=device).float()
        x_coords = torch.arange(W, device=device).float()
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        grid_y = grid_y.reshape(-1)
        grid_x = grid_x.reshape(-1)

        all_dists, all_sdr3, all_sdr5 = [], [], []
        for batch_idx, (imgs, landmarks) in enumerate(tqdm(data_loader, dynamic_ncols=True, file=sys.stdout)):
            imgs = imgs.to(device)
            pred = model(imgs)
            if isinstance(pred, list):
                pred = pred[-1]

            B, C, _, _ = pred.shape
            flat = pred.reshape(B, C, -1)
            norm = flat.sum(dim=-1, keepdim=True) + 1e-6
            x_exp = (grid_x * flat).sum(-1) / norm.squeeze(-1)
            y_exp = (grid_y * flat).sum(-1) / norm.squeeze(-1)
            coords = torch.stack([x_exp, y_exp], dim=-1).cpu()   # [B, K, 2]

            lm = landmarks.float()
            if batch_idx == 0:
                # #region agent log
                _debug_log(
                    'H2',
                    'tasks/landmark/trainer.py:100',
                    'eval first batch tensor shapes',
                    {
                        'pred_shape': list(pred.shape),
                        'coords_shape': list(coords.shape),
                        'landmarks_shape': list(lm.shape),
                        'landmarks_dtype': str(lm.dtype),
                    }
                )
                # #endregion
            if lm.dim() != 3 or lm.shape[-1] != 2:
                # #region agent log
                _debug_log(
                    'H2',
                    'tasks/landmark/trainer.py:114',
                    'landmarks shape incompatible with coordinate distance',
                    {
                        'coords_shape': list(coords.shape),
                        'landmarks_shape': list(lm.shape),
                        'expected_last_dim': 2,
                    }
                )
                # #endregion
            dist = torch.norm(lm - coords, dim=-1)  # [B, K]
            all_dists.append(dist)
            all_sdr3.append((dist <= 3.0).float().mean())
            all_sdr5.append((dist <= 5.0).float().mean())

        avg_dist = torch.cat(all_dists).mean().item()
        avg_sdr3 = torch.stack(all_sdr3).mean().item()
        avg_sdr5 = torch.stack(all_sdr5).mean().item()

        self.logger.info(f'[{sce}] Avg Euclidean: {avg_dist:.3f}px | '
                         f'SDR@3mm: {avg_sdr3:.3f} | SDR@5mm: {avg_sdr5:.3f}')
        tb_writer.add_scalar(f'{sce}/EuclideanDist', avg_dist, epoch)
        tb_writer.add_scalar(f'{sce}/SDR_3mm', avg_sdr3, epoch)
        tb_writer.add_scalar(f'{sce}/SDR_5mm', avg_sdr5, epoch)

        return avg_dist
