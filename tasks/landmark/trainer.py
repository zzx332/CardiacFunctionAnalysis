"""
tasks/landmark/trainer.py
关键点检测 Trainer：MSE 热图损失，欧氏距离 + SDR 评估指标
迁移自 Cardiac_Landmark/trainer.py
"""
import sys
import os
import json
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

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

    @torch.no_grad()
    def predict_test(self, data_loader):
        self.model.eval()
        self.logger.info(f'*********** predict on {self.args.test_data_path}')
        test_data_name = os.path.basename(self.args.test_data_path).split('.')[0]
        tb_writer = SummaryWriter(log_dir=self.args.tensorboard_output_path)

        H, W = getattr(self.args, 'target_size', (160, 160))
        device = self.device
        y_coords = torch.arange(H, device=device).float()
        x_coords = torch.arange(W, device=device).float()
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        grid_y = grid_y.reshape(-1)
        grid_x = grid_x.reshape(-1)

        soft_dists, soft_sdr3, soft_sdr5 = [], [], []
        hard_dists, hard_sdr3, hard_sdr5 = [], [], []
        sample_count = 0
        for imgs, landmarks in tqdm(data_loader, desc="predict...", dynamic_ncols=True, file=sys.stdout):
            imgs = imgs.to(device, non_blocking=True)
            pred = self.model(imgs)
            if isinstance(pred, list):
                pred = pred[-1]

            bsz, channels, _, _ = pred.shape
            flat = pred.reshape(bsz, channels, -1)

            # 普通版本：argmax 解码坐标
            hard_index = torch.argmax(flat, dim=-1)
            rows, cols = torch.unravel_index(hard_index, (H, W))
            hard_coords = torch.stack([cols, rows], dim=-1).cpu().float()

            # 高斯拟合版本：期望坐标解码
            norm = flat.sum(dim=-1, keepdim=True) + 1e-6
            x_exp = (grid_x * flat).sum(-1) / norm.squeeze(-1)
            y_exp = (grid_y * flat).sum(-1) / norm.squeeze(-1)
            soft_coords = torch.stack([x_exp, y_exp], dim=-1).cpu()

            lm = landmarks.float()
            hard_dist = torch.norm(lm - hard_coords, dim=-1)
            soft_dist = torch.norm(lm - soft_coords, dim=-1)

            hard_dists.append(hard_dist)
            hard_sdr3.append((hard_dist <= 3.0).float().mean())
            hard_sdr5.append((hard_dist <= 5.0).float().mean())

            soft_dists.append(soft_dist)
            soft_sdr3.append((soft_dist <= 3.0).float().mean())
            soft_sdr5.append((soft_dist <= 5.0).float().mean())
            sample_count += bsz

        hard_avg_dist = torch.cat(hard_dists).mean().item()
        hard_avg_sdr3 = torch.stack(hard_sdr3).mean().item()
        hard_avg_sdr5 = torch.stack(hard_sdr5).mean().item()
        soft_avg_dist = torch.cat(soft_dists).mean().item()
        soft_avg_sdr3 = torch.stack(soft_sdr3).mean().item()
        soft_avg_sdr5 = torch.stack(soft_sdr5).mean().item()
        metrics = {
            'hard_avg_euclidean_distance': hard_avg_dist,
            'hard_sdr_3mm': hard_avg_sdr3,
            'hard_sdr_5mm': hard_avg_sdr5,
            'soft_avg_euclidean_distance': soft_avg_dist,
            'soft_sdr_3mm': soft_avg_sdr3,
            'soft_sdr_5mm': soft_avg_sdr5,
            'num_samples': sample_count,
        }

        os.makedirs(self.args.model_output_path, exist_ok=True)
        result_file = os.path.join(self.args.model_output_path, f'{test_data_name}_test_results.txt')
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write("Landmark Test Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"checkpoint: {getattr(self.args, 'checkpoint', '')}\n")
            f.write(f"target_size: {tuple(getattr(self.args, 'target_size', (160, 160)))}\n")
            f.write(f"num_samples: {sample_count}\n\n")
            f.write("Argmax Metrics:\n")
            f.write(f"avg_euclidean_distance: {hard_avg_dist:.6f}\n")
            f.write(f"sdr_3mm: {hard_avg_sdr3:.6f}\n")
            f.write(f"sdr_5mm: {hard_avg_sdr5:.6f}\n\n")
            f.write("Gaussian-Fit Metrics:\n")
            f.write(f"avg_euclidean_distance: {soft_avg_dist:.6f}\n")
            f.write(f"sdr_3mm: {soft_avg_sdr3:.6f}\n")
            f.write(f"sdr_5mm: {soft_avg_sdr5:.6f}\n")

        tb_writer.add_scalar(f'Test{test_data_name}/Argmax_EuclideanDist', hard_avg_dist, 0)
        tb_writer.add_scalar(f'Test{test_data_name}/Argmax_SDR_3mm', hard_avg_sdr3, 0)
        tb_writer.add_scalar(f'Test{test_data_name}/Argmax_SDR_5mm', hard_avg_sdr5, 0)
        tb_writer.add_scalar(f'Test{test_data_name}/Gaussian_EuclideanDist', soft_avg_dist, 0)
        tb_writer.add_scalar(f'Test{test_data_name}/Gaussian_SDR_3mm', soft_avg_sdr3, 0)
        tb_writer.add_scalar(f'Test{test_data_name}/Gaussian_SDR_5mm', soft_avg_sdr5, 0)
        self.logger.info(
            f"Argmax metrics: avg_euclidean_distance={hard_avg_dist:.4f}, "
            f"sdr_3mm={hard_avg_sdr3:.4f}, sdr_5mm={hard_avg_sdr5:.4f}"
        )
        self.logger.info(
            f"Gaussian-fit metrics: avg_euclidean_distance={soft_avg_dist:.4f}, "
            f"sdr_3mm={soft_avg_sdr3:.4f}, sdr_5mm={soft_avg_sdr5:.4f}"
        )
        self.logger.info(f"Saved test report: {result_file}")
        return metrics
