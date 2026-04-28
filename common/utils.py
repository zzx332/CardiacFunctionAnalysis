"""
common/utils.py
通用工具：SlidingAverageMeter, PolyLRScheduler, git 工具函数
合并自：Cardiac_Landmark/utils.py, CH4_Segmentation/utils.py, Cardiac_View/utils.py
"""
import subprocess
import re
from typing import Dict

import torch
from torch.optim.lr_scheduler import _LRScheduler
from termcolor import colored


# ---------------------------------------------------------------------------
# 滑动平均指标
# ---------------------------------------------------------------------------
class SlidingAverageMeter:
    """按样本权重累计均值，支持 Tensor 和标量。"""

    def __init__(self):
        self.total_values: Dict[str, torch.Tensor] = {}
        self.total_weights: Dict[str, float] = {}

    def update(self, metrics: Dict[str, torch.Tensor], weight: float = 1.0):
        for key, value in metrics.items():
            if key not in self.total_values:
                self.total_values[key] = torch.zeros_like(value)
                self.total_weights[key] = 0.0
            self.total_values[key] += value * weight
            self.total_weights[key] += weight

    def get_average(self) -> Dict[str, torch.Tensor]:
        return {
            key: (value / self.total_weights[key]
                  if self.total_weights[key] != 0 else torch.zeros_like(value))
            for key, value in self.total_values.items()
        }

    def reset(self):
        self.total_values = {}
        self.total_weights = {}


# ---------------------------------------------------------------------------
# 多项式学习率衰减
# ---------------------------------------------------------------------------
class PolyLRScheduler(_LRScheduler):
    """lr = initial_lr * (1 - step / max_steps) ^ exponent"""

    def __init__(self, optimizer, initial_lr: float, max_steps: int,
                 exponent: float = 0.9, current_step: int = None):
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1
        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for pg in self.optimizer.param_groups:
            pg['lr'] = new_lr


# ---------------------------------------------------------------------------
# Git 工具
# ---------------------------------------------------------------------------
def get_git_commit_id() -> str | None:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).strip().decode('utf-8')
    except subprocess.CalledProcessError:
        return None


def check_git_status():
    try:
        status = subprocess.check_output(['git', 'status', '--porcelain']).strip()
        if status:
            print(colored(
                "There are uncommitted changes in the git repository. "
                "Please commit or stash them before running.", 'red'))
            exit(1)
    except subprocess.CalledProcessError:
        print(colored("Failed to check git status.", 'red'))
        exit(1)

def center_by_centroid(img, mask, output_size=(128, 128)):
    """
    Center img and mask based on the centroid of the mask in each D slice,
    and crop/pad to output_size.
    
    Args:
        img: Tensor of shape [D, H, W]
        mask: Tensor of shape [D, H, W] (binary mask)
        output_size: Tuple of (height, width) for output
        
    Returns:
        Centered img and mask tensors of shape [D, output_height, output_width]
    """
    assert img.shape == mask.shape, "img and mask must have the same shape"
    D, H, W = img.shape
    out_h, out_w = output_size
    
    # Initialize output tensors
    centered_img = torch.zeros((D, out_h, out_w), dtype=img.dtype, device=img.device)
    centered_mask = torch.zeros((D, out_h, out_w), dtype=mask.dtype, device=mask.device)
    
    for d in range(D):
        # Calculate centroid for current slice
        slice_mask = mask[d%(D//2) + (D//2)].cpu().numpy()
        coords = np.argwhere(slice_mask == 1)
        
        if len(coords) > 0:
            cy, cx = coords.mean(axis=0).astype(int)  # (y, x)
        else:
            cy, cx = H // 2, W // 2  # Fallback to center
        
        # Calculate translation needed to center the centroid
        translate_y = out_h // 2 - cy
        translate_x = out_w // 2 - cx
        
        # Calculate source and target regions
        src_y_start = max(0, -translate_y)
        src_y_end = min(H, out_h - translate_y)
        src_x_start = max(0, -translate_x)
        src_x_end = min(W, out_w - translate_x)
        
        tgt_y_start = max(0, translate_y)
        tgt_y_end = min(out_h, H + translate_y)
        tgt_x_start = max(0, translate_x)
        tgt_x_end = min(out_w, W + translate_x)
        
        # Copy valid regions
        if src_y_end > src_y_start and src_x_end > src_x_start:
            centered_img[d, tgt_y_start:tgt_y_end, tgt_x_start:tgt_x_end] = \
                img[d, src_y_start:src_y_end, src_x_start:src_x_end]
            centered_mask[d, tgt_y_start:tgt_y_end, tgt_x_start:tgt_x_end] = \
                mask[d, src_y_start:src_y_end, src_x_start:src_x_end]
    
    return centered_img, centered_mask