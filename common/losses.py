"""
common/losses.py
统一损失函数库，合并自：
  - Cardiac_Landmark/models/losses.py   (MSELoss, MyLoss, DeepSupervisionWrapper)
  - CH4_Segmentation/models/losses.py   (MultiClassBatchDiceLoss, CELoss)
  - SAX_Segmentation/loss.py            (MulticlassDiceLoss, MulticlassBatchDiceLoss)
  - Cardiac_View/models/losses.py       (CrossEntropyLoss)
"""
from typing import Optional, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LoggerSingleton


# ---------------------------------------------------------------------------
# 基础损失
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss 实现

        参数:
            alpha (Tensor, optional): 类别权重，形状为 (num_classes,)。默认为 None。
            gamma (float): 调节因子，默认为 2.0。
            reduction (str): 损失 reduction 方式，支持 'mean'、'sum' 或 'none'。
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        """
        计算 Focal Loss

        参数:
            y_pred (Tensor): 模型预测的 logits，形状为 (batch_size, num_classes)。
            y_true (Tensor): 真实标签，形状为 (batch_size,)，每个值为类别索引。

        返回:
            loss (Tensor): 计算得到的 Focal Loss。
        """
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(y_pred, y_true, reduction='none')

        # 计算概率 p_t
        p_t = torch.exp(-ce_loss)

        # 计算 Focal Loss
        focal_loss = (1 - p_t) ** self.gamma * ce_loss

        # 应用类别权重
        if self.alpha is not None:
            alpha_t = self.alpha[y_true]
            focal_loss = alpha_t * focal_loss

        # 根据 reduction 方式返回损失
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MSELoss(nn.Module):
    """热图回归损失（Landmark 任务）"""
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert input.shape == target.shape, "input and target must have the same shape"
        return F.mse_loss(input, target)


class CrossEntropyLoss(nn.Module):
    """分类 / 分割用 CE Loss（target 可为 one-hot 或 class index）"""
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        input: [B, C, ...]
        target: [B, C, ...] one-hot  or  [B, ...] class-index
        """
        return nn.CrossEntropyLoss()(input, target)


class DiceLoss(nn.Module):
    """
    多类别 Batch Dice Loss（SAX_Segmentation 风格）
    input: [B, C, H, W] or [B, C, D, H, W]，未经 softmax
    target: [B, C, H, W] or [B, C, D, H, W]，one-hot
    """
    def __init__(self, batch_dice: bool = True):
        super().__init__()
        self.batch_dice = batch_dice

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = F.softmax(input, dim=1)
        if self.batch_dice:
            num_classes = input.shape[1]
            # flatten 空间维度，再做 batch dice
            flat_input = input.permute(0, *range(2, input.dim()), 1).reshape(-1, num_classes)
            flat_target = target.permute(0, *range(2, target.dim()), 1).reshape(-1, num_classes)
            intersect = torch.sum(flat_input * flat_target, 0)
            denominator = torch.sum(flat_input, 0) + torch.sum(flat_target, 0)
            dice_scores = 2 * intersect / (denominator + 1e-6)
        else:
            reduce_dims = tuple(range(2, input.dim()))
            intersect = torch.sum(input * target, reduce_dims)
            denominator = torch.sum(input, reduce_dims) + torch.sum(target, reduce_dims)
            dice_scores = 2 * intersect / (denominator + 1e-6)
            dice_scores = dice_scores.mean(dim=0)
        return 1 - dice_scores.mean()

class MultiClassBatchDiceLoss(nn.Module):
    def __init__(self, class_weights=None, include_background=True):
        super().__init__()
        self.weights = class_weights
        self.include_background = include_background
    def forward(self, input, target):
        """
        input: [n,c,h,w] or [n,c,d,h,w]
        target: [n,h,w] or [n,d,h,w]
        requires one hot encoded target. Applies DiceLoss on each class iteratively.
        requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
        batch size and C is number of classes
        """
        
        if len(input.shape) not in [3, 4]:
            raise ValueError("Input and target must be 3D or 4D tensors.")
        
        num_classes = input.shape[1]

        # 确保通道维度在最后
        dims = list(range(len(input.shape)))
        dims.append(dims.pop(1))  # 将通道维度移到最后一位

        # 软化输入并调整维度
        input_soft = F.softmax(input, dim=1)
        input_reshaped = input_soft.permute(*dims).reshape((-1, num_classes))
        ### one-hot + permute
        target = F.one_hot(target.long(), num_classes=num_classes) # [B,H,W,C]
        target_reshaped = target.reshape((-1, num_classes))
        
        # 计算交集和分母
        intersect = torch.sum(input_reshaped * target_reshaped, dim=0)  # [num_class]
        denominator = torch.sum(input_reshaped, dim=0) + torch.sum(target_reshaped, dim=0)
        
        # 避免除零错误
        smooth = 1
        dice_scores = (2 * intersect + smooth)/ (denominator + smooth)
        
        # 应用权重
        if self.weights is not None:
            if self.weights.shape[0] != num_classes:
                raise ValueError("Weights must have the same length as the number of classes.")
            dice_scores = dice_scores * self.weights
        
        # 排除背景类
        if not self.include_background:
            dice_scores = dice_scores[1:]  # 排除背景类（假设背景类是第0类）

        return 1 - dice_scores.mean()
# ---------------------------------------------------------------------------
# 组合损失
# ---------------------------------------------------------------------------
class MyLoss(nn.Module):
    """按配置字典组合多个损失函数，支持加权求和。"""
    REGISTRY = {
        'mse_loss': MSELoss,
        'dice_loss': DiceLoss,
        'focal_loss': FocalLoss,
        'multiclass_batch_dice_loss': MultiClassBatchDiceLoss,
        'ce_loss': CrossEntropyLoss,
    }

    def __init__(self, loss_config: Dict[str, Union[float, Dict]]):
        super().__init__()
        assert loss_config, "loss_config must be provided"
        logger = LoggerSingleton()
        self.loss_weights = []
        self.loss_fns = []
        for loss_name, config in loss_config.items():
            if loss_name not in self.REGISTRY:
                raise ValueError(f"Unknown loss function: {loss_name}. "
                                 f"Available: {list(self.REGISTRY.keys())}")
            cfg = dict(config)  # shallow copy
            weight = cfg.pop('weight', 1.0)
            self.loss_weights.append(weight)
            fn = self.REGISTRY[loss_name](**cfg)
            self.loss_fns.append((loss_name, fn))
            logger.info(f"Loss: {loss_name}, weight={weight}")

    def forward(self, y_pred, y_true):
        total_loss = 0
        losses = {}
        for weight, (name, fn) in zip(self.loss_weights, self.loss_fns):
            v = fn(y_pred, y_true)
            losses[name] = v
            total_loss = total_loss + weight * v
        return total_loss, losses


class DeepSupervisionWrapper(nn.Module):
    """深度监督包装：对多尺度输出分别计算损失并加权求和。"""

    def __init__(self, loss: nn.Module, weight_factors=None):
        super().__init__()
        assert any(w != 0 for w in weight_factors), \
            "At least one weight factor should be != 0.0"
        self.weight_factors = tuple(weight_factors)
        self.loss = loss

    def forward(self, *args):
        # args 为 (preds_list, labels_list) 或 (preds_list, label_tensor)
        assert all(isinstance(a, (tuple, list)) for a in args), \
            f"all args must be list or tuple, got {[type(a) for a in args]}"
        total_loss = 0
        all_losses: Dict = {}
        for i, inputs in enumerate(zip(*args)):
            w = self.weight_factors[i] if i < len(self.weight_factors) else 0.0
            if w != 0.0:
                loss_val, loss_dict = self.loss(*inputs)
                total_loss = total_loss + w * loss_val
                for k, v in loss_dict.items():
                    all_losses[k] = all_losses.get(k, 0) + w * v
        return total_loss, all_losses
