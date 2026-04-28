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
    def __init__(self, class_weights=None, ignore_index=-100, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.weight = class_weights
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [b] or [b,1]
        target: [b] or [b,1]
        """
        input = input.squeeze().float()
        target = target.squeeze().float()
        if len(input.shape)!=1 or len(target.shape)!=1:
            raise ValueError("Input and Target must be 1D tensor.")
        weight = torch.ones_like(target)
        weight[target == 0] = self.weight[0]
        loss_fn = nn.BCEWithLogitsLoss(weight=weight)
        return loss_fn(input, target)

class CrossEntropyLoss_ch4seg(nn.Module):
    def __init__(self, class_weights=None, ignore_index=-100, reduction='mean'):
        super(CrossEntropyLoss_ch4seg, self).__init__()
        self.weight = class_weights
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [n,c,h,w] or [n,c,d,h,w]
        target: [n,h,w] or [n,d,h,w]
        """
        if len(input.shape) not in [4, 5]:
            raise ValueError("Input must be 4D or 5D tensor.")
        
        if len(target.shape) not in [3, 4]:
            raise ValueError("Target must be 3D or 4D tensor.")
        
        return F.cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
  
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

class MulticlassBatchDiceLoss_seg3d(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input, target):
        """
        input: [n,c,h,w] or [n,c,d,h,w]
        target: [n,c,h,w] or [n,c,d,h,w]
        requires one hot encoded target. Applies DiceLoss on each class iteratively.
        requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
        batch size and C is number of classes
        """
        num_classes = input.shape[1]
        input = F.softmax(input, dim=1).permute(0, 2, 3, 1).reshape((-1, num_classes))
        target = target.permute(0, 2, 3, 1).reshape((-1, num_classes))
        intersect = torch.sum(input * target, 0) # [num_class]
        denominator = torch.sum(input, 0) + torch.sum(target, 0)
        dice_scores = 2 * intersect / (denominator + 1e-6)
        return 1-dice_scores.mean()
# ---------------------------------------------------------------------------
# strain 专用损失
# ---------------------------------------------------------------------------

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """
    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)

class MultiClassBatchDiceLoss_strain(nn.Module):
    def __init__(self, class_weights=None, include_background=True, num_classes=3):
        super(MultiClassBatchDiceLoss_strain, self).__init__()
        self.weights = class_weights
        self.include_background = include_background
        self.num_classes = num_classes
    def forward(self, input, target):
        """
        input: [n,h,w] or [n,d,h,w]
        target: [n,h,w] or [n,d,h,w]
        requires one hot encoded target. Applies DiceLoss on each class iteratively.
        requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
        batch size and C is number of classes
        """
        
        if len(input.shape) not in [3, 4]:
            raise ValueError("Input and target must be 3D or 4D tensors.")

        # one-hot + permute
        input = F.one_hot(input.long(), num_classes=self.num_classes) # [B,H,W,C]
        input_reshaped = input.reshape((-1, self.num_classes))
        ### one-hot + permute
        target = F.one_hot(target.long(), num_classes=self.num_classes) # [B,H,W,C]
        target_reshaped = target.reshape((-1, self.num_classes))
        
        # 计算交集和分母
        intersect = torch.sum(input_reshaped * target_reshaped, dim=0)  # [num_class]
        denominator = torch.sum(input_reshaped, dim=0) + torch.sum(target_reshaped, dim=0)
        
        # 避免除零错误
        smooth = 1
        dice_scores = (2 * intersect + smooth)/ (denominator + smooth)
        
        # 应用权重
        if self.weights is not None:
            if self.weights.shape[0] != self.num_classes:
                raise ValueError("Weights must have the same length as the number of classes.")
            dice_scores = dice_scores * self.weights.to(dice_scores.device)
        
        # 排除背景类
        if not self.include_background:
            dice_scores = dice_scores[1:]  # 排除背景类（假设背景类是第0类）

        return 1 - dice_scores.mean()
    
class RadialDirectionLoss(nn.Module):
    def forward(self, flow, myocardium_mask, directions):
            """
            计算心肌位移场方向约束损失
            Args:
                flow: [B, 2, H, W] 位移场
                myocardium_mask: [B, H, W] 心肌二值mask
            Returns:
                direction_loss: 标量
            """
            image_center = torch.tensor([s//2 for s in flow.shape[-2:]], 
                                       dtype=flow.dtype, 
                                       device=flow.device)
            # 获取心肌区域的坐标
            batch_coords = torch.nonzero(myocardium_mask, as_tuple=False)
            b, *pos = batch_coords.unbind(dim=1)
            
            # 获取对应位置的位移向量
            displacements = flow[b, :, pos[0], pos[1]]  # [N, 3]
            
            # 计算从心肌点到图像中心的向量（理想位移方向）
            all_voxel_directions = torch.gather(torch.tensor(directions).to(b.device), dim=0, index=b)
            coords = torch.stack(pos, dim=1)  # 关键操作
            radial_vectors =  all_voxel_directions[:,None]*(coords - image_center) # [N, 3]
            radial_vectors = F.normalize(radial_vectors, p=2, dim=1)
            
            # 归一化实际位移向量
            displacements_norm = F.normalize(displacements, p=2, dim=1)
            
            # 计算余弦相似度（最大化方向一致性）
            cosine_sim = torch.sum(radial_vectors * displacements_norm, dim=1)
            direction_loss = 1 - torch.mean(cosine_sim)  # 越小表示方向越一致
            
            return direction_loss

class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()

class RadialDirectionLoss(nn.Module):
    def forward(self, flow, myocardium_mask, directions):
        """
        计算心肌位移场方向约束损失
        Args:
            flow: [B, 2, H, W] 位移场
            myocardium_mask: [B, H, W] 心肌二值mask
        Returns:
            direction_loss: 标量
        """
        image_center = torch.tensor([s//2 for s in flow.shape[-2:]], 
                                    dtype=flow.dtype, 
                                    device=flow.device)
        # 获取心肌区域的坐标
        batch_coords = torch.nonzero(myocardium_mask, as_tuple=False)
        b, *pos = batch_coords.unbind(dim=1)
        
        # 获取对应位置的位移向量
        displacements = flow[b, :, pos[0], pos[1]]  # [N, 3]
        
        # 计算从心肌点到图像中心的向量（理想位移方向）
        all_voxel_directions = torch.gather(torch.tensor(directions).to(b.device), dim=0, index=b)
        coords = torch.stack(pos, dim=1)  # 关键操作
        radial_vectors =  all_voxel_directions[:,None]*(coords - image_center) # [N, 3]
        radial_vectors = F.normalize(radial_vectors, p=2, dim=1)
        
        # 归一化实际位移向量
        displacements_norm = F.normalize(displacements, p=2, dim=1)
        
        # 计算余弦相似度（最大化方向一致性）
        cosine_sim = torch.sum(radial_vectors * displacements_norm, dim=1)
        direction_loss = 1 - torch.mean(cosine_sim)  # 越小表示方向越一致
        
        return direction_loss

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
        'ce_loss_ch4seg': CrossEntropyLoss_ch4seg,
        'multiclass_batch_dice_loss_seg3d': MulticlassBatchDiceLoss_seg3d,
        'radial_direction_loss': RadialDirectionLoss,
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
