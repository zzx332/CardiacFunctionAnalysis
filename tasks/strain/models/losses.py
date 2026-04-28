import torch
import torch.nn.functional as F
import numpy as np
import math


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

import torch.nn as nn
class MultiClassBatchDiceLoss(nn.Module):
    def __init__(self, class_weights=None, include_background=True, num_classes=3):
        super(MultiClassBatchDiceLoss, self).__init__()
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
