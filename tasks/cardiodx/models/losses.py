# ！/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = "Lxx"

from typing import Optional, Tuple, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.config import LoggerSingleton

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

class BatchCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=-100, reduction='mean'):
        super(BatchCrossEntropyLoss, self).__init__()
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
        
        return F.cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)

class MyLoss(nn.Module):
    def __init__(self, loss_config: Optional[Dict[str, Union[float, Dict]]] = None):
        super(MyLoss, self).__init__()
        self.loss_config = loss_config if loss_config is not None else {}
        self.loss_functions = {
            'cross_entropy_loss': CrossEntropyLoss,
            'batch_cross_entropy_loss': BatchCrossEntropyLoss,
        }
        self.loss_weights = []
        self.loss_fns = []
        logger = LoggerSingleton()
        for loss_name, config in self.loss_config.items():
            if loss_name in self.loss_functions:
                weight = config.pop('weight', 1.0)
                self.loss_weights.append(weight)
                loss_fn = self.loss_functions[loss_name](**config)
                self.loss_fns.append((loss_name, loss_fn))
                logger.info((loss_name,loss_fn))
                logger.info(weight)
                logger.info(vars(loss_fn))

    def forward(self, y_pred, y_true):
        losses = {}
        total_loss = 0

        for weight, (loss_name,loss_fn) in zip(self.loss_weights, self.loss_fns):
            loss_value = loss_fn(y_pred, y_true)
            losses[loss_name] = loss_value
            total_loss += weight * loss_value

        return total_loss, losses
    

class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        Wraps a loss function so that it can be applied to multiple outputs. Forward accepts an arbitrary number of
        inputs. Each input is expected to be a tuple/list. Each tuple/list must have the same length. The loss is then
        applied to each entry like this:
        l = w0 * loss(input0[0], input1[0], ...) +  w1 * loss(input0[1], input1[1], ...) + ...
        If weights are None, all w will be 1.
        """
        super(DeepSupervisionWrapper, self).__init__()
        assert any([x != 0 for x in weight_factors]), "At least one weight factor should be != 0.0"
        self.weight_factors = tuple(weight_factors)
        self.loss = loss

    def forward(self, *args):
        assert all([isinstance(i, (tuple, list)) for i in args]), \
            f"all args must be either tuple or list, got {[type(i) for i in args]}"
        # we could check for equal lengths here as well, but we really shouldn't overdo it with checks because
        # this code is executed a lot of times!

        if self.weight_factors is None:
            weights = (1, ) * len(args[0])
        else:
            weights = self.weight_factors
        
        total_loss = 0
        all_losses = {}

        for i, inputs in enumerate(zip(*args)):
            if weights[i] != 0.0:
                loss_value, loss_dict = self.loss(*inputs)
                total_loss += weights[i] * loss_value
                for key, value in loss_dict.items():
                    if key not in all_losses:
                        all_losses[key] = weights[i] * value
                    else:
                        all_losses[key] += weights[i] * value

        return total_loss, all_losses