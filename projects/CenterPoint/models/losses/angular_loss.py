# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import torch.nn as nn
from mmdet.models.losses.utils import weight_reduce_loss, weighted_loss
from mmdet.registry import MODELS
from torch import Tensor
import torch.nn.functional as F

@weighted_loss
def angular_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    
    pred_vec = F.normalize(pred, dim=-1)
    gt_vec = F.normalize(target, dim=-1)
    
    # Cosine similarity
    loss = 1 - torch.sum(pred_vec * gt_vec, dim=-1, keepdim=True) 
    return loss



@MODELS.register_module(force=True)
class AngularLoss(nn.Module):
    """AmpGaussianFocalLoss is a variant of GaussianFocalLoss to increase eps to stabilize the AMP training.

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_
    Code is modified from `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    Please notice that the target in GaussianFocalLoss is a gaussian heatmap,
    not 0/1 binary target.

    Args:
        alpha (float): Power of prediction.
        gamma (float): Power of target for negative samples.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
        pos_weight(float): Positive sample loss weight. Defaults to 1.0.
        neg_weight(float): Negative sample loss weight. Defaults to 1.0.
    """

    def __init__(
        self,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * angular_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox
