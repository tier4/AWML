# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import torch.nn as nn
from mmdet.models.losses.utils import weight_reduce_loss, weighted_loss
from mmdet.registry import MODELS
from torch import Tensor
from torch.nn import BCEWithLogitsLoss


@MODELS.register_module(force=True)
class BCEWithLogitsLoss(nn.Module):
    """BCEWithLogitsLoss"""

    def __init__(
        self,
				weight=None, 
				size_average=None, 
				reduce=None, 
				reduction='mean', 
				pos_weight=None
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pos_inds: Optional[Tensor] = None,
        pos_labels: Optional[Tensor] = None,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[Union[int, float]] = None,
        reduction_override: Optional[str] = None,
    ) -> Tensor:
        """Forward function.

        If you want to manually determine which positions are
        positive samples, you can set the pos_index and pos_label
        parameter. Currently, only the CenterNet update version uses
        the parameter.

        Args:
            pred (torch.Tensor): The prediction. The shape is (N, num_classes).
            target (torch.Tensor): The learning target of the prediction
                in gaussian distribution. The shape is (N, num_classes).
            pos_inds (torch.Tensor): The positive sample index.
                Defaults to None.
            pos_labels (torch.Tensor): The label corresponding to the positive
                sample index. Defaults to None.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, float, optional): Average factor that is used to
                average the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if pos_inds is not None:
            assert pos_labels is not None
            # Only used by centernet update version
            loss_reg = self.loss_weight * gaussian_focal_loss_with_pos_inds(
                pred,
                target,
                pos_inds,
                pos_labels,
                alpha=self.alpha,
                gamma=self.gamma,
                pos_weight=self.pos_weight,
                neg_weight=self.neg_weight,
                reduction=reduction,
                avg_factor=avg_factor,
            )
        else:
            loss_reg = self.loss_weight * gaussian_focal_loss(
                pred,
                target,
                weight,
                alpha=self.alpha,
                gamma=self.gamma,
                pos_weight=self.pos_weight,
                neg_weight=self.neg_weight,
                reduction=reduction,
                avg_factor=avg_factor,
            )
        return loss_reg
