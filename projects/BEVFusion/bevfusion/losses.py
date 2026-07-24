# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import torch.nn as nn
from mmdet.models.losses.utils import weight_reduce_loss, weighted_loss
from mmdet3d.registry import MODELS
from torch import Tensor
from torch.nn import functional as F


@weighted_loss
def bce_with_logits_loss(
    pred: Tensor,
    target: Tensor,
    weight: Tensor, 
    reduction: str = 'mean',
    avg_factor: Optional[int] = None,
    pos_weight: Optional[Tensor] = None
) -> Tensor:
    """
    """
    
    losses = F.binary_cross_entropy_with_logits(
      pred,
      target,
      None, # Always None since the weight will be used in the weighted_loss wrapper
      pos_weight=pos_weight,
      reduction='none', # Always none since the reduction will happen in the weighted_loss wrapper
    )
    return losses


@MODELS.register_module()
class CustomBCEWithLogitsLoss(nn.Module):
    """BCEWithLogitsLoss"""

    def __init__(
        self,
        weight=None, 
        reduction='mean', 
        pos_weight=None
    ) -> None:
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
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

        losses = bce_with_logits_loss(
          pred,
          target,
          weight, 
          reduction=reduction, 
          avg_factor=avg_factor,
          pos_weight=self.pos_weight
        )
        return losses
