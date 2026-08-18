# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import torch
import torch.nn as nn
from mmdet3d.registry import MODELS
from mmdet.models.losses.utils import weighted_loss
from torch import Tensor


@weighted_loss
def iou_loss(pred: Tensor, target: Tensor) -> Tensor:
    """ """

    losses = target - pred
    return losses


@MODELS.register_module()
class RotatedBEVIOULoss(nn.Module):
    """Compute rotated GIOU loss between predictions and gt boxes."""

    def __init__(
        self,
        loss_weight=1.0,
        reduction="mean",
    ) -> None:

        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(
        self,
        ious: Tensor,
        weight: Tensor,
        avg_factor: Optional[Union[int, float]] = None,
        reduction_override: Optional[str] = None,
    ) -> Tensor:
        """
        preds_bboxes (B, num_proposals, 10)
        gts_bboxes (B, num_proposals, 10)
        labels (B, num_proposals, )
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        targets = torch.ones_like(ious)

        losses = iou_loss(
            ious,
            targets,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
        )
        return self.loss_weight * losses
