from typing import Optional, Union
from mmdet.models.losses.smooth_l1_loss import L1Loss as _L1Loss

import torch
import torch.nn as nn
from torch import Tensor 

from mmdet3d.registry import MODELS


@MODELS.register_module()
class CustomL1Loss(_L1Loss):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super(CustomL1Loss, self).__init__(
					reduction=reduction,
					loss_weight=loss_weight
				)

    # @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Calculated loss
        """
        losses = super().forward(
					pred=pred,
					target=target,
					weight=weight, 
					avg_factor=avg_factor,
					reduction_override=reduction_override
				)
        
        return losses