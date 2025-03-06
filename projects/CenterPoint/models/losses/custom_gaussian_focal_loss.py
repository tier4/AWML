from typing import Optional, Union
from mmdet.models.losses.gaussian_focal_loss import GaussianFocalLoss as _GaussianFocalLoss

import torch
import torch.nn as nn
from torch import Tensor 

from mmdet3d.registry import MODELS


@MODELS.register_module()
class CustomGaussianFocalLoss(_GaussianFocalLoss):
    """GaussianFocalLoss is a variant of focal loss.

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

    def __init__(self,
                 alpha: float = 2.0,
                 gamma: float = 4.0,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 pos_weight: float = 1.0,
                 neg_weight: float = 1.0) -> None:
        super(CustomGaussianFocalLoss, self).__init__(
          alpha=alpha, 
          gamma=gamma, 
          reduction=reduction, 
          loss_weight=loss_weight, 
          pos_weight=pos_weight,
          neg_weight=neg_weight
        )

    # @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self,
                pred: Tensor,
                target: Tensor,
                pos_inds: Optional[Tensor] = None,
                pos_labels: Optional[Tensor] = None,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[Union[int, float]] = None,
                reduction_override: Optional[str] = None) -> Tensor:
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
        losses = super().forward(
          pred=pred,
          target=target, 
          pos_inds=pos_inds,
          pos_labels=pos_labels,
          weight=weight,
          avg_factor=avg_factor,
          reduction_override=reduction_override
        )
        
        return losses