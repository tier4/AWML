# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import torch.nn as nn
from mmdet3d.registry import MODELS
from mmdet.models.losses.utils import weight_reduce_loss, weighted_loss
from torch import Tensor
from torch.nn import functional as F
import torch 


@weighted_loss
def bce_with_logits_loss(pred: Tensor, target: Tensor, pos_weight: Optional[Tensor] = None) -> Tensor:
    """ """

    losses = F.binary_cross_entropy_with_logits(
        pred,
        target,
        None,  # Always None since the weight will be used in the weighted_loss wrapper
        pos_weight=pos_weight,
        reduction="none",  # Always none since the reduction will happen in the weighted_loss wrapper
    )
    return losses


@weighted_loss
def corner_l1_losses(pred: Tensor, target: Tensor, norm_values: Optional[Tensor] = None, barrier_masks=None) -> Tensor:
    """ """

    # Shape: [B, N, 4, 2] -> [B, N, 4] -> [B, N]
    direct_losses = (pred - target).abs().sum(dim=-1).mean(dim=-1)

    losses = direct_losses
    # Branch out barrier if set
    if barrier_masks is not None:

        # [c0, c1, c2, c3] -> [c2, c3, c0, c1]
        target_shifted_180 = torch.roll(
            target,
            shifts=2,
            dims=-2,
        )

        shifted_losses = (pred - target_shifted_180).abs().sum(dim=-1).mean(dim=-1)  # [B, N]

        barrier_losses = torch.minimum(
            direct_losses,
            shifted_losses,
        )

        # Use symmetry-aware loss only for barriers.
        losses = torch.where(
            barrier_masks,
            barrier_losses,
            direct_losses,
        )

    if norm_values is not None:
        losses = losses / norm_values
    return losses


@MODELS.register_module()
class CustomBCEWithLogitsLoss(nn.Module):
    """BCEWithLogitsLoss"""

    def __init__(self, loss_weight=None, reduction="mean", pos_weight=None) -> None:
        super().__init__()
        self.loss_weight = loss_weight
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

        losses = self.loss_weight * bce_with_logits_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor, pos_weight=self.pos_weight
        )
        return losses


@MODELS.register_module()
class BEVCornerLoss(nn.Module):
    """Compute 4 corner losses between predictions and gt boxes."""

    def __init__(
        self,
        out_size_factor,
        voxel_size,
        pc_range,
        gt_diagonal_norm: bool,
        cone_label_index: Optional[int],
        barrier_label_index: Optional[int],
        loss_weight=1.0,
        reduction="mean",
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        self.gt_diagonal_norm = gt_diagonal_norm
        self.cone_label_index = cone_label_index
        self.barrier_label_index = barrier_label_index

    def _convert_to_bev_corners(self, bboxes: Tensor, labels: Tensor) -> Tensor:
        """
        bboxes (B, num_proposal, 10)
        """
        batch_size = bboxes.shape[0]
        center_x = bboxes[:, :, 0] * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
        center_y = bboxes[:, :, 1] * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
        lw = bboxes[:, :, 3:5].exp()
        rot_sin = bboxes[:, :, 6:7]
        rot_cos = bboxes[:, :, 7:8]

        norm_rotation = torch.sqrt(rot_sin.square() + rot_cos.square() + 1e-6)
        rot_sin = rot_sin / norm_rotation
        rot_cos = rot_cos / norm_rotation

        row1 = torch.cat([rot_cos, rot_sin], dim=-1)
        row2 = torch.cat([-rot_sin, rot_cos], dim=-1)  # (B, N, 2)
        rotation_matrix_transpose = torch.stack([row1, row2], dim=-2)  # (B, N, 2, 2)

        if self.cone_label_index is not None:
            cone_mask = (labels == self.cone_label_index)[..., None, None]  # [B, N, 1, 1]

            identity = torch.eye(
                2,
                device=rotation_matrix_transpose.device,
                dtype=rotation_matrix_transpose.dtype,
            ).view(1, 1, 2, 2)

            rotation_matrix_transpose = torch.where(
                cone_mask,
                identity,
                rotation_matrix_transpose,
            )

        x4 = lw.new_tensor([0.5, -0.5, -0.5, 0.5]).to(lw.device)
        x4 = x4 * lw[:, :, 0]  # (B, N, 4)
        y4 = lw.new_tensor([0.5, 0.5, -0.5, -0.5]).to(lw.device)
        y4 = y4 * lw[:, :, 1]  # (B, N, 4)
        # (top right, top left, bottom left, bottom right)
        corners = torch.stack([x4, y4], dim=-1)  # (B, N, 4, 2)

        # (B * N, 4, 2) @ (B * N, 2, 2) -> (B * N, 4, 2)
        rotated = torch.bmm(corners.view([-1, 4, 2]), rotation_matrix_transpose.view([-1, 2, 2]))
        rotated = rotated.view([batch_size, -1, 4, 2])  # (B * N, 4, 2) -> (B, N, 4, 2)
        # Translation
        rotated[..., 0] += center_x
        rotated[..., 1] += center_y

        # Diagonal values, (B, N,)
        lw_diagonal = torch.sqrt(lw[:, :, 0].square() + lw[:, :, 1].square() + 1e-6)
        return rotated, lw_diagonal

    def forward(
        self,
        preds_bboxes: Tensor,
        gts_bboxes: Tensor,
        labels: Tensor,
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

        preds_corners, _ = self._convert_to_bev_corners(bboxes=preds_bboxes, labels=labels)
        gts_corners, gt_diagonal_values = self._convert_to_bev_corners(bboxes=gts_bboxes, labels=labels)
        if self.gt_diagonal_norm:
            norm_values = gt_diagonal_values
        else:
            norm_values = None

        if self.barrier_label_index is not None:
            barrier_masks = labels == self.barrier_label_index
        else:
            barrier_masks = None

        losses = corner_l1_losses(
            preds_corners,
            gts_corners,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            norm_values=norm_values,
            barrier_masks=barrier_masks,
        )  # (B, N)

        return self.loss_weight * losses
