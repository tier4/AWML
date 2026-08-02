# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import torch
import torch.nn as nn
from mmcv.ops.diff_iou_rotated import oriented_box_intersection_2d
from mmdet3d.registry import MODELS
from mmdet.models.losses.utils import weight_reduce_loss, weighted_loss
from torch import Tensor
from torch.nn import functional as F


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
        out_size_factor,
        voxel_size,
        pc_range,
        cone_label_index: Optional[int],
        barrier_label_index: Optional[int],
        num_orientation_bins=4,
        orientation_bin_offset=torch.pi / 4,
        loss_weight=1.0,
        reduction="mean",
    ) -> None:

        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        self.cone_label_index = cone_label_index
        self.barrier_label_index = barrier_label_index
        self.num_orientation_bins = num_orientation_bins
        self.orientation_bin_offset = orientation_bin_offset
        self.bin_size = 2 * torch.pi / self.num_orientation_bins

    def _convert_to_bev_corners(self, bboxes: Tensor, labels: Tensor, is_gt: bool = False) -> Tensor:
        """
        bboxes (B, num_proposal, 10)
        """
        batch_size = bboxes.shape[0]
        center_x = bboxes[:, :, 0] * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
        center_y = bboxes[:, :, 1] * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
        lw = bboxes[:, :, 3:5].exp()
        rot_sin = bboxes[:, :, 6:7]
        rot_cos = bboxes[:, :, 7:8]
        if not is_gt:
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
        x4 = x4 * lw[:, :, 0].unsqueeze(-1)  # (B, N, 4)
        y4 = lw.new_tensor([0.5, 0.5, -0.5, -0.5]).to(lw.device)
        y4 = y4 * lw[:, :, 1].unsqueeze(-1)  # (B, N, 4)
        # (top right, top left, bottom left, bottom right)
        corners = torch.stack([x4, y4], dim=-1)  # (B, N, 4, 2)

        # (B * N, 4, 2) @ (B * N, 2, 2) -> (B * N, 4, 2)
        rotated = torch.bmm(corners.view([-1, 4, 2]), rotation_matrix_transpose.view([-1, 2, 2]))
        rotated = rotated.view([batch_size, -1, 4, 2])  # (B * N, 4, 2) -> (B, N, 4, 2)
        # Translation
        rotated[..., 0] += center_x.unsqueeze(-1)
        rotated[..., 1] += center_y.unsqueeze(-1)

        # Diagonal values, (B, N,)
        # lw_diagonal = torch.sqrt(lw[:, :, 0].square() + lw[:, :, 1].square() + 1e-6)
        return rotated, lw

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

        preds_corners, preds_length_width = self._convert_to_bev_corners(
            bboxes=preds_bboxes, labels=labels, is_gt=False
        )
        gts_corners, gts_length_width = self._convert_to_bev_corners(bboxes=gts_bboxes, labels=labels, is_gt=True)

        intersection, _ = oriented_box_intersection_2d(preds_corners, gts_corners)  # (B, N)
        area1 = preds_length_width[:, :, 0] * preds_length_width[:, :, 1]
        area2 = gts_length_width[:, :, 0] * gts_length_width[:, :, 1]
        union = area1 + area2 - intersection
        iou = intersection / (union + 1e-8)
        targets = torch.ones_like(iou)

        losses = iou_loss(
            iou,
            targets,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
        )
        return self.loss_weight * losses
