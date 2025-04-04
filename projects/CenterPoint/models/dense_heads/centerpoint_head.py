from typing import List, Tuple

import numpy as np
import torch
from mmdet3d.models import CenterHead as _CenterHead
from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.registry import MODELS
from mmdet3d.structures.bbox_3d.utils import limit_period
from mmengine import print_log
from mmengine.structures import InstanceData


def get_rotation_bin(rot, dir_offset=0, num_bins=2, one_hot=True):
    """Encode rotation to 0 ~ num_bins-1.

    Args:
        rot (torch.Tensor): The rotation to encode.
        dir_offset (int): Direction offset.
        num_bins (int): Number of bins to divide 2*PI.
        one_hot (bool): Whether to encode as one hot.

    Returns:
        torch.Tensor: Encoded rotation bin.
    """
    offset_rot = limit_period(rot - dir_offset, 0, np.pi)
    bin_cls_targets = torch.floor(offset_rot / (np.pi / num_bins)).long()
    bin_cls_targets = torch.clamp(bin_cls_targets, min=0, max=num_bins - 1)
    if one_hot:
        bin_targets = torch.zeros(
            *list(bin_cls_targets.shape), num_bins, dtype=rot.dtype, device=bin_cls_targets.device
        )
        scatter_dim = len(bin_cls_targets.shape)
        bin_targets.scatter_(scatter_dim, bin_cls_targets.unsqueeze(dim=-1).long(), 1.0)
        bin_cls_targets = bin_targets
    return bin_cls_targets


def get_direction_bin(rot, dir_offset=0, one_hot=True):
    """Encode direction to 2 bins.

    Args:
        rot (torch.Tensor): The rotation to encode.
        dir_offset (int): Direction offset..
        one_hot (bool): Whether to encode as one hot.

    Returns:
        torch.Tensor: Encoded direction targets.
    """
    offset_rot = limit_period(rot - dir_offset, 0, 2 * np.pi)
    dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / 2)).long()
    dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=2 - 1)
    if one_hot:
        dir_targets = torch.zeros(*list(dir_cls_targets.shape), 2, dtype=rot.dtype, device=dir_cls_targets.device)
        scatter_dim = len(dir_cls_targets.shape)
        dir_targets.scatter_(scatter_dim, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
        dir_cls_targets = dir_targets
    return dir_cls_targets


@MODELS.register_module(force=True)
class CenterHead(_CenterHead):
    """overwritten class of CenterHead
    Note:
        We add class-wise loss implementation.
    TODO(KokSeang):
        We still using `loss_bbox` in this implementation, loss_reg, loss_height, loss_dim,
        loss_rot and loss_vel will be implemented in the next version.
        For this reason, we need to set `code_weights` for each loss in the config `model.train_cfg`.
    """

    def __init__(
        self,
        freeze_shared_conv: bool = False,
        freeze_task_heads: bool = False,
        **kwargs,
    ):
        super(CenterHead, self).__init__(**kwargs)
        loss_cls = kwargs["loss_cls"]
        self._class_wise_loss = loss_cls.get("reduction") == "none"
        if not self._class_wise_loss:
            print_log("If you want to see a class-wise heatmap loss, use reduction='none' of 'loss_cls'.")

        self.freeze_shared_conv = freeze_shared_conv
        self.freeze_task_heads = freeze_task_heads
        self._freeze_parameters()

    def _freeze_parameters(self) -> None:
        """Freeze parameters in the head."""
        if self.freeze_shared_conv:
            print_log("Freeze shared conv")
            for params in self.shared_conv.parameters():
                params.requires_grad = False

        if self.freeze_task_heads:
            print_log("Freeze task heads")
            for task in self.task_heads:
                for params in task.parameters():
                    params.requires_grad = False

    def loss_by_feat(self, preds_dicts: Tuple[List[dict]], batch_gt_instances_3d: List[InstanceData], *args, **kwargs):
        """Loss function for CenterHead.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results of
                multiple tasks. The outer tuple indicate  different
                tasks head, and the internal list indicate different
                FPN level.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes_3d`` and\
                ``labels_3d`` attributes.

        Returns:
            dict[str,torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        heatmaps, anno_boxes, inds, masks = self.get_targets(batch_gt_instances_3d)
        loss_dict = dict()

        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict[0]["heatmap"] = clip_sigmoid(preds_dict[0]["heatmap"])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            class_names: List[str] = self.class_names[task_id]

            if self._class_wise_loss:
                loss_heatmap_cls: torch.Tensor = self.loss_cls(
                    preds_dict[0]["heatmap"],
                    heatmaps[task_id],
                )
                loss_heatmap_cls = loss_heatmap_cls.sum((0, 2, 3)) / max(num_pos, 1)
                for cls_i, class_name in enumerate(class_names):
                    loss_dict[f"task{task_id}.loss_heatmap_{class_name}"] = loss_heatmap_cls[cls_i]
            else:
                loss_heatmap = self.loss_cls(preds_dict[0]["heatmap"], heatmaps[task_id], avg_factor=max(num_pos, 1))
                loss_dict[f"task{task_id}.loss_heatmap"] = loss_heatmap

            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            preds_dict[0]["anno_box"] = torch.cat(
                (
                    preds_dict[0]["reg"],
                    preds_dict[0]["height"],
                    preds_dict[0]["dim"],
                    preds_dict[0]["rot"],
                    preds_dict[0]["vel"],
                ),
                dim=1,
            )

            # Regression loss for dimension, offset, height, rotation
            ind = inds[task_id]
            num = masks[task_id].float().sum()
            pred = preds_dict[0]["anno_box"].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = self._gather_feat(pred, ind)
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan

            code_weights = self.train_cfg.get("code_weights", None)
            bbox_weights = mask * mask.new_tensor(code_weights)
            loss_bbox = self.loss_bbox(pred, target_box, bbox_weights, avg_factor=(num + 1e-4))
            loss_dict[f"task{task_id}.loss_bbox"] = loss_bbox
        return loss_dict
