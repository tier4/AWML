from typing import Dict, List, Optional

import numpy as np
import torch
from mmdet3d.models.task_modules.coders.centerpoint_bbox_coders import CenterPointBBoxCoder as _CenterPointBBoxCoder
from mmdet3d.registry import TASK_UTILS
from torch import Tensor


@TASK_UTILS.register_module(force=True)
class MultiHeadCenterPointBBoxCoder(_CenterPointBBoxCoder):
    """Bbox coder for CenterPoint.

    Args:
        pc_range (list[float]): Range of point cloud.
        out_size_factor (int): Downsample factor of the model.
        voxel_size (list[float]): Size of voxel.
        post_center_range (list[float], optional): Limit of the center.
            Default: None.
        max_num (int, optional): Max number to be kept. Default: 100.
        score_threshold (float, optional): Threshold to filter boxes
            based on score. Default: None.
        code_size (int, optional): Code size of bboxes. Default: 9
        :param y_axis_reference: Set True if the rotation output is based on the clockwise y-axis.
    """

    def __init__(self, y_axis_reference: bool = False, **kwargs) -> None:
        self.y_axis_reference = y_axis_reference
        super(MultiHeadCenterPointBBoxCoder, self).__init__(**kwargs)

    def multihead_decode(
        self,
        heat: Tensor,
        rot_sine: Tensor,
        rot_cosine: Tensor,
        hei: Tensor,
        dim: Tensor,
        vel: Tensor,
        reg: Optional[Tensor] = None,
        task_id: int = -1,
    ) -> List[Dict[str, Tensor]]:
        """Decode bboxes for multi-head.

        Args:
            heat (torch.Tensor): Heatmap with the shape of [B, N, W, H].
            rot_sine (torch.Tensor): Sine of rotation with the shape of
                [B, 1, W, H].
            rot_cosine (torch.Tensor): Cosine of rotation with the shape of
                [B, 1, W, H].
            hei (torch.Tensor): Height of the boxes with the shape
                of [B, 1, W, H].
            dim (torch.Tensor): Dim of the boxes with the shape of
                [B, 1, W, H].
            vel (torch.Tensor): Velocity with the shape of [B, 1, W, H].
            reg (torch.Tensor, optional): Regression value of the boxes in
                2D with the shape of [B, 2, W, H]. Default: None.
            task_id (int, optional): Index of task. Default: -1.

        Returns:
            list[dict]: Decoded boxes.
        """
        batch, cat, _, _ = heat.size()

        scores, inds, clses, ys, xs = self._topk(heat, K=self.max_num)

        if reg is not None:
            reg = self._transpose_and_gather_feat(reg, inds)
            reg = reg.view(batch, self.max_num, 2)
            xs = xs.view(batch, self.max_num, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, self.max_num, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, self.max_num, 1) + 0.5
            ys = ys.view(batch, self.max_num, 1) + 0.5

        # rotation value and direction label
        rot_sine = self._transpose_and_gather_feat(rot_sine, inds)
        rot_sine = rot_sine.view(batch, self.max_num, 1)

        rot_cosine = self._transpose_and_gather_feat(rot_cosine, inds)
        rot_cosine = rot_cosine.view(batch, self.max_num, 1)
        rot = torch.atan2(rot_sine, rot_cosine)

        # height in the bev
        hei = self._transpose_and_gather_feat(hei, inds)
        hei = hei.view(batch, self.max_num, 1)

        # dim of the box
        dim = self._transpose_and_gather_feat(dim, inds)
        dim = dim.view(batch, self.max_num, 3)

        # class label
        clses = clses.view(batch, self.max_num).float()
        scores = scores.view(batch, self.max_num)

        xs = xs.view(batch, self.max_num, 1) * self.out_size_factor[task_id] * self.voxel_size[0] + self.pc_range[0]
        ys = ys.view(batch, self.max_num, 1) * self.out_size_factor[task_id] * self.voxel_size[1] + self.pc_range[1]

        if vel is None:  # KITTI FORMAT
            final_box_preds = torch.cat([xs, ys, hei, dim, rot], dim=2)
        else:  # exist velocity, nuscene format
            vel = self._transpose_and_gather_feat(vel, inds)
            vel = vel.view(batch, self.max_num, 2)
            final_box_preds = torch.cat([xs, ys, hei, dim, rot, vel], dim=2)

        final_scores = scores
        final_preds = clses

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(self.post_center_range, device=heat.device)
            mask = (final_box_preds[..., :3] >= self.post_center_range[:3]).all(2)
            mask &= (final_box_preds[..., :3] <= self.post_center_range[3:]).all(2)

            predictions_dicts = []
            for i in range(batch):
                cmask = mask[i, :]
                if self.score_threshold:
                    cmask &= thresh_mask[i]

                boxes3d = final_box_preds[i, cmask]
                scores = final_scores[i, cmask]
                labels = final_preds[i, cmask]
                predictions_dict = {"bboxes": boxes3d, "scores": scores, "labels": labels}

                predictions_dicts.append(predictions_dict)
        else:
            raise NotImplementedError(
                "Need to reorganize output as a batch, only " "support post_center_range is not None for now!"
            )

        return predictions_dicts

    def decode(
        self,
        heat: Tensor,
        rot_sine: Tensor,
        rot_cosine: Tensor,
        hei: Tensor,
        dim: Tensor,
        vel: Tensor,
        reg: Optional[Tensor] = None,
        task_id: int = -1,
    ) -> List[Dict[str, Tensor]]:
        """Decode bboxes.

        Args:
            heat (torch.Tensor): Heatmap with the shape of [B, N, W, H].
            rot_sine (torch.Tensor): Sine of rotation with the shape of
                [B, 1, W, H].
            rot_cosine (torch.Tensor): Cosine of rotation with the shape of
                [B, 1, W, H].
            hei (torch.Tensor): Height of the boxes with the shape
                of [B, 1, W, H].
            dim (torch.Tensor): Dim of the boxes with the shape of
                [B, 1, W, H].
            vel (torch.Tensor): Velocity with the shape of [B, 1, W, H].
            reg (torch.Tensor, optional): Regression value of the boxes in
                2D with the shape of [B, 2, W, H]. Default: None.
            task_id (int, optional): Index of task. Default: -1.

        Returns:
            list[dict]: Decoded boxes.
        """
        if not isinstance(self.out_size_factor):
            predictions_dicts = super().decode(
                heat=heat,
                rot_sine=rot_sine,
                rot_cosine=rot_cosine,
                hei=hei,
                dim=dim,
                vel=vel,
                reg=reg,
                task_id=task_id,
            )
        else:
            predictions_dicts = self.multihead_decode(
                heat=heat,
                rot_sine=rot_sine,
                rot_cosine=rot_cosine,
                hei=hei,
                dim=dim,
                vel=vel,
                reg=reg,
                task_id=task_id,
            )

        if not self.y_axis_reference:
            return predictions_dicts

        for predictions_dict in predictions_dicts:
            if self.y_axis_reference:
                # Switch width and length
                predictions_dict["bboxes"][:, [3, 4]] = predictions_dict["bboxes"][:, [4, 3]]

                # Change the rotation to clockwise y-axis
                predictions_dict["bboxes"][:, 6] = -predictions_dict["bboxes"][:, 6] - np.pi / 2

        return predictions_dicts
