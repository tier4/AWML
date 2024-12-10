from typing import Dict, List, Optional

import numpy as np
from mmdet3d.models.task_modules.coders.centerpoint_bbox_coders import CenterPointBBoxCoder as _CenterPointBBoxCoder
from mmdet3d.registry import TASK_UTILS
from torch import Tensor


@TASK_UTILS.register_module(force=True)
class CenterPointBBoxCoder(_CenterPointBBoxCoder):
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
        super(CenterPointBBoxCoder, self).__init__(**kwargs)

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

        if not self.y_axis_reference:
            return predictions_dicts

        for predictions_dict in predictions_dicts:
            if self.y_axis_reference:
                # Switch width and length
                predictions_dict["bboxes"][:, [3, 4]] = predictions_dict["bboxes"][:, [4, 3]]

                # Change the rotation to clockwise y-axis
                predictions_dict["bboxes"][:, 6] = -predictions_dict["bboxes"][:, 6] - np.pi / 2

        return predictions_dicts
