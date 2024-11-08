from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmengine.model import BaseDataPreprocessor
from torch import Tensor


@MODELS.register_module()
class FrustumRangePreprocessor(BaseDataPreprocessor):
    """Frustum-Range Segmentor pre-processor for frustum region group.

    Args:
        H (int): Height of the 2D representation.
        W (int): Width of the 2D representation.
        fov_up (float): Front-of-View at upward direction of the sensor.
        fov_down (float): Front-of-View at downward direction of the sensor.
        ignore_index (int): The label index to be ignored.
        non_blocking (bool): Whether to block current process when transferring
            data to device. Defaults to False.
    """

    def __init__(self,
                 H: int,
                 W: int,
                 fov_up: float,
                 fov_down: float,
                 ignore_index: int,
                 non_blocking: bool = False) -> None:
        super(FrustumRangePreprocessor,
              self).__init__(non_blocking=non_blocking)
        self.H = H
        self.W = W
        self.fov_up = fov_up / 180 * np.pi
        self.fov_down = fov_down / 180 * np.pi
        self.fov = abs(self.fov_down) + abs(self.fov_up)
        self.ignore_index = ignore_index

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform frustum region group based on ``BaseDataPreprocessor``.

        Args:
            data (dict): Data from dataloader. The dict contains the whole
                batch data.
            training (bool): Whether to enable training time augmentation.
                Defaults to False.

        Returns:
            dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)
        data.setdefault('data_samples', None)

        inputs, data_samples = data['inputs'], data['data_samples']
        batch_inputs = dict()

        assert 'points' in inputs
        batch_inputs = self.frustum_region_group(inputs['points'], data_samples)

        return {'inputs': batch_inputs, 'data_samples': data_samples}

    @torch.no_grad()
    def frustum_region_group(self, points_batch: List[Tensor],
                             data_samples: SampleList) -> dict:
        """Calculate frustum region of each point.

        Args:
            points (List[Tensor]): Point cloud in one data batch.

        Returns:
            dict: Frustum region information.
        """
        voxel_dict = dict()

        coors = []
        points = []

        for i, res in enumerate(points_batch):
            depth = torch.linalg.norm(res[:, :3], 2, dim=1)
            yaw = -torch.atan2(res[:, 1], res[:, 0])
            pitch = torch.arcsin(res[:, 2] / depth)

            coors_x = 0.5 * (yaw / np.pi + 1.0)
            coors_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov

            # scale to image size using angular resolution
            coors_x *= self.W
            coors_y *= self.H

            # round and clamp for use as index
            coors_x = torch.floor(coors_x)
            coors_x = torch.clamp(
                coors_x, min=0, max=self.W - 1).type(torch.int64)

            coors_y = torch.floor(coors_y)
            coors_y = torch.clamp(
                coors_y, min=0, max=self.H - 1).type(torch.int64)

            res_coors = torch.stack([coors_y, coors_x], dim=1)
            res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
            coors.append(res_coors)
            points.append(res)

            if 'pts_semantic_mask' in data_samples[i].gt_pts_seg:
                pts_semantic_mask = data_samples[
                    i].gt_pts_seg.pts_semantic_mask
                seg_label = torch.ones(
                    (self.H, self.W),
                    dtype=torch.long,
                    device=pts_semantic_mask.device) * self.ignore_index

                # Sort
                indices_2 = torch.argsort(res_coors[:, 2])
                sorted_coors = res_coors[indices_2]
                indices_1 = torch.argsort(sorted_coors[:, 1])
                sorted_coors = sorted_coors[indices_1]
                indices = indices_2[indices_1]
                # Calculate differences between rows
                min_value_row = torch.full((1, sorted_coors.size(1)), torch.iinfo(sorted_coors.dtype).min, dtype=sorted_coors.dtype, device=sorted_coors.device)
                sorted_coors_ex = torch.cat([min_value_row, sorted_coors], dim=0)
                diffs = sorted_coors_ex[1:] - sorted_coors_ex[:-1]
                # Create a mask for rows that are unique
                unique_mask = (diffs != 0).any(dim=1)
                # Extract unique rows
                res_voxel_coors = sorted_coors[unique_mask]
                # Calculate inverse indices
                inverse_map = torch.cumsum(unique_mask.to(torch.int64), dim=0) - 1
                inverse_map = inverse_map[torch.argsort(indices)]

                features = F.one_hot(pts_semantic_mask).float()
                voxel_sum = torch.zeros(inverse_map.max() + 1, features.size(1), device=features.device, dtype=features.dtype)
                voxel_count = voxel_sum.clone()
                ones_tensor = torch.ones_like(features)
                index = inverse_map.unsqueeze(-1).expand_as(features)

                voxel_sum.scatter_add_(dim=0, index=index, src=features)
                voxel_count.scatter_add_(dim=0, index=index, src=ones_tensor)
                voxel_semantic_mask = voxel_sum / voxel_count

                voxel_semantic_mask = torch.argmax(voxel_semantic_mask, dim=-1)
                seg_label[res_voxel_coors[:, 1],
                          res_voxel_coors[:, 2]] = voxel_semantic_mask
                data_samples[i].gt_pts_seg.semantic_seg = seg_label

        points = torch.cat(points, dim=0)
        coors = torch.cat(coors, dim=0)
        voxel_coors, inverse_map = torch.unique(coors, return_inverse=True, dim=0)
        voxel_dict['points'] = points
        voxel_dict['coors'] = coors
        voxel_dict['voxel_coors'] = voxel_coors
        voxel_dict['inverse_map'] = inverse_map
        return voxel_dict
