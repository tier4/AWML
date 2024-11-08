from time import time

import numpy.typing as npt
import numpy as np
import torch
import torch.nn.functional as F

from utils import load_preprocessing_cfg


class Preprocessing:

    def __init__(self, model_path: str):
        preprocessing_config = load_preprocessing_cfg(model_path)
        self.interpolation_cfg = preprocessing_config['range_interpolation']
        self.region_group_cfg = preprocessing_config['frustum_region_group']

    def _range_interpolation(self, points: npt.ArrayLike, fov_up: float,
                             fov_down: float, H: int, W: int,
                             **kwargs) -> npt.ArrayLike:
        fov_up = fov_up / 180 * np.pi
        fov_down = fov_down / 180 * np.pi
        fov = abs(fov_down) + abs(fov_up)
        proj_image = np.full((H, W, 4), -1, dtype=np.float32)
        proj_idx = np.full((H, W), -1, dtype=np.int64)

        # get depth of all points
        depth = np.linalg.norm(points[:, :3], 2, axis=1)

        # get angles of all points
        yaw = -np.arctan2(points[:, 1], points[:, 0])
        pitch = np.arcsin(points[:, 2] / depth)

        # get projection in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov

        # scale to image size using angular resolution
        proj_x *= W
        proj_y *= H

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int64)

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int64)

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        proj_idx[proj_y[order], proj_x[order]] = indices[order]
        proj_image[proj_y[order], proj_x[order]] = points[order]
        proj_mask = (proj_idx > 0).astype(np.int32)

        interpolated_points = []

        # scan all the pixels
        for y in range(H):
            for x in range(W):
                # check whether the current pixel is valid
                # if valid, just skip this pixel
                if proj_mask[y, x]:
                    continue

                if (x - 1 >= 0) and (x + 1 < W):
                    # only when both of right and left pixels are valid,
                    # the interpolated points will be calculated
                    if proj_mask[y, x - 1] and proj_mask[y, x + 1]:
                        # calculated the potential points
                        mean_points = (proj_image[y, x - 1] +
                                       proj_image[y, x + 1]) / 2
                        # change the current pixel to be valid
                        proj_mask[y, x] = 1
                        proj_image[y, x] = mean_points
                        interpolated_points.append(mean_points)

        # concatenate all the interpolated points and labels
        if len(interpolated_points) > 0:
            interpolated_points = np.array(
                interpolated_points, dtype=np.float32)
            points = np.concatenate((points, interpolated_points), axis=0)

        return points

    def _frustum_region_group(self, points: torch.Tensor, fov_up: float,
                              fov_down: float, H: int, W: int,
                              **kwargs) -> dict:
        fov_up = fov_up / 180 * np.pi
        fov_down = fov_down / 180 * np.pi
        fov = abs(fov_down) + abs(fov_up)

        assert type(points) == torch.Tensor

        depth = torch.linalg.norm(points[:, :3], 2, dim=1)
        yaw = -torch.atan2(points[:, 1], points[:, 0])
        pitch = torch.arcsin(points[:, 2] / depth)

        coors_x = 0.5 * (yaw / np.pi + 1.0)
        coors_y = 1.0 - (pitch + abs(fov_down)) / fov

        # scale to image size using angular resolution
        coors_x *= W
        coors_y *= H

        # round and clamp for use as index
        coors_x = torch.floor(coors_x)
        coors_x = torch.clamp(coors_x, min=0, max=W - 1).type(torch.int64)

        coors_y = torch.floor(coors_y)
        coors_y = torch.clamp(coors_y, min=0, max=H - 1).type(torch.int64)

        coors = torch.stack([coors_y, coors_x], dim=1)
        coors = F.pad(coors, (1, 0), mode='constant', value=0)

        voxel_coors, inverse_map = torch.unique(
            coors, return_inverse=True, dim=0)

        batch_inputs_dict = {}
        batch_inputs_dict['points'] = points
        batch_inputs_dict['coors'] = coors
        batch_inputs_dict['voxel_coors'] = voxel_coors
        batch_inputs_dict['inverse_map'] = inverse_map
        return batch_inputs_dict

    def preprocess(self, points: npt.ArrayLike) -> dict:
        num_points = points.shape[0]
        t_start = time()
        interpolated_points = self._range_interpolation(
            points, **self.interpolation_cfg)
        batch_inputs_dict = self._frustum_region_group(
            torch.from_numpy(interpolated_points), **self.region_group_cfg)
        t_end = time()
        latency = np.round((t_end - t_start) * 1e3, 2)
        print(f'Preprocessing latency: {latency} ms')
        batch_inputs_dict['num_points'] = num_points
        return batch_inputs_dict
