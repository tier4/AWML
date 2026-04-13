from typing import Optional, Union

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS
from scipy.spatial.transform import Rotation as R


@TRANSFORMS.register_module()
class PointOffset(BaseTransform):
    """Apply translation and rotation offsets to points.

    Args:
        offset (list[float], optional): Translation values [dx, dy, dz].
            Defaults to [0.0, 0.0, 0.0].
        rotation (list[float], optional): Rotation values [roll, pitch, yaw] in degrees.
            Defaults to [0.0, 0.0, 0.0].
        sensor_id (int or str, optional): If provided, only points with this ID will be transformed.
            For lidarseg, it can be the index in the sources list, the sensor name, or the sensor token.
            Defaults to None.
        sensor_dim (int, optional): The index of the dimension containing the sensor ID.
            Used if range-based sensor info is not available. Defaults to 4.
    """

    def __init__(
        self,
        offset: list[float] = [0.0, 0.0, 0.0],
        rotation: list[float] = [0.0, 0.0, 0.0],
        sensor_id: Optional[Union[int, str]] = None,
        sensor_dim: int = 4,
    ) -> None:
        assert len(offset) == 3
        assert len(rotation) == 3
        self.offset = np.array(offset, dtype=np.float32)
        # scipy uses [roll, pitch, yaw] for 'xyz' extrinsic rotation
        self.rotation_angles = rotation
        self.rot_mat = R.from_euler("xyz", rotation, degrees=True).as_matrix().astype(np.float32)
        self.sensor_id = sensor_id
        self.sensor_dim = sensor_dim

    def transform(self, input_dict: dict) -> dict:
        """Apply transformation to points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results with transformed points.
        """
        points = input_dict["points"]
        points_tensor = points.tensor

        rot_mat_torch = points_tensor.new_tensor(self.rot_mat)
        offset_torch = points_tensor.new_tensor(self.offset)

        if self.sensor_id is not None:
            # Check for lidarseg-specific range information
            if "lidar_sources_info" in input_dict and "sources" in input_dict["lidar_sources_info"]:
                sources = input_dict["lidar_sources_info"]["sources"]
                idx_begin = None
                length = None

                if isinstance(self.sensor_id, int):
                    if 0 <= self.sensor_id < len(sources):
                        idx_begin = sources[self.sensor_id].get("idx_begin")
                        length = sources[self.sensor_id].get("length")
                elif isinstance(self.sensor_id, str):
                    # Try to match by sensor name or token
                    target_token = self.sensor_id
                    if "lidar_sources" in input_dict and self.sensor_id in input_dict["lidar_sources"]:
                        target_token = input_dict["lidar_sources"][self.sensor_id].get("sensor_token")

                    for s in sources:
                        if s.get("sensor_token") == target_token:
                            idx_begin = s.get("idx_begin")
                            length = s.get("length")
                            break

                if idx_begin is not None and length is not None:
                    # Apply transformation only to the specified range of points
                    points_tensor[idx_begin : idx_begin + length, :3] = (
                        points_tensor[idx_begin : idx_begin + length, :3] @ rot_mat_torch.T + offset_torch
                    )
                else:
                    # Fallback to dimension-based masking if range not found (and sensor_id is comparable)
                    if not isinstance(self.sensor_id, str):
                        mask = points_tensor[:, self.sensor_dim] == self.sensor_id
                        if mask.any():
                            points_tensor[mask, :3] = points_tensor[mask, :3] @ rot_mat_torch.T + offset_torch
            else:
                # Traditional dimension-based masking
                mask = points_tensor[:, self.sensor_dim] == self.sensor_id
                if mask.any():
                    points_tensor[mask, :3] = points_tensor[mask, :3] @ rot_mat_torch.T + offset_torch
        else:
            # Apply to all points
            points_tensor[:, :3] = points_tensor[:, :3] @ rot_mat_torch.T + offset_torch

        input_dict["points"] = points
        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(offset={self.offset.tolist()}, "
        repr_str += f"rotation={self.rotation_angles}, "
        repr_str += f"sensor_id={self.sensor_id}, "
        repr_str += f"sensor_dim={self.sensor_dim})"
        return repr_str
