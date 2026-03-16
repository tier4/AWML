import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.registry import TRANSFORMS
from scipy.spatial.transform import Rotation as R


@TRANSFORMS.register_module()
class PointOffset(BaseTransform):
    """Apply translation and rotation offsets to points.

    Args:
        offset (list[float], optional): Translation values [dx, dy, dz].
            Defaults to [0.0, 0.0, 0.0].
        rotation (list[float], optional): Rotation values [roll, pitch, yaw] in degrees.
            Defaults to [0.0, 0.0, 0.0].
        sensor_id (int, optional): If provided, only points with this ID will be transformed.
            Defaults to None.
        sensor_dim (int, optional): The index of the dimension containing the sensor ID.
            Defaults to 4.
    """

    def __init__(
        self,
        offset: list[float] = [0.0, 0.0, 0.0],
        rotation: list[float] = [0.0, 0.0, 0.0],
        sensor_id: int = None,
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

        if self.sensor_id is not None:
            points_np = points_tensor.numpy()
            mask = points_np[:, self.sensor_dim] == self.sensor_id
            
            if np.any(mask):
                # Apply rotation: v' = R * v
                points_to_transform = points_np[mask, :3]
                points_np[mask, :3] = points_to_transform @ self.rot_mat.T + self.offset
                
                points.tensor = points_tensor.new_tensor(points_np)
        else:
            # Apply to all points
            rot_mat_torch = points_tensor.new_tensor(self.rot_mat)
            offset_torch = points_tensor.new_tensor(self.offset)
            
            # points_tensor[:, :3] is (N, 3). Transformation: points @ R.T + offset
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
