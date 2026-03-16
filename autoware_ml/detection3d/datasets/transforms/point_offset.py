import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PointOffset(BaseTransform):
    """Offset points on a given axis by a specified amount.

    Args:
        offset (list[float]): Offset values [dx, dy, dz].
        sensor_id (int, optional): If provided, only points with this ID will be offset.
            Defaults to None.
        sensor_dim (int, optional): The index of the dimension containing the sensor ID.
            Defaults to 4.
    """

    def __init__(
        self, offset: list[float], sensor_id: int = None, sensor_dim: int = 4
    ) -> None:
        assert len(offset) == 3
        self.offset = np.array(offset, dtype=np.float32)
        self.sensor_id = sensor_id
        self.sensor_dim = sensor_dim

    def transform(self, input_dict: dict) -> dict:
        """Apply offset to points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results with offset points.
        """
        points = input_dict["points"]

        if self.sensor_id is not None:
            # points.tensor is a torch.Tensor
            points_np = points.tensor.numpy()
            mask = points_np[:, self.sensor_dim] == self.sensor_id
            points_np[mask, :3] += self.offset
            # Update the tensor in place if possible, or re-assign
            points.tensor = points.tensor.new_tensor(points_np)
        else:
            points.tensor[:, :3] += points.tensor.new_tensor(self.offset)

        input_dict["points"] = points
        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(offset={self.offset.tolist()}, "
        repr_str += f"sensor_id={self.sensor_id}, "
        repr_str += f"sensor_dim={self.sensor_dim})"
        return repr_str
