from mmcv.transforms import BaseTransform
from mmdet3d.structures.ops import box_np_ops
from mmengine.registry import TRANSFORMS
import numpy as np 


@TRANSFORMS.register_module()
class ObjectMinPointsFilter(BaseTransform):
    """Filter objects by the number of points in them, if it's less than min_num_points.

    Args:
        min_num_points: (int): the number of points to filter objects
    """

    def __init__(self, min_num_points: int = 5) -> None:
        assert isinstance(min_num_points, int)
        self.min_num_points = min_num_points

    def transform(self, input_dict: dict) -> dict:
        """Call function to filter objects the number of points in them.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        points = input_dict["points"]
        gt_bboxes_3d = input_dict["gt_bboxes_3d"]

        # TODO(kminoda): There is a scary comment in the original code:
        # # TODO: this function is different from PointCloud3D, be careful
        # # when start to use nuscene, check the input
        indices = box_np_ops.points_in_rbbox(
            points.tensor.numpy()[:, :3],
            gt_bboxes_3d.tensor.numpy()[:, :7],
        )
        num_points_in_gt = indices.sum(0)
        gt_bboxes_mask = num_points_in_gt >= self.min_num_points
        input_dict["gt_bboxes_3d"] = input_dict["gt_bboxes_3d"][gt_bboxes_mask]
        input_dict["gt_labels_3d"] = input_dict["gt_labels_3d"][gt_bboxes_mask]

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(min_num_points={self.min_num_points})"
        return repr_str


@TRANSFORMS.register_module()
class ObjectRangeMinPointsFilter(BaseTransform):
    """Filter objects by bev ranges and the number of points in them, if it's less than min_num_points.

    Args:
        range_radius: (list[float]): the radius of the bev range in [min and max]. This will be transformed to
        [-max, -max, -min, -min] and [min, min, max, max] to select objects in the bev range.
        min_num_points: (int): the number of points to filter objects.
    """

    def __init__(self, range_radius: list[float], min_num_points: int = 5) -> None:
        assert isinstance(min_num_points, int)
        self.range_radius = range_radius
        assert (
            len(self.range_radius) == 2
        ), f"range_radius must be a list of length 2, but got {len(self.range_radius)}"
        assert (
            range_radius[0] < range_radius[1]
        ), f"range_radius[0] must be less than range_radius[1], but got {range_radius[0]} and {range_radius[1]}"

        self.min_num_points = min_num_points
        assert self.min_num_points > 0, f"min_num_points must be greater than 0, but got {self.min_num_points}"

    def transform(self, input_dict: dict) -> dict:
        """Call function to filter objects the number of points in them.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_bboxes_3d = input_dict["gt_bboxes_3d"]

        # ---- radius of each box center in BEV ----
        # (xmin, ymin, xmax, ymax)
        lower_bev_range = [-self.range_radius[1], -self.range_radius[1], -self.range_radius[0], -self.range_radius[0]]
        upper_bev_range = [self.range_radius[0], self.range_radius[0], self.range_radius[1], self.range_radius[1]]

        # ---- lower/upper bound for radius ----
        lower_mask = gt_bboxes_3d.in_range_bev(lower_bev_range)  # e.g. min_radius = 5m
        upper_mask = gt_bboxes_3d.in_range_bev(upper_bev_range)  # e.g. max_radius = 70m
        bev_radius_mask = (lower_mask | upper_mask).numpy().astype(bool)  # final mask

        # Out of range gt masks are all valid
        out_of_range_gt_masks = ~bev_radius_mask

        points = input_dict["points"]
        # NOTE: box_np_ops.points_in_rbbox may differ from PointCloud3D in terms of input format and output mask shape.
        # For nuScenes compatibility, ensure that:
        # - The input points are in the expected coordinate system (e.g., lidar vs. camera coordinates).
        # - The bounding box tensor shape matches the expected format (N, 7) for [x, y, z, dx, dy, dz, heading].
        # - The output mask correctly identifies points inside each bounding box for nuScenes data.
        # See https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/structures/ops/box_np_ops.py for details.
        indices = box_np_ops.points_in_rbbox(
            points.tensor.numpy()[:, :3],
            gt_bboxes_3d.tensor.numpy()[:, :7],
        )

        num_points_in_gt = indices.sum(0)
        gt_bboxes_mask = ((num_points_in_gt >= self.min_num_points) & (bev_radius_mask)) | out_of_range_gt_masks

        input_dict["gt_bboxes_3d"] = input_dict["gt_bboxes_3d"][gt_bboxes_mask]
        input_dict["gt_labels_3d"] = input_dict["gt_labels_3d"][gt_bboxes_mask]

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(range_radius={self.range_radius}, min_num_points={self.min_num_points})"
        return repr_str


@TRANSFORMS.register_module()
class IntensityLogNormalization(BaseTransform):
    """Filter objects by the number of points in them, if it's less than min_num_points.

    Args:
        min_num_points: (int): the number of points to filter objects
    """

    def __init__(self) -> None:
        pass

    def transform(self, input_dict: dict) -> dict:
        """Call function to filter objects the number of points in them.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        points = input_dict["points"]
        intensity = points.tensor[:, 3]
        intensity = np.log(1 + intensity)
        points.tensor[:, 3] = intensity
        # input_dict["points"] = points
        
        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str
