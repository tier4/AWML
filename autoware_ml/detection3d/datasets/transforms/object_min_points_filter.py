from mmcv.transforms import BaseTransform
from mmdet3d.structures.ops import box_np_ops
from mmengine.registry import TRANSFORMS


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
