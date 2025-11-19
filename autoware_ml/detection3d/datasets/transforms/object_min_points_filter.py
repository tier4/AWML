import torch 

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
        # self.remove_points = remove_points

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
    """Filter objects by the number of points in them, if it's less than min_num_points.

    Args:
        min_num_points: (int): the number of points to filter objects
    """

    def __init__(self, range_radius: list[float], min_num_points: int = 5) -> None:
        assert isinstance(min_num_points, int)
        self.range_radius = range_radius
        self.min_num_points = min_num_points
        # self.remove_points = remove_points

    def transform(self, input_dict: dict) -> dict:
        """Call function to filter objects the number of points in them.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']    

        # ---- radius of each box center in BEV ----
        # (xmin, ymin, xmax, ymax)
        lower_bev_range = [-self.range_radius[1], -self.range_radius[1], -self.range_radius[0], -self.range_radius[0]]
        upper_bev_range = [self.range_radius[0], self.range_radius[0], self.range_radius[1], self.range_radius[1]]
        
        # ---- lower/upper bound for radius ----
        lower_mask = gt_bboxes_3d.in_range_bev(lower_bev_range)            # e.g. min_radius = 5m
        upper_mask = gt_bboxes_3d.in_range_bev(upper_bev_range)            # e.g. max_radius = 70m
        bev_radius_mask = (lower_mask | upper_mask).numpy().astype(bool)         # final mask
        
        # Out of range gt masks are all valid
        out_of_range_gt_masks = ~bev_radius_mask


        points = input_dict["points"]
        # TODO(kminoda): There is a scary comment in the original code:
        # # TODO: this function is different from PointCloud3D, be careful
        # # when start to use nuscene, check the input
        indices = box_np_ops.points_in_rbbox(
            points.tensor.numpy()[:, :3],
            gt_bboxes_3d.tensor.numpy()[:, :7],
        )
        
        num_points_in_gt = indices.sum(0)
        gt_bboxes_mask = ((num_points_in_gt >= self.min_num_points) & (bev_radius_mask)) | out_of_range_gt_masks
        # # mask is a torch tensor but gt_labels_3d is still numpy array
        # # using mask to index gt_labels_3d will cause bug when
        # # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # # as gt_labels_3d[1] and cause out of index error
        # gt_labels_mask = (num_points_in_gt >= self.min_num_points) & (bev_radius_mask)

        input_dict["gt_bboxes_3d"] = input_dict["gt_bboxes_3d"][gt_bboxes_mask]
        input_dict["gt_labels_3d"] = input_dict["gt_labels_3d"][gt_bboxes_mask]

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(range_radius={self.range_radius}, min_num_points={self.min_num_points})"
        return repr_str
