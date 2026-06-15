from typing import List

import numpy as np
from mmcv.transforms import BaseTransform
from mmdet3d.structures.ops import box_np_ops
from mmengine.registry import TRANSFORMS


@TRANSFORMS.register_module()
class Local3DBBoxExpand(BaseTransform):
    """Locally expand the 3D bounding boxes by scaling the width, which it doesn't scale the points.

    Args:
        expand_widths: (List[float]): Uniformly sampled expand width.
        width_dim: (int): The dimension of the width. Default is 4, which is the width dimension of the 3D
                  bounding box. Since 3D Bbox is in the format of [x, y, z, dx, dy, dz, heading], the width dimension is the
                    4th dimension.
            label_ids: (List[int]): The label IDs to expand. If None, all label IDs will be expanded.
    """

    def __init__(self, expand_widths: List[float], width_dim: int = 4, label_ids: List[int] = None) -> None:
        assert isinstance(expand_widths, list)
        assert len(expand_widths) == 2
        assert expand_widths[0] < expand_widths[1]
        self.expand_widths = expand_widths
        self.width_dim = width_dim
        self.label_ids = label_ids

    def transform(self, input_dict: dict) -> dict:
        """Call function to locally augment the 3D bounding boxes by scaling the width.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after locally augmenting the 3D bounding boxes by scaling the width, 'gt_bboxes_3d' \
                key is updated in the result dict.
        """
        # Label mask
        if self.label_ids is not None:
            label_masks = [True if label in self.label_ids else False for label in input_dict["gt_labels_3d"]]
        else:
            label_masks = np.ones(len(input_dict["gt_labels_3d"]), dtype=bool)

        for i in range(len(input_dict["gt_bboxes_3d"])):
            if not label_masks[i]:
                continue

            expand_width = np.random.uniform(self.expand_widths[0], self.expand_widths[1])
            input_dict["gt_bboxes_3d"].tensor[i, self.width_dim] += expand_width

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(expand_widths={self.expand_widths}, width_dim={self.width_dim}, label_ids={self.label_ids})"
        return repr_str
