"""Preprocessing for FRNet deployment.

Applies range-view interpolation + frustum region grouping to convert a raw
point cloud into the tensor dict expected by the FRNet backbone.  Reuses the
same RangeInterpolation and FrustumRangePreprocessor used during training.
"""

from __future__ import annotations

import copy
from time import time

import numpy as np
import numpy.typing as npt
from mmdet3d.registry import MODELS, TRANSFORMS
from mmdet3d.structures import Det3DDataSample, PointData
from mmdet3d.structures.points import LiDARPoints
from mmengine.config import Config
from mmengine.logging import MMLogger


class Preprocessing:
    """FRNet point-cloud preprocessor.

    Uses RangeInterpolation (projects points to range image, fills gaps)
    and FrustumRangePreprocessor (computes voxel coords, builds batch dict).
    Both are instantiated from the model config to stay in sync with training.
    """

    def __init__(self, config: Config) -> None:
        config = copy.deepcopy(config)
        self.logger = MMLogger.get_current_instance()

        ri_cfg = next(c for c in config.test_pipeline if c["type"] == "RangeInterpolation")
        ri_cfg = {k: v for k, v in ri_cfg.items() if k != "type"}
        RangeInterpolation = TRANSFORMS.get("RangeInterpolation")
        self._range_interpolation = RangeInterpolation(**ri_cfg)

        dp_cfg = dict(config.model["data_preprocessor"])
        dp_cfg.pop("type", None)
        FrustumRangePreprocessor = MODELS.get("FrustumRangePreprocessor")
        self._frustum_preprocessor = FrustumRangePreprocessor(**dp_cfg)

    def preprocess(self, points: npt.NDArray[np.float32]) -> dict:
        """Run range interpolation + frustum grouping on raw points (N, D)."""
        t_start = time()

        # 1. Range interpolation
        lidar_points = LiDARPoints(points, points_dim=points.shape[1])
        ri_result = self._range_interpolation.transform({"points": lidar_points})
        num_points = ri_result["num_points"]
        interpolated = ri_result["points"]  # LiDARPoints (torch-backed)

        # 2. Frustum region grouping (expects batched input)
        data_sample = Det3DDataSample()
        data_sample.gt_pts_seg = PointData()
        data = {
            "inputs": {"points": [interpolated.tensor]},
            "data_samples": [data_sample],
        }
        batch_inputs_dict = self._frustum_preprocessor.forward(data, training=False)["inputs"]

        t_end = time()
        latency = np.round((t_end - t_start) * 1e3, 2)
        self.logger.info(f"Preprocessing latency: {latency} ms")

        batch_inputs_dict["num_points"] = num_points
        return batch_inputs_dict
