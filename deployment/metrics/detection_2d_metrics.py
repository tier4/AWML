"""
2D Detection Metrics Interface using autoware_perception_evaluation.

This module provides an interface to compute 2D detection metrics (mAP)
using autoware_perception_evaluation in 2D mode, ensuring consistent metrics
between training evaluation and deployment evaluation.

For 2D detection, the interface uses:
- IoU 2D thresholds for matching (e.g., 0.5, 0.75)
- Only AP is computed (no APH since there's no heading in 2D)

Usage:
    config = Detection2DMetricsConfig(
        class_names=["car", "truck", "bus", "bicycle", "pedestrian", "motorcycle", "trailer", "unknown"],
        frame_id="cam_front",
    )
    interface = Detection2DMetricsInterface(config)

    # Add frames
    for pred, gt in zip(predictions_list, ground_truths_list):
        interface.add_frame(
            predictions=pred,  # List[Dict] with bbox (x1,y1,x2,y2), label, score
            ground_truths=gt,  # List[Dict] with bbox (x1,y1,x2,y2), label
        )

    # Compute metrics
    metrics = interface.compute_metrics()
    # Returns: {"mAP_iou_2d_0.5": 0.7, "mAP_iou_2d_0.75": 0.65, ...}
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.object2d import DynamicObject2D
from perception_eval.common.schema import FrameID
from perception_eval.evaluation.metrics import MetricsScore

from deployment.metrics.base_metrics_interface import BaseMetricsConfig, validate_2d_frame_id
from deployment.metrics.detection_base import DetectionMetricsInterface

logger = logging.getLogger(__name__)

# Single-evaluator tasks register their evaluator under this name (empty => no key prefix).
_DEFAULT_EVALUATOR = ""


@dataclass(frozen=True)
class Detection2DMetricsConfig(BaseMetricsConfig):
    """Configuration for 2D detection metrics.

    Attributes:
        class_names: List of class names for evaluation.
        frame_id: Frame ID for evaluation. Valid values for 2D:
            "cam_front", "cam_front_right", "cam_front_left", "cam_front_lower",
            "cam_back", "cam_back_left", "cam_back_right",
            "cam_traffic_light_near", "cam_traffic_light_far", "cam_traffic_light"
        iou_thresholds: List of IoU thresholds for evaluation.
        evaluation_config_dict: Configuration dict for perception evaluation.
        critical_object_filter_config: Config for filtering critical objects.
        frame_pass_fail_config: Config for pass/fail criteria.
    """

    # Override default frame_id for 2D detection (camera frame instead of base_link)
    frame_id: str = "cam_front"
    iou_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.75])
    evaluation_config_dict: Optional[Dict[str, Any]] = None
    critical_object_filter_config: Optional[Dict[str, Any]] = None
    frame_pass_fail_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        validate_2d_frame_id(self.frame_id, "2D detection")

        if self.evaluation_config_dict is None:
            object.__setattr__(
                self,
                "evaluation_config_dict",
                {
                    "evaluation_task": "detection2d",
                    "target_labels": self.class_names,
                    "iou_2d_thresholds": self.iou_thresholds,
                    "center_distance_bev_thresholds": None,
                    "plane_distance_thresholds": None,
                    "iou_3d_thresholds": None,
                    "label_prefix": "autoware",
                },
            )

        if self.critical_object_filter_config is None:
            object.__setattr__(
                self,
                "critical_object_filter_config",
                {
                    "target_labels": self.class_names,
                    "ignore_attributes": None,
                },
            )

        if self.frame_pass_fail_config is None:
            object.__setattr__(
                self,
                "frame_pass_fail_config",
                {
                    "target_labels": self.class_names,
                    "matching_threshold_list": [0.5] * len(self.class_names),
                    "confidence_threshold_list": None,
                },
            )


class Detection2DMetricsInterface(DetectionMetricsInterface):
    """
    Interface for computing 2D detection metrics using autoware_perception_evaluation.

    Unlike 3D detection, 2D detection:
    - Uses IoU 2D for matching (based on bounding box overlap)
    - Does not compute APH (no heading information in 2D)
    - Works with image-space bounding boxes [x1, y1, x2, y2]
    """

    _supports_aph = False

    def __init__(
        self,
        config: Detection2DMetricsConfig,
        data_root: str = "data/t4dataset/",
        result_root_directory: str = "/tmp/perception_eval_2d/",
    ) -> None:
        """
        Initialize the 2D detection metrics interface.

        Args:
            config: Configuration for 2D detection metrics.
            data_root: Root directory of the dataset.
            result_root_directory: Directory for saving evaluation results.
        """
        super().__init__(config)
        self.config: Detection2DMetricsConfig = config

        self._evaluator_specs[_DEFAULT_EVALUATOR] = self._build_evaluator_spec(
            config.evaluation_config_dict,
            data_root=data_root,
            result_root_directory=result_root_directory,
            critical_object_filter_config=config.critical_object_filter_config,
            frame_pass_fail_config=config.frame_pass_fail_config,
        )

    def _to_dynamic_objects_2d(
        self,
        entries: List[Dict[str, Any]],
        unix_time: int,
        *,
        is_gt: bool,
    ) -> List[DynamicObject2D]:
        """Convert prediction/ground-truth dicts to DynamicObject2D instances.

        Args:
            entries: List of dicts with keys:
                - bbox: [x1, y1, x2, y2] (image coordinates)
                - label: int (class index)
                - score: float (confidence; ignored for ground truth, which is always 1.0)
            unix_time: Unix timestamp in microseconds.
            is_gt: Whether ``entries`` are ground truths (forces score 1.0).

        Returns:
            List of DynamicObject2D instances.
        """
        kind = "ground truth" if is_gt else "prediction"
        frame_id = FrameID.from_value(self.frame_id)

        objects: List[DynamicObject2D] = []
        for entry in entries:
            bbox = entry.get("bbox", [])
            if len(bbox) < 4:
                continue

            # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            if x2 <= x1 or y2 <= y1:
                logger.warning("Skipping %s with degenerate bbox: %s", kind, bbox)
                continue

            # Convert [x1, y1, x2, y2] to roi format (xmin, ymin, width, height).
            xmin = int(x1)
            ymin = int(y1)
            width = int(x2 - x1)
            height = int(y2 - y1)

            semantic_label = self._convert_index_to_label(int(entry.get("label", 0)))
            score = 1.0 if is_gt else float(entry.get("score", 0.0))

            objects.append(
                DynamicObject2D(
                    unix_time=unix_time,
                    frame_id=frame_id,
                    semantic_score=score,
                    semantic_label=semantic_label,
                    roi=(xmin, ymin, width, height),
                    uuid=None,
                )
            )

        return objects

    def add_frame(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        frame_name: Optional[str] = None,
    ) -> None:
        """Buffer a frame of predictions and ground truths for evaluation.

        Args:
            predictions: List of prediction dicts with keys bbox [x1,y1,x2,y2], label, score.
            ground_truths: List of ground truth dicts with keys bbox [x1,y1,x2,y2], label.
            frame_name: Optional name for the frame.
        """
        unix_time = int(time.time() * 1e6)
        if frame_name is None:
            frame_name = str(self._frame_count)

        estimated_objects = self._to_dynamic_objects_2d(predictions, unix_time, is_gt=False)
        gt_objects = self._to_dynamic_objects_2d(ground_truths, unix_time, is_gt=True)

        frame_ground_truth = FrameGroundTruth(
            unix_time=unix_time,
            frame_name=frame_name,
            objects=gt_objects,
            transforms=None,
            raw_data=None,
        )

        self._buffer_frame(unix_time, estimated_objects, frame_ground_truth)

    def _select_summary_score(self) -> Optional[MetricsScore]:
        """2D uses a single evaluator, so the summary is built from its score."""
        return self._last_scene_results.get(_DEFAULT_EVALUATOR)
