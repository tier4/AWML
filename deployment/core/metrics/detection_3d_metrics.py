"""
3D Detection Metrics Interface using autoware_perception_evaluation.

This module provides an interface to compute 3D detection metrics (mAP, mAPH)
using autoware_perception_evaluation, ensuring consistent metrics between
training evaluation (T4MetricV2) and deployment evaluation.

Usage:
    config = Detection3DMetricsConfig(
        class_names=["car", "truck", "bus", "bicycle", "pedestrian"],
        frame_id="base_link",
    )
    interface = Detection3DMetricsInterface(config)

    # Add frames
    for pred, gt in zip(predictions_list, ground_truths_list):
        interface.add_frame(
            predictions=pred,  # List[Dict] with bbox_3d, label, score
            ground_truths=gt,  # List[Dict] with bbox_3d, label
        )

    # Compute metrics
    metrics = interface.compute_metrics()
    # Returns: {"mAP_center_distance_bev_0.5": 0.7, ...}
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.label import AutowareLabel, Label
from perception_eval.common.object import DynamicObject
from perception_eval.common.shape import Shape, ShapeType
from perception_eval.config.perception_evaluation_config import PerceptionEvaluationConfig
from perception_eval.evaluation.metrics import MetricsScore
from perception_eval.evaluation.result.perception_frame_config import (
    CriticalObjectFilterConfig,
    PerceptionPassFailConfig,
)
from perception_eval.manager import PerceptionEvaluationManager
from pyquaternion import Quaternion

from deployment.core.metrics.base_metrics_interface import BaseMetricsConfig, BaseMetricsInterface, DetectionSummary

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection3DMetricsConfig(BaseMetricsConfig):
    """Configuration for 3D detection metrics.

    Attributes:
        class_names: List of class names for evaluation.
        frame_id: Frame ID for evaluation (e.g., "base_link").
        evaluation_config_dict: Configuration dict for perception evaluation.
            Example:
                {
                    "evaluation_task": "detection",
                    "target_labels": ["car", "truck", "bus", "bicycle", "pedestrian"],
                    "center_distance_bev_thresholds": [0.5, 1.0, 2.0, 4.0],
                    "plane_distance_thresholds": [2.0, 4.0],
                    "iou_2d_thresholds": None,
                    "iou_3d_thresholds": None,
                    "label_prefix": "autoware",
                    "max_distance": 121.0,
                    "min_distance": -121.0,
                    "min_point_numbers": 0,
                }
        critical_object_filter_config: Config for filtering critical objects.
            Example:
                {
                    "target_labels": ["car", "truck", "bus", "bicycle", "pedestrian"],
                    "ignore_attributes": None,
                    "max_distance_list": [121.0, 121.0, 121.0, 121.0, 121.0],
                    "min_distance_list": [-121.0, -121.0, -121.0, -121.0, -121.0],
                }
        frame_pass_fail_config: Config for pass/fail criteria.
            Example:
                {
                    "target_labels": ["car", "truck", "bus", "bicycle", "pedestrian"],
                    "matching_threshold_list": [2.0, 2.0, 2.0, 2.0, 2.0],
                    "confidence_threshold_list": None,
                }
    """

    evaluation_config_dict: Optional[Dict[str, Any]] = None
    critical_object_filter_config: Optional[Dict[str, Any]] = None
    frame_pass_fail_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # Set default evaluation config if not provided
        if self.evaluation_config_dict is None:
            default_eval_config = {
                "evaluation_task": "detection",
                "target_labels": self.class_names,
                "center_distance_bev_thresholds": [0.5, 1.0, 2.0, 4.0],
                "plane_distance_thresholds": [2.0, 4.0],
                "iou_2d_thresholds": None,
                "iou_3d_thresholds": None,
                "label_prefix": "autoware",
                "max_distance": 121.0,
                "min_distance": -121.0,
                "min_point_numbers": 0,
            }
            object.__setattr__(self, "evaluation_config_dict", default_eval_config)

        # Set default critical object filter config if not provided
        if self.critical_object_filter_config is None:
            num_classes = len(self.class_names)
            default_filter_config = {
                "target_labels": self.class_names,
                "ignore_attributes": None,
                "max_distance_list": [121.0] * num_classes,
                "min_distance_list": [-121.0] * num_classes,
            }
            object.__setattr__(self, "critical_object_filter_config", default_filter_config)

        # Set default frame pass fail config if not provided
        if self.frame_pass_fail_config is None:
            num_classes = len(self.class_names)
            default_pass_fail_config = {
                "target_labels": self.class_names,
                "matching_threshold_list": [2.0] * num_classes,
                "confidence_threshold_list": None,
            }
            object.__setattr__(self, "frame_pass_fail_config", default_pass_fail_config)


class Detection3DMetricsInterface(BaseMetricsInterface):
    # TODO(vividf): refactor this class after refactoring T4MetricV2
    """
    Interface for computing 3D detection metrics using autoware_perception_evaluation.

    This interface provides a simplified interface for the deployment framework to
    compute mAP, mAPH, and other detection metrics that are consistent with
    the T4MetricV2 used during training.
    """

    _UNKNOWN = "unknown"

    def __init__(
        self,
        config: Detection3DMetricsConfig,
        data_root: str = "data/t4dataset/",
        result_root_directory: str = "/tmp/perception_eval/",
    ):
        """
        Initialize the 3D detection metrics interface.

        Args:
            config: Configuration for 3D detection metrics.
            data_root: Root directory of the dataset.
            result_root_directory: Directory for saving evaluation results.
        """
        super().__init__(config)
        self.data_root = data_root
        self.result_root_directory = result_root_directory

        cfg_dict = config.evaluation_config_dict or {}
        self._evaluation_cfg_dict: Dict[str, Any] = dict(cfg_dict)

        # Create multiple evaluators for different distance ranges (like T4MetricV2)
        min_distance = cfg_dict.get("min_distance")
        max_distance = cfg_dict.get("max_distance")

        if isinstance(min_distance, (int, float)) and isinstance(max_distance, (int, float)):
            min_distance = [float(min_distance)]
            max_distance = [float(max_distance)]
        elif not isinstance(min_distance, list) or not isinstance(max_distance, list):
            raise ValueError(
                "min_distance and max_distance must be either scalars (int/float) or lists for multi-evaluator mode. "
                f"Got min_distance={type(min_distance)}, max_distance={type(max_distance)}"
            )

        if len(min_distance) != len(max_distance):
            raise ValueError(
                f"min_distance and max_distance must have the same length. "
                f"Got len(min_distance)={len(min_distance)}, len(max_distance)={len(max_distance)}"
            )

        if len(min_distance) == 0:
            raise ValueError("min_distance and max_distance lists cannot be empty")

        # Create distance ranges and evaluators
        self._bev_distance_ranges = list(zip(min_distance, max_distance))
        self.evaluators: Dict[str, Dict[str, Any]] = {}
        self._create_evaluators(config)

        self.gt_count_total: int = 0
        self.pred_count_total: int = 0
        self.gt_count_by_label: Dict[str, int] = {}
        self.pred_count_by_label: Dict[str, int] = {}
        self._last_metrics_by_eval_name: Dict[str, MetricsScore] = {}

    def _create_evaluators(self, config: Detection3DMetricsConfig) -> None:
        """Create multiple evaluators for different distance ranges (like T4MetricV2)."""
        range_filter_name = "bev_center"

        for min_dist, max_dist in self._bev_distance_ranges:
            # Create a copy of evaluation_config_dict with single distance values
            eval_config_dict = dict(config.evaluation_config_dict or {})
            eval_config_dict["min_distance"] = min_dist
            eval_config_dict["max_distance"] = max_dist

            # Create perception evaluation config for this range
            evaluator_config = PerceptionEvaluationConfig(
                dataset_paths=self.data_root,
                frame_id=config.frame_id,
                result_root_directory=self.result_root_directory,
                evaluation_config_dict=eval_config_dict,
                load_raw_data=False,
            )

            # Create critical object filter config
            critical_object_filter_config = CriticalObjectFilterConfig(
                evaluator_config=evaluator_config,
                **config.critical_object_filter_config,
            )

            # Create frame pass fail config
            frame_pass_fail_config = PerceptionPassFailConfig(
                evaluator_config=evaluator_config,
                **config.frame_pass_fail_config,
            )

            evaluator_name = f"{range_filter_name}_{min_dist}-{max_dist}"

            self.evaluators[evaluator_name] = {
                "evaluator": None,  # Will be created on reset
                "evaluator_config": evaluator_config,
                "critical_object_filter_config": critical_object_filter_config,
                "frame_pass_fail_config": frame_pass_fail_config,
                "bev_distance_range": (min_dist, max_dist),
            }

    def reset(self) -> None:
        """Reset the interface for a new evaluation session."""
        # Reset all evaluators
        for eval_name, eval_data in self.evaluators.items():
            eval_data["evaluator"] = PerceptionEvaluationManager(
                evaluation_config=eval_data["evaluator_config"],
                load_ground_truth=False,
                metric_output_dir=None,
            )

        self._frame_count = 0
        self.gt_count_total = 0
        self.pred_count_total = 0
        self.gt_count_by_label = {}
        self.pred_count_by_label = {}
        self._last_metrics_by_eval_name = {}

    def _convert_index_to_label(self, label_index: int) -> Label:
        """Convert a label index to a Label object.

        Args:
            label_index: Index of the label in class_names.

        Returns:
            Label object with AutowareLabel.
        """
        if 0 <= label_index < len(self.class_names):
            class_name = self.class_names[label_index]
        else:
            class_name = self._UNKNOWN

        autoware_label = AutowareLabel.__members__.get(class_name.upper(), AutowareLabel.UNKNOWN)
        return Label(label=autoware_label, name=class_name)

    def _predictions_to_dynamic_objects(
        self,
        predictions: List[Dict[str, Any]],
        unix_time: float,
    ) -> List[DynamicObject]:
        """Convert prediction dicts to DynamicObject instances.

        Args:
            predictions: List of prediction dicts with keys:
                - bbox_3d: [x, y, z, l, w, h, yaw] or [x, y, z, l, w, h, yaw, vx, vy]
                  (Same format as mmdet3d LiDARInstance3DBoxes)
                - label: int (class index)
                - score: float (confidence score)
            unix_time: Unix timestamp for the frame.

        Returns:
            List of DynamicObject instances.
        """
        estimated_objects = []
        for pred in predictions:
            bbox = pred.get("bbox_3d", [])
            if len(bbox) < 7:
                continue

            # Extract bbox components
            # mmdet3d LiDARInstance3DBoxes format: [x, y, z, l, w, h, yaw, vx, vy]
            # where l=length, w=width, h=height
            x, y, z = bbox[0], bbox[1], bbox[2]
            l, w, h = bbox[3], bbox[4], bbox[5]
            yaw = bbox[6]

            # Velocity (optional)
            vx = bbox[7] if len(bbox) > 7 else 0.0
            vy = bbox[8] if len(bbox) > 8 else 0.0

            # Create quaternion from yaw
            orientation = Quaternion(np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))

            # Get label
            label_idx = pred.get("label", 0)
            semantic_label = self._convert_index_to_label(int(label_idx))

            # Get score
            score = float(pred.get("score", 0.0))

            # Shape size follows autoware_perception_evaluation convention: (length, width, height)
            dynamic_obj = DynamicObject(
                unix_time=unix_time,
                frame_id=self.frame_id,
                position=(x, y, z),
                orientation=orientation,
                shape=Shape(shape_type=ShapeType.BOUNDING_BOX, size=(l, w, h)),
                velocity=(vx, vy, 0.0),
                semantic_score=score,
                semantic_label=semantic_label,
            )
            estimated_objects.append(dynamic_obj)

        return estimated_objects

    def _ground_truths_to_frame_ground_truth(
        self,
        ground_truths: List[Dict[str, Any]],
        unix_time: float,
        frame_name: str = "0",
    ) -> FrameGroundTruth:
        """Convert ground truth dicts to FrameGroundTruth instance.

        Args:
            ground_truths: List of ground truth dicts with keys:
                - bbox_3d: [x, y, z, l, w, h, yaw] or [x, y, z, l, w, h, yaw, vx, vy]
                  (Same format as mmdet3d LiDARInstance3DBoxes)
                - label: int (class index)
                - num_lidar_pts: int (optional, number of lidar points)
            unix_time: Unix timestamp for the frame.
            frame_name: Name/ID of the frame.

        Returns:
            FrameGroundTruth instance.
        """
        gt_objects = []
        for gt in ground_truths:
            bbox = gt.get("bbox_3d", [])
            if len(bbox) < 7:
                continue

            # Extract bbox components
            # mmdet3d LiDARInstance3DBoxes format: [x, y, z, l, w, h, yaw, vx, vy]
            # where l=length, w=width, h=height
            x, y, z = bbox[0], bbox[1], bbox[2]
            l, w, h = bbox[3], bbox[4], bbox[5]
            yaw = bbox[6]

            # Velocity (optional)
            vx = bbox[7] if len(bbox) > 7 else 0.0
            vy = bbox[8] if len(bbox) > 8 else 0.0

            # Create quaternion from yaw
            orientation = Quaternion(np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))

            # Get label
            label_idx = gt.get("label", 0)
            semantic_label = self._convert_index_to_label(int(label_idx))

            # Get point count (optional)
            num_pts = gt.get("num_lidar_pts", 0)

            # Shape size follows autoware_perception_evaluation convention: (length, width, height)
            dynamic_obj = DynamicObject(
                unix_time=unix_time,
                frame_id=self.frame_id,
                position=(x, y, z),
                orientation=orientation,
                shape=Shape(shape_type=ShapeType.BOUNDING_BOX, size=(l, w, h)),
                velocity=(vx, vy, 0.0),
                semantic_score=1.0,  # GT always has score 1.0
                semantic_label=semantic_label,
                pointcloud_num=int(num_pts),
            )
            gt_objects.append(dynamic_obj)

        return FrameGroundTruth(
            unix_time=unix_time,
            frame_name=frame_name,
            objects=gt_objects,
            transforms=None,
            raw_data=None,
        )

    def add_frame(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        frame_name: Optional[str] = None,
    ) -> None:
        """Add a frame of predictions and ground truths for evaluation.

        Args:
            predictions: List of prediction dicts with keys:
                - bbox_3d: [x, y, z, l, w, h, yaw] or [x, y, z, l, w, h, yaw, vx, vy]
                - label: int (class index)
                - score: float (confidence score)
            ground_truths: List of ground truth dicts with keys:
                - bbox_3d: [x, y, z, l, w, h, yaw] or [x, y, z, l, w, h, yaw, vx, vy]
                - label: int (class index)
                - num_lidar_pts: int (optional)
            frame_name: Optional name for the frame.
        """
        needs_reset = any(eval_data["evaluator"] is None for eval_data in self.evaluators.values())
        if needs_reset:
            self.reset()

        unix_time = time.time()
        if frame_name is None:
            frame_name = str(self._frame_count)

        self.pred_count_total += len(predictions)
        self.gt_count_total += len(ground_truths)

        for p in predictions:
            try:
                label = int(p.get("label", -1))
            except Exception:
                label = -1
            if 0 <= label < len(self.class_names):
                name = self.class_names[label]
                self.pred_count_by_label[name] = self.pred_count_by_label.get(name, 0) + 1

        for g in ground_truths:
            try:
                label = int(g.get("label", -1))
            except Exception:
                label = -1
            if 0 <= label < len(self.class_names):
                name = self.class_names[label]
                self.gt_count_by_label[name] = self.gt_count_by_label.get(name, 0) + 1

        # Convert predictions to DynamicObject
        estimated_objects = self._predictions_to_dynamic_objects(predictions, unix_time)

        # Convert ground truths to FrameGroundTruth
        frame_ground_truth = self._ground_truths_to_frame_ground_truth(ground_truths, unix_time, frame_name)

        # Add frame result to all evaluators
        try:
            for eval_name, eval_data in self.evaluators.items():
                if eval_data["evaluator"] is None:
                    eval_data["evaluator"] = PerceptionEvaluationManager(
                        evaluation_config=eval_data["evaluator_config"],
                        load_ground_truth=False,
                        metric_output_dir=None,
                    )
                eval_data["evaluator"].add_frame_result(
                    unix_time=unix_time,
                    ground_truth_now_frame=frame_ground_truth,
                    estimated_objects=estimated_objects,
                    critical_object_filter_config=eval_data["critical_object_filter_config"],
                    frame_pass_fail_config=eval_data["frame_pass_fail_config"],
                )
            self._frame_count += 1
        except Exception as e:
            logger.warning(f"Failed to add frame {frame_name}: {e}")

    def compute_metrics(self) -> Dict[str, float]:
        """Compute metrics from all added frames.

        Returns:
            Dictionary of metrics with keys like:
                - mAP_center_distance_bev (mean AP across all classes, no threshold)
                - mAPH_center_distance_bev (mean APH across all classes, no threshold)
                - car_AP_center_distance_bev_0.5 (per-class AP with threshold)
                - car_AP_center_distance_bev_1.0 (per-class AP with threshold)
                - car_APH_center_distance_bev_0.5 (per-class APH with threshold)
                - etc.
                For multi-evaluator mode, metrics are prefixed with evaluator name:
                - bev_center_0.0-50.0_mAP_center_distance_bev
                - bev_center_0.0-50.0_car_AP_center_distance_bev_0.5
                - bev_center_50.0-90.0_mAP_center_distance_bev
                - etc.
                Note: mAP/mAPH keys do not include threshold; only per-class AP/APH keys do.
        """
        if self._frame_count == 0:
            logger.warning("No frames to evaluate")
            return {}

        try:
            # Cache scene results to avoid recomputing
            scene_results = {}
            for eval_name, eval_data in self.evaluators.items():
                evaluator = eval_data["evaluator"]
                if evaluator is None:
                    continue

                try:
                    metrics_score = evaluator.get_scene_result()
                    scene_results[eval_name] = metrics_score
                except Exception as e:
                    logger.warning(f"Error computing metrics for {eval_name}: {e}")

            # Process cached metrics with evaluator name prefix
            all_metrics = {}
            for eval_name, metrics_score in scene_results.items():
                eval_metrics = self._process_metrics_score(metrics_score, prefix=eval_name)
                all_metrics.update(eval_metrics)

            # Cache results for reuse by format_last_report() and get_summary()
            self._last_metrics_by_eval_name = scene_results

            return all_metrics

        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            import traceback

            traceback.print_exc()
            return {}

    def format_last_report(self) -> str:
        """Format the last metrics report using perception_eval's own __str__ implementation.

        For multi-evaluator mode, returns reports for all evaluators with distance range labels.
        Uses cached results from compute_metrics() if available to avoid recomputation.
        """
        # Use cached results if available, otherwise compute them
        if not self._last_metrics_by_eval_name:
            # Cache not available, compute now
            self.compute_metrics()

        # Format reports for all evaluators using cached results
        reports = []
        for eval_name, metrics_score in self._last_metrics_by_eval_name.items():
            try:
                # Extract distance range from evaluator name (e.g., "bev_center_0.0-50.0" -> "0.0-50.0")
                distance_range = eval_name.replace("bev_center_", "")
                report = f"\n{'='*80}\nDistance Range: {distance_range} m\n{'='*80}\n{str(metrics_score)}"
                reports.append(report)
            except Exception as e:
                logger.warning(f"Error formatting report for {eval_name}: {e}")

        return "\n".join(reports) if reports else ""

    def _process_metrics_score(self, metrics_score: MetricsScore, prefix: Optional[str] = None) -> Dict[str, float]:
        """Process MetricsScore into a flat dictionary.

        Args:
            metrics_score: MetricsScore instance from evaluator.
            prefix: Optional prefix to add to metric keys (for multi-evaluator mode).

        Returns:
            Flat dictionary of metrics.
        """
        metric_dict = {}
        key_prefix = f"{prefix}_" if prefix else ""

        for map_instance in metrics_score.mean_ap_values:
            matching_mode = map_instance.matching_mode.value.lower().replace(" ", "_")

            # Process individual AP values
            for label, aps in map_instance.label_to_aps.items():
                label_name = label.value

                for ap in aps:
                    threshold = ap.matching_threshold
                    ap_value = ap.ap

                    # Create the metric key
                    key = f"{key_prefix}{label_name}_AP_{matching_mode}_{threshold}"
                    metric_dict[key] = ap_value

            # Process individual APH values
            label_to_aphs = getattr(map_instance, "label_to_aphs", None)
            if label_to_aphs:
                for label, aphs in label_to_aphs.items():
                    label_name = label.value
                    for aph in aphs:
                        threshold = aph.matching_threshold
                        aph_value = getattr(aph, "aph", None)
                        if aph_value is None:
                            aph_value = getattr(aph, "ap", None)
                        if aph_value is None:
                            continue
                        key = f"{key_prefix}{label_name}_APH_{matching_mode}_{threshold}"
                        metric_dict[key] = aph_value

            # Add mAP and mAPH values
            map_key = f"{key_prefix}mAP_{matching_mode}"
            maph_key = f"{key_prefix}mAPH_{matching_mode}"
            metric_dict[map_key] = map_instance.map
            metric_dict[maph_key] = map_instance.maph

        return metric_dict

    @staticmethod
    def _extract_matching_modes(metrics: Mapping[str, float]) -> List[str]:
        """Extract matching modes from metrics dict keys (e.g., 'mAP_center_distance_bev' -> 'center_distance_bev').

        Supports both prefixed and non-prefixed formats:
        - Non-prefixed: "mAP_center_distance_bev"
        - Prefixed: "bev_center_0.0-50.0_mAP_center_distance_bev"
        """
        # Matches either "mAP_<mode>" or "<prefix>_mAP_<mode>"
        pat = re.compile(r"(?:^|_)mAP_(.+)$")
        modes: List[str] = []
        for k in metrics.keys():
            m = pat.search(k)
            if m:
                modes.append(m.group(1))
        # Remove duplicates while preserving order
        return list(dict.fromkeys(modes))

    def get_thresholds_for_mode(
        self, mode: str, metrics: Optional[Mapping[str, float]] = None
    ) -> Optional[List[float]]:
        """Return thresholds for a matching mode from config or inferred from metric keys."""
        cfg = self._evaluation_cfg_dict
        threshold_key = f"{mode}_thresholds"
        thresholds = cfg.get(threshold_key)
        if thresholds is not None:
            return [float(x) for x in thresholds]

        if not metrics:
            return None

        pattern = re.compile(rf"_AP(H)?_{re.escape(mode)}_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$")
        found: List[float] = []
        for k in metrics.keys():
            m = pattern.search(k)
            if m:
                try:
                    found.append(float(m.group(2)))
                except Exception:
                    pass
        return sorted(set(found)) if found else None

    def get_summary(self) -> DetectionSummary:
        """Get a summary of the evaluation including mAP and per-class metrics for all matching modes."""
        metrics = self.compute_metrics()

        modes = self._extract_matching_modes(metrics)
        if not modes:
            return DetectionSummary(
                mAP_by_mode={},
                mAPH_by_mode={},
                per_class_ap_by_mode={},
                num_frames=self._frame_count,
                detailed_metrics=metrics,
            )

        # Collect mAP/mAPH and per-class AP for each matching mode
        # Handle both prefixed (multi-evaluator) and non-prefixed metrics
        mAP_by_mode: Dict[str, float] = {}
        mAPH_by_mode: Dict[str, float] = {}
        per_class_ap_by_mode: Dict[str, Dict[str, float]] = {}

        for mode in modes:
            map_values = []
            maph_values = []

            # Use regex to match both prefixed and non-prefixed formats
            map_pattern = re.compile(rf"(?:^|_)mAP_{re.escape(mode)}$")
            maph_pattern = re.compile(rf"(?:^|_)mAPH_{re.escape(mode)}$")

            for key, value in metrics.items():
                if map_pattern.search(key):
                    map_values.append(float(value))
                if maph_pattern.search(key):
                    maph_values.append(float(value))

            if map_values:
                mAP_by_mode[mode] = float(np.mean(map_values))
            else:
                mAP_by_mode[mode] = 0.0

            if maph_values:
                mAPH_by_mode[mode] = float(np.mean(maph_values))

            # Collect AP values per class for this mode
            # Parse class name from key format: "{prefix}_<label>_AP_{mode}_{threshold}"
            # Find the label token right before "_AP_{mode}_"
            per_class_ap_values: Dict[str, List[float]] = {}
            ap_key_separator = f"_AP_{mode}_"
            for key, value in metrics.items():
                idx = key.find(ap_key_separator)
                if idx < 0:
                    continue
                # Label is the token right before "_AP_{mode}_"
                prefix_part = key[:idx]
                class_name = prefix_part.split("_")[-1]
                per_class_ap_values.setdefault(class_name, []).append(float(value))

            if per_class_ap_values:
                per_class_ap_by_mode[mode] = {k: float(np.mean(v)) for k, v in per_class_ap_values.items() if v}

        return DetectionSummary(
            mAP_by_mode=mAP_by_mode,
            mAPH_by_mode=mAPH_by_mode,
            per_class_ap_by_mode=per_class_ap_by_mode,
            num_frames=self._frame_count,
            detailed_metrics=metrics,
        )
