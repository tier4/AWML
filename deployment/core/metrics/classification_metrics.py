"""
Classification Metrics Interface using autoware_perception_evaluation.

This module provides an interface to compute classification metrics (accuracy, precision,
recall, F1) using autoware_perception_evaluation, ensuring consistent metrics between
training evaluation and deployment evaluation.

Usage:
    config = ClassificationMetricsConfig(
        class_names=["miscalibrated", "calibrated"],
    )
    interface = ClassificationMetricsInterface(config)

    for pred_label, gt_label in zip(predictions, ground_truths):
        interface.add_frame(prediction=pred_label, ground_truth=gt_label)

    metrics = interface.compute_metrics()
    # Returns: {"accuracy": 0.95, "precision": 0.94, "recall": 0.96, "f1score": 0.95, ...}
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.label import AutowareLabel, Label
from perception_eval.common.object2d import DynamicObject2D
from perception_eval.common.schema import FrameID
from perception_eval.config.perception_evaluation_config import PerceptionEvaluationConfig
from perception_eval.evaluation.metrics import MetricsScore
from perception_eval.evaluation.result.perception_frame_config import (
    CriticalObjectFilterConfig,
    PerceptionPassFailConfig,
)
from perception_eval.manager import PerceptionEvaluationManager

from deployment.core.metrics.base_metrics_interface import (
    BaseMetricsConfig,
    BaseMetricsInterface,
    ClassificationSummary,
)

logger = logging.getLogger(__name__)

# Valid 2D frame IDs for camera-based classification
VALID_2D_FRAME_IDS = [
    "cam_front",
    "cam_front_right",
    "cam_front_left",
    "cam_front_lower",
    "cam_back",
    "cam_back_left",
    "cam_back_right",
    "cam_traffic_light_near",
    "cam_traffic_light_far",
    "cam_traffic_light",
]


@dataclass(frozen=True)
class ClassificationMetricsConfig(BaseMetricsConfig):
    """Configuration for classification metrics.

    Attributes:
        class_names: List of class names for evaluation.
        frame_id: Camera frame ID for evaluation (default: "cam_front").
        evaluation_config_dict: Configuration dict for perception evaluation.
        critical_object_filter_config: Config for filtering critical objects.
        frame_pass_fail_config: Config for pass/fail criteria.
    """

    frame_id: str = "cam_front"
    evaluation_config_dict: Optional[Dict[str, Any]] = None
    critical_object_filter_config: Optional[Dict[str, Any]] = None
    frame_pass_fail_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.frame_id not in VALID_2D_FRAME_IDS:
            raise ValueError(
                f"Invalid frame_id '{self.frame_id}' for classification. " f"Valid options: {VALID_2D_FRAME_IDS}"
            )

        if self.evaluation_config_dict is None:
            object.__setattr__(
                self,
                "evaluation_config_dict",
                {
                    "evaluation_task": "classification2d",
                    "target_labels": self.class_names,
                    "center_distance_thresholds": None,
                    "center_distance_bev_thresholds": None,
                    "plane_distance_thresholds": None,
                    "iou_2d_thresholds": None,
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
                    "matching_threshold_list": [1.0] * len(self.class_names),
                    "confidence_threshold_list": None,
                },
            )


class ClassificationMetricsInterface(BaseMetricsInterface):
    """Interface for computing classification metrics using autoware_perception_evaluation.

    Metrics computed:
    - Accuracy: TP / (num_predictions + num_gt - TP)
    - Precision: TP / (TP + FP)
    - Recall: TP / num_gt
    - F1 Score: 2 * precision * recall / (precision + recall)
    - Per-class accuracy, precision, recall, F1
    """

    def __init__(
        self,
        config: ClassificationMetricsConfig,
        data_root: str = "data/t4dataset/",
        result_root_directory: str = "/tmp/perception_eval_classification/",
    ):
        """Initialize the classification metrics interface.

        Args:
            config: Configuration for classification metrics.
            data_root: Root directory of the dataset.
            result_root_directory: Directory for saving evaluation results.
        """
        super().__init__(config)
        self.config: ClassificationMetricsConfig = config

        self.perception_eval_config = PerceptionEvaluationConfig(
            dataset_paths=data_root,
            frame_id=config.frame_id,
            result_root_directory=result_root_directory,
            evaluation_config_dict=config.evaluation_config_dict,
            load_raw_data=False,
        )

        self.critical_object_filter_config = CriticalObjectFilterConfig(
            evaluator_config=self.perception_eval_config,
            **config.critical_object_filter_config,
        )

        self.frame_pass_fail_config = PerceptionPassFailConfig(
            evaluator_config=self.perception_eval_config,
            **config.frame_pass_fail_config,
        )

        self.evaluator: Optional[PerceptionEvaluationManager] = None

    def reset(self) -> None:
        """Reset the interface for a new evaluation session."""
        self.evaluator = PerceptionEvaluationManager(
            evaluation_config=self.perception_eval_config,
            load_ground_truth=False,
            metric_output_dir=None,
        )
        self._frame_count = 0

    def _convert_index_to_label(self, label_index: int) -> Label:
        """Convert a label index to a Label object."""
        if 0 <= label_index < len(self.class_names):
            class_name = self.class_names[label_index]
        else:
            class_name = "unknown"

        autoware_label = AutowareLabel.__members__.get(class_name.upper(), AutowareLabel.UNKNOWN)
        return Label(label=autoware_label, name=class_name)

    def _create_dynamic_object_2d(
        self,
        label_index: int,
        unix_time: int,
        score: float = 1.0,
        uuid: Optional[str] = None,
    ) -> DynamicObject2D:
        """Create a DynamicObject2D for classification (roi=None for image-level)."""
        return DynamicObject2D(
            unix_time=unix_time,
            frame_id=FrameID.from_value(self.frame_id),
            semantic_score=score,
            semantic_label=self._convert_index_to_label(label_index),
            roi=None,
            uuid=uuid,
        )

    def add_frame(
        self,
        prediction: int,
        ground_truth: int,
        probabilities: Optional[List[float]] = None,
        frame_name: Optional[str] = None,
    ) -> None:
        """Add a single prediction and ground truth for evaluation.

        Args:
            prediction: Predicted class index.
            ground_truth: Ground truth class index.
            probabilities: Optional probability scores for each class.
            frame_name: Optional name for the frame.
        """
        if self.evaluator is None:
            self.reset()

        unix_time = int(time.time() * 1e6)
        if frame_name is None:
            frame_name = str(self._frame_count)

        # Get confidence score from probabilities if available
        score = 1.0
        if probabilities is not None and len(probabilities) > prediction:
            score = float(probabilities[prediction])

        # Create prediction and ground truth objects
        estimated_object = self._create_dynamic_object_2d(
            label_index=prediction, unix_time=unix_time, score=score, uuid=frame_name
        )
        gt_object = self._create_dynamic_object_2d(
            label_index=ground_truth, unix_time=unix_time, score=1.0, uuid=frame_name
        )

        frame_ground_truth = FrameGroundTruth(
            unix_time=unix_time,
            frame_name=frame_name,
            objects=[gt_object],
            transforms=None,
            raw_data=None,
        )

        try:
            self.evaluator.add_frame_result(
                unix_time=unix_time,
                ground_truth_now_frame=frame_ground_truth,
                estimated_objects=[estimated_object],
                critical_object_filter_config=self.critical_object_filter_config,
                frame_pass_fail_config=self.frame_pass_fail_config,
            )
            self._frame_count += 1
        except Exception as e:
            logger.warning(f"Failed to add frame {frame_name}: {e}")

    def compute_metrics(self) -> Dict[str, float]:
        """Compute metrics from all added predictions.

        Returns:
            Dictionary of metrics including accuracy, precision, recall, f1score,
            and per-class metrics.
        """
        if self.evaluator is None or self._frame_count == 0:
            logger.warning("No samples to evaluate")
            return {}

        try:
            metrics_score: MetricsScore = self.evaluator.get_scene_result()
            return self._process_metrics_score(metrics_score)
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            import traceback

            traceback.print_exc()
            return {}

    def _process_metrics_score(self, metrics_score: MetricsScore) -> Dict[str, float]:
        """Process MetricsScore into a flat dictionary."""
        metric_dict = {}

        for classification_score in metrics_score.classification_scores:
            # Get overall metrics
            accuracy, precision, recall, f1score = classification_score._summarize()

            # Handle inf values (replace with 0.0)
            metric_dict["accuracy"] = 0.0 if accuracy == float("inf") else accuracy
            metric_dict["precision"] = 0.0 if precision == float("inf") else precision
            metric_dict["recall"] = 0.0 if recall == float("inf") else recall
            metric_dict["f1score"] = 0.0 if f1score == float("inf") else f1score

            # Process per-class metrics
            for acc in classification_score.accuracies:
                if not acc.target_labels:
                    continue

                target_label = acc.target_labels[0]
                class_name = getattr(target_label, "name", str(target_label))

                metric_dict[f"{class_name}_accuracy"] = 0.0 if acc.accuracy == float("inf") else acc.accuracy
                metric_dict[f"{class_name}_precision"] = 0.0 if acc.precision == float("inf") else acc.precision
                metric_dict[f"{class_name}_recall"] = 0.0 if acc.recall == float("inf") else acc.recall
                metric_dict[f"{class_name}_f1score"] = 0.0 if acc.f1score == float("inf") else acc.f1score
                metric_dict[f"{class_name}_tp"] = acc.num_tp
                metric_dict[f"{class_name}_fp"] = acc.num_fp
                metric_dict[f"{class_name}_num_gt"] = acc.num_ground_truth
                metric_dict[f"{class_name}_num_pred"] = acc.objects_results_num

        metric_dict["total_samples"] = self._frame_count
        return metric_dict

    # TODO(vividf): Remove after autoware_perception_evaluation supports confusion matrix.
    def get_confusion_matrix(self) -> np.ndarray:
        """Get the confusion matrix.

        Returns:
            2D numpy array where cm[i][j] = count of ground truth i predicted as j.
        """
        num_classes = len(self.class_names)
        if self.evaluator is None or self._frame_count == 0:
            return np.zeros((num_classes, num_classes), dtype=int)

        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

        for frame_result in self.evaluator.frame_results:
            if not frame_result.object_results:
                continue

            for obj_result in frame_result.object_results:
                if obj_result.ground_truth_object is None:
                    continue

                pred_name = obj_result.estimated_object.semantic_label.name
                gt_name = obj_result.ground_truth_object.semantic_label.name

                # Find indices
                pred_idx = next(
                    (i for i, n in enumerate(self.class_names) if n.lower() == pred_name.lower()),
                    -1,
                )
                gt_idx = next(
                    (i for i, n in enumerate(self.class_names) if n.lower() == gt_name.lower()),
                    -1,
                )

                if 0 <= pred_idx < num_classes and 0 <= gt_idx < num_classes:
                    confusion_matrix[gt_idx, pred_idx] += 1

        return confusion_matrix

    def get_summary(self) -> ClassificationSummary:
        """Get a summary of the evaluation.

        Returns:
            ClassificationSummary with aggregate metrics.
        """
        metrics = self.compute_metrics()

        if not metrics:
            return ClassificationSummary()

        per_class_accuracy = {
            name: metrics[f"{name}_accuracy"] for name in self.class_names if f"{name}_accuracy" in metrics
        }

        return ClassificationSummary(
            accuracy=metrics.get("accuracy", 0.0),
            precision=metrics.get("precision", 0.0),
            recall=metrics.get("recall", 0.0),
            f1score=metrics.get("f1score", 0.0),
            per_class_accuracy=per_class_accuracy,
            confusion_matrix=self.get_confusion_matrix().tolist(),
            num_samples=self._frame_count,
            detailed_metrics=metrics,
        )
