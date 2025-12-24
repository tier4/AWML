"""
Base Metrics Interface for unified metric computation.

This module provides the abstract base class that all task-specific metrics interfaces
must implement. It ensures a consistent contract across 3D detection, 2D detection,
and classification tasks.

All metric interfaces use autoware_perception_evaluation as the underlying computation
engine to ensure consistency between training (T4MetricV2) and deployment evaluation.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaseMetricsConfig:
    """Base configuration for all metrics interfaces.

    Attributes:
        class_names: List of class names for evaluation.
        frame_id: Frame ID for evaluation (e.g., "base_link" for 3D, "camera" for 2D).
    """

    class_names: List[str]
    frame_id: str = "base_link"


@dataclass(frozen=True)
class ClassificationSummary:
    """Structured summary for classification metrics."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1score: float = 0.0
    per_class_accuracy: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: List[List[int]] = field(default_factory=list)
    num_samples: int = 0
    detailed_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1score": self.f1score,
            "per_class_accuracy": dict(self.per_class_accuracy),
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
            "num_samples": self.num_samples,
            "detailed_metrics": dict(self.detailed_metrics),
        }


@dataclass(frozen=True)
class DetectionSummary:
    """Structured summary for detection metrics (2D/3D)."""

    mAP: float = 0.0
    per_class_ap: Dict[str, float] = field(default_factory=dict)
    num_frames: int = 0
    detailed_metrics: Dict[str, float] = field(default_factory=dict)
    mAPH: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "mAP": self.mAP,
            "per_class_ap": dict(self.per_class_ap),
            "num_frames": self.num_frames,
            "detailed_metrics": dict(self.detailed_metrics),
        }
        if self.mAPH is not None:
            data["mAPH"] = self.mAPH
        return data


class BaseMetricsInterface(ABC):
    """
    Abstract base class for all task-specific metrics interfaces.

    This class defines the common interface that all metric interfaces must implement.
    Each interface wraps autoware_perception_evaluation to compute metrics consistent
    with training evaluation (T4MetricV2).

    The workflow is:
        1. Create interface with task-specific config
        2. Call reset() to start a new evaluation session
        3. Call add_frame() for each sample
        4. Call compute_metrics() to get final metrics
        5. Optionally call get_summary() for a human-readable summary

    Example:
        interface = SomeMetricsInterface(config)
        interface.reset()
        for pred, gt in data:
            interface.add_frame(pred, gt)
        metrics = interface.compute_metrics()
    """

    def __init__(self, config: BaseMetricsConfig):
        """
        Initialize the metrics interface.

        Args:
            config: Configuration for the metrics interface.
        """
        self.config = config
        self.class_names = config.class_names
        self.frame_id = config.frame_id
        self._frame_count = 0

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the interface for a new evaluation session.

        This method should clear all accumulated frame data and reinitialize
        the underlying evaluator.
        """
        pass

    @abstractmethod
    def add_frame(self, *args, **kwargs) -> None:
        """
        Add a frame of predictions and ground truths for evaluation.

        The specific arguments depend on the task type:
        - 3D Detection: predictions: List[Dict], ground_truths: List[Dict]
        - 2D Detection: predictions: List[Dict], ground_truths: List[Dict]
        - Classification: prediction: int, ground_truth: int, probabilities: List[float]
        """
        pass

    @abstractmethod
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute metrics from all added frames.

        Returns:
            Dictionary of metric names to values.
        """
        pass

    @abstractmethod
    def get_summary(self) -> Any:
        """
        Get a summary of the evaluation including primary metrics.

        Returns:
            Dictionary with summary metrics and additional information.
        """
        pass

    @property
    def frame_count(self) -> int:
        """Return the number of frames added so far."""
        return self._frame_count
