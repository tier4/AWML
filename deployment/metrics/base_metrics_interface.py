"""
Base Metrics Interface for unified metric computation.

This module provides the abstract base class that all task-specific metrics interfaces
must implement. It ensures a consistent contract across 3D detection, 2D detection,
and classification tasks.

All metric interfaces use autoware_perception_evaluation as the underlying computation
engine to ensure consistency between training (T4MetricV2) and deployment evaluation.

Workflow (Template Method):
    1. Create interface with a task-specific config; the subclass populates
       ``self._evaluator_specs`` (one entry per evaluator, e.g. one per distance range).
    2. ``reset()`` clears buffered frames.
    3. ``add_frame()`` converts a sample to perception_eval objects and buffers them via
       ``_buffer_frame``; no evaluator is touched yet.
    4. ``compute_metrics()`` builds each evaluator lazily, replays the buffered frames into
       it, summarizes, releases it, and flattens the result via the subclass's
       ``_process_metrics_score``. Results are cached until the next ``add_frame``/``reset``.

Building evaluators at compute time (instead of accumulating into a live evaluator during
``add_frame``) keeps peak memory independent of the number of evaluators/distance ranges.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.label import AutowareLabel, Label
from perception_eval.config.perception_evaluation_config import PerceptionEvaluationConfig
from perception_eval.evaluation.metrics import MetricsScore
from perception_eval.evaluation.result.perception_frame_config import (
    CriticalObjectFilterConfig,
    PerceptionPassFailConfig,
)
from perception_eval.manager import PerceptionEvaluationManager

logger = logging.getLogger(__name__)

# Valid 2D frame IDs for camera-based tasks (2D detection, classification)
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


def validate_2d_frame_id(frame_id: str, task_name: str) -> None:
    """Raise ValueError if ``frame_id`` is not a valid camera frame for 2D tasks."""
    if frame_id not in VALID_2D_FRAME_IDS:
        raise ValueError(f"Invalid frame_id '{frame_id}' for {task_name}. Valid options: {VALID_2D_FRAME_IDS}")


@dataclass(frozen=True)
class BaseMetricsConfig:
    """Base configuration for all metrics interfaces.

    Attributes:
        class_names: List of class names for evaluation.
        frame_id: Frame ID for evaluation (e.g., "base_link" for 3D, "cam_front" for 2D).
    """

    class_names: List[str]
    frame_id: str


@dataclass
class _EvaluatorSpec:
    """Everything needed to build and run one perception_eval evaluator.

    One spec per evaluator: 2D/classification use a single spec; 3D uses one per
    distance range. Built once at ``__init__`` and reused for every ``compute_metrics``.
    """

    eval_config: PerceptionEvaluationConfig
    filter_config: CriticalObjectFilterConfig
    passfail_config: PerceptionPassFailConfig


# Buffered per-frame perception_eval objects: (unix_time, estimated_objects, ground_truth).
BufferedFrame = Tuple[Any, List[Any], FrameGroundTruth]


class BaseMetricsInterface(ABC):
    """
    Abstract base class for all task-specific metrics interfaces.

    Subclasses implement only what differs between tasks:
        - ``add_frame``: convert a sample to perception_eval objects and call ``_buffer_frame``.
        - ``_process_metrics_score``: flatten a ``MetricsScore`` into a metric dict.
        - ``summary``: build a task-specific structured summary.

    The shared workflow (frame buffering, evaluator lifecycle, caching) lives here.

    Example:
        interface = SomeMetricsInterface(config)
        interface.reset()
        for pred, gt in data:
            interface.add_frame(pred, gt)
        metrics = interface.compute_metrics()
    """

    _UNKNOWN = "unknown"

    def __init__(self, config: BaseMetricsConfig) -> None:
        """Initialize the metrics interface.

        Args:
            config: Configuration for the metrics interface.
        """
        self.config = config
        self.class_names = config.class_names
        self.frame_id = config.frame_id

        self._frames: List[BufferedFrame] = []
        self._frame_count = 0
        self._cached_metrics: Optional[Dict[str, float]] = None
        self._last_scene_results: Dict[str, MetricsScore] = {}

        # Populated by the subclass (one entry per evaluator). Keyed by evaluator name;
        # the key becomes the metric-key prefix (empty string => no prefix).
        self._evaluator_specs: Dict[str, _EvaluatorSpec] = {}

    def _build_evaluator_spec(
        self,
        evaluation_config_dict: Dict[str, Any],
        *,
        data_root: str,
        result_root_directory: str,
        critical_object_filter_config: Dict[str, Any],
        frame_pass_fail_config: Dict[str, Any],
    ) -> _EvaluatorSpec:
        """Build an :class:`_EvaluatorSpec` from the three perception_eval config dicts."""
        eval_config = PerceptionEvaluationConfig(
            dataset_paths=data_root,
            frame_id=self.frame_id,
            result_root_directory=result_root_directory,
            evaluation_config_dict=evaluation_config_dict,
            load_raw_data=False,
        )
        return _EvaluatorSpec(
            eval_config=eval_config,
            filter_config=CriticalObjectFilterConfig(
                evaluator_config=eval_config,
                **critical_object_filter_config,
            ),
            passfail_config=PerceptionPassFailConfig(
                evaluator_config=eval_config,
                **frame_pass_fail_config,
            ),
        )

    def reset(self) -> None:
        """Reset the interface for a new evaluation session.

        Clears buffered frames and cached results. Evaluators are created on demand in
        ``compute_metrics`` and never held between runs, so there is nothing else to free.
        """
        self._frames = []
        self._frame_count = 0
        self._cached_metrics = None
        self._last_scene_results = {}

    def _buffer_frame(
        self, unix_time: Any, estimated_objects: List[Any], frame_ground_truth: FrameGroundTruth
    ) -> None:
        """Buffer one frame's perception_eval objects and invalidate cached metrics."""
        self._frames.append((unix_time, estimated_objects, frame_ground_truth))
        self._frame_count += 1
        self._cached_metrics = None

    def compute_metrics(self) -> Dict[str, float]:
        """Compute metrics from all buffered frames.

        Each evaluator is built, replayed, summarized, and released one at a time so peak
        memory stays independent of the number of evaluators. Results are cached until the
        next ``add_frame``/``reset``.

        Returns:
            Flat dictionary of metric names to values (empty if no frames were added).
        """
        if self._cached_metrics is not None:
            return self._cached_metrics

        if self._frame_count == 0:
            logger.warning("No frames to evaluate")
            return {}

        scene_results: Dict[str, MetricsScore] = {}
        all_metrics: Dict[str, float] = {}
        for name, spec in self._evaluator_specs.items():
            try:
                evaluator = PerceptionEvaluationManager(
                    evaluation_config=spec.eval_config,
                    load_ground_truth=False,
                    metric_output_dir=None,
                )
                for unix_time, estimated_objects, frame_ground_truth in self._frames:
                    evaluator.add_frame_result(
                        unix_time=unix_time,
                        ground_truth_now_frame=frame_ground_truth,
                        estimated_objects=estimated_objects,
                        critical_object_filter_config=spec.filter_config,
                        frame_pass_fail_config=spec.passfail_config,
                    )
                metrics_score = evaluator.get_scene_result()
                scene_results[name] = metrics_score
                all_metrics.update(self._process_metrics_score(metrics_score, prefix=name or None))
                self._capture_evaluator(name, evaluator)
            except Exception:
                logger.exception("Error computing metrics for evaluator '%s'", name)

        self._last_scene_results = scene_results
        self._cached_metrics = all_metrics
        return all_metrics

    def _capture_evaluator(self, name: str, evaluator: PerceptionEvaluationManager) -> None:
        """Hook called with each evaluator before it is released.

        Default no-op. Subclasses that need per-frame results (e.g. classification's
        confusion matrix) override this to capture state from the evaluator.
        """

    @abstractmethod
    def add_frame(self, *args: Any, **kwargs: Any) -> None:
        """
        Convert a frame of predictions/ground truths to perception_eval objects and buffer
        them via ``_buffer_frame``. The specific arguments depend on the task type:
        - 3D Detection: predictions: List[Dict], ground_truths: List[Dict]
        - 2D Detection: predictions: List[Dict], ground_truths: List[Dict]
        - Classification: prediction: int, ground_truth: int, probabilities: List[float]
        """

    @abstractmethod
    def _process_metrics_score(self, metrics_score: MetricsScore, prefix: Optional[str] = None) -> Dict[str, float]:
        """Flatten a perception_eval ``MetricsScore`` into a flat metric dictionary.

        Args:
            metrics_score: Score returned by ``PerceptionEvaluationManager.get_scene_result``.
            prefix: Optional prefix for metric keys (set per evaluator, e.g. distance range).
        """

    @property
    @abstractmethod
    def summary(self) -> Any:
        """Get a structured summary of the evaluation (task-specific dataclass)."""

    def _convert_index_to_label(self, label_index: int) -> Label:
        """Convert a label index to a perception_eval Label object.

        Args:
            label_index: Index of the label in class_names.

        Returns:
            Label object with AutowareLabel (UNKNOWN for out-of-range indices).
        """
        if 0 <= label_index < len(self.class_names):
            class_name = self.class_names[label_index]
        else:
            class_name = self._UNKNOWN

        autoware_label = AutowareLabel.__members__.get(class_name.upper(), AutowareLabel.UNKNOWN)
        return Label(label=autoware_label, name=class_name)

    def format_metrics_report(self) -> Optional[str]:
        """Format the metrics report as a human-readable string.

        Optional hook overridden by subclasses for task-specific formatting. Returns None
        by default.
        """
        return None
