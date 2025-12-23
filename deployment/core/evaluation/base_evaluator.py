"""
Base evaluator for model evaluation in deployment.

This module provides:
- Type definitions (EvalResultDict, VerifyResultDict, ModelSpec)
- BaseEvaluator: the single base class for all task evaluators
- TaskProfile: describes task-specific metadata

All project evaluators should extend BaseEvaluator and implement
the required hooks for their specific task.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch

from deployment.core.backend import Backend
from deployment.core.evaluation.evaluator_types import (
    EvalResultDict,
    InferenceResult,
    LatencyBreakdown,
    LatencyStats,
    ModelSpec,
    VerifyResultDict,
)
from deployment.core.evaluation.verification_mixin import VerificationMixin
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.core.metrics import BaseMetricsInterface

# Re-export types
__all__ = [
    "EvalResultDict",
    "VerifyResultDict",
    "ModelSpec",
    "InferenceResult",
    "LatencyStats",
    "LatencyBreakdown",
    "TaskProfile",
    "BaseEvaluator",
    "EvaluationDefaults",
    "EVALUATION_DEFAULTS",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationDefaults:
    """Default values for evaluation settings."""

    LOG_INTERVAL: int = 50
    GPU_CLEANUP_INTERVAL: int = 10


EVALUATION_DEFAULTS = EvaluationDefaults()


@dataclass(frozen=True)
class TaskProfile:
    """
    Profile describing task-specific evaluation behavior.

    Attributes:
        task_name: Internal identifier for the task
        class_names: Tuple of class names for the task
        num_classes: Number of classes
        display_name: Human-readable name for display (defaults to task_name)
    """

    task_name: str
    class_names: Tuple[str, ...]
    num_classes: int
    display_name: str = ""

    def __post_init__(self):
        if not self.display_name:
            object.__setattr__(self, "display_name", self.task_name)


class BaseEvaluator(VerificationMixin, ABC):
    """
    Base class for all task-specific evaluators.

    This class provides:
    - A unified evaluation loop (iterate samples → infer → accumulate → compute metrics)
    - Verification support via VerificationMixin
    - Common utilities (latency stats, model device management)

    Subclasses implement task-specific hooks:
    - _create_pipeline(): Create backend-specific pipeline
    - _prepare_input(): Prepare model input from sample
    - _parse_predictions(): Normalize pipeline output
    - _parse_ground_truths(): Extract ground truth from sample
    - _add_to_interface(): Feed a single frame to the metrics interface
    - _build_results(): Construct final results dict from interface metrics
    - print_results(): Format and display results
    """

    def __init__(
        self,
        metrics_interface: BaseMetricsInterface,
        task_profile: TaskProfile,
        model_cfg: Any,
    ):
        """
        Initialize evaluator.

        Args:
            metrics_interface: Metrics interface for computing task-specific metrics
            task_profile: Profile describing the task
            model_cfg: Model configuration (MMEngine Config or similar)
        """
        self.metrics_interface = metrics_interface
        self.task_profile = task_profile
        self.model_cfg = model_cfg
        self.pytorch_model: Any = None

    @property
    def class_names(self) -> Tuple[str, ...]:
        """Get class names from task profile."""
        return self.task_profile.class_names

    def set_pytorch_model(self, pytorch_model: Any) -> None:
        """Set PyTorch model (called by deployment runner)."""
        self.pytorch_model = pytorch_model

    def _ensure_model_on_device(self, device: str) -> Any:
        """Ensure PyTorch model is on the correct device."""
        if self.pytorch_model is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.pytorch_model is None. "
                "DeploymentRunner must set evaluator.pytorch_model before calling verify/evaluate."
            )

        current_device = next(self.pytorch_model.parameters()).device
        target_device = torch.device(device)

        if current_device != target_device:
            logger.info(f"Moving PyTorch model from {current_device} to {target_device}")
            self.pytorch_model = self.pytorch_model.to(target_device)

        return self.pytorch_model

    # ================== Abstract Methods (Task-Specific) ==================

    @abstractmethod
    def _create_pipeline(self, model_spec: ModelSpec, device: str) -> Any:
        """Create a pipeline for the specified backend."""
        raise NotImplementedError

    @abstractmethod
    def _prepare_input(
        self,
        sample: Mapping[str, Any],
        data_loader: BaseDataLoader,
        device: str,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Prepare model input from a sample. Returns (input_data, inference_kwargs)."""
        raise NotImplementedError

    @abstractmethod
    def _parse_predictions(self, pipeline_output: Any) -> Any:
        """Normalize pipeline output to standard format."""
        raise NotImplementedError

    @abstractmethod
    def _parse_ground_truths(self, gt_data: Mapping[str, Any]) -> Any:
        """Extract ground truth from sample data."""
        raise NotImplementedError

    @abstractmethod
    def _add_to_interface(self, predictions: Any, ground_truths: Any) -> None:
        """Add a single frame to the metrics interface."""
        raise NotImplementedError

    @abstractmethod
    def _build_results(
        self,
        latencies: List[float],
        latency_breakdowns: List[Dict[str, float]],
        num_samples: int,
    ) -> EvalResultDict:
        """Build final results dict from interface metrics."""
        raise NotImplementedError

    @abstractmethod
    def print_results(self, results: EvalResultDict) -> None:
        """Pretty print evaluation results."""
        raise NotImplementedError

    # ================== VerificationMixin Implementation ==================

    def _create_pipeline_for_verification(
        self,
        model_spec: ModelSpec,
        device: str,
        log: logging.Logger,
    ) -> Any:
        """Create pipeline for verification."""
        self._ensure_model_on_device(device)
        return self._create_pipeline(model_spec, device)

    def _get_verification_input(
        self,
        sample_idx: int,
        data_loader: BaseDataLoader,
        device: str,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Get verification input."""
        sample = data_loader.load_sample(sample_idx)
        return self._prepare_input(sample, data_loader, device)

    # ================== Core Evaluation Loop ==================

    def evaluate(
        self,
        model: ModelSpec,
        data_loader: BaseDataLoader,
        num_samples: int,
        verbose: bool = False,
    ) -> EvalResultDict:
        """
        Run evaluation on the specified model.

        Args:
            model: Model specification (backend/device/path)
            data_loader: Data loader for samples
            num_samples: Number of samples to evaluate
            verbose: Whether to print progress

        Returns:
            Evaluation results dictionary
        """
        logger.info(f"\nEvaluating {model.backend.value} model: {model.path}")
        logger.info(f"Number of samples: {num_samples}")

        self._ensure_model_on_device(model.device)
        pipeline = self._create_pipeline(model, model.device)
        self.metrics_interface.reset()

        latencies = []
        latency_breakdowns = []

        actual_samples = min(num_samples, data_loader.get_num_samples())

        for idx in range(actual_samples):
            if verbose and idx % EVALUATION_DEFAULTS.LOG_INTERVAL == 0:
                logger.info(f"Processing sample {idx + 1}/{actual_samples}")

            sample = data_loader.load_sample(idx)
            input_data, infer_kwargs = self._prepare_input(sample, data_loader, model.device)

            gt_data = data_loader.get_ground_truth(idx)
            ground_truths = self._parse_ground_truths(gt_data)

            infer_result = pipeline.infer(input_data, **infer_kwargs)
            latencies.append(infer_result.latency_ms)
            if infer_result.breakdown:
                latency_breakdowns.append(infer_result.breakdown)

            predictions = self._parse_predictions(infer_result.output)
            self._add_to_interface(predictions, ground_truths)

            if model.backend is Backend.TENSORRT and idx % EVALUATION_DEFAULTS.GPU_CLEANUP_INTERVAL == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Cleanup pipeline resources
        try:
            pipeline.cleanup()
        except Exception as e:
            logger.warning(f"Error during pipeline cleanup: {e}")

        return self._build_results(latencies, latency_breakdowns, actual_samples)

    # ================== Utilities ==================

    def compute_latency_stats(self, latencies: List[float]) -> LatencyStats:
        """Compute latency statistics from a list of measurements."""
        if not latencies:
            return LatencyStats.empty()

        arr = np.array(latencies)
        return LatencyStats(
            mean_ms=float(np.mean(arr)),
            std_ms=float(np.std(arr)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            median_ms=float(np.median(arr)),
        )

    def _compute_latency_breakdown(
        self,
        latency_breakdowns: List[Dict[str, float]],
    ) -> LatencyBreakdown:
        """Compute statistics for each latency stage."""
        if not latency_breakdowns:
            return LatencyBreakdown.empty()

        stage_order = list(dict.fromkeys(stage for breakdown in latency_breakdowns for stage in breakdown.keys()))

        return LatencyBreakdown(
            stages={
                stage: self.compute_latency_stats([bd[stage] for bd in latency_breakdowns if stage in bd])
                for stage in stage_order
            }
        )

    def format_latency_stats(self, stats: Union[Mapping[str, float], LatencyStats]) -> str:
        """Format latency statistics as a readable string."""
        stats_dict = stats.to_dict() if isinstance(stats, LatencyStats) else stats
        return (
            f"Latency: {stats_dict['mean_ms']:.2f} ± {stats_dict['std_ms']:.2f} ms "
            f"(min: {stats_dict['min_ms']:.2f}, max: {stats_dict['max_ms']:.2f}, "
            f"median: {stats_dict['median_ms']:.2f})"
        )
