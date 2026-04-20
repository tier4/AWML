"""
Base evaluator for model evaluation in deployment.

All project evaluators should extend `BaseEvaluator` and implement the
required hooks for their specific task. The base class provides:

- A unified evaluation loop (iterate samples -> infer -> accumulate -> metrics)
- A `verify()` entry point that delegates to `VerificationRunner`
- Common utilities (latency stats, model device management)

Module constants:

    LOG_INTERVAL
        Sample interval for verbose progress logs in `BaseEvaluator.evaluate`.
    GPU_CLEANUP_INTERVAL
        Sample interval for optional GPU cache clears when running TensorRT during `BaseEvaluator.evaluate`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
from mmengine.config import Config

from deployment.core.backend import Backend
from deployment.core.device import DeviceSpec
from deployment.core.evaluation.evaluator_types import (
    EvalResultDict,
    InferenceInput,
    LatencyBreakdown,
    LatencyStats,
    ModelSpec,
    VerifyResultDict,
)
from deployment.core.evaluation.output_comparator import OutputComparator
from deployment.core.evaluation.verification_runner import VerificationRunner
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.core.metrics.base_metrics_interface import BaseMetricsInterface
from deployment.pipelines.base_pipeline import BaseInferencePipeline

logger = logging.getLogger(__name__)

# Verbose ``evaluate()`` logs every LOG_INTERVAL samples; TensorRT empty_cache every GPU_CLEANUP_INTERVAL.
LOG_INTERVAL = 50
GPU_CLEANUP_INTERVAL = 10


class BaseEvaluator(ABC):
    """
    Base class for all task-specific evaluators.

    Subclasses implement task-specific hooks:

    - _create_pipeline: Create backend-specific pipeline
    - _prepare_input: Prepare model input from sample
    - _parse_predictions: Normalize pipeline output
    - _parse_ground_truths: Extract ground truth from sample
    - _add_to_interface: Feed a single frame to the metrics interface
    - _build_results: Construct final results dict from interface metrics
    - print_results: Format and display results

    Subclasses may optionally override:

    - `_get_output_names`: Provide meaningful names for list/tuple outputs
      during verification comparison.
    """

    def __init__(
        self,
        metrics_interface: BaseMetricsInterface,
        model_cfg: Config,
    ) -> None:
        """Wire task metrics and model configuration into the evaluator.

        Args:
            metrics_interface: Task-specific metrics accumulator (reset per ``evaluate()`` run).
            model_cfg: MMEngine config for the model (class names, heads, etc.).
        """
        self.metrics_interface = metrics_interface
        self.model_cfg = model_cfg
        self.pytorch_model: Any = None
        self.export_model_cfg: Optional[Config] = None

    def set_pytorch_model(self, pytorch_model: Any) -> None:
        """Attach the loaded PyTorch module used when building ONNX/TRT or for reference runs."""
        self.pytorch_model = pytorch_model

    def set_export_model_cfg(self, export_model_cfg: Config) -> None:
        """Attach the MMEngine config that matches the loaded export model (``model.cfg`` after load)."""
        self.export_model_cfg = export_model_cfg

    def _ensure_model_on_device(self, device: DeviceSpec) -> Any:
        """Ensure ``pytorch_model`` lives on ``device`` (used before infer / pipeline creation)."""
        if self.pytorch_model is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.pytorch_model is None. "
                "DeploymentRunner must set evaluator.pytorch_model before calling verify/evaluate."
            )

        current_device = next(self.pytorch_model.parameters()).device
        target_device = device.to_torch_device()

        if current_device != target_device:
            logger.info("Moving PyTorch model from %s to %s", current_device, target_device)
            self.pytorch_model = self.pytorch_model.to(target_device)

        return self.pytorch_model

    def _normalize_verification_device(self, backend: Backend, device: DeviceSpec) -> DeviceSpec:
        """Enforce backend runtime constraints on a concrete DeviceSpec."""
        if backend is Backend.TENSORRT and not device.is_cuda:
            raise ValueError(f"TensorRT verification requires CUDA, got '{device}'.")
        return device

    # ================== Abstract Methods (Task-Specific) ==================

    @abstractmethod
    def _create_pipeline(self, model_spec: ModelSpec, device: DeviceSpec) -> BaseInferencePipeline:
        """Create an inference pipeline for ``model_spec.backend`` on ``device``.

        Args:
            model_spec: Backend, device, and artifact path for the deployment model.
            device: Concrete device for this run.

        Returns:
            A ``BaseInferencePipeline`` subclass exposing ``infer()`` and ``cleanup()``.
        """
        raise NotImplementedError

    @abstractmethod
    def _prepare_input(
        self,
        sample: Mapping[str, Any],
        data_loader: BaseDataLoader,
        device: DeviceSpec,
    ) -> InferenceInput:
        """Prepare model input from a sample on the given device.

        Verification calls this once per side (reference and test) with each
        backend's own device, so implementations should create tensors directly
        on ``device`` rather than relying on downstream moves.

        Args:
            sample: Sample data from the data loader.
            data_loader: Data loader to load the sample from.
            device: Device to prepare the input on.

        Returns:
            InferenceInput containing the actual input data and metadata.
        """
        raise NotImplementedError

    @abstractmethod
    def _parse_predictions(self, pipeline_output: Any) -> Any:
        """Normalize raw pipeline output into the shape expected by `_add_to_interface`."""
        raise NotImplementedError

    @abstractmethod
    def _parse_ground_truths(self, gt_data: Mapping[str, Any]) -> Any:
        """Parse `sample["ground_truth"]` into ground-truth structures for metrics."""
        raise NotImplementedError

    @abstractmethod
    def _add_to_interface(self, predictions: Any, ground_truths: Any) -> None:
        """Feed one sample's predictions and labels into ``metrics_interface``."""
        raise NotImplementedError

    @abstractmethod
    def _build_results(
        self,
        latencies: List[float],
        latency_breakdowns: List[Dict[str, float]],
        num_samples: int,
    ) -> EvalResultDict:
        """Aggregate metrics and latencies into the final `EvalResultDict`."""
        raise NotImplementedError

    @abstractmethod
    def print_results(self, results: EvalResultDict) -> None:
        """Render ``results`` for human-readable logs (prefer ``logging``, not ``print``)."""
        raise NotImplementedError

    def _get_output_names(self) -> Optional[List[str]]:
        """Optional names for list/tuple raw outputs during verification logging.

        Override in subclasses when the pipeline returns a sequence of tensors with
        known semantic names (e.g. detection heads). The names are forwarded to the
        `~deployment.core.evaluation.output_comparator.OutputComparator` to
        label positions in diagnostic paths.

        Returns:
            Names aligned with output index order, or `None` to fall back to
            `output_0`, `output_1`, ...
        """
        return None

    # ================== Core Evaluation Loop ==================

    def evaluate(
        self,
        model: ModelSpec,
        data_loader: BaseDataLoader,
        num_samples: int,
        verbose: bool = False,
    ) -> EvalResultDict:
        """Run inference over samples and compute task metrics via ``metrics_interface``.

        Args:
            model: Backend, device, and artifact for the model under test.
            data_loader: Provides ``load_sample(i)`` with ``ground_truth`` for each sample.
            num_samples: Requested batch count (capped by ``data_loader.num_samples``).
            verbose: If True, log progress every :data:`LOG_INTERVAL` samples.

        Returns:
            Task-specific evaluation dict from ``_build_results``.

        Raises:
            KeyError: If a loaded sample lacks ``\"ground_truth\"``.
        """
        logger.info("\nEvaluating %s model: %s", model.backend.value, model.artifact.path)
        logger.info("Number of samples: %s", num_samples)

        self._ensure_model_on_device(model.device)
        pipeline = self._create_pipeline(model, model.device)
        self.metrics_interface.reset()

        latencies: List[float] = []
        latency_breakdowns: List[Dict[str, float]] = []

        actual_samples = min(num_samples, data_loader.num_samples)

        for idx in range(actual_samples):
            if verbose and idx % LOG_INTERVAL == 0:
                logger.info("Processing sample %s/%s", idx + 1, actual_samples)

            sample = data_loader.load_sample(idx)
            inference_input = self._prepare_input(sample, data_loader, model.device)

            if "ground_truth" not in sample:
                raise KeyError("DataLoader.load_sample() must return 'ground_truth' for evaluation.")
            ground_truths = self._parse_ground_truths(sample["ground_truth"])

            infer_result = pipeline.infer(inference_input.data, metadata=inference_input.metadata)
            latencies.append(infer_result.latency_ms)
            if infer_result.breakdown:
                latency_breakdowns.append(infer_result.breakdown)

            predictions = self._parse_predictions(infer_result.output)
            self._add_to_interface(predictions, ground_truths)

            if model.backend is Backend.TENSORRT and idx % GPU_CLEANUP_INTERVAL == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        try:
            pipeline.cleanup()
        except Exception as e:
            logger.warning("Error during pipeline cleanup: %s", e)

        return self._build_results(latencies, latency_breakdowns, actual_samples)

    # ================== Verification ==================

    def verify(
        self,
        reference: ModelSpec,
        test: ModelSpec,
        data_loader: BaseDataLoader,
        num_samples: int = 1,
        tolerance: float = 0.1,
    ) -> VerifyResultDict:
        """Compare raw outputs of two `ModelSpec` backends.

        Thin wrapper: assembles an `OutputComparator` and a `VerificationRunner`,
        then delegates the actual work.

        Args:
            reference: Reference backend (typically PyTorch).
            test: Backend under verification.
            data_loader: Verification samples (same schema as evaluation).
            num_samples: Requested sample count (capped by loader length).
            tolerance: Maximum allowed absolute per-element difference.

        Returns:
            `VerifyResultDict` from the runner.
        """
        comparator = OutputComparator(output_names=self._get_output_names())
        runner = VerificationRunner(self, comparator)
        return runner.run(reference, test, data_loader, num_samples, tolerance)

    # ================== Utilities ==================

    def compute_latency_stats(self, latencies: List[float]) -> LatencyStats:
        """Compute mean, std, min, max, median over per-sample latencies (milliseconds).

        Args:
            latencies: Per-inference `latency_ms` values (empty list yields zeros via `LatencyStats.empty()`).

        Returns:
            Immutable `LatencyStats`.
        """
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
        """Aggregate per-sample stage timings into a `LatencyBreakdown`.

        Args:
            latency_breakdowns: One dict per sample, keys are stage names, values are ms.

        Returns:
            Per-stage `LatencyStats` keyed by stage name.
        """
        if not latency_breakdowns:
            return LatencyBreakdown.empty()

        stage_order = list(dict.fromkeys(stage for bd in latency_breakdowns for stage in bd.keys()))

        return LatencyBreakdown(
            stages={
                stage: self.compute_latency_stats([bd[stage] for bd in latency_breakdowns if stage in bd])
                for stage in stage_order
            }
        )
