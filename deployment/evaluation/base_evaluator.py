"""
Base evaluator for model evaluation in deployment.

All project evaluators should extend `BaseEvaluator` and implement the
required hooks for their specific task. The base class provides:

- A unified evaluation loop (iterate samples -> infer -> accumulate -> metrics)
- Common utilities (latency stats, model device management)

Module constants:

    LOG_INTERVAL
        Sample interval for verbose progress logs in `BaseEvaluator.evaluate`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping

import numpy as np
from mmengine.config import Config

from deployment.evaluation.backend_executor import BackendExecutor
from deployment.evaluation.evaluator_types import (
    EvalResultDict,
    LatencyBreakdown,
    LatencyStats,
    ModelSpec,
)
from deployment.inference.base_inference_pipeline import BaseInferencePipeline
from deployment.io.base_data_loader import BaseDataLoader
from deployment.metrics.base_metrics_interface import BaseMetricsInterface

logger = logging.getLogger(__name__)

# Verbose ``evaluate()`` logs every LOG_INTERVAL samples.
LOG_INTERVAL = 50


class BaseEvaluator(ABC):
    """
    Base class for all task-specific evaluators.

    Backend execution (pipeline creation, input preparation, device handling) is
    delegated to a `BackendExecutor`. Subclasses implement task-specific metrics hooks:

    - _parse_predictions: Convert pipeline output to the format the metrics interface expects
    - _parse_ground_truths: Extract ground truth from sample
    - _add_to_interface: Feed a single frame to the metrics interface
    - _build_results: Construct final results dict from interface metrics
    - print_results: Format and display results
    """

    def __init__(
        self,
        metrics_interface: BaseMetricsInterface,
        model_cfg: Config,
        executor: BackendExecutor,
    ) -> None:
        """Wire task metrics, model configuration, and backend executor into the evaluator.

        Args:
            metrics_interface: Task-specific metrics accumulator (reset per ``evaluate()`` run).
            model_cfg: MMEngine config for the model (class names, heads, etc.).
            executor: Backend execution primitives (pipeline / input / device handling),
                shared with `BackendVerifier`.
        """
        self.metrics_interface = metrics_interface
        self.model_cfg = model_cfg
        self._executor = executor

    # ================== Abstract Methods (Task-Specific) ==================

    @abstractmethod
    def _parse_predictions(self, pipeline_output: Any) -> Any:
        """Convert raw pipeline output into the format `_add_to_interface` expects."""
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

    def summarize_for_comparison(self, results: EvalResultDict) -> List[str]:
        """Return metric lines for the cross-backend comparison table.

        Args:
            results: A successful ``EvalResultDict``.

        Returns:
            Pre-formatted, indented log lines (may be empty).
        """
        lines: List[str] = []
        latency = results.get("latency")
        if latency is not None:
            lines.append(f"  Latency: {latency.mean_ms:.2f} ± {latency.std_ms:.2f} ms")
        return lines

    # ================== Core Evaluation Loop ==================

    def evaluate(
        self,
        model: ModelSpec,
        data_loader: BaseDataLoader,
        num_samples: int,
        verbose: bool = False,
        num_warmup: int = 0,
    ) -> EvalResultDict:
        """Run inference over samples and compute task metrics via ``metrics_interface``.

        Args:
            model: Backend, device, and artifact for the model under test.
            data_loader: Provides ``load_sample(i)`` with ``ground_truth`` for each sample.
            num_samples: Requested batch count (capped by ``data_loader.num_samples``).
            verbose: If True, log progress every :data:`LOG_INTERVAL` samples.
            num_warmup: Number of warm-up inferences to run before timing begins. These
                prime GPU/CUDA/TensorRT state and are excluded from latency and metrics,
                so they do not affect the ``num_samples`` totals.

        Returns:
            Task-specific evaluation dict from ``_build_results``.

        Raises:
            KeyError: If a loaded sample lacks ``\"ground_truth\"``.
        """
        logger.info("\nEvaluating %s model: %s", model.backend.value, model.artifact.path)
        logger.info("Number of samples: %s", num_samples)

        self._executor.ensure_model_on_device(model.device)
        pipeline = self._executor.create_pipeline(model, model.device)
        self.metrics_interface.reset()

        latencies: List[float] = []
        latency_breakdowns: List[Dict[str, float]] = []

        actual_samples = min(num_samples, data_loader.num_samples)

        try:
            self._run_warmup(pipeline, data_loader, model, num_warmup, verbose)

            for idx in range(actual_samples):
                if verbose and idx % LOG_INTERVAL == 0:
                    logger.info("Processing sample %s/%s", idx + 1, actual_samples)

                sample = data_loader.load_sample(idx)
                inference_input = self._executor.prepare_input(sample, data_loader, model.device)

                if "ground_truth" not in sample:
                    raise KeyError("DataLoader.load_sample() must return 'ground_truth' for evaluation.")
                ground_truths = self._parse_ground_truths(sample["ground_truth"])

                infer_result = pipeline.infer(inference_input.data, metadata=inference_input.metadata)
                latencies.append(infer_result.latency_ms)
                if infer_result.breakdown:
                    latency_breakdowns.append(infer_result.breakdown)

                predictions = self._parse_predictions(infer_result.output)
                self._add_to_interface(predictions, ground_truths)

                pipeline.periodic_cleanup(idx)
        finally:
            try:
                pipeline.cleanup()
            except Exception as e:
                logger.warning("Error during pipeline cleanup: %s", e)

        return self._build_results(latencies, latency_breakdowns, actual_samples)

    def _run_warmup(
        self,
        pipeline: BaseInferencePipeline,
        data_loader: BaseDataLoader,
        model: ModelSpec,
        num_warmup: int,
        verbose: bool,
    ) -> None:
        """Run throwaway inferences to prime GPU/CUDA/TensorRT state before timing.

        Reuses the first ``num_warmup`` samples (capped by dataset size). Outputs,
        latency, and metrics are intentionally discarded so warm-up does not affect the
        measured ``num_samples`` results.
        """
        warmup_count = min(num_warmup, data_loader.num_samples)
        if warmup_count <= 0:
            return

        if verbose:
            logger.info("Warming up on %s sample(s) (excluded from metrics/latency)...", warmup_count)

        for idx in range(warmup_count):
            sample = data_loader.load_sample(idx)
            inference_input = self._executor.prepare_input(sample, data_loader, model.device)
            pipeline.infer(inference_input.data, metadata=inference_input.metadata)

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
