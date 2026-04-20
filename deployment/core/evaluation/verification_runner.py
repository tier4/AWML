"""
Verification orchestration.

This module contains `VerificationRunner`, which coordinates the
per-sample verification loop for a reference/test `ModelSpec` pair.

Division of responsibilities (intentionally minimal):

- `VerificationHooks` is the narrow structural contract the runner
  depends on: pipeline creation, input preparation, and device handling.
  `~deployment.core.evaluation.base_evaluator.BaseEvaluator` satisfies
  this protocol implicitly, but the runner does not depend on it directly.
- `~deployment.core.evaluation.output_comparator.OutputComparator`
  performs pure structural comparison of raw outputs.
- `VerificationRunner` wires both together, handles pipeline lifecycle,
  and owns all verification logging.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Protocol

import torch

from deployment.core.backend import Backend
from deployment.core.device import DeviceSpec
from deployment.core.evaluation.evaluator_types import (
    InferenceInput,
    ModelSpec,
    VerifyResultDict,
)
from deployment.core.evaluation.output_comparator import (
    OutputComparator,
    OutputDiffSummary,
    TensorDiffDetail,
)
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.pipelines.base_pipeline import BaseInferencePipeline

logger = logging.getLogger(__name__)


def _fmt_finite_diff(value: float) -> str:
    """Format a diff for logs; ``inf`` is spelled ``inf`` (not ``inf`` via ``%f`` quirks)."""
    return "inf" if math.isinf(value) else f"{value:.6f}"


class VerificationHooks(Protocol):
    """Minimal structural contract required by `VerificationRunner`.

    Any object exposing these four hooks can drive a verification run.
    `~deployment.core.evaluation.base_evaluator.BaseEvaluator` satisfies
    this protocol implicitly; the runner avoids depending on the full evaluator
    surface to keep responsibilities cleanly separated.
    """

    def _normalize_verification_device(self, backend: Backend, device: DeviceSpec) -> DeviceSpec:
        """Enforce backend runtime constraints and return the resolved device."""
        ...

    def _ensure_model_on_device(self, device: DeviceSpec) -> Any:
        """Ensure the reference PyTorch model lives on ``device`` before use."""
        ...

    def _create_pipeline(self, model_spec: ModelSpec, device: DeviceSpec) -> BaseInferencePipeline:
        """Create an inference pipeline for ``model_spec.backend`` on ``device``."""
        ...

    def _prepare_input(
        self,
        sample: Mapping[str, Any],
        data_loader: BaseDataLoader,
        device: DeviceSpec,
    ) -> InferenceInput:
        """Build an `InferenceInput` for ``sample`` on ``device``."""
        ...


@dataclass(frozen=True)
class SampleVerificationResult:
    """Result of verifying a single sample.

    Attributes:
        sample_idx: Index used with ``data_loader.load_sample``.
        passed: Whether reference and test outputs match within tolerance.
        max_diff: Maximum absolute difference observed.
        mean_diff: Mean absolute difference weighted by element count.
        reason: First discovered mismatch description (``None`` when passed).
    """

    sample_idx: int
    passed: bool
    max_diff: float
    mean_diff: float
    reason: Optional[str] = None


class VerificationRunner:
    """Drive a reference vs test verification run over ``N`` samples.

    Args:
        hooks: Object implementing `VerificationHooks` (pipeline / input /
            device hooks). Typically a `BaseEvaluator`, but the runner
            only depends on the narrower protocol.
        comparator: Pure comparator used on each sample's raw outputs.
    """

    def __init__(self, hooks: VerificationHooks, comparator: OutputComparator) -> None:
        self._hooks = hooks
        self._comparator = comparator

    def run(
        self,
        reference: ModelSpec,
        test: ModelSpec,
        data_loader: BaseDataLoader,
        num_samples: int,
        tolerance: float,
    ) -> VerifyResultDict:
        """Run verification for `min(num_samples, data_loader.num_samples)` samples.

        Args:
            reference: Reference backend model specification.
            test: Backend-under-test specification.
            data_loader: Same loader used for evaluation.
            num_samples: Requested sample count (capped by loader length).
            tolerance: Per-element absolute tolerance for numeric comparison.

        Returns:
            `VerifyResultDict` with ``summary`` + per-sample pass map. ``error``
            is set when device normalization fails before any inference runs.
        """
        results: VerifyResultDict = {
            "summary": {"passed": 0, "failed": 0, "total": 0},
            "samples": {},
        }

        try:
            ref_device = self._hooks._normalize_verification_device(reference.backend, reference.device)
            test_device = self._hooks._normalize_verification_device(test.backend, test.device)
        except ValueError as exc:
            results["error"] = str(exc)
            return results

        self._log_header(reference, test, ref_device, test_device, num_samples, tolerance)

        logger.info("\nInitializing %s reference pipeline...", reference.backend.value)
        self._hooks._ensure_model_on_device(ref_device)
        ref_pipeline = self._hooks._create_pipeline(reference, ref_device)

        logger.info("\nInitializing %s test pipeline...", test.backend.value)
        self._hooks._ensure_model_on_device(test_device)
        test_pipeline = self._hooks._create_pipeline(test, test_device)

        actual_samples = min(num_samples, data_loader.num_samples)
        sample_results: List[SampleVerificationResult] = []
        for i in range(actual_samples):
            sr = self._run_single_sample(
                i,
                ref_pipeline,
                test_pipeline,
                data_loader,
                ref_device,
                test_device,
                reference.backend,
                test.backend,
                tolerance,
            )
            sample_results.append(sr)
            results["samples"][f"sample_{i}"] = sr.passed
            self._log_sample_result(sr)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for pipeline in (ref_pipeline, test_pipeline):
            if pipeline is None:
                continue
            try:
                pipeline.cleanup()
            except Exception as e:
                logger.warning("Error during pipeline cleanup in verification: %s", e)

        passed_count = sum(1 for r in sample_results if r.passed)
        failed_count = sum(1 for r in sample_results if not r.passed)
        results["summary"] = {
            "passed": passed_count,
            "failed": failed_count,
            "total": len(sample_results),
        }

        self._log_summary(sample_results)
        return results

    def _run_single_sample(
        self,
        sample_idx: int,
        ref_pipeline: BaseInferencePipeline,
        test_pipeline: BaseInferencePipeline,
        data_loader: BaseDataLoader,
        ref_device: DeviceSpec,
        test_device: DeviceSpec,
        ref_backend: Backend,
        test_backend: Backend,
        tolerance: float,
    ) -> SampleVerificationResult:
        """Run both pipelines on one sample and compare their raw outputs.

        Each side calls ``_prepare_input`` with its own device so that tensors
        are created directly on the right device (no post-hoc ``.to(device)``
        shuffling).
        """
        hooks = self._hooks

        logger.info("\n%s", "=" * 60)
        logger.info("Verifying sample %s", sample_idx)
        logger.info("%s", "=" * 60)

        sample = data_loader.load_sample(sample_idx)

        hooks._ensure_model_on_device(ref_device)
        ref_input = hooks._prepare_input(sample, data_loader, ref_device)
        ref_label = f"{ref_backend.value} ({ref_device})"
        logger.info("Running %s reference...", ref_label)
        ref_result = ref_pipeline.infer(
            ref_input.data,
            metadata=ref_input.metadata,
            return_raw_outputs=True,
        )
        logger.info("  %s latency: %.2f ms", ref_label, ref_result.latency_ms)

        hooks._ensure_model_on_device(test_device)
        test_input = hooks._prepare_input(sample, data_loader, test_device)
        test_label = f"{test_backend.value} ({test_device})"
        logger.info("Running %s test...", test_label)
        test_result = test_pipeline.infer(
            test_input.data,
            metadata=test_input.metadata,
            return_raw_outputs=True,
        )
        logger.info("  %s latency: %.2f ms", test_label, test_result.latency_ms)

        summary, per_tensor = self._comparator.compare(ref_result.output, test_result.output, tolerance)
        self._log_per_output_comparison(test_label, per_tensor, summary)

        return SampleVerificationResult(
            sample_idx=sample_idx,
            passed=summary.passed,
            max_diff=summary.max_diff,
            mean_diff=summary.mean_diff,
            reason=summary.reason,
        )

    def _log_per_output_comparison(
        self,
        test_label: str,
        per_tensor: List[TensorDiffDetail],
        summary: OutputDiffSummary,
    ) -> None:
        """Emit one line per tensor, then overall max/mean, then a verification line."""
        logger.info("")
        for d in per_tensor:
            logger.info(
                "  %s: shape=%s, max_diff=%s, mean_diff=%s",
                d.path,
                d.shape,
                _fmt_finite_diff(d.max_diff),
                _fmt_finite_diff(d.mean_diff),
            )
        logger.info("  Overall Max difference: %s", _fmt_finite_diff(summary.max_diff))
        logger.info("  Overall Mean difference: %s", _fmt_finite_diff(summary.mean_diff))
        verdict = "PASSED ✓" if summary.passed else "FAILED ✗"
        logger.info("  %s verification %s", test_label, verdict)

    def _log_header(
        self,
        reference: ModelSpec,
        test: ModelSpec,
        ref_device: DeviceSpec,
        test_device: DeviceSpec,
        num_samples: int,
        tolerance: float,
    ) -> None:
        """Emit a banner with models, devices, sample count and tolerance."""
        logger.info("\n" + "=" * 60)
        logger.info("Model Verification")
        logger.info("=" * 60)
        logger.info("Reference: %s on %s - %s", reference.backend.value, ref_device, reference.artifact.path)
        logger.info("Test: %s on %s - %s", test.backend.value, test_device, test.artifact.path)
        logger.info("Number of samples: %s", num_samples)
        logger.info("Tolerance: %s", tolerance)
        logger.info("=" * 60)

    def _log_sample_result(self, result: SampleVerificationResult) -> None:
        """Log a single sample's pass/fail verdict plus max/mean diff (and reason on fail)."""
        if result.passed:
            logger.info(
                "  sample_%s PASSED ✓ (max_diff=%.6f, mean_diff=%.6f)",
                result.sample_idx,
                result.max_diff,
                result.mean_diff,
            )
        else:
            logger.warning(
                "  sample_%s FAILED ✗ (max_diff=%.6f, mean_diff=%.6f) - %s",
                result.sample_idx,
                result.max_diff,
                result.mean_diff,
                result.reason or "no diagnostic",
            )

    def _log_summary(self, sample_results: List[SampleVerificationResult]) -> None:
        """Log per-sample verdicts then an aggregate pass/fail counter."""
        logger.info("\n" + "=" * 60)
        logger.info("Verification Summary")
        logger.info("=" * 60)

        for r in sample_results:
            status = "PASSED" if r.passed else "FAILED"
            logger.info("  sample_%s: %s", r.sample_idx, status)

        total = len(sample_results)
        passed = sum(1 for r in sample_results if r.passed)
        failed = total - passed
        logger.info("=" * 60)
        logger.info("Total: %s/%s passed, %s/%s failed", passed, total, failed, total)
        logger.info("=" * 60)
