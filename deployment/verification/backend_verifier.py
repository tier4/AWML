"""
Backend verification.

This module contains `BackendVerifier`, which drives the per-sample
verification loop for a reference/test `ModelSpec` pair: it runs both
backends, compares their outputs via `OutputComparator`, and owns all
verification logging.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional

import torch

from deployment.config.enums import Backend
from deployment.execution.backend_executor import BackendExecutor
from deployment.inference.base_inference_pipeline import BaseInferencePipeline
from deployment.io.base_data_loader import BaseDataLoader
from deployment.primitives.device import DeviceSpec
from deployment.primitives.evaluator_types import (
    ModelSpec,
    VerifyResultDict,
)
from deployment.verification.output_comparator import (
    OutputComparator,
    OutputDiffSummary,
    TensorDiffDetail,
)
from deployment.verification.reporting import banner, format_verdict

logger = logging.getLogger(__name__)


def _fmt_finite_diff(value: float) -> str:
    """Format a diff for logs: literal ``inf`` for infinities, else 6-decimal fixed-point."""
    return "inf" if math.isinf(value) else f"{value:.6f}"


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


class BackendVerifier:
    """Drive a reference vs test verification run over ``N`` samples.

    Args:
        executor: `BackendExecutor` providing pipeline creation, input
            preparation, and device handling for each side.
        comparator: Pure comparator used on each sample's raw outputs.
    """

    def __init__(self, executor: BackendExecutor, comparator: OutputComparator) -> None:
        self._executor = executor
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
            ref_device = self._executor.validate_device(reference.backend, reference.device)
            test_device = self._executor.validate_device(test.backend, test.device)
        except ValueError as exc:
            results["error"] = str(exc)
            return results

        self._log_header(reference, test, ref_device, test_device, num_samples, tolerance)

        actual_samples = min(num_samples, data_loader.num_samples)
        sample_results: List[SampleVerificationResult] = []
        ref_pipeline = None
        test_pipeline = None
        try:
            logger.info("\nInitializing %s reference pipeline...", reference.backend.value)
            self._executor.ensure_model_on_device(ref_device)
            ref_pipeline = self._executor.create_pipeline(reference, ref_device)

            logger.info("\nInitializing %s test pipeline...", test.backend.value)
            self._executor.ensure_model_on_device(test_device)
            test_pipeline = self._executor.create_pipeline(test, test_device)

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
        finally:
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

        Each side calls ``prepare_input`` with its own device so that tensors
        are created directly on the right device (no post-hoc ``.to(device)``
        shuffling).
        """
        executor = self._executor

        logger.info("\n%s", banner())
        logger.info("Verifying sample %s", sample_idx)
        logger.info("%s", banner())

        sample = data_loader.load_sample(sample_idx)

        executor.ensure_model_on_device(ref_device)
        ref_input = executor.prepare_input(sample, data_loader, ref_device)
        ref_label = f"{ref_backend.value} ({ref_device})"
        logger.info("Running %s reference...", ref_label)
        ref_result = ref_pipeline.infer(
            ref_input.data,
            metadata=ref_input.metadata,
            return_raw_outputs=True,
        )
        logger.info("  %s latency: %.2f ms", ref_label, ref_result.latency_ms)

        executor.ensure_model_on_device(test_device)
        test_input = executor.prepare_input(sample, data_loader, test_device)
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
        logger.info("  %s verification %s", test_label, format_verdict(summary.passed))

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
        logger.info("\n" + banner())
        logger.info("Model Verification")
        logger.info(banner())
        logger.info("Reference: %s on %s - %s", reference.backend.value, ref_device, reference.artifact.path)
        logger.info("Test: %s on %s - %s", test.backend.value, test_device, test.artifact.path)
        logger.info("Number of samples: %s", num_samples)
        logger.info("Tolerance: %s", tolerance)
        logger.info(banner())

    def _log_sample_result(self, result: SampleVerificationResult) -> None:
        """Log a single sample's pass/fail verdict plus max/mean diff (and reason on fail)."""
        log = logger.info if result.passed else logger.warning
        suffix = "" if result.passed else f" - {result.reason or 'no diagnostic'}"
        log(
            "  sample_%s %s (max_diff=%.6f, mean_diff=%.6f)%s",
            result.sample_idx,
            format_verdict(result.passed),
            result.max_diff,
            result.mean_diff,
            suffix,
        )

    def _log_summary(self, sample_results: List[SampleVerificationResult]) -> None:
        """Log per-sample verdicts then an aggregate pass/fail counter."""
        logger.info("\n" + banner())
        logger.info("Verification Summary")
        logger.info(banner())

        for r in sample_results:
            logger.info("  sample_%s: %s", r.sample_idx, format_verdict(r.passed))

        total = len(sample_results)
        passed = sum(1 for r in sample_results if r.passed)
        failed = total - passed
        logger.info(banner())
        logger.info("Total: %s/%s passed, %s/%s failed", passed, total, failed, total)
        logger.info(banner())
