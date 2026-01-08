"""
Verification mixin providing shared verification logic for evaluators.

This mixin extracts the common verification workflow that was duplicated
across CenterPointEvaluator, YOLOXOptElanEvaluator, and ClassificationEvaluator.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch

from deployment.core.backend import Backend
from deployment.core.evaluation.evaluator_types import ModelSpec, VerifyResultDict
from deployment.core.io.base_data_loader import BaseDataLoader


@dataclass(frozen=True)
class ComparisonResult:
    """Result of comparing two outputs (immutable)."""

    passed: bool
    max_diff: float
    mean_diff: float
    num_elements: int = 0
    details: Tuple[Tuple[str, ComparisonResult], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "passed": self.passed,
            "max_diff": self.max_diff,
            "mean_diff": self.mean_diff,
            "num_elements": self.num_elements,
        }
        if self.details:
            result["details"] = {k: v.to_dict() for k, v in self.details}
        return result


class VerificationMixin:
    """
    Mixin providing shared verification logic for all evaluators.

    Subclasses must implement:
    - _create_pipeline_for_verification(): Create backend-specific pipeline
    - _get_verification_input(): Extract inputs for verification

    Subclasses may optionally override:
    - _get_output_names(): Provide meaningful names for list/tuple outputs
    """

    @abstractmethod
    def _create_pipeline_for_verification(
        self,
        model_spec: ModelSpec,
        device: str,
        logger: logging.Logger,
    ) -> Any:
        """Create a pipeline for the specified backend."""
        raise NotImplementedError

    @abstractmethod
    def _get_verification_input(
        self,
        sample_idx: int,
        data_loader: BaseDataLoader,
        device: str,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Get input data for verification."""
        raise NotImplementedError

    def _get_output_names(self) -> Optional[List[str]]:
        """
        Optional: Provide meaningful names for list/tuple outputs.

        Override this method to provide task-specific output names for better logging.
        Returns None by default, which uses generic naming (output_0, output_1, ...).
        """
        return None

    def _compare_outputs(
        self,
        reference: Any,
        test: Any,
        tolerance: float,
        logger: logging.Logger,
        path: str = "output",
    ) -> ComparisonResult:
        """
        Recursively compare outputs of any structure.

        Handles:
        - Tensors (torch.Tensor, np.ndarray)
        - Scalars (int, float)
        - Dictionaries
        - Lists/Tuples
        - None values

        Args:
            reference: Reference output
            test: Test output
            tolerance: Maximum allowed difference
            logger: Logger instance
            path: Current path in the structure (for logging)

        Returns:
            ComparisonResult with comparison statistics
        """
        # Handle None
        if reference is None and test is None:
            return ComparisonResult(passed=True, max_diff=0.0, mean_diff=0.0)

        if reference is None or test is None:
            logger.error(f"  {path}: One output is None while the other is not")
            return ComparisonResult(passed=False, max_diff=float("inf"), mean_diff=float("inf"))

        # Handle dictionaries
        if isinstance(reference, dict) and isinstance(test, dict):
            return self._compare_dicts(reference, test, tolerance, logger, path)

        # Handle lists/tuples
        if isinstance(reference, (list, tuple)) and isinstance(test, (list, tuple)):
            return self._compare_sequences(reference, test, tolerance, logger, path)

        # Handle tensors and arrays
        if self._is_array_like(reference) and self._is_array_like(test):
            return self._compare_arrays(reference, test, tolerance, logger, path)

        # Handle scalars
        if isinstance(reference, (int, float)) and isinstance(test, (int, float)):
            diff = abs(float(reference) - float(test))
            passed = diff < tolerance
            if not passed:
                logger.warning(f"  {path}: scalar diff={diff:.6f} > tolerance={tolerance:.6f}")
            return ComparisonResult(passed=passed, max_diff=diff, mean_diff=diff, num_elements=1)

        # Type mismatch
        logger.error(f"  {path}: Type mismatch - {type(reference).__name__} vs {type(test).__name__}")
        return ComparisonResult(passed=False, max_diff=float("inf"), mean_diff=float("inf"))

    def _compare_dicts(
        self,
        reference: Mapping[str, Any],
        test: Mapping[str, Any],
        tolerance: float,
        logger: logging.Logger,
        path: str,
    ) -> ComparisonResult:
        """Compare dictionary outputs."""
        ref_keys = set(reference.keys())
        test_keys = set(test.keys())

        if ref_keys != test_keys:
            missing = ref_keys - test_keys
            extra = test_keys - ref_keys
            if missing:
                logger.error(f"  {path}: Missing keys in test: {missing}")
            if extra:
                logger.warning(f"  {path}: Extra keys in test: {extra}")
            return ComparisonResult(passed=False, max_diff=float("inf"), mean_diff=float("inf"))

        max_diff = 0.0
        total_diff = 0.0
        total_elements = 0
        all_passed = True
        details_list = []

        for key in sorted(ref_keys):
            child_path = f"{path}.{key}"
            result = self._compare_outputs(reference[key], test[key], tolerance, logger, child_path)
            details_list.append((key, result))

            max_diff = max(max_diff, result.max_diff)
            total_diff += result.mean_diff * result.num_elements
            total_elements += result.num_elements
            all_passed = all_passed and result.passed

        mean_diff = total_diff / total_elements if total_elements > 0 else 0.0
        return ComparisonResult(
            passed=all_passed,
            max_diff=max_diff,
            mean_diff=mean_diff,
            num_elements=total_elements,
            details=tuple(details_list),
        )

    def _compare_sequences(
        self,
        reference: Union[List, Tuple],
        test: Union[List, Tuple],
        tolerance: float,
        logger: logging.Logger,
        path: str,
    ) -> ComparisonResult:
        """Compare list/tuple outputs."""
        if len(reference) != len(test):
            logger.error(f"  {path}: Length mismatch - {len(reference)} vs {len(test)}")
            return ComparisonResult(passed=False, max_diff=float("inf"), mean_diff=float("inf"))

        # Get optional output names from subclass
        output_names = self._get_output_names()

        max_diff = 0.0
        total_diff = 0.0
        total_elements = 0
        all_passed = True
        details_list = []

        for idx, (ref_item, test_item) in enumerate(zip(reference, test)):
            # Use provided names or generic naming
            if output_names and idx < len(output_names):
                name = output_names[idx]
            else:
                name = f"output_{idx}"

            child_path = f"{path}[{name}]"
            result = self._compare_outputs(ref_item, test_item, tolerance, logger, child_path)
            details_list.append((name, result))

            max_diff = max(max_diff, result.max_diff)
            total_diff += result.mean_diff * result.num_elements
            total_elements += result.num_elements
            all_passed = all_passed and result.passed

        mean_diff = total_diff / total_elements if total_elements > 0 else 0.0
        return ComparisonResult(
            passed=all_passed,
            max_diff=max_diff,
            mean_diff=mean_diff,
            num_elements=total_elements,
            details=tuple(details_list),
        )

    def _compare_arrays(
        self,
        reference: Any,
        test: Any,
        tolerance: float,
        logger: logging.Logger,
        path: str,
    ) -> ComparisonResult:
        """Compare array-like outputs (tensors, numpy arrays)."""
        ref_np = self._to_numpy(reference)
        test_np = self._to_numpy(test)

        if ref_np.shape != test_np.shape:
            logger.error(f"  {path}: Shape mismatch - {ref_np.shape} vs {test_np.shape}")
            return ComparisonResult(passed=False, max_diff=float("inf"), mean_diff=float("inf"))

        diff = np.abs(ref_np - test_np)
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))
        num_elements = int(diff.size)

        passed = max_diff < tolerance
        logger.info(f"  {path}: shape={ref_np.shape}, max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        return ComparisonResult(
            passed=passed,
            max_diff=max_diff,
            mean_diff=mean_diff,
            num_elements=num_elements,
        )

    @staticmethod
    def _is_array_like(obj: Any) -> bool:
        """Check if object is array-like (tensor or numpy array)."""
        return isinstance(obj, (torch.Tensor, np.ndarray))

    @staticmethod
    def _to_numpy(tensor: Any) -> np.ndarray:
        """Convert tensor to numpy array."""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        if isinstance(tensor, np.ndarray):
            return tensor
        return np.array(tensor)

    def _compare_backend_outputs(
        self,
        reference_output: Any,
        test_output: Any,
        tolerance: float,
        backend_name: str,
        logger: logging.Logger,
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Compare outputs from reference and test backends.

        This is the main entry point for output comparison.
        Uses recursive comparison to handle any output structure.
        """
        result = self._compare_outputs(reference_output, test_output, tolerance, logger)

        logger.info(f"\n  Overall Max difference: {result.max_diff:.6f}")
        logger.info(f"  Overall Mean difference: {result.mean_diff:.6f}")

        if result.passed:
            logger.info(f"  {backend_name} verification PASSED ✓")
        else:
            logger.warning(
                f"  {backend_name} verification FAILED ✗ "
                f"(max diff: {result.max_diff:.6f} > tolerance: {tolerance:.6f})"
            )

        return result.passed, {"max_diff": result.max_diff, "mean_diff": result.mean_diff}

    def _normalize_verification_device(
        self,
        backend: Backend,
        device: str,
        logger: logging.Logger,
    ) -> Optional[str]:
        """Normalize device for verification based on backend requirements."""
        if backend is Backend.PYTORCH and device.startswith("cuda"):
            logger.warning("PyTorch verification is forced to CPU; overriding device to 'cpu'")
            return "cpu"

        if backend is Backend.TENSORRT:
            if not device.startswith("cuda"):
                return None
            if device != "cuda:0":
                logger.warning("TensorRT verification only supports 'cuda:0'. Overriding.")
                return "cuda:0"

        return device

    def verify(
        self,
        reference: ModelSpec,
        test: ModelSpec,
        data_loader: BaseDataLoader,
        num_samples: int = 1,
        tolerance: float = 0.1,
        verbose: bool = False,
    ) -> VerifyResultDict:
        """Verify exported models using policy-based verification."""
        logger = logging.getLogger(__name__)

        results: VerifyResultDict = {
            "summary": {"passed": 0, "failed": 0, "total": 0},
            "samples": {},
        }

        ref_device = self._normalize_verification_device(reference.backend, reference.device, logger)
        test_device = self._normalize_verification_device(test.backend, test.device, logger)

        if test_device is None:
            results["error"] = f"{test.backend.value} requires CUDA"
            return results

        self._log_verification_header(reference, test, ref_device, test_device, num_samples, tolerance, logger)

        logger.info(f"\nInitializing {reference.backend.value} reference pipeline...")
        ref_pipeline = self._create_pipeline_for_verification(reference, ref_device, logger)

        logger.info(f"\nInitializing {test.backend.value} test pipeline...")
        test_pipeline = self._create_pipeline_for_verification(test, test_device, logger)

        actual_samples = min(num_samples, data_loader.get_num_samples())
        for i in range(actual_samples):
            logger.info(f"\n{'='*60}")
            logger.info(f"Verifying sample {i}")
            logger.info(f"{'='*60}")

            passed = self._verify_single_sample(
                i,
                ref_pipeline,
                test_pipeline,
                data_loader,
                ref_device,
                test_device,
                reference.backend,
                test.backend,
                tolerance,
                logger,
            )
            results["samples"][f"sample_{i}"] = passed

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Cleanup pipeline resources - all pipelines now have cleanup() via base class
        for pipeline in [ref_pipeline, test_pipeline]:
            if pipeline is not None:
                try:
                    pipeline.cleanup()
                except Exception as e:
                    logger.warning(f"Error during pipeline cleanup in verification: {e}")

        sample_values = results["samples"].values()
        passed_count = sum(1 for v in sample_values if v is True)
        failed_count = sum(1 for v in sample_values if v is False)

        results["summary"] = {"passed": passed_count, "failed": failed_count, "total": len(results["samples"])}
        self._log_verification_summary(results, logger)

        return results

    def _verify_single_sample(
        self,
        sample_idx: int,
        ref_pipeline: Any,
        test_pipeline: Any,
        data_loader: BaseDataLoader,
        ref_device: str,
        test_device: str,
        ref_backend: Backend,
        test_backend: Backend,
        tolerance: float,
        logger: logging.Logger,
    ) -> bool:
        """Verify a single sample."""
        input_data, infer_kwargs = self._get_verification_input(sample_idx, data_loader, ref_device)

        ref_name = f"{ref_backend.value} ({ref_device})"
        logger.info(f"\nRunning {ref_name} reference...")
        ref_result = ref_pipeline.infer(input_data, return_raw_outputs=True, **infer_kwargs)
        logger.info(f"  {ref_name} latency: {ref_result.latency_ms:.2f} ms")

        test_input = self._move_input_to_device(input_data, test_device)
        test_name = f"{test_backend.value} ({test_device})"
        logger.info(f"\nRunning {test_name} test...")
        test_result = test_pipeline.infer(test_input, return_raw_outputs=True, **infer_kwargs)
        logger.info(f"  {test_name} latency: {test_result.latency_ms:.2f} ms")

        passed, _ = self._compare_backend_outputs(ref_result.output, test_result.output, tolerance, test_name, logger)
        return passed

    def _move_input_to_device(self, input_data: Any, device: str) -> Any:
        """Move input data to specified device."""
        device_obj = torch.device(device)

        if isinstance(input_data, torch.Tensor):
            return input_data.to(device_obj) if input_data.device != device_obj else input_data
        if isinstance(input_data, dict):
            return {k: self._move_input_to_device(v, device) for k, v in input_data.items()}
        if isinstance(input_data, (list, tuple)):
            return type(input_data)(self._move_input_to_device(item, device) for item in input_data)
        return input_data

    def _log_verification_header(
        self,
        reference: ModelSpec,
        test: ModelSpec,
        ref_device: str,
        test_device: str,
        num_samples: int,
        tolerance: float,
        logger: logging.Logger,
    ) -> None:
        """Log verification header information."""
        logger.info("\n" + "=" * 60)
        logger.info("Model Verification (Policy-Based)")
        logger.info("=" * 60)
        logger.info(f"Reference: {reference.backend.value} on {ref_device} - {reference.path}")
        logger.info(f"Test: {test.backend.value} on {test_device} - {test.path}")
        logger.info(f"Number of samples: {num_samples}")
        logger.info(f"Tolerance: {tolerance}")
        logger.info("=" * 60)

    def _log_verification_summary(self, results: VerifyResultDict, logger: logging.Logger) -> None:
        """Log verification summary."""
        logger.info("\n" + "=" * 60)
        logger.info("Verification Summary")
        logger.info("=" * 60)

        for key, value in results["samples"].items():
            status = "PASSED" if value else "FAILED"
            logger.info(f"  {key}: {status}")

        summary = results["summary"]
        logger.info("=" * 60)
        logger.info(
            f"Total: {summary['passed']}/{summary['total']} passed, {summary['failed']}/{summary['total']} failed"
        )
        logger.info("=" * 60)
