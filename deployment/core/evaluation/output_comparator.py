"""
Pure output comparison for model verification.

This module contains `OutputComparator`, a stateless recursive comparator
for structured model outputs. It handles dicts, lists/tuples, tensors, arrays
and scalars under an absolute tolerance and returns a `ComparisonResult`.

Design notes:
    - No logging is performed here; callers (e.g. ``VerificationRunner``) are
      responsible for rendering results.
    - The comparator is intentionally minimal: no task-specific logic, no
      strategy/plugin pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch


@dataclass(frozen=True)
class ComparisonResult:
    """Outcome of recursively comparing reference and test structured outputs.

    Attributes:
        passed: True if all nested comparisons satisfy ``tolerance``.
        max_diff: Largest absolute difference seen in any numeric leaf.
        mean_diff: Absolute difference mean weighted by ``num_elements``.
        num_elements: Element count used for weighted mean aggregation.
        reason: Short description of the first discovered mismatch (or ``None`` on pass).
    """

    passed: bool
    max_diff: float
    mean_diff: float
    num_elements: int = 0
    reason: Optional[str] = None


class OutputComparator:
    """Recursively compare structured outputs within an absolute tolerance.

    The comparator is stateless aside from optional ``output_names`` used to
    label positions when comparing list/tuple outputs (for clearer diagnostic
    paths in ``reason``).

    Args:
        output_names: Optional names aligned with sequence output indices.
            When provided, sequence children are labelled ``output[name]``
            instead of the default ``output[output_i]``.
    """

    def __init__(self, output_names: Optional[Sequence[str]] = None) -> None:
        self._output_names: Optional[Tuple[str, ...]] = tuple(output_names) if output_names else None

    def compare(
        self,
        reference: Any,
        test: Any,
        tolerance: float,
        path: str = "output",
    ) -> ComparisonResult:
        """Recursively compare two structured outputs under ``tolerance``.

        Args:
            reference: Expected structure from the reference backend.
            test: Structure from the backend under test.
            tolerance: Maximum absolute difference for numeric leaves.
            path: Dot/bracket path used to describe the first mismatch.

        Returns:
            Aggregated `ComparisonResult` for this subtree.
        """
        if reference is None and test is None:
            return ComparisonResult(passed=True, max_diff=0.0, mean_diff=0.0)

        if reference is None or test is None:
            return _fail(path, "one side is None while the other is not")

        if isinstance(reference, dict) and isinstance(test, dict):
            return self._compare_dicts(reference, test, tolerance, path)

        if isinstance(reference, (list, tuple)) and isinstance(test, (list, tuple)):
            return self._compare_sequences(reference, test, tolerance, path)

        if self._is_array_like(reference) and self._is_array_like(test):
            return self._compare_arrays(reference, test, tolerance, path)

        if isinstance(reference, (int, float)) and isinstance(test, (int, float)):
            diff = abs(float(reference) - float(test))
            passed = diff < tolerance
            reason = None if passed else f"{path}: scalar diff={diff:.6f} > tolerance={tolerance:.6f}"
            return ComparisonResult(passed=passed, max_diff=diff, mean_diff=diff, num_elements=1, reason=reason)

        return _fail(
            path,
            f"type mismatch {type(reference).__name__} vs {type(test).__name__}",
        )

    def _compare_dicts(
        self,
        reference: Mapping[str, Any],
        test: Mapping[str, Any],
        tolerance: float,
        path: str,
    ) -> ComparisonResult:
        """Compare dict outputs key-by-key (sorted keys for determinism)."""
        ref_keys = set(reference.keys())
        test_keys = set(test.keys())

        if ref_keys != test_keys:
            missing = ref_keys - test_keys
            extra = test_keys - ref_keys
            parts: List[str] = []
            if missing:
                parts.append(f"missing {sorted(missing)}")
            if extra:
                parts.append(f"extra {sorted(extra)}")
            return _fail(path, "key mismatch: " + ", ".join(parts))

        return self._merge_comparison_results(
            self.compare(reference[k], test[k], tolerance, f"{path}.{k}") for k in sorted(ref_keys)
        )

    def _compare_sequences(
        self,
        reference: Union[List, Tuple],
        test: Union[List, Tuple],
        tolerance: float,
        path: str,
    ) -> ComparisonResult:
        """Compare list/tuple outputs element-wise using ``output_names`` when provided."""
        if len(reference) != len(test):
            return _fail(path, f"length mismatch {len(reference)} vs {len(test)}")

        names = self._output_names

        def _child_results():
            for idx, (ref_item, test_item) in enumerate(zip(reference, test)):
                name = names[idx] if names and idx < len(names) else f"output_{idx}"
                yield self.compare(ref_item, test_item, tolerance, f"{path}[{name}]")

        return self._merge_comparison_results(_child_results())

    def _compare_arrays(
        self,
        reference: Any,
        test: Any,
        tolerance: float,
        path: str,
    ) -> ComparisonResult:
        """Compare array-like leaves after converting to NumPy (same shape required)."""
        ref_np = self._to_numpy(reference)
        test_np = self._to_numpy(test)

        if ref_np.shape != test_np.shape:
            return _fail(path, f"shape mismatch {ref_np.shape} vs {test_np.shape}")

        diff = np.abs(ref_np - test_np)
        max_diff = float(np.max(diff)) if diff.size else 0.0
        mean_diff = float(np.mean(diff)) if diff.size else 0.0
        num_elements = int(diff.size)

        passed = max_diff < tolerance
        reason = (
            None if passed else f"{path}: max_diff={max_diff:.6f} > tolerance={tolerance:.6f} (shape={ref_np.shape})"
        )
        return ComparisonResult(
            passed=passed,
            max_diff=max_diff,
            mean_diff=mean_diff,
            num_elements=num_elements,
            reason=reason,
        )

    @staticmethod
    def _merge_comparison_results(results) -> ComparisonResult:
        """Merge several subtree `ComparisonResult` values into one.

        ``max_diff`` is the max across subtrees; ``mean_diff`` is weighted by
        ``num_elements``; ``reason`` is the first failing subtree's reason.
        """
        max_diff = 0.0
        total_diff = 0.0
        total_elements = 0
        all_passed = True
        first_reason: Optional[str] = None

        for result in results:
            max_diff = max(max_diff, result.max_diff)
            total_diff += result.mean_diff * result.num_elements
            total_elements += result.num_elements
            if not result.passed and all_passed:
                all_passed = False
                first_reason = result.reason

        mean_diff = total_diff / total_elements if total_elements > 0 else 0.0
        return ComparisonResult(
            passed=all_passed,
            max_diff=max_diff,
            mean_diff=mean_diff,
            num_elements=total_elements,
            reason=first_reason,
        )

    @staticmethod
    def _is_array_like(obj: Any) -> bool:
        """Return True when ``obj`` is a tensor or ndarray (leaf comparison path)."""
        return isinstance(obj, (torch.Tensor, np.ndarray))

    @staticmethod
    def _to_numpy(tensor: Any) -> np.ndarray:
        """Convert tensors to CPU NumPy arrays; pass through ``ndarray``."""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        if isinstance(tensor, np.ndarray):
            return tensor
        return np.array(tensor)


def _fail(path: str, reason: str) -> ComparisonResult:
    """Build a failing `ComparisonResult` with an inf diff and a short reason."""
    return ComparisonResult(
        passed=False,
        max_diff=float("inf"),
        mean_diff=float("inf"),
        reason=f"{path}: {reason}",
    )
