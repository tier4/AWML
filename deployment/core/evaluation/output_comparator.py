"""
Pure output comparison for model verification.

This module contains `OutputComparator`, a stateless recursive comparator
for structured model outputs. Deployment verification expects pipeline raw
outputs that are **sequences (list/tuple) of tensors/arrays** and/or tensor
leaves; dict and bare scalar outputs are not handled (they fail with a type
mismatch).

Naming:
    - **OutputDiffSummary**: one object for the **whole output** — whether it
      passed, overall max/mean diff (aggregated), and first failure reason.
    - **TensorDiffDetail**: one row per **tensor** in the structure — path,
      shape, and that tensor's max/mean diff (for per-head logging).

Design notes:
    - No logging here; callers (e.g. ``VerificationRunner``) render logs.
    - :meth:`OutputComparator.compare` returns ``(OutputDiffSummary, list of
      TensorDiffDetail)`` in a single traversal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch


@dataclass(frozen=True)
class OutputDiffSummary:
    """Rolled-up comparison for an entire structured output (or subtree).

    Use this for pass/fail and **global** max/mean diff. For each tensor's
    own stats, see :class:`TensorDiffDetail`.

    Attributes:
        passed: True if the full structure is within ``tolerance``.
        max_diff: Largest per-tensor max diff anywhere in the tree.
        mean_diff: Element-weighted mean of absolute differences over the tree.
        num_elements: Total tensor elements compared (for weighted mean).
        reason: First failing tensor's message, or ``None`` if passed.
    """

    passed: bool
    max_diff: float
    mean_diff: float
    num_elements: int = 0
    reason: Optional[str] = None


@dataclass(frozen=True)
class TensorDiffDetail:
    """Stats for **one tensor** at a path (used for verbose per-output logs).

    Attributes:
        path: Dot/bracket path (e.g. ``output[heatmap]``).
        shape: NumPy shape of this tensor.
        max_diff: Max absolute difference on this tensor.
        mean_diff: Mean absolute difference on this tensor.
        passed: Whether this tensor alone satisfies ``tolerance``.
    """

    path: str
    shape: Tuple[int, ...]
    max_diff: float
    mean_diff: float
    passed: bool


class OutputComparator:
    """Recursively compare structured outputs within an absolute tolerance.

    Optional ``output_names`` label sequence slots (e.g. head names) in paths.

    Args:
        output_names: Names aligned with sequence indices; children become
            ``output[name]`` instead of ``output_0``, ``output_1``, ...
    """

    def __init__(self, output_names: Optional[Sequence[str]] = None) -> None:
        self._output_names: Optional[Tuple[str, ...]] = tuple(output_names) if output_names else None

    def compare(
        self,
        reference: Any,
        test: Any,
        tolerance: float,
        path: str = "output",
    ) -> Tuple[OutputDiffSummary, List[TensorDiffDetail]]:
        """Compare two structured outputs; collect per-tensor rows and a summary."""
        tensor_details: List[TensorDiffDetail] = []
        summary = self._compare_nested(reference, test, tolerance, path, tensor_details)
        return summary, tensor_details

    def _compare_nested(
        self,
        reference: Any,
        test: Any,
        tolerance: float,
        path: str,
        tensor_details: List[TensorDiffDetail],
    ) -> OutputDiffSummary:
        """Recursive compare; appends one :class:`TensorDiffDetail` per tensor leaf."""
        if reference is None and test is None:
            return OutputDiffSummary(passed=True, max_diff=0.0, mean_diff=0.0)

        if reference is None or test is None:
            return _fail(path, "one side is None while the other is not")

        if isinstance(reference, (list, tuple)) and isinstance(test, (list, tuple)):
            return self._compare_sequences(reference, test, tolerance, path, tensor_details)

        if self._is_array_like(reference) and self._is_array_like(test):
            return self._compare_arrays(reference, test, tolerance, path, tensor_details)

        return _fail(
            path,
            f"type mismatch {type(reference).__name__} vs {type(test).__name__}",
        )

    def _compare_sequences(
        self,
        reference: Union[List, Tuple],
        test: Union[List, Tuple],
        tolerance: float,
        path: str,
        tensor_details: List[TensorDiffDetail],
    ) -> OutputDiffSummary:
        """Compare list/tuple outputs element-wise using ``output_names`` when provided."""
        if len(reference) != len(test):
            return _fail(path, f"length mismatch {len(reference)} vs {len(test)}")

        names = self._output_names

        def _child_summaries():
            for idx, (ref_item, test_item) in enumerate(zip(reference, test)):
                name = names[idx] if names and idx < len(names) else f"output_{idx}"
                yield self._compare_nested(ref_item, test_item, tolerance, f"{path}[{name}]", tensor_details)

        return self._merge_summaries(_child_summaries())

    def _compare_arrays(
        self,
        reference: Any,
        test: Any,
        tolerance: float,
        path: str,
        tensor_details: List[TensorDiffDetail],
    ) -> OutputDiffSummary:
        """Compare tensor/ndarray leaves (same shape required)."""
        ref_np = self._to_numpy(reference)
        test_np = self._to_numpy(test)

        if ref_np.shape != test_np.shape:
            tensor_details.append(
                TensorDiffDetail(
                    path=path,
                    shape=tuple(int(x) for x in ref_np.shape),
                    max_diff=float("inf"),
                    mean_diff=float("inf"),
                    passed=False,
                )
            )
            return _fail(path, f"shape mismatch {ref_np.shape} vs {test_np.shape}")

        diff = np.abs(ref_np - test_np)
        max_diff = float(np.max(diff)) if diff.size else 0.0
        mean_diff = float(np.mean(diff)) if diff.size else 0.0
        num_elements = int(diff.size)

        passed = max_diff < tolerance
        reason = (
            None if passed else f"{path}: max_diff={max_diff:.6f} > tolerance={tolerance:.6f} (shape={ref_np.shape})"
        )
        tensor_details.append(
            TensorDiffDetail(
                path=path,
                shape=tuple(int(x) for x in ref_np.shape),
                max_diff=max_diff,
                mean_diff=mean_diff,
                passed=passed,
            )
        )
        return OutputDiffSummary(
            passed=passed,
            max_diff=max_diff,
            mean_diff=mean_diff,
            num_elements=num_elements,
            reason=reason,
        )

    @staticmethod
    def _merge_summaries(results) -> OutputDiffSummary:
        """Combine child :class:`OutputDiffSummary` values into one rollup."""
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
        return OutputDiffSummary(
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


def _fail(path: str, reason: str) -> OutputDiffSummary:
    """Build a failing summary with infinite diffs and a short reason."""
    return OutputDiffSummary(
        passed=False,
        max_diff=float("inf"),
        mean_diff=float("inf"),
        reason=f"{path}: {reason}",
    )
