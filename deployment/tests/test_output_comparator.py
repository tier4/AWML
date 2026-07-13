"""Unit tests for OutputComparator, focused on the shape-mismatch path."""

from __future__ import annotations

import numpy as np

from deployment.verification.output_comparator import OutputComparator


class TestOutputComparator:
    def test_identical_arrays_pass(self):
        a = [np.zeros((2, 3), dtype=np.float32)]
        b = [np.zeros((2, 3), dtype=np.float32)]
        summary, details = OutputComparator().compare(a, b, tolerance=1e-6)
        assert summary.passed
        assert summary.max_diff == 0.0
        assert len(details) == 1

    def test_within_tolerance_passes(self):
        a = [np.zeros((4,), dtype=np.float32)]
        b = [np.full((4,), 0.05, dtype=np.float32)]
        summary, _ = OutputComparator().compare(a, b, tolerance=0.1)
        assert summary.passed
        assert summary.max_diff <= 0.1

    def test_exceeds_tolerance_fails(self):
        a = [np.zeros((4,), dtype=np.float32)]
        b = [np.full((4,), 1.0, dtype=np.float32)]
        summary, _ = OutputComparator().compare(a, b, tolerance=0.1)
        assert not summary.passed
        assert "tolerance" in (summary.reason or "")

    def test_shape_mismatch_fails_with_reason(self):
        a = [np.zeros((2, 3), dtype=np.float32)]
        b = [np.zeros((2, 4), dtype=np.float32)]
        summary, details = OutputComparator().compare(a, b, tolerance=1.0)
        assert not summary.passed
        assert "shape mismatch" in (summary.reason or "")
        # The mismatched tensor is recorded with infinite diffs, not silently dropped.
        assert details and details[0].max_diff == float("inf")

    def test_length_mismatch_fails(self):
        a = [np.zeros((2,), dtype=np.float32), np.zeros((2,), dtype=np.float32)]
        b = [np.zeros((2,), dtype=np.float32)]
        summary, _ = OutputComparator().compare(a, b, tolerance=1.0)
        assert not summary.passed
        assert "length mismatch" in (summary.reason or "")

    def test_named_outputs_label_paths(self):
        a = [np.zeros((2,), dtype=np.float32)]
        b = [np.ones((2,), dtype=np.float32)]
        summary, details = OutputComparator(output_names=["heatmap"]).compare(a, b, tolerance=0.1)
        assert not summary.passed
        assert "heatmap" in details[0].path
