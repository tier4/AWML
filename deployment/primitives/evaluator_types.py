"""
Type definitions for model evaluation in deployment.

This module contains the shared type definitions used by evaluators,
runners, and orchestrators.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, TypedDict

from deployment.config.enums import Backend
from deployment.primitives.artifacts import Artifact
from deployment.primitives.device import DeviceSpec


class EvalResultDict(TypedDict, total=False):
    """
    Structured evaluation result produced by ``BaseEvaluator._build_results``.

    Every key is optional (``total=False``): each task surfaces only the subset
    relevant to it (detection emits ``mAP_*``; classification emits ``accuracy``).

    Attributes:
        mAP_by_mode: Detection mAP keyed by evaluation mode/distance bucket.
        mAPH_by_mode: Detection mAPH (heading-aware) keyed by mode.
        per_class_ap_by_mode: Per-class AP nested by mode.
        accuracy: Top-line scalar for classification tasks.
        detailed_metrics: Raw task-specific metric payload for deep inspection.
        latency: End-to-end latency statistics from ``compute_latency_stats``.
        latency_breakdown: Per-stage latency statistics (optional).
        num_samples: Number of samples actually evaluated.
        error: Set instead of metrics when evaluation failed for this backend.
    """

    mAP_by_mode: Dict[str, float]
    mAPH_by_mode: Dict[str, float]
    per_class_ap_by_mode: Dict[str, Any]
    accuracy: float
    detailed_metrics: Dict[str, Any]
    latency: "LatencyStats"
    latency_breakdown: "LatencyBreakdown"
    num_samples: int
    error: str


class VerifyResultDict(TypedDict, total=False):
    """
    Structured verification outcome shared between runners and evaluators.

    Attributes:
        summary: Aggregate pass/fail counts.
        samples: Mapping of sample identifiers to boolean pass/fail states.
    """

    summary: Dict[str, int]
    samples: Dict[str, bool]
    error: str


@dataclass(frozen=True)
class LatencyStats:
    """
    Immutable latency statistics for a batch of inferences.

    Provides a typed alternative to loose dictionaries and a convenient
    ``to_dict`` helper for interoperability with existing call sites.
    """

    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    median_ms: float

    @classmethod
    def empty(cls) -> LatencyStats:
        """Return a zero-initialized stats object."""
        return cls(0.0, 0.0, 0.0, 0.0, 0.0)

    def to_dict(self) -> Dict[str, float]:
        """Convert to a plain dictionary for serialization."""
        return asdict(self)


@dataclass(frozen=True)
class LatencyBreakdown:
    """
    Stage-wise latency statistics keyed by stage name.

    Stored as a mapping of stage -> LatencyStats, with a ``to_dict`` helper
    to preserve backward compatibility with existing dictionary consumers.
    """

    stages: Dict[str, LatencyStats]

    @classmethod
    def empty(cls) -> LatencyBreakdown:
        """Return an empty breakdown."""
        return cls(stages={})

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Convert to ``Dict[str, Dict[str, float]]`` for downstream use."""
        return {stage: stats.to_dict() for stage, stats in self.stages.items()}


@dataclass(frozen=True)
class InferenceInput:
    """Prepared input for pipeline inference.

    Attributes:
        data: The actual input data (e.g., points tensor, image tensor).
        metadata: Sample metadata forwarded to postprocess().
    """

    data: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceResult:
    """Standard inference return payload."""

    output: Any
    latency_ms: float
    breakdown: Optional[Dict[str, float]] = None


@dataclass(frozen=True)
class ModelSpec:
    """
    Minimal description of a concrete model artifact to evaluate or verify.

    Attributes:
        backend: Backend identifier such as 'pytorch', 'onnx', or 'tensorrt'.
        device: Target runtime device.
        artifact: Filesystem representation of the produced model.
    """

    backend: Backend
    device: DeviceSpec
    artifact: Artifact
