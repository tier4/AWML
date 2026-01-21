"""
Type definitions for model evaluation in deployment.

This module contains the shared type definitions used by evaluators,
runners, and orchestrators.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, TypedDict

from deployment.core.artifacts import Artifact
from deployment.core.backend import Backend


class EvalResultDict(TypedDict, total=False):
    """
    Structured evaluation result used across deployments.

    Attributes:
        primary_metric: Main scalar metric for quick ranking (e.g., accuracy, mAP).
        metrics: Flat dictionary of additional scalar metrics.
        per_class: Optional nested metrics keyed by class/label name.
        latency: Latency statistics as returned by compute_latency_stats().
        metadata: Arbitrary metadata that downstream components might need.
    """

    primary_metric: float
    metrics: Dict[str, float]
    per_class: Dict[str, Any]
    latency: Dict[str, float]
    metadata: Dict[str, Any]


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
    def empty(cls) -> "LatencyStats":
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
    def empty(cls) -> "LatencyBreakdown":
        """Return an empty breakdown."""
        return cls(stages={})

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Convert to ``Dict[str, Dict[str, float]]`` for downstream use."""
        return {stage: stats.to_dict() for stage, stats in self.stages.items()}


@dataclass(frozen=True)
class InferenceResult:
    """Standard inference return payload."""

    output: Any
    latency_ms: float
    breakdown: Optional[Dict[str, float]] = None

    @classmethod
    def empty(cls) -> "InferenceResult":
        """Return an empty inference result."""
        return cls(output=None, latency_ms=0.0, breakdown={})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary for logging/serialization."""
        return {
            "output": self.output,
            "latency_ms": self.latency_ms,
            "breakdown": dict(self.breakdown or {}),
        }


@dataclass(frozen=True)
class ModelSpec:
    """
    Minimal description of a concrete model artifact to evaluate or verify.

    Attributes:
        backend: Backend identifier such as 'pytorch', 'onnx', or 'tensorrt'.
        device: Target device string (e.g., 'cpu', 'cuda:0').
        artifact: Filesystem representation of the produced model.
    """

    backend: Backend
    device: str
    artifact: Artifact

    @property
    def path(self) -> str:
        """Backward-compatible access to artifact path."""
        return self.artifact.path

    @property
    def multi_file(self) -> bool:
        """True if the artifact represents a multi-file bundle."""
        return self.artifact.multi_file
