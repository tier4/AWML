"""
Pure enums and constants for deployment config.

No dependency on torch or mmengine. Safe to import from exporters, evaluators, CLI.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Union

# Constants
DEFAULT_WORKSPACE_SIZE = 1 << 30  # 1 GB


class PrecisionPolicy(str, Enum):
    """Precision policy options for TensorRT.

    The concrete TensorRT flags each policy maps to are applied by the TensorRT
    exporter (see ``TensorRTExporter._apply_precision_policy``); this enum is the
    single source of truth for the policy itself.
    """

    AUTO = "auto"
    FP16 = "fp16"
    FP32_TF32 = "fp32_tf32"
    STRONGLY_TYPED = "strongly_typed"

    @classmethod
    def from_value(cls, value: Optional[Union[str, PrecisionPolicy]]) -> PrecisionPolicy:
        """Parse strings or enum members into PrecisionPolicy (defaults to AUTO)."""
        if value is None:
            return cls.AUTO
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            for member in cls:
                if member.value == normalized:
                    return member
        raise ValueError(f"Invalid precision_policy '{value}'. Must be one of {[m.value for m in cls]}.")


class Backend(str, Enum):
    """Supported deployment backends."""

    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"

    @classmethod
    def from_value(cls, value: Union[str, Backend]) -> Backend:
        """
        Normalize backend identifiers coming from configs or enums.

        Args:
            value: Backend as string or Backend enum

        Returns:
            Backend enum instance

        Raises:
            ValueError: If value cannot be mapped to a supported backend
        """
        if isinstance(value, cls):
            return value

        if isinstance(value, str):
            normalized = value.strip().lower()
            try:
                return cls(normalized)
            except ValueError as exc:
                raise ValueError(f"Unsupported backend '{value}'. Expected one of {[b.value for b in cls]}.") from exc

        raise TypeError(f"Backend must be a string or Backend enum, got {type(value)}")

    @property
    def requires_cuda(self) -> bool:
        """Whether this backend can only run on a CUDA device.

        Single source of truth for the runtime constraint enforced by config validation, evaluation, and verification.
        """
        return self is Backend.TENSORRT

    def __str__(self) -> str:  # pragma: no cover - convenience for logging
        return self.value


class ExportMode(str, Enum):
    """Export pipeline modes."""

    ONNX = "onnx"
    TRT = "trt"
    BOTH = "both"
    NONE = "none"

    @classmethod
    def from_value(cls, value: Optional[Union[str, ExportMode]]) -> ExportMode:
        """Parse strings or enum members into ExportMode (defaults to BOTH)."""
        if value is None:
            return cls.BOTH
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            for member in cls:
                if member.value == normalized:
                    return member
        raise ValueError(f"Invalid export mode '{value}'. Must be one of {[m.value for m in cls]}.")
