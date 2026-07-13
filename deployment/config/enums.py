"""
Pure enums and constants for deployment config.

No dependency on torch or mmengine. Safe to import from exporters, evaluators, CLI.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Type, TypeVar, Union

# Constants
DEFAULT_WORKSPACE_SIZE = 1 << 30  # 1 GB

_E = TypeVar("_E", bound=Enum)


def _enum_from_value(
    enum_cls: Type[_E],
    value: object,
    *,
    default: Optional[_E] = None,
    label: Optional[str] = None,
) -> _E:
    """Normalize a string or enum member into ``enum_cls`` (shared by the config enums).

    Matching is case-insensitive on the member ``value``. ``None`` returns ``default`` when
    one is given (for optional config sections) and is otherwise an error, so every enum
    parses identically instead of each hand-rolling its own ``from_value``.

    Raises:
        ValueError: If ``value`` is ``None`` without a default, or is an unknown string.
        TypeError: If ``value`` is neither ``None``, a ``str``, nor an ``enum_cls`` member.
    """
    label = label or enum_cls.__name__
    valid = [member.value for member in enum_cls]
    if value is None:
        if default is not None:
            return default
        raise ValueError(f"{label} is required; must be one of {valid}.")
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        for member in enum_cls:
            if member.value == normalized:
                return member
        raise ValueError(f"Invalid {label} '{value}'. Must be one of {valid}.")
    raise TypeError(f"{label} must be a string or {enum_cls.__name__}, got {type(value).__name__}.")


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
        return _enum_from_value(cls, value, default=cls.AUTO, label="precision_policy")

    def __str__(self) -> str:  # pragma: no cover - convenience for logging
        return self.value


class Backend(str, Enum):
    """Supported deployment backends."""

    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"

    @classmethod
    def from_value(cls, value: Union[str, Backend]) -> Backend:
        """Normalize a backend identifier (string or enum) into a ``Backend`` member."""
        return _enum_from_value(cls, value, label="backend")

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
        return _enum_from_value(cls, value, default=cls.BOTH, label="export mode")

    def __str__(self) -> str:  # pragma: no cover - convenience for logging
        return self.value
