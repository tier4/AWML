"""Backend enum used across deployment configs and runtime components."""

from __future__ import annotations

from enum import Enum
from typing import Union


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

    def __str__(self) -> str:  # pragma: no cover - convenience for logging
        return self.value
