"""Runtime device descriptor used across deployment backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

import torch


@dataclass(frozen=True)
class DeviceSpec:
    """Validated runtime device representation.

    This type is intentionally runtime-focused (single concrete device), unlike
    config-level `DeviceConfig` which stores defaults and aliases.
    """

    kind: Literal["cpu", "cuda"]
    index: int = 0

    def __post_init__(self) -> None:
        if self.kind == "cpu":
            object.__setattr__(self, "index", 0)
            return

        if self.kind != "cuda":
            raise ValueError(f"Unsupported device kind '{self.kind}'.")
        if self.index < 0:
            raise ValueError("CUDA device index must be non-negative.")

    @classmethod
    def from_value(cls, value: Union[str, torch.device, "DeviceSpec"]) -> "DeviceSpec":
        """Normalize strings/torch.device/DeviceSpec into DeviceSpec."""
        if isinstance(value, cls):
            return value

        if isinstance(value, torch.device):
            if value.type == "cuda":
                return cls(kind="cuda", index=0 if value.index is None else int(value.index))
            if value.type == "cpu":
                return cls(kind="cpu", index=0)
            raise ValueError(f"Unsupported torch device type '{value.type}'.")

        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized == "cpu":
                return cls(kind="cpu", index=0)
            if normalized == "cuda":
                return cls(kind="cuda", index=0)
            if normalized.startswith("cuda:"):
                suffix = normalized.split(":", 1)[1].strip()
                if not suffix.isdigit():
                    raise ValueError(f"Invalid CUDA device index in '{value}'.")
                return cls(kind="cuda", index=int(suffix))

        raise TypeError(f"Unsupported device value type: {type(value)}")

    @property
    def is_cuda(self) -> bool:
        return self.kind == "cuda"

    def to_torch_device(self) -> torch.device:
        """Return torch.device equivalent."""
        return torch.device(str(self))

    def to_ort_provider(self) -> list[str]:
        """Return ONNX Runtime execution providers for this device."""
        if self.is_cuda:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def to_trt_device_str(self) -> str:
        """Return TensorRT-compatible CUDA device string."""
        if not self.is_cuda:
            raise ValueError("TensorRT requires CUDA device.")
        return str(self)

    def __str__(self) -> str:
        if self.is_cuda:
            return f"cuda:{self.index}"
        return "cpu"
