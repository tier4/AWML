"""Artifact descriptors for deployment outputs."""

from __future__ import annotations

import os.path as osp
from dataclasses import dataclass


@dataclass(frozen=True)
class Artifact:
    """Represents a produced deployment artifact such as ONNX or TensorRT outputs."""

    path: str
    multi_file: bool = False

    def exists(self) -> bool:
        """Return True if the artifact path currently exists on disk."""
        return osp.exists(self.path)
