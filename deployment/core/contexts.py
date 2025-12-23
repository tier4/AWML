"""
Typed context objects for deployment workflows.

This module defines typed dataclasses that replace **kwargs with explicit,
type-checked parameters. This improves:
- Type safety: Catches mismatches at type-check time
- Discoverability: IDE autocomplete shows available parameters
- Refactoring safety: Renamed fields are caught by type checkers

Design Principles:
    1. Base contexts define common parameters across all projects
    2. Project-specific contexts extend base with additional fields
    3. Optional fields have sensible defaults
    4. Contexts are immutable (frozen=True) for safety

Usage:
    # Create context for export
    ctx = ExportContext(sample_idx=0)

    # Project-specific context
    ctx = CenterPointExportContext(rot_y_axis_reference=True)

    # Pass to orchestrator
    result = export_orchestrator.run(ctx)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class ExportContext:
    """
    Base context for export operations.

    This context carries parameters needed during the export workflow,
    including model loading and ONNX/TensorRT export settings.

    Attributes:
        sample_idx: Index of sample to use for tracing/shape inference (default: 0)
        extra: Dictionary for project-specific or debug-only options that don't
               warrant a dedicated field. Use sparingly.
    """

    sample_idx: int = 0
    extra: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from extra dict with a default."""
        return self.extra.get(key, default)


@dataclass(frozen=True)
class YOLOXExportContext(ExportContext):
    """
    YOLOX-specific export context.

    Attributes:
        model_cfg_path: Path to model configuration file. If None, attempts
                        to extract from model_cfg.filename.
    """

    model_cfg: Optional[str] = None


@dataclass(frozen=True)
class CenterPointExportContext(ExportContext):
    """
    CenterPoint-specific export context.

    Attributes:
        rot_y_axis_reference: Whether to use y-axis rotation reference for
                              ONNX-compatible output format. This affects
                              how rotation and dimensions are encoded.
    """

    rot_y_axis_reference: bool = False


@dataclass(frozen=True)
class CalibrationExportContext(ExportContext):
    """
    Calibration model export context.

    Currently uses only base ExportContext fields.
    Extend with calibration-specific parameters as needed.
    """

    pass
