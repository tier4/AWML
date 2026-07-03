"""CenterPoint-specific export context."""

from __future__ import annotations

from dataclasses import dataclass

from deployment.export.contexts import ExportContext


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
