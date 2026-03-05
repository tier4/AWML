"""CenterPoint ONNX-compatible model definitions.

This module contains model variants that support ONNX export:

- ``CenterPointONNX``: ONNX-compatible detector with feature extraction helpers.
- ``CenterHeadONNX``: ONNX-compatible detection head with stable output ordering.
- ``SeparateHeadONNX``: ONNX-compatible separate head with deterministic ordering.
- ``PillarFeatureNetONNX``: ONNX-compatible pillar feature network.
- ``BackwardPillarFeatureNetONNX``: Backward-compatible pillar feature network.

**Note**: These are model *definitions* for ONNX export, not exported model artifacts.
Importing this package (or its submodules) triggers ``@MODELS.register_module()``,
registering types for config strings (e.g., "CenterPointONNX", "CenterHeadONNX").

Usage:
    Import this package before building models that reference ONNX types:

    >>> from deployment.projects.centerpoint import onnx_models  # noqa: F401
    >>> # Now you can build models with type="CenterPointONNX" in config
"""

from __future__ import annotations

from deployment.projects.centerpoint.onnx_models import centerpoint_head_onnx, centerpoint_onnx, pillar_encoder_onnx

__all__ = [
    "centerpoint_head_onnx",
    "centerpoint_onnx",
    "pillar_encoder_onnx",
]
