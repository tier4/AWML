"""CenterPoint ONNX-compatible model definitions.

This module contains model variants that support ONNX export:

- ``CenterPointONNX``: ONNX-compatible detector with feature extraction helpers.
- ``CenterHeadONNX``: ONNX-compatible detection head with stable output ordering.
- ``SeparateHeadONNX``: ONNX-compatible separate head with deterministic ordering.
- ``PillarFeatureNetONNX``: ONNX-compatible pillar feature network.
- ``BackwardPillarFeatureNetONNX``: Backward-compatible pillar feature network.

**Note**: These are model *definitions* for ONNX export, not exported model artifacts.
They are registered into MMEngine's ``MODELS`` registry via ``@MODELS.register_module()``.

Usage:
    Call ``register_models()`` before building models that reference types like
    "CenterPointONNX", "CenterHeadONNX", etc.

Example:
    >>> from deployment.projects.centerpoint.onnx_models import register_models
    >>> register_models()  # Register ONNX model variants
    >>> # Now you can build models with type="CenterPointONNX" in config
"""

from __future__ import annotations


def register_models() -> None:
    """Register CenterPoint ONNX model variants into MMEngine's MODELS registry.

    Importing the submodules triggers ``@MODELS.register_module()`` decorators,
    which registers the types referenced by config strings (e.g., "CenterPointONNX").

    This function should be called before ``MODELS.build()`` for configs that
    use ONNX model variants.
    """
    # Import triggers @MODELS.register_module() registrations
    from deployment.projects.centerpoint.onnx_models import centerpoint_head_onnx as _head  # noqa: F401
    from deployment.projects.centerpoint.onnx_models import centerpoint_onnx as _model  # noqa: F401
    from deployment.projects.centerpoint.onnx_models import pillar_encoder_onnx as _encoder  # noqa: F401


__all__ = ["register_models"]
