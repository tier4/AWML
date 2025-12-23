"""CenterPoint deploy-only ONNX model definitions.

These modules exist to support ONNX export / ONNX-friendly execution graphs.
They are registered into MMEngine's `MODELS` registry via import side-effects
(`@MODELS.register_module()`).

Important:
- Call `register_models()` before building models that reference types like
  "CenterPointONNX", "CenterHeadONNX", "SeparateHeadONNX",
  "PillarFeatureNetONNX", "BackwardPillarFeatureNetONNX".
"""

from __future__ import annotations


def register_models() -> None:
    # Importing modules triggers `@MODELS.register_module()` registrations.
    from deployment.projects.centerpoint.onnx_models import centerpoint_head_onnx as _  # noqa: F401
    from deployment.projects.centerpoint.onnx_models import centerpoint_onnx as _  # noqa: F401
    from deployment.projects.centerpoint.onnx_models import pillar_encoder_onnx as _  # noqa: F401


__all__ = ["register_models"]
