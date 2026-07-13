"""
CenterPoint model loading utilities.

This module provides ONNX-compatible model building from MMEngine configs.
"""

from __future__ import annotations

import copy
import logging

import torch
from mmengine.config import Config

from deployment.io.mmdet3d_model import build_mmdet3d_model
from deployment.primitives.device import DeviceSpec

# Imported for their side effect: registering CenterPoint's ONNX module variants into the
# MMDet3D registries so ``MODELS.build`` can resolve them during export.
from deployment.projects.centerpoint.export.onnx_models import (  # noqa: F401
    centerpoint_head_onnx,
    centerpoint_onnx,
    pillar_encoder_onnx,
)

logger = logging.getLogger(__name__)


def create_onnx_model_cfg(
    model_cfg: Config,
    device: DeviceSpec,
    rot_y_axis_reference: bool = False,
) -> Config:
    """Create a model config that swaps modules to ONNX-friendly variants.

    This mutates the `model_cfg.model` subtree to reference classes registered by
    `deployment.projects.centerpoint.export.onnx_models` (e.g., `CenterPointONNX`).

    Args:
        model_cfg: Original MMEngine model configuration.
        device: Target device specification.
        rot_y_axis_reference: Whether to use y-axis rotation reference.

    Returns:
        New config whose ``model`` subtree builds the deployment export graph (e.g. ONNX-friendly types).
    """
    export_model_cfg = model_cfg.copy()
    model_config = copy.deepcopy(export_model_cfg.model)

    model_config.type = "CenterPointONNX"
    model_config.point_channels = model_config.pts_voxel_encoder.in_channels
    model_config.device = device

    if model_config.pts_voxel_encoder.type == "PillarFeatureNet":
        model_config.pts_voxel_encoder.type = "PillarFeatureNetONNX"
    elif model_config.pts_voxel_encoder.type == "BackwardPillarFeatureNet":
        model_config.pts_voxel_encoder.type = "BackwardPillarFeatureNetONNX"

    model_config.pts_bbox_head.type = "CenterHeadONNX"
    model_config.pts_bbox_head.separate_head.type = "SeparateHeadONNX"
    model_config.pts_bbox_head.rot_y_axis_reference = rot_y_axis_reference

    if (
        getattr(model_config, "pts_backbone", None)
        and getattr(model_config.pts_backbone, "type", None) == "ConvNeXt_PC"
    ):
        model_config.pts_backbone.with_cp = False

    export_model_cfg.model = model_config
    return export_model_cfg


def build_centerpoint_model(
    model_cfg: Config,
    checkpoint_path: str,
    device: DeviceSpec,
    *,
    rot_y_axis_reference: bool = False,
) -> torch.nn.Module:
    """Build a CenterPoint model from config and load checkpoint weights (for export + reference eval).

    Swaps the model config to CenterPoint's ONNX-friendly module variants, then builds and
    loads it via the shared mmdet3d model core (mirrors :func:`build_bevfusion_model`).

    Args:
        model_cfg: MMEngine model configuration.
        checkpoint_path: Path to the checkpoint file.
        device: Target device specification.
        rot_y_axis_reference: Whether to use y-axis rotation reference.

    Returns:
        The loaded model; the export config it was built from is available as ``model.cfg``.
    """
    export_model_cfg = create_onnx_model_cfg(
        model_cfg,
        device=device,
        rot_y_axis_reference=rot_y_axis_reference,
    )

    return build_mmdet3d_model(export_model_cfg, checkpoint_path, device)
