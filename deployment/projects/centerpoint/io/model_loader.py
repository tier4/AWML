"""
CenterPoint model loading utilities.

This module provides ONNX-compatible model building from MMEngine configs.
"""

from __future__ import annotations

import copy
from typing import Tuple

import torch
from mmengine.config import Config
from mmengine.registry import MODELS, init_default_scope
from mmengine.runner import load_checkpoint

from deployment.core.device import DeviceSpec
from deployment.projects.centerpoint.onnx_models import (  # noqa: F401 - register MODELS
    centerpoint_head_onnx,
    centerpoint_onnx,
    pillar_encoder_onnx,
)


def create_onnx_model_cfg(
    model_cfg: Config,
    device: DeviceSpec,
    rot_y_axis_reference: bool = False,
) -> Config:
    """Create a model config that swaps modules to ONNX-friendly variants.

    This mutates the `model_cfg.model` subtree to reference classes registered by
    `deployment.projects.centerpoint.onnx_models` (e.g., `CenterPointONNX`).

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


def build_model_from_cfg(
    model_cfg: Config,
    checkpoint_path: str,
    device: DeviceSpec,
) -> torch.nn.Module:
    """Build a model from MMEngine config and load checkpoint weights.

    Args:
        model_cfg: MMEngine model configuration.
        checkpoint_path: Path to the checkpoint file.
        device: Target device specification.

    Returns:
        Loaded and initialized PyTorch model in eval mode.
    """
    # Importing onnx_models above triggers MODELS registration for ONNX variants.
    init_default_scope("mmdet3d")

    model_config = copy.deepcopy(model_cfg.model)
    model = MODELS.build(model_config)
    torch_device = device.to_torch_device()
    model.to(torch_device)
    load_checkpoint(model, checkpoint_path, map_location=torch_device)
    model.eval()
    model.cfg = model_cfg
    return model


def build_centerpoint_onnx_model(
    base_model_cfg: Config,
    checkpoint_path: str,
    device: DeviceSpec,
    rot_y_axis_reference: bool = False,
) -> Tuple[torch.nn.Module, Config]:
    """Build an ONNX-compatible CenterPoint model.

    Convenience wrapper that creates ONNX config and builds the model.

    Args:
        base_model_cfg: Base MMEngine model configuration.
        checkpoint_path: Path to the checkpoint file.
        device: Target device specification.
        rot_y_axis_reference: Whether to use y-axis rotation reference.

    Returns:
        Tuple of ``(model, export_model_cfg)``; the latter matches ``model.cfg``.
    """
    export_model_cfg = create_onnx_model_cfg(
        base_model_cfg,
        device=device,
        rot_y_axis_reference=rot_y_axis_reference,
    )
    model = build_model_from_cfg(export_model_cfg, checkpoint_path, device=device)
    return model, export_model_cfg
