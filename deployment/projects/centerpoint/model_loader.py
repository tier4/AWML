"""
CenterPoint deployment utilities: ONNX-compatible model building and metrics config extraction.

Moved from projects/CenterPoint/deploy/utils.py into the unified deployment bundle.
"""

import copy
import logging
from typing import List, Optional, Tuple

import torch
from mmengine.config import Config
from mmengine.registry import MODELS, init_default_scope
from mmengine.runner import load_checkpoint

from deployment.core.metrics.detection_3d_metrics import Detection3DMetricsConfig
from deployment.projects.centerpoint.onnx_models import register_models


def create_onnx_model_cfg(
    model_cfg: Config,
    device: str,
    rot_y_axis_reference: bool = False,
) -> Config:
    onnx_cfg = model_cfg.copy()
    model_config = copy.deepcopy(onnx_cfg.model)

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

    onnx_cfg.model = model_config
    return onnx_cfg


def build_model_from_cfg(model_cfg: Config, checkpoint_path: str, device: str) -> torch.nn.Module:
    # Ensure CenterPoint ONNX variants are registered into MODELS before building.
    # This is required because the config uses string types like "CenterPointONNX", "CenterHeadONNX", etc.
    register_models()
    init_default_scope("mmdet3d")
    model_config = copy.deepcopy(model_cfg.model)
    model = MODELS.build(model_config)
    model.to(device)
    load_checkpoint(model, checkpoint_path, map_location=device)
    model.eval()
    model.cfg = model_cfg
    return model


def build_centerpoint_onnx_model(
    base_model_cfg: Config,
    checkpoint_path: str,
    device: str,
    rot_y_axis_reference: bool = False,
) -> Tuple[torch.nn.Module, Config]:
    onnx_cfg = create_onnx_model_cfg(
        base_model_cfg,
        device=device,
        rot_y_axis_reference=rot_y_axis_reference,
    )
    model = build_model_from_cfg(onnx_cfg, checkpoint_path, device=device)
    return model, onnx_cfg


def extract_t4metric_v2_config(
    model_cfg: Config,
    class_names: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> Detection3DMetricsConfig:
    if logger is None:
        logger = logging.getLogger(__name__)

    if class_names is None:
        if hasattr(model_cfg, "class_names"):
            class_names = model_cfg.class_names
        else:
            raise ValueError("class_names must be provided or defined in model_cfg.class_names")

    evaluator_cfg = None
    if hasattr(model_cfg, "val_evaluator"):
        evaluator_cfg = model_cfg.val_evaluator
    elif hasattr(model_cfg, "test_evaluator"):
        evaluator_cfg = model_cfg.test_evaluator
    else:
        raise ValueError("No val_evaluator or test_evaluator found in model_cfg")

    def get_cfg_value(cfg, key, default=None):
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    evaluator_type = get_cfg_value(evaluator_cfg, "type")
    if evaluator_type != "T4MetricV2":
        raise ValueError(f"Evaluator type is '{evaluator_type}', not 'T4MetricV2'")

    perception_configs = get_cfg_value(evaluator_cfg, "perception_evaluator_configs", {})
    evaluation_config_dict = get_cfg_value(perception_configs, "evaluation_config_dict")
    frame_id = get_cfg_value(perception_configs, "frame_id", "base_link")

    critical_object_filter_config = get_cfg_value(evaluator_cfg, "critical_object_filter_config")
    frame_pass_fail_config = get_cfg_value(evaluator_cfg, "frame_pass_fail_config")

    if evaluation_config_dict and hasattr(evaluation_config_dict, "to_dict"):
        evaluation_config_dict = dict(evaluation_config_dict)
    if critical_object_filter_config and hasattr(critical_object_filter_config, "to_dict"):
        critical_object_filter_config = dict(critical_object_filter_config)
    if frame_pass_fail_config and hasattr(frame_pass_fail_config, "to_dict"):
        frame_pass_fail_config = dict(frame_pass_fail_config)

    return Detection3DMetricsConfig(
        class_names=class_names,
        frame_id=frame_id,
        evaluation_config_dict=evaluation_config_dict,
        critical_object_filter_config=critical_object_filter_config,
        frame_pass_fail_config=frame_pass_fail_config,
    )
