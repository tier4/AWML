"""
CenterPoint metrics utilities.

This module extracts metrics configuration from MMEngine model configs.
"""

import logging
from typing import List, Optional

from mmengine.config import Config

from deployment.core.metrics.detection_3d_metrics import Detection3DMetricsConfig


def extract_t4metric_v2_config(
    model_cfg: Config,
    class_names: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> Detection3DMetricsConfig:
    """Extract `Detection3DMetricsConfig` from an MMEngine model config.

    Expects the config to contain a `T4MetricV2` evaluator (val or test).

    Args:
        model_cfg: MMEngine model configuration.
        class_names: Optional list of class names. If not provided,
                    extracted from model_cfg.class_names.
        logger: Optional logger instance.

    Returns:
        Detection3DMetricsConfig instance with extracted settings.

    Raises:
        ValueError: If class_names not provided and not found in model_cfg,
                   or if evaluator config is missing or not T4MetricV2 type.
    """
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
