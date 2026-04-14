"""
CenterPoint metrics utilities.

This module extracts metrics configuration from MMEngine model configs.
"""

import logging
from typing import Any

from mmengine.config import Config, ConfigDict

from deployment.core.metrics.detection_3d_metrics import Detection3DMetricsConfig

_T4METRIC_V2_EVALUATOR_TYPE = "T4MetricV2"


def extract_t4metric_v2_config(
    model_cfg: Config,
    logger: logging.Logger,
) -> Detection3DMetricsConfig:
    """Extract `Detection3DMetricsConfig` from an MMEngine model config.

    Expects the config to contain a `T4MetricV2` val evaluator.

    Args:
        model_cfg: MMEngine model configuration.
        logger: Logger instance.

    Returns:
        Detection3DMetricsConfig instance with extracted settings.

    Raises:
        ValueError: If class_names not provided and not found in model_cfg,
                   or if evaluator config is missing or not T4MetricV2 type.
    """

    def read_required_cfg_value(cfg: Config | ConfigDict, key: str) -> Any:
        """Read a required key/attribute from config object.

        Args:
            cfg: MMEngine Config or ConfigDict to read from.
            key: Required key/attribute name.

        Returns:
            Value stored at the given key/attribute.

        Raises:
            ValueError: If key/attribute does not exist in cfg.
        """
        if key in cfg:
            return cfg[key]
        if hasattr(cfg, key):
            return getattr(cfg, key)
        raise ValueError(f"Missing required key/attribute '{key}'")

    class_names = read_required_cfg_value(model_cfg, "class_names")
    evaluator_cfg = read_required_cfg_value(model_cfg, "val_evaluator")

    evaluator_type = read_required_cfg_value(evaluator_cfg, "type")
    if evaluator_type != _T4METRIC_V2_EVALUATOR_TYPE:
        raise ValueError(f"Evaluator type is '{evaluator_type}', not '{_T4METRIC_V2_EVALUATOR_TYPE}'")

    perception_configs = read_required_cfg_value(evaluator_cfg, "perception_evaluator_configs")
    evaluation_config_dict = read_required_cfg_value(perception_configs, "evaluation_config_dict")
    frame_id = read_required_cfg_value(perception_configs, "frame_id")

    critical_object_filter_config = read_required_cfg_value(evaluator_cfg, "critical_object_filter_config")
    frame_pass_fail_config = read_required_cfg_value(evaluator_cfg, "frame_pass_fail_config")

    return Detection3DMetricsConfig(
        class_names=class_names,
        frame_id=frame_id,
        evaluation_config_dict=evaluation_config_dict,
        critical_object_filter_config=critical_object_filter_config,
        frame_pass_fail_config=frame_pass_fail_config,
    )
