"""
CenterPoint metrics utilities.

This module extracts metrics configuration from MMEngine model configs.
"""

import logging
from typing import Dict, List, Optional, Union

from mmengine.config import Config

from deployment.core.metrics.detection_3d_metrics import Detection3DMetricsConfig


def extract_t4metric_v2_config(
    model_cfg: Config,
    logger: Optional[logging.Logger] = None,
) -> Detection3DMetricsConfig:
    """Extract `Detection3DMetricsConfig` from an MMEngine model config.

    Expects the config to contain a `T4MetricV2` evaluator (val or test).

    Args:
        model_cfg: MMEngine model configuration.
        logger: Optional logger instance.

    Returns:
        Detection3DMetricsConfig instance with extracted settings.

    Raises:
        ValueError: If class_names not provided and not found in model_cfg,
                   or if evaluator config is missing or not T4MetricV2 type.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    class_names = model_cfg.class_names
    evaluator_cfg = model_cfg.val_evaluator or model_cfg.test_evaluator

    def read_cfg_value(
        cfg: Optional[Union[Dict[str, object], object]],
        key: str,
        *,
        required: bool,
        default: Optional[object] = None,
    ) -> object:
        """Read a key from config dict or object; return default or raise if required and missing.

        Args:
            cfg: Config dict or object with attributes.
            key: Key or attribute name to read.
            required: If True, raise KeyError when key is missing.
            default: Value to return when key is missing and not required.

        Returns:
            The value for key, or default.

        Raises:
            KeyError: If required and key/attribute is missing.
        """
        if cfg is None:
            if required:
                raise KeyError(f"Missing required config object while reading '{key}'")
            return default

        if isinstance(cfg, dict):
            if key in cfg:
                return cfg[key]
            if required:
                raise KeyError(f"Missing required key '{key}' in evaluator config dict")
            return default

        if hasattr(cfg, key):
            return getattr(cfg, key)
        if required:
            raise KeyError(f"Missing required attribute '{key}' in evaluator config object")
        return default

    evaluator_type = read_cfg_value(evaluator_cfg, "type", required=True)
    if evaluator_type != "T4MetricV2":
        raise ValueError(f"Evaluator type is '{evaluator_type}', not 'T4MetricV2'")

    perception_configs = read_cfg_value(evaluator_cfg, "perception_evaluator_configs", required=True)
    evaluation_config_dict = read_cfg_value(perception_configs, "evaluation_config_dict", required=True)
    frame_id = read_cfg_value(perception_configs, "frame_id", required=True)

    critical_object_filter_config = read_cfg_value(
        evaluator_cfg, "critical_object_filter_config", required=False, default=None
    )
    frame_pass_fail_config = read_cfg_value(evaluator_cfg, "frame_pass_fail_config", required=False, default=None)

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
