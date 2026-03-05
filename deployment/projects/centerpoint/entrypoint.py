"""CenterPoint deployment entrypoint invoked by the unified CLI."""

from __future__ import annotations

import argparse
import logging

from mmengine.config import Config

from deployment.core.config.base_config import BaseDeploymentConfig, setup_logging
from deployment.core.contexts import CenterPointExportContext
from deployment.projects.centerpoint.data_loader import CenterPointDataLoader
from deployment.projects.centerpoint.evaluator import CenterPointEvaluator
from deployment.projects.centerpoint.metrics_utils import extract_t4metric_v2_config
from deployment.projects.centerpoint.runner import CenterPointDeploymentRunner

_REQUIRED_COMPONENTS = ("pts_voxel_encoder", "pts_backbone_neck_head")


def _validate_required_components(components_cfg) -> None:
    """Validate that all CenterPoint required components exist."""
    for component_name in _REQUIRED_COMPONENTS:
        components_cfg.get_component(component_name)


def run(args: argparse.Namespace) -> int:
    """Run the CenterPoint deployment workflow for the unified CLI.

    Args:
        args: Parsed command-line arguments containing deploy_cfg and model_cfg paths.

    Returns:
        Exit code (0 for success).
    """
    logger = setup_logging(args.log_level)

    deploy_cfg = Config.fromfile(args.deploy_cfg)
    model_cfg = Config.fromfile(args.model_cfg)
    config = BaseDeploymentConfig(deploy_cfg)

    _validate_required_components(config.components_cfg)

    logger.info("=" * 80)
    logger.info("CenterPoint Deployment Pipeline (Unified CLI)")
    logger.info("=" * 80)

    data_loader = CenterPointDataLoader(
        info_file=config.runtime_config.info_file,
        model_cfg=model_cfg,
        device="cpu",
        task_type=config.task_type,
    )
    logger.info(f"Loaded {data_loader.num_samples} samples")

    metrics_config = extract_t4metric_v2_config(model_cfg, logger=logger)

    evaluator = CenterPointEvaluator(
        model_cfg=model_cfg,
        metrics_config=metrics_config,
        components_cfg=config.components_cfg,
    )

    runner = CenterPointDeploymentRunner(
        data_loader=data_loader,
        evaluator=evaluator,
        config=config,
        model_cfg=model_cfg,
        logger=logger,
    )

    context = CenterPointExportContext(rot_y_axis_reference=bool(getattr(args, "rot_y_axis_reference", False)))
    runner.run(context=context)
    return 0
