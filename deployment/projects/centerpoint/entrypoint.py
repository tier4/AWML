"""CenterPoint deployment entrypoint invoked by the unified CLI."""

from __future__ import annotations

import argparse

from mmengine.config import Config

from deployment.cli.args import add_deployment_file_logging, setup_logging
from deployment.config.base import BaseDeploymentConfig
from deployment.metrics.detection_3d_metrics import extract_t4metric_v2_config
from deployment.projects.centerpoint.contexts import CenterPointExportContext
from deployment.projects.centerpoint.evaluation.evaluator import CenterPointEvaluator
from deployment.projects.centerpoint.evaluation.executor import CenterPointExecutor
from deployment.projects.centerpoint.io.data_loader import CenterPointDataLoader
from deployment.projects.centerpoint.runner import CenterPointDeploymentRunner
from deployment.projects.registry import project_registry


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

    log_file = config.resolved_deploy_log_file
    if log_file:
        add_deployment_file_logging(log_file)
        logger.info("Deployment log file: %s", log_file)

    project_registry.validate_required_components("centerpoint", config.components_cfg)

    logger.info("=" * 80)
    logger.info("CenterPoint Deployment Pipeline")
    logger.info("=" * 80)

    data_loader = CenterPointDataLoader(
        model_cfg=model_cfg,
    )
    logger.info("Loaded %s samples", data_loader.num_samples)

    metrics_config = extract_t4metric_v2_config(model_cfg)

    # One executor instance, shared by the evaluator (evaluate/verify) and the runner
    # (which hands it the loaded reference model after export).
    executor = CenterPointExecutor(components_cfg=config.components_cfg)

    evaluator = CenterPointEvaluator(
        model_cfg=model_cfg,
        metrics_config=metrics_config,
        executor=executor,
    )

    runner = CenterPointDeploymentRunner(
        data_loader=data_loader,
        evaluator=evaluator,
        executor=executor,
        config=config,
        model_cfg=model_cfg,
    )

    context = CenterPointExportContext(rot_y_axis_reference=bool(getattr(args, "rot_y_axis_reference", False)))
    runner.run(context=context)
    return 0
