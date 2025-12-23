"""CenterPoint deployment entrypoint invoked by the unified CLI."""

from __future__ import annotations

import logging

from mmengine.config import Config

from deployment.core.config.base_config import BaseDeploymentConfig, setup_logging
from deployment.core.contexts import CenterPointExportContext
from deployment.core.metrics.detection_3d_metrics import Detection3DMetricsConfig
from deployment.projects.centerpoint.data_loader import CenterPointDataLoader
from deployment.projects.centerpoint.evaluator import CenterPointEvaluator
from deployment.projects.centerpoint.model_loader import extract_t4metric_v2_config
from deployment.projects.centerpoint.runner import CenterPointDeploymentRunner


def run(args) -> int:
    logger = setup_logging(args.log_level)

    deploy_cfg = Config.fromfile(args.deploy_cfg)
    model_cfg = Config.fromfile(args.model_cfg)
    config = BaseDeploymentConfig(deploy_cfg)

    logger.info("=" * 80)
    logger.info("CenterPoint Deployment Pipeline (Unified CLI)")
    logger.info("=" * 80)

    data_loader = CenterPointDataLoader(
        info_file=config.runtime_config.info_file,
        model_cfg=model_cfg,
        device="cpu",
        task_type=config.task_type,
    )
    logger.info(f"Loaded {data_loader.get_num_samples()} samples")

    metrics_config = extract_t4metric_v2_config(model_cfg, logger=logger)

    evaluator = CenterPointEvaluator(
        model_cfg=model_cfg,
        metrics_config=metrics_config,
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
