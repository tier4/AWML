"""Shared entrypoint wiring for 3D-detection deployment projects.

Every point-cloud 3D detector (CenterPoint, BEVFusion-L, …) wires up its deployment the same
way: parse the two configs, build the typed deploy config, set up file logging, build the
MMDet3D point-cloud data loader, derive the T4MetricV2 metrics config, then hand a shared
``Detection3DEvaluator`` to the project's runner. Only three things vary between projects — the
typed config class, how the backend executor is constructed, and the runner class — so that
variation is injected and the identical wiring lives here once instead of in each ``entrypoint.py``.
"""

from __future__ import annotations

import argparse
from typing import Callable

from mmengine.config import Config

from deployment.cli.args import add_deployment_file_logging, setup_logging
from deployment.config.base import BaseDeploymentConfig
from deployment.evaluation.detection_3d_evaluator import Detection3DEvaluator
from deployment.execution.backend_executor import BackendExecutor
from deployment.io.point_cloud_data_loader import PointCloudDataLoader
from deployment.metrics.detection_3d_metrics import extract_t4metric_v2_config
from deployment.runtime.runner import BaseDeploymentRunner

#: Builds the typed deploy config from the raw MMEngine ``deploy_cfg``.
ConfigFactory = Callable[[Config], BaseDeploymentConfig]
#: Builds the project's backend executor from the typed config and the raw ``deploy_cfg``
#: (the latter carries project-specific extras such as ``tensorrt_config.plugin_libraries``).
ExecutorFactory = Callable[[BaseDeploymentConfig, Config], BackendExecutor]
#: Constructs the project's deployment runner (same keyword contract as ``BaseDeploymentRunner``).
RunnerFactory = Callable[..., BaseDeploymentRunner]


def run_detection3d_deployment(
    args: argparse.Namespace,
    *,
    pipeline_name: str,
    config_factory: ConfigFactory,
    executor_factory: ExecutorFactory,
    runner_factory: RunnerFactory,
) -> int:
    """Run a 3D-detection deployment workflow with the shared wiring.

    Args:
        args: Parsed CLI args carrying ``deploy_cfg``, ``model_cfg`` and ``log_level``.
        pipeline_name: Human-readable project name for the log banner (e.g. ``"BEVFusion"``).
        config_factory: Builds the typed deploy config from the raw ``deploy_cfg``.
        executor_factory: Builds the backend executor from ``(config, deploy_cfg)``.
        runner_factory: Builds the deployment runner (``BaseDeploymentRunner`` keyword contract).

    Returns:
        Process exit code (0 on success).
    """
    logger = setup_logging(args.log_level)

    deploy_cfg = Config.fromfile(args.deploy_cfg)
    model_cfg = Config.fromfile(args.model_cfg)
    config = config_factory(deploy_cfg)

    log_file = config.resolved_deploy_log_file
    if log_file:
        add_deployment_file_logging(log_file)
        logger.info("Deployment log file: %s", log_file)

    logger.info("=" * 80)
    logger.info("%s Deployment Pipeline", pipeline_name)
    logger.info("=" * 80)

    # ``runtime_io.info_file`` overrides the dataset's ann_file when set; absent (CenterPoint) it
    # is "" and the loader keeps the model config's own ann_file.
    info_file = (deploy_cfg.get("runtime_io", {}) or {}).get("info_file", "")
    data_loader = PointCloudDataLoader(info_file=info_file, model_cfg=model_cfg)
    logger.info("Loaded %s samples", data_loader.num_samples)

    metrics_config = extract_t4metric_v2_config(model_cfg)

    # One executor instance, shared by the evaluator (evaluate/verify) and the runner (which hands
    # it the loaded reference model after export).
    executor = executor_factory(config, deploy_cfg)
    evaluator = Detection3DEvaluator(model_cfg=model_cfg, metrics_config=metrics_config, executor=executor)
    runner = runner_factory(
        data_loader=data_loader,
        evaluator=evaluator,
        executor=executor,
        config=config,
        model_cfg=model_cfg,
    )

    runner.run()
    return 0
