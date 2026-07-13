"""CenterPoint deployment entrypoint invoked by the unified CLI."""

from __future__ import annotations

import argparse

from mmengine.config import Config

from deployment.config.base import BaseDeploymentConfig
from deployment.execution.backend_executor import BackendExecutor
from deployment.projects.centerpoint.config.centerpoint_deployment_config import CenterPointDeploymentConfig
from deployment.projects.centerpoint.evaluation.executor import CenterPointExecutor
from deployment.projects.centerpoint.runner import CenterPointDeploymentRunner
from deployment.runtime.detection3d_entrypoint import run_detection3d_deployment


def _build_executor(config: BaseDeploymentConfig, deploy_cfg: Config) -> BackendExecutor:
    """Build the CenterPoint executor (no custom TensorRT plugins needed)."""
    return CenterPointExecutor(components_cfg=config.components_cfg)


def run(args: argparse.Namespace) -> int:
    """Run the CenterPoint deployment workflow via the shared 3D-detection entrypoint."""
    return run_detection3d_deployment(
        args,
        pipeline_name="CenterPoint",
        config_factory=CenterPointDeploymentConfig,
        executor_factory=_build_executor,
        runner_factory=CenterPointDeploymentRunner,
    )
