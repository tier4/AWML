"""BEVFusion-L deployment entrypoint invoked by the unified CLI."""

from __future__ import annotations

import argparse

from mmengine.config import Config

from deployment.config.base import BaseDeploymentConfig
from deployment.execution.backend_executor import BackendExecutor
from deployment.projects.bevfusion_l.config.bevfusion_deployment_config import BEVFusionDeploymentConfig
from deployment.projects.bevfusion_l.evaluation.executor import BEVFusionExecutor
from deployment.projects.bevfusion_l.runner import BEVFusionDeploymentRunner
from deployment.runtime.detection3d_entrypoint import run_detection3d_deployment


def _build_executor(config: BaseDeploymentConfig, deploy_cfg: Config) -> BackendExecutor:
    """Build the BEVFusion executor, forwarding the spconv ImplicitGemm plugin ``.so`` paths."""
    plugin_libraries = tuple((deploy_cfg.get("tensorrt_config", {}) or {}).get("plugin_libraries", ()) or ())
    return BEVFusionExecutor(components_cfg=config.components_cfg, plugin_libraries=plugin_libraries)


def run(args: argparse.Namespace) -> int:
    """Run the BEVFusion-L deployment workflow via the shared 3D-detection entrypoint."""
    return run_detection3d_deployment(
        args,
        pipeline_name="BEVFusion",
        config_factory=BEVFusionDeploymentConfig,
        executor_factory=_build_executor,
        runner_factory=BEVFusionDeploymentRunner,
    )
