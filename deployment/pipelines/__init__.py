"""Deployment pipeline infrastructure.

Project-specific pipeline implementations live under `deployment/projects/<project>/pipelines/`
and should register themselves into `deployment.pipelines.registry.pipeline_registry`.
"""

from deployment.pipelines.base_factory import BasePipelineFactory
from deployment.pipelines.base_pipeline import BaseDeploymentPipeline
from deployment.pipelines.factory import PipelineFactory
from deployment.pipelines.registry import PipelineRegistry, pipeline_registry

__all__ = [
    "BaseDeploymentPipeline",
    "BasePipelineFactory",
    "PipelineRegistry",
    "pipeline_registry",
    "PipelineFactory",
]
