"""
Base Pipeline Classes for Deployment Framework.

This module provides the base abstract class for all deployment pipelines,
along with the factory base class and registry for project-specific factories.
"""

from deployment.pipelines.common.base_factory import BasePipelineFactory
from deployment.pipelines.common.base_pipeline import BaseDeploymentPipeline
from deployment.pipelines.common.project_names import ProjectNames
from deployment.pipelines.common.registry import PipelineRegistry, pipeline_registry

__all__ = [
    "BaseDeploymentPipeline",
    "BasePipelineFactory",
    "PipelineRegistry",
    "pipeline_registry",
    "ProjectNames",
]
