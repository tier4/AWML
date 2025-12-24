"""
Deployment Pipelines for Complex Models.

This module provides pipeline abstractions for models that require
multi-stage processing with mixed PyTorch and optimized backend inference.

Architecture:
    - BasePipelineFactory: Abstract base class for project-specific factories
    - pipeline_registry: Registry for dynamic project registration
    - PipelineFactory: Unified interface for creating pipelines

Adding a New Project:
    1. Create a factory.py in your project directory (e.g., pipelines/myproject/factory.py)
    2. Implement a class inheriting from BasePipelineFactory
    3. Use @pipeline_registry.register decorator
    4. Import the factory in this __init__.py to trigger registration

Example:
    >>> from deployment.pipelines import PipelineFactory, pipeline_registry
    >>> pipeline = PipelineFactory.create("centerpoint", model_spec, pytorch_model)
    >>> print(pipeline_registry.list_projects())
"""

# CenterPoint pipelines
from deployment.pipelines.centerpoint import (
    CenterPointDeploymentPipeline,
    CenterPointONNXPipeline,
    CenterPointPipelineFactory,
    CenterPointPyTorchPipeline,
    CenterPointTensorRTPipeline,
)

# Base classes and registry
from deployment.pipelines.common import (
    BaseDeploymentPipeline,
    BasePipelineFactory,
    PipelineRegistry,
    ProjectNames,
    pipeline_registry,
)

# Pipeline factory
from deployment.pipelines.factory import PipelineFactory

# Add pipelines here


__all__ = [
    # Base classes and registry
    "BaseDeploymentPipeline",
    "BasePipelineFactory",
    "PipelineRegistry",
    "pipeline_registry",
    "ProjectNames",
    # Factory
    "PipelineFactory",
    # CenterPoint
    "CenterPointDeploymentPipeline",
    "CenterPointPyTorchPipeline",
    "CenterPointONNXPipeline",
    "CenterPointTensorRTPipeline",
    "CenterPointPipelineFactory",
    # Add pipelines here
]
