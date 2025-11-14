"""
Base Pipeline Classes for Deployment Framework.

This module provides the base abstract classes for all deployment pipelines,
including base pipeline, classification, 2D detection, and 3D detection pipelines.
"""

from .base_pipeline import BaseDeploymentPipeline
from .classification_pipeline import ClassificationPipeline
from .detection_2d_pipeline import Detection2DPipeline
from .detection_3d_pipeline import Detection3DPipeline

__all__ = [
    "BaseDeploymentPipeline",
    "ClassificationPipeline",
    "Detection2DPipeline",
    "Detection3DPipeline",
]
