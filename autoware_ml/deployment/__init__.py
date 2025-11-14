"""
Autoware ML Unified Deployment Framework
This package provides a unified, task-agnostic deployment framework for
exporting, verifying, and evaluating machine learning models across different
tasks (classification, detection, segmentation, etc.) and backends (ONNX,
TensorRT, TorchScript, etc.).
"""

from .core.base_config import BaseDeploymentConfig
from .core.base_data_loader import BaseDataLoader
from .core.base_evaluator import BaseEvaluator
from .core.preprocessing_builder import build_preprocessing_pipeline
from .runners import DeploymentRunner

__all__ = [
    "BaseDeploymentConfig",
    "BaseDataLoader",
    "BaseEvaluator",
    "DeploymentRunner",
    "build_preprocessing_pipeline",
]

__version__ = "1.0.0"
