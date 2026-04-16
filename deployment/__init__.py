"""
Autoware ML Deployment Framework

This package provides a task-agnostic deployment framework for
exporting, verifying, and evaluating machine learning models across different
tasks (classification, detection, segmentation, etc.) and backends (ONNX,
TensorRT).
"""

from deployment.configs import BaseDeploymentConfig
from deployment.core.evaluation.base_evaluator import BaseEvaluator
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.runtime.runner import BaseDeploymentRunner

__all__ = [
    "BaseDeploymentConfig",
    "BaseDataLoader",
    "BaseEvaluator",
    "BaseDeploymentRunner",
]

__version__ = "1.0.0"
