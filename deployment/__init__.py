"""
Autoware ML Unified Deployment Framework

This package provides a unified, task-agnostic deployment framework for
exporting, verifying, and evaluating machine learning models across different
tasks (classification, detection, segmentation, etc.) and backends (ONNX,
TensorRT).
"""

from deployment.core.config.base_config import BaseDeploymentConfig
from deployment.core.evaluation.base_evaluator import BaseEvaluator
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.core.io.preprocessing_builder import build_preprocessing_pipeline
from deployment.runtime.runner import BaseDeploymentRunner

__all__ = [
    "BaseDeploymentConfig",
    "BaseDataLoader",
    "BaseEvaluator",
    "BaseDeploymentRunner",
    "build_preprocessing_pipeline",
]

__version__ = "1.0.0"
