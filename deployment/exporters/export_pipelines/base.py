"""
Base export pipeline interfaces for specialized export flows.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from deployment.core.artifacts import Artifact
from deployment.core.config.base_config import BaseDeploymentConfig
from deployment.core.io.base_data_loader import BaseDataLoader


class OnnxExportPipeline(ABC):
    """
    Base interface for ONNX export pipelines.
    """

    @abstractmethod
    def export(
        self,
        *,
        model: Any,
        data_loader: BaseDataLoader,
        output_dir: str,
        config: BaseDeploymentConfig,
        sample_idx: int = 0,
    ) -> Artifact:
        """
        Execute the ONNX export pipeline and return the produced artifact.

        Args:
            model: PyTorch model to export
            data_loader: Data loader for samples
            output_dir: Directory for output files
            config: Deployment configuration
            sample_idx: Sample index for tracing

        Returns:
            Artifact describing the exported ONNX output
        """


class TensorRTExportPipeline(ABC):
    """
    Base interface for TensorRT export pipelines.
    """

    @abstractmethod
    def export(
        self,
        *,
        onnx_path: str,
        output_dir: str,
        config: BaseDeploymentConfig,
        device: str,
    ) -> Artifact:
        """
        Execute the TensorRT export pipeline and return the produced artifact.

        Args:
            onnx_path: Path to ONNX model file/directory
            output_dir: Directory for output files
            config: Deployment configuration
            device: CUDA device string

        Returns:
            Artifact describing the exported TensorRT output
        """
