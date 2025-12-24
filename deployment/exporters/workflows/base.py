"""
Base workflow interfaces for specialized export flows.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from deployment.core.artifacts import Artifact
from deployment.core.config.base_config import BaseDeploymentConfig
from deployment.core.contexts import ExportContext
from deployment.core.io.base_data_loader import BaseDataLoader


class OnnxExportWorkflow(ABC):
    """
    Base interface for ONNX export workflows.
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
        context: Optional[ExportContext] = None,
    ) -> Artifact:
        """
        Execute the ONNX export workflow and return the produced artifact.

        Args:
            model: PyTorch model to export
            data_loader: Data loader for samples
            output_dir: Directory for output files
            config: Deployment configuration
            sample_idx: Sample index for tracing
            context: Typed export context with project-specific parameters.
                     Use project-specific context subclasses (e.g., CenterPointExportContext)
                     for type-safe access to parameters.

        Returns:
            Artifact describing the exported ONNX output
        """


class TensorRTExportWorkflow(ABC):
    """
    Base interface for TensorRT export workflows.
    """

    @abstractmethod
    def export(
        self,
        *,
        onnx_path: str,
        output_dir: str,
        config: BaseDeploymentConfig,
        device: str,
        data_loader: BaseDataLoader,
        context: Optional[ExportContext] = None,
    ) -> Artifact:
        """
        Execute the TensorRT export workflow and return the produced artifact.

        Args:
            onnx_path: Path to ONNX model file/directory
            output_dir: Directory for output files
            config: Deployment configuration
            device: CUDA device string
            data_loader: Data loader for samples
            context: Typed export context with project-specific parameters.
                     Use project-specific context subclasses for type-safe access.

        Returns:
            Artifact describing the exported TensorRT output
        """
