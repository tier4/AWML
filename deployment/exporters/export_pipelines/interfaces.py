"""
Interfaces for export pipeline components.

This module defines interfaces that allow project-specific code to provide
model-specific knowledge to generic deployment export pipelines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch

from deployment.exporters.common.configs import ONNXExportConfig


@dataclass(frozen=True)
class ExportableComponent:
    """
    A model component ready for ONNX export.

    Attributes:
        name: Component name (e.g., "voxel_encoder", "backbone_head")
        module: PyTorch module to export
        sample_input: Sample input tensor for tracing
        config_override: Optional ONNX export config override
    """

    name: str
    module: torch.nn.Module
    sample_input: Any
    config_override: Optional[ONNXExportConfig] = None


class ModelComponentExtractor(ABC):
    """
    Interface for extracting exportable model components.

    This interface allows project-specific code to provide model-specific
    knowledge (model structure, component extraction, input preparation)
    without the deployment framework needing to know about specific models.

    This solves the dependency inversion problem: instead of deployment
    framework importing from projects/, projects/ implement this interface
    and inject it into export pipelines.
    """

    @abstractmethod
    def extract_components(self, model: torch.nn.Module, sample_data: Any) -> List[ExportableComponent]:
        """
        Extract all components that need to be exported to ONNX.

        This method should handle all model-specific logic:
        - Running model inference to prepare inputs
        - Creating combined modules (e.g., backbone+neck+head)
        - Preparing sample inputs for each component
        - Specifying ONNX export configs for each component

        Args:
            model: PyTorch model to extract components from
            sample_data: Sample data for preparing inputs

        Returns:
            List of ExportableComponent instances ready for ONNX export
        """
        pass
