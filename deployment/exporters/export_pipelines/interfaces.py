"""
Interfaces for export pipeline components.

This module defines interfaces that allow project-specific code to provide
model-specific knowledge to generic deployment export pipelines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List

import torch


@dataclass(frozen=True)
class ExportableComponent:
    """A model component ready for ONNX export.

    Attributes:
        name: Component identifier (same as key in deploy config components). Used for
              config lookup, output filename, and logs.
        module: PyTorch module to export.
        sample_input: Sample input tensor for tracing.
    """

    name: str
    module: torch.nn.Module
    sample_input: Any


class ModelComponentExtractor(ABC):
    """Interface for extracting exportable model components.

    This interface allows project-specific code to provide model-specific
    knowledge (model structure, component extraction, input preparation)
    without the deployment framework needing to know about specific models.
    """

    @abstractmethod
    def extract_components(self, model: torch.nn.Module, sample_data: Any) -> List[ExportableComponent]:
        """Extract all components that need to be exported to ONNX.

        Args:
            model: PyTorch model to extract components from
            sample_data: Sample data for preparing inputs

        Returns:
            List of ExportableComponent instances ready for ONNX export
        """
        ...

    @abstractmethod
    def extract_features(
        self,
        model: torch.nn.Module,
        data_loader: Any,
        sample_idx: int,
    ) -> Any:
        """Extract model-specific intermediate features required for multi-component export.

        Args:
            model: PyTorch model used for feature extraction
            data_loader: Data loader used to access the sample
            sample_idx: Sample index used for tracing/feature extraction

        Returns:
            Model-specific payload that ``extract_components`` expects.
        """
        ...
