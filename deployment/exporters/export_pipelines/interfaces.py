"""
Interfaces for export pipeline components.

This module defines interfaces that allow project-specific code to provide
model-specific knowledge to generic deployment export pipelines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

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


class ExportSampleAdapter(ABC):
    """Interface for adapting model-specific sample extraction for export.

    Implementations convert model-specific feature extraction outputs
    into a sample object that component builders can consume.
    """

    @abstractmethod
    def extract_sample(
        self,
        model: torch.nn.Module,
        data_loader: Any,
        sample_idx: int,
    ) -> Any:
        """Extract model-specific sample payload for export.

        Args:
            model: PyTorch model used for feature extraction
            data_loader: Data loader used to access the sample
            sample_idx: Sample index used for tracing/feature extraction

        Returns:
            Model-specific typed sample payload.
        """
        ...


class ModelComponentBuilder(ABC):
    """Interface for building exportable ONNX components from model and sample."""

    @abstractmethod
    def build_components(
        self,
        model: torch.nn.Module,
        sample: Any,
    ) -> list[ExportableComponent]:
        """Build all ONNX-exportable components.

        Args:
            model: PyTorch model to build components from
            sample: Typed sample payload for preparing component inputs

        Returns:
            List of exportable model components ready for ONNX export.
        """
        ...
