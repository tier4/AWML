"""
Base model wrappers for ONNX export.

This module provides the base classes for model wrappers that prepare models
for ONNX export with specific output formats and processing requirements.

Each project should define its own wrapper in {project}/model_wrappers.py,
either by using IdentityWrapper or by creating a custom wrapper that inherits
from BaseModelWrapper.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseModelWrapper(nn.Module, ABC):
    """
    Abstract base class for ONNX export model wrappers.

    Wrappers modify model forward pass to produce ONNX-compatible outputs
    with specific formats required by deployment backends.

    Each project should create its own wrapper class that inherits from this
    base class if special output format conversion is needed.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize wrapper.

        Args:
            model: PyTorch model to wrap
        """
        super().__init__()
        self.model = model

    @abstractmethod
    def forward(self, *args):
        """
        Forward pass for ONNX export.

        Must be implemented by subclasses to define ONNX-specific output format.
        """
        raise NotImplementedError


class IdentityWrapper(BaseModelWrapper):
    """
    Identity wrapper that doesn't modify the model.

    Useful for models that don't need special ONNX export handling.
    This is the default wrapper for most models.
    """

    def __init__(self, model: nn.Module):
        super().__init__(model)

    def forward(self, *args):
        """Forward pass without modification."""
        return self.model(*args)
