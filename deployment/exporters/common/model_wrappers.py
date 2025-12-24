"""
Base model wrappers for ONNX export.

This module provides the base classes for model wrappers that prepare models
for ONNX export with specific output formats and processing requirements.

Each project should define its own wrapper in {project}/model_wrappers.py,
either by using IdentityWrapper or by creating a custom wrapper that inherits
from BaseModelWrapper.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

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

    def __init__(self, model: nn.Module, **kwargs):
        """
        Initialize wrapper.

        Args:
            model: PyTorch model to wrap
            **kwargs: Wrapper-specific arguments
        """
        super().__init__()
        self.model = model
        self._wrapper_config = kwargs

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass for ONNX export.

        Must be implemented by subclasses to define ONNX-specific output format.
        """
        raise NotImplementedError

    def get_config(self) -> Dict[str, Any]:
        """Get wrapper configuration."""
        return self._wrapper_config


class IdentityWrapper(BaseModelWrapper):
    """
    Identity wrapper that doesn't modify the model.

    Useful for models that don't need special ONNX export handling.
    This is the default wrapper for most models.
    """

    def __init__(self, model: nn.Module, **kwargs):
        super().__init__(model, **kwargs)

    def forward(self, *args, **kwargs):
        """Forward pass without modification."""
        return self.model(*args, **kwargs)
