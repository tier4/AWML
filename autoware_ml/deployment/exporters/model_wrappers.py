"""
Model wrappers for ONNX export.
This module provides wrapper classes that prepare models for ONNX export
with specific output formats and processing requirements.
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
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get wrapper configuration."""
        return self._wrapper_config


# TODO(vividf): class YOLOXONNXWrapper


class IdentityWrapper(BaseModelWrapper):
    """
    Identity wrapper that doesn't modify the model.

    Useful for models that don't need special ONNX export handling.
    """

    def __init__(self, model: nn.Module, **kwargs):
        super().__init__(model, **kwargs)

    def forward(self, *args, **kwargs):
        """Forward pass without modification."""
        return self.model(*args, **kwargs)


# Model wrapper registry
_MODEL_WRAPPERS = {
    # 'yolox': YOLOXONNXWrapper,
    "identity": IdentityWrapper,
}


def register_model_wrapper(name: str, wrapper_class: type):
    """
    Register a custom model wrapper.

    Args:
        name: Wrapper name
        wrapper_class: Wrapper class (must inherit from BaseModelWrapper)
    """
    if not issubclass(wrapper_class, BaseModelWrapper):
        raise TypeError(f"Wrapper class must inherit from BaseModelWrapper, got {wrapper_class}")
    _MODEL_WRAPPERS[name] = wrapper_class


def get_model_wrapper(name: str):
    """
    Get model wrapper class by name.

    Args:
        name: Wrapper name

    Returns:
        Wrapper class

    Raises:
        KeyError: If wrapper name not found
    """
    if name not in _MODEL_WRAPPERS:
        raise KeyError(f"Model wrapper '{name}' not found. " f"Available wrappers: {list(_MODEL_WRAPPERS.keys())}")
    return _MODEL_WRAPPERS[name]


def list_model_wrappers():
    """List all registered model wrappers."""
    return list(_MODEL_WRAPPERS.keys())
