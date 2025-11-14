"""
Abstract base class for model exporters.
Provides a unified interface for exporting models to different formats.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import torch


class BaseExporter(ABC):
    """
    Abstract base class for model exporters.
    This class defines a unified interface for exporting models
    to different backend formats (ONNX, TensorRT, TorchScript, etc.).

    Enhanced features:
    - Support for model wrappers (preprocessing before export)
    - Flexible configuration with overrides
    - Better logging and error handling
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger = None):
        """
        Initialize exporter.
        Args:
            config: Configuration dictionary for export settings
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._model_wrapper_fn: Optional[Callable] = None

        # Extract wrapper configuration if present
        wrapper_config = config.get("model_wrapper")
        if wrapper_config:
            self._setup_model_wrapper(wrapper_config)

    def _setup_model_wrapper(self, wrapper_config):
        """
        Setup model wrapper from configuration.

        Args:
            wrapper_config: Either a string (wrapper name) or dict with 'type' and kwargs
        """
        from .model_wrappers import get_model_wrapper

        if isinstance(wrapper_config, str):
            # Simple string: wrapper name only
            wrapper_class = get_model_wrapper(wrapper_config)
            self._model_wrapper_fn = lambda model: wrapper_class(model)
        elif isinstance(wrapper_config, dict):
            # Dict with type and additional arguments
            wrapper_type = wrapper_config.get("type")
            if not wrapper_type:
                raise ValueError("Model wrapper config must have 'type' field")

            wrapper_class = get_model_wrapper(wrapper_type)
            wrapper_kwargs = {k: v for k, v in wrapper_config.items() if k != "type"}
            self._model_wrapper_fn = lambda model: wrapper_class(model, **wrapper_kwargs)
        else:
            raise TypeError(f"Model wrapper config must be str or dict, got {type(wrapper_config)}")

    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Prepare model for export (apply wrapper if configured).

        Args:
            model: Original PyTorch model

        Returns:
            Prepared model (wrapped if wrapper configured)
        """
        if self._model_wrapper_fn:
            self.logger.info("Applying model wrapper for export")
            return self._model_wrapper_fn(model)
        return model

    @abstractmethod
    def export(self, model: torch.nn.Module, sample_input: torch.Tensor, output_path: str, **kwargs) -> bool:
        """
        Export model to target format.
        Args:
            model: PyTorch model to export
            sample_input: Sample input tensor for tracing/shape inference
            output_path: Path to save exported model
            **kwargs: Additional format-specific arguments
        Returns:
            True if export succeeded, False otherwise
        Raises:
            RuntimeError: If export fails
        """
        pass

    def validate_export(self, output_path: str) -> bool:
        """
        Validate that the exported model file is valid.
        Override this in subclasses to add format-specific validation.
        Args:
            output_path: Path to exported model file
        Returns:
            True if valid, False otherwise
        """
        import os

        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
