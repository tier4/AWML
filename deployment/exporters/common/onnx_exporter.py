"""ONNX model exporter."""

import logging
import os
from dataclasses import replace
from typing import Any, Optional

import onnx
import onnxsim
import torch

from deployment.exporters.common.base_exporter import BaseExporter
from deployment.exporters.common.configs import ONNXExportConfig


class ONNXExporter(BaseExporter):
    """
    ONNX model exporter with enhanced features.

    Exports PyTorch models to ONNX format with:
    - Optional model wrapping for ONNX-specific output formats
    - Optional model simplification
    - Multi-file export support for complex models
    - Configuration override capability
    """

    def __init__(
        self,
        config: ONNXExportConfig,
        model_wrapper: Optional[Any] = None,
        logger: logging.Logger = None,
    ):
        """
        Initialize ONNX exporter.

        Args:
            config: ONNX export configuration dataclass instance.
            model_wrapper: Optional model wrapper class (e.g., YOLOXOptElanONNXWrapper)
            logger: Optional logger instance
        """
        super().__init__(config, model_wrapper=model_wrapper, logger=logger)
        self._validate_config(config)

    def _validate_config(self, config: ONNXExportConfig) -> None:
        """
        Validate ONNX export configuration.

        Args:
            config: Configuration to validate

        Raises:
            ValueError: If configuration is invalid
        """
        if config.opset_version < 11:
            raise ValueError(f"opset_version must be >= 11, got {config.opset_version}")

        if not config.input_names:
            raise ValueError("input_names cannot be empty")

        if not config.output_names:
            raise ValueError("output_names cannot be empty")

        if len(config.input_names) != len(set(config.input_names)):
            raise ValueError("input_names contains duplicates")

        if len(config.output_names) != len(set(config.output_names)):
            raise ValueError("output_names contains duplicates")

    def export(
        self,
        model: torch.nn.Module,
        sample_input: Any,
        output_path: str,
        *,
        config_override: Optional[ONNXExportConfig] = None,
    ) -> None:
        """
        Export model to ONNX format.

        Args:
            model: PyTorch model to export
            sample_input: Sample input tensor
            output_path: Path to save ONNX model
            config_override: Optional configuration override. If provided, will be merged
                           with base config using dataclasses.replace.

        Raises:
            RuntimeError: If export fails
            ValueError: If configuration is invalid
        """
        model = self._prepare_for_onnx(model)
        export_cfg = self._build_export_config(config_override)
        self._do_onnx_export(model, sample_input, output_path, export_cfg)
        if export_cfg.simplify:
            self._simplify_model(output_path)

    def _prepare_for_onnx(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Prepare model for ONNX export.

        Applies model wrapper if configured and sets model to eval mode.

        Args:
            model: PyTorch model to prepare

        Returns:
            Prepared model ready for ONNX export
        """
        model = self.prepare_model(model)
        model.eval()
        return model

    def _build_export_config(self, config_override: Optional[ONNXExportConfig] = None) -> ONNXExportConfig:
        """
        Build export configuration by merging base config with override.

        Args:
            config_override: Optional configuration override. If provided, all fields
                           from the override will replace corresponding fields in base config.

        Returns:
            Merged configuration ready for export

        Raises:
            ValueError: If merged configuration is invalid
        """
        if config_override is None:
            export_cfg = self.config
        else:
            export_cfg = replace(self.config, **config_override.__dict__)

        # Validate merged config
        self._validate_config(export_cfg)
        return export_cfg

    def _do_onnx_export(
        self,
        model: torch.nn.Module,
        sample_input: Any,
        output_path: str,
        export_cfg: ONNXExportConfig,
    ) -> None:
        """
        Perform ONNX export using torch.onnx.export.

        Args:
            model: Prepared PyTorch model
            sample_input: Sample input tensor
            output_path: Path to save ONNX model
            export_cfg: Export configuration

        Raises:
            RuntimeError: If export fails
        """
        self.logger.info("Exporting model to ONNX format...")
        if hasattr(sample_input, "shape"):
            self.logger.info(f"  Input shape: {sample_input.shape}")
        self.logger.info(f"  Output path: {output_path}")
        self.logger.info(f"  Opset version: {export_cfg.opset_version}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        try:
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    sample_input,
                    output_path,
                    export_params=export_cfg.export_params,
                    keep_initializers_as_inputs=export_cfg.keep_initializers_as_inputs,
                    opset_version=export_cfg.opset_version,
                    do_constant_folding=export_cfg.do_constant_folding,
                    input_names=list(export_cfg.input_names),
                    output_names=list(export_cfg.output_names),
                    dynamic_axes=export_cfg.dynamic_axes,
                    verbose=export_cfg.verbose,
                )

            self.logger.info(f"ONNX export completed: {output_path}")

        except Exception as e:
            self.logger.error(f"ONNX export failed: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            raise RuntimeError("ONNX export failed") from e

    def _simplify_model(self, onnx_path: str) -> None:
        """
        Simplify ONNX model using onnxsim.

        Args:
            onnx_path: Path to ONNX model file
        """
        self.logger.info("Simplifying ONNX model...")
        try:
            model_simplified, success = onnxsim.simplify(onnx_path)
            if success:
                onnx.save(model_simplified, onnx_path)
                self.logger.info("ONNX model simplified successfully")
            else:
                self.logger.warning("ONNX model simplification failed")
        except Exception as e:
            self.logger.warning(f"ONNX simplification error: {e}")
