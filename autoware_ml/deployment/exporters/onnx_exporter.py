"""ONNX model exporter."""

import logging
import os
from typing import Any, Dict, Optional

import onnx
import onnxsim
import torch

from .base_exporter import BaseExporter


class ONNXExporter(BaseExporter):
    """
    ONNX model exporter with enhanced features.
    Exports PyTorch models to ONNX format with:
    - Optional model wrapping for ONNX-specific output formats
    - Optional model simplification
    - Multi-file export support for complex models
    - Configuration override capability
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger = None):
        """
        Initialize ONNX exporter.
        Args:
            config: ONNX export configuration
            logger: Optional logger instance
        """
        super().__init__(config, logger)

    def export(
        self,
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        output_path: str,
        config_override: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Export model to ONNX format.
        Args:
            model: PyTorch model to export
            sample_input: Sample input tensor
            output_path: Path to save ONNX model
            config_override: Optional config overrides for this specific export
        Returns:
            True if export succeeded
        """
        # Apply model wrapper if configured
        model = self.prepare_model(model)
        model.eval()

        # Merge config with overrides
        export_config = self.config.copy()
        if config_override:
            export_config.update(config_override)

        self.logger.info("Exporting model to ONNX format...")
        self.logger.info(f"  Input shape: {sample_input.shape}")
        self.logger.info(f"  Output path: {output_path}")
        self.logger.info(f"  Opset version: {export_config.get('opset_version', 16)}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        try:
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    sample_input,
                    output_path,
                    export_params=export_config.get("export_params", True),
                    keep_initializers_as_inputs=export_config.get("keep_initializers_as_inputs", False),
                    opset_version=export_config.get("opset_version", 16),
                    do_constant_folding=export_config.get("do_constant_folding", True),
                    input_names=export_config.get("input_names", ["input"]),
                    output_names=export_config.get("output_names", ["output"]),
                    dynamic_axes=export_config.get("dynamic_axes"),
                    verbose=export_config.get("verbose", False),
                )

            self.logger.info(f"ONNX export completed: {output_path}")

            # Optional model simplification
            if export_config.get("simplify", True):
                self._simplify_model(output_path)

            return True

        except Exception as e:
            self.logger.error(f"ONNX export failed: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            return False

    def export_multi(
        self,
        models: Dict[str, torch.nn.Module],
        sample_inputs: Dict[str, torch.Tensor],
        output_dir: str,
        configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> bool:
        """
        Export multiple models to separate ONNX files.

        Useful for complex models that need to be split into multiple files
        (e.g., CenterPoint: voxel encoder + backbone/neck/head).

        Args:
            models: Dict of {filename: model}
            sample_inputs: Dict of {filename: input_tensor}
            output_dir: Directory to save ONNX files
            configs: Optional dict of {filename: config_override}

        Returns:
            True if all exports succeeded
        """
        self.logger.info(f"Exporting {len(models)} models to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        success_count = 0
        configs = configs or {}

        for name, model in models.items():
            if name not in sample_inputs:
                self.logger.error(f"No sample input provided for model: {name}")
                continue

            output_path = os.path.join(output_dir, name)
            if not output_path.endswith(".onnx"):
                output_path += ".onnx"

            config_override = configs.get(name)
            success = self.export(
                model=model,
                sample_input=sample_inputs[name],
                output_path=output_path,
                config_override=config_override,
            )

            if success:
                success_count += 1
                self.logger.info(f"✅ Exported {name}")
            else:
                self.logger.error(f"❌ Failed to export {name}")

        total = len(models)
        if success_count == total:
            self.logger.info(f"✅ All {total} models exported successfully")
            return True
        elif success_count > 0:
            self.logger.warning(f"⚠️  Partial success: {success_count}/{total} models exported")
            return False
        else:
            self.logger.error(f"❌ All exports failed")
            return False

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
                self.logger.info(f"ONNX model simplified successfully")
            else:
                self.logger.warning("ONNX model simplification failed")
        except Exception as e:
            self.logger.warning(f"ONNX simplification error: {e}")
