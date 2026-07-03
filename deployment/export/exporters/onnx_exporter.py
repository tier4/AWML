"""ONNX model exporter."""

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Optional

import onnx
import onnxsim
import torch

from deployment.export.exporters.configs import ONNXExportConfig
from deployment.primitives.artifacts import Artifact

logger = logging.getLogger(__name__)


class ONNXExporter:
    """
    ONNX model exporter with enhanced features.

    Exports PyTorch models to ONNX format with:
    - Optional model wrapping for ONNX-specific output formats
    - Optional model simplification
    - Configuration override capability
    """

    def __init__(
        self,
        config: ONNXExportConfig,
        model_wrapper: Optional[Any] = None,
    ) -> None:
        """
        Initialize ONNX exporter.

        Args:
            config: ONNX export configuration dataclass instance.
            model_wrapper: Optional model wrapper class (e.g., YOLOXOptElanONNXWrapper)
        """
        self.config = config
        self._model_wrapper = model_wrapper
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
    ) -> Artifact:
        """Export model to ONNX format.

        Args:
            model: PyTorch model to export
            sample_input: Sample input tensor
            output_path: Path to save ONNX model

        Returns:
            Artifact describing the exported ONNX model.

        Raises:
            RuntimeError: If export fails
            ValueError: If configuration is invalid
        """
        model = self._prepare_for_onnx(model)
        self._do_onnx_export(model, sample_input, output_path, self.config)
        if self.config.simplify:
            self._simplify_model(output_path)
        return Artifact(path=output_path)

    def _prepare_for_onnx(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Prepare model for ONNX export.

        Applies model wrapper if configured and sets model to eval mode.

        Args:
            model: PyTorch model to prepare

        Returns:
            Prepared model ready for ONNX export
        """
        if self._model_wrapper is not None:
            logger.info("Applying model wrapper for export")
            model = self._model_wrapper(model)
        model.eval()
        return model

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
        logger.info("Exporting model to ONNX format...")
        if hasattr(sample_input, "shape"):
            logger.info("  Input shape: %s", sample_input.shape)
        logger.info("  Output path: %s", output_path)

        logger.info("  Opset version: %s", export_cfg.opset_version)

        # Ensure output directory exists
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        # Export into a private staging directory, then publish the result into place.
        # torch.onnx.export may emit external-data sidecar files next to the .onnx (for models
        # whose weights exceed the 2GB protobuf limit). Staging the whole set and publishing it
        # together means a failed/interrupted export never leaves a partial model in the target
        # directory, and the .onnx never becomes visible before the data files it references.
        staging = output.parent / f".{output.name}.staging"
        produced = staging / output.name
        self._reset_dir(staging)
        try:
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    sample_input,
                    str(produced),
                    export_params=export_cfg.export_params,
                    keep_initializers_as_inputs=export_cfg.keep_initializers_as_inputs,
                    opset_version=export_cfg.opset_version,
                    do_constant_folding=export_cfg.do_constant_folding,
                    input_names=list(export_cfg.input_names),
                    output_names=list(export_cfg.output_names),
                    dynamic_axes=export_cfg.dynamic_axes,
                    verbose=export_cfg.verbose,
                )
            self._publish(staging, produced, output)

            logger.info("ONNX export completed: %s", output_path)

        except Exception as exc:
            logger.exception("ONNX export failed: %s", output_path)
            raise RuntimeError(f"ONNX export failed: {output_path}") from exc
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _simplify_model(self, onnx_path: str) -> None:
        """
        Simplify ONNX model using onnxsim.

        Args:
            onnx_path: Path to ONNX model file
        """
        logger.info("Simplifying ONNX model...")
        target = Path(onnx_path)
        # Save into a staging dir and publish, for the same external-data and atomicity reasons
        # as the export above: never overwrite the valid exported model in place.
        staging = target.parent / f".{target.name}.simplify.staging"
        produced = staging / target.name
        try:
            model_simplified, success = onnxsim.simplify(onnx_path)
            if not success:
                logger.error("ONNX model simplification failed; keeping unsimplified model")
                return
            self._reset_dir(staging)
            onnx.save(model_simplified, str(produced))
            self._publish(staging, produced, target)
            logger.info("ONNX model simplified successfully")
        except Exception as e:
            logger.error("ONNX simplification error: %s", e)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    @staticmethod
    def _reset_dir(path: Path) -> None:
        """Create an empty staging directory, removing any leftovers from a prior run."""
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True)

    @staticmethod
    def _publish(staging: Path, produced: Path, target: Path) -> None:
        """Move a freshly produced ONNX (and any external-data sidecars) into place.

        Sidecar files are moved first and the main ``.onnx`` (``produced``) last, so a reader
        that observes the model file always sees the data files it references. ``os.replace``
        is atomic within the destination directory.

        Args:
            staging: Directory holding the freshly produced files.
            produced: The main ``.onnx`` file inside ``staging``.
            target: Final path for the main ``.onnx`` file.
        """
        dest_dir = target.parent
        for item in sorted(staging.iterdir()):
            if item == produced:
                continue
            os.replace(item, dest_dir / item.name)
        os.replace(produced, target)
