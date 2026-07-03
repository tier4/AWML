"""
Model-agnostic TensorRT export pipeline.

Converting ONNX → TensorRT does not touch the PyTorch model, so a single
pipeline handles every project: it iterates the deploy config's ``components``
(in order), resolves each component's ``onnx_file`` under ``onnx_path``, and
builds the corresponding ``engine_file`` under ``output_dir``. The
single-component case is just a one-iteration loop.

Because there is nothing model-specific to inject here, this is the only
TensorRT pipeline; projects do not subclass it. (Should a project ever need a
model-specific build — e.g. INT8 calibration with bespoke data — it can supply
its own ``tensorrt_pipeline`` to the runner.)
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from deployment.config.base import BaseDeploymentConfig
from deployment.export.exporters.tensorrt_exporter import TensorRTExporter
from deployment.primitives.artifacts import Artifact, resolve_artifact_path
from deployment.primitives.device import DeviceSpec

logger = logging.getLogger(__name__)


class TensorRTExportPipeline:
    """TensorRT export pipeline (one engine per config component).

    Iterates the deploy config's ``components`` (in order), resolves each
    component's ``onnx_file`` under ``onnx_path``, and builds the corresponding
    ``engine_file`` under ``output_dir``. Handles the single-component case too.
    """

    def export(
        self,
        *,
        onnx_path: str,
        output_dir: str,
        config: BaseDeploymentConfig,
        device: DeviceSpec,
    ) -> Artifact:
        """Convert each component's ONNX to a TensorRT engine under ``output_dir``.

        Args:
            onnx_path: Directory containing ONNX files (layout matches deploy config).
            output_dir: Directory where TensorRT engine files are written.
            config: Deployment config for TensorRT exporter options and component layout.
            device: CUDA device for building engines.

        Returns:
            Artifact whose path is the output directory.

        Raises:
            ValueError: If ``onnx_path`` is not a directory, CUDA is invalid, or
                ``components`` is empty.
            FileNotFoundError: If a configured ONNX file is missing under ``onnx_path``.
        """
        onnx_dir_path = Path(onnx_path)
        if not onnx_dir_path.is_dir():
            raise ValueError(f"onnx_path must be a directory for multi-file export, got: {onnx_path}")

        components_cfg = config.components_cfg
        components = list(components_cfg.items())
        if not components:
            raise ValueError("components config is empty; nothing to export to TensorRT.")

        device_id = self._validate_cuda_device(device)
        logger.info("Using CUDA device: %s", device)

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        onnx_dir_str = str(onnx_dir_path)
        num = len(components)
        # Scope the active CUDA device to this export instead of mutating the process-global
        # device via torch.cuda.set_device(); this keeps concurrent/repeat exports isolated.
        with torch.cuda.device(device_id):
            # Start at 1 so progress logs are human-friendly: [1/N] ... [N/N].
            for i, (component_name, comp) in enumerate(components, 1):
                onnx_file = resolve_artifact_path(
                    base_dir=onnx_dir_str,
                    components_cfg=components_cfg,
                    component_name=component_name,
                    file_key="onnx_file",
                )
                trt_path = output_dir_path / comp.engine_file
                trt_path.parent.mkdir(parents=True, exist_ok=True)

                logger.info(
                    "\n[%s/%s] Converting %s → %s...",
                    i,
                    num,
                    Path(onnx_file).name,
                    trt_path.name,
                )

                exporter = TensorRTExporter(config=config.get_tensorrt_settings(component_name))

                artifact = exporter.export(
                    onnx_path=onnx_file,
                    output_path=str(trt_path),
                )
                logger.info("TensorRT engine saved: %s", artifact.path)

        logger.info("\nAll TensorRT engines exported successfully to %s", output_dir_path)
        return Artifact(path=str(output_dir_path))

    def _validate_cuda_device(self, device: DeviceSpec) -> int:
        """Ensure device is CUDA and return the device index.

        Args:
            device: CUDA device specification.

        Returns:
            The integer device index.

        Raises:
            ValueError: If device is not CUDA.
        """
        if not device.is_cuda:
            raise ValueError(f"TensorRT export requires CUDA device, got: {device}")
        return device.index
