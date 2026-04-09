"""
CenterPoint TensorRT export pipeline using composition.

Reads ONNX paths from ``deploy_config`` ``components`` (same rules as
``resolve_artifact_path``) and builds one TensorRT engine per component into
``output_dir``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from typing_extensions import override

from deployment.configs.base import BaseDeploymentConfig
from deployment.configs.schema import ComponentsConfig
from deployment.core.artifacts import Artifact, resolve_artifact_path
from deployment.core.device import DeviceSpec
from deployment.exporters.common.factory import ExporterFactory
from deployment.exporters.export_pipelines.base import TensorRTExportPipeline


class CenterPointTensorRTExportPipeline(TensorRTExportPipeline):
    """TensorRT export pipeline for CenterPoint.

    Iterates ``components`` in deploy config order and builds one engine per
    component from the configured ``onnx_file`` under ``onnx_path``.
    """

    def __init__(
        self,
        exporter_factory: type[ExporterFactory],
        components_cfg: ComponentsConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the pipeline with exporter factory and components config.

        Args:
            exporter_factory: Factory used to create TensorRT exporters per component.
            components_cfg: Config defining component names, onnx_file and engine_file paths.
            logger: Optional logger; defaults to module logger if not provided.
        """
        self.exporter_factory = exporter_factory
        self._components_cfg = components_cfg
        self.logger = logger or logging.getLogger(__name__)

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

    @override
    def export(
        self,
        *,
        onnx_path: str,
        output_dir: str,
        config: BaseDeploymentConfig,
        device: DeviceSpec,
    ) -> Artifact:
        """Convert each component's ONNX to a TensorRT engine under ``output_dir``.

        For every entry in ``components``, resolves ``onnx_file`` under ``onnx_path``
        (must exist) and writes ``engine_file`` relative to ``output_dir``.

        Args:
            onnx_path: Directory containing ONNX files (layout matches deploy config).
            output_dir: Directory where TensorRT engine files are written.
            config: Deployment config for TensorRT exporter options.
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

        components = list(self._components_cfg.items())
        if not components:
            raise ValueError("components config is empty; nothing to export to TensorRT.")

        device_id = self._validate_cuda_device(device)
        torch.cuda.set_device(device_id)
        self.logger.info(f"Using CUDA device: {device}")

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        onnx_dir_str = str(onnx_dir_path)
        num = len(components)
        for i, (component_name, comp) in enumerate(components, 1):
            onnx_file = resolve_artifact_path(
                base_dir=onnx_dir_str,
                components_cfg=self._components_cfg,
                component_name=component_name,
                file_key="onnx_file",
            )
            trt_path = output_dir_path / comp.engine_file
            trt_path.parent.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"\n[{i}/{num}] Converting {Path(onnx_file).name} → {trt_path.name}...")

            exporter = self.exporter_factory.create_tensorrt_exporter(
                config=config,
                logger=self.logger,
                component_name=component_name,
            )

            artifact = exporter.export(
                model=None,
                sample_input=None,
                output_path=str(trt_path),
                onnx_path=onnx_file,
            )
            self.logger.info(f"TensorRT engine saved: {artifact.path}")

        self.logger.info(f"\nAll TensorRT engines exported successfully to {output_dir_path}")
        return Artifact(path=str(output_dir_path))
