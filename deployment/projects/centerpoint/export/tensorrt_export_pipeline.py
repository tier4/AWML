"""
CenterPoint TensorRT export pipeline using composition.

Consumes a directory of ONNX files (from the CenterPoint ONNX export) and
converts each to a TensorRT engine, writing engines according to the
components config (e.g. engine_file paths).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch

from deployment.configs import BaseDeploymentConfig, ComponentsConfig
from deployment.core.artifacts import Artifact
from deployment.core.device import DeviceSpec
from deployment.exporters.common.factory import ExporterFactory
from deployment.exporters.export_pipelines.base import TensorRTExportPipeline


class CenterPointTensorRTExportPipeline(TensorRTExportPipeline):
    """TensorRT export pipeline for CenterPoint.

    Consumes a directory of ONNX files (multi-file export) and builds a TensorRT
    engine per component into ``output_dir``.
    """

    def __init__(
        self,
        exporter_factory: type[ExporterFactory],
        components_cfg: ComponentsConfig,
        logger: Optional[logging.Logger] = None,
    ):
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

    def export(
        self,
        *,
        onnx_path: str,
        output_dir: str,
        config: BaseDeploymentConfig,
        device: DeviceSpec,
    ) -> Artifact:
        """Convert all ONNX files in onnx_path to TensorRT engines in output_dir.

        Discovers *.onnx files in the given directory, maps each to a component
        and engine path from config, and builds one TensorRT engine per file.

        Args:
            onnx_path: Directory containing ONNX files (multi-file export).
            output_dir: Directory where TensorRT engine files are written.
            config: Deployment config for TensorRT exporter options.
            device: CUDA device for building engines.

        Returns:
            Artifact whose path is the output directory.

        Raises:
            ValueError: If device or onnx_path is invalid or onnx_path is not a directory.
            FileNotFoundError: If no ONNX files are found in onnx_path.
            KeyError: If an ONNX file stem is not declared in components config.
        """
        if device is None:
            raise ValueError("CUDA device must be provided for TensorRT export")
        if onnx_path is None:
            raise ValueError("onnx_path must be provided for CenterPoint TensorRT export")

        onnx_dir_path = Path(onnx_path)
        if not onnx_dir_path.is_dir():
            raise ValueError(f"onnx_path must be a directory for multi-file export, got: {onnx_path}")

        device_id = self._validate_cuda_device(device)
        torch.cuda.set_device(device_id)
        self.logger.info(f"Using CUDA device: {device}")

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        onnx_files = self._discover_onnx_files(onnx_dir_path)
        if not onnx_files:
            raise FileNotFoundError(f"No ONNX files found in {onnx_dir_path}")

        engine_file_map = self._build_engine_file_map()
        onnx_stem_to_component = self._build_onnx_stem_to_component_map()

        num_files = len(onnx_files)
        for i, onnx_file in enumerate(onnx_files, 1):
            onnx_stem = onnx_file.stem
            if onnx_stem not in engine_file_map:
                raise KeyError(f"ONNX file '{onnx_file.name}' is not declared in deploy config components.*.onnx_file")
            engine_file = engine_file_map[onnx_stem]
            trt_path = output_dir_path / engine_file
            trt_path.parent.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"\n[{i}/{num_files}] Converting {onnx_file.name} → {trt_path.name}...")

            component_name = onnx_stem_to_component[onnx_stem]
            exporter = self.exporter_factory.create_tensorrt_exporter(
                config=config,
                logger=self.logger,
                component_name=component_name,
            )

            artifact = exporter.export(
                model=None,
                sample_input=None,
                output_path=str(trt_path),
                onnx_path=str(onnx_file),
            )
            self.logger.info(f"TensorRT engine saved: {artifact.path}")

        self.logger.info(f"\nAll TensorRT engines exported successfully to {output_dir_path}")
        return Artifact(path=str(output_dir_path))

    def _discover_onnx_files(self, onnx_dir: Path) -> List[Path]:
        """Return sorted list of .onnx file paths in the given directory."""
        return sorted(
            (path for path in onnx_dir.iterdir() if path.is_file() and path.suffix.lower() == ".onnx"),
            key=lambda p: p.name,
        )

    def _build_engine_file_map(self) -> Dict[str, str]:
        """Build mapping from ONNX file stem to engine_file path from ComponentsConfig.

        Returns:
            Dict mapping each component's ONNX stem (e.g. 'pts_voxel_encoder') to
            its configured engine_file name/path.
        """
        mapping: Dict[str, str] = {}
        for name, comp in self._components_cfg.items():
            mapping[Path(comp.onnx_file).stem] = comp.engine_file
        return mapping

    def _build_onnx_stem_to_component_map(self) -> Dict[str, str]:
        """Build mapping from ONNX file stem to component name from ComponentsConfig.

        Returns:
            Dict mapping each ONNX stem (e.g. 'pts_voxel_encoder') to its
            component name in the config.
        """
        return {
            Path(component_cfg.onnx_file).stem: component_name
            for component_name, component_cfg in self._components_cfg.items()
        }
