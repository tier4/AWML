"""
CenterPoint TensorRT export pipeline using composition.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

import torch

from deployment.core import Artifact, BaseDeploymentConfig
from deployment.exporters.common.factory import ExporterFactory
from deployment.exporters.export_pipelines.base import TensorRTExportPipeline


class CenterPointTensorRTExportPipeline(TensorRTExportPipeline):
    """TensorRT export pipeline for CenterPoint.

    Consumes a directory of ONNX files (multi-file export) and builds a TensorRT
    engine per component into `output_dir`.
    """

    _CUDA_DEVICE_PATTERN = re.compile(r"^cuda:\d+$")

    def __init__(
        self,
        exporter_factory: type[ExporterFactory],
        logger: Optional[logging.Logger] = None,
    ):
        self.exporter_factory = exporter_factory
        self.logger = logger or logging.getLogger(__name__)

    def _validate_cuda_device(self, device: str) -> int:
        if not self._CUDA_DEVICE_PATTERN.match(device):
            raise ValueError(
                f"Invalid CUDA device format: '{device}'. Expected format: 'cuda:N' (e.g., 'cuda:0', 'cuda:1')"
            )
        return int(device.split(":")[1])

    def export(
        self,
        *,
        onnx_path: str,
        output_dir: str,
        config: BaseDeploymentConfig,
        device: str,
    ) -> Artifact:
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

        engine_file_by_onnx_stem = self._build_engine_file_map(config)
        num_files = len(onnx_files)
        for i, onnx_file in enumerate(onnx_files, 1):
            engine_file = engine_file_by_onnx_stem.get(onnx_file.stem, f"{onnx_file.stem}.engine")
            trt_path = output_dir_path / engine_file
            trt_path.parent.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"\n[{i}/{num_files}] Converting {onnx_file.name} → {trt_path.name}...")
            exporter = self._build_tensorrt_exporter(config)

            artifact = exporter.export(
                model=None,
                sample_input=None,
                output_path=str(trt_path),
                onnx_path=str(onnx_file),
            )
            self.logger.info(f"TensorRT engine saved: {artifact.path}")

        self.logger.info(f"\nAll TensorRT engines exported successfully to {output_dir_path}")
        return Artifact(path=str(output_dir_path), multi_file=True)

    def _discover_onnx_files(self, onnx_dir: Path) -> List[Path]:
        return sorted(
            (path for path in onnx_dir.iterdir() if path.is_file() and path.suffix.lower() == ".onnx"),
            key=lambda p: p.name,
        )

    def _build_engine_file_map(self, config: BaseDeploymentConfig) -> dict[str, str]:
        """
        Build mapping from ONNX stem -> engine_file from config (if present).

        This allows `deploy_cfg.tensorrt_config.components[*].engine_file` to control output naming.
        """
        trt_cfg = getattr(config, "tensorrt_config", None)
        components = dict(getattr(trt_cfg, "components", {}) or {})
        mapping: dict[str, str] = {}
        for comp in components.values():
            onnx_file = comp.get("onnx_file")
            engine_file = comp.get("engine_file")
            if not onnx_file or not engine_file:
                continue
            mapping[Path(onnx_file).stem] = str(engine_file)
        return mapping

    def _build_tensorrt_exporter(self, config: BaseDeploymentConfig):
        return self.exporter_factory.create_tensorrt_exporter(config=config, logger=self.logger)
