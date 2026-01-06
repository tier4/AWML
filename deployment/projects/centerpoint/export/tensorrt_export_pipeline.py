"""
CenterPoint TensorRT export pipeline using composition.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

import torch

from deployment.core import Artifact, BaseDataLoader, BaseDeploymentConfig
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
        config: BaseDeploymentConfig,
        logger: Optional[logging.Logger] = None,
    ):
        self.exporter_factory = exporter_factory
        self.config = config
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
        data_loader: BaseDataLoader,
    ) -> Artifact:
        onnx_dir = onnx_path

        if device is None:
            raise ValueError("CUDA device must be provided for TensorRT export")
        if onnx_dir is None:
            raise ValueError("onnx_dir must be provided for CenterPoint TensorRT export")

        onnx_dir_path = Path(onnx_dir)
        if not onnx_dir_path.is_dir():
            raise ValueError(f"onnx_path must be a directory for multi-file export, got: {onnx_dir}")

        device_id = self._validate_cuda_device(device)
        torch.cuda.set_device(device_id)
        self.logger.info(f"Using CUDA device: {device}")

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        onnx_files = self._discover_onnx_files(onnx_dir_path)
        if not onnx_files:
            raise FileNotFoundError(f"No ONNX files found in {onnx_dir_path}")

        num_files = len(onnx_files)
        for i, onnx_file in enumerate(onnx_files, 1):
            trt_path = output_dir_path / f"{onnx_file.stem}.engine"

            self.logger.info(f"\n[{i}/{num_files}] Converting {onnx_file.name} → {trt_path.name}...")
            exporter = self._build_tensorrt_exporter()

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

    def _build_tensorrt_exporter(self):
        return self.exporter_factory.create_tensorrt_exporter(config=self.config, logger=self.logger)
