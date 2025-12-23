"""
Export orchestration for deployment workflows.

This module handles all model export logic (PyTorch loading, ONNX export, TensorRT export)
in a unified orchestrator, keeping the deployment runner thin.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Type

import torch

from deployment.core.artifacts import Artifact
from deployment.core.backend import Backend
from deployment.core.config.base_config import BaseDeploymentConfig
from deployment.core.contexts import ExportContext
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.exporters.common.factory import ExporterFactory
from deployment.exporters.common.model_wrappers import BaseModelWrapper
from deployment.exporters.common.onnx_exporter import ONNXExporter
from deployment.exporters.common.tensorrt_exporter import TensorRTExporter
from deployment.exporters.export_pipelines.base import OnnxExportPipeline, TensorRTExportPipeline
from deployment.runtime.artifact_manager import ArtifactManager


@dataclass
class ExportResult:
    """Result of export orchestration."""

    pytorch_model: Optional[Any] = None
    onnx_path: Optional[str] = None
    tensorrt_path: Optional[str] = None


class ExportOrchestrator:
    """Orchestrates model export workflows (PyTorch loading, ONNX, TensorRT)."""

    ONNX_DIR_NAME = "onnx"
    TENSORRT_DIR_NAME = "tensorrt"
    DEFAULT_ENGINE_FILENAME = "model.engine"

    def __init__(
        self,
        config: BaseDeploymentConfig,
        data_loader: BaseDataLoader,
        artifact_manager: ArtifactManager,
        logger: logging.Logger,
        model_loader: Callable[..., Any],
        evaluator: Any,
        onnx_wrapper_cls: Optional[Type[BaseModelWrapper]] = None,
        onnx_pipeline: Optional[OnnxExportPipeline] = None,
        tensorrt_pipeline: Optional[TensorRTExportPipeline] = None,
    ):
        self.config = config
        self.data_loader = data_loader
        self.artifact_manager = artifact_manager
        self.logger = logger
        self._model_loader = model_loader
        self._evaluator = evaluator
        self._onnx_wrapper_cls = onnx_wrapper_cls
        self._onnx_pipeline = onnx_pipeline
        self._tensorrt_pipeline = tensorrt_pipeline

        self._onnx_exporter: Optional[ONNXExporter] = None
        self._tensorrt_exporter: Optional[TensorRTExporter] = None

    def run(self, context: Optional[ExportContext] = None) -> ExportResult:
        if context is None:
            context = ExportContext()

        result = ExportResult()

        should_export_onnx = self.config.export_config.should_export_onnx()
        should_export_trt = self.config.export_config.should_export_tensorrt()
        checkpoint_path = self.config.checkpoint_path
        external_onnx_path = self.config.export_config.onnx_path

        requires_pytorch = self._determine_pytorch_requirements()

        pytorch_model = None
        if requires_pytorch:
            pytorch_model, success = self._ensure_pytorch_model_loaded(pytorch_model, checkpoint_path, context, result)
            if not success:
                return result

        if should_export_onnx:
            result.onnx_path = self._run_onnx_export(pytorch_model, context)

        if should_export_trt:
            onnx_path = self._resolve_onnx_path_for_trt(result.onnx_path, external_onnx_path)
            if not onnx_path:
                return result
            result.onnx_path = onnx_path
            self._register_external_onnx_artifact(onnx_path)
            result.tensorrt_path = self._run_tensorrt_export(onnx_path, context)

        self._resolve_external_artifacts(result)
        return result

    def _determine_pytorch_requirements(self) -> bool:
        if self.config.export_config.should_export_onnx():
            return True

        eval_config = self.config.evaluation_config
        if eval_config.enabled:
            backends_cfg = eval_config.backends
            pytorch_cfg = backends_cfg.get(Backend.PYTORCH.value) or backends_cfg.get(Backend.PYTORCH, {})
            if pytorch_cfg and pytorch_cfg.get("enabled", False):
                return True

        verification_cfg = self.config.verification_config
        if verification_cfg.enabled:
            export_mode = self.config.export_config.mode
            scenarios = self.config.get_verification_scenarios(export_mode)
            if scenarios and any(
                policy.ref_backend is Backend.PYTORCH or policy.test_backend is Backend.PYTORCH for policy in scenarios
            ):
                return True

        return False

    def _load_and_register_pytorch_model(self, checkpoint_path: str, context: ExportContext) -> Optional[Any]:
        if not checkpoint_path:
            self.logger.error(
                "Checkpoint required but not provided. Please set export.checkpoint_path in config or pass via CLI."
            )
            return None

        self.logger.info("\nLoading PyTorch model...")
        try:
            pytorch_model = self._model_loader(checkpoint_path, context)
            self.artifact_manager.register_artifact(Backend.PYTORCH, Artifact(path=checkpoint_path))

            if hasattr(self._evaluator, "set_pytorch_model"):
                self._evaluator.set_pytorch_model(pytorch_model)
                self.logger.info("Updated evaluator with pre-built PyTorch model via set_pytorch_model()")

            return pytorch_model
        except Exception as e:
            self.logger.error(f"Failed to load PyTorch model: {e}")
            return None

    def _ensure_pytorch_model_loaded(
        self,
        pytorch_model: Optional[Any],
        checkpoint_path: str,
        context: ExportContext,
        result: ExportResult,
    ) -> tuple[Optional[Any], bool]:
        if pytorch_model is not None:
            return pytorch_model, True

        if not checkpoint_path:
            self.logger.error("PyTorch model required but checkpoint_path not provided.")
            return None, False

        pytorch_model = self._load_and_register_pytorch_model(checkpoint_path, context)
        if pytorch_model is None:
            self.logger.error("Failed to load PyTorch model; aborting export.")
            return None, False

        result.pytorch_model = pytorch_model
        return pytorch_model, True

    def _run_onnx_export(self, pytorch_model: Any, context: ExportContext) -> Optional[str]:
        onnx_artifact = self._export_onnx(pytorch_model, context)
        if onnx_artifact:
            return onnx_artifact.path
        self.logger.error("ONNX export requested but no artifact was produced.")
        return None

    def _resolve_onnx_path_for_trt(
        self, exported_onnx_path: Optional[str], external_onnx_path: Optional[str]
    ) -> Optional[str]:
        onnx_path = exported_onnx_path or external_onnx_path
        if not onnx_path:
            self.logger.error(
                "TensorRT export requires an ONNX path. "
                "Please set export.onnx_path in config or enable ONNX export."
            )
            return None
        return onnx_path

    def _register_external_onnx_artifact(self, onnx_path: str) -> None:
        if not os.path.exists(onnx_path):
            return
        multi_file = os.path.isdir(onnx_path)
        self.artifact_manager.register_artifact(Backend.ONNX, Artifact(path=onnx_path, multi_file=multi_file))

    def _run_tensorrt_export(self, onnx_path: str, context: ExportContext) -> Optional[str]:
        trt_artifact = self._export_tensorrt(onnx_path, context)
        if trt_artifact:
            return trt_artifact.path
        self.logger.error("TensorRT export requested but no artifact was produced.")
        return None

    def _export_onnx(self, pytorch_model: Any, context: ExportContext) -> Optional[Artifact]:
        if not self.config.export_config.should_export_onnx():
            return None

        if self._onnx_pipeline is None and self._onnx_wrapper_cls is None:
            raise RuntimeError("ONNX export requested but no wrapper class or export pipeline provided.")

        onnx_settings = self.config.get_onnx_settings()
        sample_idx = context.sample_idx if context.sample_idx != 0 else self.config.runtime_config.sample_idx

        onnx_dir = os.path.join(self.config.export_config.work_dir, self.ONNX_DIR_NAME)
        os.makedirs(onnx_dir, exist_ok=True)
        output_path = os.path.join(onnx_dir, onnx_settings.save_file)

        if self._onnx_pipeline is not None:
            self.logger.info("=" * 80)
            self.logger.info(f"Exporting to ONNX via pipeline ({type(self._onnx_pipeline).__name__})")
            self.logger.info("=" * 80)
            artifact = self._onnx_pipeline.export(
                model=pytorch_model,
                data_loader=self.data_loader,
                output_dir=onnx_dir,
                config=self.config,
                sample_idx=sample_idx,
            )
            self.artifact_manager.register_artifact(Backend.ONNX, artifact)
            self.logger.info(f"ONNX export successful: {artifact.path}")
            return artifact

        exporter = self._get_onnx_exporter()
        self.logger.info("=" * 80)
        self.logger.info(f"Exporting to ONNX (Using {type(exporter).__name__})")
        self.logger.info("=" * 80)

        sample = self.data_loader.load_sample(sample_idx)
        single_input = self.data_loader.preprocess(sample)

        batch_size = onnx_settings.batch_size
        if batch_size is None:
            input_tensor = single_input
            self.logger.info("Using dynamic batch size")
        else:
            if isinstance(single_input, (list, tuple)):
                input_tensor = tuple(
                    inp.repeat(batch_size, *([1] * (len(inp.shape) - 1))) if len(inp.shape) > 0 else inp
                    for inp in single_input
                )
            else:
                input_tensor = single_input.repeat(batch_size, *([1] * (len(single_input.shape) - 1)))
            self.logger.info(f"Using fixed batch size: {batch_size}")

        exporter.export(pytorch_model, input_tensor, output_path)

        multi_file = bool(self.config.onnx_config.get("multi_file", False))
        artifact_path = onnx_dir if multi_file else output_path
        artifact = Artifact(path=artifact_path, multi_file=multi_file)
        self.artifact_manager.register_artifact(Backend.ONNX, artifact)
        self.logger.info(f"ONNX export successful: {artifact.path}")
        return artifact

    def _export_tensorrt(self, onnx_path: str, context: ExportContext) -> Optional[Artifact]:
        if not self.config.export_config.should_export_tensorrt():
            return None

        if not onnx_path:
            self.logger.warning("ONNX path not available, skipping TensorRT export")
            return None

        exporter_label = None if self._tensorrt_pipeline else type(self._get_tensorrt_exporter()).__name__
        self.logger.info("=" * 80)
        if self._tensorrt_pipeline:
            self.logger.info(f"Exporting to TensorRT via pipeline ({type(self._tensorrt_pipeline).__name__})")
        else:
            self.logger.info(f"Exporting to TensorRT (Using {exporter_label})")
        self.logger.info("=" * 80)

        tensorrt_dir = os.path.join(self.config.export_config.work_dir, self.TENSORRT_DIR_NAME)
        os.makedirs(tensorrt_dir, exist_ok=True)
        output_path = self._get_tensorrt_output_path(onnx_path, tensorrt_dir)

        cuda_device = self.config.devices.cuda
        device_id = self.config.devices.get_cuda_device_index()
        if cuda_device is None or device_id is None:
            raise RuntimeError("TensorRT export requires a CUDA device. Set deploy_cfg.devices['cuda'].")
        torch.cuda.set_device(device_id)
        self.logger.info(f"Using CUDA device for TensorRT export: {cuda_device}")

        sample_idx = context.sample_idx if context.sample_idx != 0 else self.config.runtime_config.sample_idx
        sample_input = self.data_loader.get_shape_sample(sample_idx)

        if self._tensorrt_pipeline is not None:
            artifact = self._tensorrt_pipeline.export(
                onnx_path=onnx_path,
                output_dir=tensorrt_dir,
                config=self.config,
                device=cuda_device,
                data_loader=self.data_loader,
            )
            self.artifact_manager.register_artifact(Backend.TENSORRT, artifact)
            self.logger.info(f"TensorRT export successful: {artifact.path}")
            return artifact

        exporter = self._get_tensorrt_exporter()
        artifact = exporter.export(
            model=None,
            sample_input=sample_input,
            output_path=output_path,
            onnx_path=onnx_path,
        )
        self.artifact_manager.register_artifact(Backend.TENSORRT, artifact)
        self.logger.info(f"TensorRT export successful: {artifact.path}")
        return artifact

    def _get_onnx_exporter(self) -> ONNXExporter:
        if self._onnx_exporter is None:
            if self._onnx_wrapper_cls is None:
                raise RuntimeError("ONNX wrapper class not provided. Cannot create ONNX exporter.")
            self._onnx_exporter = ExporterFactory.create_onnx_exporter(
                config=self.config,
                wrapper_cls=self._onnx_wrapper_cls,
                logger=self.logger,
            )
        return self._onnx_exporter

    def _get_tensorrt_exporter(self) -> TensorRTExporter:
        if self._tensorrt_exporter is None:
            self._tensorrt_exporter = ExporterFactory.create_tensorrt_exporter(
                config=self.config,
                logger=self.logger,
            )
        return self._tensorrt_exporter

    def _get_tensorrt_output_path(self, onnx_path: str, tensorrt_dir: str) -> str:
        if os.path.isdir(onnx_path):
            return os.path.join(tensorrt_dir, self.DEFAULT_ENGINE_FILENAME)
        onnx_filename = os.path.basename(onnx_path)
        engine_filename = onnx_filename.replace(".onnx", ".engine")
        return os.path.join(tensorrt_dir, engine_filename)

    def _resolve_external_artifacts(self, result: ExportResult) -> None:
        if not result.onnx_path:
            self._resolve_and_register_artifact(Backend.ONNX, result, "onnx_path")

        if not result.tensorrt_path:
            self._resolve_and_register_artifact(Backend.TENSORRT, result, "tensorrt_path")

    def _resolve_and_register_artifact(self, backend: Backend, result: ExportResult, attr_name: str) -> None:
        eval_models = self.config.evaluation_config.models
        artifact_path = self._get_backend_entry(eval_models, backend)

        if artifact_path and os.path.exists(artifact_path):
            setattr(result, attr_name, artifact_path)
            multi_file = os.path.isdir(artifact_path)
            self.artifact_manager.register_artifact(backend, Artifact(path=artifact_path, multi_file=multi_file))
        elif artifact_path:
            self.logger.warning(f"{backend.value} file from config does not exist: {artifact_path}")

    @staticmethod
    def _get_backend_entry(mapping: Optional[Mapping[Any, Any]], backend: Backend) -> Any:
        if not mapping:
            return None
        if backend.value in mapping:
            return mapping[backend.value]
        return mapping.get(backend)
