"""
Export orchestration for deployment workflows.

This module handles all model export logic (PyTorch loading, ONNX export, TensorRT export)
in a unified orchestrator, keeping the deployment runner thin.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from deployment.config.base import BaseDeploymentConfig
from deployment.config.enums import Backend
from deployment.export.contexts import ExportContext
from deployment.export.pipelines.onnx_export_pipeline import OnnxExportPipeline
from deployment.export.pipelines.tensorrt_export_pipeline import TensorRTExportPipeline
from deployment.io.base_data_loader import BaseDataLoader
from deployment.primitives.artifacts import Artifact
from deployment.runtime.artifact_manager import ArtifactManager

logger = logging.getLogger(__name__)


@dataclass
class ExportResult:
    """
    Result of the export orchestration.

    Attributes:
        pytorch_model: Loaded PyTorch model (if loaded)
        onnx_path: Path to exported ONNX artifact
        tensorrt_path: Path to exported TensorRT engine
    """

    pytorch_model: Optional[Any] = None
    onnx_path: Optional[str] = None
    tensorrt_path: Optional[str] = None


class ExportOrchestrator:
    """
    Orchestrates model export workflows (PyTorch loading, ONNX, TensorRT).

    This class centralizes all export-related logic:
    - Loading PyTorch from checkpoint_path (required for this deployment stack)
    - ONNX / TensorRT export (pipeline or per-component) and artifact registration

    By extracting this logic from the runner, the runner becomes a thin
    orchestrator that coordinates Export, Verification, and Evaluation.
    """

    ONNX_DIR_NAME = "onnx"
    TENSORRT_DIR_NAME = "tensorrt"

    def __init__(
        self,
        config: BaseDeploymentConfig,
        data_loader: BaseDataLoader,
        artifact_manager: ArtifactManager,
        model_loader: Callable[..., Any],
        onnx_pipeline: Optional[OnnxExportPipeline] = None,
        tensorrt_pipeline: Optional[TensorRTExportPipeline] = None,
    ) -> None:
        """
        Initialize export orchestrator.

        Args:
            config: Deployment configuration
            data_loader: Data loader for loading samples
            artifact_manager: Artifact manager for resolving model paths
            model_loader: Model loader for loading PyTorch model
            onnx_pipeline: ONNX export pipeline (required when ONNX export is requested)
            tensorrt_pipeline: TensorRT export pipeline (required when TensorRT export is requested)
        """
        self.config = config
        self.data_loader = data_loader
        self.artifact_manager = artifact_manager
        self._model_loader = model_loader
        self._onnx_pipeline = onnx_pipeline
        self._tensorrt_pipeline = tensorrt_pipeline

    def run(self, context: Optional[ExportContext] = None) -> ExportResult:
        """
        Execute the complete export workflow.

        This method:
        1. Loads PyTorch model from checkpoint_path
        2. Exports to ONNX if configured
        3. Exports to TensorRT if configured
        4. Resolves external artifact paths

        Args:
            context: Typed export context with parameters. If None, a default
                     ExportContext is created.

        Returns:
            ExportResult containing model and artifact paths
        """
        if context is None:
            context = ExportContext()

        result = ExportResult()

        should_export_onnx = self.config.export_config.should_export_onnx
        should_export_trt = self.config.export_config.should_export_tensorrt
        external_onnx_path = self.config.export_config.onnx_path

        pytorch_model = self._load_and_register_pytorch_model(self.config.checkpoint_path, context)
        result.pytorch_model = pytorch_model

        if should_export_onnx:
            result.onnx_path = self._run_onnx_export(pytorch_model)
            if not result.onnx_path:
                # ONNX export was explicitly requested for this run but produced nothing.
                # Failing here is critical: otherwise the TensorRT stage below would silently
                # fall back to `export.onnx_path` (often the same dir) and build an engine from
                # a STALE ONNX left by a previous run, yielding an engine that does not match
                # the current checkpoint with no error surfaced.
                raise RuntimeError(
                    "ONNX export was requested (export.mode includes ONNX) but no ONNX artifact "
                    "was produced. Refusing to continue, as TensorRT export would otherwise reuse "
                    "a stale ONNX file. Check the ONNX export logs above."
                )

        if should_export_trt:
            # When this run also produced ONNX, reuse that fresh path (guaranteed present by the
            # raise above). In trt-only mode fall back to the externally configured ONNX path.
            onnx_path = result.onnx_path if should_export_onnx else external_onnx_path
            if not onnx_path:
                raise RuntimeError(
                    "TensorRT export requires an ONNX path but none is available. "
                    "Set export.onnx_path in config or enable ONNX export (export.mode)."
                )
            result.onnx_path = onnx_path
            self._register_external_onnx_artifact(onnx_path)
            result.tensorrt_path = self._run_tensorrt_export(onnx_path)

        self._resolve_external_artifacts(result)
        return result

    def _load_and_register_pytorch_model(self, checkpoint_path: str, context: ExportContext) -> Any:
        """
        Load and register a PyTorch model from checkpoint.

        Args:
            checkpoint_path: Path to the PyTorch checkpoint
            context: Export context with sample index
        Returns:
            Loaded PyTorch model
        Raises:
            RuntimeError: If the checkpoint cannot be loaded.
        """
        logger.info("\nLoading PyTorch model...")
        try:
            pytorch_model = self._model_loader(checkpoint_path, context)
            self.artifact_manager.register_artifact(Backend.PYTORCH, Artifact(path=checkpoint_path))
            return pytorch_model
        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model from '{checkpoint_path}': {e}") from e

    def _run_onnx_export(self, pytorch_model: Any) -> Optional[str]:
        """
        Run the ONNX export workflow.

        Args:
            pytorch_model: PyTorch model to export
        Returns:
            Path to the exported ONNX artifact or None if export failed
        """
        onnx_artifact = self._export_onnx(pytorch_model)
        if onnx_artifact:
            return onnx_artifact.path
        logger.error("ONNX export requested but no artifact was produced.")
        return None

    def _register_external_onnx_artifact(self, onnx_path: str) -> None:
        """
        Register an external ONNX artifact.

        Args:
            onnx_path: Path to the ONNX artifact
        """
        if not Path(onnx_path).exists():
            return
        self.artifact_manager.register_artifact(Backend.ONNX, Artifact(path=onnx_path))

    def _run_tensorrt_export(self, onnx_path: str) -> Optional[str]:
        """
        Run the TensorRT export workflow.

        Args:
            onnx_path: Path to the ONNX artifact
        Returns:
            Path to the exported TensorRT engine or None if export failed
        """
        trt_artifact = self._export_tensorrt(onnx_path)
        if trt_artifact:
            return trt_artifact.path
        logger.error("TensorRT export requested but no artifact was produced.")
        return None

    def _export_onnx(self, pytorch_model: Any) -> Optional[Artifact]:
        """
        Export a PyTorch model to ONNX via the configured ONNX pipeline.

        Args:
            pytorch_model: PyTorch model to export
        Returns:
            Artifact representing the exported ONNX model
        """
        if self._onnx_pipeline is None:
            raise RuntimeError("ONNX export requested but no ONNX export pipeline was provided.")

        sample_idx = self.config.export_config.sample_idx
        onnx_dir = Path(self.config.export_config.work_dir) / self.ONNX_DIR_NAME
        onnx_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("Exporting to ONNX via pipeline (%s)", type(self._onnx_pipeline).__name__)
        logger.info("=" * 80)
        artifact = self._onnx_pipeline.export(
            model=pytorch_model,
            data_loader=self.data_loader,
            output_dir=str(onnx_dir),
            config=self.config,
            sample_idx=sample_idx,
        )
        self.artifact_manager.register_artifact(Backend.ONNX, artifact)
        logger.info("ONNX export successful: %s", artifact.path)
        return artifact

    def _export_tensorrt(self, onnx_path: str) -> Optional[Artifact]:
        """
        Export an ONNX model to TensorRT via the configured TensorRT pipeline.

        Device scoping is the pipeline's responsibility (it receives the CUDA
        ``device`` and isolates the active device for the build).

        Args:
            onnx_path: Path to the ONNX artifact
        Returns:
            Artifact representing the exported TensorRT engine
        """
        if self._tensorrt_pipeline is None:
            raise RuntimeError("TensorRT export requested but no TensorRT export pipeline was provided.")

        cuda_device = self.config.device_config.cuda
        if cuda_device is None:
            raise RuntimeError("TensorRT export requires a CUDA device. Set deploy_cfg.devices['cuda'].")

        tensorrt_dir = Path(self.config.export_config.work_dir) / self.TENSORRT_DIR_NAME
        tensorrt_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("Exporting to TensorRT via pipeline (%s)", type(self._tensorrt_pipeline).__name__)
        logger.info("=" * 80)
        logger.info("Using CUDA device for TensorRT export: %s", cuda_device)

        artifact = self._tensorrt_pipeline.export(
            onnx_path=onnx_path,
            output_dir=str(tensorrt_dir),
            config=self.config,
            device=cuda_device,
        )
        self.artifact_manager.register_artifact(Backend.TENSORRT, artifact)
        logger.info("TensorRT export successful: %s", artifact.path)
        return artifact

    def _resolve_external_artifacts(self, result: ExportResult) -> None:
        """
        Fill in artifact paths not produced by this run from configured fallbacks.

        Config-based artifact lookup is delegated to ``ArtifactManager.resolve_artifact`` so
        that the resolution rules (registered artifacts, then ``evaluation.backends``, then
        per-backend fallbacks) live in exactly one place rather than being duplicated here.

        Args:
            result: Export result object to store the artifacts
        """
        if not result.onnx_path:
            result.onnx_path = self._resolve_configured_artifact(Backend.ONNX)

        if not result.tensorrt_path:
            result.tensorrt_path = self._resolve_configured_artifact(Backend.TENSORRT)

    def _resolve_configured_artifact(self, backend: Backend) -> Optional[str]:
        """
        Resolve a backend artifact path from configuration, if one exists on disk.

        Args:
            backend: Backend to resolve the artifact for
        Returns:
            The artifact path if it is configured and exists, otherwise None.
        """
        artifact, exists = self.artifact_manager.resolve_artifact(backend)
        if artifact and exists:
            return artifact.path
        if artifact:
            logger.warning("%s artifact path from config does not exist: %s", backend.value, artifact.path)
        return None
