"""
Unified deployment runner for common deployment workflows.

This module provides a unified runner that handles the common deployment workflow
across different projects, while allowing project-specific customization.

Architecture:
    The runner orchestrates three specialized orchestrators:
    - ExportOrchestrator: Handles PyTorch loading, ONNX export, TensorRT export
    - VerificationOrchestrator: Handles output verification across backends
    - EvaluationOrchestrator: Handles model evaluation with metrics

    This design keeps the runner thin (~150 lines vs original 850+) while
    maintaining flexibility for project-specific customization.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Type

from mmengine.config import Config

from deployment.core import BaseDataLoader, BaseDeploymentConfig, BaseEvaluator
from deployment.core.contexts import ExportContext
from deployment.exporters.common.model_wrappers import BaseModelWrapper
from deployment.exporters.workflows.base import OnnxExportWorkflow, TensorRTExportWorkflow
from deployment.runners.common.artifact_manager import ArtifactManager
from deployment.runners.common.evaluation_orchestrator import EvaluationOrchestrator
from deployment.runners.common.export_orchestrator import ExportOrchestrator
from deployment.runners.common.verification_orchestrator import VerificationOrchestrator


@dataclass
class DeploymentResult:
    """
    Standardized structure returned by `BaseDeploymentRunner.run()`.

    Fields:
        pytorch_model: In-memory model instance loaded from the checkpoint (if requested).
        onnx_path: Filesystem path to the exported ONNX artifact (single file or directory).
        tensorrt_path: Filesystem path to the exported TensorRT engine.
        verification_results: Arbitrary dictionary produced by `BaseEvaluator.verify()`.
        evaluation_results: Arbitrary dictionary produced by `BaseEvaluator.evaluate()`.
    """

    pytorch_model: Optional[Any] = None
    onnx_path: Optional[str] = None
    tensorrt_path: Optional[str] = None
    verification_results: Dict[str, Any] = field(default_factory=dict)
    evaluation_results: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a dict view for compatibility/serialization."""
        return asdict(self)


class BaseDeploymentRunner:
    """
    Base deployment runner for common deployment workflows.

    This runner orchestrates three specialized components:
    1. ExportOrchestrator: Load PyTorch, export ONNX, export TensorRT
    2. VerificationOrchestrator: Verify outputs across backends
    3. EvaluationOrchestrator: Evaluate models with metrics

    Projects should extend this class and override methods as needed:
    - Override load_pytorch_model() for project-specific model loading
    - Provide project-specific ONNX/TensorRT workflows via constructor
    """

    def __init__(
        self,
        data_loader: BaseDataLoader,
        evaluator: BaseEvaluator,
        config: BaseDeploymentConfig,
        model_cfg: Config,
        logger: logging.Logger,
        onnx_wrapper_cls: Optional[Type[BaseModelWrapper]] = None,
        onnx_workflow: Optional[OnnxExportWorkflow] = None,
        tensorrt_workflow: Optional[TensorRTExportWorkflow] = None,
    ):
        """
        Initialize base deployment runner.

        Args:
            data_loader: Data loader for samples
            evaluator: Evaluator for model evaluation
            config: Deployment configuration
            model_cfg: Model configuration
            logger: Logger instance
            onnx_wrapper_cls: Optional ONNX model wrapper class for exporter creation
            onnx_workflow: Optional specialized ONNX workflow
            tensorrt_workflow: Optional specialized TensorRT workflow
        """
        self.data_loader = data_loader
        self.evaluator = evaluator
        self.config = config
        self.model_cfg = model_cfg
        self.logger = logger

        # Store workflow references for subclasses to modify
        self._onnx_wrapper_cls = onnx_wrapper_cls
        self._onnx_workflow = onnx_workflow
        self._tensorrt_workflow = tensorrt_workflow

        # Initialize artifact manager (shared across orchestrators)
        self.artifact_manager = ArtifactManager(config, logger)

        # Initialize orchestrators (export orchestrator created lazily to allow subclass workflow setup)
        self._export_orchestrator: Optional[ExportOrchestrator] = None
        self.verification_orchestrator = VerificationOrchestrator(config, evaluator, data_loader, logger)
        self.evaluation_orchestrator = EvaluationOrchestrator(config, evaluator, data_loader, logger)

    @property
    def export_orchestrator(self) -> ExportOrchestrator:
        """
        Get export orchestrator (created lazily to allow subclass workflow setup).

        This allows subclasses to set _onnx_workflow and _tensorrt_workflow in __init__
        before the export orchestrator is created.
        """
        if self._export_orchestrator is None:
            self._export_orchestrator = ExportOrchestrator(
                config=self.config,
                data_loader=self.data_loader,
                artifact_manager=self.artifact_manager,
                logger=self.logger,
                model_loader=self.load_pytorch_model,
                evaluator=self.evaluator,
                onnx_wrapper_cls=self._onnx_wrapper_cls,
                onnx_workflow=self._onnx_workflow,
                tensorrt_workflow=self._tensorrt_workflow,
            )
        return self._export_orchestrator

    def load_pytorch_model(self, checkpoint_path: str, context: ExportContext) -> Any:
        """
        Load PyTorch model from checkpoint.

        Subclasses must implement this method to provide project-specific model loading logic.
        Project-specific parameters should be accessed from the typed context object.

        Args:
            checkpoint_path: Path to checkpoint file
            context: Export context containing project-specific parameters.
                     Use project-specific context subclasses (e.g., YOLOXExportContext,
                     CenterPointExportContext) for type-safe access to parameters.

        Returns:
            Loaded PyTorch model

        Raises:
            NotImplementedError: If not implemented by subclass

        Example:
            # In YOLOXDeploymentRunner:
            def load_pytorch_model(self, checkpoint_path: str, context: ExportContext) -> Any:
                # Type narrow to access YOLOX-specific fields
                if isinstance(context, YOLOXExportContext):
                    model_cfg_path = context.model_cfg_path
                else:
                    model_cfg_path = context.get("model_cfg_path")
                ...
        """
        raise NotImplementedError(f"{self.__class__.__name__}.load_pytorch_model() must be implemented by subclasses.")

    def run(
        self,
        context: Optional[ExportContext] = None,
    ) -> DeploymentResult:
        """
        Execute the complete deployment workflow.

        The workflow consists of three phases:
        1. Export: Load PyTorch model, export to ONNX/TensorRT
        2. Verification: Verify outputs across backends
        3. Evaluation: Evaluate models with metrics

        Args:
            context: Typed export context with parameters. If None, a default
                     ExportContext is created.

        Returns:
            DeploymentResult: Structured summary of all deployment artifacts and reports.
        """
        # Create default context if not provided
        if context is None:
            context = ExportContext()

        results = DeploymentResult()

        # Phase 1: Export
        export_result = self.export_orchestrator.run(context)
        results.pytorch_model = export_result.pytorch_model
        results.onnx_path = export_result.onnx_path
        results.tensorrt_path = export_result.tensorrt_path

        # Phase 2: Verification
        verification_results = self.verification_orchestrator.run(
            artifact_manager=self.artifact_manager,
        )
        results.verification_results = verification_results

        # Phase 3: Evaluation
        evaluation_results = self.evaluation_orchestrator.run(self.artifact_manager)
        results.evaluation_results = evaluation_results

        self.logger.info("\n" + "=" * 80)
        self.logger.info("Deployment Complete!")
        self.logger.info("=" * 80)

        return results
