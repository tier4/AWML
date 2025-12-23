"""
Unified deployment runner for common deployment workflows.

Project-agnostic runtime runner that orchestrates:
- Export (PyTorch -> ONNX -> TensorRT)
- Verification (scenario-based comparisons)
- Evaluation (metrics/latency across backends)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Type

from mmengine.config import Config

from deployment.core import BaseDataLoader, BaseDeploymentConfig, BaseEvaluator
from deployment.core.contexts import ExportContext
from deployment.exporters.common.model_wrappers import BaseModelWrapper
from deployment.exporters.export_pipelines.base import OnnxExportPipeline, TensorRTExportPipeline
from deployment.runtime.artifact_manager import ArtifactManager
from deployment.runtime.evaluation_orchestrator import EvaluationOrchestrator
from deployment.runtime.export_orchestrator import ExportOrchestrator
from deployment.runtime.verification_orchestrator import VerificationOrchestrator


@dataclass
class DeploymentResult:
    """Standardized structure returned by `BaseDeploymentRunner.run()`."""

    pytorch_model: Optional[Any] = None
    onnx_path: Optional[str] = None
    tensorrt_path: Optional[str] = None
    verification_results: Dict[str, Any] = field(default_factory=dict)
    evaluation_results: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseDeploymentRunner:
    """Base deployment runner for common deployment pipelines."""

    def __init__(
        self,
        data_loader: BaseDataLoader,
        evaluator: BaseEvaluator,
        config: BaseDeploymentConfig,
        model_cfg: Config,
        logger: logging.Logger,
        onnx_wrapper_cls: Optional[Type[BaseModelWrapper]] = None,
        onnx_pipeline: Optional[OnnxExportPipeline] = None,
        tensorrt_pipeline: Optional[TensorRTExportPipeline] = None,
    ):
        self.data_loader = data_loader
        self.evaluator = evaluator
        self.config = config
        self.model_cfg = model_cfg
        self.logger = logger

        self._onnx_wrapper_cls = onnx_wrapper_cls
        self._onnx_pipeline = onnx_pipeline
        self._tensorrt_pipeline = tensorrt_pipeline

        self.artifact_manager = ArtifactManager(config, logger)

        self._export_orchestrator: Optional[ExportOrchestrator] = None
        self.verification_orchestrator = VerificationOrchestrator(config, evaluator, data_loader, logger)
        self.evaluation_orchestrator = EvaluationOrchestrator(config, evaluator, data_loader, logger)

    @property
    def export_orchestrator(self) -> ExportOrchestrator:
        if self._export_orchestrator is None:
            self._export_orchestrator = ExportOrchestrator(
                config=self.config,
                data_loader=self.data_loader,
                artifact_manager=self.artifact_manager,
                logger=self.logger,
                model_loader=self.load_pytorch_model,
                evaluator=self.evaluator,
                onnx_wrapper_cls=self._onnx_wrapper_cls,
                onnx_pipeline=self._onnx_pipeline,
                tensorrt_pipeline=self._tensorrt_pipeline,
            )
        return self._export_orchestrator

    def load_pytorch_model(self, checkpoint_path: str, context: ExportContext) -> Any:
        raise NotImplementedError(f"{self.__class__.__name__}.load_pytorch_model() must be implemented by subclasses.")

    def run(self, context: Optional[ExportContext] = None) -> DeploymentResult:
        if context is None:
            context = ExportContext()

        results = DeploymentResult()

        export_result = self.export_orchestrator.run(context)
        results.pytorch_model = export_result.pytorch_model
        results.onnx_path = export_result.onnx_path
        results.tensorrt_path = export_result.tensorrt_path

        results.verification_results = self.verification_orchestrator.run(artifact_manager=self.artifact_manager)
        results.evaluation_results = self.evaluation_orchestrator.run(self.artifact_manager)

        self.logger.info("\n" + "=" * 80)
        self.logger.info("Deployment Complete!")
        self.logger.info("=" * 80)

        return results
