"""
Unified deployment runner for common deployment workflows.

Project-agnostic runtime runner that orchestrates:
- Export (PyTorch -> ONNX -> TensorRT)
- Verification (scenario-based comparisons)
- Evaluation (metrics/latency across backends)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type

from mmengine.config import Config

from deployment.config.base import BaseDeploymentConfig
from deployment.evaluation.backend_executor import BackendExecutor
from deployment.evaluation.backend_verifier import BackendVerifier
from deployment.evaluation.base_evaluator import BaseEvaluator
from deployment.evaluation.output_comparator import OutputComparator
from deployment.export.contexts import ExportContext
from deployment.export.exporters.model_wrappers import BaseModelWrapper
from deployment.export.pipelines.component_builder import DefaultComponentBuilder
from deployment.export.pipelines.onnx_export_pipeline import OnnxExportPipeline
from deployment.export.pipelines.sample_extractor import DefaultSampleExtractor
from deployment.export.pipelines.tensorrt_export_pipeline import TensorRTExportPipeline
from deployment.io.base_data_loader import BaseDataLoader
from deployment.runtime.artifact_manager import ArtifactManager
from deployment.runtime.evaluation_orchestrator import EvaluationOrchestrator
from deployment.runtime.export_orchestrator import ExportOrchestrator
from deployment.runtime.verification_orchestrator import VerificationOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class DeploymentResult:
    """Standardized structure returned by `BaseDeploymentRunner.run()`."""

    pytorch_model: Optional[Any] = None
    onnx_path: Optional[str] = None
    tensorrt_path: Optional[str] = None
    verification_results: Dict[str, Any] = field(default_factory=dict)
    evaluation_results: Dict[str, Any] = field(default_factory=dict)


class BaseDeploymentRunner:
    """Base deployment runner for common deployment pipelines."""

    def __init__(
        self,
        data_loader: BaseDataLoader,
        evaluator: BaseEvaluator,
        executor: BackendExecutor,
        config: BaseDeploymentConfig,
        model_cfg: Config,
        onnx_wrapper_cls: Optional[Type[BaseModelWrapper]] = None,
        onnx_pipeline: Optional[OnnxExportPipeline] = None,
        tensorrt_pipeline: Optional[TensorRTExportPipeline] = None,
    ) -> None:
        self.data_loader = data_loader
        self.evaluator = evaluator
        self._executor = executor
        self.config = config
        self.model_cfg = model_cfg

        self.artifact_manager = ArtifactManager(config)

        # Default to the model-agnostic whole-model export (one ONNX/engine per config
        # component) when a project does not supply its own pipeline. Projects needing
        # model-specific decomposition pass an explicit onnx_pipeline; the whole-model
        # builder requires a single-component config and a wrapper class to build.
        if onnx_pipeline is None and onnx_wrapper_cls is not None:
            onnx_pipeline = OnnxExportPipeline(
                sample_extractor=DefaultSampleExtractor(),
                component_builder=DefaultComponentBuilder(config.components_cfg),
                onnx_wrapper_cls=onnx_wrapper_cls,
            )
        if tensorrt_pipeline is None:
            tensorrt_pipeline = TensorRTExportPipeline()

        self.export_orchestrator = ExportOrchestrator(
            config=config,
            data_loader=data_loader,
            artifact_manager=self.artifact_manager,
            model_loader=self.load_pytorch_model,
            onnx_pipeline=onnx_pipeline,
            tensorrt_pipeline=tensorrt_pipeline,
        )
        comparator = OutputComparator(output_names=executor.get_output_names())
        verifier = BackendVerifier(executor, comparator)
        self.verification_orchestrator = VerificationOrchestrator(config, verifier, data_loader, self.artifact_manager)
        self.evaluation_orchestrator = EvaluationOrchestrator(config, evaluator, data_loader, self.artifact_manager)

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

        # Hand the loaded reference model to the executor shared by verification and evaluation.
        self._executor.set_pytorch_model(export_result.pytorch_model)

        results.verification_results = self.verification_orchestrator.run()
        results.evaluation_results = self.evaluation_orchestrator.run()

        logger.info("\n" + "=" * 80)
        logger.info("Deployment Complete!")
        logger.info("=" * 80)

        return results
