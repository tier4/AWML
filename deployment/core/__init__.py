"""Core components for deployment framework."""

from deployment.core.artifacts import (
    Artifact,
    get_component_files,
    resolve_artifact_path,
    resolve_engine_path,
    resolve_onnx_path,
)
from deployment.core.backend import Backend
from deployment.core.contexts import (
    CalibrationExportContext,
    CenterPointExportContext,
    ExportContext,
    YOLOXExportContext,
)
from deployment.core.device import DeviceSpec
from deployment.core.evaluation.base_evaluator import (
    BaseEvaluator,
    EvalResultDict,
    InferenceInput,
    ModelSpec,
    VerifyResultDict,
)
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.core.metrics import (
    BaseMetricsConfig,
    BaseMetricsInterface,
    ClassificationMetricsConfig,
    ClassificationMetricsInterface,
    Detection2DMetricsConfig,
    Detection2DMetricsInterface,
    Detection3DMetricsConfig,
    Detection3DMetricsInterface,
)

__all__ = [
    # Backend
    "Backend",
    "DeviceSpec",
    # Typed contexts
    "ExportContext",
    "YOLOXExportContext",
    "CenterPointExportContext",
    "CalibrationExportContext",
    # Data loading
    "BaseDataLoader",
    # Evaluation
    "BaseEvaluator",
    "InferenceInput",
    "EvalResultDict",
    "VerifyResultDict",
    # Artifacts
    "Artifact",
    "resolve_artifact_path",
    "resolve_onnx_path",
    "resolve_engine_path",
    "get_component_files",
    "ModelSpec",
    # Metrics interfaces (using autoware_perception_evaluation)
    "BaseMetricsInterface",
    "BaseMetricsConfig",
    "Detection3DMetricsInterface",
    "Detection3DMetricsConfig",
    "Detection2DMetricsInterface",
    "Detection2DMetricsConfig",
    "ClassificationMetricsInterface",
    "ClassificationMetricsConfig",
]
