"""Core components for deployment framework."""

from deployment.core.artifacts import Artifact
from deployment.core.backend import Backend
from deployment.core.config.base_config import (
    BackendConfig,
    BaseDeploymentConfig,
    DeviceConfig,
    EvaluationConfig,
    ExportConfig,
    ExportMode,
    RuntimeConfig,
    VerificationConfig,
    VerificationScenario,
    parse_base_args,
    setup_logging,
)
from deployment.core.contexts import (
    CalibrationExportContext,
    CenterPointExportContext,
    ExportContext,
    YOLOXExportContext,
)
from deployment.core.evaluation.base_evaluator import (
    EVALUATION_DEFAULTS,
    BaseEvaluator,
    EvalResultDict,
    EvaluationDefaults,
    ModelSpec,
    TaskProfile,
    VerifyResultDict,
)
from deployment.core.evaluation.verification_mixin import VerificationMixin
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.core.io.preprocessing_builder import build_preprocessing_pipeline
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
    # Backend and configuration
    "Backend",
    # Typed contexts
    "ExportContext",
    "YOLOXExportContext",
    "CenterPointExportContext",
    "CalibrationExportContext",
    "BaseDeploymentConfig",
    "ExportConfig",
    "ExportMode",
    "RuntimeConfig",
    "BackendConfig",
    "DeviceConfig",
    "EvaluationConfig",
    "VerificationConfig",
    "VerificationScenario",
    "setup_logging",
    "parse_base_args",
    # Constants
    "EVALUATION_DEFAULTS",
    "EvaluationDefaults",
    # Data loading
    "BaseDataLoader",
    # Evaluation
    "BaseEvaluator",
    "TaskProfile",
    "EvalResultDict",
    "VerifyResultDict",
    "VerificationMixin",
    # Artifacts
    "Artifact",
    "ModelSpec",
    # Preprocessing
    "build_preprocessing_pipeline",
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
