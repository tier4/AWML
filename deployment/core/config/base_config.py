"""
Backward-compatible re-exports from deployment.configs and deployment.cli.args.

New code should prefer: from deployment.configs import ... and from deployment.cli.args import ...
"""

from __future__ import annotations

from deployment.cli.args import parse_base_args, setup_logging
from deployment.configs.base import BaseDeploymentConfig
from deployment.configs.enums import (
    DEFAULT_WORKSPACE_SIZE,
    PRECISION_POLICIES,
    ExportMode,
    PrecisionPolicy,
)
from deployment.configs.schema import (
    ComponentCfg,
    ComponentIO,
    ComponentsConfig,
    DeviceConfig,
    EvaluationConfig,
    ExportConfig,
    InputSpec,
    OnnxConfig,
    OutputSpec,
    RuntimeConfig,
    TensorRTConfig,
    VerificationConfig,
    VerificationScenario,
)

__all__ = [
    "BaseDeploymentConfig",
    "ComponentCfg",
    "ComponentIO",
    "ComponentsConfig",
    "DeviceConfig",
    "DEFAULT_WORKSPACE_SIZE",
    "EvaluationConfig",
    "ExportConfig",
    "ExportMode",
    "InputSpec",
    "OnnxConfig",
    "OutputSpec",
    "PRECISION_POLICIES",
    "PrecisionPolicy",
    "RuntimeConfig",
    "TensorRTConfig",
    "VerificationConfig",
    "VerificationScenario",
    "parse_base_args",
    "setup_logging",
]
