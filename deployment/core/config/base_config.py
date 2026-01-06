"""
Base configuration classes for deployment framework.

This module provides the foundation for task-agnostic deployment configuration.
Task-specific deployment configs should extend BaseDeploymentConfig.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

import torch
from mmengine.config import Config

from deployment.core.backend import Backend
from deployment.exporters.common.configs import (
    ONNXExportConfig,
    TensorRTExportConfig,
    TensorRTModelInputConfig,
)

# Constants
DEFAULT_WORKSPACE_SIZE = 1 << 30  # 1 GB


def _empty_mapping() -> Mapping[Any, Any]:
    """Return an immutable empty mapping."""
    return MappingProxyType({})


class PrecisionPolicy(str, Enum):
    """Precision policy options for TensorRT."""

    AUTO = "auto"
    FP16 = "fp16"
    FP32_TF32 = "fp32_tf32"
    STRONGLY_TYPED = "strongly_typed"


class ExportMode(str, Enum):
    """Export pipeline modes."""

    ONNX = "onnx"
    TRT = "trt"
    BOTH = "both"
    NONE = "none"

    @classmethod
    def from_value(cls, value: Optional[Union[str, ExportMode]]) -> ExportMode:
        """Parse strings or enum members into ExportMode (defaults to BOTH)."""
        if value is None:
            return cls.BOTH
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            for member in cls:
                if member.value == normalized:
                    return member
        raise ValueError(f"Invalid export mode '{value}'. Must be one of {[m.value for m in cls]}.")


# Precision policy mapping for TensorRT
PRECISION_POLICIES = {
    PrecisionPolicy.AUTO.value: {},  # No special flags, TensorRT decides
    PrecisionPolicy.FP16.value: {"FP16": True},
    PrecisionPolicy.FP32_TF32.value: {"TF32": True},  # TF32 for FP32 operations
    PrecisionPolicy.STRONGLY_TYPED.value: {"STRONGLY_TYPED": True},  # Network creation flag
}


@dataclass(frozen=True)
class ExportConfig:
    """Configuration for model export settings."""

    mode: ExportMode = ExportMode.BOTH
    work_dir: str = "work_dirs"
    onnx_path: Optional[str] = None

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> ExportConfig:
        """Create ExportConfig from dict."""
        return cls(
            mode=ExportMode.from_value(config_dict.get("mode", ExportMode.BOTH)),
            work_dir=config_dict.get("work_dir", cls.work_dir),
            onnx_path=config_dict.get("onnx_path"),
        )

    def should_export_onnx(self) -> bool:
        """Check if ONNX export is requested."""
        return self.mode in (ExportMode.ONNX, ExportMode.BOTH)

    def should_export_tensorrt(self) -> bool:
        """Check if TensorRT export is requested."""
        return self.mode in (ExportMode.TRT, ExportMode.BOTH)


@dataclass(frozen=True)
class DeviceConfig:
    """Normalized device settings shared across deployment stages."""

    cpu: str = "cpu"
    cuda: Optional[str] = "cuda:0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "cpu", self._normalize_cpu(self.cpu))
        object.__setattr__(self, "cuda", self._normalize_cuda(self.cuda))

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> DeviceConfig:
        """Create DeviceConfig from dict."""
        return cls(cpu=config_dict.get("cpu", cls.cpu), cuda=config_dict.get("cuda", cls.cuda))

    @staticmethod
    def _normalize_cpu(device: Optional[str]) -> str:
        """Normalize CPU device string."""
        if not device:
            return "cpu"
        normalized = str(device).strip().lower()
        if normalized.startswith("cuda"):
            raise ValueError("CPU device cannot be a CUDA device")
        return normalized

    @staticmethod
    def _normalize_cuda(device: Optional[str]) -> Optional[str]:
        """Normalize CUDA device string to 'cuda:N' format."""
        if device is None:
            return None
        if not isinstance(device, str):
            raise ValueError("cuda device must be a string (e.g., 'cuda:0')")
        normalized = device.strip().lower()
        if normalized == "":
            return None
        if normalized == "cuda":
            normalized = "cuda:0"
        if not normalized.startswith("cuda"):
            raise ValueError(f"Invalid CUDA device '{device}'. Must start with 'cuda'")
        suffix = normalized.split(":", 1)[1] if ":" in normalized else "0"
        suffix = suffix.strip() or "0"
        if not suffix.isdigit():
            raise ValueError(f"Invalid CUDA device index in '{device}'")
        device_id = int(suffix)
        if device_id < 0:
            raise ValueError("CUDA device index must be non-negative")
        return f"cuda:{device_id}"

    def get_cuda_device_index(self) -> Optional[int]:
        """Return CUDA device index as integer (if configured)."""
        if self.cuda is None:
            return None
        return int(self.cuda.split(":", 1)[1])


@dataclass(frozen=True)
class RuntimeConfig:
    """Configuration for runtime I/O settings."""

    info_file: str = ""
    sample_idx: int = 0

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> RuntimeConfig:
        """Create RuntimeConfig from dictionary."""
        return cls(
            info_file=config_dict.get("info_file", ""),
            sample_idx=config_dict.get("sample_idx", 0),
        )


@dataclass(frozen=True)
class TensorRTConfig:
    """
    Configuration for TensorRT backend-specific settings.

    Note:
        The deploy config key for this section is **`tensorrt_config`**.
    """

    common_config: Mapping[str, Any] = field(default_factory=_empty_mapping)
    model_inputs: Tuple[TensorRTModelInputConfig, ...] = field(default_factory=tuple)
    components: Mapping[str, Mapping[str, Any]] = field(default_factory=_empty_mapping)

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> "TensorRTConfig":
        common_config = dict(config_dict.get("common_config", {}))
        model_inputs_raw: Iterable[Mapping[str, Any]] = config_dict.get("model_inputs", []) or []
        model_inputs: Tuple[TensorRTModelInputConfig, ...] = tuple(
            TensorRTModelInputConfig.from_dict(item) for item in model_inputs_raw
        )
        components_raw = dict(config_dict.get("components", {}) or {})
        components_frozen = {k: MappingProxyType(dict(v or {})) for k, v in components_raw.items()}
        return cls(
            common_config=MappingProxyType(common_config),
            model_inputs=model_inputs,
            components=MappingProxyType(components_frozen),
        )

    def get_precision_policy(self) -> str:
        """Get precision policy name."""
        return self.common_config.get("precision_policy", PrecisionPolicy.AUTO.value)

    def get_precision_flags(self) -> Mapping[str, bool]:
        """Get TensorRT precision flags for the configured policy."""
        policy = self.get_precision_policy()
        return PRECISION_POLICIES.get(policy, {})

    def get_max_workspace_size(self) -> int:
        """Get maximum workspace size for TensorRT."""
        return self.common_config.get("max_workspace_size", DEFAULT_WORKSPACE_SIZE)


@dataclass(frozen=True)
class EvaluationConfig:
    """Typed configuration for evaluation settings."""

    enabled: bool = False
    num_samples: int = 10
    verbose: bool = False
    backends: Mapping[Any, Mapping[str, Any]] = field(default_factory=_empty_mapping)
    models: Mapping[Any, Any] = field(default_factory=_empty_mapping)
    devices: Mapping[str, str] = field(default_factory=_empty_mapping)

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> EvaluationConfig:
        backends_raw = config_dict.get("backends", {}) or {}
        backends_frozen = {key: MappingProxyType(dict(value)) for key, value in backends_raw.items()}

        return cls(
            enabled=config_dict.get("enabled", False),
            num_samples=config_dict.get("num_samples", 10),
            verbose=config_dict.get("verbose", False),
            backends=MappingProxyType(backends_frozen),
            models=MappingProxyType(dict(config_dict.get("models", {}))),
            devices=MappingProxyType(dict(config_dict.get("devices", {}))),
        )


@dataclass(frozen=True)
class VerificationConfig:
    """Typed configuration for verification settings."""

    enabled: bool = True
    num_verify_samples: int = 3
    tolerance: float = 0.1
    devices: Mapping[str, str] = field(default_factory=_empty_mapping)
    scenarios: Mapping[ExportMode, Tuple[VerificationScenario, ...]] = field(default_factory=_empty_mapping)

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> VerificationConfig:
        scenarios_raw = config_dict.get("scenarios", {}) or {}
        scenario_map: Dict[ExportMode, Tuple[VerificationScenario, ...]] = {}
        for mode_key, scenario_list in scenarios_raw.items():
            mode = ExportMode.from_value(mode_key)
            scenario_entries = tuple(VerificationScenario.from_dict(entry) for entry in (scenario_list or []))
            scenario_map[mode] = scenario_entries

        return cls(
            enabled=config_dict.get("enabled", True),
            num_verify_samples=config_dict.get("num_verify_samples", 3),
            tolerance=config_dict.get("tolerance", 0.1),
            devices=MappingProxyType(dict(config_dict.get("devices", {}))),
            scenarios=MappingProxyType(scenario_map),
        )

    def get_scenarios(self, mode: ExportMode) -> Tuple[VerificationScenario, ...]:
        """Return scenarios for a specific export mode."""
        return self.scenarios.get(mode, ())


@dataclass(frozen=True)
class VerificationScenario:
    """Immutable verification scenario specification."""

    ref_backend: Backend
    ref_device: str
    test_backend: Backend
    test_device: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VerificationScenario:
        missing_keys = {"ref_backend", "ref_device", "test_backend", "test_device"} - data.keys()
        if missing_keys:
            raise ValueError(f"Verification scenario missing keys: {missing_keys}")

        return cls(
            ref_backend=Backend.from_value(data["ref_backend"]),
            ref_device=str(data["ref_device"]),
            test_backend=Backend.from_value(data["test_backend"]),
            test_device=str(data["test_device"]),
        )


class BaseDeploymentConfig:
    """
    Base configuration container for deployment settings.

    This class provides a task-agnostic interface for deployment configuration.
    Task-specific configs should extend this class and add task-specific settings.

    Attributes:
        checkpoint_path: Single source of truth for the PyTorch checkpoint path.
                        Used by both export (for ONNX conversion) and evaluation
                        (for PyTorch backend). Defined at top-level of deploy config.
    """

    def __init__(self, deploy_cfg: Config):
        """
        Initialize deployment configuration.

        Args:
            deploy_cfg: MMEngine Config object containing deployment settings
        """
        self.deploy_cfg = deploy_cfg
        self._validate_config()

        self._checkpoint_path: Optional[str] = deploy_cfg.get("checkpoint_path")
        self._device_config = DeviceConfig.from_dict(deploy_cfg.get("devices", {}) or {})

        # Initialize config sections
        self.export_config = ExportConfig.from_dict(deploy_cfg.get("export", {}))
        self.runtime_config = RuntimeConfig.from_dict(deploy_cfg.get("runtime_io", {}))
        self.tensorrt_config = TensorRTConfig.from_dict(deploy_cfg.get("tensorrt_config", {}) or {})
        self._evaluation_config = EvaluationConfig.from_dict(deploy_cfg.get("evaluation", {}))
        self._verification_config = VerificationConfig.from_dict(deploy_cfg.get("verification", {}))

        self._validate_cuda_device()

    def _validate_config(self) -> None:
        """Validate configuration structure and required fields."""
        # Validate required sections
        if "export" not in self.deploy_cfg:
            raise ValueError(
                "Missing 'export' section in deploy config. " "Please update your config to include 'export' section."
            )

        # Validate export mode
        try:
            ExportMode.from_value(self.deploy_cfg.get("export", {}).get("mode", ExportMode.BOTH))
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

        # Validate precision policy if present
        tensorrt_config = self.deploy_cfg.get("tensorrt_config", {})
        common_cfg = tensorrt_config.get("common_config", {})
        precision_policy = common_cfg.get("precision_policy", PrecisionPolicy.AUTO.value)
        if precision_policy not in PRECISION_POLICIES:
            raise ValueError(
                f"Invalid precision_policy '{precision_policy}'. " f"Must be one of {list(PRECISION_POLICIES.keys())}"
            )

    def _validate_cuda_device(self) -> None:
        """Validate CUDA device availability once at config stage."""
        if not self._needs_cuda_device():
            return

        cuda_device = self.devices.cuda
        device_idx = self.devices.get_cuda_device_index()

        if cuda_device is None or device_idx is None:
            raise RuntimeError(
                "CUDA device is required (TensorRT export/verification/evaluation enabled) but no CUDA device was"
                " configured in deploy_cfg.devices."
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device is required (TensorRT export/verification/evaluation enabled) "
                "but torch.cuda.is_available() returned False."
            )

        device_count = torch.cuda.device_count()
        if device_idx >= device_count:
            raise ValueError(
                f"Requested CUDA device '{cuda_device}' but only {device_count} CUDA device(s) are available."
            )

    def _needs_cuda_device(self) -> bool:
        """Determine if current deployment config requires a CUDA device."""
        if self.export_config.should_export_tensorrt():
            return True

        evaluation_cfg = self.evaluation_config
        backends_cfg = evaluation_cfg.backends
        tensorrt_backend = backends_cfg.get(Backend.TENSORRT.value) or backends_cfg.get(Backend.TENSORRT, {})
        if tensorrt_backend and tensorrt_backend.get("enabled", False):
            return True

        verification_cfg = self.verification_config

        for scenario_list in verification_cfg.scenarios.values():
            for scenario in scenario_list:
                if Backend.TENSORRT in (scenario.ref_backend, scenario.test_backend):
                    return True

        return False

    @property
    def checkpoint_path(self) -> Optional[str]:
        """
        Get checkpoint path - single source of truth for PyTorch model.

        This path is used by:
        - Export pipeline: to load the PyTorch model for ONNX conversion
        - Evaluation: for PyTorch backend evaluation
        - Verification: when PyTorch is used as reference or test backend

        Returns:
            Path to the PyTorch checkpoint file, or None if not configured
        """
        return self._checkpoint_path

    @property
    def evaluation_config(self) -> EvaluationConfig:
        """Get evaluation configuration."""
        return self._evaluation_config

    @property
    def onnx_config(self) -> Mapping[str, Any]:
        """Get ONNX configuration."""
        return self.deploy_cfg.get("onnx_config", {})

    @property
    def verification_config(self) -> VerificationConfig:
        """Get verification configuration."""
        return self._verification_config

    @property
    def devices(self) -> DeviceConfig:
        """Get normalized device settings."""
        return self._device_config

    def get_evaluation_backends(self) -> Mapping[Any, Mapping[str, Any]]:
        """
        Get evaluation backends configuration.

        Returns:
            Dictionary mapping backend names to their configuration
        """
        return self.evaluation_config.backends

    def get_verification_scenarios(self, export_mode: ExportMode) -> Tuple[VerificationScenario, ...]:
        """
        Get verification scenarios for the given export mode.

        Args:
            export_mode: Export mode (`ExportMode`)

        Returns:
            Tuple of verification scenarios
        """
        return self.verification_config.get_scenarios(export_mode)

    @property
    def task_type(self) -> Optional[str]:
        """Get task type for pipeline building."""
        return self.deploy_cfg.get("task_type")

    def get_onnx_settings(self) -> ONNXExportConfig:
        """
        Get ONNX export settings.

        Returns:
            ONNXExportConfig instance containing ONNX export parameters
        """
        onnx_config = self.onnx_config
        model_io = self.deploy_cfg.get("model_io", {})

        # Get batch size and dynamic axes from model_io
        batch_size = model_io.get("batch_size", None)
        dynamic_axes = model_io.get("dynamic_axes", None)

        # If batch_size is set to a number, disable dynamic_axes
        if batch_size is not None and isinstance(batch_size, int):
            dynamic_axes = None

        # Handle multiple inputs and outputs
        input_names = [model_io.get("input_name", "input")]
        output_names = [model_io.get("output_name", "output")]

        # Add additional inputs if specified
        additional_inputs = model_io.get("additional_inputs", [])
        for additional_input in additional_inputs:
            if isinstance(additional_input, dict):
                input_names.append(additional_input.get("name", "input"))

        # Add additional outputs if specified
        additional_outputs = model_io.get("additional_outputs", [])
        for additional_output in additional_outputs:
            if isinstance(additional_output, str):
                output_names.append(additional_output)

        settings_dict = {
            "opset_version": onnx_config.get("opset_version", 16),
            "do_constant_folding": onnx_config.get("do_constant_folding", True),
            "input_names": tuple(input_names),
            "output_names": tuple(output_names),
            "dynamic_axes": dynamic_axes,
            "export_params": onnx_config.get("export_params", True),
            "keep_initializers_as_inputs": onnx_config.get("keep_initializers_as_inputs", False),
            "verbose": onnx_config.get("verbose", False),
            "save_file": onnx_config.get("save_file", "model.onnx"),
            "batch_size": batch_size,
        }

        # Note: simplify is typically True by default, but can be overridden
        if "simplify" in onnx_config:
            settings_dict["simplify"] = onnx_config["simplify"]

        return ONNXExportConfig.from_mapping(settings_dict)

    def get_tensorrt_settings(self) -> TensorRTExportConfig:
        """
        Get TensorRT export settings with precision policy support.

        Returns:
            TensorRTExportConfig instance containing TensorRT export parameters
        """
        settings_dict = {
            "max_workspace_size": self.tensorrt_config.get_max_workspace_size(),
            "precision_policy": self.tensorrt_config.get_precision_policy(),
            "policy_flags": self.tensorrt_config.get_precision_flags(),
            "model_inputs": self.tensorrt_config.model_inputs,
        }
        return TensorRTExportConfig.from_mapping(settings_dict)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    logging.basicConfig(level=getattr(logging, level), format="%(levelname)s:%(name)s:%(message)s")
    return logging.getLogger("deployment")


def parse_base_args(parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    """
    Create argument parser with common deployment arguments.

    Args:
        parser: Optional existing ArgumentParser to add arguments to

    Returns:
        ArgumentParser with deployment arguments
    """
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Deploy model to ONNX/TensorRT",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

    parser.add_argument("deploy_cfg", help="Deploy config path")
    parser.add_argument("model_cfg", help="Model config path")
    # Optional overrides
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    return parser
