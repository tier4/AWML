"""
Base deployment config: single entry point container with runtime validation and helpers.

Torch/CUDA validation lives here. Schema/enums are in configs.schema and configs.enums.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

import torch
from mmengine.config import Config

from deployment.configs.enums import ExportMode
from deployment.configs.schema import (
    ComponentsConfig,
    DeviceConfig,
    EvaluationConfig,
    ExportConfig,
    OnnxConfig,
    RuntimeConfig,
    TensorRTConfig,
    VerificationConfig,
    VerificationScenario,
)
from deployment.core.backend import Backend
from deployment.exporters.common.configs import (
    ONNXExportConfig,
    TensorRTExportConfig,
    TensorRTModelInputConfig,
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
        self._ensure_required_sections()

        self._checkpoint_path = deploy_cfg.get("checkpoint_path")
        self._device_config = DeviceConfig.from_dict(deploy_cfg.get("devices", {}))
        self.components_cfg = ComponentsConfig.from_dict(deploy_cfg.get("components", {}))
        self._onnx_export_config = OnnxConfig.from_dict(deploy_cfg.get("onnx_config"))

        # Schema/type validation in each from_dict and __post_init__
        self.export_config = ExportConfig.from_dict(deploy_cfg.get("export", {}))
        self.runtime_config = RuntimeConfig.from_dict(deploy_cfg.get("runtime_io", {}))
        self.tensorrt_config = TensorRTConfig.from_dict(deploy_cfg.get("tensorrt_config", {}))
        self._evaluation_config = EvaluationConfig.from_dict(deploy_cfg.get("evaluation", {}))
        self._verification_config = VerificationConfig.from_dict(deploy_cfg.get("verification", {}))

        # Runtime/environment validation (torch/cuda)
        self._validate_cuda_device()

    def _ensure_required_sections(self) -> None:
        """Ensure required deploy config sections exist. Schema/type validation is done by typed configs in from_dict/__post_init__."""
        if "export" not in self.deploy_cfg:
            raise ValueError("Missing 'export' section in deploy config.")
        export_raw = self.deploy_cfg.get("export")
        if export_raw is not None and not isinstance(export_raw, Mapping):
            raise TypeError("deploy config 'export' must be a mapping.")

        if "components" not in self.deploy_cfg:
            raise ValueError("Missing 'components' section in deploy config.")
        components_raw = self.deploy_cfg.get("components")
        if components_raw is not None and not isinstance(components_raw, Mapping):
            raise TypeError("deploy config 'components' must be a mapping.")

    def _validate_cuda_device(self) -> None:
        """Validate CUDA device availability once at config stage."""
        if not self._needs_cuda_device():
            return

        cuda_device = self.devices.cuda
        device_idx = self.devices.cuda_device_index

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
        if self.export_config.should_export_tensorrt:
            return True

        evaluation_cfg = self.evaluation_config
        backends_cfg = evaluation_cfg.backends
        tensorrt_backend = backends_cfg.get(Backend.TENSORRT.value, {})
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
    def onnx_config(self) -> OnnxConfig:
        """Get ONNX export configuration (typed)."""
        return self._onnx_export_config

    @property
    def verification_config(self) -> VerificationConfig:
        """Get verification configuration."""
        return self._verification_config

    @property
    def devices(self) -> DeviceConfig:
        """Get normalized device settings."""
        return self._device_config

    @property
    def evaluation_backends(self) -> Mapping[Any, Mapping[str, Any]]:
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

    def get_onnx_settings(self, component_name: str) -> ONNXExportConfig:
        """Get ONNX export settings for a component. I/O and save_file come from ComponentCfg."""
        component_cfg = self.components_cfg.get_component(component_name)
        onnx_config = self._onnx_export_config
        input_names = tuple(inp.name for inp in component_cfg.io.inputs)
        output_names = tuple(out.name for out in component_cfg.io.outputs)
        if not input_names:
            input_names = ("input",)
        if not output_names:
            output_names = ("output",)
        settings_dict = {
            "opset_version": onnx_config.opset_version,
            "do_constant_folding": onnx_config.do_constant_folding,
            "input_names": input_names,
            "output_names": output_names,
            "dynamic_axes": component_cfg.io.dynamic_axes,
            "export_params": onnx_config.export_params,
            "keep_initializers_as_inputs": onnx_config.keep_initializers_as_inputs,
            "verbose": False,
            "save_file": component_cfg.onnx_file,
            "batch_size": None,
            "simplify": onnx_config.simplify,
        }
        return ONNXExportConfig.from_mapping(settings_dict)

    def get_tensorrt_settings(self, component_name: str) -> TensorRTExportConfig:
        """Get TensorRT export settings for a component. Profile and I/O come from ComponentCfg."""
        component_cfg = self.components_cfg.get_component(component_name)
        if not component_cfg.tensorrt_profile:
            return TensorRTExportConfig.from_mapping(
                {
                    "max_workspace_size": self.tensorrt_config.max_workspace_size,
                    "precision_policy": self.tensorrt_config.precision_policy,
                    "policy_flags": self.tensorrt_config.precision_flags,
                    "model_inputs": None,
                }
            )
        input_shapes = dict(component_cfg.tensorrt_profile)
        model_inputs = (TensorRTModelInputConfig(input_shapes=MappingProxyType(input_shapes)),)
        return TensorRTExportConfig.from_mapping(
            {
                "max_workspace_size": self.tensorrt_config.max_workspace_size,
                "precision_policy": self.tensorrt_config.precision_policy,
                "policy_flags": self.tensorrt_config.precision_flags,
                "model_inputs": model_inputs,
            }
        )
