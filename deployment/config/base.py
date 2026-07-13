"""
Base deployment config: single entry point container with runtime validation and helpers.

Torch/CUDA validation lives here. Schema/enums are in configs.schema and configs.enums.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Optional, Tuple

import torch
from mmengine.config import Config

from deployment.config.enums import Backend, ExportMode
from deployment.config.schema import (
    ComponentsConfig,
    DeviceConfig,
    EvaluationConfig,
    ExportConfig,
    OnnxConfig,
    TensorRTConfig,
    VerificationConfig,
    VerificationScenario,
)
from deployment.export.exporters.configs import (
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

    def __init__(self, deploy_cfg: Config) -> None:
        """
        Initialize deployment configuration.

        Args:
            deploy_cfg: MMEngine Config object containing deployment settings
        """
        checkpoint_path = deploy_cfg.get("checkpoint_path")
        self.checkpoint_path = self._validate_checkpoint_path(checkpoint_path)
        self.device_config = DeviceConfig.from_dict(deploy_cfg.get("devices", {}))
        self.components_cfg = ComponentsConfig.from_dict(deploy_cfg.get("components"))
        self._onnx_config = OnnxConfig.from_dict(deploy_cfg.get("onnx_config"))
        self.export_config = ExportConfig.from_dict(deploy_cfg.get("export"))
        self._tensorrt_config = TensorRTConfig.from_dict(deploy_cfg.get("tensorrt_config", {}))
        self.evaluation_config = EvaluationConfig.from_dict(deploy_cfg.get("evaluation", {}))
        self.verification_config = VerificationConfig.from_dict(deploy_cfg.get("verification", {}))
        self._deploy_log_path = self._parse_deploy_log_path(deploy_cfg.get("deploy_log_path", "deployment.log"))

        # Runtime/environment validation (torch/cuda)
        self._validate_cuda_device()

    @staticmethod
    def _parse_deploy_log_path(raw: Optional[str]) -> Optional[str]:
        """Parse deploy_log_path; None or blank disables file logging."""
        return raw.strip() or None if raw is not None else None

    @staticmethod
    def _validate_checkpoint_path(checkpoint_path: str) -> str:
        """Require a non-empty checkpoint path that exists as a regular file."""
        if not isinstance(checkpoint_path, str):
            raise TypeError(f"checkpoint_path must be a string, got {type(checkpoint_path).__name__}.")
        path = Path(checkpoint_path).expanduser()

        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint file not found: '{checkpoint_path}' (resolved to '{path.resolve()}'). "
                f"Deploy-config paths are relative to the current working directory "
                f"('{Path.cwd()}'); run from the repository root or set an absolute checkpoint_path."
            )

        return str(path)

    def _validate_cuda_device(self) -> None:
        """Validate CUDA device availability once at config stage."""
        if not self._uses_tensorrt():
            return

        cuda_device = self.device_config.cuda
        device_idx = self.device_config.cuda_device_index

        if cuda_device is None or device_idx is None:
            raise RuntimeError(
                "CUDA device is required (TensorRT export/verification/evaluation enabled) but no CUDA device was"
                " configured in devices."
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

    def _uses_tensorrt(self) -> bool:
        """Whether TensorRT is used by any stage (export, evaluation, or verification)."""
        if self.export_config.should_export_tensorrt:
            return True

        if self.evaluation_config.enabled:
            tensorrt_backend = self.evaluation_config.backends.get(Backend.TENSORRT.value)
            if tensorrt_backend is not None and tensorrt_backend.enabled:
                return True

        if self.verification_config.enabled:
            for scenario_list in self.verification_config.scenarios.values():
                for scenario in scenario_list:
                    if Backend.TENSORRT in (scenario.ref_backend, scenario.test_backend):
                        return True

        return False

    @property
    def resolved_deploy_log_file(self) -> Optional[str]:
        """Absolute path for the deployment log file, or None if file logging is disabled."""
        if self._deploy_log_path is None:
            return None
        log_path = Path(self._deploy_log_path).expanduser()
        if log_path.is_absolute():
            return str(log_path.resolve(strict=False))
        work_dir = Path(self.export_config.work_dir).expanduser()
        return str((work_dir / log_path).resolve(strict=False))

    def get_verification_scenarios(self, export_mode: ExportMode) -> Tuple[VerificationScenario, ...]:
        """
        Get verification scenarios for the given export mode.

        Args:
            export_mode: Export mode (`ExportMode`)

        Returns:
            Tuple of verification scenarios
        """
        return self.verification_config.get_scenarios(export_mode)

    def get_onnx_settings(self, component_name: str) -> ONNXExportConfig:
        """Get ONNX export settings for a component. I/O and save_file come from ComponentCfg."""
        component_cfg = self.components_cfg.get_component(component_name)
        onnx_config = self._onnx_config
        input_names = tuple(inp.name for inp in component_cfg.io.inputs)
        output_names = tuple(out.name for out in component_cfg.io.outputs)
        if not input_names:
            input_names = ("input",)
        if not output_names:
            output_names = ("output",)
        return ONNXExportConfig(
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=component_cfg.io.dynamic_axes,
            simplify=onnx_config.simplify,
            opset_version=onnx_config.opset_version,
            export_params=onnx_config.export_params,
            keep_initializers_as_inputs=onnx_config.keep_initializers_as_inputs,
            verbose=False,
            do_constant_folding=onnx_config.do_constant_folding,
            save_file=component_cfg.onnx_file,
            batch_size=None,
        )

    def get_tensorrt_settings(self, component_name: str) -> TensorRTExportConfig:
        """Get TensorRT export settings for a component. Profile and I/O come from ComponentCfg."""
        component_cfg = self.components_cfg.get_component(component_name)

        model_input: Optional[TensorRTModelInputConfig] = None
        if component_cfg.tensorrt_profile:
            input_shapes = MappingProxyType(dict(component_cfg.tensorrt_profile))
            model_input = TensorRTModelInputConfig(input_shapes=input_shapes)

        return TensorRTExportConfig(
            precision_policy=self._tensorrt_config.precision_policy,
            max_workspace_size=self._tensorrt_config.max_workspace_size,
            model_input=model_input,
            plugin_libraries=self._tensorrt_config.plugin_libraries,
        )
