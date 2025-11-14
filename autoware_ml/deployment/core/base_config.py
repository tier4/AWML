"""
Base configuration classes for deployment framework.
This module provides the foundation for task-agnostic deployment configuration.
Task-specific deployment configs should extend BaseDeploymentConfig.
"""

import argparse
import logging
from typing import Any, Dict, List, Optional

from mmengine.config import Config

# Constants
DEFAULT_VERIFICATION_TOLERANCE = 1e-3
DEFAULT_WORKSPACE_SIZE = 1 << 30  # 1 GB

# Precision policy mapping for TensorRT
PRECISION_POLICIES = {
    "auto": {},  # No special flags, TensorRT decides
    "fp16": {"FP16": True},
    "fp32_tf32": {"TF32": True},  # TF32 for FP32 operations
    "explicit_int8": {"INT8": True},
    "strongly_typed": {"STRONGLY_TYPED": True},  # Network creation flag
}


class ExportConfig:
    """Configuration for model export settings."""

    def __init__(self, config_dict: Dict[str, Any]):
        self.mode = config_dict.get("mode", "both")
        # Note: verify has been moved to verification.enabled in v2 config format
        # Device is optional in v2 format (devices are specified per-backend in evaluation/verification)
        # Default to cuda:0 for backward compatibility
        self.device = config_dict.get("device", "cuda:0")
        self.work_dir = config_dict.get("work_dir", "work_dirs")
        self.checkpoint_path = config_dict.get("checkpoint_path")
        self.onnx_path = config_dict.get("onnx_path")

    def should_export_onnx(self) -> bool:
        """Check if ONNX export is requested."""
        return self.mode in ["onnx", "both"]

    def should_export_tensorrt(self) -> bool:
        """Check if TensorRT export is requested."""
        return self.mode in ["trt", "both"]


class RuntimeConfig:
    """Configuration for runtime I/O settings."""

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict

    def get(self, key: str, default: Any = None) -> Any:
        """Get a runtime configuration value."""
        return self._config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access to runtime config."""
        return self._config[key]


class BackendConfig:
    """Configuration for backend-specific settings."""

    def __init__(self, config_dict: Dict[str, Any]):
        self.common_config = config_dict.get("common_config", {})
        self.model_inputs = config_dict.get("model_inputs", [])

    def get_precision_policy(self) -> str:
        """Get precision policy name."""
        return self.common_config.get("precision_policy", "auto")

    def get_precision_flags(self) -> Dict[str, bool]:
        """Get TensorRT precision flags for the configured policy."""
        policy = self.get_precision_policy()
        return PRECISION_POLICIES.get(policy, {})

    def get_max_workspace_size(self) -> int:
        """Get maximum workspace size for TensorRT."""
        return self.common_config.get("max_workspace_size", DEFAULT_WORKSPACE_SIZE)


class BaseDeploymentConfig:
    """
    Base configuration container for deployment settings.
    This class provides a task-agnostic interface for deployment configuration.
    Task-specific configs should extend this class and add task-specific settings.
    """

    def __init__(self, deploy_cfg: Config):
        """
        Initialize deployment configuration.
        Args:
            deploy_cfg: MMEngine Config object containing deployment settings
        """
        self.deploy_cfg = deploy_cfg
        self._validate_config()

        # Initialize config sections
        self.export_config = ExportConfig(deploy_cfg.get("export", {}))
        self.runtime_config = RuntimeConfig(deploy_cfg.get("runtime_io", {}))
        self.backend_config = BackendConfig(deploy_cfg.get("backend_config", {}))

    def _validate_config(self) -> None:
        """Validate configuration structure and required fields."""
        # Validate required sections
        if "export" not in self.deploy_cfg:
            raise ValueError(
                "Missing 'export' section in deploy config. " "Please update your config to include 'export' section."
            )

        # Validate export mode
        valid_modes = ["onnx", "trt", "both", "none"]
        mode = self.deploy_cfg.get("export", {}).get("mode", "both")
        if mode not in valid_modes:
            raise ValueError(f"Invalid export mode '{mode}'. Must be one of {valid_modes}")

        # Validate precision policy if present
        backend_cfg = self.deploy_cfg.get("backend_config", {})
        common_cfg = backend_cfg.get("common_config", {})
        precision_policy = common_cfg.get("precision_policy", "auto")
        if precision_policy not in PRECISION_POLICIES:
            raise ValueError(
                f"Invalid precision_policy '{precision_policy}'. " f"Must be one of {list(PRECISION_POLICIES.keys())}"
            )

    @property
    def evaluation_config(self) -> Dict:
        """Get evaluation configuration."""
        return self.deploy_cfg.get("evaluation", {})

    @property
    def onnx_config(self) -> Dict:
        """Get ONNX configuration."""
        return self.deploy_cfg.get("onnx_config", {})

    @property
    def verification_config(self) -> Dict:
        """Get verification configuration."""
        return self.deploy_cfg.get("verification", {})

    def get_evaluation_backends(self) -> Dict[str, Dict[str, Any]]:
        """
        Get evaluation backends configuration.

        Returns:
            Dictionary mapping backend names to their configuration
        """
        eval_config = self.evaluation_config
        return eval_config.get("backends", {})

    def get_verification_scenarios(self, export_mode: str) -> List[Dict[str, str]]:
        """
        Get verification scenarios for the given export mode.

        Args:
            export_mode: Export mode ('onnx', 'trt', 'both', 'none')

        Returns:
            List of verification scenarios dictionaries
        """
        verification_cfg = self.verification_config
        scenarios = verification_cfg.get("scenarios", {})
        return scenarios.get(export_mode, [])

    @property
    def task_type(self) -> Optional[str]:
        """Get task type for pipeline building."""
        return self.deploy_cfg.get("task_type")

    def get_onnx_settings(self) -> Dict[str, Any]:
        """
        Get ONNX export settings.
        Returns:
            Dictionary containing ONNX export parameters
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

        settings = {
            "opset_version": onnx_config.get("opset_version", 16),
            "do_constant_folding": onnx_config.get("do_constant_folding", True),
            "input_names": input_names,
            "output_names": output_names,
            "dynamic_axes": dynamic_axes,
            "export_params": onnx_config.get("export_params", True),
            "keep_initializers_as_inputs": onnx_config.get("keep_initializers_as_inputs", False),
            "save_file": onnx_config.get("save_file", "model.onnx"),
            "decode_in_inference": onnx_config.get("decode_in_inference", True),
            "batch_size": batch_size,
        }

        # Include model_wrapper config if present in onnx_config
        if "model_wrapper" in onnx_config:
            settings["model_wrapper"] = onnx_config["model_wrapper"]

        return settings

    def get_tensorrt_settings(self) -> Dict[str, Any]:
        """
        Get TensorRT export settings with precision policy support.
        Returns:
            Dictionary containing TensorRT export parameters
        """
        return {
            "max_workspace_size": self.backend_config.get_max_workspace_size(),
            "precision_policy": self.backend_config.get_precision_policy(),
            "policy_flags": self.backend_config.get_precision_flags(),
            "model_inputs": self.backend_config.model_inputs,
        }

    def update_batch_size(self, batch_size: int) -> None:
        """
        Update batch size in backend config model_inputs.
        Args:
            batch_size: New batch size to set
        """
        if batch_size is not None:
            # Check if model_inputs already has TensorRT-specific configuration
            existing_model_inputs = self.backend_config.model_inputs

            # If model_inputs is None or empty, generate from model_io
            if existing_model_inputs is None or len(existing_model_inputs) == 0:
                # Get model_io configuration
                model_io = self.deploy_cfg.get("model_io", {})
                input_name = model_io.get("input_name", "input")
                input_shape = model_io.get("input_shape", (3, 960, 960))
                input_dtype = model_io.get("input_dtype", "float32")

                # Create model_inputs list
                model_inputs = []

                # Add primary input
                full_shape = (batch_size,) + input_shape
                model_inputs.append(
                    dict(
                        name=input_name,
                        shape=full_shape,
                        dtype=input_dtype,
                    )
                )

                # Add additional inputs if specified
                additional_inputs = model_io.get("additional_inputs", [])
                for additional_input in additional_inputs:
                    if isinstance(additional_input, dict):
                        add_name = additional_input.get("name", "input")
                        add_shape = additional_input.get("shape", (-1,))
                        add_dtype = additional_input.get("dtype", "float32")

                        # Handle dynamic shapes (e.g., (-1,) for variable length)
                        if isinstance(add_shape, tuple) and len(add_shape) > 0 and add_shape[0] == -1:
                            # Keep dynamic shape for variable length inputs
                            full_add_shape = add_shape
                        else:
                            # Add batch dimension for fixed shapes
                            full_add_shape = (batch_size,) + add_shape

                        model_inputs.append(
                            dict(
                                name=add_name,
                                shape=full_add_shape,
                                dtype=add_dtype,
                            )
                        )

                # Update model_inputs in backend config
                self.backend_config.model_inputs = model_inputs
            else:
                # If model_inputs already exists (e.g., TensorRT shape ranges),
                # update batch size in existing shapes if they are simple shapes
                for model_input in existing_model_inputs:
                    if isinstance(model_input, dict) and "shape" in model_input:
                        # Simple shape format: {"name": "input", "shape": (batch, ...), "dtype": "float32"}
                        if isinstance(model_input["shape"], tuple) and len(model_input["shape"]) > 0:
                            # Update batch dimension (first dimension)
                            shape = list(model_input["shape"])
                            shape[0] = batch_size
                            model_input["shape"] = tuple(shape)
                    elif isinstance(model_input, dict) and "input_shapes" in model_input:
                        # TensorRT shape ranges format: {"input_shapes": {"input": {"min_shape": [...], ...}}}
                        # For TensorRT shape ranges, we don't modify batch size as it's handled by dynamic_axes
                        pass


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
    parser.add_argument(
        "checkpoint", nargs="?", default=None, help="Model checkpoint path (optional when mode='none')"
    )

    # Optional overrides
    parser.add_argument("--work-dir", help="Override output directory from config")
    parser.add_argument("--device", help="Override device from config")
    parser.add_argument("--log-level", default="INFO", choices=list(logging._nameToLevel.keys()), help="Logging level")

    return parser
