"""
CenterPoint TensorRT export pipeline using composition.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping

import torch

from deployment.core import Artifact, BaseDeploymentConfig
from deployment.exporters.common.configs import TensorRTExportConfig, TensorRTModelInputConfig, TensorRTProfileConfig
from deployment.exporters.common.factory import ExporterFactory
from deployment.exporters.export_pipelines.base import TensorRTExportPipeline


class CenterPointTensorRTExportPipeline(TensorRTExportPipeline):
    """TensorRT export pipeline for CenterPoint.

    Consumes a directory of ONNX files (multi-file export) and builds a TensorRT
    engine per component into `output_dir`.

    """

    _CUDA_DEVICE_PATTERN = re.compile(r"^cuda:\d+$")

    def __init__(
        self,
        exporter_factory: type[ExporterFactory],
        logger: Optional[logging.Logger] = None,
    ):
        self.exporter_factory = exporter_factory
        self.logger = logger or logging.getLogger(__name__)

    def _validate_cuda_device(self, device: str) -> int:
        if not self._CUDA_DEVICE_PATTERN.match(device):
            raise ValueError(
                f"Invalid CUDA device format: '{device}'. Expected format: 'cuda:N' (e.g., 'cuda:0', 'cuda:1')"
            )
        return int(device.split(":")[1])

    def export(
        self,
        *,
        onnx_path: str,
        output_dir: str,
        config: BaseDeploymentConfig,
        device: str,
    ) -> Artifact:
        if device is None:
            raise ValueError("CUDA device must be provided for TensorRT export")
        if onnx_path is None:
            raise ValueError("onnx_path must be provided for CenterPoint TensorRT export")

        onnx_dir_path = Path(onnx_path)
        if not onnx_dir_path.is_dir():
            raise ValueError(f"onnx_path must be a directory for multi-file export, got: {onnx_path}")

        device_id = self._validate_cuda_device(device)
        torch.cuda.set_device(device_id)
        self.logger.info(f"Using CUDA device: {device}")

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        onnx_files = self._discover_onnx_files(onnx_dir_path)
        if not onnx_files:
            raise FileNotFoundError(f"No ONNX files found in {onnx_dir_path}")

        components_cfg = self._get_components_cfg(config)
        engine_file_map = self._build_engine_file_map(components_cfg)
        onnx_stem_to_component = self._build_onnx_stem_to_component_map(components_cfg)

        num_files = len(onnx_files)
        for i, onnx_file in enumerate(onnx_files, 1):
            onnx_stem = onnx_file.stem
            if onnx_stem not in engine_file_map:
                raise KeyError(f"ONNX file '{onnx_file.name}' is not declared in deploy config components.*.onnx_file")
            engine_file = engine_file_map[onnx_stem]
            trt_path = output_dir_path / engine_file
            trt_path.parent.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"\n[{i}/{num_files}] Converting {onnx_file.name} → {trt_path.name}...")

            # Get component-specific TensorRT profile
            component_name = onnx_stem_to_component[onnx_stem]
            exporter = self._build_tensorrt_exporter_for_component(config, components_cfg, component_name)

            artifact = exporter.export(
                model=None,
                sample_input=None,
                output_path=str(trt_path),
                onnx_path=str(onnx_file),
            )
            self.logger.info(f"TensorRT engine saved: {artifact.path}")

        self.logger.info(f"\nAll TensorRT engines exported successfully to {output_dir_path}")
        return Artifact(path=str(output_dir_path), multi_file=True)

    def _discover_onnx_files(self, onnx_dir: Path) -> List[Path]:
        return sorted(
            (path for path in onnx_dir.iterdir() if path.is_file() and path.suffix.lower() == ".onnx"),
            key=lambda p: p.name,
        )

    def _build_engine_file_map(self, components_cfg: Mapping[str, Any]) -> Dict[str, str]:
        """Build mapping from ONNX stem -> engine_file."""
        mapping: Dict[str, str] = {}
        for comp_name, comp in components_cfg.items():
            if not isinstance(comp, Mapping):
                raise TypeError(f"components['{comp_name}'] must be a mapping, got {type(comp).__name__}")
            if "onnx_file" not in comp or "engine_file" not in comp:
                raise KeyError(f"components['{comp_name}'] must define both 'onnx_file' and 'engine_file'.")
            onnx_file = comp["onnx_file"]
            engine_file = comp["engine_file"]
            mapping[Path(onnx_file).stem] = str(engine_file)
        return mapping

    def _build_onnx_stem_to_component_map(self, components_cfg: Mapping[str, Any]) -> Dict[str, str]:
        """Build mapping from ONNX stem -> component name."""
        mapping: Dict[str, str] = {}
        for comp_name, comp_cfg in components_cfg.items():
            if not isinstance(comp_cfg, Mapping):
                raise TypeError(f"components['{comp_name}'] must be a mapping, got {type(comp_cfg).__name__}")
            if "onnx_file" not in comp_cfg:
                raise KeyError(f"components['{comp_name}'] must define 'onnx_file'.")
            onnx_file = comp_cfg["onnx_file"]
            mapping[Path(onnx_file).stem] = comp_name
        return mapping

    def _get_components_cfg(self, config: BaseDeploymentConfig) -> Mapping[str, Any]:
        """Get unified components configuration from deploy config."""
        if "components" not in config.deploy_cfg:
            raise KeyError("deploy_cfg must define 'components' for CenterPoint TensorRT export.")
        return dict(config.deploy_cfg["components"])

    def _build_tensorrt_exporter_for_component(
        self,
        config: BaseDeploymentConfig,
        components_cfg: Mapping[str, Any],
        component_name: str,
    ):
        """Build TensorRT exporter with component-specific profile.

        Converts the unified `tensorrt_profile` from the component config
        into the `model_inputs` format expected by TensorRTExporter.
        """
        # Get base TensorRT settings from config
        if "tensorrt_config" not in config.deploy_cfg:
            raise KeyError("deploy_cfg must define 'tensorrt_config' for TensorRT export.")
        trt_cfg = config.deploy_cfg["tensorrt_config"]
        precision_policy = trt_cfg["precision_policy"]
        max_workspace_size = trt_cfg["max_workspace_size"]

        # Build model_inputs from component's tensorrt_profile
        model_inputs = ()
        if component_name not in components_cfg:
            raise KeyError(f"Component '{component_name}' missing from deploy config.")

        comp_cfg = components_cfg[component_name]
        if "tensorrt_profile" not in comp_cfg:
            raise KeyError(f"components['{component_name}'] must define 'tensorrt_profile'.")
        tensorrt_profile = comp_cfg["tensorrt_profile"]
        if not isinstance(tensorrt_profile, Mapping):
            raise TypeError(
                f"components['{component_name}']['tensorrt_profile'] must be a mapping, "
                f"got {type(tensorrt_profile).__name__}"
            )
        if not tensorrt_profile:
            raise ValueError(f"components['{component_name}']['tensorrt_profile'] must not be empty.")

        # Convert tensorrt_profile to TensorRTModelInputConfig format
        input_shapes = {}
        for input_name, shape_cfg in tensorrt_profile.items():
            if not isinstance(shape_cfg, Mapping):
                raise TypeError(
                    f"Profile for input '{input_name}' in component '{component_name}' must be a mapping, "
                    f"got {type(shape_cfg).__name__}"
                )
            input_shapes[input_name] = TensorRTProfileConfig(
                min_shape=tuple(shape_cfg["min_shape"]),
                opt_shape=tuple(shape_cfg["opt_shape"]),
                max_shape=tuple(shape_cfg["max_shape"]),
            )

        from types import MappingProxyType

        model_inputs = (TensorRTModelInputConfig(input_shapes=MappingProxyType(input_shapes)),)
        self.logger.info(f"Using TensorRT profile for component '{component_name}': {list(input_shapes.keys())}")

        # Create TensorRT export config with component-specific model_inputs
        trt_export_config = TensorRTExportConfig(
            precision_policy=precision_policy,
            max_workspace_size=max_workspace_size,
            model_inputs=model_inputs,
        )

        return self.exporter_factory.create_tensorrt_exporter(
            config=config,
            logger=self.logger,
            config_override=trt_export_config,
        )
