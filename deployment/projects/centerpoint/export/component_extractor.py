"""
CenterPoint-specific component extractor.

Extracts exportable submodules from CenterPoint using the unified component config.
"""

import logging
from typing import Any, Dict, List, Mapping, Tuple

import torch

from deployment.core import BaseDeploymentConfig
from deployment.exporters.common.configs import ONNXExportConfig
from deployment.exporters.export_pipelines.interfaces import ExportableComponent, ModelComponentExtractor
from deployment.projects.centerpoint.onnx_models.centerpoint_onnx import CenterPointHeadONNX

logger = logging.getLogger(__name__)


class CenterPointComponentExtractor(ModelComponentExtractor):
    """Extract exportable CenterPoint submodules for multi-file ONNX export.

    For CenterPoint we export two components:
    - `voxel_encoder` (pts_voxel_encoder)
    - `backbone_neck_head` (pts_backbone + pts_neck + pts_bbox_head)

    Uses the unified `components` config structure where each component defines
    its own IO specification, filenames, and TensorRT profiles.
    """

    def __init__(self, config: BaseDeploymentConfig, logger: logging.Logger = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._validate_config()

    @property
    def _components_cfg(self) -> Dict[str, Any]:
        """Get unified components configuration."""
        return dict((self.config.deploy_cfg or {}).get("components", {}) or {})

    @property
    def _onnx_config(self) -> Dict[str, Any]:
        """Get shared ONNX export settings."""
        return dict(self.config.onnx_config or {})

    def extract_components(self, model: torch.nn.Module, sample_data: Any) -> List[ExportableComponent]:
        input_features, voxel_dict = self._unpack_sample(sample_data)
        self.logger.info("Extracting CenterPoint components for export...")

        voxel_component = self._create_voxel_encoder_component(model, input_features)
        backbone_component = self._create_backbone_component(model, input_features, voxel_dict)

        self.logger.info("Extracted 2 components: voxel_encoder, backbone_neck_head")
        return [voxel_component, backbone_component]

    def _validate_config(self) -> None:
        """Validate component configuration."""
        components = self._components_cfg

        missing = []
        for required_key in ("voxel_encoder", "backbone_head"):
            if required_key not in components:
                missing.append(required_key)
        if missing:
            raise KeyError(
                "Missing required `components` entries for CenterPoint export: "
                f"{missing}. Please set them in deploy config."
            )

        def _require_fields(comp_key: str, fields: Tuple[str, ...]) -> None:
            comp = dict(components.get(comp_key, {}) or {})
            missing_fields = [f for f in fields if not comp.get(f)]
            if missing_fields:
                raise KeyError(
                    f"Missing required fields in components['{comp_key}']: {missing_fields}. "
                    "Expected at least: " + ", ".join(fields)
                )

        _require_fields("voxel_encoder", ("name", "onnx_file"))
        _require_fields("backbone_head", ("name", "onnx_file"))

    def _unpack_sample(self, sample_data: Any) -> Tuple[torch.Tensor, dict]:
        """
        Unpack extractor output into `(input_features, voxel_dict)`.

        We intentionally keep this contract simple to avoid extra project-specific types.
        """
        if not (isinstance(sample_data, (list, tuple)) and len(sample_data) == 2):
            raise TypeError(
                "Invalid sample_data for CenterPoint export. Expected a 2-tuple "
                "`(input_features: torch.Tensor, voxel_dict: dict)`."
            )
        input_features, voxel_dict = sample_data
        if not isinstance(input_features, torch.Tensor):
            raise TypeError(f"input_features must be a torch.Tensor, got: {type(input_features)}")
        if not isinstance(voxel_dict, dict):
            raise TypeError(f"voxel_dict must be a dict, got: {type(voxel_dict)}")
        if "coors" not in voxel_dict:
            raise KeyError("voxel_dict must contain key 'coors' for CenterPoint export")
        return input_features, voxel_dict

    def _get_component_io(self, component: str) -> Mapping[str, Any]:
        """Get IO specification for a component."""
        comp_cfg = self._components_cfg.get(component, {})
        return comp_cfg.get("io", {})

    def _build_onnx_config_for_component(
        self,
        component: str,
        input_names: Tuple[str, ...],
        output_names: Tuple[str, ...],
        dynamic_axes: Dict[str, Dict[int, str]] | None = None,
    ) -> ONNXExportConfig:
        """Build ONNX export config for a component using unified config."""
        comp_cfg = self._components_cfg.get(component, {})
        comp_io = comp_cfg.get("io", {})
        onnx_settings = self._onnx_config

        # Use dynamic_axes from component IO config if not explicitly provided
        if dynamic_axes is None:
            dynamic_axes = comp_io.get("dynamic_axes", {})

        return ONNXExportConfig(
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=onnx_settings.get("opset_version", 16),
            do_constant_folding=onnx_settings.get("do_constant_folding", True),
            simplify=bool(onnx_settings.get("simplify", True)),
            save_file=comp_cfg.get("onnx_file", f"{component}.onnx"),
        )

    def _create_voxel_encoder_component(
        self, model: torch.nn.Module, input_features: torch.Tensor
    ) -> ExportableComponent:
        """Create exportable component for voxel encoder."""
        comp_cfg = self._components_cfg["voxel_encoder"]
        comp_io = comp_cfg.get("io", {})

        # Get input/output names from IO config
        inputs = comp_io.get("inputs", [])
        outputs = comp_io.get("outputs", [])
        input_names = tuple(inp.get("name", "input_features") for inp in inputs) or ("input_features",)
        output_names = tuple(out.get("name", "pillar_features") for out in outputs) or ("pillar_features",)

        return ExportableComponent(
            name=comp_cfg["name"],
            module=model.pts_voxel_encoder,
            sample_input=input_features,
            config_override=self._build_onnx_config_for_component(
                component="voxel_encoder",
                input_names=input_names,
                output_names=output_names,
            ),
        )

    def _create_backbone_component(
        self, model: torch.nn.Module, input_features: torch.Tensor, voxel_dict: dict
    ) -> ExportableComponent:
        """Create exportable component for backbone + neck + head."""
        backbone_input = self._prepare_backbone_input(model, input_features, voxel_dict)
        backbone_module = self._create_backbone_module(model)

        comp_cfg = self._components_cfg["backbone_head"]
        comp_io = comp_cfg.get("io", {})

        # Get input/output names from IO config
        inputs = comp_io.get("inputs", [])
        outputs = comp_io.get("outputs", [])
        input_names = tuple(inp.get("name", "spatial_features") for inp in inputs) or ("spatial_features",)
        output_names = self._get_output_names(model, outputs)

        return ExportableComponent(
            name=comp_cfg["name"],
            module=backbone_module,
            sample_input=backbone_input,
            config_override=self._build_onnx_config_for_component(
                component="backbone_head",
                input_names=input_names,
                output_names=output_names,
            ),
        )

    def _prepare_backbone_input(
        self, model: torch.nn.Module, input_features: torch.Tensor, voxel_dict: dict
    ) -> torch.Tensor:
        with torch.no_grad():
            voxel_features = model.pts_voxel_encoder(input_features).squeeze(1)
            coors = voxel_dict["coors"]
            batch_size = int(coors[-1, 0].item()) + 1 if len(coors) > 0 else 1
            spatial_features = model.pts_middle_encoder(voxel_features, coors, batch_size)
        return spatial_features

    def _create_backbone_module(self, model: torch.nn.Module) -> torch.nn.Module:
        return CenterPointHeadONNX(model.pts_backbone, model.pts_neck, model.pts_bbox_head)

    def _get_output_names(self, model: torch.nn.Module, io_outputs: List[Dict[str, Any]]) -> Tuple[str, ...]:
        """Get output names from config or model.

        Priority:
        1. Component IO config outputs
        2. model.pts_bbox_head.output_names
        3. Raise error if neither available
        """
        # Try from component IO config first
        if io_outputs:
            return tuple(out.get("name") for out in io_outputs if out.get("name"))

        # Try from model
        if hasattr(model, "pts_bbox_head") and hasattr(model.pts_bbox_head, "output_names"):
            output_names = model.pts_bbox_head.output_names
            if isinstance(output_names, (list, tuple)):
                return tuple(output_names)
            return (output_names,)

        raise KeyError(
            "Missing head output names for CenterPoint export. "
            "Set `components.backbone_head.io.outputs` in the deployment config, "
            "or define `model.pts_bbox_head.output_names`."
        )

    def extract_features(self, model: torch.nn.Module, data_loader: Any, sample_idx: int) -> Tuple[torch.Tensor, dict]:
        if hasattr(model, "_extract_features"):
            raw = model._extract_features(data_loader, sample_idx)
            return self._unpack_sample(raw)
        raise AttributeError(
            "CenterPoint model must have _extract_features method for ONNX export. "
            "Please ensure the model is built with ONNX compatibility."
        )
