"""
CenterPoint-specific component extractor.
"""

import logging
from typing import Any, List, Tuple

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
    """

    def __init__(self, config: BaseDeploymentConfig, logger: logging.Logger = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._validate_config()

    @property
    def _onnx_config(self) -> dict:
        return dict(self.config.onnx_config or {})

    @property
    def _model_io(self) -> dict:
        return dict((self.config.deploy_cfg or {}).get("model_io", {}) or {})

    def extract_components(self, model: torch.nn.Module, sample_data: Any) -> List[ExportableComponent]:
        input_features, voxel_dict = self._unpack_sample(sample_data)
        self.logger.info("Extracting CenterPoint components for export...")

        voxel_component = self._create_voxel_encoder_component(model, input_features)
        backbone_component = self._create_backbone_component(model, input_features, voxel_dict)

        self.logger.info("Extracted 2 components: voxel_encoder, backbone_neck_head")
        return [voxel_component, backbone_component]

    def _validate_config(self) -> None:
        onnx_config = self._onnx_config
        components = dict(onnx_config.get("components", {}) or {})

        missing = []
        for required_key in ("voxel_encoder", "backbone_head"):
            if required_key not in components:
                missing.append(required_key)
        if missing:
            raise KeyError(
                "Missing required `onnx_config.components` entries for CenterPoint export: "
                f"{missing}. Please set them in deploy config."
            )

        def _require_fields(comp_key: str, fields: Tuple[str, ...]) -> None:
            comp = dict(components.get(comp_key, {}) or {})
            missing_fields = [f for f in fields if not comp.get(f)]
            if missing_fields:
                raise KeyError(
                    f"Missing required fields in onnx_config.components['{comp_key}']: {missing_fields}. "
                    "Expected at least: " + ", ".join(fields)
                )

        _require_fields("voxel_encoder", ("name", "onnx_file"))
        _require_fields("backbone_head", ("name", "onnx_file"))

        # Make it easier to debug later failures by flagging likely misconfigurations early.
        model_io = self._model_io
        if not model_io.get("head_output_names"):
            self.logger.warning(
                "deploy_cfg.model_io.head_output_names is not set. "
                "Export will rely on `model.pts_bbox_head.output_names` at runtime."
            )

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

    def _create_voxel_encoder_component(
        self, model: torch.nn.Module, input_features: torch.Tensor
    ) -> ExportableComponent:
        onnx_config = self._onnx_config
        voxel_cfg = onnx_config["components"]["voxel_encoder"]
        return ExportableComponent(
            name=voxel_cfg["name"],
            module=model.pts_voxel_encoder,
            sample_input=input_features,
            config_override=ONNXExportConfig(
                input_names=("input_features",),
                output_names=("pillar_features",),
                dynamic_axes={
                    "input_features": {0: "num_voxels", 1: "num_max_points"},
                    "pillar_features": {0: "num_voxels"},
                },
                opset_version=onnx_config.get("opset_version", 16),
                do_constant_folding=True,
                simplify=bool(onnx_config.get("simplify", True)),
                save_file=voxel_cfg["onnx_file"],
            ),
        )

    def _create_backbone_component(
        self, model: torch.nn.Module, input_features: torch.Tensor, voxel_dict: dict
    ) -> ExportableComponent:
        backbone_input = self._prepare_backbone_input(model, input_features, voxel_dict)
        backbone_module = self._create_backbone_module(model)
        output_names = self._get_output_names(model)

        dynamic_axes = {
            "spatial_features": {0: "batch_size", 2: "height", 3: "width"},
        }
        for name in output_names:
            dynamic_axes[name] = {0: "batch_size", 2: "height", 3: "width"}

        onnx_config = self._onnx_config
        backbone_cfg = onnx_config["components"]["backbone_head"]
        return ExportableComponent(
            name=backbone_cfg["name"],
            module=backbone_module,
            sample_input=backbone_input,
            config_override=ONNXExportConfig(
                input_names=("spatial_features",),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=onnx_config.get("opset_version", 16),
                do_constant_folding=True,
                simplify=bool(onnx_config.get("simplify", True)),
                save_file=backbone_cfg["onnx_file"],
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

    def _get_output_names(self, model: torch.nn.Module) -> Tuple[str, ...]:
        if hasattr(model, "pts_bbox_head") and hasattr(model.pts_bbox_head, "output_names"):
            output_names = model.pts_bbox_head.output_names
            if isinstance(output_names, (list, tuple)):
                return tuple(output_names)
            return (output_names,)
        model_io = self._model_io
        output_names = model_io.get("head_output_names", ())
        if not output_names:
            raise KeyError(
                "Missing head output names for CenterPoint export. "
                "Set `model_io.head_output_names` in the deployment config, "
                "or define `model.pts_bbox_head.output_names`."
            )
        return tuple(output_names)

    def extract_features(self, model: torch.nn.Module, data_loader: Any, sample_idx: int) -> Tuple[torch.Tensor, dict]:
        if hasattr(model, "_extract_features"):
            raw = model._extract_features(data_loader, sample_idx)
            return self._unpack_sample(raw)
        raise AttributeError(
            "CenterPoint model must have _extract_features method for ONNX export. "
            "Please ensure the model is built with ONNX compatibility."
        )
