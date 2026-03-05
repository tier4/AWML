"""
CenterPoint-specific component extractor.

Extracts exportable submodules from CenterPoint using typed component config.
"""

import logging
from typing import Any, List, Tuple

import torch

from deployment.core.config.base_config import ComponentsConfig
from deployment.exporters.export_pipelines.interfaces import ExportableComponent, ModelComponentExtractor
from deployment.projects.centerpoint.onnx_models.centerpoint_onnx import CenterPointHeadONNX

logger = logging.getLogger(__name__)


class CenterPointComponentExtractor(ModelComponentExtractor):
    """Extract exportable CenterPoint submodules for multi-file ONNX export.

    For CenterPoint we export two components:
    - ``pts_voxel_encoder`` (pts_voxel_encoder)
    - ``pts_backbone_neck_head`` (pts_backbone + pts_neck + pts_bbox_head)
    """

    def __init__(
        self,
        components_cfg: ComponentsConfig,
        logger: logging.Logger,
    ):
        self._components_cfg = components_cfg
        self.logger = logger or logging.getLogger(__name__)

    def extract_components(self, model: torch.nn.Module, sample_data: Any) -> List[ExportableComponent]:
        """Extract exportable submodules from the CenterPoint model for multi-file ONNX export."""
        input_features, voxel_dict = self._unpack_sample(sample_data)
        self.logger.info("Extracting CenterPoint components for export...")

        voxel_component = self._create_voxel_encoder_component(model, input_features)
        backbone_component = self._create_backbone_component(model, input_features, voxel_dict)

        self.logger.info("Extracted 2 components: pts_voxel_encoder, pts_backbone_neck_head")
        return [voxel_component, backbone_component]

    def _unpack_sample(self, sample_data: Any) -> Tuple[torch.Tensor, dict]:
        """Unpack (input_features, voxel_dict) from sample_data. Validates structure once."""
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
        """Create exportable component for voxel encoder."""
        component_cfg = self._components_cfg.get_component("pts_voxel_encoder")
        return ExportableComponent(
            name=component_cfg.name,
            module=model.pts_voxel_encoder,
            sample_input=input_features,
        )

    def _create_backbone_component(
        self, model: torch.nn.Module, input_features: torch.Tensor, voxel_dict: dict
    ) -> ExportableComponent:
        """Create exportable component for backbone + neck + head."""
        backbone_input = self._prepare_backbone_input(model, input_features, voxel_dict)
        backbone_module = self._create_backbone_module(model)

        component_cfg = self._components_cfg.get_component("pts_backbone_neck_head")
        return ExportableComponent(
            name=component_cfg.name,
            module=backbone_module,
            sample_input=backbone_input,
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

    def extract_features(self, model: torch.nn.Module, data_loader: Any, sample_idx: int) -> Tuple[torch.Tensor, dict]:
        if hasattr(model, "_extract_features"):
            raw = model._extract_features(data_loader, sample_idx)
            return self._unpack_sample(raw)
        raise AttributeError(
            "CenterPoint model must have _extract_features method for ONNX export. "
            "Please ensure the model is built with ONNX compatibility."
        )
