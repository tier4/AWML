"""
CenterPoint-specific component builder.

Builds exportable submodules from CenterPoint using typed component config.
"""

from __future__ import annotations

import logging

import torch

from deployment.configs import ComponentsConfig
from deployment.exporters.export_pipelines.interfaces import ExportableComponent, ModelComponentBuilder
from deployment.projects.centerpoint.export_types import CenterPointExportSample
from deployment.projects.centerpoint.onnx_models.centerpoint_onnx import CenterPointHeadONNX


class CenterPointComponentBuilder(ModelComponentBuilder):
    """Build exportable CenterPoint submodules for multi-file ONNX export.

    For CenterPoint we export two components:
    - ``pts_voxel_encoder`` (pts_voxel_encoder)
    - ``pts_backbone_neck_head`` (pts_backbone + pts_neck + pts_bbox_head)
    """

    def __init__(
        self,
        components_cfg: ComponentsConfig,
        logger: logging.Logger,
    ) -> None:
        self._components_cfg = components_cfg
        self.logger = logger

    def build_components(
        self,
        model: torch.nn.Module,
        sample: CenterPointExportSample,
    ) -> list[ExportableComponent]:
        """Build exportable CenterPoint components from a typed sample."""
        self.logger.info("Extracting CenterPoint components for export...")

        voxel_component = self._create_voxel_encoder_component(model, sample)
        backbone_component = self._create_backbone_component(model, sample)

        self.logger.info("Extracted 2 components: pts_voxel_encoder, pts_backbone_neck_head")
        return [voxel_component, backbone_component]

    def _create_voxel_encoder_component(
        self,
        model: torch.nn.Module,
        sample: CenterPointExportSample,
    ) -> ExportableComponent:
        """Create exportable component for the voxel encoder (pts_voxel_encoder)."""
        component_cfg = self._components_cfg.get_component("pts_voxel_encoder")
        return ExportableComponent(
            name=component_cfg.name,
            module=model.pts_voxel_encoder,
            sample_input=sample.input_features,
        )

    def _create_backbone_component(
        self,
        model: torch.nn.Module,
        sample: CenterPointExportSample,
    ) -> ExportableComponent:
        """Create exportable component for backbone + neck + head (pts_backbone_neck_head)."""
        backbone_input = self._prepare_backbone_input(model, sample)
        backbone_module = self._create_backbone_module(model)

        component_cfg = self._components_cfg.get_component("pts_backbone_neck_head")
        return ExportableComponent(
            name=component_cfg.name,
            module=backbone_module,
            sample_input=backbone_input,
        )

    def _prepare_backbone_input(
        self,
        model: torch.nn.Module,
        sample: CenterPointExportSample,
    ) -> torch.Tensor:
        """Compute spatial features for the backbone from typed sample tensors."""
        with torch.no_grad():
            voxel_features = model.pts_voxel_encoder(sample.input_features).squeeze(1)
            coors = sample.coors
            batch_size = int(coors[-1, 0].item()) + 1 if len(coors) > 0 else 1
            spatial_features = model.pts_middle_encoder(voxel_features, coors, batch_size)
        return spatial_features

    def _create_backbone_module(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap pts_backbone, pts_neck, and pts_bbox_head into a single ONNX module."""
        return CenterPointHeadONNX(model.pts_backbone, model.pts_neck, model.pts_bbox_head)
