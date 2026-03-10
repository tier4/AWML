"""
CenterPoint-specific component extractor.

Extracts exportable submodules from CenterPoint using typed component config.
"""

import logging
from typing import List, Tuple

import torch

from deployment.configs import ComponentsConfig
from deployment.core.io.base_data_loader import BaseDataLoader
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

    def extract_components(
        self, model: torch.nn.Module, sample_data: Tuple[torch.Tensor, dict]
    ) -> List[ExportableComponent]:
        """Extract exportable submodules from the CenterPoint model for multi-file ONNX export.

        Builds two components: pts_voxel_encoder and pts_backbone_neck_head, each with
        name, module, and sample_input from the config and the given sample.

        Args:
            model: CenterPoint model containing pts_voxel_encoder, pts_middle_encoder,
                pts_backbone, pts_neck, pts_bbox_head.
            sample_data: 2-tuple (input_features, voxel_dict) for a single sample.

        Returns:
            List of two ExportableComponent instances (voxel encoder, backbone+neck+head).
        """
        input_features, voxel_dict = self._unpack_sample(sample_data)
        self.logger.info("Extracting CenterPoint components for export...")

        voxel_component = self._create_voxel_encoder_component(model, input_features)
        backbone_component = self._create_backbone_component(model, input_features, voxel_dict)

        self.logger.info("Extracted 2 components: pts_voxel_encoder, pts_backbone_neck_head")
        return [voxel_component, backbone_component]

    def _unpack_sample(self, sample_data: Tuple[torch.Tensor, dict]) -> Tuple[torch.Tensor, dict]:
        """Unpack (input_features, voxel_dict) from sample_data. Validates structure once.

        Args:
            sample_data: 2-tuple of (input_features, voxel_dict).

        Returns:
            input_features: Tensor for pts_voxel_encoder.
            voxel_dict: Dict with at least "coors" for middle encoder.

        Raises:
            TypeError: If sample_data is not a 2-tuple or types are wrong.
            KeyError: If voxel_dict does not contain "coors".
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
        """Create exportable component for the voxel encoder (pts_voxel_encoder).

        Args:
            model: CenterPoint model with pts_voxel_encoder.
            input_features: Sample voxel features used as sample_input for export.

        Returns:
            ExportableComponent with name from config, model.pts_voxel_encoder, sample_input.
        """
        component_cfg = self._components_cfg.get_component("pts_voxel_encoder")
        return ExportableComponent(
            name=component_cfg.name,
            module=model.pts_voxel_encoder,
            sample_input=input_features,
        )

    def _create_backbone_component(
        self, model: torch.nn.Module, input_features: torch.Tensor, voxel_dict: dict
    ) -> ExportableComponent:
        """Create exportable component for backbone + neck + head (pts_backbone_neck_head).

        Computes backbone input via _prepare_backbone_input and wraps backbone, neck,
        and bbox head in CenterPointHeadONNX.

        Args:
            model: CenterPoint model (voxel/middle encoders and backbone/neck/head).
            input_features: Voxel features for the voxel encoder.
            voxel_dict: Dict with "coors" for middle encoder and batch size.

        Returns:
            ExportableComponent with name from config, backbone module, and spatial sample_input.
        """
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
        """Compute spatial features for the backbone from voxel input and coordinates.

        Runs voxel encoder and middle encoder (no grad) to produce the tensor
        expected by pts_backbone. Uses voxel_dict["coors"] to infer batch_size.

        Args:
            model: CenterPoint model (pts_voxel_encoder, pts_middle_encoder).
            input_features: Voxel features tensor for pts_voxel_encoder.
            voxel_dict: Dict with "coors" used for middle encoder and batch size.

        Returns:
            spatial_features: Tensor fed to pts_backbone (e.g. [B, C, H, W]).
        """
        with torch.no_grad():
            voxel_features = model.pts_voxel_encoder(input_features).squeeze(1)
            coors = voxel_dict["coors"]
            batch_size = int(coors[-1, 0].item()) + 1 if len(coors) > 0 else 1
            spatial_features = model.pts_middle_encoder(voxel_features, coors, batch_size)
        return spatial_features

    def _create_backbone_module(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap pts_backbone, pts_neck, and pts_bbox_head into a single ONNX module."""
        return CenterPointHeadONNX(model.pts_backbone, model.pts_neck, model.pts_bbox_head)

    def extract_features(
        self, model: torch.nn.Module, data_loader: BaseDataLoader, sample_idx: int
    ) -> Tuple[torch.Tensor, dict]:
        """Extract (input_features, voxel_dict) for a sample for ONNX export.

        Delegates to model._extract_features if present; otherwise raises.
        Return value is suitable for _unpack_sample and component extraction.

        Args:
            model: CenterPoint model with optional _extract_features.
            data_loader: Loader to fetch sample from.
            sample_idx: Index of the sample to extract.

        Returns:
            Tuple of (input_features, voxel_dict) for the given sample.

        Raises:
            AttributeError: If model does not define _extract_features.
        """
        if hasattr(model, "_extract_features"):
            raw = model._extract_features(data_loader, sample_idx)
            return self._unpack_sample(raw)
        raise AttributeError(
            "CenterPoint model must have _extract_features method for ONNX export. "
            "Please ensure the model is built with ONNX compatibility."
        )
