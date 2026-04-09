"""CenterPoint deploy-only ONNX model variants.

These modules provide ONNX-friendly model wrappers and detector variants used by
the deployment/export pipeline (not training).
"""

from typing import Dict, List, Tuple

import torch
from mmdet3d.models.detectors.centerpoint import CenterPoint
from mmdet3d.registry import MODELS
from mmengine.logging import MMLogger
from torch import nn

from deployment.core.device import DeviceSpec


class CenterPointHeadONNX(nn.Module):
    """Head module for centerpoint with BACKBONE, NECK and BBOX_HEAD"""

    def __init__(self, backbone: nn.Module, neck: nn.Module, bbox_head: nn.Module):
        """Wrap backbone, neck, and bbox_head into a single module for ONNX export.

        Args:
            backbone: pts_backbone module.
            neck: pts_neck module (may be None).
            bbox_head: pts_bbox_head module.
        """
        super(CenterPointHeadONNX, self).__init__()
        self.backbone: nn.Module = backbone
        self.neck: nn.Module = neck
        self.bbox_head: nn.Module = bbox_head
        self._logger = MMLogger.get_current_instance()
        self._logger.info("Running CenterPointHeadONNX!")

    def forward(self, x: torch.Tensor) -> Tuple[List[Dict[str, torch.Tensor]]]:
        """
        Note:
            torch.onnx.export() doesn't support triple-nested output

        Args:
            x (torch.Tensor): (B, C, H, W)
        Returns:
            tuple[list[dict[str, any]]]:
                (num_classes x [num_detect x {'reg', 'height', 'dim', 'rot', 'vel', 'heatmap'}])
        """
        x = self.backbone(x)
        if self.neck is not None:
            x = self.neck(x)
        x = self.bbox_head(x)

        return x


@MODELS.register_module()
class CenterPointONNX(CenterPoint):
    """onnx support impl of mmdet3d.models.detectors.CenterPoint"""

    def __init__(
        self,
        point_channels: int = 5,
        device: DeviceSpec = DeviceSpec.from_value("cpu"),
        **kwargs,
    ) -> None:
        """Initialize CenterPoint ONNX detector.

        Args:
            point_channels: Number of point feature channels (e.g. from voxel encoder).
            device: Target device specification.
            **kwargs: Passed to CenterPoint base class.
        """
        super().__init__(**kwargs)
        self._point_channels = point_channels
        self._device = device
        self._torch_device = self._device.to_torch_device()
        self._logger = MMLogger.get_current_instance()
        self._logger.info("Running CenterPointONNX!")

    def _get_inputs(self, data_loader, sample_idx=0) -> Dict[str, Any]:
        """
        Generate inputs from the provided data loader.

        Args:
            data_loader: Loader that implements ``load_sample``.
            sample_idx: Index of the sample to fetch.
        """
        if data_loader is None:
            raise ValueError("data_loader is required for CenterPoint ONNX export")

        if not hasattr(data_loader, "load_sample"):
            raise AttributeError("data_loader must implement 'load_sample(sample_idx)'")

        sample = data_loader.load_sample(sample_idx)

        if "points" not in sample:
            raise KeyError(f"Sample must contain 'points' (processed tensor). Got keys: {list(sample.keys())}")

        points = sample["points"]
        if not isinstance(points, torch.Tensor):
            raise TypeError(f"Expected points to be torch.Tensor, got {type(points)}")

        # Ensure points are on the correct device
        points = points.to(self._torch_device)
        points = [points]
        return {"points": points, "data_samples": None}

    def _extract_features(self, data_loader, sample_idx=0) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Extract (input_features, voxel_dict) using a sample from the data loader.

        Runs data preprocessor voxelization and voxel encoder get_input_features.
        Used by the ONNX export pipeline to get sample data for tracing.

        Args:
            data_loader: Loader with load_sample(sample_idx); must not be None.
            sample_idx: Index of the sample to use (default 0).

        Returns:
            Tuple of (input_features, voxel_dict) for backbone input and export.

        Raises:
            ValueError: If data_loader is None.
            KeyError/AttributeError: If sample or model components are invalid.
        """
        if data_loader is None:
            raise ValueError("data_loader is required to extract features")

        assert self.data_preprocessor is not None and hasattr(self.data_preprocessor, "voxelize")

        # Ensure data preprocessor is on the correct device
        if hasattr(self.data_preprocessor, "to"):
            self.data_preprocessor.to(self._torch_device)

        inputs = self._get_inputs(data_loader, sample_idx)
        voxel_dict = self.data_preprocessor.voxelize(points=inputs["points"], data_samples=inputs["data_samples"])

        # Ensure all voxel tensors are on the correct device
        for key in ["voxels", "num_points", "coors"]:
            if key in voxel_dict and isinstance(voxel_dict[key], torch.Tensor):
                voxel_dict[key] = voxel_dict[key].to(self._torch_device)

        assert self.pts_voxel_encoder is not None and hasattr(self.pts_voxel_encoder, "get_input_features")
        input_features = self.pts_voxel_encoder.get_input_features(
            voxel_dict["voxels"], voxel_dict["num_points"], voxel_dict["coors"]
        )
        return input_features, voxel_dict
