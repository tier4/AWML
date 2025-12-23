import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from mmdet3d.models.detectors.centerpoint import CenterPoint
from mmdet3d.registry import MODELS
from mmengine.logging import MMLogger
from torch import nn


class CenterPointHeadONNX(nn.Module):
    """Head module for centerpoint with BACKBONE, NECK and BBOX_HEAD"""

    def __init__(self, backbone: nn.Module, neck: nn.Module, bbox_head: nn.Module):
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

    def __init__(self, point_channels: int = 5, device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        self._point_channels = point_channels
        self._device = device
        # Handle both "cuda:0" and "gpu" device strings
        if self._device.startswith("cuda") or self._device == "gpu":
            self._torch_device = torch.device(self._device if self._device.startswith("cuda") else "cuda:0")
        else:
            self._torch_device = torch.device("cpu")
        self._logger = MMLogger.get_current_instance()
        self._logger.info("Running CenterPointONNX!")

    def _get_inputs(self, data_loader, sample_idx=0):
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

        if "lidar_points" not in sample:
            raise KeyError("Sample does not contain 'lidar_points'")

        lidar_path = sample["lidar_points"].get("lidar_path")
        if not lidar_path:
            raise ValueError("Sample must provide 'lidar_path' inside 'lidar_points'")

        if not os.path.exists(lidar_path):
            raise FileNotFoundError(f"Lidar path not found: {lidar_path}")

        points = self._load_point_cloud(lidar_path)
        points = torch.from_numpy(points).to(self._torch_device)
        points = [points]
        return {"points": points, "data_samples": None}

    def _load_point_cloud(self, lidar_path: str) -> np.ndarray:
        """
        Load point cloud from file.

        Args:
            lidar_path: Path to point cloud file (.bin or .pcd)

        Returns:
            Point cloud array (N, 5) where 5 = (x, y, z, intensity, ring_id)
        """
        if lidar_path.endswith(".bin"):
            # Load binary point cloud (KITTI/nuScenes format)
            # T4 dataset has 5 features: x, y, z, intensity, ring_id
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)

            # Don't pad here - let the voxelization process handle feature expansion
            # The voxelization process will add cluster_center (+3) and voxel_center (+3) features
            # So 5 + 3 + 3 = 11 features total

        elif lidar_path.endswith(".pcd"):
            # Load PCD format (placeholder - would need pypcd or similar)
            raise NotImplementedError("PCD format loading not implemented yet")
        else:
            raise ValueError(f"Unsupported point cloud format: {lidar_path}")

        return points

    def _extract_features(self, data_loader, sample_idx=0):
        """
        Extract features using samples from the provided data loader.
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
