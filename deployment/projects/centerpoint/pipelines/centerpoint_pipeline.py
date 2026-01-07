"""
CenterPoint Deployment Pipeline Base Class.

Provides common preprocessing, postprocessing, and inference logic
shared by PyTorch, ONNX, and TensorRT backend implementations.
"""

from __future__ import annotations

import logging
import time
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes

from deployment.pipelines.base_pipeline import BaseDeploymentPipeline

logger = logging.getLogger(__name__)


class CenterPointDeploymentPipeline(BaseDeploymentPipeline):
    """Base pipeline for CenterPoint staged inference.

    This normalizes preprocessing/postprocessing for CenterPoint and provides
    common helpers (e.g., middle encoder processing) used by PyTorch/ONNX/TensorRT
    backend-specific pipelines.

    Attributes:
        pytorch_model: Reference PyTorch model for preprocessing/postprocessing.
        num_classes: Number of detection classes.
        class_names: List of class names.
        point_cloud_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
        voxel_size: Voxel size [vx, vy, vz].
    """

    def __init__(
        self,
        pytorch_model: torch.nn.Module,
        device: str = "cuda",
        backend_type: str = "unknown",
    ) -> None:
        """Initialize CenterPoint pipeline.

        Args:
            pytorch_model: PyTorch model for preprocessing/postprocessing.
            device: Target device ('cpu' or 'cuda:N').
            backend_type: Backend identifier ('pytorch', 'onnx', 'tensorrt').

        Raises:
            ValueError: If class_names not found in pytorch_model.cfg.
        """
        cfg = getattr(pytorch_model, "cfg", None)

        class_names = getattr(cfg, "class_names", None)
        if class_names is None:
            raise ValueError("class_names not found in pytorch_model.cfg")

        point_cloud_range = getattr(cfg, "point_cloud_range", None)
        voxel_size = getattr(cfg, "voxel_size", None)

        super().__init__(
            model=pytorch_model,
            device=device,
            task_type="detection3d",
            backend_type=backend_type,
        )

        self.num_classes: int = len(class_names)
        self.class_names: List[str] = class_names
        self.point_cloud_range: Optional[List[float]] = point_cloud_range
        self.voxel_size: Optional[List[float]] = voxel_size
        self.pytorch_model: torch.nn.Module = pytorch_model
        self._stage_latencies: Dict[str, float] = {}

    def to_device_tensor(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Convert data to tensor on the pipeline's device.

        Args:
            data: Input data (torch.Tensor or np.ndarray).

        Returns:
            Tensor on self.device.
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        return data.to(self.device)

    def to_numpy(self, data: torch.Tensor, dtype: np.dtype = np.float32) -> np.ndarray:
        """Convert tensor to contiguous numpy array.

        Args:
            data: Input tensor.
            dtype: Target numpy dtype.

        Returns:
            Contiguous numpy array.
        """
        arr = data.cpu().numpy().astype(dtype)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return arr

    def preprocess(
        self,
        points: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Preprocess point cloud data for inference.

        Performs voxelization and feature extraction using the data_preprocessor
        and pts_voxel_encoder from the PyTorch model.

        Args:
            points: Point cloud tensor of shape [N, point_features].
            **kwargs: Additional arguments (unused).

        Returns:
            Tuple of (preprocessed_dict, metadata_dict).
            preprocessed_dict contains: input_features, voxels, num_points, coors.
        """
        points_tensor = self.to_device_tensor(points)

        data_samples = [Det3DDataSample()]
        with torch.no_grad():
            batch_inputs = self.pytorch_model.data_preprocessor(
                {"inputs": {"points": [points_tensor]}, "data_samples": data_samples}
            )

        voxel_dict = batch_inputs["inputs"]["voxels"]
        voxels = voxel_dict["voxels"]
        num_points = voxel_dict["num_points"]
        coors = voxel_dict["coors"]

        input_features: Optional[torch.Tensor] = None
        with torch.no_grad():
            if hasattr(self.pytorch_model.pts_voxel_encoder, "get_input_features"):
                input_features = self.pytorch_model.pts_voxel_encoder.get_input_features(voxels, num_points, coors)

        preprocessed_dict = {
            "input_features": input_features,
            "voxels": voxels,
            "num_points": num_points,
            "coors": coors,
        }

        return preprocessed_dict, {}

    def process_middle_encoder(
        self,
        voxel_features: torch.Tensor,
        coors: torch.Tensor,
    ) -> torch.Tensor:
        """Process voxel features through middle encoder (scatter to BEV).

        This step runs on PyTorch regardless of backend because it involves
        sparse-to-dense conversion that's not easily exportable to ONNX.

        Args:
            voxel_features: Encoded voxel features [N, feature_dim].
            coors: Voxel coordinates [N, 4] (batch_idx, z, y, x).

        Returns:
            Spatial features tensor [B, C, H, W].
        """
        voxel_features = self.to_device_tensor(voxel_features)
        coors = self.to_device_tensor(coors)

        batch_size = int(coors[-1, 0].item()) + 1 if len(coors) > 0 else 1

        with torch.no_grad():
            spatial_features = self.pytorch_model.pts_middle_encoder(voxel_features, coors, batch_size)

        return spatial_features

    def postprocess(
        self,
        head_outputs: List[torch.Tensor],
        sample_meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Postprocess head outputs to detection results.

        Args:
            head_outputs: List of 6 tensors [heatmap, reg, height, dim, rot, vel].
            sample_meta: Sample metadata dict.

        Returns:
            List of detection dicts with keys: bbox_3d, score, label.

        Raises:
            ValueError: If head_outputs doesn't contain exactly 6 tensors.
        """
        head_outputs = [self.to_device_tensor(out) for out in head_outputs]

        if len(head_outputs) != 6:
            raise ValueError(f"Expected 6 head outputs, got {len(head_outputs)}")

        heatmap, reg, height, dim, rot, vel = head_outputs

        # Apply rotation axis correction if configured
        if hasattr(self.pytorch_model, "pts_bbox_head"):
            rot_y_axis_reference = getattr(self.pytorch_model.pts_bbox_head, "_rot_y_axis_reference", False)
            if rot_y_axis_reference:
                dim = dim[:, [1, 0, 2], :, :]
                rot = rot * (-1.0)
                rot = rot[:, [1, 0], :, :]

        preds_dict = {
            "heatmap": heatmap,
            "reg": reg,
            "height": height,
            "dim": dim,
            "rot": rot,
            "vel": vel,
        }
        preds_dicts = ([preds_dict],)

        if "box_type_3d" not in sample_meta:
            sample_meta["box_type_3d"] = LiDARInstance3DBoxes
        batch_input_metas = [sample_meta]

        with torch.no_grad():
            predictions_list = self.pytorch_model.pts_bbox_head.predict_by_feat(
                preds_dicts=preds_dicts, batch_input_metas=batch_input_metas
            )

        results: List[Dict[str, Any]] = []
        for pred_instances in predictions_list:
            bboxes_3d = pred_instances.bboxes_3d.tensor.cpu().numpy()
            scores_3d = pred_instances.scores_3d.cpu().numpy()
            labels_3d = pred_instances.labels_3d.cpu().numpy()

            for i in range(len(bboxes_3d)):
                results.append(
                    {
                        "bbox_3d": bboxes_3d[i][:7].tolist(),
                        "score": float(scores_3d[i]),
                        "label": int(labels_3d[i]),
                    }
                )

        return results

    @abstractmethod
    def run_voxel_encoder(self, input_features: torch.Tensor) -> torch.Tensor:
        """Run voxel encoder inference.

        Args:
            input_features: Input features [N, max_points, C].

        Returns:
            Voxel features [N, feature_dim].
        """
        raise NotImplementedError

    @abstractmethod
    def run_backbone_head(self, spatial_features: torch.Tensor) -> List[torch.Tensor]:
        """Run backbone and head inference.

        Args:
            spatial_features: Spatial features [B, C, H, W].

        Returns:
            List of 6 head output tensors.
        """
        raise NotImplementedError

    def run_model(
        self,
        preprocessed_input: Dict[str, torch.Tensor],
    ) -> Tuple[List[torch.Tensor], Dict[str, float]]:
        """Run the full model pipeline with latency tracking.

        Args:
            preprocessed_input: Dict with keys: input_features, coors.

        Returns:
            Tuple of (head_outputs, stage_latencies).
        """
        stage_latencies: Dict[str, float] = {}

        start = time.perf_counter()
        voxel_features = self.run_voxel_encoder(preprocessed_input["input_features"])
        stage_latencies["voxel_encoder_ms"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        spatial_features = self.process_middle_encoder(voxel_features, preprocessed_input["coors"])
        stage_latencies["middle_encoder_ms"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        head_outputs = self.run_backbone_head(spatial_features)
        stage_latencies["backbone_head_ms"] = (time.perf_counter() - start) * 1000

        return head_outputs, stage_latencies

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device}, backend={self.backend_type})"
