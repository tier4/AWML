"""
CenterPoint inference pipeline base class.

Provides common preprocessing, postprocessing, and inference logic
shared by PyTorch, ONNX, and TensorRT backend implementations.
"""

from __future__ import annotations

import logging
import time
from abc import abstractmethod
from typing import Any, Dict, List, Mapping, Tuple, Union

import torch
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes
from typing_extensions import override

from deployment.config.enums import Backend
from deployment.inference.base_inference_pipeline import BaseInferencePipeline
from deployment.primitives.device import DeviceSpec
from deployment.projects.centerpoint.io.sample_types import compute_batch_size

logger = logging.getLogger(__name__)


class CenterPointInferencePipeline(BaseInferencePipeline):
    """Base pipeline for CenterPoint staged inference.

    This normalizes preprocessing/postprocessing for CenterPoint and provides
    common helpers (e.g., middle encoder processing) used by PyTorch/ONNX/TensorRT
    backend-specific pipelines.

    Attributes:
        pytorch_model: Reference PyTorch model for preprocessing/postprocessing.
        num_classes: Number of detection classes.
        class_names: List of class names.
    """

    def __init__(
        self,
        pytorch_model: torch.nn.Module,
        backend_type: Backend,
        device: DeviceSpec,
    ) -> None:
        """Initialize CenterPoint pipeline.

        Args:
            pytorch_model: PyTorch model for preprocessing/postprocessing.
            device: Target runtime device (DeviceSpec).
            backend_type: Deployment backend enum. Required.

        Raises:
            ValueError: If class_names not found in pytorch_model.cfg.
        """
        cfg = getattr(pytorch_model, "cfg", None)

        class_names = getattr(cfg, "class_names", None)
        if class_names is None:
            raise ValueError("class_names not found in pytorch_model.cfg")

        super().__init__(
            model=pytorch_model,
            backend_type=backend_type,
            device=device,
        )

        self.pytorch_model: torch.nn.Module = pytorch_model
        self.class_names: List[str] = class_names
        self._rot_y_axis_reference: bool = pytorch_model.pts_bbox_head.rot_y_axis_reference

    @staticmethod
    def squeeze_voxel_features(voxel_features: torch.Tensor) -> torch.Tensor:
        """Collapse the singleton channel of the voxel-encoder output ``[N, 1, F] -> [N, F]``.

        All backends (PyTorch/ONNX/TensorRT) emit ``[N, 1, F]`` for CenterPoint; the guard
        fails loud if a future model variant changes that, instead of silently squeezing
        the wrong axis.
        """
        if voxel_features.ndim != 3 or voxel_features.shape[1] != 1:
            raise RuntimeError(f"Expected voxel encoder output [N, 1, F], got shape {tuple(voxel_features.shape)}.")
        return voxel_features.squeeze(1)

    @override
    def preprocess(
        self,
        points: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Preprocess point cloud data for inference.

        Performs voxelization and feature extraction using the data_preprocessor
        and pts_voxel_encoder from the PyTorch model.

        Args:
            points: Point cloud tensor of shape [N, point_features].

        Returns:
            Dict with input_features, voxels, num_points, coors.
        """
        points_tensor = self.to_device_tensor(points)

        data_samples = [Det3DDataSample()]
        # Run data preprocessor
        with torch.no_grad():
            batch_inputs = self.pytorch_model.data_preprocessor(
                {"inputs": {"points": [points_tensor]}, "data_samples": data_samples}
            )

        voxel_dict = batch_inputs["inputs"]["voxels"]
        voxels = voxel_dict["voxels"]
        num_points = voxel_dict["num_points"]
        coors = voxel_dict["coors"]

        with torch.no_grad():
            input_features = self.pytorch_model.pts_voxel_encoder.get_input_features(voxels, num_points, coors)

        preprocessed_dict = {
            "input_features": input_features,
            "voxels": voxels,
            "num_points": num_points,
            "coors": coors,
        }

        return preprocessed_dict

    def run_middle_encoder(
        self,
        voxel_features: torch.Tensor,
        coors: torch.Tensor,
    ) -> torch.Tensor:
        """Run voxel features through the middle encoder (scatter to BEV).

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

        batch_size = compute_batch_size(coors)

        with torch.no_grad():
            spatial_features = self.pytorch_model.pts_middle_encoder(voxel_features, coors, batch_size)

        return spatial_features

    @override
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
        spatial_features = self.run_middle_encoder(voxel_features, preprocessed_input["coors"])
        stage_latencies["middle_encoder_ms"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        model_outputs = self.run_backbone_head(spatial_features)
        stage_latencies["backbone_head_ms"] = (time.perf_counter() - start) * 1000

        return model_outputs, stage_latencies

    @override
    def postprocess(
        self,
        model_output: List[torch.Tensor],
        metadata: Mapping[str, Any],
    ) -> List[Dict[str, Union[List[float], float, int]]]:
        """Postprocess head outputs to detection results.

        Args:
            model_output: List of 6 tensors [heatmap, reg, height, dim, rot, vel].
            metadata: Sample metadata dict.

        Returns:
            List of detection dicts with keys: bbox_3d, score, label.

        Raises:
            ValueError: If model_output doesn't contain exactly 6 tensors.
        """
        model_output = [self.to_device_tensor(out) for out in model_output]

        if len(model_output) != 6:
            raise ValueError(f"Expected 6 head outputs, got {len(model_output)}")

        heatmap, reg, height, dim, rot, vel = model_output

        # Apply rotation axis correction to mirror the head's export-time convention.
        if self._rot_y_axis_reference:
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

        # Build a new dict instead of mutating the caller's metadata (the same metadata
        # may be reused across backends for the same frame).
        batch_input_metas = [{**metadata, "box_type_3d": metadata.get("box_type_3d", LiDARInstance3DBoxes)}]

        with torch.no_grad():
            predictions_list = self.pytorch_model.pts_bbox_head.predict_by_feat(
                preds_dicts=preds_dicts, batch_input_metas=batch_input_metas
            )

        results: List[Dict[str, Union[List[float], float, int]]] = []
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
