"""BEVFusion Inference Pipeline Base Class.

Provides common preprocessing, postprocessing, and inference logic shared by the PyTorch and
TensorRT backend implementations. ONNXRuntime is not a runtime backend for BEVFusion: the sparse
(spconv) graph needs TensorRT-only ``autoware`` plugins, so ONNX is an export format only.
"""

from __future__ import annotations

import logging
import time
from abc import abstractmethod
from typing import Any, Dict, List, Mapping, Tuple, Union

import torch
from typing_extensions import override

from deployment.config.enums import Backend
from deployment.inference.base_inference_pipeline import BaseInferencePipeline
from deployment.primitives.device import DeviceSpec
from projects.BEVFusion.bevfusion.utils import apply_cluster_nms

logger = logging.getLogger(__name__)


class BEVFusionInferencePipeline(BaseInferencePipeline):
    """Base pipeline for BEVFusion inference.

    Handles voxelization in preprocessing and bbox decoding in postprocessing.
    The model (ONNX/TensorRT) takes voxels/coors/num_points_per_voxel and
    outputs bbox_pred/score/label_pred directly.
    """

    def __init__(
        self,
        pytorch_model: torch.nn.Module,
        backend_type: Backend,
        device: DeviceSpec,
    ) -> None:
        """Initialize BEVFusion pipeline.

        Args:
            pytorch_model: PyTorch model for preprocessing/postprocessing.
            backend_type: Deployment backend enum. Required.
            device: Target runtime device (DeviceSpec).

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
        self.num_classes: int = len(class_names)
        self.class_names: List[str] = class_names

    @override
    def preprocess(
        self,
        points: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Voxelize point cloud into voxels/coors/num_points_per_voxel.

        Uses the BEVFusion model's voxelization layer (outside the ONNX graph).

        Args:
            points: Point cloud tensor [N, point_features].

        Returns:
            Dict with voxels, coors, num_points_per_voxel.
        """
        points_tensor = self.to_device_tensor(points).float()

        with torch.no_grad():
            voxel_output = self.pytorch_model.pts_voxel_layer(points_tensor)
            if not (isinstance(voxel_output, (tuple, list)) and len(voxel_output) == 3):
                raise NotImplementedError(
                    "BEVFusion deployment only supports hard voxelization "
                    "(max_num_points > 0); got a voxel layer output that is not "
                    "(voxels, coors, num_points_per_voxel)."
                )
            voxels, coors, num_points_per_voxel = voxel_output

        preprocessed_dict = {
            "voxels": voxels,
            "coors": coors,
            "num_points_per_voxel": num_points_per_voxel,
        }
        return preprocessed_dict

    @override
    def run_model(
        self,
        preprocessed_input: Dict[str, torch.Tensor],
    ) -> Tuple[List[torch.Tensor], Dict[str, float]]:
        """Run the BEVFusion model as two stages and report per-stage latency.

        Mirrors :class:`~deployment.projects.centerpoint.inference.centerpoint_inference_pipeline.CenterPointInferencePipeline`:
        the base orchestrates named seams and times each, so every backend shares one
        ``sparse`` / ``dense`` breakdown instead of hand-rolling its own. BEVFusion's two
        seams line up with the split ONNX/TensorRT graphs:

        - :meth:`run_sparse_encoder`: ``pts_voxel_encoder`` + ``pts_middle_encoder`` (spconv)
          -> the dense BEV feature map (the ``bevfusion_sparse`` component).
        - :meth:`run_dense`: ``pts_backbone`` + ``pts_neck`` + ``bbox_head`` + scoring
          -> ``[bbox_pred, score, label_pred]`` (the ``bevfusion_dense`` component).

        Plain wall-clock timing, exactly like CenterPoint's base orchestration: accurate for
        ONNX (``ort.run`` is a blocking call) and reported as reference timings for the native
        PyTorch backend. TensorRT overrides this to substitute pure-GPU CUDA-event times, and the
        merged single-graph backends override it to report one total (they cannot be split cleanly).

        Args:
            preprocessed_input: Dict with voxels, coors, num_points_per_voxel.

        Returns:
            Tuple of ([bbox_pred, score, label_pred], {"sparse_ms", "dense_ms"}).
        """
        stage_latencies: Dict[str, float] = {}

        start = time.perf_counter()
        bev_features = self.run_sparse_encoder(
            preprocessed_input["voxels"],
            preprocessed_input["coors"],
            preprocessed_input["num_points_per_voxel"],
        )
        stage_latencies["sparse_ms"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        model_outputs = self.run_dense(bev_features)
        stage_latencies["dense_ms"] = (time.perf_counter() - start) * 1000

        return model_outputs, stage_latencies

    @override
    def postprocess(
        self,
        model_output: List[torch.Tensor],
        metadata: Mapping[str, Any],
    ) -> List[Dict[str, Union[List[float], float, int]]]:
        """Decode bbox_pred/score/label_pred into detection dicts.

        The ONNX graph already bakes in the query scoring that produces the (bbox_pred, score,
        label_pred) triple, but the remaining reference-eval selection is not in the graph and is
        reproduced here so PyTorch-deploy / TensorRT match test.py: the bbox_coder decode with
        ``filter=True`` (per-class ``score_threshold`` + ``post_center_range``) followed by
        per-cluster circle NMS. bbox outputs arrive in head-encoded space and are decoded to
        metric coordinates:
        - bbox_pred: [10, num_proposals]
          (center_x_feat, center_y_feat, z_gravity, dim0_log, dim1_log, dim2_log, sin, cos, vx, vy)
        - score: [num_proposals]
        - label_pred: [num_proposals]

        Args:
            model_output: [bbox_pred, score, label_pred] tensors.
            metadata: Sample metadata.

        Returns:
            List of detection dicts with bbox_3d, score, label.
        """
        bbox_pred, score, label_pred = [self.to_device_tensor(o) for o in model_output]

        # Normalize common export/runtime shapes to [10, num_proposals], [num_proposals], [num_proposals].
        if bbox_pred.ndim == 3 and bbox_pred.shape[0] == 1:
            bbox_pred = bbox_pred[0]
        if bbox_pred.ndim == 2 and bbox_pred.shape[0] != 10 and bbox_pred.shape[1] == 10:
            bbox_pred = bbox_pred.transpose(0, 1).contiguous()
        if bbox_pred.ndim != 2 or bbox_pred.shape[0] != 10:
            logger.warning("Unexpected bbox_pred shape %s; skipping frame.", tuple(bbox_pred.shape))
            return []

        score = score.reshape(-1)
        label_pred = label_pred.reshape(-1)

        # bbox_pred/score/label_pred all carry the same num_proposals by construction
        # (head_dict_to_detection_outputs derives them from one head-output dict), so the
        # bbox_pred column count is the single source of truth for the proposal count.
        num_proposals = bbox_pred.shape[1]

        # Decode via BEVFusion's own bbox_coder to avoid convention drift.
        bbox_coder = getattr(self.pytorch_model.bbox_head, "bbox_coder", None)
        if bbox_coder is None:
            logger.warning("bbox_coder not found on model.bbox_head; skipping frame.")
            return []

        center = bbox_pred[0:2, :num_proposals].unsqueeze(0)
        height = bbox_pred[2:3, :num_proposals].unsqueeze(0)
        dim = bbox_pred[3:6, :num_proposals].unsqueeze(0)
        rot = bbox_pred[6:8, :num_proposals].unsqueeze(0)
        vel = bbox_pred[8:10, :num_proposals].unsqueeze(0)

        labels = label_pred[:num_proposals].long()
        scores = score[:num_proposals].to(dtype=bbox_pred.dtype)
        heatmap = torch.zeros((1, self.num_classes, num_proposals), device=self.torch_device, dtype=bbox_pred.dtype)
        valid = (labels >= 0) & (labels < self.num_classes)
        if valid.any():
            valid_idx = torch.nonzero(valid, as_tuple=False).reshape(-1)
            heatmap[0, labels[valid_idx], valid_idx] = scores[valid_idx]

        # filter=True applies the coder's per-class ``score_threshold`` and ``post_center_range``,
        # matching what ``BEVFusionHead.predict_by_feat`` runs during the reference eval (test.py).
        decoded = bbox_coder.decode(heatmap, rot, dim, center, height, vel, filter=True)[0]
        bbox_head = self.pytorch_model.bbox_head
        boxes3d, scores, labels = apply_cluster_nms(
            decoded["bboxes"],
            decoded["scores"],
            decoded["labels"],
            nms_type=bbox_head.test_cfg.get("nms_type"),
            nms_clusters=getattr(bbox_head, "nms_clusters", []),
            box_type_3d=metadata.get("box_type_3d"),
            pre_max_size=bbox_head.test_cfg.get("pre_max_size"),
            post_max_size=bbox_head.test_cfg.get("post_max_size"),
        )

        results: List[Dict[str, Union[List[float], float, int]]] = []
        for i in range(boxes3d.shape[0]):
            bbox = boxes3d[i].detach().cpu().numpy()
            # decoded box format: [x, y, z, dx, dy, dz, yaw, vx, vy]
            if bbox.shape[0] < 7:
                continue

            cx, cy, z = float(bbox[0]), float(bbox[1]), float(bbox[2])
            d0, d1, d2 = float(bbox[3]), float(bbox[4]), float(bbox[5])
            yaw = float(bbox[6])
            vx = float(bbox[7]) if bbox.shape[0] > 7 else 0.0
            vy = float(bbox[8]) if bbox.shape[0] > 8 else 0.0

            results.append(
                {
                    "bbox_3d": [cx, cy, z, d0, d1, d2, yaw, vx, vy],
                    "score": float(scores[i].item()),
                    "label": int(labels[i].item()),
                }
            )

        return results

    @abstractmethod
    def run_sparse_encoder(
        self,
        voxels: torch.Tensor,
        coors: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
    ) -> torch.Tensor:
        """Run the sparse branch: voxel encoder + spconv middle encoder -> dense BEV.

        Analogous to CenterPoint's :meth:`run_voxel_encoder`; corresponds to the
        ``bevfusion_sparse`` ONNX/TensorRT component. Voxelization already happened in
        :meth:`preprocess`, so this consumes the voxel tensors directly.

        Args:
            voxels: [M, max_points, C]
            coors: [M, 3] (z, y, x)
            num_points_per_voxel: [M]

        Returns:
            The dense BEV feature map [B, C, H, W] fed to :meth:`run_dense`.
        """
        raise NotImplementedError

    @abstractmethod
    def run_dense(self, bev_features: torch.Tensor) -> List[torch.Tensor]:
        """Run the dense branch: backbone + neck + head + scoring -> detection tensors.

        Analogous to CenterPoint's :meth:`run_backbone_head`; corresponds to the
        ``bevfusion_dense`` ONNX/TensorRT component.

        Args:
            bev_features: Dense BEV feature map [B, C, H, W] from :meth:`run_sparse_encoder`.

        Returns:
            ``[bbox_pred, score, label_pred]`` in the config's declared output order.
        """
        raise NotImplementedError
