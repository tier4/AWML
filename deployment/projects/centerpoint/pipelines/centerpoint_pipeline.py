"""
CenterPoint Deployment Pipeline Base Class.

"""

import logging
import time
from abc import abstractmethod
from typing import Dict, List, Tuple

import torch
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes

from deployment.pipelines.base_pipeline import BaseDeploymentPipeline

logger = logging.getLogger(__name__)


class CenterPointDeploymentPipeline(BaseDeploymentPipeline):
    """Base pipeline for CenterPoint staged inference.

    This normalizes preprocessing/postprocessing for CenterPoint and provides
    common helpers (e.g., middle encoder processing) used by PyTorch/ONNX/TensorRT
    backend-specific pipelines.
    """

    def __init__(self, pytorch_model, device: str = "cuda", backend_type: str = "unknown"):
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

        self.num_classes = len(class_names)
        self.class_names = class_names
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.pytorch_model = pytorch_model
        self._stage_latencies = {}

    def preprocess(self, points: torch.Tensor, **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict]:
        points_tensor = points.to(self.device)

        data_samples = [Det3DDataSample()]
        with torch.no_grad():
            batch_inputs = self.pytorch_model.data_preprocessor(
                {"inputs": {"points": [points_tensor]}, "data_samples": data_samples}
            )

        voxel_dict = batch_inputs["inputs"]["voxels"]
        voxels = voxel_dict["voxels"]
        num_points = voxel_dict["num_points"]
        coors = voxel_dict["coors"]

        input_features = None
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

    def process_middle_encoder(self, voxel_features: torch.Tensor, coors: torch.Tensor) -> torch.Tensor:
        voxel_features = voxel_features.to(self.device)
        coors = coors.to(self.device)
        batch_size = int(coors[-1, 0].item()) + 1 if len(coors) > 0 else 1
        with torch.no_grad():
            spatial_features = self.pytorch_model.pts_middle_encoder(voxel_features, coors, batch_size)
        return spatial_features

    def postprocess(self, head_outputs: List[torch.Tensor], sample_meta: Dict) -> List[Dict]:
        head_outputs = [out.to(self.device) for out in head_outputs]
        if len(head_outputs) != 6:
            raise ValueError(f"Expected 6 head outputs, got {len(head_outputs)}")

        heatmap, reg, height, dim, rot, vel = head_outputs

        if hasattr(self.pytorch_model, "pts_bbox_head"):
            rot_y_axis_reference = getattr(self.pytorch_model.pts_bbox_head, "_rot_y_axis_reference", False)
            if rot_y_axis_reference:
                dim = dim[:, [1, 0, 2], :, :]
                rot = rot * (-1.0)
                rot = rot[:, [1, 0], :, :]

        preds_dict = {"heatmap": heatmap, "reg": reg, "height": height, "dim": dim, "rot": rot, "vel": vel}
        preds_dicts = ([preds_dict],)

        if "box_type_3d" not in sample_meta:
            sample_meta["box_type_3d"] = LiDARInstance3DBoxes
        batch_input_metas = [sample_meta]

        with torch.no_grad():
            predictions_list = self.pytorch_model.pts_bbox_head.predict_by_feat(
                preds_dicts=preds_dicts, batch_input_metas=batch_input_metas
            )

        results = []
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
        raise NotImplementedError

    @abstractmethod
    def run_backbone_head(self, spatial_features: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    def run_model(self, preprocessed_input: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor], Dict[str, float]]:
        stage_latencies = {}

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

    def __repr__(self):
        return f"{self.__class__.__name__}(device={self.device})"
