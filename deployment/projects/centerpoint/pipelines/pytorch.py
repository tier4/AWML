"""
CenterPoint PyTorch Pipeline Implementation.
"""

import logging
from typing import Dict, List

import torch

from deployment.projects.centerpoint.pipelines.centerpoint_pipeline import CenterPointDeploymentPipeline

logger = logging.getLogger(__name__)


class CenterPointPyTorchPipeline(CenterPointDeploymentPipeline):
    """PyTorch-based CenterPoint pipeline (staged to match ONNX/TensorRT outputs)."""

    def __init__(self, pytorch_model, device: str = "cuda"):
        super().__init__(pytorch_model, device, backend_type="pytorch")
        logger.info("PyTorch pipeline initialized (ONNX-compatible staged inference)")

    def infer(self, points: torch.Tensor, sample_meta: Dict = None, return_raw_outputs: bool = False):
        if sample_meta is None:
            sample_meta = {}
        return super().infer(points, sample_meta, return_raw_outputs=return_raw_outputs)

    def run_voxel_encoder(self, input_features: torch.Tensor) -> torch.Tensor:
        if input_features is None:
            raise ValueError("input_features is None. This should not happen for ONNX models.")

        input_features = input_features.to(self.device)

        with torch.no_grad():
            voxel_features = self.pytorch_model.pts_voxel_encoder(input_features)

        if voxel_features.ndim == 3:
            if voxel_features.shape[1] == 1:
                voxel_features = voxel_features.squeeze(1)
            elif voxel_features.shape[2] == 1:
                voxel_features = voxel_features.squeeze(2)
            else:
                raise RuntimeError(
                    f"Voxel encoder output has unexpected 3D shape: {voxel_features.shape}. "
                    f"Expected 2D output [N_voxels, feature_dim]. Input was: {input_features.shape}"
                )
        elif voxel_features.ndim > 3:
            raise RuntimeError(
                f"Voxel encoder output has {voxel_features.ndim}D shape: {voxel_features.shape}. "
                "Expected 2D output [N_voxels, feature_dim]."
            )

        return voxel_features

    def run_backbone_head(self, spatial_features: torch.Tensor) -> List[torch.Tensor]:
        spatial_features = spatial_features.to(self.device)

        with torch.no_grad():
            x = self.pytorch_model.pts_backbone(spatial_features)

            if hasattr(self.pytorch_model, "pts_neck") and self.pytorch_model.pts_neck is not None:
                x = self.pytorch_model.pts_neck(x)

            head_outputs_tuple = self.pytorch_model.pts_bbox_head(x)

            if isinstance(head_outputs_tuple, tuple) and len(head_outputs_tuple) > 0:
                first_element = head_outputs_tuple[0]

                if isinstance(first_element, torch.Tensor):
                    head_outputs = list(head_outputs_tuple)
                elif isinstance(first_element, list) and len(first_element) > 0:
                    preds_dict = first_element[0]
                    head_outputs = [
                        preds_dict["heatmap"],
                        preds_dict["reg"],
                        preds_dict["height"],
                        preds_dict["dim"],
                        preds_dict["rot"],
                        preds_dict["vel"],
                    ]
                else:
                    raise ValueError(f"Unexpected task_outputs format: {type(first_element)}")
            else:
                raise ValueError(f"Unexpected head_outputs format: {type(head_outputs_tuple)}")

        return head_outputs
