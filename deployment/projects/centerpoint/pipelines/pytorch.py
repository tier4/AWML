"""
CenterPoint PyTorch Pipeline Implementation.
"""

from __future__ import annotations

import logging
from typing import List

import torch

from deployment.projects.centerpoint.pipelines.centerpoint_pipeline import CenterPointDeploymentPipeline

logger = logging.getLogger(__name__)


class CenterPointPyTorchPipeline(CenterPointDeploymentPipeline):
    """PyTorch-based CenterPoint pipeline (staged to match ONNX/TensorRT outputs).

    This pipeline runs inference using the native PyTorch model, but structures
    the execution to match the ONNX/TensorRT staged inference for consistency.
    """

    def __init__(self, pytorch_model: torch.nn.Module, device: str = "cuda") -> None:
        """Initialize PyTorch pipeline.

        Args:
            pytorch_model: PyTorch model for inference.
            device: Target device ('cpu' or 'cuda:N').
        """
        super().__init__(pytorch_model, device, backend_type="pytorch")
        logger.info("PyTorch pipeline initialized (ONNX-compatible staged inference)")

    def run_voxel_encoder(self, input_features: torch.Tensor) -> torch.Tensor:
        """Run voxel encoder using PyTorch model.

        Args:
            input_features: Input features [N, max_points, C].

        Returns:
            Voxel features [N, feature_dim].

        Raises:
            ValueError: If input_features is None.
            RuntimeError: If output shape is unexpected.
        """
        if input_features is None:
            raise ValueError("input_features is None. This should not happen for ONNX models.")

        input_features = self.to_device_tensor(input_features)

        with torch.no_grad():
            voxel_features = self.pytorch_model.pts_voxel_encoder(input_features)

        # Handle various output shapes from different encoder variants
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
        """Run backbone and head using PyTorch model.

        Args:
            spatial_features: Spatial features [B, C, H, W].

        Returns:
            List of 6 head output tensors.

        Raises:
            ValueError: If head output format is unexpected.
        """
        spatial_features = self.to_device_tensor(spatial_features)

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
