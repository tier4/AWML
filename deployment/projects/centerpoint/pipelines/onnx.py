"""
CenterPoint ONNX Pipeline Implementation.
"""

from __future__ import annotations

import logging
import os.path as osp
from typing import Any, List, Mapping

import numpy as np
import onnxruntime as ort
import torch

from deployment.projects.centerpoint.pipelines.artifacts import resolve_component_artifact_path
from deployment.projects.centerpoint.pipelines.centerpoint_pipeline import CenterPointDeploymentPipeline

logger = logging.getLogger(__name__)


class CenterPointONNXPipeline(CenterPointDeploymentPipeline):
    """ONNXRuntime-based CenterPoint pipeline (componentized inference).

    Loads separate ONNX models for voxel_encoder and backbone_head components
    and runs inference using ONNXRuntime.

    Attributes:
        onnx_dir: Directory containing ONNX model files.
        voxel_encoder_session: ONNXRuntime session for voxel encoder.
        backbone_head_session: ONNXRuntime session for backbone + head.
    """

    def __init__(
        self,
        pytorch_model: torch.nn.Module,
        onnx_dir: str,
        device: str = "cpu",
        components_cfg: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize ONNX pipeline.

        Args:
            pytorch_model: Reference PyTorch model for preprocessing.
            onnx_dir: Directory containing ONNX model files.
            device: Target device ('cpu' or 'cuda:N').
            components_cfg: Component configuration dict from deploy_config.
                           If None, uses default component names.
        """
        super().__init__(pytorch_model, device, backend_type="onnx")

        self.onnx_dir = onnx_dir
        self._components_cfg = components_cfg or {}
        self._load_onnx_models(device)
        logger.info(f"ONNX pipeline initialized with models from: {onnx_dir}")

    def _load_onnx_models(self, device: str) -> None:
        """Load ONNX models for each component.

        Args:
            device: Target device for execution provider selection.

        Raises:
            FileNotFoundError: If ONNX model files are not found.
            RuntimeError: If model loading fails.
        """
        voxel_encoder_path = resolve_component_artifact_path(
            base_dir=self.onnx_dir,
            components_cfg=self._components_cfg,
            component="voxel_encoder",
            file_key="onnx_file",
            default_filename="pts_voxel_encoder.onnx",
        )
        backbone_head_path = resolve_component_artifact_path(
            base_dir=self.onnx_dir,
            components_cfg=self._components_cfg,
            component="backbone_head",
            file_key="onnx_file",
            default_filename="pts_backbone_neck_head.onnx",
        )

        if not osp.exists(voxel_encoder_path):
            raise FileNotFoundError(f"Voxel encoder ONNX not found: {voxel_encoder_path}")
        if not osp.exists(backbone_head_path):
            raise FileNotFoundError(f"Backbone head ONNX not found: {backbone_head_path}")

        # Configure session options
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.log_severity_level = 3

        # Select execution providers based on device
        if device.startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            logger.info("Using CUDA execution provider for ONNX")
        else:
            providers = ["CPUExecutionProvider"]
            logger.info("Using CPU execution provider for ONNX")

        try:
            self.voxel_encoder_session = ort.InferenceSession(voxel_encoder_path, sess_options=so, providers=providers)
            logger.info(f"Loaded voxel encoder: {voxel_encoder_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load voxel encoder ONNX: {e}")

        try:
            self.backbone_head_session = ort.InferenceSession(backbone_head_path, sess_options=so, providers=providers)
            logger.info(f"Loaded backbone+head: {backbone_head_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load backbone+head ONNX: {e}")

    def run_voxel_encoder(self, input_features: torch.Tensor) -> torch.Tensor:
        """Run voxel encoder using ONNXRuntime.

        Args:
            input_features: Input features [N, max_points, C].

        Returns:
            Voxel features [N, feature_dim].
        """
        input_array = self.to_numpy(input_features, dtype=np.float32)
        input_name = self.voxel_encoder_session.get_inputs()[0].name
        output_name = self.voxel_encoder_session.get_outputs()[0].name

        outputs = self.voxel_encoder_session.run([output_name], {input_name: input_array})

        voxel_features = torch.from_numpy(outputs[0]).to(self.device)
        if voxel_features.ndim == 3 and voxel_features.shape[1] == 1:
            voxel_features = voxel_features.squeeze(1)
        return voxel_features

    def run_backbone_head(self, spatial_features: torch.Tensor) -> List[torch.Tensor]:
        """Run backbone and head using ONNXRuntime.

        Args:
            spatial_features: Spatial features [B, C, H, W].

        Returns:
            List of 6 head output tensors.

        Raises:
            ValueError: If output count is not 6.
        """
        input_array = self.to_numpy(spatial_features, dtype=np.float32)

        input_name = self.backbone_head_session.get_inputs()[0].name
        output_names = [output.name for output in self.backbone_head_session.get_outputs()]

        outputs = self.backbone_head_session.run(output_names, {input_name: input_array})
        head_outputs = [torch.from_numpy(out).to(self.device) for out in outputs]

        if len(head_outputs) != 6:
            raise ValueError(f"Expected 6 head outputs, got {len(head_outputs)}")

        return head_outputs
