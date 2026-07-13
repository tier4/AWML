"""
CenterPoint ONNX Pipeline Implementation.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
import torch
from typing_extensions import override

from deployment.config.enums import Backend
from deployment.config.schema import ComponentsConfig
from deployment.primitives.artifacts import resolve_artifact_path
from deployment.primitives.device import DeviceSpec
from deployment.projects.centerpoint.inference.centerpoint_inference_pipeline import CenterPointInferencePipeline

logger = logging.getLogger(__name__)


class CenterPointONNXInferencePipeline(CenterPointInferencePipeline):
    """ONNXRuntime-based CenterPoint pipeline (componentized inference).

    Loads separate ONNX models for pts_voxel_encoder and pts_backbone_neck_head components
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
        device: DeviceSpec,
        components_cfg: ComponentsConfig,
    ) -> None:
        """Initialize ONNX pipeline.

        Args:
            pytorch_model: Reference PyTorch model for preprocessing.
            onnx_dir: Directory containing ONNX model files.
            device: Target runtime device (DeviceSpec).
            components_cfg: Component configuration from deploy_config (use ComponentsConfig.from_dict).
                           If None, raises.
        """
        super().__init__(pytorch_model=pytorch_model, backend_type=Backend.ONNX, device=device)

        self.onnx_dir = onnx_dir
        self._components_cfg = components_cfg
        self.voxel_encoder_session, self.backbone_head_session = self._load_onnx_models()
        logger.info("ONNX pipeline initialized with models from: %s", onnx_dir)

    def _load_onnx_models(self) -> Tuple[ort.InferenceSession, ort.InferenceSession]:
        """Load ONNX models for each component (voxel encoder and backbone+head).

        Uses self.onnx_dir, self._components_cfg, and self.device to resolve paths
        and select execution providers.

        Returns:
            The (voxel_encoder_session, backbone_head_session) ONNXRuntime sessions.

        Raises:
            FileNotFoundError: If ONNX model files are not found.
            RuntimeError: If model loading fails.
        """
        voxel_encoder_path = resolve_artifact_path(
            base_dir=self.onnx_dir,
            components_cfg=self._components_cfg,
            component_name="pts_voxel_encoder",
            file_key="onnx_file",
        )
        backbone_head_path = resolve_artifact_path(
            base_dir=self.onnx_dir,
            components_cfg=self._components_cfg,
            component_name="pts_backbone_neck_head",
            file_key="onnx_file",
        )

        # Configure session options
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.log_severity_level = 2  # Warning

        # Select execution providers based on device
        providers = self.device.to_ort_provider()
        device_message = "CUDA" if self.device.is_cuda else "CPU"
        logger.info("Using %s execution provider for ONNX", device_message)

        try:
            voxel_encoder_session = ort.InferenceSession(voxel_encoder_path, sess_options=so, providers=providers)
            logger.info("Loaded voxel encoder: %s", voxel_encoder_path)
            backbone_head_session = ort.InferenceSession(backbone_head_path, sess_options=so, providers=providers)
            logger.info("Loaded backbone+head: %s", backbone_head_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}") from e

        return voxel_encoder_session, backbone_head_session

    @override
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

        voxel_features = torch.from_numpy(outputs[0]).to(self.torch_device)

        return self.squeeze_voxel_features(voxel_features)

    @override
    def run_backbone_head(self, spatial_features: torch.Tensor) -> List[torch.Tensor]:
        """Run backbone and head using ONNXRuntime.

        Args:
            spatial_features: Spatial features [B, C, H, W].

        Returns:
            List of head output tensors in configured order.

        Raises:
            ValueError: If the ONNX outputs don't match the configured head outputs.
        """
        input_array = self.to_numpy(spatial_features, dtype=np.float32)

        input_name = self.backbone_head_session.get_inputs()[0].name
        onnx_output_names = [output.name for output in self.backbone_head_session.get_outputs()]
        expected_output_names = [
            out.name for out in self._components_cfg.get_component("pts_backbone_neck_head").io.outputs
        ]
        output_names = self.order_outputs_by_config(onnx_output_names, expected_output_names)

        # Run inference with ordered output names (ONNX Runtime returns outputs in the same order)
        outputs = self.backbone_head_session.run(output_names, {input_name: input_array})
        return [torch.from_numpy(out).to(self.torch_device) for out in outputs]
