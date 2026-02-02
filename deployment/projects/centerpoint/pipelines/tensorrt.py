"""
CenterPoint TensorRT Pipeline Implementation.
"""

from __future__ import annotations

import logging
import os.path as osp
import time
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
import torch

from deployment.core.artifacts import resolve_artifact_path
from deployment.pipelines.gpu_resource_mixin import (
    GPUResourceMixin,
    TensorRTResourceManager,
    release_tensorrt_resources,
)
from deployment.projects.centerpoint.pipelines.centerpoint_pipeline import CenterPointDeploymentPipeline

logger = logging.getLogger(__name__)


class CenterPointTensorRTPipeline(GPUResourceMixin, CenterPointDeploymentPipeline):
    """TensorRT-based CenterPoint pipeline (engine-per-component inference).

    Loads separate TensorRT engines for voxel_encoder and backbone_head components
    and runs inference using TensorRT execution contexts.

    Attributes:
        tensorrt_dir: Directory containing TensorRT engine files.
    """

    def __init__(
        self,
        pytorch_model: torch.nn.Module,
        tensorrt_dir: str,
        device: str = "cuda",
        components_cfg: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize TensorRT pipeline.

        Args:
            pytorch_model: Reference PyTorch model for preprocessing.
            tensorrt_dir: Directory containing TensorRT engine files.
            device: Target CUDA device ('cuda:N').
            components_cfg: Component configuration dict from deploy_config.
                           If None, uses default component names.

        Raises:
            ValueError: If device is not a CUDA device.
        """
        if not device.startswith("cuda"):
            raise ValueError("TensorRT requires CUDA device")

        super().__init__(pytorch_model, device, backend_type="tensorrt")

        self.tensorrt_dir = tensorrt_dir
        if components_cfg is None:
            components_cfg = {}
        if not isinstance(components_cfg, Mapping):
            raise TypeError(f"components_cfg must be a mapping, got {type(components_cfg).__name__}")
        self._components_cfg = components_cfg
        self._engines: dict = {}
        self._contexts: dict = {}
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._cleanup_called = False

        # Create CUDA events for GPU timing measurements
        self._backbone_start_event = cuda.Event()
        self._backbone_end_event = cuda.Event()
        self._voxel_encoder_start_event = cuda.Event()
        self._voxel_encoder_end_event = cuda.Event()

        self._load_tensorrt_engines()
        logger.info(f"TensorRT pipeline initialized with engines from: {tensorrt_dir}")

    def _load_tensorrt_engines(self) -> None:
        """Load TensorRT engines for each component.

        Raises:
            FileNotFoundError: If engine files are not found.
            RuntimeError: If engine loading or context creation fails.
        """
        trt.init_libnvinfer_plugins(self._logger, "")
        runtime = trt.Runtime(self._logger)

        engine_files = {
            "voxel_encoder": resolve_artifact_path(
                base_dir=self.tensorrt_dir,
                components_cfg=self._components_cfg,
                component="voxel_encoder",
                file_key="engine_file",
            ),
            "backbone_head": resolve_artifact_path(
                base_dir=self.tensorrt_dir,
                components_cfg=self._components_cfg,
                component="backbone_head",
                file_key="engine_file",
            ),
        }

        for component, engine_path in engine_files.items():
            if not osp.exists(engine_path):
                raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

            with open(engine_path, "rb") as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

            context = engine.create_execution_context()
            if context is None:
                raise RuntimeError(
                    f"Failed to create execution context for {component}. " "This is likely due to GPU out-of-memory."
                )

            self._engines[component] = engine
            self._contexts[component] = context
            logger.info(f"Loaded TensorRT engine: {component}")

    def _get_io_names(
        self,
        engine: Any,
        single_output: bool = False,
    ) -> Tuple[str, Any]:
        """Get input and output tensor names from engine.

        Args:
            engine: TensorRT engine.
            single_output: If True, return single output name instead of list.

        Returns:
            Tuple of (input_name, output_name(s)).

        Raises:
            RuntimeError: If input or output names cannot be found.
        """
        input_name = None
        output_names = []

        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                input_name = tensor_name
            elif engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
                output_names.append(tensor_name)

        if input_name is None:
            raise RuntimeError("Could not find input tensor name")
        if not output_names:
            raise RuntimeError("Could not find output tensor names")

        if single_output:
            return input_name, output_names[0]
        return input_name, output_names

    def run_voxel_encoder(self, input_features: torch.Tensor) -> torch.Tensor:
        """Run voxel encoder using TensorRT.

        Args:
            input_features: Input features [N, max_points, C].

        Returns:
            Voxel features [N, feature_dim].

        Raises:
            RuntimeError: If context is None (initialization failed).
        """
        engine = self._engines["voxel_encoder"]
        context = self._contexts["voxel_encoder"]
        if context is None:
            raise RuntimeError("voxel_encoder context is None - likely failed to initialize due to GPU OOM")

        input_array = self.to_numpy(input_features, dtype=np.float32)

        input_name, output_name = self._get_io_names(engine, single_output=True)
        context.set_input_shape(input_name, input_array.shape)
        output_shape = context.get_tensor_shape(output_name)
        output_array = np.empty(output_shape, dtype=np.float32)
        if not output_array.flags["C_CONTIGUOUS"]:
            output_array = np.ascontiguousarray(output_array)

        with TensorRTResourceManager() as manager:
            d_input = manager.allocate(input_array.nbytes)
            d_output = manager.allocate(output_array.nbytes)
            stream = manager.stream

            context.set_tensor_address(input_name, int(d_input))
            context.set_tensor_address(output_name, int(d_output))

            # Memory transfer: CPU -> GPU (not timed)
            cuda.memcpy_htod_async(d_input, input_array, stream)

            # Record start event and execute inference (pure GPU time)
            self._voxel_encoder_start_event.record(stream)
            context.execute_async_v3(stream_handle=stream.handle)
            self._voxel_encoder_end_event.record(stream)

            # Memory transfer: GPU -> CPU (not timed)
            cuda.memcpy_dtoh_async(output_array, d_output, stream)
            manager.synchronize()

        voxel_features = torch.from_numpy(output_array).to(self.device)
        voxel_features = voxel_features.squeeze(1)
        return voxel_features

    def run_backbone_head(self, spatial_features: torch.Tensor) -> List[torch.Tensor]:
        """Run backbone and head using TensorRT.

        Args:
            spatial_features: Spatial features [B, C, H, W].

        Returns:
            List of 6 head output tensors.

        Raises:
            RuntimeError: If context is None (initialization failed).
            ValueError: If output count is not 6.
        """
        engine = self._engines["backbone_head"]
        context = self._contexts["backbone_head"]
        if context is None:
            raise RuntimeError("backbone_head context is None - likely failed to initialize due to GPU OOM")

        input_array = self.to_numpy(spatial_features, dtype=np.float32)

        input_name, output_names = self._get_io_names(engine, single_output=False)
        context.set_input_shape(input_name, input_array.shape)

        output_arrays = {}
        for output_name in output_names:
            output_shape = context.get_tensor_shape(output_name)
            output_array = np.empty(output_shape, dtype=np.float32)
            if not output_array.flags["C_CONTIGUOUS"]:
                output_array = np.ascontiguousarray(output_array)
            output_arrays[output_name] = output_array

        with TensorRTResourceManager() as manager:
            d_input = manager.allocate(input_array.nbytes)
            d_outputs = {name: manager.allocate(arr.nbytes) for name, arr in output_arrays.items()}
            stream = manager.stream

            context.set_tensor_address(input_name, int(d_input))
            for output_name in output_names:
                context.set_tensor_address(output_name, int(d_outputs[output_name]))

            # Memory transfer: CPU -> GPU (not timed)
            cuda.memcpy_htod_async(d_input, input_array, stream)

            # Record start event and execute inference (pure GPU time)
            self._backbone_start_event.record(stream)
            context.execute_async_v3(stream_handle=stream.handle)
            self._backbone_end_event.record(stream)

            # Memory transfer: GPU -> CPU (not timed)
            for output_name in output_names:
                cuda.memcpy_dtoh_async(output_arrays[output_name], d_outputs[output_name], stream)

            manager.synchronize()

        head_outputs = [torch.from_numpy(output_arrays[name]).to(self.device) for name in output_names]

        if len(head_outputs) != 6:
            raise ValueError(f"Expected 6 head outputs, got {len(head_outputs)}")

        return head_outputs

    def run_model(
        self,
        preprocessed_input: Dict[str, torch.Tensor],
    ) -> Tuple[List[torch.Tensor], Dict[str, float]]:
        """Run complete multi-stage model inference with GPU timing using CUDA events.

        This override uses CUDA events to measure pure GPU inference time for
        TensorRT operations, matching the C++ implementation's timing methodology.

        Args:
            preprocessed_input: Dict from preprocess() containing:
                - 'input_features': Input features for voxel encoder [N_voxels, max_points, 11]
                - 'coors': Voxel coordinates [N_voxels, 4]
                - 'voxels': Raw voxel data
                - 'num_points': Number of points per voxel

        Returns:
            Tuple of (head_outputs, stage_latencies):
            - head_outputs: List of head outputs [heatmap, reg, height, dim, rot, vel]
            - stage_latencies: Dict mapping stage names to latency in ms
                - 'voxel_encoder_ms': Pure GPU inference time (CUDA events)
                - 'middle_encoder_ms': Wall-clock time (PyTorch)
                - 'backbone_head_ms': Pure GPU inference time (CUDA events)
        """
        # Use local variable for thread safety
        stage_latencies: Dict[str, float] = {}

        # Stage 1: Voxel Encoder
        voxel_features = self.run_voxel_encoder(preprocessed_input["input_features"])
        self._voxel_encoder_end_event.synchronize()
        voxel_encoder_gpu_time_ms = self._voxel_encoder_end_event.time_since(self._voxel_encoder_start_event)
        stage_latencies["voxel_encoder_ms"] = voxel_encoder_gpu_time_ms

        # Stage 2: Middle Encoder
        start = time.perf_counter()
        spatial_features = self.process_middle_encoder(voxel_features, preprocessed_input["coors"])
        stage_latencies["middle_encoder_ms"] = (time.perf_counter() - start) * 1000

        # Stage 3: Backbone + Head
        head_outputs = self.run_backbone_head(spatial_features)
        self._backbone_end_event.synchronize()
        backbone_head_gpu_time_ms = self._backbone_end_event.time_since(self._backbone_start_event)
        stage_latencies["backbone_head_ms"] = backbone_head_gpu_time_ms

        return head_outputs, stage_latencies

    def _release_gpu_resources(self) -> None:
        """Release TensorRT resources (engines and contexts) and CUDA events."""
        # Destroy CUDA events
        if hasattr(self, "_backbone_start_event"):
            try:
                del self._backbone_start_event
            except Exception:
                pass
        if hasattr(self, "_backbone_end_event"):
            try:
                del self._backbone_end_event
            except Exception:
                pass
        if hasattr(self, "_voxel_encoder_start_event"):
            try:
                del self._voxel_encoder_start_event
            except Exception:
                pass
        if hasattr(self, "_voxel_encoder_end_event"):
            try:
                del self._voxel_encoder_end_event
            except Exception:
                pass

        release_tensorrt_resources(
            engines=getattr(self, "_engines", None),
            contexts=getattr(self, "_contexts", None),
        )
