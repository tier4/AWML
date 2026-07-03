"""
CenterPoint TensorRT Pipeline Implementation.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple, Union

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
import torch
from typing_extensions import override

from deployment.config.enums import Backend
from deployment.config.schema import ComponentsConfig
from deployment.inference.gpu_resource_mixin import (
    GPUResourceMixin,
    TensorRTResourceManager,
    release_tensorrt_resources,
)
from deployment.primitives.artifacts import resolve_artifact_path
from deployment.primitives.device import DeviceSpec
from deployment.projects.centerpoint.inference.centerpoint_inference_pipeline import CenterPointInferencePipeline

logger = logging.getLogger(__name__)


class CenterPointTensorRTInferencePipeline(GPUResourceMixin, CenterPointInferencePipeline):
    """TensorRT-based CenterPoint pipeline (engine-per-component inference).

    Loads separate TensorRT engines for pts_voxel_encoder and pts_backbone_neck_head components
    and runs inference using TensorRT execution contexts.

    Attributes:
        tensorrt_dir: Directory containing TensorRT engine files.
    """

    # Free the CUDA cache every N evaluated samples
    _GPU_CLEANUP_INTERVAL = 10

    def __init__(
        self,
        pytorch_model: torch.nn.Module,
        tensorrt_dir: str,
        components_cfg: ComponentsConfig,
        device: DeviceSpec,
    ) -> None:
        """Initialize TensorRT pipeline.

        Args:
            pytorch_model: Reference PyTorch model for preprocessing.
            tensorrt_dir: Directory containing TensorRT engine files.
            components_cfg: Component configuration from deploy_config (use ComponentsConfig.from_dict).
            device: Target CUDA device ('cuda:N').

        Raises:
            ValueError: If device is not a CUDA device or components_cfg is None.
        """
        super().__init__(pytorch_model=pytorch_model, backend_type=Backend.TENSORRT, device=device)

        self.tensorrt_dir = tensorrt_dir
        self._components_cfg = components_cfg
        self._engines: Dict[str, trt.ICudaEngine] = {}
        self._contexts: Dict[str, trt.IExecutionContext] = {}
        self._logger = trt.Logger(trt.Logger.WARNING)

        # Per-stage pure-GPU times (ms), filled by each stage while its CUDA stream is
        # still alive and read back in run_model.
        self._gpu_stage_ms: Dict[str, float] = {}

        self._load_tensorrt_engines()
        logger.info("TensorRT pipeline initialized with engines from: %s", tensorrt_dir)

    def _load_tensorrt_engines(self) -> None:
        """Load TensorRT engines for each component.

        Raises:
            FileNotFoundError: If engine files are not found.
            RuntimeError: If engine loading or context creation fails.
        """
        trt.init_libnvinfer_plugins(self._logger, "")
        runtime = trt.Runtime(self._logger)

        engine_files = {
            "pts_voxel_encoder": resolve_artifact_path(
                base_dir=self.tensorrt_dir,
                components_cfg=self._components_cfg,
                component_name="pts_voxel_encoder",
                file_key="engine_file",
            ),
            "pts_backbone_neck_head": resolve_artifact_path(
                base_dir=self.tensorrt_dir,
                components_cfg=self._components_cfg,
                component_name="pts_backbone_neck_head",
                file_key="engine_file",
            ),
        }

        for component_name, engine_path in engine_files.items():
            with open(engine_path, "rb") as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

            context = engine.create_execution_context()
            if context is None:
                raise RuntimeError(
                    f"Failed to create execution context for {component_name}. "
                    "This is likely due to GPU out-of-memory."
                )

            self._engines[component_name] = engine
            self._contexts[component_name] = context
            logger.info("Loaded TensorRT engine: %s", component_name)

    def _get_io_names(
        self,
        engine: trt.ICudaEngine,
        single_output: bool = False,
    ) -> Tuple[str, Union[str, List[str]]]:
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

    def _run_engine_inference(
        self,
        context: trt.IExecutionContext,
        input_name: str,
        input_array: np.ndarray,
        output_names: List[str],
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """Run one TensorRT context end-to-end and return outputs plus pure-GPU time.

        Allocates device buffers, copies the input host->device, executes the context
        while timing it with CUDA events, copies every output device->host, and reads
        the elapsed GPU time back while the stream is still alive. Shared by the
        single-output voxel encoder and the multi-output backbone/head stages.

        Args:
            context: Execution context whose engine exposes ``input_name``/``output_names``.
            input_name: Engine input tensor name (its shape is set from ``input_array``).
            input_array: Contiguous float32 host input.
            output_names: Engine output tensor names, in the desired return order.

        Returns:
            Tuple of (outputs-by-name as host ndarrays, pure-GPU time in ms).
        """
        context.set_input_shape(input_name, input_array.shape)

        # Output shapes can depend on the input shape, so read them only after set_input_shape.
        outputs: Dict[str, np.ndarray] = {}
        for name in output_names:
            output_array = np.empty(context.get_tensor_shape(name), dtype=np.float32)
            if not output_array.flags["C_CONTIGUOUS"]:
                output_array = np.ascontiguousarray(output_array)
            outputs[name] = output_array

        with TensorRTResourceManager() as manager:
            d_input = manager.allocate(input_array.nbytes)
            d_outputs = {name: manager.allocate(arr.nbytes) for name, arr in outputs.items()}
            stream = manager.stream

            context.set_tensor_address(input_name, int(d_input))
            for name in output_names:
                context.set_tensor_address(name, int(d_outputs[name]))

            # Memory transfer: CPU -> GPU
            cuda.memcpy_htod_async(d_input, input_array, stream)

            # Record start event and execute inference
            start_event = cuda.Event()
            end_event = cuda.Event()
            start_event.record(stream)
            context.execute_async_v3(stream_handle=stream.handle)
            end_event.record(stream)

            # Memory transfer: GPU -> CPU
            for name in output_names:
                cuda.memcpy_dtoh_async(outputs[name], d_outputs[name], stream)
            manager.synchronize()

            # Read GPU timing while the stream is still alive (events are complete after
            # synchronize); avoids reading across a stream that has been released.
            gpu_ms = end_event.time_since(start_event)

        return outputs, gpu_ms

    @override
    def run_voxel_encoder(self, input_features: torch.Tensor) -> torch.Tensor:
        """Run voxel encoder using TensorRT.

        Args:
            input_features: Input features [N, max_points, C].

        Returns:
            Voxel features [N, feature_dim].

        Raises:
            RuntimeError: If context is None (initialization failed).
        """
        engine = self._engines["pts_voxel_encoder"]
        context = self._contexts["pts_voxel_encoder"]
        if context is None:
            raise RuntimeError("pts_voxel_encoder context is None - likely failed to initialize due to GPU OOM")

        input_array = self.to_numpy(input_features, dtype=np.float32)
        input_name, output_name = self._get_io_names(engine, single_output=True)

        outputs, gpu_ms = self._run_engine_inference(context, input_name, input_array, [output_name])
        self._gpu_stage_ms["voxel_encoder_ms"] = gpu_ms

        voxel_features = torch.from_numpy(outputs[output_name]).to(self.torch_device)
        return self.squeeze_voxel_features(voxel_features)

    @override
    def run_backbone_head(self, spatial_features: torch.Tensor) -> List[torch.Tensor]:
        """Run backbone and head using TensorRT.

        Args:
            spatial_features: Spatial features [B, C, H, W].

        Returns:
            List of 6 head output tensors.

        Raises:
            RuntimeError: If context is None (initialization failed).
            ValueError: If the engine outputs don't match the configured head outputs.
        """
        engine = self._engines["pts_backbone_neck_head"]
        context = self._contexts["pts_backbone_neck_head"]
        if context is None:
            raise RuntimeError("pts_backbone_neck_head context is None - likely failed to initialize due to GPU OOM")

        input_array = self.to_numpy(spatial_features, dtype=np.float32)
        input_name, trt_output_names = self._get_io_names(engine, single_output=False)

        expected_output_names = [
            out.name for out in self._components_cfg.get_component("pts_backbone_neck_head").io.outputs
        ]
        # Validate and order outputs (CenterPoint postprocess depends on the config order).
        output_names = self.order_head_outputs(trt_output_names, expected_output_names)

        outputs, gpu_ms = self._run_engine_inference(context, input_name, input_array, output_names)
        self._gpu_stage_ms["backbone_head_ms"] = gpu_ms

        return [torch.from_numpy(outputs[name]).to(self.torch_device) for name in output_names]

    @override
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
        stage_latencies: Dict[str, float] = {}

        # Stage 1: Voxel Encoder (pure-GPU time recorded inside run_voxel_encoder).
        voxel_features = self.run_voxel_encoder(preprocessed_input["input_features"])
        stage_latencies["voxel_encoder_ms"] = self._gpu_stage_ms["voxel_encoder_ms"]

        # Stage 2: Middle Encoder (PyTorch, wall-clock).
        start = time.perf_counter()
        spatial_features = self.process_middle_encoder(voxel_features, preprocessed_input["coors"])
        stage_latencies["middle_encoder_ms"] = (time.perf_counter() - start) * 1000

        # Stage 3: Backbone + Head (pure-GPU time recorded inside run_backbone_head).
        head_outputs = self.run_backbone_head(spatial_features)
        stage_latencies["backbone_head_ms"] = self._gpu_stage_ms["backbone_head_ms"]

        return head_outputs, stage_latencies

    @override
    def periodic_cleanup(self, sample_idx: int) -> None:
        """Free the CUDA cache every ``_GPU_CLEANUP_INTERVAL`` samples during long eval loops."""
        if sample_idx > 0 and sample_idx % self._GPU_CLEANUP_INTERVAL == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _release_gpu_resources(self) -> None:
        """Release TensorRT resources (engines and contexts)."""
        release_tensorrt_resources(
            engines=self._engines,
            contexts=self._contexts,
        )
