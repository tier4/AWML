"""
CenterPoint TensorRT Pipeline Implementation.

Moved from deployment/pipelines/centerpoint/centerpoint_tensorrt.py into the CenterPoint deployment bundle.
"""

import logging
import os.path as osp
from typing import List

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
import torch

from deployment.pipelines.gpu_resource_mixin import (
    GPUResourceMixin,
    TensorRTResourceManager,
    release_tensorrt_resources,
)
from deployment.projects.centerpoint.config.deploy_config import onnx_config
from deployment.projects.centerpoint.pipelines.centerpoint_pipeline import CenterPointDeploymentPipeline

logger = logging.getLogger(__name__)


class CenterPointTensorRTPipeline(GPUResourceMixin, CenterPointDeploymentPipeline):
    def __init__(self, pytorch_model, tensorrt_dir: str, device: str = "cuda"):
        if not device.startswith("cuda"):
            raise ValueError("TensorRT requires CUDA device")

        super().__init__(pytorch_model, device, backend_type="tensorrt")

        self.tensorrt_dir = tensorrt_dir
        self._engines = {}
        self._contexts = {}
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._cleanup_called = False

        self._load_tensorrt_engines()
        logger.info(f"TensorRT pipeline initialized with engines from: {tensorrt_dir}")

    def _load_tensorrt_engines(self):
        trt.init_libnvinfer_plugins(self._logger, "")
        runtime = trt.Runtime(self._logger)

        component_cfg = onnx_config.get("components", {})
        voxel_cfg = component_cfg.get("voxel_encoder", {})
        backbone_cfg = component_cfg.get("backbone_head", {})
        engine_files = {
            "voxel_encoder": voxel_cfg.get("engine_file", "pts_voxel_encoder.engine"),
            "backbone_neck_head": backbone_cfg.get("engine_file", "pts_backbone_neck_head.engine"),
        }

        for component, engine_file in engine_files.items():
            engine_path = osp.join(self.tensorrt_dir, engine_file)
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

    def run_voxel_encoder(self, input_features: torch.Tensor) -> torch.Tensor:
        engine = self._engines["voxel_encoder"]
        context = self._contexts["voxel_encoder"]
        if context is None:
            raise RuntimeError("voxel_encoder context is None - likely failed to initialize due to GPU OOM")

        input_array = input_features.cpu().numpy().astype(np.float32)
        if not input_array.flags["C_CONTIGUOUS"]:
            input_array = np.ascontiguousarray(input_array)

        input_name, output_name = self._get_io_names(engine, single_output=True)
        context.set_input_shape(input_name, input_array.shape)
        output_shape = context.get_tensor_shape(output_name)
        output_array = np.empty(output_shape, dtype=np.float32)
        if not output_array.flags["C_CONTIGUOUS"]:
            output_array = np.ascontiguousarray(output_array)

        with TensorRTResourceManager() as manager:
            d_input = manager.allocate(input_array.nbytes)
            d_output = manager.allocate(output_array.nbytes)
            stream = manager.get_stream()

            context.set_tensor_address(input_name, int(d_input))
            context.set_tensor_address(output_name, int(d_output))

            cuda.memcpy_htod_async(d_input, input_array, stream)
            context.execute_async_v3(stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(output_array, d_output, stream)
            manager.synchronize()

        voxel_features = torch.from_numpy(output_array).to(self.device)
        voxel_features = voxel_features.squeeze(1)
        return voxel_features

    def _get_io_names(self, engine, single_output: bool = False):
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

    def run_backbone_head(self, spatial_features: torch.Tensor) -> List[torch.Tensor]:
        engine = self._engines["backbone_neck_head"]
        context = self._contexts["backbone_neck_head"]
        if context is None:
            raise RuntimeError("backbone_neck_head context is None - likely failed to initialize due to GPU OOM")

        input_array = spatial_features.cpu().numpy().astype(np.float32)
        if not input_array.flags["C_CONTIGUOUS"]:
            input_array = np.ascontiguousarray(input_array)

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
            stream = manager.get_stream()

            context.set_tensor_address(input_name, int(d_input))
            for output_name in output_names:
                context.set_tensor_address(output_name, int(d_outputs[output_name]))

            cuda.memcpy_htod_async(d_input, input_array, stream)
            context.execute_async_v3(stream_handle=stream.handle)

            for output_name in output_names:
                cuda.memcpy_dtoh_async(output_arrays[output_name], d_outputs[output_name], stream)
            manager.synchronize()

        head_outputs = [torch.from_numpy(output_arrays[name]).to(self.device) for name in output_names]
        if len(head_outputs) != 6:
            raise ValueError(f"Expected 6 head outputs, got {len(head_outputs)}")
        return head_outputs

    def _release_gpu_resources(self) -> None:
        release_tensorrt_resources(
            engines=getattr(self, "_engines", None),
            contexts=getattr(self, "_contexts", None),
        )
