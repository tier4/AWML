"""TensorRT model wrapper for FRNet deployment.

Builds a TensorRT engine from ONNX and runs inference with PyCUDA.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import pycuda.autoinit  # noqa: F401 – required to initialise the CUDA context
import pycuda.driver as cuda
import tensorrt as trt
from mmengine.config import Config
from mmengine.logging import MMLogger


class TrtModel:
    """FRNet TensorRT model wrapper.

    Optionally builds the engine from ONNX on construction (when deploy=True),
    then loads it for inference.  The engine file is saved as frnet.engine
    next to the ONNX file.
    """

    def __init__(
        self,
        deploy_cfg: Config,
        onnx_path: str,
        deploy: bool = True,
        verbose: bool = False,
    ) -> None:
        self._deploy_cfg = deploy_cfg
        self.logger = MMLogger.get_current_instance()
        self._trt_logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self._trt_logger, "")

        self._start = cuda.Event()
        self._end = cuda.Event()
        self._stream = cuda.Stream()

        if deploy:
            self._engine = self._build_engine(onnx_path)
        else:
            self._engine = self._load_engine(onnx_path)

    def _build_engine(self, onnx_path: str) -> trt.ICudaEngine:
        """Build a TensorRT engine from an ONNX model and save it to disk."""
        runtime = trt.Runtime(self._trt_logger)
        builder = trt.Builder(self._trt_logger)

        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(pool=trt.MemoryPoolType.WORKSPACE, pool_size=1 << 32)

        # Optimisation profile (dynamic shapes)
        profile = builder.create_optimization_profile()
        for name, shapes in self._deploy_cfg.tensorrt_config.items():
            profile.set_shape(name, shapes["min_shape"], shapes["opt_shape"], shapes["max_shape"])
        config.add_optimization_profile(profile)

        # Parse ONNX
        parser = trt.OnnxParser(network, self._trt_logger)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                self.logger.error("Failed to parse the ONNX file")
                for i in range(parser.num_errors):
                    self.logger.error(parser.get_error(i))
            else:
                self.logger.info("Successfully parsed the ONNX file")

        # Serialise engine
        serialized_engine = builder.build_serialized_network(network, config)
        engine_path = os.path.join(os.path.dirname(onnx_path), "frnet.engine")
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
            self.logger.info(f"TensorRT engine saved to {engine_path}")

        return runtime.deserialize_cuda_engine(serialized_engine)

    def _load_engine(self, onnx_path: str) -> trt.ICudaEngine:
        """Load a pre-built TensorRT engine from disk."""
        runtime = trt.Runtime(self._trt_logger)
        engine_path = os.path.join(os.path.dirname(onnx_path), "frnet.engine")
        with open(engine_path, "rb") as f:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self, shapes_dict: Dict[str, Tuple[int, ...]]) -> Dict[str, Dict]:
        """Allocate GPU buffers for all input and output tensors."""
        tensors: Dict[str, Dict] = {"input": {}, "output": {}}

        def _alloc(target: Dict, indices: List[int]) -> None:
            for i in indices:
                name = self._engine.get_tensor_name(i)
                dtype = trt.nptype(self._engine.get_tensor_dtype(name))
                shape = shapes_dict[name]
                if len(shape) > 1:
                    assert (
                        shape[-1] == self._engine.get_tensor_shape(name)[-1]
                    ), f"Last dim of {shape} != engine shape {self._engine.get_tensor_shape(name)}"
                size = trt.volume(shape) * np.array(1, dtype=dtype).itemsize
                target[name] = {"device_ptr": cuda.mem_alloc(size), "shape": shape}

        _alloc(tensors["input"], [0, 1, 2, 3])
        _alloc(tensors["output"], [4])
        return tensors

    def _transfer_input_to_device(self, batch_inputs_dict: dict, input_tensors: Dict) -> None:
        """Copy input tensors from host to device."""
        input_data = [
            batch_inputs_dict["points"],
            batch_inputs_dict["coors"],
            batch_inputs_dict["voxel_coors"],
            batch_inputs_dict["inverse_map"],
        ]
        for (device_ptr, shape), data in zip(
            ((v["device_ptr"], v["shape"]) for v in input_tensors.values()),
            input_data,
        ):
            np_data = np.array(data, dtype=data.numpy().dtype).reshape(shape)
            cuda.memcpy_htod_async(device_ptr, np_data, self._stream)
        self._stream.synchronize()

    def _transfer_output_from_device(self, output_tensors: Dict) -> npt.NDArray[np.float32]:
        """Copy the first output tensor from device to host."""
        results = []
        for value in output_tensors.values():
            np_output = np.empty(value["shape"], dtype=np.float32)
            cuda.memcpy_dtoh_async(np_output, value["device_ptr"], self._stream)
            results.append(np_output)
        self._stream.synchronize()
        return results[0]

    def _run_engine(self, tensors: Dict[str, Dict]) -> None:
        """Execute the TensorRT engine."""
        with self._engine.create_execution_context() as context:
            for key, value in tensors["input"].items():
                context.set_input_shape(key, value["shape"])
                context.set_tensor_address(key, int(value["device_ptr"]))
            for key, value in tensors["output"].items():
                context.set_tensor_address(key, int(value["device_ptr"]))

            self._start.record(self._stream)
            context.execute_async_v3(stream_handle=self._stream.handle)
            self._end.record(self._stream)
            self._stream.synchronize()

            latency = self._end.time_since(self._start)
            self.logger.info(f"Inference latency: {latency} ms")

    def inference(self, batch_inputs_dict: dict) -> npt.NDArray[np.float32]:
        """Run TensorRT inference, returns logits (N, num_classes)."""
        shapes_dict = {
            "points": batch_inputs_dict["points"].shape,
            "coors": batch_inputs_dict["coors"].shape,
            "voxel_coors": batch_inputs_dict["voxel_coors"].shape,
            "inverse_map": batch_inputs_dict["inverse_map"].shape,
            "seg_logit": (batch_inputs_dict["points"].shape[0], self._deploy_cfg.num_classes),
        }
        tensors = self._allocate_buffers(shapes_dict)
        self._transfer_input_to_device(batch_inputs_dict, tensors["input"])
        self._run_engine(tensors)
        return self._transfer_output_from_device(tensors["output"])
