"""Shared TensorRT engine runner.

One battle-tested implementation of the TensorRT run loop — allocate device buffers,
copy host->device, execute with CUDA-event timing, copy device->host — reused by every
per-project TensorRT pipeline (BEVFusion, CenterPoint). Project pipelines keep only their
model-specific pieces (which engine, how to name/order inputs and outputs); the GPU plumbing
lives here so it cannot drift between backends.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

from deployment.inference.gpu_resource_mixin import TensorRTResourceManager

logger = logging.getLogger(__name__)


def list_trt_io_names(engine: trt.ICudaEngine) -> Tuple[List[str], List[str]]:
    """Return ``(input_names, output_names)`` in TensorRT tensor-index order.

    Shared by every per-project TensorRT pipeline so input/output discovery cannot drift
    between backends.
    """
    inputs: List[str] = []
    outputs: List[str] = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append(name)
        else:
            outputs.append(name)
    return inputs, outputs


def load_trt_engine(
    runtime: trt.Runtime,
    engine_path: str,
    *,
    component_name: Optional[str] = None,
) -> Tuple[trt.ICudaEngine, trt.IExecutionContext]:
    """Deserialize a TensorRT engine and create its execution context, failing loud.

    One implementation of the deserialize -> null-check -> create-context -> null-check
    boilerplate, reused by every per-project TensorRT pipeline so the error messages (and the
    OOM hint) stay identical across backends.

    Args:
        runtime: TensorRT runtime used to deserialize the engine.
        engine_path: Path to the serialized ``.engine`` file.
        component_name: Optional component label for error messages (defaults to ``engine_path``).

    Returns:
        Tuple of (engine, execution context).

    Raises:
        RuntimeError: If deserialization or context creation fails (context failure is usually OOM).
    """
    label = component_name or engine_path
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(f"Failed to create TensorRT execution context for {label} (likely GPU out-of-memory).")

    return engine, context


def _trt_dtype_to_numpy(trt_dtype: trt.DataType) -> np.dtype:
    """Return the numpy dtype matching a TensorRT dtype, for correctly sized host buffers.

    Delegates to TensorRT's own ``nptype`` mapping and lets it raise for a dtype it cannot map:
    failing loud is safer than guessing a size (e.g. defaulting to float32), which would
    mis-size the GPU buffer and silently corrupt the data.
    """
    return np.dtype(trt.nptype(trt_dtype))


def _cast_to_binding_dtype(engine: trt.ICudaEngine, tensor_name: str, arr: np.ndarray) -> np.ndarray:
    """Return ``arr`` as a C-contiguous buffer whose dtype matches the engine binding.

    Matching the binding dtype is critical for FP16 engines: split sparse ONNX is often traced
    with FP32 voxels, but TensorRT ``fp16`` builds may bind ``voxels`` as ``HALF``. Feeding
    float32 nbytes into a HALF binding misaligns the GPU buffer and corrupts the first
    ImplicitGemm inputs (lidar_bev explosion while numpy voxel stats still look sane). Returns
    ``arr`` unchanged when it already matches the binding dtype and is contiguous.
    """
    binding_dtype = engine.get_tensor_dtype(tensor_name)
    target_dtype = _trt_dtype_to_numpy(binding_dtype)
    if arr.dtype != target_dtype:
        logger.info(
            "[trt-io] casting host buffer for tensor %r: numpy %s → %s (engine binding %s)",
            tensor_name,
            arr.dtype,
            target_dtype,
            binding_dtype,
        )
        arr = np.asarray(arr, dtype=target_dtype)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def run_trt_engine(
    engine: trt.ICudaEngine,
    context: trt.IExecutionContext,
    inputs_by_name: Dict[str, np.ndarray],
    output_names: List[str],
) -> Tuple[Dict[str, np.ndarray], float]:
    """Run one engine end-to-end and return (outputs-by-name, pure-GPU time in ms).

    Handles all the dtype bookkeeping so callers pass plain arrays: input buffers are cast to
    each binding's dtype and output buffers are allocated with the engine's actual output dtype,
    so the same code path serves FP32 and FP16 engines. Timing uses CUDA events bracketing only
    ``execute_async_v3`` on one stream, so the returned time is the engine's GPU compute and
    excludes the H2D/D2H copies; it is read back while that stream is still alive.

    Args:
        engine: TensorRT engine that owns ``context`` (needed to query binding dtypes/shapes).
        context: Execution context; input shapes are set from ``inputs_by_name``.
        inputs_by_name: Engine input tensor name -> host ndarray. A single-input engine is just a
            one-entry map.
        output_names: Engine output tensor names, in the desired return order.

    Returns:
        Tuple of (outputs-by-name as host ndarrays, pure-GPU time in ms).

    Raises:
        RuntimeError: If ``execute_async_v3`` reports a failure status.
    """
    inputs_by_name = {name: _cast_to_binding_dtype(engine, name, arr) for name, arr in inputs_by_name.items()}
    for name, arr in inputs_by_name.items():
        context.set_input_shape(name, arr.shape)

    # Output shapes can depend on the input shape, so read them only after set_input_shape.
    host_outputs: Dict[str, np.ndarray] = {}
    for name in output_names:
        shape = context.get_tensor_shape(name)
        arr = np.empty(shape, dtype=_trt_dtype_to_numpy(engine.get_tensor_dtype(name)))
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        host_outputs[name] = arr

    with TensorRTResourceManager() as resources:
        device_inputs = {name: resources.allocate(arr.nbytes) for name, arr in inputs_by_name.items()}
        device_outputs = {name: resources.allocate(arr.nbytes) for name, arr in host_outputs.items()}
        stream = resources.stream

        for name, arr in inputs_by_name.items():
            context.set_tensor_address(name, int(device_inputs[name]))
            cuda.memcpy_htod_async(device_inputs[name], arr, stream)

        for name in output_names:
            context.set_tensor_address(name, int(device_outputs[name]))

        start_event = cuda.Event()
        end_event = cuda.Event()
        start_event.record(stream)
        succeeded = context.execute_async_v3(stream_handle=stream.handle)
        if not succeeded:
            raise RuntimeError("TensorRT execute_async_v3 returned failure status.")
        end_event.record(stream)

        for name in output_names:
            cuda.memcpy_dtoh_async(host_outputs[name], device_outputs[name], stream)

        resources.synchronize()
        gpu_time_ms = float(end_event.time_since(start_event))

    return host_outputs, gpu_time_ms
