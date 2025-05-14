import argparse
import logging
import time
from typing import Dict, Tuple

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_engine(engine_path: str) -> trt.ICudaEngine:
    """Load a serialized TensorRT engine from file."""
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        logger.info(f"Loading TensorRT engine from: {engine_path}")
        return runtime.deserialize_cuda_engine(f.read())


def allocate_buffers(
    engine: trt.ICudaEngine, context: trt.IExecutionContext
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]], cuda.Stream]:
    """Allocate input and output buffers for the TensorRT engine."""
    logger.info("Allocating buffers for TensorRT engine...")

    inputs: Dict[str, Dict[str, np.ndarray]] = {}
    outputs: Dict[str, Dict[str, np.ndarray]] = {}
    stream = cuda.Stream()

    for idx in range(engine.num_io_tensors):
        name = engine.get_tensor_name(idx)
        size = trt.volume(engine.get_tensor_shape(name))
        dtype = trt.nptype(engine.get_tensor_dtype(name))

        # Allocate host and device memory
        host_mem = np.empty(size, dtype=dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # Assign buffers to dict based on input/output mode
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs[name] = {"host": host_mem, "device": device_mem}
        else:
            outputs[name] = {"host": host_mem, "device": device_mem}

        # Set tensor address for execution context
        context.set_tensor_address(name, int(device_mem))
        logger.debug(f"Allocated buffer for tensor: {name}, shape: {engine.get_tensor_shape(name)}, dtype: {dtype}")

    logger.info("Buffer allocation completed.")
    return inputs, outputs, stream


def infer(
    engine: trt.ICudaEngine,
    context: trt.IExecutionContext,
    inputs: Dict[str, Dict[str, np.ndarray]],
    outputs: Dict[str, Dict[str, np.ndarray]],
    stream: cuda.Stream,
    iterations: int = 100,
) -> Dict[str, float]:
    """Run inference using execute_async_v3 and measure execution time with statistics."""

    logger.info(f"Running inference for {iterations} iterations...")

    # Generate random input data and copy it to the device
    for inp in inputs.values():
        inp["host"] = np.random.random(inp["host"].shape).astype(inp["host"].dtype)
        cuda.memcpy_htod_async(inp["device"], inp["host"], stream)

    # Warm-up run
    context.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()

    times = []

    # Run inference multiple times
    for _ in range(iterations):
        start_time = time.perf_counter()
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()
        end_time = time.perf_counter()

        times.append((end_time - start_time) * 1000)  # Convert to milliseconds

    # Copy outputs back to host
    for out in outputs.values():
        cuda.memcpy_dtoh_async(out["host"], out["device"], stream)
    stream.synchronize()

    # Compute statistics
    mean_time = np.mean(times)
    std_dev = np.std(times)
    percentiles = np.percentile(times, [50, 80, 90, 95, 99])

    results = {
        "iterations": iterations,
        "mean_time": mean_time,
        "std_dev": std_dev,
        "50th_percentile": percentiles[0],
        "80th_percentile": percentiles[1],
        "90th_percentile": percentiles[2],
        "95th_percentile": percentiles[3],
        "99th_percentile": percentiles[4],
    }

    # Log formatted results
    logger.info("\nInference Execution Time Statistics:")
    logger.info("-" * 50)
    for key, value in results.items():
        logger.info(f"{key.replace('_', ' ').title():<20}: {value:.3f} ms")
    logger.info("-" * 50)

    return results


def get_device_info(device_id: int = 0) -> None:
    """Log the GPU device being used."""
    cuda.init()
    device = cuda.Device(device_id)
    logger.info(f"Using GPU: {device.name()} (Compute Capability: {device.compute_capability()})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TensorRT Inference Script")
    parser.add_argument("--engine_path", type=str, required=True, help="Path to the TensorRT engine file")
    parser.add_argument("--iterations", type=int, default=100, help="Number of inference iterations")
    args = parser.parse_args()

    get_device_info()

    engine = load_engine(args.engine_path)
    context = engine.create_execution_context()
    inputs, outputs, stream = allocate_buffers(engine, context)
    infer(engine, context, inputs, outputs, stream, iterations=args.iterations)
