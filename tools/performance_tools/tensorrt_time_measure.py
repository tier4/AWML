import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import argparse

def load_engine(engine_path):
    """Load a serialized TensorRT engine from file."""
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine, context):
    """Allocate input and output buffers for the TensorRT engine."""
    inputs = {}
    outputs = {}
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

    return inputs, outputs, stream

def infer(engine, context, inputs, outputs, stream, iterations=100):
    """Run inference using execute_async_v3 and measure execution time with statistics."""

    # Generate random input data and copy it to the device
    for name, inp in inputs.items():
        inp["host"] = np.random.random(inp["host"].shape).astype(inp["host"].dtype)
        cuda.memcpy_htod_async(inp["device"], inp["host"], stream)

    # Warm-up run
    context.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()

    # Timing variables
    times = []

    # Run inference multiple times
    for _ in range(iterations):
        start_time = time.perf_counter()
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()
        end_time = time.perf_counter()

        times.append((end_time - start_time) * 1000)  # Convert to milliseconds

    # Copy outputs back to host
    for name, out in outputs.items():
        cuda.memcpy_dtoh_async(out["host"], out["device"], stream)

    stream.synchronize()

    # Compute statistics
    mean_time = np.mean(times)
    std_dev = np.std(times)
    percentiles = np.percentile(times, [50, 80, 90, 95, 99])

    # Print formatted results
    print("\nInference Execution Time Statistics:")
    print("-" * 50)
    print(f"Iterations       : {iterations}")
    print(f"Mean Time        : {mean_time:.3f} ms")
    print(f"Standard Dev.    : {std_dev:.3f} ms")
    print(f"50th Percentile  : {percentiles[0]:.3f} ms")
    print(f"80th Percentile  : {percentiles[1]:.3f} ms")
    print(f"90th Percentile  : {percentiles[2]:.3f} ms")
    print(f"95th Percentile  : {percentiles[3]:.3f} ms")
    print(f"99th Percentile  : {percentiles[4]:.3f} ms")
    print("-" * 50)


def get_device_info(device_id=0):
    """Print the GPU device being used."""
    cuda.init()
    device = cuda.Device(device_id)  # Assume using the first GPU
    print(f"Using GPU: {device.name()} (Compute Capability: {device.compute_capability()})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TensorRT Inference Script")
    parser.add_argument("--engine_path", type=str, required=True, help="Path to the TensorRT engine file")
    parser.add_argument("--iterations", type=int, default=100, help="Number of inference iterations")
    args = parser.parse_args()

    get_device_info()
    
    engine = load_engine(args.engine_path)
    context = engine.create_execution_context()
    inputs, outputs, stream = allocate_buffers(engine, context)
    infer(engine, context, inputs, outputs, stream, iterations=100)

# CUDA_VISIBLE_DEVICES=1 python3 tools/performance_tools/tensorrt_time_measure.py --engine_path work_dirs/yolox_s_tlr_416x416_pedcar_t4dataset/tlr_car_ped_yolox_s_batch_6.engine --iterations 1000