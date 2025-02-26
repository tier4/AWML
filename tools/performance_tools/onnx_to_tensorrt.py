import tensorrt as trt
import argparse

def build_engine(onnx_file_path, engine_file_path, fp16_mode=True, workspace_size=2, max_dynamic_shape=[]):
    """Converts ONNX model to TensorRT engine."""
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size * (1 << 30))  # Convert GB to bytes

    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)

    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("ONNX parsing failed!")

    # Handle dynamic input shapes (if any)
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        shape = input_tensor.shape
        if -1 in shape:  # Dynamic shape detected
            print(f"dynamic shape detected: {shape}. {max_dynamic_shape} will be used")
            min_shape = [s if s != -1 else max_dynamic_shape[i]  for i,s in enumerate(shape)]
            opt_shape = [s if s != -1 else max_dynamic_shape[i]  for i,s in enumerate(shape)]
            max_shape = [s if s != -1 else max_dynamic_shape[i]  for i,s in enumerate(shape)]

            profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)

    # Build serialized network
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine!")

    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)

    print(f"Successfully created TensorRT engine: {engine_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT Engine")
    parser.add_argument("onnx_file", type=str, help="Path to ONNX model file")
    parser.add_argument("engine_file", type=str, help="Path to save TensorRT engine")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 precision")
    parser.add_argument("--workspace", type=int, default=8, help="Workspace size in GB")
    parser.add_argument("--max_dynamic_shape", type=int, nargs='+', default=[], help="Max sizes for dynamic axes (provide space-separated integers)")


    args = parser.parse_args()
    build_engine(args.onnx_file, args.engine_file, args.fp16, args.workspace, args.max_dynamic_shape)
