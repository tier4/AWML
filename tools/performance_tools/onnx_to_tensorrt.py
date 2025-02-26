import tensorrt as trt
import argparse

def build_engine(onnx_file_path, engine_file_path, fp16_mode=False, workspace_size=16):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Read the ONNX file
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse the ONNX file.")

    # Configure builder
    config = builder.create_builder_config()
    config.max_workspace_size = workspace_size * (1 << 30)  # Convert GB to bytes
    if fp16_mode:
        print("Using FP16 mode")
        config.set_flag(trt.BuilderFlag.FP16)

    # Build and save the engine
    with builder.build_engine(network, config) as engine:
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
    print("Completed building TensorRT engine.")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ONNX model to TensorRT engine.")
    parser.add_argument("onnx_file", type=str, help="Path to the ONNX model file.")
    parser.add_argument("engine_file", type=str, help="Path to save the TensorRT engine file.")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 precision mode.")
    parser.add_argument("--workspace", type=int, default=4, help="Workspace size in GB (default: 4GB).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_engine(args.onnx_file, args.engine_file, args.fp16, args.workspace)


# python3 tools/performance_tools/onnx_to_tensorrt.py work_dirs/yolox_s_tlr_416x416_pedcar_t4dataset/tlr_car_ped_yolox_s_batch_6.onnx work_dirs/yolox_s_tlr_416x416_pedcar_t4dataset/tlr_car_ped_yolox_s_batch_6.engine
