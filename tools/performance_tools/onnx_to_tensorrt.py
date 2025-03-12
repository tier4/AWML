import argparse
import logging
from typing import List, Optional

import tensorrt as trt

# Configure logger
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more verbosity
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_engine(
    onnx_file_path: str,
    engine_file_path: str,
    fp16_mode: bool = True,
    workspace_size: int = 2,
    max_dynamic_shape: Optional[List[int]] = None,
) -> None:
    """Converts ONNX model to TensorRT engine."""
    if max_dynamic_shape is None:
        max_dynamic_shape = []

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
                logger.error(parser.get_error(error))
            logger.error("ONNX parsing failed!")
            return

    # Handle dynamic input shapes (if any)
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        shape = input_tensor.shape
        if -1 in shape:  # Dynamic shape detected
            logger.info(f"Dynamic shape detected: {shape}. {max_dynamic_shape} will be used")
            min_shape = [s if s != -1 else max_dynamic_shape[i] for i, s in enumerate(shape)]
            opt_shape = [s if s != -1 else max_dynamic_shape[i] for i, s in enumerate(shape)]
            max_shape = [s if s != -1 else max_dynamic_shape[i] for i, s in enumerate(shape)]

            profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)

    # Build serialized network
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        logger.error("Failed to build TensorRT engine!")
        return

    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)

    logger.info(f"Successfully created TensorRT engine: {engine_file_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT Engine")
    parser.add_argument("onnx_file", type=str, help="Path to ONNX model file")
    parser.add_argument("engine_file", type=str, help="Path to save TensorRT engine")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 precision")
    parser.add_argument("--workspace", type=int, default=8, help="Workspace size in GB")
    parser.add_argument(
        "--max_dynamic_shape",
        type=int,
        nargs="+",
        default=None,
        help="Max sizes for dynamic axes (provide space-separated integers)",
    )

    args = parser.parse_args()
    build_engine(args.onnx_file, args.engine_file, args.fp16, args.workspace, args.max_dynamic_shape)


if __name__ == "__main__":
    main()
