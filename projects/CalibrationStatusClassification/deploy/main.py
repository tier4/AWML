"""
CalibrationStatusClassification Model Deployment Script

This script exports CalibrationStatusClassification models to ONNX and TensorRT formats,
with comprehensive verification and performance benchmarking.

Features:
- ONNX export with optimization
- TensorRT conversion
- Dual verification (ONNX + TensorRT)
- Performance benchmarking
"""

import argparse
import logging
import os
import os.path as osp
import pickle
import time
from typing import Any, Dict, Optional, Tuple

import mmengine
import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
import pycuda.driver as cuda
import tensorrt as trt
import torch
from mmengine.config import Config
from mmpretrain.apis import get_model

from autoware_ml.calibration_classification.datasets.transforms.calibration_classification_transform import (
    CalibrationClassificationTransform,
)

# Constants
DEFAULT_VERIFICATION_TOLERANCE = 1e-3
DEFAULT_WORKSPACE_SIZE = 1 << 30  # 1 GB
EXPECTED_CHANNELS = 5  # RGB + Depth + Intensity
LABELS = {"0": "miscalibrated", "1": "calibrated"}


def load_info_pkl_data(info_pkl_path: str, sample_idx: int = 0) -> Dict[str, Any]:
    """
    Load a single sample from info.pkl file.

    Args:
        info_pkl_path: Path to the info.pkl file
        sample_idx: Index of the sample to load (default: 0)

    Returns:
        Sample dictionary with the required structure for CalibrationClassificationTransform

    Raises:
        FileNotFoundError: If info.pkl file doesn't exist
        ValueError: If data format is unexpected or sample index is invalid
    """
    if not os.path.exists(info_pkl_path):
        raise FileNotFoundError(f"Info.pkl file not found: {info_pkl_path}")

    try:
        with open(info_pkl_path, "rb") as f:
            info_data = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load info.pkl file: {e}")

    # Extract samples from info.pkl
    if isinstance(info_data, dict):
        if "data_list" in info_data:
            samples_list = info_data["data_list"]
        else:
            raise ValueError(f"Expected 'data_list' key in info_data, found keys: {list(info_data.keys())}")
    else:
        raise ValueError(f"Expected dict format, got {type(info_data)}")

    if not samples_list:
        raise ValueError("No samples found in info.pkl")

    if sample_idx >= len(samples_list):
        raise ValueError(f"Sample index {sample_idx} out of range (0-{len(samples_list)-1})")

    sample = samples_list[sample_idx]

    # Validate sample structure
    required_keys = ["image", "lidar_points"]
    if not all(key in sample for key in required_keys):
        raise ValueError(f"Sample {sample_idx} has invalid structure. Required keys: {required_keys}")

    return sample


def load_sample_data_from_info_pkl(
    info_pkl_path: str,
    model_cfg: Config,
    sample_idx: int = 0,
    force_generate_miscalibration: bool = False,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Load and preprocess sample data from info.pkl using CalibrationClassificationTransform.

    Args:
        info_pkl_path: Path to the info.pkl file
        model_cfg: Model configuration containing data_root setting
        sample_idx: Index of the sample to load (default: 0)
        force_generate_miscalibration: Whether to force generation of miscalibration
        device: Device to load tensor on

    Returns:
        Preprocessed tensor ready for model inference
    """
    # Load sample data from info.pkl
    sample_data = load_info_pkl_data(info_pkl_path, sample_idx)

    # Get data_root from model config
    data_root = model_cfg.get("data_root", None)
    if data_root is None:
        raise ValueError("data_root not found in model configuration")

    # Create transform for deployment
    mode = "test" if not force_generate_miscalibration else "train"
    transform = CalibrationClassificationTransform(
        mode=mode,
        data_root=data_root,
        projection_vis_dir=None,
        results_vis_dir=None,
        enable_augmentation=False,
    )

    # Apply transform
    results = transform.transform(sample_data, force_generate_miscalibration=force_generate_miscalibration)
    input_data_processed = results["fused_img"]  # (H, W, 5)

    # Convert to tensor
    input_tensor = torch.from_numpy(input_data_processed).permute(2, 0, 1).float()  # (5, H, W)
    input_tensor = input_tensor.unsqueeze(0).to(device)  # (1, 5, H, W)

    return input_tensor


class DeploymentConfig:
    """Configuration container for deployment settings."""

    def __init__(self, deploy_cfg: Config):
        self.deploy_cfg = deploy_cfg
        self.backend_config = deploy_cfg.get("backend_config", {})
        self.onnx_config = deploy_cfg.get("onnx_config", {})

    @property
    def onnx_settings(self) -> Dict:
        """Get ONNX export settings."""
        return {
            "opset_version": self.onnx_config.get("opset_version", 11),
            "do_constant_folding": self.onnx_config.get("do_constant_folding", True),
            "input_names": self.onnx_config.get("input_names", ["input"]),
            "output_names": self.onnx_config.get("output_names", ["output"]),
            "dynamic_axes": self.onnx_config.get("dynamic_axes"),
            "export_params": self.onnx_config.get("export_params", True),
            "keep_initializers_as_inputs": self.onnx_config.get("keep_initializers_as_inputs", False),
            "save_file": self.onnx_config.get("save_file", "calibration_classifier.onnx"),
        }

    @property
    def tensorrt_settings(self) -> Dict:
        """Get TensorRT export settings."""
        common_config = self.backend_config.get("common_config", {})
        return {
            "max_workspace_size": common_config.get("max_workspace_size", DEFAULT_WORKSPACE_SIZE),
            "fp16_mode": common_config.get("fp16_mode", False),
            "model_inputs": self.backend_config.get("model_inputs", []),
        }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export CalibrationStatusClassification model to ONNX/TensorRT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("deploy_cfg", help="deploy config path")
    parser.add_argument("model_cfg", help="model config path")
    parser.add_argument("checkpoint", help="model checkpoint path")
    parser.add_argument("--info_pkl", required=True, help="info.pkl file path containing calibration data")
    parser.add_argument("--sample_idx", type=int, default=0, help="sample index from info.pkl (default: 0)")
    parser.add_argument("--work-dir", default=os.getcwd(), help="output directory")
    parser.add_argument("--device", default="cpu", help="device for conversion")
    parser.add_argument("--log-level", default="INFO", choices=list(logging._nameToLevel.keys()), help="logging level")
    parser.add_argument("--verify", action="store_true", help="verify model outputs")
    return parser.parse_args()


def setup_logging(level: str) -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(level=getattr(logging, level), format="%(levelname)s:%(name)s:%(message)s")
    return logging.getLogger("mmdeploy")


def export_to_onnx(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    output_path: str,
    config: DeploymentConfig,
    logger: logging.Logger,
) -> None:
    """Export model to ONNX format."""
    settings = config.onnx_settings
    model.eval()

    logger.info("Exporting model to ONNX format...")
    logger.info(f"Input shape: {input_tensor.shape}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"ONNX opset version: {settings['opset_version']}")

    with torch.no_grad():
        torch.onnx.export(
            model,
            input_tensor,
            output_path,
            export_params=settings["export_params"],
            keep_initializers_as_inputs=settings["keep_initializers_as_inputs"],
            opset_version=settings["opset_version"],
            do_constant_folding=settings["do_constant_folding"],
            input_names=settings["input_names"],
            output_names=settings["output_names"],
            dynamic_axes=settings["dynamic_axes"],
            verbose=False,
        )

    logger.info(f"ONNX export completed: {output_path}")

    # Optional model simplification
    _optimize_onnx_model(output_path, logger)


def _optimize_onnx_model(onnx_path: str, logger: logging.Logger) -> None:
    """Optimize ONNX model using onnxsim."""
    logger.info("Simplifying ONNX model...")
    model_simplified, success = onnxsim.simplify(onnx_path)
    if success:
        onnx.save(model_simplified, onnx_path)
        logger.info(f"ONNX model simplified successfully. Saved to {onnx_path}")
    else:
        logger.warning("ONNX model simplification failed")


def export_to_tensorrt(
    onnx_path: str, output_path: str, input_tensor: torch.Tensor, config: DeploymentConfig, logger: logging.Logger
) -> bool:
    """Export ONNX model to TensorRT format."""
    settings = config.tensorrt_settings

    # Initialize TensorRT
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    runtime = trt.Runtime(TRT_LOGGER)
    builder = trt.Builder(TRT_LOGGER)

    # Create network and config
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config_trt = builder.create_builder_config()
    config_trt.set_memory_pool_limit(pool=trt.MemoryPoolType.WORKSPACE, pool_size=settings["max_workspace_size"])

    # Enable FP16 if specified
    if settings["fp16_mode"]:
        config_trt.set_flag(trt.BuilderFlag.FP16)
        logger.info("FP16 mode enabled")

    # Setup optimization profile
    profile = builder.create_optimization_profile()
    _configure_input_shapes(profile, input_tensor, settings["model_inputs"], logger)
    config_trt.add_optimization_profile(profile)

    # Parse ONNX model
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            _log_parser_errors(parser, logger)
            return False
        logger.info("Successfully parsed the ONNX file")

    # Build engine
    logger.info("Building TensorRT engine...")
    serialized_engine = builder.build_serialized_network(network, config_trt)

    if serialized_engine is None:
        logger.error("Failed to build TensorRT engine")
        return False

    # Save engine
    with open(output_path, "wb") as f:
        f.write(serialized_engine)

    workspace_gb = settings["max_workspace_size"] / (1024**3)
    logger.info(f"TensorRT engine saved to {output_path}")
    logger.info(f"Engine max workspace size: {workspace_gb:.2f} GB")

    return True


def _configure_input_shapes(profile, input_tensor: torch.Tensor, model_inputs: list, logger: logging.Logger) -> None:
    """Configure input shapes for TensorRT optimization profile."""
    if model_inputs:
        input_shapes = model_inputs[0].get("input_shapes", {})
        for input_name, shapes in input_shapes.items():
            min_shape = shapes.get("min_shape", list(input_tensor.shape))
            opt_shape = shapes.get("opt_shape", list(input_tensor.shape))
            max_shape = shapes.get("max_shape", list(input_tensor.shape))

            logger.info(f"Setting input shapes - min: {min_shape}, opt: {opt_shape}, max: {max_shape}")
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    else:
        # Default shapes based on input tensor
        input_shape = list(input_tensor.shape)
        logger.info(f"Using default input shape: {input_shape}")
        profile.set_shape("input", input_shape, input_shape, input_shape)


def _log_parser_errors(parser, logger: logging.Logger) -> None:
    """Log TensorRT parser errors."""
    logger.error("Failed to parse ONNX model")
    for error in range(parser.num_errors):
        logger.error(f"Parser error: {parser.get_error(error)}")


def run_pytorch_inference(
    model: torch.nn.Module, input_tensor: torch.Tensor, logger: logging.Logger
) -> Tuple[torch.Tensor, float]:
    """Run PyTorch inference on CPU for verification and return output with latency."""
    # Move to CPU to avoid GPU memory issues
    model_cpu = model.cpu()
    input_cpu = input_tensor.cpu()

    model_cpu.eval()
    with torch.no_grad():
        # Measure inference time
        start_time = time.perf_counter()
        output = model_cpu(input_cpu)
        end_time = time.perf_counter()

        latency = (end_time - start_time) * 1000  # Convert to milliseconds

        # Handle different output formats
        if hasattr(output, "output"):
            output = output.output
        elif isinstance(output, dict) and "output" in output:
            output = output["output"]

        if not isinstance(output, torch.Tensor):
            raise ValueError(f"Unexpected PyTorch output type: {type(output)}")

    logger.info(f"PyTorch inference latency: {latency:.2f} ms")
    logger.info(f"Output verification:")
    logger.info(f"  Output: {output.cpu().numpy()}")
    return output, latency


def run_onnx_inference(
    onnx_path: str,
    input_tensor: torch.Tensor,
    ref_output: torch.Tensor,
    logger: logging.Logger,
) -> bool:
    """Verify ONNX model output against PyTorch model."""
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ONNX inference with timing
    providers = ["CPUExecutionProvider"]
    ort_session = ort.InferenceSession(onnx_path, providers=providers)
    onnx_input = {ort_session.get_inputs()[0].name: input_tensor.cpu().numpy()}

    start_time = time.perf_counter()
    onnx_output = ort_session.run(None, onnx_input)[0]
    end_time = time.perf_counter()
    onnx_latency = (end_time - start_time) * 1000

    logger.info(f"ONNX inference latency: {onnx_latency:.2f} ms")

    # Ensure onnx_output is numpy array before comparison
    if not isinstance(onnx_output, np.ndarray):
        logger.error(f"Unexpected ONNX output type: {type(onnx_output)}")
        return False

    # Compare outputs
    return _compare_outputs(ref_output.cpu().numpy(), onnx_output, "ONNX", logger)


def run_tensorrt_inference(
    tensorrt_path: str,
    input_tensor: torch.Tensor,
    ref_output: torch.Tensor,
    logger: logging.Logger,
) -> bool:
    """Verify TensorRT model output against PyTorch model."""
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load TensorRT engine
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    runtime = trt.Runtime(TRT_LOGGER)

    with open(tensorrt_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        logger.error("Failed to deserialize TensorRT engine")
        return False

    # Run TensorRT inference with timing
    trt_output, latency = _run_tensorrt_inference(engine, input_tensor.cpu(), logger)
    logger.info(f"TensorRT inference latency: {latency:.2f} ms")

    # Compare outputs
    return _compare_outputs(ref_output.cpu().numpy(), trt_output, "TensorRT", logger)


def _run_tensorrt_inference(engine, input_tensor: torch.Tensor, logger: logging.Logger) -> Tuple[np.ndarray, float]:
    """Run TensorRT inference and return output with timing."""
    context = engine.create_execution_context()
    stream = cuda.Stream()
    start = cuda.Event()
    end = cuda.Event()

    # Get tensor names and shapes
    input_name, output_name = None, None
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            input_name = tensor_name
        elif engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
            output_name = tensor_name

    if input_name is None or output_name is None:
        raise RuntimeError("Could not find input/output tensor names")

    # Prepare arrays
    input_np = input_tensor.numpy().astype(np.float32)
    if not input_np.flags["C_CONTIGUOUS"]:
        input_np = np.ascontiguousarray(input_np)

    context.set_input_shape(input_name, input_np.shape)
    output_shape = context.get_tensor_shape(output_name)
    output_np = np.empty(output_shape, dtype=np.float32)
    if not output_np.flags["C_CONTIGUOUS"]:
        output_np = np.ascontiguousarray(output_np)

    # Allocate GPU memory
    d_input = cuda.mem_alloc(input_np.nbytes)
    d_output = cuda.mem_alloc(output_np.nbytes)

    try:
        # Set tensor addresses
        context.set_tensor_address(input_name, int(d_input))
        context.set_tensor_address(output_name, int(d_output))

        # Run inference with timing
        cuda.memcpy_htod_async(d_input, input_np, stream)
        start.record(stream)
        context.execute_async_v3(stream_handle=stream.handle)
        end.record(stream)
        cuda.memcpy_dtoh_async(output_np, d_output, stream)
        stream.synchronize()

        latency = end.time_since(start)
        return output_np, latency

    finally:
        # Cleanup
        try:
            d_input.free()
            d_output.free()
        except:
            pass


def _compare_outputs(
    pytorch_output: np.ndarray, backend_output: np.ndarray, backend_name: str, logger: logging.Logger
) -> bool:
    """Compare outputs between PyTorch and backend."""
    if not isinstance(backend_output, np.ndarray):
        logger.error(f"Unexpected {backend_name} output type: {type(backend_output)}")
        return False

    max_diff = np.abs(pytorch_output - backend_output).max()
    mean_diff = np.abs(pytorch_output - backend_output).mean()

    logger.info(f"Output verification:")
    logger.info(f"  {backend_name} output: {backend_output}")
    logger.info(f"  Max difference with PyTorch: {max_diff:.6f}")
    logger.info(f"  Mean difference with PyTorch: {mean_diff:.6f}")

    success = max_diff < DEFAULT_VERIFICATION_TOLERANCE
    if not success:
        logger.warning(f"Large difference detected: {max_diff:.6f}")

    return success


def run_verification(
    model: torch.nn.Module,
    onnx_path: str,
    trt_path: Optional[str],
    input_tensors: Dict[str, torch.Tensor],
    logger: logging.Logger,
) -> None:
    """Run model verification for available backends."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for key, input_tensor in input_tensors.items():
        logger.info("=" * 50)
        logger.info(f"Verifying {LABELS[key]} sample...")
        logger.info("-" * 50)
        logger.info("Verifying PyTorch model...")
        pytorch_output, pytorch_latency = run_pytorch_inference(model, input_tensor, logger)

        if onnx_path and osp.exists(onnx_path):
            logger.info("-" * 50)
            logger.info("Verifying ONNX model...")
            if run_onnx_inference(onnx_path, input_tensor, pytorch_output, logger):
                logger.info("ONNX model verification passed!")
            else:
                logger.error("ONNX model verification failed!")

        if trt_path and osp.exists(trt_path):
            logger.info("-" * 50)
            logger.info("Verifying TensorRT model...")
            if run_tensorrt_inference(trt_path, input_tensor, pytorch_output, logger):
                logger.info("TensorRT model verification passed!")
            else:
                logger.error("TensorRT model verification failed!")
    logger.info("=" * 50)


def main():
    """Main deployment function."""
    args = parse_args()
    logger = setup_logging(args.log_level)

    # Setup
    mmengine.mkdir_or_exist(osp.abspath(args.work_dir))

    # Load configurations
    logger.info(f"Loading deploy config from: {args.deploy_cfg}")
    deploy_cfg = Config.fromfile(args.deploy_cfg)
    config = DeploymentConfig(deploy_cfg)

    logger.info(f"Loading model config from: {args.model_cfg}")
    model_cfg = Config.fromfile(args.model_cfg)

    # Setup file paths
    onnx_settings = config.onnx_settings
    onnx_path = osp.join(args.work_dir, onnx_settings["save_file"])
    trt_path: Optional[str] = None
    trt_file = onnx_settings["save_file"].replace(".onnx", ".engine")
    trt_path = osp.join(args.work_dir, trt_file)

    # Load model
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    device = torch.device(args.device)
    model = get_model(model_cfg, args.checkpoint, device=device)

    # Load sample data
    logger.info(f"Loading sample data from info.pkl: {args.info_pkl}")
    input_tensor_calibrated = load_sample_data_from_info_pkl(
        args.info_pkl, model_cfg, args.sample_idx, force_generate_miscalibration=False, device=args.device
    )
    input_tensor_miscalibrated = load_sample_data_from_info_pkl(
        args.info_pkl, model_cfg, args.sample_idx, force_generate_miscalibration=True, device=args.device
    )

    # Export models
    export_to_onnx(model, input_tensor_calibrated, onnx_path, config, logger)

    logger.info("Converting ONNX to TensorRT...")

    # Ensure CUDA device for TensorRT
    if args.device == "cpu":
        logger.warning("TensorRT requires CUDA device, switching to cuda")
        device = torch.device("cuda")
        input_tensor_calibrated = input_tensor_calibrated.to(device)
        input_tensor_miscalibrated = input_tensor_miscalibrated.to(device)

    success = export_to_tensorrt(onnx_path, trt_path, input_tensor_calibrated, config, logger)
    if success:
        logger.info(f"TensorRT conversion successful: {trt_path}")
    else:
        logger.error("TensorRT conversion failed, keeping ONNX model")

    # Run verification if requested
    if args.verify:
        logger.info(
            "Running verification for miscalibrated and calibrated samples with an output array [SCORE_MISCALIBRATED, SCORE_CALIBRATED]..."
        )
        input_tensors = {"0": input_tensor_miscalibrated, "1": input_tensor_calibrated}
        run_verification(model, onnx_path, trt_path, input_tensors, logger)

    logger.info("Deployment completed successfully!")


# TODO: make deployment script inherit from awml base deploy script or use awml deploy script directly
if __name__ == "__main__":
    main()
