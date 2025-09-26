#!/usr/bin/env python3
"""
Integrated model evaluation script for calibration classification.
Supports both ONNX and TensorRT inference modes.
Simplified version that directly uses info.pkl files instead of dataset.
"""

import argparse
import gc
import logging
import os
import pickle
import signal
import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config

# Now we can import the class directly
from autoware_ml.calibration_classification.datasets.transforms.calibration_classification_transform import (
    CalibrationClassificationTransform,
)

# TensorRT imports (only if needed) - match original test_tensorrt.py
try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    import tensorrt as trt

    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

# Constants
LABELS = {"0": "miscalibrated", "1": "calibrated"}


def signal_handler(signum, frame):
    """Handle segmentation faults and other signals gracefully."""
    print(f"\nReceived signal {signum}. Cleaning up...")
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("Cleanup completed.")
    except:
        pass
    sys.exit(1)


# Register signal handlers
signal.signal(signal.SIGSEGV, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(level=getattr(logging, level.upper()), format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


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
    device: str = "cpu",
    transform_test: Optional[CalibrationClassificationTransform] = None,
) -> torch.Tensor:
    """
    Load and preprocess sample data from info.pkl using CalibrationClassificationTransform.
    Args:
        info_pkl_path: Path to the info.pkl file
        model_cfg: Model configuration containing data_root setting
        sample_idx: Index of the sample to load (default: 0)
        device: Device to load tensor on
        transform_test: Pre-created test transform instance (optional)
    Returns:
        Preprocessed tensor ready for model inference
    """
    # Load sample data from info.pkl
    sample_data = load_info_pkl_data(info_pkl_path, sample_idx)

    # Get data_root from model config
    data_root = model_cfg.get("data_root", None)
    if data_root is None:
        raise ValueError("data_root not found in model configuration")

    # Use pre-created transform or create new one (always use test mode)
    if transform_test is None:
        transform = CalibrationClassificationTransform(
            transform_config=model_cfg.get("transform_config", None),
            mode="test",
            data_root=data_root,
            projection_vis_dir=None,
            results_vis_dir=None,
            enable_augmentation=False,
        )
    else:
        transform = transform_test

    # Apply transform with miscalibration control
    results = transform.transform(sample_data)
    input_data_processed = results["fused_img"]  # (H, W, 5)

    # Convert to tensor
    input_tensor = torch.from_numpy(input_data_processed).permute(2, 0, 1).float()  # (5, H, W)
    input_tensor = input_tensor.unsqueeze(0).to(device)  # (1, 5, H, W)

    return input_tensor


def create_test_transform(model_cfg: Config) -> CalibrationClassificationTransform:
    """Create test transform instance once for reuse."""
    data_root = model_cfg.get("data_root", None)
    if data_root is None:
        raise ValueError("data_root not found in model configuration")

    transform_test = CalibrationClassificationTransform(
        transform_config=model_cfg.get("transform_config", None),
        mode="test",
        data_root=data_root,
        projection_vis_dir=None,
        results_vis_dir=None,
        enable_augmentation=False,
    )

    return transform_test


def run_onnx_inference(
    onnx_path: str,
    input_tensor: torch.Tensor,
    logger: logging.Logger,
) -> Tuple[np.ndarray, float]:
    """Run ONNX inference directly and return output and latency."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "onnxruntime is required for ONNX inference. Please install it with: pip install onnxruntime"
        )

    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Convert input tensor to float32
    input_tensor = input_tensor.float()

    # Debug: Print input tensor info before preprocessing
    logger.debug(
        f"Input tensor before preprocessing - Shape: {input_tensor.shape}, Dtype: {input_tensor.dtype}, Min: {input_tensor.min():.4f}, Max: {input_tensor.max():.4f}"
    )

    # Add batch dimension if needed (ONNX expects 4D input: batch, channels, height, width)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        logger.debug(f"Added batch dimension - Shape: {input_tensor.shape}")

    # ONNX inference with timing
    providers = ["CPUExecutionProvider"]
    ort_session = ort.InferenceSession(onnx_path, providers=providers)

    # Debug: Print ONNX model input info
    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    input_type = ort_session.get_inputs()[0].type
    logger.debug(f"ONNX model expects - Input name: {input_name}, Shape: {input_shape}, Type: {input_type}")

    onnx_input = {input_name: input_tensor.cpu().numpy().astype(np.float32)}

    start_time = time.perf_counter()
    onnx_output = ort_session.run(None, onnx_input)[0]
    end_time = time.perf_counter()
    onnx_latency = (end_time - start_time) * 1000

    logger.info(f"ONNX inference latency: {onnx_latency:.2f} ms")

    # Ensure onnx_output is numpy array
    if not isinstance(onnx_output, np.ndarray):
        logger.error(f"Unexpected ONNX output type: {type(onnx_output)}")
        return None, 0.0

    return onnx_output, onnx_latency


def run_tensorrt_inference(engine, input_tensor: torch.Tensor, logger: logging.Logger) -> Tuple[np.ndarray, float]:
    """Run TensorRT inference and return output with timing."""
    if not TENSORRT_AVAILABLE:
        raise ImportError("TensorRT and PyCUDA are required for TensorRT inference. Please install them.")

    context = None
    stream = None
    start = None
    end = None
    d_input = None
    d_output = None

    try:
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

        # Validate input shape
        expected_shape = engine.get_tensor_shape(input_name)
        if input_np.shape != tuple(expected_shape):
            logger.warning(f"Input shape mismatch: expected {expected_shape}, got {input_np.shape}")
            # Try to reshape if possible
            if len(input_np.shape) == len(expected_shape):
                try:
                    input_np = input_np.reshape(expected_shape)
                    logger.info(f"Reshaped input to {input_np.shape}")
                except Exception as e:
                    raise RuntimeError(f"Cannot reshape input from {input_np.shape} to {expected_shape}: {e}")
            elif len(input_np.shape) == len(expected_shape) - 1 and expected_shape[0] == 1:
                # Add batch dimension if missing
                try:
                    input_np = input_np.reshape(expected_shape)
                    logger.info(f"Added batch dimension to input: {input_np.shape}")
                except Exception as e:
                    raise RuntimeError(
                        f"Cannot add batch dimension to input from {input_np.shape} to {expected_shape}: {e}"
                    )
            else:
                raise RuntimeError(
                    f"Input shape mismatch: expected {expected_shape}, got {input_np.shape}. Please ensure input has correct batch dimension."
                )

        context.set_input_shape(input_name, input_np.shape)
        output_shape = context.get_tensor_shape(output_name)
        output_np = np.empty(output_shape, dtype=np.float32)
        if not output_np.flags["C_CONTIGUOUS"]:
            output_np = np.ascontiguousarray(output_np)

        # Allocate GPU memory
        d_input = cuda.mem_alloc(input_np.nbytes)
        d_output = cuda.mem_alloc(output_np.nbytes)

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

    except Exception as e:
        logger.error(f"TensorRT inference failed: {e}")
        raise
    finally:
        # Cleanup with better error handling
        try:
            if d_input is not None:
                d_input.free()
        except Exception as e:
            logger.warning(f"Failed to free input memory: {e}")

        try:
            if d_output is not None:
                d_output.free()
        except Exception as e:
            logger.warning(f"Failed to free output memory: {e}")

        # Note: Don't try to free stream, start, end, or context as they are managed by TensorRT
        # and may cause issues if freed manually


def load_tensorrt_engine(engine_path: str, logger: logging.Logger):
    """Load TensorRT engine from file."""
    if not TENSORRT_AVAILABLE:
        raise ImportError("TensorRT is required for TensorRT inference. Please install it.")

    trt_logger = trt.Logger(trt.Logger.WARNING)

    try:
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        engine = trt.Runtime(trt_logger).deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")
        return engine
    except Exception as e:
        logger.error(f"Error loading TensorRT engine: {e}")
        logger.error("This might be due to TensorRT version incompatibility.")
        logger.error("Please rebuild the TensorRT engine with the current TensorRT version.")
        raise


def evaluate_model(
    model_path: str,
    model_type: str,
    model_cfg_path: str,
    info_pkl_path: str,
    logger: logging.Logger,
    device: str = "cpu",
    num_samples: int = 10,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model using info.pkl data and return predictions, ground truth, probabilities, and latencies."""

    # Load model config
    model_cfg = Config.fromfile(model_cfg_path)

    # Load info.pkl data
    try:
        with open(info_pkl_path, "rb") as f:
            info_data = pickle.load(f)

        if "data_list" not in info_data:
            raise ValueError("Expected 'data_list' key in info.pkl")

        samples_list = info_data["data_list"]
        logger.info(f"Loaded {len(samples_list)} samples from info.pkl")

        # Limit number of samples for evaluation
        num_samples = min(num_samples, len(samples_list))
        logger.info(f"Evaluating {num_samples} samples")

    except Exception as e:
        logger.error(f"Failed to load info.pkl: {e}")
        raise

    # Create transform instances once
    transform_test = create_test_transform(model_cfg)

    # Initialize inference engine
    if model_type == "onnx":
        logger.info(f"Using ONNX model: {model_path}")
        inference_func = lambda input_tensor: run_onnx_inference(model_path, input_tensor, logger)
    elif model_type == "tensorrt":
        logger.info(f"Using TensorRT model: {model_path}")
        engine = load_tensorrt_engine(model_path, logger)
        inference_func = lambda input_tensor: run_tensorrt_inference(engine, input_tensor, logger)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Lists to store results
    all_predictions = []
    all_ground_truth = []
    all_probabilities = []
    all_latencies = []

    # Evaluate samples
    for sample_idx in range(num_samples):
        if sample_idx % 5 == 0:
            logger.info(f"Processing sample {sample_idx + 1}/{num_samples}")

        try:
            # Load sample data directly from info.pkl
            input_tensor_calibrated = load_sample_data_from_info_pkl(
                info_pkl_path,
                model_cfg,
                sample_idx,
                device=device,
                transform_test=transform_test,
            )
            input_tensor_miscalibrated = load_sample_data_from_info_pkl(
                info_pkl_path,
                model_cfg,
                sample_idx,
                device=device,
                transform_test=transform_test,
            )

            # Test both calibrated and miscalibrated samples
            test_samples = [
                (input_tensor_calibrated, 1),  # calibrated sample
                (input_tensor_miscalibrated, 0),  # miscalibrated sample
            ]

            for input_tensor, gt_label in test_samples:
                # Debug: Print input tensor info only in verbose mode
                if verbose:
                    logger.info(f"Sample {sample_idx + 1} (GT={gt_label}) input tensor:")
                    logger.info(f"  Dtype: {input_tensor.dtype}")

                # Run inference
                output_np, latency = inference_func(input_tensor)

                if output_np is None:
                    logger.error(f"Failed to get output for sample {sample_idx}")
                    continue

                # Convert logits to probabilities
                logits = torch.from_numpy(output_np)
                probabilities = F.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities.max().item()

                # Store results
                all_predictions.append(predicted_class)
                all_ground_truth.append(gt_label)
                all_probabilities.append(probabilities.cpu().numpy())
                all_latencies.append(latency)

                # Print sample results only in verbose mode
                if verbose:
                    logger.info(
                        f"Sample {sample_idx + 1} (GT={gt_label}): Pred={predicted_class}, Confidence={confidence:.4f}, Latency={latency:.2f}ms"
                    )

            # Clear GPU memory periodically for TensorRT
            if model_type == "tensorrt" and sample_idx % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            logger.error(f"Error processing sample {sample_idx}: {e}")
            continue

    return (
        np.array(all_predictions),
        np.array(all_ground_truth),
        np.array(all_probabilities),
        np.array(all_latencies),
    )


def print_results(
    all_predictions: np.ndarray,
    all_ground_truth: np.ndarray,
    all_probabilities: np.ndarray,
    all_latencies: np.ndarray,
    model_type: str,
    logger: logging.Logger,
):
    """Print evaluation results."""
    if len(all_predictions) == 0:
        logger.error("No samples were processed successfully.")
        return

    correct_predictions = (all_predictions == all_ground_truth).sum()
    total_samples = len(all_predictions)
    accuracy = correct_predictions / total_samples
    avg_latency = np.mean(all_latencies)

    # Print results
    logger.info(f"\n{'='*50}")
    logger.info(f"{model_type.upper()} Model Evaluation Results")
    logger.info(f"{'='*50}")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Correct predictions: {correct_predictions}")
    logger.info(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"Average latency: {avg_latency:.2f} ms")

    # Calculate per-class accuracy
    unique_classes = np.unique(all_ground_truth)
    logger.info(f"\nPer-class accuracy:")
    for cls in unique_classes:
        cls_mask = all_ground_truth == cls
        cls_correct = (all_predictions[cls_mask] == all_ground_truth[cls_mask]).sum()
        cls_total = cls_mask.sum()
        cls_accuracy = cls_correct / cls_total if cls_total > 0 else 0
        logger.info(
            f"  Class {cls} ({LABELS[str(cls)]}): {cls_correct}/{cls_total} = {cls_accuracy:.4f} ({cls_accuracy*100:.2f}%)"
        )

    # Calculate average confidence
    avg_confidence = np.mean([prob.max() for prob in all_probabilities])
    logger.info(f"\nAverage confidence: {avg_confidence:.4f}")

    # Calculate latency statistics
    min_latency = np.min(all_latencies)
    max_latency = np.max(all_latencies)
    std_latency = np.std(all_latencies)

    logger.info(f"\nLatency Statistics:")
    logger.info(f"  Average latency: {avg_latency:.2f} ms")
    logger.info(f"  Min latency: {min_latency:.2f} ms")
    logger.info(f"  Max latency: {max_latency:.2f} ms")
    logger.info(f"  Std latency: {std_latency:.2f} ms")

    # Show confusion matrix
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"Predicted ->")
    logger.info(f"Actual    0    1")
    for true_cls in unique_classes:
        row = []
        for pred_cls in unique_classes:
            count = ((all_ground_truth == true_cls) & (all_predictions == pred_cls)).sum()
            row.append(f"{count:4d}")
        logger.info(f"  {true_cls}    {' '.join(row)}")

    logger.info(f"\n{model_type.upper()} model evaluation completed successfully!")
    logger.info(f"Model accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"Average latency: {avg_latency:.2f} ms")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evaluate calibration classification model using info.pkl")
    parser.add_argument("--onnx", type=str, help="Path to ONNX model file")
    parser.add_argument("--tensorrt", type=str, help="Path to TensorRT engine file")
    parser.add_argument(
        "--model-cfg",
        type=str,
        default="projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb8-25e_j6gen2.py",
        help="Path to model config file",
    )
    parser.add_argument(
        "--info-pkl",
        type=str,
        required=True,
        help="Path to info.pkl file containing calibration data",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to evaluate (default: 10)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use for inference"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging for detailed sample information"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.onnx and not args.tensorrt:
        parser.error("Either --onnx or --tensorrt must be specified")

    if args.onnx and args.tensorrt:
        parser.error("Only one of --onnx or --tensorrt can be specified")

    # Setup logging
    logger = setup_logging(args.log_level)

    try:
        # Determine model type and path
        if args.onnx:
            model_type = "onnx"
            model_path = args.onnx
        else:
            model_type = "tensorrt"
            model_path = args.tensorrt

        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(args.model_cfg):
            raise FileNotFoundError(f"Model config file not found: {args.model_cfg}")
        if not os.path.exists(args.info_pkl):
            raise FileNotFoundError(f"Info.pkl file not found: {args.info_pkl}")

        logger.info(f"Starting {model_type.upper()} model evaluation...")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Model config: {args.model_cfg}")
        logger.info(f"Info.pkl: {args.info_pkl}")
        logger.info(f"Device: {args.device}")
        logger.info(f"Number of samples: {args.num_samples}")
        logger.info(f"Verbose logging: {args.verbose}")

        # Evaluate model
        all_predictions, all_ground_truth, all_probabilities, all_latencies = evaluate_model(
            model_path, model_type, args.model_cfg, args.info_pkl, logger, args.device, args.num_samples, args.verbose
        )

        # Print results
        print_results(all_predictions, all_ground_truth, all_probabilities, all_latencies, model_type, logger)

        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        logger.info("Script completed successfully!")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        # Cleanup on error
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
