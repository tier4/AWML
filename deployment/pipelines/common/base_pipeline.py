"""
Base Deployment Pipeline for Unified Model Deployment.

This module provides the abstract base class for all deployment pipelines,
defining a unified interface across different backends (PyTorch, ONNX, TensorRT)
and task types (detection, classification, segmentation).

Architecture:
    Input → preprocess() → run_model() → postprocess() → Output

Key Design Principles:
    1. Shared Logic: preprocess/postprocess are shared across backends
    2. Backend-Specific: run_model() is implemented per backend
    3. Unified Interface: infer() provides consistent API
    4. Flexible Output: Can return raw or processed outputs
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from deployment.core.evaluation.evaluator_types import InferenceResult

logger = logging.getLogger(__name__)


class BaseDeploymentPipeline(ABC):
    """
    Abstract base class for all deployment pipelines.

    This class defines the unified interface for model deployment across
    different backends and task types.

    Attributes:
        model: Model object (PyTorch model, ONNX session, TensorRT engine, etc.)
        device: Device for inference
        task_type: Type of task ("detection2d", "detection3d", "classification", etc.)
        backend_type: Type of backend ("pytorch", "onnx", "tensorrt", etc.)
    """

    def __init__(self, model: Any, device: str = "cpu", task_type: str = "unknown", backend_type: str = "unknown"):
        """
        Initialize deployment pipeline.

        Args:
            model: Model object (backend-specific)
            device: Device for inference ('cpu', 'cuda', 'cuda:0', etc.)
            task_type: Type of task
            backend_type: Type of backend
        """
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.task_type = task_type
        self.backend_type = backend_type
        self._stage_latencies: Dict[str, float] = {}

        logger.info(f"Initialized {self.__class__.__name__} on device: {self.device}")

    @abstractmethod
    def preprocess(self, input_data: Any, **kwargs) -> Any:
        """
        Preprocess input data.

        This method should handle all preprocessing steps required before
        feeding data to the model (normalization, resizing, etc.).

        Args:
            input_data: Raw input (image, point cloud, etc.)
            **kwargs: Additional preprocessing parameters

        Returns:
            Preprocessed data ready for model
        """
        raise NotImplementedError

    @abstractmethod
    def run_model(self, preprocessed_input: Any) -> Union[Any, Tuple[Any, Dict[str, float]]]:
        """
        Run model inference (backend-specific).

        This is the only method that differs across backends.
        Each backend (PyTorch, ONNX, TensorRT) implements its own version.

        Args:
            preprocessed_input: Preprocessed input data

        Returns:
            Model output, or Tuple of (model_output, stage_latencies)

            If a tuple is returned:
            - model_output: Raw tensors or backend-specific format
            - stage_latencies: Dict mapping stage names to latency in ms

            If single value is returned, it's treated as model_output with no stage latencies.

        Note:
            Returning stage latencies as a tuple is the recommended pattern to avoid
            race conditions when pipelines are reused across multiple threads.
            Use local variables instead of instance variables for per-request data.
        """
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, model_output: Any, metadata: Dict = None) -> Any:
        """
        Postprocess model output to final predictions.

        This method should handle all postprocessing steps like NMS,
        coordinate transformation, score filtering, etc.

        Args:
            model_output: Raw model output from run_model()
            metadata: Additional metadata (image size, point cloud range, etc.)

        Returns:
            Final predictions in standard format
        """
        raise NotImplementedError

    def infer(
        self, input_data: Any, metadata: Optional[Dict] = None, return_raw_outputs: bool = False, **kwargs
    ) -> InferenceResult:
        """
        Complete inference pipeline.

        This method orchestrates the entire inference flow:
        1. Preprocessing
        2. Model inference
        3. Postprocessing (optional)

        This unified interface allows:
        - Evaluation: infer(..., return_raw_outputs=False) → get final predictions
        - Verification: infer(..., return_raw_outputs=True) → get raw outputs for comparison

        Args:
            input_data: Raw input data
            metadata: Additional metadata for preprocessing/postprocessing
            return_raw_outputs: If True, skip postprocessing (for verification)
            **kwargs: Additional arguments passed to preprocess()

        Returns:
            InferenceResult containing:
            - output: raw model output (return_raw_outputs=True) or final predictions
            - latency_ms: total inference latency in milliseconds
            - breakdown: stage-wise latencies (may be empty) with keys such as
              preprocessing_ms, model_ms, postprocessing_ms
        """
        if metadata is None:
            metadata = {}

        latency_breakdown: Dict[str, float] = {}

        try:
            start_time = time.perf_counter()

            # Preprocess
            preprocessed = self.preprocess(input_data, **kwargs)

            # Unpack preprocess outputs
            preprocess_metadata = {}
            model_input = preprocessed
            if isinstance(preprocessed, tuple) and len(preprocessed) == 2 and isinstance(preprocessed[1], dict):
                model_input, preprocess_metadata = preprocessed

            preprocess_time = time.perf_counter()
            latency_breakdown["preprocessing_ms"] = (preprocess_time - start_time) * 1000

            # Merge caller metadata with preprocess metadata
            merged_metadata = {}
            merged_metadata.update(metadata or {})
            merged_metadata.update(preprocess_metadata)

            # Run model (backend-specific)
            model_start = time.perf_counter()
            model_result = self.run_model(model_input)
            model_time = time.perf_counter()
            latency_breakdown["model_ms"] = (model_time - model_start) * 1000

            # Handle returned stage latencies (new pattern - thread-safe)
            stage_latencies = {}
            if isinstance(model_result, tuple) and len(model_result) == 2:
                model_output, stage_latencies = model_result
                if isinstance(stage_latencies, dict):
                    latency_breakdown.update(stage_latencies)
            else:
                model_output = model_result

            # Legacy: Merge stage-wise latencies from instance variable (deprecated)
            # This is kept for backward compatibility but should be removed eventually
            if hasattr(self, "_stage_latencies") and isinstance(self._stage_latencies, dict):
                latency_breakdown.update(self._stage_latencies)
                # Clear for next inference
                self._stage_latencies = {}

            total_latency = (time.perf_counter() - start_time) * 1000

            # Postprocess (optional)
            if return_raw_outputs:
                return InferenceResult(output=model_output, latency_ms=total_latency, breakdown=latency_breakdown)
            else:
                postprocess_start = time.perf_counter()
                predictions = self.postprocess(model_output, merged_metadata)
                postprocess_time = time.perf_counter()
                latency_breakdown["postprocessing_ms"] = (postprocess_time - postprocess_start) * 1000

                total_latency = (time.perf_counter() - start_time) * 1000
                return InferenceResult(output=predictions, latency_ms=total_latency, breakdown=latency_breakdown)

        except Exception:
            logger.exception("Inference failed.")
            raise

    def cleanup(self) -> None:
        """
        Cleanup pipeline resources.

        Subclasses should override this to release backend-specific resources
        (e.g., TensorRT contexts, ONNX sessions, CUDA streams).

        This method is called automatically when using the pipeline as a
        context manager, or can be called explicitly when done with the pipeline.
        """
        pass

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"device={self.device}, "
            f"task={self.task_type}, "
            f"backend={self.backend_type})"
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.cleanup()
        return False
