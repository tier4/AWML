"""
Base inference pipeline for unified model deployment.

Flattened from `deployment/pipelines/common/base_pipeline.py`.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from deployment.core.backend import Backend
from deployment.core.device import DeviceSpec
from deployment.core.evaluation.evaluator_types import InferenceResult

logger = logging.getLogger(__name__)


class BaseInferencePipeline(ABC):
    """Base contract for a deployment-time inference pipeline.

    A pipeline is responsible for the classic 3-stage inference flow:
    `preprocess -> run_model -> postprocess`.

    The default `infer()` implementation measures per-stage latency and returns an
    `InferenceResult` with optional breakdown information.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        backend_type: Backend,
        device: DeviceSpec,
    ) -> None:
        """Create a pipeline bound to a model and a device.

        Args:
            model: Backend-specific callable/model wrapper used by `run_model`.
            device: Target runtime device (string/torch.device/DeviceSpec).
            backend_type: Deployment backend enum for logging/metrics. Required.
        """
        self.model = model
        self.device = device
        self.backend_type = backend_type

        logger.info(f"Initialized {self.__class__.__name__} on device: {self.device}")

    @property
    def torch_device(self) -> torch.device:
        """Return torch.device converted from canonical DeviceSpec."""
        return self.device.to_torch_device()

    @abstractmethod
    def preprocess(self, input_data: Any) -> Tuple[Any, Dict[str, Any]]:
        """Convert raw input into model-ready tensors/arrays.

        Returns:
            A 2-tuple ``(model_input, preprocess_metadata)``:
            - ``model_input``: Tensors or structure consumed by :meth:`run_model`.
            - ``preprocess_metadata``: Dict merged into the ``metadata`` argument of
              :meth:`infer` (together with any ``metadata`` passed by the caller) and
              then passed to :meth:`postprocess`. Use an empty dict when nothing extra
              is needed.
        """
        raise NotImplementedError

    @abstractmethod
    def run_model(self, preprocessed_input: Any) -> Tuple[Any, Dict[str, float]]:
        """Run the underlying model and return its raw outputs.

        Returns:
            A 2-tuple ``(model_output, stage_latencies)``:
            - ``model_output``: Raw tensors or structure for :meth:`postprocess` (or
              returned as-is when ``infer(..., return_raw_outputs=True)``).
            - ``stage_latencies``: Per-substage timings in milliseconds; merged into
              :class:`~deployment.core.evaluation.evaluator_types.InferenceResult`
              ``breakdown`` (e.g. ``voxel_encoder_ms``).
        """
        raise NotImplementedError

    @abstractmethod
    def postprocess(
        self,
        model_output: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """Convert raw model outputs into final predictions/results.

        Args:
            model_output: Value returned by :meth:`run_model` (first element of its tuple).
            metadata: Merged dict from ``infer(..., metadata=...)`` plus
                ``preprocess_metadata`` from :meth:`preprocess`. May be empty.
        """
        raise NotImplementedError

    def infer(
        self, input_data: Any, metadata: Optional[Mapping[str, Any]] = None, return_raw_outputs: bool = False
    ) -> InferenceResult:
        """Run end-to-end inference with latency breakdown.

        Flow:
            1) preprocess(input_data)
            2) run_model(model_input)
            3) postprocess(model_output, merged_metadata) unless `return_raw_outputs=True`

        Args:
            input_data: Raw input sample(s) in a project-defined format.
            metadata: Optional auxiliary context merged with preprocess metadata.
            return_raw_outputs: If True, skip `postprocess` and return raw model output.

        Returns:
            InferenceResult with `output`, total latency, and per-stage breakdown.
        """
        if metadata is None:
            metadata = {}

        latency_breakdown: Dict[str, float] = {}

        try:
            # Preprocess
            start_time = time.perf_counter()
            model_input, preprocess_metadata = self.preprocess(input_data)
            preprocess_time = time.perf_counter()
            latency_breakdown["preprocessing_ms"] = (preprocess_time - start_time) * 1000
            metadata.update(preprocess_metadata)

            # Run model
            model_start = time.perf_counter()
            model_output, model_latency = self.run_model(model_input)
            model_time = time.perf_counter()
            latency_breakdown["model_ms"] = (model_time - model_start) * 1000

            latency_breakdown.update(model_latency)

            total_latency = (time.perf_counter() - start_time) * 1000

            if return_raw_outputs:
                return InferenceResult(output=model_output, latency_ms=total_latency, breakdown=latency_breakdown)

            # Postprocess
            postprocess_start = time.perf_counter()
            postprocess_output = self.postprocess(model_output, metadata)
            postprocess_time = time.perf_counter()
            latency_breakdown["postprocessing_ms"] = (postprocess_time - postprocess_start) * 1000

            total_latency = (time.perf_counter() - start_time) * 1000
            return InferenceResult(output=postprocess_output, latency_ms=total_latency, breakdown=latency_breakdown)

        except Exception:
            logger.exception("Inference failed.")
            raise

    def cleanup(self) -> None:
        """Release resources owned by the pipeline.

        Subclasses should override when they hold external resources (e.g., CUDA
        buffers, TensorRT engines/contexts, file handles). `infer()` does not call
        this automatically; use the context manager (`with pipeline:`) or call it
        explicitly.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(" f"device={self.device}, " f"backend={self.backend_type})"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
