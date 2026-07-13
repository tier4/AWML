"""Base inference pipeline for unified model deployment."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from deployment.config.enums import Backend
from deployment.primitives.device import DeviceSpec
from deployment.primitives.evaluator_types import InferenceResult

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

        logger.info("Initialized %s on device: %s", self.__class__.__name__, self.device)

    @property
    def torch_device(self) -> torch.device:
        """Return torch.device converted from canonical DeviceSpec."""
        return self.device.to_torch_device()

    def to_device_tensor(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Convert an array/tensor to a tensor on the pipeline's device."""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        return data.to(self.torch_device)

    def to_numpy(self, data: torch.Tensor, dtype: np.dtype = np.float32) -> np.ndarray:
        """Convert a tensor to a contiguous numpy array of ``dtype``."""
        arr = data.cpu().numpy().astype(dtype)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return arr

    @staticmethod
    def order_outputs_by_config(
        actual_names: Sequence[str],
        expected_names: Sequence[str],
        *,
        strict: bool = True,
    ) -> List[str]:
        """Return output names in the config's declared order.

        ONNX/TensorRT may report outputs in arbitrary order, but postprocess depends on the
        exact order declared in the component config. This returns the config order.

        Args:
            actual_names: Output names reported by the runtime session/engine.
            expected_names: Output names in the config's declared order.
            strict: If True, raise when the two name sets differ (any missing/extra). If False,
                return the expected names that are present (in order), then append any extras.

        Raises:
            ValueError: If ``strict`` and the name sets do not match exactly.
        """
        if strict:
            expected_set, actual_set = set(expected_names), set(actual_names)
            missing = expected_set - actual_set
            extra = actual_set - expected_set
            if missing or extra:
                raise ValueError(
                    f"Output name mismatch: missing={sorted(missing)}, extra={sorted(extra)}; "
                    f"expected={sorted(expected_set)}, got={sorted(actual_set)}."
                )
            return list(expected_names)
        ordered = [n for n in expected_names if n in actual_names]
        ordered += [n for n in actual_names if n not in ordered]
        return ordered

    @abstractmethod
    def preprocess(self, input_data: Any) -> Any:
        """Convert raw input into model-ready tensors/arrays.

        Returns:
            ``model_input``: Tensors or structure consumed by :meth:`run_model`.
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
              `~deployment.primitives.evaluator_types.InferenceResult`
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
            metadata: Dict passed by the caller via ``infer(..., metadata=...)``.
                May be empty.
        """
        raise NotImplementedError

    def infer(
        self, input_data: Any, metadata: Optional[Mapping[str, Any]] = None, return_raw_outputs: bool = False
    ) -> InferenceResult:
        """Run end-to-end inference with latency breakdown.

        Flow:
            1) preprocess(input_data)
            2) run_model(model_input)
            3) postprocess(model_output, metadata) unless `return_raw_outputs=True`

        Args:
            input_data: Raw input sample(s) in a project-defined format.
            metadata: Optional auxiliary context passed through to :meth:`postprocess`.
            return_raw_outputs: If True, skip `postprocess` and return raw model output.

        Returns:
            InferenceResult with `output`, total latency, and per-stage breakdown.
        """
        latency_breakdown: Dict[str, float] = {}

        try:
            # Preprocess
            start_time = time.perf_counter()
            model_input = self.preprocess(input_data)
            preprocess_time = time.perf_counter()
            latency_breakdown["preprocessing_ms"] = (preprocess_time - start_time) * 1000
            # Build a new dict from caller-provided metadata (passed to postprocess).
            metadata = dict(metadata or {})

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

    def periodic_cleanup(self, sample_idx: int) -> None:
        """Per-sample cleanup hook, always called once per sample by the evaluation loop.

        The default does nothing; overriding is optional. Backends with their own
        caching concerns (e.g. TensorRT freeing the CUDA cache every N samples)
        override this so the loop never has to special-case a backend.
        """

    def cleanup(self) -> None:
        """Release resources owned by the pipeline.

        Subclasses should override when they hold external resources (e.g., CUDA
        buffers, TensorRT engines/contexts, file handles). `infer()` does not call
        this automatically; use the context manager (`with pipeline:`) or call it
        explicitly.
        """

    def __repr__(self):
        return f"{self.__class__.__name__}(" f"device={self.device}, " f"backend={self.backend_type})"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logger.error("Pipeline failed with: %s", exc_val)

        self.cleanup()
        return False
