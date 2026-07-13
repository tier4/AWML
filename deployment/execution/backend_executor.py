"""
Backend execution primitives.

`BackendExecutor` is the task-specific collaborator that knows how to run a single
backend on a device for one sample: create the inference pipeline, prepare the
model input, and manage the reference PyTorch model's device placement.

It is shared by both `~deployment.evaluation.base_evaluator.BaseEvaluator`
(the evaluation loop) and
`~deployment.verification.backend_verifier.BackendVerifier` (the
reference/test verification loop), so neither has to depend on the other.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Mapping, Optional

from deployment.config.enums import Backend
from deployment.inference.base_inference_pipeline import BaseInferencePipeline
from deployment.io.base_data_loader import BaseDataLoader
from deployment.primitives.device import DeviceSpec
from deployment.primitives.evaluator_types import InferenceInput, ModelSpec

logger = logging.getLogger(__name__)


class BackendExecutor(ABC):
    """Run a backend on a device for one sample (pipeline / input / device handling).

    Holds the loaded reference PyTorch model (set via `set_pytorch_model`), which is
    needed both to move the model onto a device and to build PyTorch/ONNX/TensorRT
    pipelines. Subclasses implement the two task-specific hooks: `create_pipeline`
    and `prepare_input`.
    """

    def __init__(self) -> None:
        self.pytorch_model: Any = None

    def set_pytorch_model(self, pytorch_model: Any) -> None:
        """Attach the loaded PyTorch module used for reference runs and pipeline creation."""
        self.pytorch_model = pytorch_model

    def ensure_model_on_device(self, device: DeviceSpec) -> Any:
        """Ensure ``pytorch_model`` lives on ``device`` (used before infer / pipeline creation)."""
        if self.pytorch_model is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.pytorch_model is None. "
                "DeploymentRunner must call set_pytorch_model() before verify/evaluate."
            )

        current_device = next(self.pytorch_model.parameters()).device
        target_device = device.to_torch_device()

        if current_device != target_device:
            logger.info("Moving PyTorch model from %s to %s", current_device, target_device)
            self.pytorch_model = self.pytorch_model.to(target_device)

        return self.pytorch_model

    def validate_device(self, backend: Backend, device: DeviceSpec) -> DeviceSpec:
        """Validate backend runtime constraints on a concrete DeviceSpec and return it."""
        if backend.requires_cuda and not device.is_cuda:
            raise ValueError(f"{backend.value} verification requires CUDA, got '{device}'.")
        return device

    def get_output_names(self) -> Optional[List[str]]:
        """Optional names for list/tuple raw outputs during verification comparison.

        Override when the backend's pipeline returns a sequence of tensors with known
        semantic names (e.g. detection heads). The names are forwarded to the
        `~deployment.verification.output_comparator.OutputComparator` to label
        positions in diagnostic paths.

        Returns:
            Names aligned with output index order, or `None` to fall back to
            `output_0`, `output_1`, ...
        """
        return None

    def get_supported_backends(self) -> List[Backend]:
        """Return the backends this executor can instantiate (override to restrict)."""
        return [Backend.PYTORCH, Backend.ONNX, Backend.TENSORRT]

    def _validate_backend(self, backend: Backend) -> None:
        """Raise a ValueError if ``backend`` is not supported by this executor."""
        supported = self.get_supported_backends()
        if backend not in supported:
            supported_names = [b.value for b in supported]
            raise ValueError(
                f"Unsupported backend '{backend.value}' for {self.__class__.__name__}. "
                f"Supported backends: {supported_names}"
            )

    @abstractmethod
    def create_pipeline(self, model_spec: ModelSpec, device: DeviceSpec) -> BaseInferencePipeline:
        """Create an inference pipeline for ``model_spec.backend`` on ``device``.

        Args:
            model_spec: Backend, device, and artifact path for the deployment model.
            device: Concrete device for this run.

        Returns:
            A ``BaseInferencePipeline`` subclass exposing ``infer()`` and ``cleanup()``.
        """
        raise NotImplementedError

    @abstractmethod
    def prepare_input(
        self,
        sample: Mapping[str, Any],
        data_loader: BaseDataLoader,
        device: DeviceSpec,
    ) -> InferenceInput:
        """Build an `InferenceInput` for ``sample`` on ``device``.

        Verification calls this once per side (reference and test) with each backend's
        own device, so implementations should create tensors directly on ``device``
        rather than relying on downstream moves.

        Args:
            sample: Sample data from the data loader.
            data_loader: Data loader to load the sample from.
            device: Device to prepare the input on.

        Returns:
            InferenceInput containing the actual input data and metadata.
        """
        raise NotImplementedError
