"""
BEVFusion backend executor.

Implements the task-specific backend execution primitives (pipeline creation and
input preparation) for BEVFusion, shared by the evaluator and the verification
runner via `~deployment.execution.backend_executor.BackendExecutor`.

This replaces the OLD ``BEVFusionPipelineFactory`` (the global pipeline registry was removed
in the refactor): pipeline construction uses the reference model on ``self.pytorch_model``
(set by the runner after export).
"""

import logging
from typing import Iterable, List, Optional

from typing_extensions import override

from deployment.config.enums import Backend
from deployment.config.schema import ComponentsConfig
from deployment.execution.point_cloud_backend_executor import PointCloudBackendExecutor
from deployment.inference.base_inference_pipeline import BaseInferencePipeline
from deployment.primitives.device import DeviceSpec
from deployment.primitives.evaluator_types import ModelSpec
from deployment.projects.bevfusion_l.config.component_layout import has_component, is_split_components
from deployment.projects.bevfusion_l.inference.pytorch_inference_pipeline import BEVFusionPyTorchInferencePipeline
from deployment.projects.bevfusion_l.inference.tensorrt_inference_pipeline import BEVFusionTensorRTInferencePipeline

logger = logging.getLogger(__name__)


class BEVFusionExecutor(PointCloudBackendExecutor):
    """Backend execution primitives for BEVFusion (pipeline creation, input prep).

    Args:
        components_cfg: Unified components configuration, forwarded to the ONNX/TensorRT
            pipelines so they can resolve split (sparse+dense) vs merged full-graph artifacts.
        plugin_libraries: Custom TensorRT plugin ``.so`` paths forwarded to the
            TensorRT pipeline (e.g. the spconv ImplicitGemm plugin); empty when none is needed.
    """

    def __init__(self, components_cfg: ComponentsConfig, plugin_libraries: Iterable[str] = ()) -> None:
        super().__init__()
        self._components_cfg = components_cfg
        self._plugin_libraries = tuple(plugin_libraries)

    @override
    def get_supported_backends(self) -> List[Backend]:
        """BEVFusion supports PyTorch and TensorRT only.

        ONNXRuntime cannot run the sparse (spconv) graph: it relies on ``autoware``-domain
        custom ops (``ImplicitGemm`` / ``GetIndicePairsImplicitGemm``) that are TensorRT plugins
        (``libautoware_tensorrt_plugins.so``), not ORT ops. ONNX *export* still happens (it is the
        PyTorch→ONNX→TensorRT bridge); only ONNX *inference* is unsupported here.
        """
        return [Backend.PYTORCH, Backend.TENSORRT]

    @override
    def get_output_names(self) -> Optional[List[str]]:
        """Return the model output names (split→dense outputs; otherwise merged-graph outputs)."""
        if is_split_components(self._components_cfg) and not has_component(self._components_cfg, "bevfusion_merged"):
            comp = self._components_cfg.get_component("bevfusion_dense")
        else:
            comp = self._components_cfg.get_component("bevfusion_merged")
        return [out.name for out in comp.io.outputs]

    @override
    def create_pipeline(self, model_spec: ModelSpec, device: DeviceSpec) -> BaseInferencePipeline:
        """Create a BEVFusion inference pipeline for the given backend and device.

        Args:
            model_spec: Model specification (backend, device, path).
            device: Target device for the pipeline.

        Returns:
            BEVFusion pipeline instance (PyTorch, ONNX, or TensorRT).

        Raises:
            ValueError: If ``model_spec.backend`` is not a supported backend.
        """
        backend = model_spec.backend
        self._validate_backend(backend)

        if backend is Backend.PYTORCH:
            logger.info("Creating BEVFusion PyTorch pipeline on %s", device)
            return BEVFusionPyTorchInferencePipeline(self.pytorch_model, device=device)

        if backend is Backend.TENSORRT:
            logger.info("Creating BEVFusion TensorRT pipeline from %s on %s", model_spec.artifact.path, device)
            return BEVFusionTensorRTInferencePipeline(
                self.pytorch_model,
                tensorrt_dir=model_spec.artifact.path,
                device=device,
                components_cfg=self._components_cfg,
                plugin_libraries=self._plugin_libraries,
            )

        raise ValueError(f"Unsupported backend: {backend.value}")
