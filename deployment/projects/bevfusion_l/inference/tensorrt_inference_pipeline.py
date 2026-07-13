"""BEVFusion TensorRT Pipeline Implementation."""

from __future__ import annotations

import logging
import os.path as osp
from typing import Dict, List, Tuple

import numpy as np
import pycuda.autoinit  # noqa: F401
import tensorrt as trt
import torch
from typing_extensions import override

from deployment.config.enums import Backend
from deployment.config.schema import ComponentsConfig
from deployment.inference.gpu_resource_mixin import GPUResourceMixin, release_tensorrt_resources
from deployment.inference.tensorrt_runner import list_trt_io_names, load_trt_engine, run_trt_engine
from deployment.primitives.artifacts import resolve_artifact_path
from deployment.primitives.device import DeviceSpec
from deployment.primitives.tensorrt_plugins import load_tensorrt_plugin_libraries
from deployment.projects.bevfusion_l.config.component_layout import has_component, is_split_components
from deployment.projects.bevfusion_l.inference.bevfusion_inference_pipeline import BEVFusionInferencePipeline
from deployment.projects.bevfusion_l.io.voxel_inputs import map_voxel_inputs, voxel_indices_xyz_to_graph_input_zyx

logger = logging.getLogger(__name__)


class BEVFusionTensorRTInferencePipeline(GPUResourceMixin, BEVFusionInferencePipeline):
    """TensorRT-based BEVFusion pipeline (one loaded engine per deploy-config component).

    Engines and contexts are held in ``self._engines`` / ``self._contexts`` keyed by component
    name (the same pattern CenterPoint uses), so the layout only decides *which* components are
    loaded, not how they are stored:

    - Split layout: ``bevfusion_sparse`` + ``bevfusion_dense``, each CUDA-timed via the
      ``run_sparse_encoder`` / ``run_dense`` seams (``sparse_ms`` / ``dense_ms``).
    - Merged layout: a single ``bevfusion_merged`` full-graph engine that cannot be split, so
      it reports one ``model_ms`` GPU total.
    """

    def __init__(
        self,
        pytorch_model: torch.nn.Module,
        tensorrt_dir: str,
        device: DeviceSpec,
        components_cfg: ComponentsConfig,
        plugin_libraries: Tuple[str, ...] = (),
    ) -> None:
        super().__init__(pytorch_model=pytorch_model, backend_type=Backend.TENSORRT, device=device)

        self.tensorrt_dir = tensorrt_dir
        self._components_cfg = components_cfg
        self._plugin_libraries = plugin_libraries
        self._trt_logger = trt.Logger(trt.Logger.WARNING)

        # Prefer the merged full-graph engine when the split+merge export produced one on disk;
        # otherwise run the split sparse+dense pair.
        split_layout = is_split_components(components_cfg)
        merged_available = (
            split_layout
            and has_component(components_cfg, "bevfusion_merged")
            and osp.exists(
                resolve_artifact_path(
                    base_dir=tensorrt_dir,
                    components_cfg=components_cfg,
                    component_name="bevfusion_merged",
                    file_key="engine_file",
                )
            )
        )
        self._split = split_layout and not merged_available

        # Engine/context per component name (like CenterPoint); the loaded keys follow the layout.
        self._engines: Dict[str, trt.ICudaEngine] = {}
        self._contexts: Dict[str, trt.IExecutionContext] = {}

        # Per-stage pure-GPU times (ms), filled by each seam while its CUDA stream is still alive
        # and read back in run_model (mirrors the CenterPoint TensorRT pipeline).
        self._gpu_stage_ms: Dict[str, float] = {}

        self._load_tensorrt_engines()
        logger.info("BEVFusion TensorRT pipeline initialized from: %s (split=%s)", tensorrt_dir, self._split)

    def _load_tensorrt_engines(self) -> None:
        """Load one engine/context per component for the active layout into the name-keyed dicts."""
        load_tensorrt_plugin_libraries(self._plugin_libraries)
        trt.init_libnvinfer_plugins(self._trt_logger, "")
        runtime = trt.Runtime(self._trt_logger)

        component_names = ["bevfusion_sparse", "bevfusion_dense"] if self._split else ["bevfusion_merged"]
        for component_name in component_names:
            engine_path = resolve_artifact_path(
                base_dir=self.tensorrt_dir,
                components_cfg=self._components_cfg,
                component_name=component_name,
                file_key="engine_file",
            )
            if not osp.exists(engine_path):
                raise FileNotFoundError(f"TensorRT engine not found for {component_name}: {engine_path}")
            engine, context = load_trt_engine(runtime, engine_path, component_name=component_name)
            self._engines[component_name] = engine
            self._contexts[component_name] = context
            logger.info("Loaded TensorRT engine: %s (%s)", component_name, engine_path)

    def _engine_context(self, component_name: str) -> Tuple[trt.ICudaEngine, trt.IExecutionContext]:
        """Return the loaded (engine, context) for a component, or fail loud if it is absent."""
        engine = self._engines.get(component_name)
        context = self._contexts.get(component_name)
        if engine is None or context is None:
            raise RuntimeError(f"TensorRT engine/context for {component_name!r} is not loaded (layout mismatch).")
        return engine, context

    def _trt_infer_voxel_inputs(
        self,
        engine: trt.ICudaEngine,
        context: trt.IExecutionContext,
        voxels_np: np.ndarray,
        coors_np: np.ndarray,
        num_points_np: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """Assemble the multi-input voxel map (sparse/merged engines) and run the engine."""
        input_names, output_names = list_trt_io_names(engine)
        input_map = map_voxel_inputs(input_names, voxels=voxels_np, coors=coors_np, num_points=num_points_np)
        return run_trt_engine(engine, context, input_map, output_names)

    def _prepare_voxel_inputs(
        self,
        voxels: torch.Tensor,
        coors: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert the voxel tensors to the numpy dtypes/axis order the engine expects."""
        voxels_np = self.to_numpy(voxels, dtype=np.float32)
        coors_np = self.to_numpy(voxel_indices_xyz_to_graph_input_zyx(coors), dtype=np.int32)
        num_points_np = self.to_numpy(num_points_per_voxel, dtype=np.int32)
        # Match ``extract_pts_feat``: mean-pool must not divide by zero (NaN BEV → dense NaN).
        num_points_np = np.maximum(num_points_np, 1)
        return voxels_np, coors_np, num_points_np

    @override
    def run_model(
        self,
        preprocessed_input: Dict[str, torch.Tensor],
    ) -> Tuple[List[torch.Tensor], Dict[str, float]]:
        """Split: pure-GPU ``sparse_ms`` / ``dense_ms`` from the two seams. Merged: one ``model_ms``.

        Overrides the base wall-clock orchestration to report CUDA-event GPU times (matching the
        CenterPoint TensorRT pipeline). The merged full-graph engine is a single ``execute`` that
        cannot be split into sparse/dense, so it reports one GPU total under ``model_ms``.
        """
        if self._split:
            bev_features = self.run_sparse_encoder(
                preprocessed_input["voxels"],
                preprocessed_input["coors"],
                preprocessed_input["num_points_per_voxel"],
            )
            outputs = self.run_dense(bev_features)
            return outputs, {"sparse_ms": self._gpu_stage_ms["sparse_ms"], "dense_ms": self._gpu_stage_ms["dense_ms"]}

        outputs, gpu_ms = self._run_merged(
            preprocessed_input["voxels"],
            preprocessed_input["coors"],
            preprocessed_input["num_points_per_voxel"],
        )
        return outputs, {"model_ms": gpu_ms}

    @override
    def run_sparse_encoder(
        self,
        voxels: torch.Tensor,
        coors: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
    ) -> torch.Tensor:
        """Sparse (spconv) engine: voxels/coors/num_points -> ``lidar_bev`` (split layout only)."""
        engine, context = self._engine_context("bevfusion_sparse")
        voxels_np, coors_np, num_points_np = self._prepare_voxel_inputs(voxels, coors, num_points_per_voxel)

        sparse_out, gpu_ms = self._trt_infer_voxel_inputs(engine, context, voxels_np, coors_np, num_points_np)
        self._gpu_stage_ms["sparse_ms"] = gpu_ms

        if len(sparse_out) != 1:
            raise RuntimeError(f"Sparse engine: expected 1 output, got {list(sparse_out.keys())}")
        bev_name = next(iter(sparse_out))
        expected = [o.name for o in self._components_cfg.get_component("bevfusion_sparse").io.outputs]
        if expected and bev_name not in expected:
            logger.warning(
                "[trt-split] sparse engine output tensor is %r but deploy_cfg bevfusion_sparse.io.outputs "
                "names=%s — check ONNX export / TRT binding names.",
                bev_name,
                expected,
            )
        bev_arr = np.ascontiguousarray(sparse_out[bev_name].astype(np.float32))
        return torch.from_numpy(bev_arr).to(self.torch_device)

    @override
    def run_dense(self, bev_features: torch.Tensor) -> List[torch.Tensor]:
        """Dense engine: ``lidar_bev`` -> detection tensors (split layout only)."""
        engine, context = self._engine_context("bevfusion_dense")
        dense_cfg = self._components_cfg.get_component("bevfusion_dense")

        bev_arr = self.to_numpy(bev_features, dtype=np.float32)
        # The dense graph has a single input (``lidar_bev``), so bind by the engine's only input name.
        input_names, output_names = list_trt_io_names(engine)
        dense_out, gpu_ms = run_trt_engine(engine, context, {input_names[0]: bev_arr}, output_names)
        self._gpu_stage_ms["dense_ms"] = gpu_ms

        expected_output_names = [out.name for out in dense_cfg.io.outputs]
        ordered_names = self.order_outputs_by_config(list(dense_out.keys()), expected_output_names, strict=False)
        return [torch.from_numpy(dense_out[name]).to(self.torch_device) for name in ordered_names]

    def _run_merged(
        self,
        voxels: torch.Tensor,
        coors: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], float]:
        """Run the single full-graph engine and return (detection tensors, pure-GPU ms)."""
        engine, context = self._engine_context("bevfusion_merged")
        voxels_np, coors_np, num_points_np = self._prepare_voxel_inputs(voxels, coors, num_points_per_voxel)

        output_arrays, gpu_ms = self._trt_infer_voxel_inputs(engine, context, voxels_np, coors_np, num_points_np)
        expected_output_names = [out.name for out in self._components_cfg.get_component("bevfusion_merged").io.outputs]
        ordered_names = self.order_outputs_by_config(list(output_arrays.keys()), expected_output_names, strict=False)
        tensors = [torch.from_numpy(output_arrays[name]).to(self.torch_device) for name in ordered_names]
        return tensors, gpu_ms

    def _release_gpu_resources(self) -> None:
        """Release every loaded engine/context (uniform across split and merged layouts)."""
        release_tensorrt_resources(engines=self._engines, contexts=self._contexts)
