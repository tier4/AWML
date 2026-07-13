"""BEVFusion-specific deployment runner."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from mmengine.config import Config
from typing_extensions import override

from deployment.evaluation.detection_3d_evaluator import Detection3DEvaluator
from deployment.execution.backend_executor import BackendExecutor
from deployment.export.pipelines.onnx_export_pipeline import OnnxExportPipeline
from deployment.export.pipelines.tensorrt_export_pipeline import TensorRTExportPipeline
from deployment.io.base_data_loader import BaseDataLoader
from deployment.projects.bevfusion_l.config.bevfusion_deployment_config import BEVFusionDeploymentConfig
from deployment.projects.bevfusion_l.export.component_builder import BEVFusionComponentBuilder
from deployment.projects.bevfusion_l.export.sample_extractor import BEVFusionSampleExtractor
from deployment.projects.bevfusion_l.export.transforms import bevfusion_merge_finalize
from deployment.projects.bevfusion_l.io.model_loader import build_bevfusion_model
from deployment.runtime.runner import BaseDeploymentRunner
from projects.SparseConvolution.sparse_functional import set_do_sort

logger = logging.getLogger(__name__)


class BEVFusionDeploymentRunner(BaseDeploymentRunner):
    """BEVFusion deployment runner.

    Implements project-specific model loading (LiDAR-only, on CUDA, with the optional SparseConv+BN
    fold and the ``spconv_do_sort`` export global) and wires BEVFusion's ``BEVFusionSampleExtractor``
    + ``BEVFusionComponentBuilder`` (plus the split→merge ``finalize`` hook when
    ``config.merge_bevfusion`` is set) into the shared ``OnnxExportPipeline``, reusing the
    project-agnostic orchestration in ``BaseDeploymentRunner`` and the shared
    ``TensorRTExportPipeline``.

    BEVFusion-only deploy-config flags (``fuse_spconv_bn``, ``spconv_do_sort``,
    ``spconv_fuse_implicit_gemm_relu``, ``merge_bevfusion``) are typed attributes on
    ``BEVFusionDeploymentConfig``.
    """

    def __init__(
        self,
        data_loader: BaseDataLoader,
        evaluator: Detection3DEvaluator,
        executor: BackendExecutor,
        config: BEVFusionDeploymentConfig,
        model_cfg: Config,
        onnx_pipeline: Optional[OnnxExportPipeline] = None,
        tensorrt_pipeline: Optional[TensorRTExportPipeline] = None,
    ) -> None:
        # The exported ONNX layout (split sparse+dense, optionally merged into one full graph) is
        # driven entirely by the deploy config's ``components``; there is no per-run module selection.

        # Construct the pipelines BEFORE super().__init__, because the base runner forwards them
        # straight to the ExportOrchestrator (there is no post-init slot).
        if onnx_pipeline is None:
            onnx_pipeline = OnnxExportPipeline(
                sample_extractor=BEVFusionSampleExtractor(),
                component_builder=BEVFusionComponentBuilder(config=config),
                finalize=bevfusion_merge_finalize if config.merge_bevfusion else None,
            )

        super().__init__(
            data_loader=data_loader,
            evaluator=evaluator,
            executor=executor,
            config=config,
            model_cfg=model_cfg,
            onnx_pipeline=onnx_pipeline,
            tensorrt_pipeline=tensorrt_pipeline,
        )

    @override
    def load_pytorch_model(self, checkpoint_path: str) -> torch.nn.Module:
        """Load the BEVFusion model onto the CUDA device for export.

        The base runner forwards the returned model to ``executor.set_pytorch_model`` after
        export, so PyTorch/ONNX/TensorRT evaluation all reuse this reference.
        """
        cuda_device = self.config.device_config.cuda
        if cuda_device is None:
            raise RuntimeError(
                "BEVFusion requires a CUDA device for sparse convolution. Set devices.cuda in deploy config."
            )

        # ``spconv_do_sort`` is a process-global read by GetIndicePairsImplicitGemm at ONNX symbolic
        # export and in the spconv forward path. Set it here — once, before any export or inference —
        # so the exported sparse graph and PyTorch inference agree. It is deploy-time config, not a
        # per-component concern, so it lives on the runner rather than in the component builder.
        set_do_sort(self.config.spconv_do_sort)
        logger.info(
            "spconv_do_sort=%s (baked into GetIndicePairsImplicitGemm.do_sort_i at ONNX export)",
            self.config.spconv_do_sort,
        )

        return build_bevfusion_model(
            model_cfg=self.model_cfg,
            checkpoint_path=checkpoint_path,
            device=cuda_device,
            fuse_spconv_bn=self.config.fuse_spconv_bn,
        )
