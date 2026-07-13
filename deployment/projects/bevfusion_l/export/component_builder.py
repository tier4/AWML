"""BEVFusion-specific component builder.

Splits a BEVFusion model into ONNX-exportable components for the shared ``OnnxExportPipeline``:

- **split** (``bevfusion_sparse`` + ``bevfusion_dense``): the sparse encoder and the dense
  (SECOND+neck+head) graph as two components. The dense component's tracing input (``lidar_bev``)
  is produced by running the sparse encoder on the sample — the same "run the earlier stage to get
  the later stage's input" pattern CenterPoint uses for its backbone component.

The single full-graph ONNX (``bevfusion_merged``) is not exported directly; it is composed from
the split sparse+dense pair as a post-export finalize step (see ``transforms.py`` and
``bevfusion_merge``), so this builder only ever produces the split pair.

The builder is a pure ``model + export-ready sample -> components`` step. :class:`BEVFusionVoxelSample`
already carries tensors on the model device with ``coors`` in the int32 ``[z, y, x]`` graph-input
layout (see :class:`BEVFusionSampleExtractor`), so the builder never touches device or dtype. The
two export-time globals live elsewhere by design: the SparseConv+BN fold is applied at model load
(``build_bevfusion_model(fuse_spconv_bn=...)``) and ``spconv_do_sort`` is set by the runner before
export — neither is a per-component concern.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import List

import torch

from deployment.export.pipelines.component_builder import ExportableComponent, ModelComponentBuilder
from deployment.projects.bevfusion_l.config.bevfusion_deployment_config import BEVFusionDeploymentConfig
from deployment.projects.bevfusion_l.config.component_layout import is_split_components
from deployment.projects.bevfusion_l.export.onnx_fuse_implicit_gemm_activation import (
    fuse_autoware_implicit_gemm_trailing_relu,
)
from deployment.projects.bevfusion_l.export.onnx_models.bevfusion_onnx import (
    BEVFusionDenseWrapper,
    BEVFusionSparseWrapper,
)
from deployment.projects.bevfusion_l.export.transforms import fix_topk_constant_k
from deployment.projects.bevfusion_l.io.sample_types import BEVFusionVoxelSample

logger = logging.getLogger(__name__)


def _voxel_inputs(sample: BEVFusionVoxelSample) -> tuple:
    """The (voxels, coors, num_points) tracing tuple, taken from the export-ready sample as-is."""
    return (sample.voxels, sample.coors, sample.num_points_per_voxel)


def _num_proposals(model: torch.nn.Module) -> int:
    """Return the head's ``num_proposals`` — the constant K baked into the exported TopK node."""
    head = getattr(model, "bbox_head", None)
    if head is None or not hasattr(head, "num_proposals"):
        raise ValueError("BEVFusion bbox_head.num_proposals is required for the TopK-constant fix.")
    return int(head.num_proposals)


def _topk_fix(model: torch.nn.Module):
    """The TopK-constant post-export transform, bound to the model's ``num_proposals``."""
    return partial(fix_topk_constant_k, num_proposals=_num_proposals(model))


def _fuse_implicit_gemm_relu(model_proto):
    """Post-export transform: fold trailing ReLU into ImplicitGemm ``act_type``."""
    n_relu = fuse_autoware_implicit_gemm_trailing_relu(model_proto)
    logger.info("Sparse ONNX postprocess: ImplicitGemm fuse done (trailing Relu=%d).", n_relu)
    return model_proto


def _run_sparse_encoder(model: torch.nn.Module, sample: BEVFusionVoxelSample) -> torch.Tensor:
    """Run the sparse encoder on the sample to get a BEV feature map for tracing the dense graph."""
    with torch.no_grad():
        return BEVFusionSparseWrapper(model).eval()(*_voxel_inputs(sample))


class BEVFusionComponentBuilder(ModelComponentBuilder):
    """Build exportable BEVFusion components (the split sparse + dense pair)."""

    def __init__(self, config: BEVFusionDeploymentConfig) -> None:
        """Store the deploy config (component layout + ``spconv_fuse_implicit_gemm_relu`` flag)."""
        self._config = config

    def build_components(
        self,
        model: torch.nn.Module,
        sample: BEVFusionVoxelSample,
    ) -> List[ExportableComponent]:
        """Build the ``sparse`` + ``dense`` component pair (the only supported export layout).

        The single full-graph ONNX (``bevfusion_merged``) is composed from this pair afterwards by
        the merge finalize hook, so the builder never exports a full graph directly.
        """
        if not is_split_components(self._config.components_cfg):
            raise ValueError(
                "BEVFusion export requires the split sparse+dense layout; the merged full graph is "
                "derived from that pair post-export (see bevfusion_merge / transforms.py)."
            )
        logger.info("Building BEVFusion split components (sparse + dense)...")
        return [self._sparse_component(model, sample), self._dense_component(model, sample)]

    def _sparse_component(self, model: torch.nn.Module, sample: BEVFusionVoxelSample) -> ExportableComponent:
        """Sparse encoder: voxels/coors/num_points -> ``lidar_bev`` (optional ImplicitGemm+ReLU fuse)."""
        if self._config.spconv_fuse_implicit_gemm_relu:
            post_transforms: tuple = (_fuse_implicit_gemm_relu,)
        else:
            post_transforms = ()
            logger.info("Sparse ONNX postprocess: ImplicitGemm ReLU fuse disabled by deploy config.")
        return self._component(
            "bevfusion_sparse", BEVFusionSparseWrapper(model), _voxel_inputs(sample), post_transforms
        )

    def _dense_component(self, model: torch.nn.Module, sample: BEVFusionVoxelSample) -> ExportableComponent:
        """Dense graph: ``lidar_bev`` -> detection triple. Traced with a BEV map from the sparse encoder."""
        lidar_bev = _run_sparse_encoder(model, sample)
        logger.info("Dense trace input lidar_bev shape: %s", tuple(lidar_bev.shape))
        return self._component("bevfusion_dense", BEVFusionDenseWrapper(model), (lidar_bev,), (_topk_fix(model),))

    def _component(
        self,
        name: str,
        module: torch.nn.Module,
        sample_input: tuple,
        post_transforms: tuple,
    ) -> ExportableComponent:
        """Assemble one ``ExportableComponent``, taking its canonical name from the deploy config.

        Looking the name up here (rather than passing a literal) keeps the exported component name
        in lockstep with the deploy config and validates the component exists — the same pattern
        CenterPoint's builder uses.
        """
        component_cfg = self._config.components_cfg.get_component(name)
        return ExportableComponent(
            name=component_cfg.name,
            module=module,
            sample_input=sample_input,
            post_transforms=post_transforms,
        )
