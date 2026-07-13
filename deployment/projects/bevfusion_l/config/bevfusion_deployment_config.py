"""BEVFusion-specific deployment config.

Extends :class:`~deployment.config.base.BaseDeploymentConfig` to model the BEVFusion-only
deploy-config keys as typed attributes, so the entrypoint and export pipeline never reach
back into the raw MMEngine ``Config``. This is the typed home for the keys the generic
sections intentionally do not model.
"""

from __future__ import annotations

from mmengine.config import Config

from deployment.config.base import BaseDeploymentConfig
from deployment.projects.bevfusion_l.config.component_layout import (
    add_merged_component,
    is_split_components,
    merge_requested,
)


class BEVFusionDeploymentConfig(BaseDeploymentConfig):
    """Deployment config for BEVFusion.

    Adds typed attributes for the BEVFusion-only deploy-config keys:

    - ``fuse_spconv_bn``: fold spconv BatchNorm into conv weights before export (default ``False``).
    - ``spconv_do_sort``: bake the pair-mask argsort into ``GetIndicePairsImplicitGemm`` at ONNX
      export (default ``True``).
    - ``spconv_fuse_implicit_gemm_relu``: fuse a trailing ReLU into ImplicitGemm nodes in the sparse
      ONNX postprocess (default ``False``).
    - ``merge_bevfusion``: keep the split (sparse+dense) export and also emit the merged
      full-graph artifacts (derived from the deploy config's ``bevfusion_merge`` key).
    """

    def __init__(self, deploy_cfg: Config) -> None:
        super().__init__(deploy_cfg)
        self.fuse_spconv_bn: bool = bool(deploy_cfg.get("fuse_spconv_bn", False))
        self.spconv_do_sort: bool = bool(deploy_cfg.get("spconv_do_sort", True))
        self.spconv_fuse_implicit_gemm_relu: bool = bool(deploy_cfg.get("spconv_fuse_implicit_gemm_relu", False))
        self.merge_bevfusion: bool = merge_requested(deploy_cfg)

        # The merged graph is *derived* from the split sparse+dense pair (sparse inputs +
        # dense outputs), so it is resolved here as part of the config rather than mutated onto
        # the config later. After construction ``components_cfg`` is the final layout: the split
        # (sparse+dense) export plus, when ``merge_bevfusion`` is set, the merged graph.
        if self.merge_bevfusion:
            self.components_cfg = add_merged_component(
                deploy_cfg=deploy_cfg,
                components_cfg=self.components_cfg,
            )

        self._validate_components()

    def _validate_components(self) -> None:
        """Fail early if the resolved component layout is incomplete.

        BEVFusion's required components vary by layout (split sparse+dense vs merged graph),
        so this layout-aware check is authoritative rather than the registry's static tuple.
        """
        if is_split_components(self.components_cfg):
            self.components_cfg.get_component("bevfusion_sparse")
            self.components_cfg.get_component("bevfusion_dense")
        else:
            self.components_cfg.get_component("bevfusion_merged")
