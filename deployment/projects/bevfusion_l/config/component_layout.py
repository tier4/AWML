"""BEVFusion deploy-config component-layout helpers.

BEVFusion can be deployed in two component layouts:

- **split**: separate ``bevfusion_sparse`` (spconv) + ``bevfusion_dense`` ONNX/TensorRT graphs.
- **merged**: a single ``bevfusion_merged`` graph (sparse inputs → dense outputs).

These helpers query the layout (:func:`is_split_components`, :func:`has_component`) and, when a
deploy config opts into ``bevfusion_merge``, derive the merged ``bevfusion_merged`` component
from the split pair (:func:`merge_requested`, :func:`add_merged_component`). They operate
purely on deploy-config structures, so they live beside the deployment config rather than in ``io``.
"""

from __future__ import annotations

from typing import Any, Mapping

from deployment.config.schema import ComponentCfg, ComponentIO, ComponentsConfig


def is_split_components(components_cfg: ComponentsConfig) -> bool:
    """True when the deploy config uses split sparse + dense graphs (not a single merged graph)."""
    names = set(components_cfg.component_names())
    return "bevfusion_sparse" in names and "bevfusion_dense" in names


def has_component(components_cfg: ComponentsConfig, component_name: str) -> bool:
    """Return True if the component exists in the layout."""
    try:
        components_cfg.get_component(component_name)
        return True
    except KeyError:
        return False


def merge_requested(deploy_cfg: Mapping[str, Any]) -> bool:
    """Return True when the deploy config requests the split->merged graph merge.

    The single canonical key is ``bevfusion_merge`` (a dict with ``enabled`` / ``onnx_file`` /
    ``engine_file``, or a plain bool).
    """
    merge_raw = deploy_cfg.get("bevfusion_merge", False)
    if isinstance(merge_raw, Mapping):
        return bool(merge_raw.get("enabled", False))
    return bool(merge_raw)


def add_merged_component(
    *,
    deploy_cfg: Mapping[str, Any],
    components_cfg: ComponentsConfig,
) -> ComponentsConfig:
    """Add a merged ``bevfusion_merged`` component while keeping the split components.

    When ``bevfusion_merge`` is enabled and the layout is split, this derives
    ``bevfusion_merged`` by reusing:

    - the split sparse input schema / TensorRT profile, and
    - the split dense output schema.

    If the layout is not split, merge is not requested, or ``bevfusion_merged`` already exists,
    the config is returned unchanged.
    """
    if not is_split_components(components_cfg):
        return components_cfg
    if not merge_requested(deploy_cfg):
        return components_cfg
    if has_component(components_cfg, "bevfusion_merged"):
        return components_cfg

    sparse_cfg = components_cfg.get_component("bevfusion_sparse")
    dense_cfg = components_cfg.get_component("bevfusion_dense")

    merge_raw = deploy_cfg.get("bevfusion_merge", {})
    merge_cfg = merge_raw if isinstance(merge_raw, Mapping) else {}
    onnx_file = str(merge_cfg.get("onnx_file", "bevfusion_lidar.onnx"))
    engine_file = str(merge_cfg.get("engine_file", "bevfusion_lidar.engine"))

    # The merged graph reuses the split sparse inputs / TensorRT profile and the split dense
    # outputs, so build it directly from the already-typed components (no raw-dict round-trip).
    merged = ComponentCfg(
        name="bevfusion_merged",
        onnx_file=onnx_file,
        engine_file=engine_file,
        io=ComponentIO(
            inputs=list(sparse_cfg.io.inputs),
            outputs=list(dense_cfg.io.outputs),
            dynamic_axes=dict(sparse_cfg.io.dynamic_axes),
        ),
        tensorrt_profile=dict(sparse_cfg.tensorrt_profile),
    )
    return components_cfg.with_component(merged)
