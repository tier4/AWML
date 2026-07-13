"""BEVFusion post-export ONNX graph transforms.

Pure ``onnx.ModelProto`` graph rewrites applied after ``torch.onnx.export``:

- :func:`fix_topk_constant_k`: replace the head's dynamic TopK ``K`` with a constant
  (``num_proposals``) — TensorRT requires a constant K. Registered as a per-component
  ``post_transforms`` on the shared ``OnnxExportPipeline``.
- :func:`merge_split_sparse_dense_onnx`: compose the split ``sparse`` + ``dense`` ONNX into a
  single ``merged`` ONNX. Used as the pipeline-level ``finalize`` hook for split+merge exports.

The ImplicitGemm+ReLU fusion lives in :mod:`onnx_fuse_implicit_gemm_activation` and is registered
directly as a ``post_transforms`` entry by the component builder.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import onnx
import onnx_graphsurgeon as gs

from deployment.projects.bevfusion_l.config.bevfusion_deployment_config import BEVFusionDeploymentConfig
from deployment.projects.bevfusion_l.config.component_layout import has_component

logger = logging.getLogger(__name__)


def fix_topk_constant_k(model: onnx.ModelProto, num_proposals: int) -> onnx.ModelProto:
    """Replace the TopK node's ``K`` with a constant ``num_proposals``.

    TensorRT requires TopK's K to be a constant, but ``torch.onnx.export`` may produce a
    dynamic K. This rewrites the (single) TopK node in place and fixes up its output shapes.

    Args:
        model: Exported ONNX model containing the head's TopK node.
        num_proposals: Constant K to bake into the TopK node.

    Returns:
        The same ``model`` with the TopK ``K`` constant-folded (returned for transform chaining).
    """
    logger.info("Fixing TopK (K=%s) in ONNX graph...", num_proposals)
    graph = gs.import_onnx(model)

    topk_nodes = [node for node in graph.nodes if node.op == "TopK"]
    if len(topk_nodes) == 0:
        logger.warning("No TopK node found; skipping fix")
        return model

    if len(topk_nodes) != 1:
        logger.warning("Expected 1 TopK node, found %s; fixing the first one", len(topk_nodes))

    topk = topk_nodes[0]
    topk.inputs[1] = gs.Constant("K", values=np.array([num_proposals], dtype=np.int64))
    topk.outputs[0].shape = [1, num_proposals]
    topk.outputs[0].dtype = topk.inputs[0].dtype if topk.inputs[0].dtype else np.float32
    topk.outputs[1].shape = [1, num_proposals]
    topk.outputs[1].dtype = np.int64

    graph.cleanup().toposort()
    fixed = gs.export_onnx(graph)
    logger.info("TopK fixed (K=%s)", num_proposals)
    return fixed


def merge_split_sparse_dense_onnx(
    *,
    config: BEVFusionDeploymentConfig,
    sparse_onnx_path: Path,
    dense_onnx_path: Path,
    output_dir_path: Path,
    logger: logging.Logger = logger,
) -> Path:
    """Merge split ``sparse`` + ``dense`` ONNX into a single ``merged`` ONNX.

    Args:
        config: BEVFusion deploy config (supplies the merged component's I/O names and filename).
        sparse_onnx_path: Path to the exported sparse-encoder ONNX.
        dense_onnx_path: Path to the exported dense ONNX.
        output_dir_path: Directory to write the merged ``merged`` ONNX into.
        logger: Logger for progress messages.

    Returns:
        The path of the written merged ONNX.

    Raises:
        KeyError: If the ``bevfusion_merged`` component is not present in the config.
        FileNotFoundError: If either split ONNX is missing.
        RuntimeError: If ONNX compose utilities are unavailable.
    """
    if not has_component(config.components_cfg, "bevfusion_merged"):
        raise KeyError(
            "bevfusion_merge is enabled but components_cfg has no 'bevfusion_merged'. "
            "Ensure merge overlay is applied before export."
        )
    merged_cfg = config.components_cfg.get_component("bevfusion_merged")
    merged_path = output_dir_path / merged_cfg.onnx_file

    try:
        from onnx import compose as onnx_compose
    except Exception as e:
        raise RuntimeError("ONNX compose utilities unavailable; cannot merge split ONNX.") from e

    if not sparse_onnx_path.exists():
        raise FileNotFoundError(f"Sparse ONNX not found: {sparse_onnx_path}")
    if not dense_onnx_path.exists():
        raise FileNotFoundError(f"Dense ONNX not found: {dense_onnx_path}")

    sparse_model = onnx.load(str(sparse_onnx_path))
    dense_model = onnx.load(str(dense_onnx_path))

    # onnx.compose.merge_models requires identical IR/opset metadata.
    target_ir = max(int(sparse_model.ir_version), int(dense_model.ir_version))
    sparse_model.ir_version = target_ir
    dense_model.ir_version = target_ir

    sparse_opsets = {imp.domain: int(imp.version) for imp in sparse_model.opset_import}
    dense_opsets = {imp.domain: int(imp.version) for imp in dense_model.opset_import}
    merged_opsets = dict(sparse_opsets)
    for domain, version in dense_opsets.items():
        merged_opsets[domain] = max(version, merged_opsets.get(domain, version))
    merged_opset_ids = [onnx.helper.make_operatorsetid(d, v) for d, v in merged_opsets.items()]
    del sparse_model.opset_import[:]
    sparse_model.opset_import.extend(merged_opset_ids)
    del dense_model.opset_import[:]
    dense_model.opset_import.extend(merged_opset_ids)

    sparse_pref = onnx_compose.add_prefix(sparse_model, prefix="sparse/")
    dense_pref = onnx_compose.add_prefix(dense_model, prefix="dense/")

    sparse_out_name = config.components_cfg.get_component("bevfusion_sparse").io.outputs[0].name
    dense_in_name = config.components_cfg.get_component("bevfusion_dense").io.inputs[0].name
    io_map = [(f"sparse/{sparse_out_name}", f"dense/{dense_in_name}")]

    merged_model = onnx_compose.merge_models(sparse_pref, dense_pref, io_map=io_map)
    merged_graph = gs.import_onnx(merged_model)

    sparse_inputs = [inp.name for inp in config.components_cfg.get_component("bevfusion_sparse").io.inputs]
    dense_outputs = [out.name for out in config.components_cfg.get_component("bevfusion_dense").io.outputs]
    if len(merged_graph.inputs) != len(sparse_inputs):
        logger.warning(
            "Merged ONNX input count mismatch: graph=%d expected=%d",
            len(merged_graph.inputs),
            len(sparse_inputs),
        )
    if len(merged_graph.outputs) != len(dense_outputs):
        logger.warning(
            "Merged ONNX output count mismatch: graph=%d expected=%d",
            len(merged_graph.outputs),
            len(dense_outputs),
        )
    for i, name in enumerate(sparse_inputs):
        if i < len(merged_graph.inputs):
            merged_graph.inputs[i].name = name
    for i, name in enumerate(dense_outputs):
        if i < len(merged_graph.outputs):
            merged_graph.outputs[i].name = name

    merged_graph.cleanup().toposort()
    final_model = gs.export_onnx(merged_graph)
    # onnx.compose.merge_models concatenated both sub-models' (already-unified) opset lists, so the
    # merged graph lists each domain twice. Restore the single deduped union computed above so the
    # merged ONNX carries the same opset_import a monolithic single-graph export would.
    del final_model.opset_import[:]
    final_model.opset_import.extend(merged_opset_ids)
    onnx.save_model(final_model, str(merged_path))
    logger.info("Merged split ONNX -> %s", merged_path)
    return merged_path


def bevfusion_merge_finalize(
    exported_paths: List[str],
    output_dir_path: Path,
    config: BEVFusionDeploymentConfig,
) -> None:
    """Pipeline finalize hook: merge the split sparse+dense ONNX into a single ``merged`` ONNX.

    Matches :data:`deployment.export.pipelines.onnx_export_pipeline.FinalizeHook`. Resolves the
    split ONNX paths from the deploy config under ``output_dir_path`` and delegates to
    :func:`merge_split_sparse_dense_onnx`.

    Args:
        exported_paths: Per-component ONNX paths already written (unused; paths are resolved
            from the deploy config to avoid depending on export order).
        output_dir_path: Directory holding the exported ONNX files.
        config: BEVFusion deploy config with the split + merged component layout.
    """
    sparse_onnx = output_dir_path / config.components_cfg.get_component("bevfusion_sparse").onnx_file
    dense_onnx = output_dir_path / config.components_cfg.get_component("bevfusion_dense").onnx_file
    merge_split_sparse_dense_onnx(
        config=config,
        sparse_onnx_path=sparse_onnx,
        dense_onnx_path=dense_onnx,
        output_dir_path=output_dir_path,
    )
