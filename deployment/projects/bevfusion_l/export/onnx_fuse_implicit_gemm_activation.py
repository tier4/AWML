"""Fuse a post-spconv activation into the ``autoware`` ImplicitGemm plugin.

TensorRT does not fuse a standard ONNX ``Relu`` with a custom op, so we fold the
pattern ``ImplicitGemm -> Relu`` by hand:

    * set ``act_type = kReLU`` on the ImplicitGemm node, and
    * delete the now-redundant standalone ``Relu`` node.

The public entry point is :func:`fuse_autoware_implicit_gemm_trailing_relu`.
"""

from __future__ import annotations

from typing import Dict, List

import onnx
from onnx import helper

# ImplicitGemm ``act_type`` values (mirrors the plugin's enum).
_ACT_NONE = 0
_ACT_RELU = 1

_ONNX_DOMAINS = ("", "ai.onnx")


def _normalize_attr(name: str) -> str:
    """Strip an ONNX type suffix (``_f``, ``_i``, ``_s``, ``_l``) from an attribute name."""
    for suffix in ("_f", "_i", "_s", "_l"):
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return name


def _read_int_and_float_attrs(node: onnx.NodeProto) -> Dict[str, object]:
    """Return the node's INT/FLOAT attributes keyed by their normalized name."""
    attrs: Dict[str, object] = {}
    for attr in node.attribute:
        base = _normalize_attr(attr.name)
        if attr.type == onnx.AttributeProto.INT:
            attrs[base] = int(attr.i)
        elif attr.type == onnx.AttributeProto.FLOAT:
            attrs[base] = float(attr.f)
    return attrs


def _set_act_type(node: onnx.NodeProto, act_type: int) -> None:
    """Replace the node's ``act_type`` attribute with ``act_type``."""
    kept = [a for a in node.attribute if _normalize_attr(a.name) != "act_type"]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute("act_type", int(act_type)))


def _rename_tensor(graph: onnx.GraphProto, old: str, new: str) -> None:
    """Rewire every reference to tensor ``old`` so it points at ``new`` instead."""
    if old == new:
        return
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp == old:
                node.input[i] = new
    for out in graph.output:
        if out.name == old:
            out.name = new
    for value_info in graph.value_info:
        if value_info.name == old:
            value_info.name = new


def _is_onnx_relu(node: onnx.NodeProto) -> bool:
    return (
        node.op_type == "Relu"
        and node.domain in _ONNX_DOMAINS
        and len(node.input) >= 1
        and bool(node.input[0])
        and len(node.output) >= 1
        and bool(node.output[0])
    )


def _is_autoware_implicit_gemm(node: onnx.NodeProto) -> bool:
    return node.op_type == "ImplicitGemm" and node.domain == "autoware"


def fuse_autoware_implicit_gemm_trailing_relu(model: onnx.ModelProto) -> int:
    """Fold each ``ImplicitGemm -> Relu`` pair into a single activated ImplicitGemm.

    For every ONNX ``Relu`` whose input is produced by an ``autoware.ImplicitGemm``
    that feeds nothing else, the ImplicitGemm's ``act_type`` is set to kReLU and the
    ``Relu`` node is removed.

    Returns the number of ``Relu`` nodes removed.
    """
    graph = model.graph

    # Map each tensor to the node index producing it, and count consumers per tensor.
    # Built once up front: the only nodes we remove are the fused Relus, which never
    # feed an ImplicitGemm, so removals cannot change any decision made below.
    producer_of: Dict[str, int] = {}
    consumer_count: Dict[str, int] = {}
    for ni, node in enumerate(graph.node):
        for out in node.output:
            if out:
                producer_of[out] = ni
        for inp in node.input:
            if inp:
                consumer_count[inp] = consumer_count.get(inp, 0) + 1

    remove_idx: set[int] = set()

    for ri, relu in enumerate(graph.node):
        if not _is_onnx_relu(relu):
            continue

        gemm_out = relu.input[0]

        # The ImplicitGemm output must feed this Relu and nothing else.
        if consumer_count.get(gemm_out, 0) != 1:
            continue

        producer_i = producer_of.get(gemm_out)
        if producer_i is None:
            continue
        producer = graph.node[producer_i]
        if not _is_autoware_implicit_gemm(producer):
            continue

        # Only fuse when the ImplicitGemm has no activation yet (or already kReLU).
        cur_act = int(_read_int_and_float_attrs(producer).get("act_type", 0) or 0)
        if cur_act not in (_ACT_NONE, _ACT_RELU):
            continue

        _set_act_type(producer, _ACT_RELU)
        _rename_tensor(graph, relu.output[0], gemm_out)
        remove_idx.add(ri)

    if not remove_idx:
        return 0

    kept: List[onnx.NodeProto] = [n for i, n in enumerate(graph.node) if i not in remove_idx]
    del graph.node[:]
    graph.node.extend(kept)
    return len(remove_idx)
