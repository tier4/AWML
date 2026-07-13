"""Typed configuration helpers shared by exporter implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Tuple

from deployment.config.enums import PrecisionPolicy


@dataclass(frozen=True)
class TensorRTProfileConfig:
    """Optimization profile description for a TensorRT input tensor."""

    min_shape: Tuple[int, ...] = field(default_factory=tuple)
    opt_shape: Tuple[int, ...] = field(default_factory=tuple)
    max_shape: Tuple[int, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TensorRTProfileConfig:
        return cls(
            min_shape=cls._to_shape_tuple(data.get("min_shape")),
            opt_shape=cls._to_shape_tuple(data.get("opt_shape")),
            max_shape=cls._to_shape_tuple(data.get("max_shape")),
        )

    @staticmethod
    def _to_shape_tuple(shape: Optional[Iterable[int]]) -> Tuple[int, ...]:
        """Convert an iterable of dimensions into an int tuple; None becomes an empty tuple."""
        if shape is None:
            return tuple()
        return tuple(int(dim) for dim in shape)


@dataclass(frozen=True)
class TensorRTModelInputConfig:
    """TensorRT model input shape settings."""

    input_shapes: Mapping[str, TensorRTProfileConfig] = field(default_factory=dict)


@dataclass(frozen=True)
class ONNXExportConfig:
    """
    Typed schema describing ONNX exporter configuration.

    Attributes:
        input_names: Ordered collection of input tensor names.
        output_names: Ordered collection of output tensor names.
        dynamic_axes: Optional dynamic axes mapping identical to torch.onnx API.
        simplify: Whether to run onnx-simplifier after export.
        opset_version: ONNX opset to target.
        export_params: Whether to embed weights inside the ONNX file.
        keep_initializers_as_inputs: Mirror of torch.onnx flag.
        verbose: Whether to log torch.onnx export graph debugging.
        do_constant_folding: Whether to enable constant folding.
        save_file: Output filename for the ONNX model.
        batch_size: Fixed batch size for export (None for dynamic batch).
    """

    input_names: Tuple[str, ...] = ("input",)
    output_names: Tuple[str, ...] = ("output",)
    dynamic_axes: Optional[Mapping[str, Mapping[int, str]]] = None
    simplify: bool = False
    opset_version: int = 17
    export_params: bool = True
    keep_initializers_as_inputs: bool = False
    verbose: bool = False
    do_constant_folding: bool = True
    save_file: str = "model.onnx"
    batch_size: Optional[int] = None


@dataclass(frozen=True)
class TensorRTExportConfig:
    """
    Typed schema describing TensorRT exporter configuration.

    Attributes:
        precision_policy: Precision policy; the exporter maps it to concrete TensorRT flags.
        max_workspace_size: Workspace size in bytes.
        model_input: Per-input optimization-profile shapes. A single config already maps
            multiple named inputs via ``input_shapes``; None means no dynamic profile.
        plugin_libraries: Custom TensorRT plugin ``.so`` paths to load before building
            the engine (e.g. the BEVFusion spconv ImplicitGemm plugin). Empty by default.
    """

    precision_policy: PrecisionPolicy = PrecisionPolicy.AUTO
    max_workspace_size: int = 1 << 30
    model_input: Optional[TensorRTModelInputConfig] = None
    plugin_libraries: Tuple[str, ...] = ()
