"""Typed configuration helpers shared by exporter implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Tuple


def _empty_mapping() -> Mapping[Any, Any]:
    """Return an immutable empty mapping."""
    return MappingProxyType({})


@dataclass(frozen=True)
class TensorRTProfileConfig:
    """Optimization profile description for a TensorRT input tensor."""

    min_shape: Tuple[int, ...] = field(default_factory=tuple)
    opt_shape: Tuple[int, ...] = field(default_factory=tuple)
    max_shape: Tuple[int, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TensorRTProfileConfig:
        return cls(
            min_shape=cls._normalize_shape(data.get("min_shape")),
            opt_shape=cls._normalize_shape(data.get("opt_shape")),
            max_shape=cls._normalize_shape(data.get("max_shape")),
        )

    @staticmethod
    def _normalize_shape(shape: Optional[Iterable[int]]) -> Tuple[int, ...]:
        if shape is None:
            return tuple()
        return tuple(int(dim) for dim in shape)

    def has_complete_profile(self) -> bool:
        return bool(self.min_shape and self.opt_shape and self.max_shape)


@dataclass(frozen=True)
class TensorRTModelInputConfig:
    """Typed container for TensorRT model input shape settings."""

    input_shapes: Mapping[str, TensorRTProfileConfig] = field(default_factory=_empty_mapping)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TensorRTModelInputConfig:
        input_shapes_raw = data.get("input_shapes", {}) or {}
        profile_map = {
            name: TensorRTProfileConfig.from_dict(shape_dict or {}) for name, shape_dict in input_shapes_raw.items()
        }
        return cls(input_shapes=MappingProxyType(profile_map))


class BaseExporterConfig:
    """
    Base class for typed exporter configuration dataclasses.

    Concrete configs should extend this class and provide typed fields
    for all configuration parameters.
    """

    pass


@dataclass(frozen=True)
class ONNXExportConfig(BaseExporterConfig):
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
    simplify: bool = True
    opset_version: int = 16
    export_params: bool = True
    keep_initializers_as_inputs: bool = False
    verbose: bool = False
    do_constant_folding: bool = True
    save_file: str = "model.onnx"
    batch_size: Optional[int] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ONNXExportConfig:
        """Instantiate config from a plain mapping."""
        return cls(
            input_names=tuple(data.get("input_names", cls.input_names)),
            output_names=tuple(data.get("output_names", cls.output_names)),
            dynamic_axes=data.get("dynamic_axes"),
            simplify=data.get("simplify", cls.simplify),
            opset_version=data.get("opset_version", cls.opset_version),
            export_params=data.get("export_params", cls.export_params),
            keep_initializers_as_inputs=data.get("keep_initializers_as_inputs", cls.keep_initializers_as_inputs),
            verbose=data.get("verbose", cls.verbose),
            do_constant_folding=data.get("do_constant_folding", cls.do_constant_folding),
            save_file=data.get("save_file", cls.save_file),
            batch_size=data.get("batch_size", cls.batch_size),
        )


@dataclass(frozen=True)
class TensorRTExportConfig(BaseExporterConfig):
    """
    Typed schema describing TensorRT exporter configuration.

    Attributes:
        precision_policy: Name of the precision policy (matches PrecisionPolicy enum).
        policy_flags: Mapping of TensorRT builder/network flags.
        max_workspace_size: Workspace size in bytes.
        model_inputs: Tuple of TensorRTModelInputConfig entries describing shapes.
    """

    precision_policy: str = "auto"
    policy_flags: Mapping[str, bool] = field(default_factory=dict)
    max_workspace_size: int = 1 << 30
    model_inputs: Tuple[TensorRTModelInputConfig, ...] = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TensorRTExportConfig:
        """Instantiate config from a plain mapping."""
        inputs_raw = data.get("model_inputs") or ()
        parsed_inputs = tuple(
            entry if isinstance(entry, TensorRTModelInputConfig) else TensorRTModelInputConfig.from_dict(entry)
            for entry in inputs_raw
        )
        return cls(
            precision_policy=str(data.get("precision_policy", cls.precision_policy)),
            policy_flags=MappingProxyType(dict(data.get("policy_flags", {}))),
            max_workspace_size=int(data.get("max_workspace_size", cls.max_workspace_size)),
            model_inputs=parsed_inputs,
        )


__all__ = [
    "BaseExporterConfig",
    "ONNXExportConfig",
    "TensorRTExportConfig",
    "TensorRTModelInputConfig",
    "TensorRTProfileConfig",
]
