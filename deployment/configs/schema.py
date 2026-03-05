"""
Typed schema for deployment config: frozen dataclasses, validation, normalize.

No torch runtime checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from deployment.configs.enums import (
    DEFAULT_WORKSPACE_SIZE,
    PRECISION_POLICIES,
    ExportMode,
    PrecisionPolicy,
)
from deployment.core.backend import Backend
from deployment.exporters.common.configs import TensorRTProfileConfig


def _empty_mapping() -> Mapping[Any, Any]:
    """Return an immutable empty mapping."""
    return MappingProxyType({})


def _normalize_dynamic_axes(raw: Mapping[str, Any]) -> Dict[str, Dict[int, str]]:
    """Normalize dynamic_axes inner dict keys to int."""
    result: Dict[str, Dict[int, str]] = {}
    for name, axes in (raw or {}).items():
        if not isinstance(axes, Mapping):
            raise TypeError(f"dynamic_axes['{name}'] must be a mapping, got {type(axes).__name__}")
        result[name] = {int(k): str(v) for k, v in axes.items()}
    return result


# -----------------------------------------------------------------------------
# Export / Device / Runtime
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportConfig:
    """Configuration for model export settings."""

    mode: ExportMode = ExportMode.BOTH
    work_dir: str = "work_dirs"
    onnx_path: Optional[str] = None

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> ExportConfig:
        """Create ExportConfig from dict."""
        return cls(
            mode=ExportMode.from_value(config_dict.get("mode", ExportMode.BOTH)),
            work_dir=config_dict.get("work_dir", cls.work_dir),
            onnx_path=config_dict.get("onnx_path"),
        )

    @property
    def should_export_onnx(self) -> bool:
        """Whether ONNX export is requested."""
        return self.mode in (ExportMode.ONNX, ExportMode.BOTH)

    @property
    def should_export_tensorrt(self) -> bool:
        """Whether TensorRT export is requested."""
        return self.mode in (ExportMode.TRT, ExportMode.BOTH)


@dataclass(frozen=True)
class DeviceConfig:
    """Normalized device settings shared across deployment stages."""

    cpu: str = "cpu"
    cuda: Optional[str] = "cuda:0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "cpu", self._normalize_cpu(self.cpu))
        object.__setattr__(self, "cuda", self._normalize_cuda(self.cuda))

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> DeviceConfig:
        """Create DeviceConfig from dict."""
        return cls(cpu=config_dict.get("cpu", cls.cpu), cuda=config_dict.get("cuda", cls.cuda))

    @staticmethod
    def _normalize_cpu(device: Optional[str]) -> str:
        """Normalize CPU device string."""
        if not device:
            return "cpu"
        normalized = str(device).strip().lower()
        if normalized.startswith("cuda"):
            raise ValueError("CPU device cannot be a CUDA device")
        return normalized

    @staticmethod
    def _normalize_cuda(device: Optional[str]) -> Optional[str]:
        """Normalize CUDA device string to 'cuda:N' format."""
        if device is None:
            return None
        if not isinstance(device, str):
            raise ValueError("cuda device must be a string (e.g., 'cuda:0')")
        normalized = device.strip().lower()
        if normalized == "":
            return None
        if normalized == "cuda":
            normalized = "cuda:0"
        if not normalized.startswith("cuda"):
            raise ValueError(f"Invalid CUDA device '{device}'. Must start with 'cuda'")
        suffix = normalized.split(":", 1)[1] if ":" in normalized else "0"
        suffix = suffix.strip() or "0"
        if not suffix.isdigit():
            raise ValueError(f"Invalid CUDA device index in '{device}'")
        device_id = int(suffix)
        if device_id < 0:
            raise ValueError("CUDA device index must be non-negative")
        return f"cuda:{device_id}"

    @property
    def cuda_device_index(self) -> Optional[int]:
        """Return CUDA device index as integer (if configured)."""
        if self.cuda is None:
            return None
        return int(self.cuda.split(":", 1)[1])


@dataclass(frozen=True)
class RuntimeConfig:
    """Configuration for runtime I/O settings."""

    info_file: str = ""
    sample_idx: int = 0

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> RuntimeConfig:
        """Create RuntimeConfig from dictionary."""
        return cls(
            info_file=config_dict.get("info_file", ""),
            sample_idx=config_dict.get("sample_idx", 0),
        )


@dataclass(frozen=True)
class OnnxConfig:
    """ONNX export settings (shared across all components)."""

    opset_version: int = 16
    do_constant_folding: bool = True
    export_params: bool = True
    keep_initializers_as_inputs: bool = False
    simplify: bool = False

    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, Any]]) -> OnnxConfig:
        """Build OnnxConfig from deploy_cfg['onnx_config']."""
        if not raw:
            return cls()
        if not isinstance(raw, Mapping):
            raise TypeError(f"onnx_config must be a mapping, got {type(raw).__name__}")
        return cls(
            opset_version=int(raw.get("opset_version", 16)),
            do_constant_folding=bool(raw.get("do_constant_folding", True)),
            export_params=bool(raw.get("export_params", True)),
            keep_initializers_as_inputs=bool(raw.get("keep_initializers_as_inputs", False)),
            simplify=bool(raw.get("simplify", False)),
        )


@dataclass(frozen=True)
class TensorRTConfig:
    """
    Configuration for TensorRT backend-specific settings.

    Uses config structure:
        tensorrt_config = dict(precision_policy="auto", max_workspace_size=1<<30)

    TensorRT profiles are defined in components.*.tensorrt_profile.

    Note:
        The deploy config key for this section is **`tensorrt_config`**.
    """

    precision_policy: str = PrecisionPolicy.AUTO.value
    max_workspace_size: int = DEFAULT_WORKSPACE_SIZE

    def __post_init__(self) -> None:
        """Validate TensorRT precision policy at construction time."""
        if self.precision_policy not in PRECISION_POLICIES:
            raise ValueError(
                f"Invalid precision_policy '{self.precision_policy}'. "
                f"Must be one of {list(PRECISION_POLICIES.keys())}"
            )

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> TensorRTConfig:
        return cls(
            precision_policy=config_dict.get("precision_policy", PrecisionPolicy.AUTO.value),
            max_workspace_size=config_dict.get("max_workspace_size", DEFAULT_WORKSPACE_SIZE),
        )

    @property
    def precision_flags(self) -> Mapping[str, bool]:
        """TensorRT precision flags for the configured policy."""
        return PRECISION_POLICIES[self.precision_policy]


# -----------------------------------------------------------------------------
# Component config (deploy_cfg["components"])
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class InputSpec:
    """Single input name/dtype for a component."""

    name: str
    dtype: str = "float32"


@dataclass(frozen=True)
class OutputSpec:
    """Single output name/dtype for a component."""

    name: str
    dtype: str = "float32"


@dataclass(frozen=True)
class ComponentIO:
    """I/O specification for a component (inputs, outputs, dynamic_axes)."""

    inputs: List[InputSpec]
    outputs: List[OutputSpec]
    dynamic_axes: Dict[str, Dict[int, str]]


@dataclass(frozen=True)
class ComponentCfg:
    """Configuration for one deployable component.

    The component identifier is the key in deploy_cfg['components']; ``name`` is always set
    from that key.
    """

    name: str
    onnx_file: str
    engine_file: str
    io: ComponentIO
    tensorrt_profile: Dict[str, TensorRTProfileConfig]


@dataclass(frozen=True)
class ComponentsConfig:
    """Unified component configuration: mapping of component id -> ComponentCfg.

    The dict key is the component identifier (e.g. "model", "pts_voxel_encoder", "pts_backbone_neck_head").
    """

    _components: Mapping[str, ComponentCfg]

    def get_component(self, component_name: str) -> ComponentCfg:
        """Get component config by name. Raises KeyError if not found."""
        if component_name not in self._components:
            raise KeyError(f"Unknown component: {component_name}. Available: {list(self._components.keys())}")
        return self._components[component_name]

    def get_artifact_filename(self, component_name: str, file_key: str) -> Optional[str]:
        """Return artifact filename for path resolution (onnx_file or engine_file)."""
        component_cfg = self._components.get(component_name)
        if component_cfg is None:
            raise KeyError(f"Unknown component: {component_name}. Available: {list(self._components.keys())}")
        return getattr(component_cfg, file_key)

    def component_names(self) -> Iterable[str]:
        """Iterate over component names."""
        return self._components.keys()

    def items(self) -> Iterable[Tuple[str, ComponentCfg]]:
        """Iterate (name, ComponentCfg) pairs."""
        return self._components.items()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ComponentsConfig:
        """Build ComponentsConfig from deploy_cfg['components'] dict. Generic: any keys allowed."""
        if not isinstance(raw, Mapping):
            raise TypeError(f"components must be a mapping, got {type(raw).__name__}")
        parsed = {}
        for component_name, comp_raw in raw.items():
            parsed[component_name] = cls._parse_component(comp_raw, component_name)
        return cls(_components=MappingProxyType(parsed))

    @classmethod
    def _parse_component(cls, comp_raw: Any, component_name: str) -> ComponentCfg:
        if not isinstance(comp_raw, Mapping):
            raise TypeError(f"components['{component_name}'] must be a mapping, got {type(comp_raw).__name__}")
        for field_name in ("onnx_file", "engine_file", "io"):
            if field_name not in comp_raw:
                raise KeyError(f"components['{component_name}'] must define '{field_name}'.")
        component_id = component_name
        io_raw = comp_raw["io"]
        if not isinstance(io_raw, Mapping):
            raise TypeError(f"components['{component_name}'].io must be a mapping, got {type(io_raw).__name__}")
        if "outputs" not in io_raw or not io_raw["outputs"]:
            raise KeyError(f"components['{component_name}'].io.outputs must be a non-empty list.")
        if "inputs" not in io_raw or not io_raw["inputs"]:
            raise KeyError(f"components['{component_name}'].io.inputs must be a non-empty list.")
        outputs = []
        for i, out in enumerate(io_raw["outputs"]):
            if not isinstance(out, Mapping) or "name" not in out:
                raise KeyError(f"components['{component_name}'].io.outputs[{i}] must define 'name'.")
            output_name = out["name"]
            if not output_name or not isinstance(output_name, str):
                raise ValueError(f"components['{component_name}'].io.outputs[{i}].name must be a non-empty string.")
            outputs.append(OutputSpec(name=output_name, dtype=out.get("dtype", "float32")))
        inputs = []
        for i, inp in enumerate(io_raw["inputs"]):
            if not isinstance(inp, Mapping) or "name" not in inp:
                raise KeyError(f"components['{component_name}'].io.inputs[{i}] must define 'name'.")
            n = inp["name"]
            if not n or not isinstance(n, str):
                raise ValueError(f"components['{component_name}'].io.inputs[{i}].name must be a non-empty string.")
            inputs.append(InputSpec(name=n, dtype=inp.get("dtype", "float32")))
        dynamic_axes = _normalize_dynamic_axes(io_raw.get("dynamic_axes") or {})
        io = ComponentIO(
            inputs=inputs,
            outputs=outputs,
            dynamic_axes=dynamic_axes,
        )
        profile_raw = comp_raw.get("tensorrt_profile") or {}
        if not isinstance(profile_raw, Mapping):
            raise TypeError(f"components['{component_name}'].tensorrt_profile must be a mapping.")
        tensorrt_profile = {}
        for input_name, shape_cfg in profile_raw.items():
            if not isinstance(shape_cfg, Mapping):
                raise TypeError(
                    f"components['{component_name}'].tensorrt_profile['{input_name}'] must be a mapping, got {type(shape_cfg).__name__}."
                )
            tensorrt_profile[input_name] = TensorRTProfileConfig.from_dict(shape_cfg)
        return ComponentCfg(
            name=component_id,
            onnx_file=str(comp_raw["onnx_file"]),
            engine_file=str(comp_raw["engine_file"]),
            io=io,
            tensorrt_profile=tensorrt_profile,
        )


# -----------------------------------------------------------------------------
# Evaluation & Verification
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationConfig:
    """Typed configuration for evaluation settings."""

    enabled: bool = False
    num_samples: int = 10
    verbose: bool = False
    backends: Mapping[Any, Mapping[str, Any]] = field(default_factory=_empty_mapping)
    models: Mapping[Any, Any] = field(default_factory=_empty_mapping)
    devices: Mapping[str, str] = field(default_factory=_empty_mapping)

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> EvaluationConfig:
        backends_raw = config_dict.get("backends", None)
        if backends_raw is None:
            backends_raw = {}
        if not isinstance(backends_raw, Mapping):
            raise TypeError(f"evaluation.backends must be a mapping, got {type(backends_raw).__name__}")
        backends_frozen = {key: MappingProxyType(dict(value)) for key, value in backends_raw.items()}

        models_raw = config_dict.get("models", None)
        if models_raw is None:
            models_raw = {}
        if not isinstance(models_raw, Mapping):
            raise TypeError(f"evaluation.models must be a mapping, got {type(models_raw).__name__}")

        devices_raw = config_dict.get("devices", None)
        if devices_raw is None:
            devices_raw = {}
        if not isinstance(devices_raw, Mapping):
            raise TypeError(f"evaluation.devices must be a mapping, got {type(devices_raw).__name__}")

        return cls(
            enabled=config_dict.get("enabled", False),
            num_samples=config_dict.get("num_samples", 10),
            verbose=config_dict.get("verbose", False),
            backends=MappingProxyType(backends_frozen),
            models=MappingProxyType(dict(models_raw)),
            devices=MappingProxyType(dict(devices_raw)),
        )


@dataclass(frozen=True)
class VerificationScenario:
    """Immutable verification scenario specification."""

    ref_backend: Backend
    ref_device: str
    test_backend: Backend
    test_device: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VerificationScenario:
        missing_keys = {"ref_backend", "ref_device", "test_backend", "test_device"} - data.keys()
        if missing_keys:
            raise ValueError(f"Verification scenario missing keys: {missing_keys}")

        return cls(
            ref_backend=Backend.from_value(data["ref_backend"]),
            ref_device=str(data["ref_device"]),
            test_backend=Backend.from_value(data["test_backend"]),
            test_device=str(data["test_device"]),
        )


@dataclass(frozen=True)
class VerificationConfig:
    """Typed configuration for verification settings."""

    enabled: bool = True
    num_verify_samples: int = 3
    tolerance: float = 0.1
    devices: Mapping[str, str] = field(default_factory=_empty_mapping)
    scenarios: Mapping[ExportMode, Tuple[VerificationScenario, ...]] = field(default_factory=_empty_mapping)

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> VerificationConfig:
        scenarios_raw = config_dict.get("scenarios")
        if scenarios_raw is None:
            scenarios_raw = {}
        if not isinstance(scenarios_raw, Mapping):
            raise TypeError(f"verification.scenarios must be a mapping, got {type(scenarios_raw).__name__}")

        scenario_map: Dict[ExportMode, Tuple[VerificationScenario, ...]] = {}
        for mode_key, scenario_list in scenarios_raw.items():
            mode = ExportMode.from_value(mode_key)
            if scenario_list is None:
                scenario_list = []
            elif not isinstance(scenario_list, (list, tuple)):
                raise TypeError(
                    f"verification.scenarios.{mode_key} must be a list or tuple, got {type(scenario_list).__name__}"
                )
            scenario_entries = tuple(VerificationScenario.from_dict(entry) for entry in scenario_list)
            scenario_map[mode] = scenario_entries

        devices_raw = config_dict.get("devices")
        if devices_raw is None:
            devices_raw = {}
        if not isinstance(devices_raw, Mapping):
            raise TypeError(f"verification.devices must be a mapping, got {type(devices_raw).__name__}")

        return cls(
            enabled=config_dict.get("enabled", True),
            num_verify_samples=config_dict.get("num_verify_samples", 3),
            tolerance=config_dict.get("tolerance", 0.1),
            devices=MappingProxyType(dict(devices_raw)),
            scenarios=MappingProxyType(scenario_map),
        )

    def get_scenarios(self, mode: ExportMode) -> Tuple[VerificationScenario, ...]:
        """Return scenarios for a specific export mode."""
        return self.scenarios.get(mode, ())
