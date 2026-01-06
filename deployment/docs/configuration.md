# Configuration Reference

Configurations remain dictionary-driven for flexibility, with typed dataclasses layered on top for validation and IDE support.

## Structure

```python
# Task type
task_type = "detection3d"  # or "detection2d", "classification"

# Checkpoint (single source of truth)
checkpoint_path = "model.pth"

devices = dict(
    cpu="cpu",
    cuda="cuda:0",
)

export = dict(
    mode="both",          # "onnx", "trt", "both", "none"
    work_dir="work_dirs/deployment",
    onnx_path=None,       # Required when mode="trt" and ONNX already exists
)

runtime_io = dict(
    info_file="data/info.pkl",
    sample_idx=0,
)

model_io = dict(
    input_name="input",
    input_shape=(3, 960, 960),
    input_dtype="float32",
    output_name="output",
    batch_size=1,
    dynamic_axes={...},
)

onnx_config = dict(
    opset_version=16,
    do_constant_folding=True,
    save_file="model.onnx",
    multi_file=False,
)

tensorrt_config = dict(
    common_config=dict(
        precision_policy="auto",
        max_workspace_size=1 << 30,
    ),
)

verification = dict(
    enabled=True,
    num_verify_samples=3,
    tolerance=0.1,
    devices=devices,
    scenarios={
        "both": [
            {"ref_backend": "pytorch", "ref_device": "cpu",
             "test_backend": "onnx", "test_device": "cuda"},
        ]
    }
)

evaluation = dict(
    enabled=True,
    num_samples=100,
    verbose=False,
    backends={
        "pytorch": {"enabled": True, "device": devices["cpu"]},
        "onnx": {"enabled": True, "device": devices["cpu"]},
        "tensorrt": {"enabled": True, "device": devices["cuda"]},
    }
)
```

### Device Aliases

Keep device definitions centralized by declaring a top-level `devices` dictionary and referencing aliases (for example, `devices["cuda"]`). Updating the mapping once automatically propagates to export, evaluation, and verification blocks without digging into nested dictionaries.

## Backend Enum

Use `deployment.core.Backend` to avoid typos while keeping backward compatibility with plain strings.

```python
from deployment.core import Backend

evaluation = dict(
    backends={
        Backend.PYTORCH: {"enabled": True, "device": devices["cpu"]},
        Backend.ONNX: {"enabled": True, "device": devices["cpu"]},
        Backend.TENSORRT: {"enabled": True, "device": devices["cuda"]},
    }
)
```

## Typed Exporter Configs

Typed classes in `deployment.exporters.common.configs` provide schema validation and IDE hints.

```python
from deployment.exporters.common.configs import (
    ONNXExportConfig,
    TensorRTExportConfig,
    TensorRTModelInputConfig,
    TensorRTProfileConfig,
)

onnx_config = ONNXExportConfig(
    input_names=("input",),
    output_names=("output",),
    opset_version=16,
    do_constant_folding=True,
    simplify=True,
    save_file="model.onnx",
    batch_size=1,
)

trt_config = TensorRTExportConfig(
    precision_policy="auto",
    max_workspace_size=1 << 30,
    model_inputs=(
        TensorRTModelInputConfig(
            input_shapes={
                "input": TensorRTProfileConfig(
                    min_shape=(1, 3, 960, 960),
                    opt_shape=(1, 3, 960, 960),
                    max_shape=(1, 3, 960, 960),
                )
            }
        ),
    ),
)
```

Use `from_mapping()` / `from_dict()` helpers to instantiate typed configs from existing dictionaries.

## Example Config Paths

- `deployment/projects/centerpoint/config/deploy_config.py`
