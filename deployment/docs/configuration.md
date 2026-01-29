# Configuration Reference

Configurations remain dictionary-driven for flexibility, with typed dataclasses layered on top for validation and IDE support.

## Structure

### Single-Model Export (Simple Models)

For simple models with a single ONNX/TensorRT output:

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
)

tensorrt_config = dict(
    precision_policy="auto",
    max_workspace_size=1 << 30,
)
```

### Multi-File Export (Complex Models like CenterPoint)

For models that export to multiple ONNX/TensorRT files, use the unified `components` config:

```python
task_type = "detection3d"
checkpoint_path = "work_dirs/centerpoint/best_checkpoint.pth"

devices = dict(
    cpu="cpu",
    cuda="cuda:0",
)

export = dict(
    mode="both",
    work_dir="work_dirs/centerpoint_deployment",
)

# Unified component configuration (single source of truth)
# Each component defines: name, file paths, IO spec, and TensorRT profile
components = dict(
    voxel_encoder=dict(
        name="pts_voxel_encoder",
        onnx_file="pts_voxel_encoder.onnx",
        engine_file="pts_voxel_encoder.engine",
        io=dict(
            inputs=[dict(name="input_features", dtype="float32")],
            outputs=[dict(name="pillar_features", dtype="float32")],
            dynamic_axes={
                "input_features": {0: "num_voxels", 1: "num_max_points"},
                "pillar_features": {0: "num_voxels"},
            },
        ),
        tensorrt_profile=dict(
            input_features=dict(
                min_shape=[1000, 32, 11],
                opt_shape=[20000, 32, 11],
                max_shape=[64000, 32, 11],
            ),
        ),
    ),
    backbone_head=dict(
        name="pts_backbone_neck_head",
        onnx_file="pts_backbone_neck_head.onnx",
        engine_file="pts_backbone_neck_head.engine",
        io=dict(
            inputs=[dict(name="spatial_features", dtype="float32")],
            outputs=[
                dict(name="heatmap", dtype="float32"),
                dict(name="reg", dtype="float32"),
                # ... more outputs
            ],
            dynamic_axes={...},
        ),
        tensorrt_profile=dict(
            spatial_features=dict(
                min_shape=[1, 32, 760, 760],
                opt_shape=[1, 32, 760, 760],
                max_shape=[1, 32, 760, 760],
            ),
        ),
    ),
)

# Shared ONNX settings (applied to all components)
onnx_config = dict(
    opset_version=16,
    do_constant_folding=True,
    simplify=False,
)

# Shared TensorRT settings (applied to all components)
tensorrt_config = dict(
    precision_policy="auto",
    max_workspace_size=2 << 30,
)
```

### Verification and Evaluation

```python
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

- `deployment/projects/centerpoint/config/deploy_config.py` - Multi-file export example
