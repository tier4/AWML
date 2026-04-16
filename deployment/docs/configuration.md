# Configuration reference

This is the single source of truth for deploy config fields: top-level keys, `components`, devices, `export`, `onnx_config`, `tensorrt_config`, `evaluation`, `verification`, and logging.

Use [runbook.md](./runbook.md) for execution behavior and [architecture.md](./architecture.md) for framework structure. This page is intentionally reference-first.

Deploy configs are plain Python dicts loaded with MMEngine `Config.fromfile`. `BaseDeploymentConfig` wraps them with typed dataclasses in `deployment.configs.schema` for validation and IDE-friendly access.

## How to read this config

Read the deploy config in this order:

1. `checkpoint_path`, `devices`, and `export` define the run boundary.
2. `components` defines what artifacts are produced and how each subgraph is named.
3. `onnx_config` and `tensorrt_config` tune exporter behavior.
4. `evaluation` and `verification` define the post-export quality gates.

## Top-level keys

| Key | Required | Purpose |
| --- | --- | --- |
| `checkpoint_path` | Yes | Path to the PyTorch checkpoint (must exist). Single source for load + PyTorch backend. |
| `deploy_log_path` | No | File for deployment logs. Default `"deployment.log"`. Relative paths are resolved under `export.work_dir`. `None` or `""` disables file logging. |
| `devices` | Yes | `cpu` / `cuda` device strings shared by export, verification, and evaluation. |
| `export` | Yes | Export mode, `work_dir`, optional `onnx_path` (e.g. when `mode="trt"`). |
| `components` | Yes | Multi-component I/O and artifact names (see below). Current runner expects this section. |
| `runtime_io` | Yes | Paths/indices for runtime data (e.g. `info_file`, `sample_idx`). |
| `onnx_config` | No | Shared ONNX export flags (`opset_version`, `simplify`, …). Per-component filenames live under `components.*.onnx_file`. |
| `tensorrt_config` | No | Shared TensorRT build flags. Per-component profiles live under `components.*.tensorrt_profile`. |
| `evaluation` | No | Backend toggles, sample counts, optional `model_dir` / `engine_dir` for ONNX and TensorRT. |
| `verification` | No | Scenarios per export mode, tolerance, `devices` map for verification. |

## Logging (`deploy_log_path`)

After `BaseDeploymentConfig` loads, the CenterPoint entrypoint attaches a root `FileHandler` via `deployment.cli.args.add_deployment_file_logging` when `resolved_deploy_log_file` is set. All standard `logging` output for the process (console + libraries that log to the root logger) is mirrored to that file.

```python
# Relative → join(export.work_dir, "deployment.log")
deploy_log_path = "deployment.log"

# Absolute path
# deploy_log_path = "/var/log/centerpoint_deploy.log"

# Disable file logging
# deploy_log_path = None
```

## Single ONNX / engine (one component)

The schema is still the unified `components` map: use **one** entry when the graph exports to a single ONNX and engine.

```python
checkpoint_path = "work_dirs/model/best.pth"
deploy_log_path = "deployment.log"

devices = dict(cpu="cpu", cuda="cuda:0")

export = dict(
    mode="both",
    work_dir="work_dirs/my_deployment",
    onnx_path=None,
)

runtime_io = dict(
    info_file="data/info.pkl",
    sample_idx=0,
)

components = dict(
    model=dict(
        onnx_file="model.onnx",
        engine_file="model.engine",
        io=dict(
            inputs=[dict(name="input", dtype="float32")],
            outputs=[dict(name="output", dtype="float32")],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        ),
        tensorrt_profile=dict(
            input=dict(
                min_shape=[1, 3, 960, 960],
                opt_shape=[1, 3, 960, 960],
                max_shape=[4, 3, 960, 960],
            ),
        ),
    ),
)

onnx_config = dict(
    opset_version=17,
    do_constant_folding=True,
    export_params=True,
    keep_initializers_as_inputs=False,
    simplify=False,
)

tensorrt_config = dict(
    precision_policy="auto",
    max_workspace_size=1 << 30,
)
```

There is **no** top-level `model_io` in the current `BaseDeploymentConfig`; I/O and filenames belong to `components`.

## Multi-component export (CenterPoint-style)

The **dict key** is the component id (e.g. `pts_voxel_encoder`, `pts_backbone_neck_head`). Typed parsing sets `ComponentCfg.name` from that key—do **not** add a separate `name` field inside each component dict.

Align shapes and dynamic axes with your model config. Example structure (abbreviated; see `deployment/projects/centerpoint/config/deploy_config.py` for the full reference):

```python
checkpoint_path = "work_dirs/centerpoint/best_checkpoint.pth"
deploy_log_path = "deployment.log"

devices = dict(cpu="cpu", cuda="cuda:0")

_WORK_DIR = "work_dirs/centerpoint_deployment"
export = dict(
    mode="both",
    work_dir=_WORK_DIR,
    onnx_path=f"{_WORK_DIR}/onnx",
)

components = dict(
    pts_voxel_encoder=dict(
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
                max_shape=[96000, 32, 11],
            ),
        ),
    ),
    pts_backbone_neck_head=dict(
        onnx_file="pts_backbone_neck_head.onnx",
        engine_file="pts_backbone_neck_head.engine",
        io=dict(
            inputs=[dict(name="spatial_features", dtype="float32")],
            outputs=[
                dict(name="heatmap", dtype="float32"),
                dict(name="reg", dtype="float32"),
                # ... remaining heads
            ],
            dynamic_axes={
                "spatial_features": {0: "batch_size", 2: "height", 3: "width"},
                # ... align per output
            },
        ),
        tensorrt_profile=dict(
            spatial_features=dict(
                min_shape=[1, 32, 1020, 1020],
                opt_shape=[1, 32, 1020, 1020],
                max_shape=[1, 32, 1020, 1020],
            ),
        ),
    ),
)

onnx_config = dict(
    opset_version=17,
    do_constant_folding=True,
    export_params=True,
    keep_initializers_as_inputs=False,
    simplify=False,
)

tensorrt_config = dict(
    precision_policy="auto",
    max_workspace_size=2 << 30,
)
```

## Evaluation

`ArtifactManager` resolves ONNX/TensorRT paths in this order: registered export artifacts, then explicit evaluation paths, then fallbacks (`export.onnx_path`, etc.). For a typical post-export evaluation, set directories explicitly:

```python
_ONNX_DIR = f"{_WORK_DIR}/onnx"
_TENSORRT_DIR = f"{_WORK_DIR}/tensorrt"

evaluation = dict(
    enabled=True,
    num_samples=100,
    verbose=False,
    backends=dict(
        pytorch=dict(enabled=True, device=devices["cuda"]),
        onnx=dict(enabled=True, device=devices["cuda"], model_dir=_ONNX_DIR),
        tensorrt=dict(enabled=True, device=devices["cuda"], engine_dir=_TENSORRT_DIR),
    ),
)
```

Optional `evaluation.models` / `evaluation.devices` exist in the typed schema for advanced overrides.

## Verification

Scenarios are grouped by **export mode** (`both`, `onnx`, `trt`, `none`), matching `deployment.configs.enums.ExportMode`. Only scenarios for the active export mode run.

```python
verification = dict(
    enabled=True,
    tolerance=0.1,
    num_verify_samples=3,
    devices=devices,
    scenarios=dict(
        both=[
            dict(ref_backend="pytorch", ref_device="cpu", test_backend="onnx", test_device="cpu"),
            dict(ref_backend="onnx", ref_device="cuda", test_backend="tensorrt", test_device="cuda"),
        ],
        onnx=[
            dict(ref_backend="pytorch", ref_device="cpu", test_backend="onnx", test_device="cpu"),
        ],
        trt=[
            dict(ref_backend="onnx", ref_device="cuda", test_backend="tensorrt", test_device="cuda"),
        ],
        none=[],
    ),
)
```

## Device aliases

Keep a single top-level `devices` dict and reference it from `evaluation.backends` and `verification` so CUDA/CPU strings stay consistent.

## Backend enum

```python
from deployment.core import Backend

evaluation = dict(
    backends={
        Backend.PYTORCH.value: {"enabled": True, "device": devices["cpu"]},
        Backend.ONNX.value: {"enabled": True, "device": devices["cpu"], "model_dir": _ONNX_DIR},
        Backend.TENSORRT.value: {"enabled": True, "device": devices["cuda"], "engine_dir": _TENSORRT_DIR},
    }
)
```

## Typed exporter configs

Low-level typed classes live in `deployment.exporters.common.configs`. `BaseDeploymentConfig.get_onnx_settings(component_name)` / `get_tensorrt_settings(component_name)` merge shared `onnx_config` / `tensorrt_config` with each `components` entry.

## Example on disk

- `deployment/projects/centerpoint/config/deploy_config.py` — full multi-component deploy config (with comments on tolerance and verification).
