# Usage & Entry Points

## Basic Commands

```bash
# Single deployment entrypoint (project is a subcommand)
python -m deployment.cli.main centerpoint \
    <deploy_cfg.py> \
    <model_cfg.py>

# Example with CenterPoint-specific flag
python -m deployment.cli.main centerpoint \
    <deploy_cfg.py> \
    <model_cfg.py> \
    --rot-y-axis-reference
```

## Creating a Project Runner

Projects pass lightweight configuration objects (wrapper classes and optional export pipelines) into the runner. Exporters are created lazily via `ExporterFactory`.

```python
# Project bundles live under deployment/projects/<project> and are resolved by the CLI.
# The runtime layer is under deployment/runtime/*.
```

Key points:

- Pass wrapper classes (and optional export pipelines) instead of exporter instances.
- Exporters are constructed lazily inside `BaseDeploymentRunner`.
- Entry points remain explicit and easily testable.

## Typed Context Objects

Typed contexts carry parameters through the workflow, improving IDE discoverability and refactor safety.

```python
from deployment.core import ExportContext, YOLOXExportContext, CenterPointExportContext

results = runner.run(context=YOLOXExportContext(
    sample_idx=0,
    model_cfg_path="/path/to/config.py",
))
```

Available contexts:

- `ExportContext` – default context with `sample_idx` and `extra` dict.
- `YOLOXExportContext` – adds `model_cfg_path`.
- `CenterPointExportContext` – adds `rot_y_axis_reference`.
- `CalibrationExportContext` – calibration-specific options.

Create custom contexts by subclassing `ExportContext` and adding dataclass fields.

## Command-Line Arguments

```bash
python deploy/main.py \
    <deploy_cfg> \          # Deployment configuration file
    <model_cfg> \           # Model configuration file
    --log-level <level>     # Optional: DEBUG, INFO, WARNING, ERROR (default: INFO)
```

## Export Modes

### ONNX Only

```python
checkpoint_path = "model.pth"

export = dict(
    mode="onnx",
    work_dir="work_dirs/deployment",
)
```

### TensorRT From Existing ONNX

```python
export = dict(
    mode="trt",
    onnx_path="work_dirs/deployment/onnx/model.onnx",
    work_dir="work_dirs/deployment",
)
```

### Full Export Pipeline

```python
checkpoint_path = "model.pth"

export = dict(
    mode="both",
    work_dir="work_dirs/deployment",
)
```

### Evaluation-Only

```python
checkpoint_path = "model.pth"

export = dict(
    mode="none",
    work_dir="work_dirs/deployment",
)
```
