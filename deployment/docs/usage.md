# Usage and entry points

## Commands

The only supported entrypoint is the unified CLI (subcommand = project name):

```bash
python -m deployment.cli.main centerpoint \
    <deploy_cfg.py> \
    <model_cfg.py> \
    [--log-level DEBUG|INFO|WARNING|ERROR|CRITICAL]

# CenterPoint-specific flag (see deployment/projects/centerpoint/cli.py)
python -m deployment.cli.main centerpoint \
    <deploy_cfg.py> \
    <model_cfg.py> \
    --rot-y-axis-reference
```

Arguments `deploy_cfg` and `model_cfg` are passed through `parse_base_args` in `deployment/cli/args.py`.

## Logging

- `--log-level` sets the root logging level (default `INFO`).
- If `deploy_log_path` is set in the deploy config (non-empty), the CenterPoint entrypoint resolves it (relative paths under `export.work_dir`) and attaches a UTF-8 file handler to the **root** logger so deployment and library logs can be captured in one file. See `deployment/cli.args.add_deployment_file_logging`.

## Project runner pattern

Project bundles under `deployment/projects/<project>/` own:

- `entrypoint.py` — load MMEngine configs, build `BaseDeploymentConfig`, data loader, evaluator, runner, then `runner.run(context=...)`.
- A thin `runner.py` subclass of `BaseDeploymentRunner` for model load and exporter/export-pipeline wiring.

Exporters are created lazily inside `BaseDeploymentRunner` / export orchestrator via `ExporterFactory`, not constructed in the CLI.

## Typed context objects

Export orchestration receives a small frozen context (sample index, project flags). CenterPoint uses `CenterPointExportContext` from CLI args:

```python
from deployment.core.contexts import ExportContext, CenterPointExportContext, YOLOXExportContext, CalibrationExportContext
```

| Context | Notes |
| --- | --- |
| `ExportContext` | `sample_idx`, optional `extra` map. |
| `CenterPointExportContext` | Adds `rot_y_axis_reference`. |
| `YOLOXExportContext` | Extends base; optional `model_cfg` string when that project is wired. |
| `CalibrationExportContext` | Placeholder for calibration-specific fields. |

Subclasses of `ExportContext` stay dataclasses/frozen for predictable wiring from `runner.run(context=...)`.

## Export modes (`export.mode`)

### ONNX only

```python
export = dict(mode="onnx", work_dir="work_dirs/deployment", onnx_path=None)
```

### TensorRT from existing ONNX

Point `onnx_path` at the ONNX **directory** (or path your export pipeline expects), consistent with `ExportConfig` and `ArtifactManager`.

```python
export = dict(
    mode="trt",
    work_dir="work_dirs/deployment",
    onnx_path="work_dirs/deployment/onnx",
)
```

### Full pipeline

```python
export = dict(mode="both", work_dir="work_dirs/deployment", onnx_path=f"work_dirs/deployment/onnx")
```

### Evaluation-only (no export)

```python
export = dict(mode="none", work_dir="work_dirs/deployment")
```

You still need a valid `checkpoint_path` and, for ONNX/TRT evaluation, resolvable artifact paths (registered during a previous run or via `evaluation.backends`).
