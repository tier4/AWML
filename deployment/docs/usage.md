# Usage and entry points

Practical guide: **commands**, **inputs**, **run modes**, and **what happens during a run**. Deploy config **schema and examples** live in [configuration.md](./configuration.md); do not treat this page as the config source of truth.

## Supported CLI syntax

The supported entrypoint is the unified CLI (subcommand = project name):

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

## Required inputs

- **Project name** — must match a registered `ProjectAdapter` (see `deployment/projects/<name>/__init__.py`).
- **`deploy_cfg.py`** — MMEngine-style deploy config loaded into `BaseDeploymentConfig` (keys in [configuration.md](./configuration.md)).
- **`model_cfg.py`** — model / training config used by the project entrypoint (for example MMDet3D config for CenterPoint).

## Most common workflows

| Goal | Typical `export.mode` | Notes |
| --- | --- | --- |
| Full quality gate: export ONNX and TRT, then verify and evaluate | `both` | Default path when you have checkpoints and want artifacts plus metrics. |
| ONNX only (no TRT build yet) | `onnx` | Verification scenarios keyed to `onnx` run; TRT stages skipped. |
| TRT only from an existing ONNX tree | `trt` | Point `export.onnx_path` at the ONNX **layout** your pipeline expects; see [configuration.md](./configuration.md). |
| Evaluation (and optional verification) on existing artifacts | `none` | Ensure `evaluation.backends` (and verification, if enabled) can resolve ONNX/TRT paths — often explicit `model_dir` / `engine_dir`. |

## Logging

- `--log-level` sets the root logging level (default `INFO`).
- If `deploy_log_path` is set in the deploy config (non-empty), the CenterPoint entrypoint resolves it (relative paths under `export.work_dir`) and attaches a UTF-8 file handler to the **root** logger so deployment and library logs can be captured in one file. See `deployment/cli.args.add_deployment_file_logging`.

**Field semantics:** [configuration.md](./configuration.md#logging-deploy_log_path).

## Project runner pattern

Project bundles under `deployment/projects/<project>/` own:

- **`entrypoint.py`** — load MMEngine configs, build `BaseDeploymentConfig`, data loader, evaluator, runner, then `runner.run(context=...)`.
- **`runner.py`** — thin subclass of `BaseDeploymentRunner` for model load and exporter / export-pipeline wiring.

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

| Mode | Meaning |
| --- | --- |
| `onnx` | PyTorch → ONNX only. |
| `trt` | Build TensorRT from existing ONNX (see `export.onnx_path` in [configuration.md](./configuration.md)). |
| `both` | Full export pipeline. |
| `none` | Skip export; run evaluation when paths resolve. |

Copy-paste config fragments: [configuration.md](./configuration.md).

## What happens during a run

At a high level, `BaseDeploymentRunner` sequences **export** (when `export.mode` is not `none`), **verification** (when enabled and scenarios exist for the active mode), and **evaluation** (when enabled). Orchestrators coordinate exporters and `ArtifactManager`; evaluators construct pipelines via `PipelineFactory` using `components_cfg` from the typed deploy config. **Deeper diagram:** [architecture.md](./architecture.md).
