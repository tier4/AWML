# Deployment runbook

Practical guide for running deployment end to end: CLI syntax, required inputs, export modes, and what the framework does during a run.

For deploy config keys and copy-paste examples, use [configuration.md](./configuration.md). For internal structure and extension rules, use [architecture.md](./architecture.md).

## Quick start

```bash
python -m deployment.cli.main centerpoint \
    <deploy_cfg.py> \
    <model_cfg.py> \
    [--log-level INFO]

# CenterPoint-specific flag
python -m deployment.cli.main centerpoint \
    <deploy_cfg.py> \
    <model_cfg.py> \
    --rot-y-axis-reference
```

## Required inputs

- `project_name` must match a registered `ProjectAdapter` under `deployment/projects/`.
- `deploy_cfg.py` is the deployment config loaded into `BaseDeploymentConfig`.
- `model_cfg.py` is the project model or training config used by the project entrypoint.

## What happens during a run

Every project follows the same high-level flow:

```text
Load checkpoint -> Export -> Verify -> Evaluate
```

More concretely:

1. `deployment.cli.main` resolves the project adapter and parses shared plus project-specific flags.
2. The project `entrypoint.py` loads configs, builds the data loader, evaluator, and project runner.
3. `BaseDeploymentRunner` executes export, verification, and evaluation in sequence.
4. Evaluators build backend-specific inference pipelines through `PipelineFactory`.

## Export modes

| `export.mode` | What runs | Typical use |
| --- | --- | --- |
| `both` | ONNX export, TensorRT export, then verification/evaluation | Full deployment quality gate |
| `onnx` | ONNX export, then ONNX-relevant verification/evaluation | Validate export before TRT |
| `trt` | TensorRT build from existing ONNX, then TRT-relevant verification/evaluation | Rebuild engines from a known ONNX layout |
| `none` | Skip export and only verify/evaluate against existing artifacts | Re-run checks on saved artifacts |

## Verification and evaluation

Verification and evaluation are part of the same deployment run, but they answer different questions.

- Verification asks whether backend outputs still match within tolerance.
- Evaluation asks how each backend performs on task metrics and latency.

### Verification

Verification uses scenario lists grouped by export mode. Only the scenarios for the active `export.mode` run.

Typical flow:

1. Build reference and test pipelines for the chosen backends.
2. Run paired inference on shared samples.
3. Compare nested outputs with `verification.tolerance`.
4. Report pass/fail statistics.

Use verification when you need to catch numerical or tensor-shape regressions before trusting exported artifacts.

### Evaluation

Evaluation runs the configured backends and reports task metrics plus latency summaries.

- CenterPoint uses 3D detection metrics through the shared evaluator interfaces.
- ONNX and TensorRT artifacts are resolved from the current export run or from explicit directories in `evaluation.backends`.

If you run with `export.mode="none"`, set `model_dir` or `engine_dir` explicitly so the evaluator can resolve artifacts without relying on in-process export state.

## Export mental model

The framework exports one or more model components defined in the `components` section of the deploy config.

- Single-component projects export one ONNX file and optionally one TensorRT engine.
- Multi-component projects such as CenterPoint export several artifacts, one per component.

For each component:

- `onnx_file` and `io` drive ONNX export.
- `engine_file` and `tensorrt_profile` drive TensorRT engine build.
- Shared `onnx_config` and `tensorrt_config` are merged with per-component settings.

## Logging

- `--log-level` controls the root logging level.
- `deploy_log_path` mirrors deployment logs to a file, usually under `export.work_dir` when given as a relative path.
- Evaluators should log summaries through the `logging` module so console and file output stay aligned.

## Where to go next

- [configuration.md](./configuration.md) for deploy config fields and examples
- [architecture.md](./architecture.md) for CLI, runner, orchestrators, pipelines, and extension boundaries
- [operations.md](./operations.md) for troubleshooting and practical advice
- [../projects/centerpoint/README.md](../projects/centerpoint/README.md) for the current project-specific guide
