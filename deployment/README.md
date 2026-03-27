# AWML Deployment Framework

AWML ships a unified, task-agnostic deployment stack that turns trained PyTorch checkpoints into production-ready ONNX and TensorRT artifacts. Verification and evaluation run across backends where configured, with shared orchestration and typed configs.

The reference implementation is **CenterPoint** under `deployment/projects/centerpoint/`. Additional tasks can follow the same bundle layout and register with `deployment.projects.registry`.

## Quick start

```bash
python -m deployment.cli.main centerpoint <deploy_cfg.py> <model_cfg.py> [--log-level INFO]

# Optional CenterPoint flag
python -m deployment.cli.main centerpoint <deploy_cfg.py> <model_cfg.py> --rot-y-axis-reference
```

Deploy configs support optional **`deploy_log_path`** (default `deployment.log`, resolved under `export.work_dir` when relative). When set, logs are mirrored to that file via a root `FileHandler` in addition to stderr.

## Documentation map

| Topic | Description |
| --- | --- |
| [`docs/overview.md`](docs/overview.md) | Design principles, key features, precision policies. |
| [`docs/architecture.md`](docs/architecture.md) | CLI → project bundles → runtime runner and orchestrators. |
| [`docs/usage.md`](docs/usage.md) | CLI, logging, typed contexts, export modes. |
| [`docs/configuration.md`](docs/configuration.md) | Top-level keys, `components`, logging, evaluation/verification. |
| [`docs/projects.md`](docs/projects.md) | CenterPoint file layout; status of other tasks. |
| [`docs/export_pipeline.md`](docs/export_pipeline.md) | ONNX/TRT export and multi-component patterns. |
| [`docs/verification_evaluation.md`](docs/verification_evaluation.md) | Scenarios, metrics, artifact resolution. |
| [`docs/best_practices.md`](docs/best_practices.md) | Practices and troubleshooting. |
| [`docs/contributing.md`](docs/contributing.md) | Adding a new project bundle. |
| [`docs/core_contract.md`](docs/core_contract.md) | Layer boundaries for runners, evaluators, pipelines. |

The same index lives in [`deployment/docs/README.md`](docs/README.md).

## Architecture snapshot

- **`deployment/cli/main.py`** — Discovers `deployment.projects.*`, registers subcommands per project.
- **`deployment/runtime/runner.py`** — `BaseDeploymentRunner`: export → verify → evaluate.
- **`deployment/runtime/*_orchestrator.py`** — Export, verification, evaluation orchestration; **`ArtifactManager`** resolves ONNX/engine paths.
- **`deployment/exporters/common/`** — Base ONNX/TensorRT exporters and `ExporterFactory`.
- **`deployment/pipelines/`** — `BaseInferencePipeline`, global `PipelineFactory` and `pipeline_registry`.
- **`deployment/projects/<project>/`** — `entrypoint.py`, `runner.py`, `config/deploy_config.py`, task `io/` / `eval/` / `pipelines/` / `export/`.

See [`docs/architecture.md`](docs/architecture.md) for the full diagram and layout.

## Export and verification flow

1. Load checkpoint; export ONNX (per `components` entries) and optionally TensorRT with the configured precision policy.
2. Register artifacts for downstream steps.
3. Run verification scenarios for the current **export mode** (`both` / `onnx` / `trt` / `none`).
4. Run evaluation on enabled backends (`evaluation.backends`, optional `model_dir` / `engine_dir` for ONNX and TensorRT).

Details: [`docs/export_pipeline.md`](docs/export_pipeline.md), [`docs/verification_evaluation.md`](docs/verification_evaluation.md).

## Project coverage

- **CenterPoint** — Multi-component export and 3D metrics; see [`docs/projects.md`](docs/projects.md) and `projects/centerpoint/config/deploy_config.py`.
- **Other tasks** — Extend via new packages under `deployment/projects/` and `ProjectAdapter` registration (see contributing doc).

## Core contract

[`docs/core_contract.md`](docs/core_contract.md) defines how runners, evaluators, `PipelineFactory`, and metrics interact. Follow it when extending shared code.

## License

See LICENSE at the repository root.
