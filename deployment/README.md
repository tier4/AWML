# AWML Deployment Framework

AWML ships a unified, task-agnostic deployment stack that turns trained PyTorch checkpoints into production-ready ONNX and TensorRT artifacts. The verification and evaluation toolchain runs across every backend, ensuring numerical parity and consistent metrics across different projects.

At the center is a shared runner/pipeline/exporter architecture that teams can extend with lightweight wrappers or workflows. CenterPoint, YOLOX, CalibrationStatusClassification, and future models plug into the same export and verification flow while still layering in task-specific logic where needed.


## Quick Start

```bash
# CenterPoint deployment
python projects/CenterPoint/deploy/main.py configs/deploy_config.py configs/model_config.py

# YOLOX deployment
python projects/YOLOX_opt_elan/deploy/main.py configs/deploy_config.py configs/model_config.py

# Calibration deployment
python projects/CalibrationStatusClassification/deploy/main.py configs/deploy_config.py configs/model_config.py
```

Only `--log-level` is available as a command-line flag. All other settings (`work_dir`, `device`, `checkpoint_path`) are configured in the deploy config file. Inject wrapper classes and optional workflows when instantiating a runner; exporters are created lazily inside `BaseDeploymentRunner`.

## Documentation Map

| Topic | Description |
| --- | --- |
| [`docs/overview.md`](docs/overview.md) | Design principles, key features, precision policies. |
| [`docs/architecture.md`](docs/architecture.md) | Workflow diagram, core components, file layout. |
| [`docs/usage.md`](docs/usage.md) | CLI usage, runner patterns, typed contexts, export modes. |
| [`docs/configuration.md`](docs/configuration.md) | Config structure, typed schemas, backend enums. |
| [`docs/projects.md`](docs/projects.md) | CenterPoint, YOLOX, and Calibration deployment specifics. |
| [`docs/export_workflow.md`](docs/export_workflow.md) | ONNX/TRT export steps and workflow patterns. |
| [`docs/verification_evaluation.md`](docs/verification_evaluation.md) | Verification scenarios, evaluation metrics, core contract. |
| [`docs/best_practices.md`](docs/best_practices.md) | Best practices, troubleshooting, roadmap. |
| [`docs/contributing.md`](docs/contributing.md) | How to add new deployment projects end-to-end. |

Refer to `deployment/docs/README.md` for the same index.

## Architecture Snapshot

- **Entry points** (`projects/*/deploy/main.py`) instantiate project runners with data loaders, evaluators, wrappers, and optional workflows.
- **Runners** coordinate load → export → verify → evaluate while delegating to shared Artifact/Verification/Evaluation orchestrators.
- **Exporters** live under `exporters/common/` with typed config classes; project wrappers/workflows compose the base exporters as needed.
- **Pipelines** (`pipelines/common/*`, `pipelines/{task}/`) provide consistent preprocessing/postprocessing with backend-specific inference implementations resolved via `PipelineFactory`.
- **Core package** (`core/`) supplies typed configs, runtime contexts, task definitions, and shared verification utilities.

See [`docs/architecture.md`](docs/architecture.md) for diagrams and component details.

## Export & Verification Flow

1. Load the PyTorch checkpoint and run ONNX export (single or multi-file) using the injected wrappers/workflows.
2. Optionally build TensorRT engines with precision policies such as `auto`, `fp16`, `fp32_tf32`, or `strongly_typed`.
3. Register artifacts via `ArtifactManager` for downstream verification and evaluation.
4. Run verification scenarios defined in config—pipelines are resolved by backend and device, and outputs are recursively compared with typed tolerances.
5. Execute evaluation across enabled backends and emit typed metrics.

Implementation details live in [`docs/export_workflow.md`](docs/export_workflow.md) and [`docs/verification_evaluation.md`](docs/verification_evaluation.md).

## Project Coverage

- **CenterPoint** – multi-file export orchestrated by dedicated ONNX/TRT workflows; see [`docs/projects.md`](docs/projects.md).
- **YOLOX** – single-file export with output reshaping via `YOLOXOptElanONNXWrapper`.
- **CalibrationStatusClassification** – binary classification deployment with identity wrappers and simplified pipelines.

Each project ships its own `deploy_config.py`, evaluator, and data loader under `projects/{Project}/deploy/`.

## Core Contract

[`core_contract.md`](docs/core_contract.md) defines the boundaries between runners, orchestrators, evaluators, pipelines, and metrics interfaces. Follow the contract when introducing new logic to keep refactors safe and dependencies explicit.

## Contributing & Best Practices

- Start with [`docs/contributing.md`](docs/contributing.md) for the required files and patterns when adding a new deployment project.
- Consult [`docs/best_practices.md`](docs/best_practices.md) for export patterns, troubleshooting tips, and roadmap items.
- Keep documentation for project-specific quirks in the appropriate file under `deployment/docs/`.

## License

See LICENSE at the repository root.
