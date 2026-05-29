# AWML Deployment Framework

The `deployment/` package is the shared path from trained PyTorch checkpoints to ONNX and TensorRT artifacts, with verification and evaluation built into the same run. Shared runtime code lives in the framework packages, while task-specific logic lives under `deployment/projects/<name>/`.

## Core workflow

```text
Load checkpoint -> Export -> Verify -> Evaluate
```

## Quick start

```bash
python -m deployment.cli.main <project_name> <deploy_cfg.py> <model_cfg.py> [--log-level INFO]

# Example (CenterPoint)
python -m deployment.cli.main centerpoint \
    deployment/projects/centerpoint/config/deploy_config.py \
    <model_cfg.py> \
    --rot-y-axis-reference
```

## What to read

| If you want to... | Start here |
| --- | --- |
| Run deployment today | [docs/runbook.md](docs/runbook.md) |
| Edit or author deploy config | [docs/configuration.md](docs/configuration.md) |
| Understand the framework structure | [docs/architecture.md](docs/architecture.md) |
| Troubleshoot or tune runs | [docs/operations.md](docs/operations.md) |
| Add a new project bundle | [docs/contributing.md](docs/contributing.md) |
| Run the current shipped project | [projects/centerpoint/README.md](projects/centerpoint/README.md) |

## Current status

- Current first-class project: [CenterPoint](projects/centerpoint/README.md)
- Shared framework responsibilities: CLI, typed config, exporters, runtime orchestration, verification, evaluation, and pipeline creation

## Repository layout

```text
deployment/
├── cli/           # Unified CLI
├── configs/       # Typed deploy config schema
├── core/          # Shared types, evaluators, verification mixins
├── exporters/     # ONNX / TensorRT exporters and export pipeline bases
├── pipelines/     # Inference pipelines and global factory
├── runtime/       # BaseDeploymentRunner, orchestrators, ArtifactManager
└── projects/      # Per-task bundles
```

## License

See `LICENSE` at the repository root.
