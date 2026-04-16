# AWML Deployment Framework

The AWML Deployment Framework is a **task-agnostic** path from trained PyTorch checkpoints to ONNX and TensorRT artifacts, with **verification** and **evaluation** wired into the same run. The core runtime, exporters, and orchestration are shared across projects; each task ships as a **project bundle** under `deployment/projects/`.

**Today:** CenterPoint is the **full reference implementation** (multi-component export, 3D metrics, pipelines). See [Project status](#project-status) for more infomation.

**Why it exists**

- Unify export, verification, and evaluation across tasks and backends (PyTorch, ONNX, TensorRT).
- Reduce duplicated deployment logic via shared runners, orchestrators, and typed config.
- Make new task integration predictable: same entrypoint, registry pattern, and [core contract](docs/core_contract.md) for extensions.

**Core workflow**

Load checkpoint → Export → Verify → Evaluate

**Quick start**

```bash
python -m deployment.cli.main <project_name> <deploy_cfg.py> <model_cfg.py> [--log-level INFO]

# Example (CenterPoint)
python -m deployment.cli.main centerpoint <deploy_cfg.py> <model_cfg.py> [--rot-y-axis-reference]
```

**Project status**

| Status | Projects |
| --- | --- |
| **Full reference implementation** | [CenterPoint](docs/projects.md#centerpoint-3d-detection) |
| **Planned / not yet first-class in this tree** | YOLOX (and similar) — follow the same bundle pattern when added |
| **Legacy references** | Calibration status classification — historical scripts may live outside `deployment/projects/` |

**Documentation guide (by what you want to do)**

| Goal | Start here |
| --- | --- |
| New to the repo — what is this and how do I orient? | [docs/getting_started.md](docs/getting_started.md), then [docs/overview.md](docs/overview.md) |
| Run an existing project (commands, modes, logging) | [docs/usage.md](docs/usage.md), project README under `deployment/projects/<name>/` |
| Edit or author deploy config (authoritative keys and examples) | [docs/configuration.md](docs/configuration.md) |
| Understand how the pieces connect (CLI → runner → exporters → evaluators) | [docs/architecture.md](docs/architecture.md) |
| Export mental model and artifact flow | [docs/export_pipeline.md](docs/export_pipeline.md) |
| Verification / evaluation behavior and reporting | [docs/verification_evaluation.md](docs/verification_evaluation.md) |
| Add a new project bundle | [docs/contributing.md](docs/contributing.md) — read [docs/core_contract.md](docs/core_contract.md) first |
| Operational tips and troubleshooting | [docs/best_practices.md](docs/best_practices.md) |

Full doc map: [deployment/docs/README.md](docs/README.md).

**Repository layout (high level)**

```
deployment/
├── cli/           # Unified CLI
├── configs/       # Typed deploy config schema
├── core/          # Shared types, evaluators, verification mixins
├── exporters/     # ONNX / TensorRT exporters and export pipeline bases
├── pipelines/     # Inference pipelines and global factory
├── runtime/       # BaseDeploymentRunner, orchestrators, ArtifactManager
└── projects/      # Per-task bundles (entrypoint, runner, config, io, eval, …)
```

Export and verification **details** live in the docs linked above, not in this README.

## License

See LICENSE at the repository root.
