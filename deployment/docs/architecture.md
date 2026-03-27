# Deployment Architecture

## High-level workflow

```
┌─────────────────────────────────────────────────────────┐
│  CLI: deployment/cli/main.py                          │
│  Discovers deployment.projects.<name>, subcommands      │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  Project bundle: deployment/projects/<project>/        │
│  entrypoint.py → builds config, loader, evaluator,     │
│  CenterPointDeploymentRunner (or future project runner) │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  BaseDeploymentRunner (deployment/runtime/runner.py)     │
│  load → export → verify → evaluate                      │
└────────────────────────┬────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
┌───────▼────────┐               ┌────────▼────────────────┐
│ Export stack   │               │ Orchestrators            │
│ exporters/*    │               │ ArtifactManager          │
│ export_pipelines│              │ Export / Verification /  │
│                │               │ Evaluation orchestrators │
└────────────────┘               └────────┬────────────────┘
                                          │
┌─────────────────────────────────────────▼────────────────┐
│  Evaluators & inference pipelines                          │
│  BaseEvaluator + PipelineFactory → project pipelines       │
└──────────────────────────────────────────────────────────┘
```

## Core components

### CLI and project bundles

- `deployment/cli/main.py` discovers packages under `deployment/projects/`, imports them so each package registers a `ProjectAdapter` in `deployment.projects.registry.project_registry`, then dispatches to `adapter.run(args)`.
- Each project lives under `deployment/projects/<project>/` with `entrypoint.py`, `runner.py`, `config/deploy_config.py`, task-specific `io/`, `eval/`, optional `export/` and `pipelines/`.

### BaseDeploymentRunner

Defined in `deployment/runtime/runner.py`. Owns the shared sequence (export, verification, evaluation), constructs `ExporterFactory` outputs, and delegates to `ExportOrchestrator`, `VerificationOrchestrator`, and `EvaluationOrchestrator`. Project runners (e.g. `CenterPointDeploymentRunner`) subclass it to plug in model loading, sample adapters, and export pipelines.

### Core package (`deployment/core/`)

- `BaseDeploymentConfig` (`configs/base.py`) — validated view of deploy config; exposes `components_cfg`, `resolved_deploy_log_file`, etc.
- `Backend`, `DeviceSpec`, `Artifact`
- `VerificationMixin`, `BaseEvaluator`, `BaseDataLoader`
- Metrics interfaces and `EvalResultDict` types

### Exporters and export pipelines

- `deployment/exporters/common/` — base ONNX/TensorRT exporters, `ExporterFactory`, shared wrappers.
- `deployment/exporters/export_pipelines/` — base pipeline interfaces.
- Complex projects add orchestration under `deployment/projects/<project>/export/` (e.g. CenterPoint multi-file export).

### Inference pipelines

- Shared abstractions: `deployment/pipelines/base_pipeline.py`, `deployment/pipelines/base_factory.py`, `deployment/pipelines/registry.py`, `deployment/pipelines/factory.py`.
- Each project registers a `BasePipelineFactory` subclass (e.g. `CenterPointPipelineFactory` in `deployment/projects/centerpoint/pipelines/factory.py`) with `@pipeline_registry.register`. Evaluators call `PipelineFactory.create(project_name, ...)` with `components_cfg` from deploy config.

### Runtime helpers

- `deployment/runtime/artifact_manager.py` — register and resolve ONNX/engine paths for evaluation and verification.
- Orchestrators in `deployment/runtime/*_orchestrator.py` keep runner code thin.

## Directory layout (snapshot)

```
deployment/
├── cli/                 # Unified CLI (main.py, args / logging helpers)
├── configs/             # BaseDeploymentConfig + schema dataclasses
├── core/                # Backend, device, evaluators, metrics, contexts
├── exporters/           # Shared exporters + export pipeline bases
├── pipelines/           # Base pipeline + global PipelineFactory / registry
├── runtime/             # BaseDeploymentRunner + orchestrators + ArtifactManager
└── projects/
    └── <project>/       # entrypoint, runner, config/, io/, eval/, pipelines/, export/
```

The older layout `projects/*/deploy/main.py` is **not** used; the supported pattern is `projects/<project>/entrypoint.py` plus CLI registration in `projects/<project>/__init__.py`.
