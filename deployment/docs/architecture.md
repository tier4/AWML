# Deployment Architecture

## High-Level Workflow

```
┌─────────────────────────────────────────────────────────┐
│              Project Entry Points                       │
│  (projects/*/deploy/main.py)                            │
│  - CenterPoint, YOLOX-ELAN, Calibration                 │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│ BaseDeploymentRunner + Project Runners                  │
│  - Coordinates load → export → verify → evaluate        │
│  - Delegates to helper orchestrators                    │
│  - Projects extend the base runner for custom logic     │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┴────────────┐
        │                       │
┌───────▼────────┐     ┌────────▼───────────────┐
│   Exporters    │     │  Helper Orchestrators │
│  - ONNX / TRT  │     │  - ArtifactManager     │
│  - Wrappers    │     │  - VerificationOrch.   │
│  - Workflows   │     │  - EvaluationOrch.     │
└────────────────┘     └────────┬───────────────┘
                                │
┌───────────────────────────────▼─────────────────────────┐
│                    Evaluators & Pipelines               │
│  - BaseDeploymentPipeline + task-specific variants      │
│  - Backend-specific implementations (PyTorch/ONNX/TRT)  │
└────────────────────────────────────────────────────────┘
```

## Core Components

### BaseDeploymentRunner & Project Runners

`BaseDeploymentRunner` orchestrates the export/verification/evaluation loop. Project runners (CenterPoint, YOLOX, Calibration, …):

- Implement model loading.
- Inject wrapper classes and optional workflows.
- Reuse `ExporterFactory` to lazily create ONNX/TensorRT exporters.
- Delegate artifact registration plus verification/evaluation to the shared orchestrators.

### Core Package (`deployment/core/`)

- `BaseDeploymentConfig` – typed deployment configuration container.
- `Backend` – enum guaranteeing backend name consistency.
- `Artifact` – dataclass describing exported artifacts.
- `VerificationMixin` – recursive comparer for nested outputs.
- `BaseEvaluator` – task-specific evaluation contract.
- `BaseDataLoader` – data-loading abstraction.
- `build_preprocessing_pipeline` – extracts preprocessing steps from MMDet/MMDet3D configs.
- Typed value objects (`constants.py`, `runtime_config.py`, `task_config.py`, `results.py`) keep configuration and metrics structured.

### Exporters & Workflows

- `exporters/common/` hosts the base exporters, typed config objects, and `ExporterFactory`.
- Project wrappers live in `exporters/{project}/model_wrappers.py`.
- Complex projects add workflows (e.g., `CenterPointONNXExportWorkflow`) that orchestrate multi-file exports by composing the base exporters.

### Pipelines

`BaseDeploymentPipeline` defines `preprocess → run_model → postprocess`, while `PipelineFactory` builds backend-specific implementations for each task (`Detection2D`, `Detection3D`, `Classification`). Pipelines are encapsulated per backend (PyTorch/ONNX/TensorRT) under `deployment/pipelines/{task}/`.

### File Structure Snapshot

```
deployment/
├── core/                 # Core dataclasses, configs, evaluators
├── exporters/            # Base exporters + project wrappers/workflows
├── pipelines/            # Task-specific pipelines per backend
├── runners/              # Shared runner + project adapters
```

Project entry points follow the same pattern under `projects/*/deploy/` with `main.py`, `data_loader.py`, `evaluator.py`, and `configs/deploy_config.py`.
