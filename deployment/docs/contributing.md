# Contributing to deployment

Use this page when adding a new project bundle or changing shared deployment infrastructure.

Before changing shared runners, evaluators, `BackendExecutor`, metrics interfaces, or orchestrators, read [architecture.md](./architecture.md). It contains the framework structure and extension contract.

## Minimal project checklist

Each directory below mirrors the framework module that owns its base class — see
the [project layout contract](./architecture.md#project-layout-contract) for the
full project-path → framework-module → base-class mapping.

1. Create `deployment/projects/<project>/__init__.py` and register a `ProjectAdapter`.
2. Add `entrypoint.py` to build `BaseDeploymentConfig`, the data loader, evaluator, and runner.
3. Add `cli.py` exposing `add_args(parser)` for project flags (may be a no-op).
4. Add `runner.py` as a thin `BaseDeploymentRunner` subclass.
5. Add `configs/deploy_config.py` with the required deploy config sections described in [configuration.md](./configuration.md).
6. Add `io/` and `evaluation/` for project-specific loading and evaluation logic (mirroring framework `io/` and `evaluation/`).
7. In `entrypoint.py`, build the task metrics config from the shared `metrics/` (e.g. `extract_t4metric_v2_config` for 3D detection) and pass it to the evaluator — do **not** add a project `metrics/` directory.
8. Add `inference/` with backend-specific inference pipelines (files `*_inference_pipeline.py`), and create them from the project's `BackendExecutor.create_pipeline`.
9. Add a project `README.md` with the project-specific quick start and links back to shared docs.

Add `contexts.py` only when the project needs extra `ExportContext` fields, and
`export/` only when the project needs multi-stage or multi-file export orchestration.

## Implementation notes

### Evaluator, executor, and data loader

- Subclass `BaseEvaluator` with task-specific metrics and output parsing.
- Subclass `BackendExecutor` for pipeline creation, input preparation, and (optionally) `get_output_names()` to label raw outputs during verification.
- Subclass `BaseDataLoader` for project dataset and preprocessing needs.
- Metrics are shared, not per-project: reuse the task config/interface and extractor in `metrics/` (e.g. `Detection3DMetricsConfig` + `extract_t4metric_v2_config`). Build the config in `entrypoint.py` and keep metric computation inside `metrics/` interfaces, not inside pipelines. Add a new `metrics/<task>_metrics.py` only for a genuinely new task.

### Runner

- Project runners should focus on project model loading, wrappers, and optional export pipeline wiring.
- Keep export sequencing in the shared runtime instead of reimplementing it per project.

### Inference pipelines

- Add backend-specific pipelines under `deployment/projects/<project>/inference/` (files `*_inference_pipeline.py`).
- Construct them in the project's `BackendExecutor.create_pipeline` (one branch per backend); override `get_supported_backends` to restrict which backends the project allows.
- Use `components_cfg` from `BaseDeploymentConfig` instead of raw config dicts where possible.

### CLI

- The shared entrypoint remains `python -m deployment.cli.main <project> <deploy_cfg.py> <model_cfg.py>`.
- Project-specific flags should be added through the project adapter, not by editing the shared CLI for one project.

### Documentation

- Keep `deployment/README.md` short and user-facing.
- Put shared behavior in shared docs and project-specific details in the project README.
