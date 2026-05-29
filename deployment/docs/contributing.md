# Contributing to deployment

Use this page when adding a new project bundle or changing shared deployment infrastructure.

Before changing shared runners, evaluators, `PipelineFactory`, metrics interfaces, or orchestrators, read [architecture.md](./architecture.md). It contains the framework structure and extension contract.

## Minimal project checklist

1. Create `deployment/projects/<project>/__init__.py` and register a `ProjectAdapter`.
2. Add `entrypoint.py` to build `BaseDeploymentConfig`, the data loader, evaluator, and runner.
3. Add `runner.py` as a thin `BaseDeploymentRunner` subclass.
4. Add `config/deploy_config.py` with the required deploy config sections described in [configuration.md](./configuration.md).
5. Add `io/` and `eval/` for project-specific loading and evaluation logic.
6. Add `pipelines/` with backend-specific inference pipelines and register a project pipeline factory.
7. Add a project `README.md` with the project-specific quick start and links back to shared docs.

Add `export/` only when the project needs multi-stage or multi-file export orchestration.

## Implementation notes

### Evaluator and data loader

- Subclass `BaseEvaluator` with task-specific metrics and output parsing.
- Subclass `BaseDataLoader` for project dataset and preprocessing needs.
- Keep metrics inside evaluators and metrics interfaces, not inside pipelines.

### Runner

- Project runners should focus on project model loading, wrappers, and optional export pipeline wiring.
- Keep export sequencing in the shared runtime instead of reimplementing it per project.

### Inference pipelines

- Add backend-specific pipelines under `deployment/projects/<project>/pipelines/`.
- Register a project `BasePipelineFactory` subclass with `@pipeline_registry.register`.
- Use `components_cfg` from `BaseDeploymentConfig` instead of raw config dicts where possible.

### CLI

- The shared entrypoint remains `python -m deployment.cli.main <project> <deploy_cfg.py> <model_cfg.py>`.
- Project-specific flags should be added through the project adapter, not by editing the shared CLI for one project.

### Documentation

- Keep `deployment/README.md` short and user-facing.
- Put shared behavior in shared docs and project-specific details in the project README.
