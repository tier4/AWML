# Contributing to Deployment

## Adding a New Project

1. **Evaluator & Data Loader**
   - Implement `BaseEvaluator` with task-specific metrics.
   - Implement `BaseDataLoader` variant for the dataset(s).

2. **Project Bundle**
   - Create a new bundle under `deployment/projects/<project>/`.
   - Put **all project deployment code** in one place: `runner.py`, `evaluator.py`, `data_loader.py`, `config/deploy_config.py`.

3. **Pipelines**
   - Add backend-specific pipelines under `deployment/projects/<project>/pipelines/` and register a factory into `deployment.pipelines.registry.pipeline_registry`.

4. **Export Pipelines (optional)**
   - If the project needs multi-stage export, implement under `deployment/projects/<project>/export/` (compose the generic exporters in `deployment/exporters/common/`).

5. **CLI wiring**
   - Register a `ProjectAdapter` in `deployment/projects/<project>/__init__.py`.
   - The unified entry point is `python -m deployment.cli.main <project> <deploy_cfg.py> <model_cfg.py> ...`

6. **Documentation**
   - Update `deployment/README.md` and the relevant docs in `deployment/docs/`.
   - Document special requirements, configuration flags, or export pipelines.

## Core Contract

Before touching shared components, review `deployment/docs/core_contract.md` to understand allowed dependencies between runners, evaluators, pipelines, and exporters. Adhering to the contract keeps refactors safe and ensures new logic lands in the correct layer.
