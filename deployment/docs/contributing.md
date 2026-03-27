# Contributing to deployment

## Adding a new project

1. **Evaluator and data loader**
   - Subclass `BaseEvaluator` with task-specific metrics and `PipelineFactory.create("<your_project>", ...)`.
   - Subclass `BaseDataLoader` for your dataset and collate/preprocess needs.

2. **Project bundle** under `deployment/projects/<project>/`
   - `entrypoint.py` — argparse wiring (or delegate to `cli.py`), `setup_logging`, `BaseDeploymentConfig`, build loader/evaluator/runner, `runner.run(context=...)`.
   - `runner.py` — subclass `BaseDeploymentRunner`; wire `ExporterFactory`, wrappers, optional export pipelines.
   - `config/deploy_config.py` — deploy config (must include required sections: `export`, `components`, `checkpoint_path`, etc.).
   - `io/` — data loader, model load helpers, sample adapters as needed.
   - `eval/` — evaluator and metric helpers (optional subpackage, e.g. `eval/evaluator.py`).
   - `__init__.py` — register `ProjectAdapter(name=..., add_args=..., run=...)` with `deployment.projects.registry.project_registry`.

3. **Inference pipelines**
   - Add `pipelines/` with backend-specific classes inheriting `BaseInferencePipeline`.
   - Register a `BasePipelineFactory` subclass with `@pipeline_registry.register` in `deployment/pipelines/registry.py` (see `deployment/projects/centerpoint/pipelines/factory.py`).

4. **Export pipelines (optional)**
   - For multi-stage or multi-file export, add modules under `deployment/projects/<project>/export/` and pass instances into your runner’s `super().__init__(..., onnx_pipeline=..., tensorrt_pipeline=...)`.

5. **CLI**
   - The unified entrypoint is `python -m deployment.cli.main <project> <deploy_cfg.py> <model_cfg.py> ...`. Project-specific flags are added via `adapter.add_args`.

6. **Documentation**
   - Update `deployment/README.md`, `deployment/docs/projects.md`, and this file when behavior or layout changes.

## Core contract

Read `deployment/docs/core_contract.md` before changing shared runners, evaluators, `PipelineFactory`, or metrics interfaces.
