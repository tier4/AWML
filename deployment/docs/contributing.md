# Contributing to deployment

## Read this first

Before changing shared runners, evaluators, `PipelineFactory`, metrics interfaces, or orchestrators, read **[core_contract.md](./core_contract.md)**. It is the architecture contract for the framework.

## Minimal new project checklist

1. **`deployment/projects/<project>/__init__.py`** — register `ProjectAdapter(name=..., add_args=..., run=...)` with `deployment.projects.registry.project_registry`.
2. **`entrypoint.py`** — build `BaseDeploymentConfig`, loader, evaluator, runner; call `runner.run(context=...)`.
3. **`runner.py`** — subclass `BaseDeploymentRunner`; wire `ExporterFactory`, wrappers, optional ONNX/TRT export pipelines.
4. **`config/deploy_config.py`** — valid deploy dict / MMEngine config with required sections (`export`, `components`, `checkpoint_path`, …); see [configuration.md](./configuration.md).
5. **`io/`**, **`eval/`** — `BaseDataLoader`, `BaseEvaluator`, sample adapters as needed.
6. **`pipelines/`** — backend pipelines + `BasePipelineFactory` registered with `@pipeline_registry.register`.
7. **Project `README.md`** — quick start, flags, config pointers.

Optional: **`export/`** for multi-stage or multi-file export orchestration (CenterPoint pattern).

## Adding a new project (expanded)

1. **Evaluator and data loader**
   - Subclass `BaseEvaluator` with task-specific metrics and `PipelineFactory.create("<your_project>", ...)`.
   - Subclass `BaseDataLoader` for your dataset and collate/preprocess needs.

2. **Project bundle** under `deployment/projects/<project>/`
   - `entrypoint.py` — argparse wiring (or delegate to `cli.py`), `setup_logging`, `BaseDeploymentConfig`, build loader/evaluator/runner, `runner.run(context=...)`.
   - `runner.py` — subclass `BaseDeploymentRunner`; wire `ExporterFactory`, wrappers, optional export pipelines.
   - `config/deploy_config.py` — deploy config (required sections per [configuration.md](./configuration.md)).
   - `io/` — data loader, model load helpers, sample adapters as needed.
   - `eval/` — evaluator and metric helpers (optional subpackage, e.g. `eval/evaluator.py`).
   - `__init__.py` — register `ProjectAdapter` as above.

3. **Inference pipelines**
   - Add `pipelines/` with backend-specific classes inheriting `BaseInferencePipeline`.
   - Register a `BasePipelineFactory` subclass with `@pipeline_registry.register` (see `deployment/projects/centerpoint/pipelines/factory.py`).

4. **Export pipelines (optional)**
   - For multi-stage or multi-file export, add modules under `deployment/projects/<project>/export/` and pass instances into your runner’s `super().__init__(..., onnx_pipeline=..., tensorrt_pipeline=...)`.

5. **CLI**
   - The unified entrypoint is `python -m deployment.cli.main <project> <deploy_cfg.py> <model_cfg.py> ...`. Project-specific flags are added via `adapter.add_args`.

6. **Documentation**
   - Update [projects.md](./projects.md) and this file when adding or changing bundles.
   - Update the top-level [README.md](../README.md) only when **user-visible** status or entrypoints change (keep that file short).

## Common mistakes when adding a project

- **Task-specific preprocessing in generic runners** — keep preprocessing and collate in project loaders / pipelines, not in `BaseDeploymentRunner`.
- **Constructing exporters in the CLI** — exporters come from `ExporterFactory` inside the runtime / export orchestrator; the CLI dispatches to the project adapter only.
- **Metrics inside pipelines** — pipelines run inference and tensor shaping; evaluators own metrics (see [core_contract.md](./core_contract.md)).
- **Duplicated component names** — the `components` **dict key** is the canonical component id; do not add a redundant nested `name` field inside each component spec (typed loading sets `ComponentCfg.name` from the key).
- **Skipping registry wiring** — without `ProjectAdapter` registration, the unified CLI will not discover the project.
