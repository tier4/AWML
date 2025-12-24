# Contributing to Deployment

## Adding a New Project

1. **Evaluator & Data Loader**
   - Implement `BaseEvaluator` with task-specific metrics.
   - Implement `BaseDataLoader` variant for the dataset(s).

2. **Exporters**
   - Add `exporters/{project}/model_wrappers.py` (reuse `IdentityWrapper` or implement a custom wrapper).
   - Introduce `onnx_workflow.py` / `tensorrt_workflow.py` only if multi-stage orchestration is required; prefer composing the base exporters instead of subclassing them.

3. **Pipelines**
   - Inherit from the appropriate task base (`Detection2D`, `Detection3D`, `Classification`).
   - Add backend-specific implementations (PyTorch, ONNX, TensorRT) only when behavior deviates from existing ones.

4. **Configuration**
   - Create `projects/{project}/deploy/configs/deploy_config.py`.
   - Configure export, verification, and evaluation settings with typed dataclasses where possible.

5. **Entry Point**
   - Add `projects/{project}/deploy/main.py`.
   - Follow the dependency injection pattern: explicitly pass wrapper classes and workflows to the runner.

6. **Documentation**
   - Update `deployment/README.md` and the relevant docs in `deployment/docs/`.
   - Document special requirements, configuration flags, or workflows.

## Core Contract

Before touching shared components, review `deployment/docs/core_contract.md` to understand allowed dependencies between runners, evaluators, pipelines, and exporters. Adhering to the contract keeps refactors safe and ensures new logic lands in the correct layer.
