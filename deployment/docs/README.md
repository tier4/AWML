# Deployment docs index

Guides for the `deployment/` package (CLI, configs, runtime, projects):

| Doc | Contents |
| --- | --- |
| [`overview.md`](./overview.md) | Goals, design principles, features, precision policies. |
| [`architecture.md`](./architecture.md) | Workflow, components, directory layout (CLI → projects → runtime). |
| [`usage.md`](./usage.md) | Unified CLI, logging, contexts, export modes. |
| [`configuration.md`](./configuration.md) | Deploy config keys, `components`, `deploy_log_path`, evaluation/verification. |
| [`projects.md`](./projects.md) | CenterPoint layout; placeholders for other tasks. |
| [`export_pipeline.md`](./export_pipeline.md) | ONNX/TRT steps, multi-component config, artifacts. |
| [`verification_evaluation.md`](./verification_evaluation.md) | Verification scenarios, evaluation backends, logging of results. |
| [`best_practices.md`](./best_practices.md) | Config, export, verification, evaluation, troubleshooting. |
| [`contributing.md`](./contributing.md) | Adding a `deployment/projects/<name>/` bundle and registry wiring. |
| [`core_contract.md`](./core_contract.md) | Responsibilities between runner, evaluator, pipelines, metrics. |

The canonical multi-component example config is `deployment/projects/centerpoint/config/deploy_config.py`.
