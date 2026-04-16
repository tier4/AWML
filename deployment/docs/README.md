# Deployment documentation

Reader-path index for the `deployment/` package. Files keep **stable names**; sections below group by **task**, not by filename alone.

## 1. Start here

| Doc | Use when |
| --- | --- |
| [getting_started.md](./getting_started.md) | First visit: what the framework does and where to click next |
| [overview.md](./overview.md) | Concepts: principles, capabilities, export modes (light on config) |
| [usage.md](./usage.md) | You will run the CLI today: syntax, modes, logging, typical flows |
| [projects.md](./projects.md) | Which project bundles exist, maturity, where to start per project |

## 2. Learn the architecture

| Doc | Use when |
| --- | --- |
| [architecture.md](./architecture.md) | How CLI, project bundles, runner, orchestrators, and pipelines connect |
| [core_contract.md](./core_contract.md) | Maintainer “law”: responsibilities between runner, evaluator, pipelines, metrics |

## 3. Configure and run

| Doc | Use when |
| --- | --- |
| [configuration.md](./configuration.md) | **Authoritative** deploy config: top-level keys, `components`, devices, export, evaluation, verification, logging |
| [export_pipeline.md](./export_pipeline.md) | Export mental model and execution flow (defers schema to `configuration.md`) |
| [verification_evaluation.md](./verification_evaluation.md) | Why verify, how scenarios work, evaluation behavior, artifacts, reporting (schemas: `configuration.md`) |

## 4. Extend and operate

| Doc | Use when |
| --- | --- |
| [contributing.md](./contributing.md) | Adding a `deployment/projects/<name>/` bundle; read `core_contract.md` first |
| [best_practices.md](./best_practices.md) | Troubleshooting and lessons from real runs — not the architecture contract |

**Canonical multi-component example config:** `deployment/projects/centerpoint/config/deploy_config.py`
