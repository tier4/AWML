# Getting started

Use this page if you are opening the deployment docs for the first time.

## What you get

A single CLI and shared runtime that run **export → verify → evaluate** for each registered project. Task-specific code stays in **`deployment/projects/<name>/`**; the framework supplies runners, orchestrators, exporters, and typed deploy config.

## Pick your path

1. **Understand the idea** — [overview.md](./overview.md) (principles and features, minimal config).
2. **Run something** — [usage.md](./usage.md) plus the project README (for CenterPoint: `deployment/projects/centerpoint/README.md`).
3. **See what ships today** — [projects.md](./projects.md) (CenterPoint is the full reference; others are planned or legacy references).

## Common next steps

| Question | Doc |
| --- | --- |
| What are the deploy config keys and examples? | [configuration.md](./configuration.md) |
| How do ONNX/TRT export and artifacts fit together? | [export_pipeline.md](./export_pipeline.md) |
| How do verification scenarios and evaluation backends work? | [verification_evaluation.md](./verification_evaluation.md) |
| How is the code layered internally? | [architecture.md](./architecture.md) |
| I want to add a new project | [contributing.md](./contributing.md) after [core_contract.md](./core_contract.md) |
