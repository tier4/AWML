# Deployment overview

Conceptual overview of the AWML Deployment Framework: **what it is for** and **what it guarantees**, without duplicating the deploy config schema (see [configuration.md](./configuration.md)) or CLI details (see [usage.md](./usage.md)).

The framework standardizes a **task-agnostic** path from PyTorch checkpoints to ONNX and TensorRT, with **verification** and **evaluation** in the same run. Project bundles (for example CenterPoint) plug in loaders, evaluators, pipelines, and optional export pipelines while reusing the shared runner and orchestration.

## Design principles

1. **Unified interface** — shared `BaseDeploymentRunner` with thin project-specific subclasses.
2. **Task-agnostic core** — no task logic in generic runners; task code lives under `deployment/projects/<name>/`.
3. **Backend flexibility** — PyTorch, ONNX, and TensorRT treated consistently.
4. **Pipeline architecture** — shared pre/postprocessing with backend-specific inference stages.
5. **Configuration-driven** — typed deploy config (`BaseDeploymentConfig`) for predictable defaults.
6. **Dependency injection** — exporters, wrappers, and export pipelines wired explicitly from project runners.
7. **Type-safe building blocks** — typed configs, runtime contexts, and result objects.
8. **Extensible verification** — scenario-based comparisons with nested output checks (see [verification_evaluation.md](./verification_evaluation.md)).
9. **Observable runs** — optional file logging via `deploy_log_path` (see [configuration.md](./configuration.md#logging-deploy_log_path)).

## Key features

### Unified deployment workflow

```
Load Model → Export ONNX → Export TensorRT → Verify → Evaluate
```

(Exact stages depend on `export.mode` and enabled backends; see [usage.md](./usage.md#most-common-workflows).)

### Scenario-based verification

Verification compares reference and test backends on shared samples, grouped by **export mode** (`both`, `onnx`, `trt`, `none`) so only relevant scenarios run. **Behavior and reporting:** [verification_evaluation.md](./verification_evaluation.md). **Config fields and examples:** [configuration.md](./configuration.md#verification).

### Multi-backend evaluation

Evaluators return structured results and use task metrics interfaces (3D / 2D / classification) so reports stay comparable across backends. **Details:** [verification_evaluation.md](./verification_evaluation.md).

### Pipeline architecture

Shared preprocessing and postprocessing plug into backend-specific inference. Project data loaders build preprocessing from the training/model stack (for example MMDet/MMDet3D test pipelines). **Internal wiring:** [architecture.md](./architecture.md).

### Flexible export modes

- `onnx` — PyTorch → ONNX only.
- `trt` — TensorRT from an existing ONNX layout.
- `both` — full export pipeline.
- `none` — skip export; verify/evaluate against existing artifacts when configured.

**Authoritative config snippets:** [configuration.md](./configuration.md).

### TensorRT precision policies

Supports `auto`, `fp16`, `fp32_tf32`, and `strongly_typed` (and related) modes via typed configuration for reproducible engine builds.
