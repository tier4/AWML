# Verification and evaluation

Framework **quality gates**: why they exist, how verification scenarios are selected, what evaluation measures, how artifacts resolve, and what to expect in logs. **Typed fields and copy-paste config** for `verification` and `evaluation` are in [configuration.md](./configuration.md); this page focuses on **behavior** and responsibilities.

## Why verification exists

Verification answers: **did export preserve numerics and tensor structure across backends?** It runs paired inference on shared samples and compares nested outputs within tolerance, so regressions surface before you rely on ONNX or TensorRT in production.

## How scenarios work

`VerificationMixin` (on `BaseEvaluator`) runs scenario-based comparisons:

1. Build reference/test pipelines through `PipelineFactory` using the same project name and `components_cfg`.
2. Normalize devices per backend where required (e.g. TensorRT on CUDA).
3. Run inference on shared samples.
4. Recursively compare nested outputs using `verification.tolerance`.
5. Aggregate pass/fail statistics.

Configuration is typed as `VerificationConfig` (`deployment/configs/schema.py`). Important details:

- **`scenarios`** — dict keyed by export mode: `both`, `onnx`, `trt`, `none` (see `ExportMode`). Only scenarios for the **current** export mode run.
- **`devices`** — optional map passed through verification (alongside top-level `devices` used elsewhere); keep scenario `ref_device` / `test_device` strings consistent with available hardware.

**Examples:** [configuration.md](./configuration.md#verification).

## What evaluation measures

Task evaluators share typed results (`EvalResultDict`) and metrics interfaces (3D / 2D / classification) so reports stay comparable across backends.

**3D detection (CenterPoint)** — mAP / mAPH (by mode), per-class AP, latency stats, optional stage-wise latency breakdown.

**Classification** — accuracy, precision/recall, confusion matrix, latency (when wired).

**Evaluation config:** see [configuration.md](./configuration.md#evaluation).

## How artifact resolution works

`ArtifactManager` resolves ONNX/TensorRT paths for evaluation and verification: registered artifacts from the current export run, explicit directories in `evaluation.backends`, then sensible fallbacks tied to `export` paths. When re-running with `export.mode="none"`, set `model_dir` / `engine_dir` for ONNX and TensorRT backends so resolution does not depend on in-process registration alone.

**Detailed resolution notes** are co-documented with evaluation examples in [configuration.md](./configuration.md#evaluation).

## Reporting and logging expectations

Evaluators implement `print_results(results)`; implementations should **emit summaries via the `logging` module** (e.g. module logger or caller’s logger), not `print`, so file logging from `deploy_log_path` captures the same text as the console.

## Related architecture rules

[core_contract.md](./core_contract.md) defines allowed dependencies between runners, evaluators, pipelines, and metrics when you extend or refactor these stages.
