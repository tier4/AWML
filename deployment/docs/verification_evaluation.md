# Verification and evaluation

## Verification

`VerificationMixin` (on `BaseEvaluator`) runs scenario-based comparisons:

1. Build reference/test pipelines through `PipelineFactory` using the same project name and `components_cfg`.
2. Normalize devices per backend where required (e.g. TensorRT on CUDA).
3. Run inference on shared samples.
4. Recursively compare nested outputs using `verification.tolerance`.
5. Aggregate pass/fail statistics.

Configuration is typed as `VerificationConfig` (`deployment/configs/schema.py`). Important details:

- **`scenarios`** — dict keyed by export mode: `both`, `onnx`, `trt`, `none` (see `ExportMode`). Only scenarios for the current export mode are executed.
- **`devices`** — optional map passed through verification (alongside top-level `devices` used elsewhere); keep scenario `ref_device` / `test_device` strings consistent with available hardware.

```python
verification = dict(
    enabled=True,
    scenarios=dict(
        both=[
            dict(
                ref_backend="pytorch",
                ref_device="cpu",
                test_backend="onnx",
                test_device="cpu",
            ),
        ],
        onnx=[],
        trt=[],
        none=[],
    ),
    tolerance=0.1,
    num_verify_samples=3,
    devices=devices,
)
```

## Evaluation

Task evaluators share typed results (`EvalResultDict`) and metrics interfaces (3D / 2D / classification) so reports stay comparable across backends.

**3D detection (CenterPoint)** — mAP / mAPH (by mode), per-class AP, latency stats, optional stage-wise latency breakdown.

**Classification** — accuracy, precision/recall, confusion matrix, latency (when wired).

### Evaluation config

Typed as `EvaluationConfig`. Backend entries are plain dicts; ONNX and TensorRT often need explicit directories so `ArtifactManager` can resolve artifacts after export or when re-running with `export.mode="none"`:

```python
evaluation = dict(
    enabled=True,
    num_samples=100,
    verbose=False,
    backends=dict(
        pytorch=dict(enabled=True, device="cuda:0"),
        onnx=dict(enabled=True, device="cuda:0", model_dir="work_dirs/deployment/onnx"),
        tensorrt=dict(enabled=True, device="cuda:0", engine_dir="work_dirs/deployment/tensorrt"),
    ),
)
```

### Reporting

Evaluators implement `print_results(results)`; implementations should **emit summaries via the `logging` module** (e.g. module logger or caller’s logger), not `print`, so file logging from `deploy_log_path` captures the same text as the console.

## Core contract

`deployment/docs/core_contract.md` defines allowed dependencies between runners, evaluators, pipelines, and metrics.
