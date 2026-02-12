# Verification & Evaluation

## Verification

`VerificationMixin` coordinates scenario-based comparisons:

1. Resolve reference/test pipelines through `PipelineFactory`.
2. Normalize devices per backend (PyTorch → CPU, TensorRT → `cuda:0`, …).
3. Run inference on shared samples.
4. Recursively compare nested outputs with tolerance controls.
5. Emit per-sample pass/fail statistics.

Example configuration:

```python
verification = dict(
    enabled=True,
    scenarios={
        "both": [
            {
                "ref_backend": "pytorch",
                "ref_device": "cpu",
                "test_backend": "onnx",
                "test_device": "cpu"
            }
        ]
    },
    tolerance=0.1,
    num_verify_samples=3,
)
```

## Evaluation

Task-specific evaluators share typed metrics so reports stay consistent across backends.

### Detection

- mAP and per-class AP.
- Latency statistics (mean, std, min, max).

### Classification

- Accuracy, precision, recall.
- Per-class metrics and confusion matrix.
- Latency statistics.

Evaluation configuration example:

```python
evaluation = dict(
    enabled=True,
    num_samples=100,
    verbose=False,
    backends={
        "pytorch": {"enabled": True, "device": "cpu"},
        "onnx": {"enabled": True, "device": "cpu"},
        "tensorrt": {"enabled": True, "device": "cuda:0"},
    }
)
```

## Core Contract

`deployment/docs/core_contract.md` documents the responsibilities and allowed dependencies between runners, evaluators, pipelines, `PipelineFactory`, and metrics interfaces. Following the contract keeps refactors safe and ensures new projects remain compatible with shared infrastructure.
