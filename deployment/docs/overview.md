# Deployment Overview

The AWML Deployment Framework provides a standardized, task-agnostic approach to exporting PyTorch models to ONNX and TensorRT with verification and evaluation baked in. It abstracts the common workflow steps while leaving space for project-specific customization so that CenterPoint, YOLOX, CalibrationStatusClassification, and future models can share the same deployment flow.

## Design Principles

1. **Unified interface** – a shared `BaseDeploymentRunner` with thin project-specific subclasses.
2. **Task-agnostic core** – base classes support detection, classification, and segmentation tasks.
3. **Backend flexibility** – PyTorch, ONNX, and TensorRT backends are first-class citizens.
4. **Pipeline architecture** – common pre/postprocessing with backend-specific inference stages.
5. **Configuration-driven** – configs plus typed dataclasses provide predictable defaults and IDE support.
6. **Dependency injection** – exporters, wrappers, and export pipelines are explicitly wired for clarity and testability.
7. **Type-safe building blocks** – typed configs, runtime contexts, and result objects reduce runtime surprises.
8. **Extensible verification** – mixins compare nested outputs so that evaluators stay lightweight.

## Key Features

### Unified Deployment Workflow

```
Load Model → Export ONNX → Export TensorRT → Verify → Evaluate
```

### Scenario-Based Verification

`VerificationMixin` normalizes devices, reuses pipelines from `PipelineFactory`, and recursively compares nested outputs with per-node logging. Scenarios define which backend pairs to compare.

```python
verification = dict(
    enabled=True,
    scenarios={
        "both": [
            {"ref_backend": "pytorch", "ref_device": "cpu",
             "test_backend": "onnx", "test_device": "cpu"},
            {"ref_backend": "onnx", "ref_device": "cpu",
             "test_backend": "tensorrt", "test_device": "cuda:0"},
        ]
    }
)
```

### Multi-Backend Evaluation

Evaluators return typed results via `EvalResultDict` (TypedDict) ensuring consistent structure across backends. Metrics interfaces (`Detection3DMetricsInterface`, `Detection2DMetricsInterface`, `ClassificationMetricsInterface`) compute task-specific metrics using `autoware_perception_evaluation`.

### Pipeline Architecture

Shared preprocessing/postprocessing steps plug into backend-specific inference. Preprocessing can be generated from MMDet/MMDet3D configs via `build_preprocessing_pipeline`.

### Flexible Export Modes

- `mode="onnx"` – PyTorch → ONNX only.
- `mode="trt"` – Build TensorRT from an existing ONNX export.
- `mode="both"` – Full export pipeline.
- `mode="none"` – Skip export and only run evaluation.

### TensorRT Precision Policies

Supports `auto`, `fp16`, `fp32_tf32`, and `strongly_typed` modes with typed configuration to keep engine builds reproducible.
