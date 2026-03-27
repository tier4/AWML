# Project guides

The repository currently ships a **full** deployment bundle for **CenterPoint**. Other projects listed below describe the intended pattern or legacy locations; they are not present as complete `deployment/projects/<name>/` trees unless noted.

## CenterPoint (3D detection)

**Highlights**

- Multi-file ONNX and TensorRT export (voxel encoder + backbone/head) driven by `components` in deploy config.
- `CenterPointONNXExportPipeline` / `CenterPointTensorRTExportPipeline` compose generic exporters per component.
- Inference: `CenterPointPyTorchPipeline`, `CenterPointONNXPipeline`, `CenterPointTensorRTPipeline` registered via `CenterPointPipelineFactory`.

**Layout**

| Path | Role |
| --- | --- |
| `deployment/cli/main.py` | Unified CLI |
| `deployment/projects/centerpoint/entrypoint.py` | Config, logging, loader, evaluator, runner |
| `deployment/projects/centerpoint/runner.py` | `CenterPointDeploymentRunner` |
| `deployment/projects/centerpoint/config/deploy_config.py` | Reference deploy config |
| `deployment/projects/centerpoint/eval/evaluator.py` | `CenterPointEvaluator` |
| `deployment/projects/centerpoint/io/data_loader.py` | Samples and preprocessing |
| `deployment/projects/centerpoint/pipelines/` | Backend pipelines + factory |
| `deployment/projects/centerpoint/export/` | ONNX/TRT export pipelines, component builder |

**Inference flow (conceptual)**

CenterPoint pipelines chain staged subgraphs: voxelization / encoder, middle processing, backbone-head decode—see `centerpoint_pipeline.py` for the exact hook sequence.

## YOLOX (2D detection) — planned pattern

**Intended highlights**

- Single-component (or single-file) ONNX export.
- Wrapper for output layout (e.g. Tier4-compatible tensors), ReLU6 → ReLU where needed.

**Status**

- No `deployment/projects/yolox_opt_elan/` (or similar) bundle is checked in yet. When added, it should follow the same `entrypoint` + `runner` + `project_registry` pattern as CenterPoint.

## Calibration status classification — legacy reference

**Highlights**

- Binary classification, identity ONNX wrapper, single export.

**Status**

- Historical scripts may live outside this tree; there is no first-class `deployment/projects/calibration_*/` bundle in the current package. New work should use the project registry pattern under `deployment/projects/<name>/`.
