# Project guides

Each first-class bundle under `deployment/projects/<name>/` should ship its own **`README.md`** next to `entrypoint.py` (quick start, config pointers, layout). This file is the cross-project index.

The repository currently ships a **full** deployment bundle for **CenterPoint**. Other projects listed below describe the intended pattern or legacy locations; they are not present as complete `deployment/projects/<name>/` trees unless noted.

## CenterPoint (3D detection)

**Project README:** [`deployment/projects/centerpoint/README.md`](../projects/centerpoint/README.md)

**Summary**

- Multi-file ONNX and TensorRT export (voxel encoder + backbone/head) driven by `components` in deploy config.
- `CenterPointONNXExportPipeline` / `CenterPointTensorRTExportPipeline` compose generic exporters per component.
- Inference: `CenterPointPyTorchPipeline`, `CenterPointONNXPipeline`, `CenterPointTensorRTPipeline` registered via `CenterPointPipelineFactory`.

The unified CLI entry remains `deployment/cli/main.py`; directory layout, required config keys, and example commands are documented in the project README above.

## YOLOX (2D detection) — planned pattern

**Intended highlights**

- Single-component (or single-file) ONNX export.
- Wrapper for output layout (e.g. Tier4-compatible tensors), ReLU6 → ReLU where needed.

**Status**

- No `deployment/projects/yolox_opt_elan/` (or similar) bundle is checked in yet. When added, it should follow the same `entrypoint` + `runner` + `project_registry` pattern as CenterPoint, and include `deployment/projects/<name>/README.md` describing that project’s CLI flags and config.

## Calibration status classification — legacy reference

**Highlights**

- Binary classification, identity ONNX wrapper, single export.

**Status**

- Historical scripts may live outside this tree; there is no first-class `deployment/projects/calibration_*/` bundle in the current package. New work should use the project registry pattern under `deployment/projects/<name>/`, with a colocated `README.md` for that project.
