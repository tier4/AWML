# Project Guides

## CenterPoint (3D Detection)

**Highlights**

- Multi-file ONNX export (voxel encoder + backbone/head) orchestrated via export pipelines.
- ONNX-compatible model configuration that mirrors training graph.
- Composed exporters keep logic reusable.

**Pipelines & Wrappers**

- `CenterPointONNXExportPipeline` – drives multiple ONNX exports using the generic `ONNXExporter`.
- `CenterPointTensorRTExportPipeline` – converts each ONNX file via the generic `TensorRTExporter`.
- `CenterPointONNXWrapper` – identity wrapper.

**Key Files**

- `deployment/cli/main.py` (single entrypoint)
- `deployment/projects/centerpoint/entrypoint.py`
- `deployment/projects/centerpoint/evaluator.py`
- `deployment/projects/centerpoint/pipelines/`
- `deployment/projects/centerpoint/export/`

**Pipeline Structure**

```
preprocess() → run_voxel_encoder() → process_middle_encoder() →
run_backbone_head() → postprocess()
```

## YOLOX (2D Detection)

**Highlights**

- Standard single-file ONNX export.
- `YOLOXOptElanONNXWrapper` reshapes output to Tier4-compatible format.
- ReLU6 → ReLU replacement for ONNX compatibility.

**Export Stack**

- `ONNXExporter` and `TensorRTExporter` instantiated via `ExporterFactory` with the YOLOX wrapper.

**Key Files**

- `deployment/cli/main.py` (single entrypoint)
- `deployment/projects/yolox_opt_elan/` (planned bundle; not migrated yet)

**Pipeline Structure**

```
preprocess() → run_model() → postprocess()
```

## CalibrationStatusClassification

**Highlights**

- Binary classification deployment with calibrated/miscalibrated data loaders.
- Single-file ONNX export with no extra output reshaping.

**Export Stack**

- `ONNXExporter` and `TensorRTExporter` with `CalibrationONNXWrapper` (identity wrapper).

**Key Files**

- `deployment/projects/calibration_status_classification/legacy/main.py` (legacy script)

**Pipeline Structure**

```
preprocess() → run_model() → postprocess()
```
