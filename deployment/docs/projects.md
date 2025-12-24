# Project Guides

## CenterPoint (3D Detection)

**Highlights**

- Multi-file ONNX export (voxel encoder + backbone/head) orchestrated via workflows.
- ONNX-compatible model configuration that mirrors training graph.
- Composed exporters keep logic reusable.

**Workflows & Wrappers**

- `CenterPointONNXExportWorkflow` – drives multiple ONNX exports using the generic `ONNXExporter`.
- `CenterPointTensorRTExportWorkflow` – converts each ONNX file via the generic `TensorRTExporter`.
- `CenterPointONNXWrapper` – identity wrapper.

**Key Files**

- `projects/CenterPoint/deploy/main.py`
- `projects/CenterPoint/deploy/evaluator.py`
- `deployment/pipelines/centerpoint/`
- `deployment/exporters/centerpoint/onnx_workflow.py`
- `deployment/exporters/centerpoint/tensorrt_workflow.py`

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

- `projects/YOLOX_opt_elan/deploy/main.py`
- `projects/YOLOX_opt_elan/deploy/evaluator.py`
- `deployment/pipelines/yolox/`
- `deployment/exporters/yolox/model_wrappers.py`

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

- `projects/CalibrationStatusClassification/deploy/main.py`
- `projects/CalibrationStatusClassification/deploy/evaluator.py`
- `deployment/pipelines/calibration/`
- `deployment/exporters/calibration/model_wrappers.py`

**Pipeline Structure**

```
preprocess() → run_model() → postprocess()
```
