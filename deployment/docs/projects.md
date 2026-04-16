# Project guides

Cross-project index: **what ships in this tree**, **maturity**, and **where to start**. Each first-class bundle under `deployment/projects/<name>/` should also ship a colocated **`README.md`** (quick start, config pointers, layout).

## Implemented

### CenterPoint (3D detection)

| | |
| --- | --- |
| **Use case** | LiDAR-based 3D object detection deployment. |
| **Export pattern** | **Multi-component** — separate ONNX/TRT artifacts per `components` entry (voxel encoder, backbone/head). |
| **Evaluator** | 3D detection metrics via shared evaluator interfaces and `autoware_perception_evaluation` (see project `eval/`). |
| **Project-specific flags** | Example: `--rot-y-axis-reference` (see `deployment/projects/centerpoint/cli.py`). |
| **Where to start** | [`deployment/projects/centerpoint/README.md`](../projects/centerpoint/README.md), then [`configuration.md`](./configuration.md) and [`export_pipeline.md`](./export_pipeline.md). |

**Technical summary**

- `CenterPointONNXExportPipeline` / `CenterPointTensorRTExportPipeline` compose generic exporters per component.
- Inference: `CenterPointPyTorchPipeline`, `CenterPointONNXPipeline`, `CenterPointTensorRTPipeline` registered via `CenterPointPipelineFactory`.
- Unified CLI: `python -m deployment.cli.main centerpoint ...`.

Canonical deploy config: `deployment/projects/centerpoint/config/deploy_config.py`.
