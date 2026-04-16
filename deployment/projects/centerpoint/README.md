# CenterPoint deployment

3D LiDAR detection: multi-component ONNX / TensorRT export, evaluation, and backend pipelines (PyTorch, ONNX, TensorRT).

## Quick start

From the repository root (with AWML on `PYTHONPATH`):

```bash
python -m deployment.cli.main centerpoint \
    deployment/projects/centerpoint/config/deploy_config.py \
    <path/to/your/model_config.py> \
    --rot-y-axis-reference \
    [--log-level DEBUG|INFO|WARNING|ERROR|CRITICAL]
```

Example:

```bash
python -m deployment.cli.main centerpoint deployment/projects/centerpoint/config/deploy_config.py   projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_8xb16_121m_j6gen2_base_amp_t4metric_v2.py   --rot-y-axis-reference
```

Global CLI behavior, logging, and the project-runner pattern are described in `deployment/docs/usage.md`.

## Configuration

| File | Role |
| --- | --- |
| `config/deploy_config.py` | Reference deploy config: checkpoint path, `export`, `components`, `runtime_io`, ONNX/TRT options, evaluation and verification. |

Adjust at least:

- `checkpoint_path` — PyTorch weights.
- `export.work_dir` / `export.mode` — output layout and whether to run ONNX, TensorRT, or both.
- `components` — per-subgraph ONNX/engine names, I/O dtypes, dynamic axes, and TensorRT profiles (must match your model grid and voxel limits).
- `runtime_io` — sample `info_file` (relative to model config `data_root`) and `sample_idx`.

Required component keys: `pts_voxel_encoder`, `pts_backbone_neck_head` (declared in project adapter metadata and validated by shared project validator).

## Layout

| Path | Role |
| --- | --- |
| `entrypoint.py` | Loads MMEngine configs, wires loader, evaluator, runner, export context. |
| `runner.py` | `CenterPointDeploymentRunner` — model load and export orchestration. |
| `cli.py` | Project-specific CLI flags. |
| `config/deploy_config.py` | Deploy-side settings. |
| `eval/` | `CenterPointEvaluator` and metric helpers. |
| `io/` | Data loading, sample types, model loader. |
| `pipelines/` | PyTorch / ONNX / TensorRT pipelines and `CenterPointPipelineFactory`. |
| `export/` | ONNX and TensorRT export pipelines, component builder. |
| `onnx_models/` | ONNX-facing model wrappers used during export. |

## Inference flow

Pipelines chain voxelization / encoder, middle processing, and backbone-head decode. The exact hook order is in `pipelines/centerpoint_pipeline.py`.

## Related docs

- `deployment/docs/getting_started.md` — first-time orientation for the deployment package.
- `deployment/docs/projects.md` — index of all deployment projects.
- `deployment/docs/usage.md` — unified CLI, logging, contexts.
- `deployment/docs/configuration.md` — shared config concepts.
