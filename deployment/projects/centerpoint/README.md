# CenterPoint deployment

CenterPoint is the current reference project for multi-component ONNX and TensorRT export inside `deployment/`.

## Quick start

From the repository root:

```bash
python -m deployment.cli.main centerpoint \
    deployment/projects/centerpoint/config/deploy_config.py \
    <path/to/your/model_config.py> \
    --rot-y-axis-reference \
    [--log-level INFO]
```

Example:

```bash
python -m deployment.cli.main centerpoint \
    deployment/projects/centerpoint/config/deploy_config.py \
    projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_8xb16_121m_j6gen2_base_amp_t4metric_v2.py \
    --rot-y-axis-reference
```

## What is project-specific here

- Multi-component export with `pts_voxel_encoder` and `pts_backbone_neck_head`
- CenterPoint-specific CLI flag `--rot-y-axis-reference`
- CenterPoint evaluator, loaders, export pipelines, and backend inference pipelines

## Config file

The reference deploy config is `deployment/projects/centerpoint/config/deploy_config.py`.

Adjust at least:

- `checkpoint_path`
- `export.work_dir` and `export.mode`
- `components`
- `runtime_io`

Required component keys are `pts_voxel_encoder` and `pts_backbone_neck_head`.

## Project layout

| Path | Role |
| --- | --- |
| `entrypoint.py` | Builds config, loader, evaluator, runner, and export context |
| `runner.py` | `CenterPointDeploymentRunner` |
| `cli.py` | Project-specific CLI flags |
| `config/` | Deploy config |
| `io/` | Data loading and model loading helpers |
| `eval/` | CenterPoint evaluator and metrics helpers |
| `pipelines/` | PyTorch, ONNX, and TensorRT pipelines |
| `export/` | CenterPoint export orchestration |
| `onnx_models/` | Export-time ONNX wrappers |

## Shared docs

- [../../docs/runbook.md](../../docs/runbook.md) for CLI behavior and run flow
- [../../docs/configuration.md](../../docs/configuration.md) for shared config reference
- [../../docs/architecture.md](../../docs/architecture.md) for framework structure
- [../../docs/operations.md](../../docs/operations.md) for troubleshooting
