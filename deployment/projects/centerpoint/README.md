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
- `export.work_dir`, `export.mode`, and `export.sample_idx`
- `components`

Required component keys are `pts_voxel_encoder` and `pts_backbone_neck_head`.

The evaluation/verification dataset comes from the **model config's** `test_dataloader.dataset.ann_file` (the test info), not from the deploy config.

## Project layout

Directory names mirror the framework module each one implements; see the
[project layout contract](../../docs/architecture.md#project-layout-contract)
for the generic project → framework → base-class mapping. The concrete
CenterPoint classes are:

| Path | Role |
| --- | --- |
| `__init__.py` | Registers the `centerpoint` `ProjectAdapter` |
| `entrypoint.py` | Builds config, loader, evaluator, runner, and export context |
| `cli.py` | Project-specific CLI flags (`--rot-y-axis-reference`) |
| `runner.py` | `CenterPointDeploymentRunner` |
| `config/` | Deploy config |
| `io/` | Data loading and model loading helpers |
| `evaluation/` | `CenterPointEvaluator` and `CenterPointExecutor` |
| `inference/` | PyTorch, ONNX, and TensorRT inference pipelines |
| `contexts.py` | `CenterPointExportContext` |
| `export/` | CenterPoint export orchestration (builder, sample extractor, `onnx_models/`) |

## Shared docs

- [../../docs/runbook.md](../../docs/runbook.md) for CLI behavior and run flow
- [../../docs/configuration.md](../../docs/configuration.md) for shared config reference
- [../../docs/architecture.md](../../docs/architecture.md) for framework structure
- [../../docs/operations.md](../../docs/operations.md) for troubleshooting
