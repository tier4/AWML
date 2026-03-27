# Export pipelines

## ONNX export (generic)

1. Load PyTorch checkpoint and optional wrapper for export-time subgraphs.
2. Take a representative sample (often via project `SampleAdapter` / data loader).
3. Run `torch.onnx.export` with merged settings from `onnx_config` and the target `components` entry.
4. Optionally simplify ONNX.
5. Write files under `export.work_dir` (typically `.../onnx/` for multi-component projects).

ONNX filenames come from **`components.<id>.onnx_file`**, not from a global `save_file` in `onnx_config`.

## TensorRT export (generic)

1. Validate ONNX inputs exist.
2. Parse ONNX and build the TensorRT network.
3. Apply `tensorrt_config.precision_policy` (`auto`, `fp16`, `fp32_tf32`, `strongly_typed`, …).
4. Apply per-input min/opt/max shapes from **`components.<id>.tensorrt_profile`**.
5. Build and serialize the engine to **`components.<id>.engine_file`** under the TensorRT output directory.

## Multi-component export (CenterPoint)

`deploy_cfg["components"]` is a map from **component id** to spec. The id is the canonical name (`pts_voxel_encoder`, `pts_backbone_neck_head`, …). Typed loading sets `ComponentCfg.name` from that key—**do not** duplicate a `name` field inside the nested dict.

Each value defines:

- `onnx_file`, `engine_file`
- `io`: `inputs` / `outputs` (name + dtype), `dynamic_axes`
- `tensorrt_profile`: map from binding name to `min_shape` / `opt_shape` / `max_shape`

Export pipelines orchestrate sequential export per component and wire tensors between stages where needed. CenterPoint uses:

- `CenterPointSampleAdapter` — trace/sample payload for export
- `CenterPointComponentBuilder` — split `nn.Module` into exportable pieces aligned with `components`

## Artifacts and downstream stages

`ArtifactManager` registers paths after export so verification and evaluation resolve ONNX directories and engine directories consistently. Wrappers keep tensor layout aligned across backends.

## Dependency injection (runner)

Project runners pass optional pipelines into `BaseDeploymentRunner`:

```python
runner = CenterPointDeploymentRunner(
    data_loader=data_loader,
    evaluator=evaluator,
    config=config,
    model_cfg=model_cfg,
    logger=logger,
    onnx_pipeline=custom_onnx_pipeline,  # optional
    tensorrt_pipeline=custom_trt_pipeline,  # optional
)
```

If omitted, defaults are constructed inside the CenterPoint runner (see `deployment/projects/centerpoint/runner.py`).

## Runtime inference pipelines

Evaluators create pipelines with `PipelineFactory.create(...)` and pass the typed **`ComponentsConfig`** instance (`config.components_cfg`), not a raw dict:

```python
PipelineFactory.create(
    project_name="centerpoint",
    model_spec=model_spec,
    pytorch_model=pytorch_model,
    device=device,
    components_cfg=self._components_cfg,
)
```

Pipelines use `components_cfg` to locate ONNX/engine filenames and I/O names for each backend.
