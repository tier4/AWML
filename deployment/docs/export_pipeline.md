# Export Pipelines

## ONNX Export

1. **Model preparation** – load PyTorch model and apply the wrapper if output reshaping is required.
2. **Input preparation** – grab a representative sample from the data loader.
3. **Export** – call `torch.onnx.export()` with the configured settings.
4. **Simplification** – optionally run ONNX simplification.
5. **Save** – store artifacts under `work_dir/onnx/`.

## TensorRT Export

1. **Validate ONNX** – ensure the ONNX model exists and is compatible.
2. **Network creation** – parse ONNX and build a TensorRT network.
3. **Precision policy** – apply the configured precision mode (`auto`, `fp16`, `fp32_tf32`, `strongly_typed`).
4. **Optimization profile** – configure dynamic-shape ranges.
5. **Engine build** – compile and serialize the engine.
6. **Save** – store artifacts under `work_dir/tensorrt/`.

## Multi-File Export (CenterPoint)

CenterPoint splits the model into multiple ONNX/TensorRT artifacts using a unified `components` configuration:

```python
components = dict(
    voxel_encoder=dict(
        name="pts_voxel_encoder",
        onnx_file="pts_voxel_encoder.onnx",     # ONNX output filename
        engine_file="pts_voxel_encoder.engine", # TensorRT output filename
        io=dict(
            inputs=[dict(name="input_features", dtype="float32")],
            outputs=[dict(name="pillar_features", dtype="float32")],
            dynamic_axes={...},
        ),
        tensorrt_profile=dict(
            input_features=dict(min_shape=[...], opt_shape=[...], max_shape=[...]),
        ),
    ),
    backbone_head=dict(
        name="pts_backbone_neck_head",
        onnx_file="pts_backbone_neck_head.onnx",
        engine_file="pts_backbone_neck_head.engine",
        io=dict(...),
        tensorrt_profile=dict(...),
    ),
)
```

### Configuration Structure

Each component in `deploy_cfg.components` defines:

- `name`: Component identifier used during export
- `onnx_file`: Output ONNX filename
- `engine_file`: Output TensorRT engine filename
- `io`: Input/output specification (names, dtypes, dynamic_axes)
- `tensorrt_profile`: TensorRT optimization profile (min/opt/max shapes)

### Export Pipeline Orchestration

Export pipelines orchestrate:

- Sequential export of each component
- Input/output wiring between stages
- Directory structure management

CenterPoint uses a project-specific `ModelComponentExtractor` implementation that provides:

- `extract_features(model, data_loader, sample_idx)`: project-specific feature extraction for tracing
- `extract_components(model, sample_data)`: splitting into ONNX-exportable submodules and per-component config overrides

## Verification-Oriented Exports

- Exporters register artifacts via `ArtifactManager`, making the exported files discoverable for verification and evaluation.
- Wrappers ensure consistent tensor ordering and shape expectations across backends.

## Dependency Injection Pattern

Projects inject wrappers and export pipelines when instantiating the runner:

```python
runner = CenterPointDeploymentRunner(
    ...,
    onnx_pipeline=CenterPointONNXExportPipeline(...),
    tensorrt_pipeline=CenterPointTensorRTExportPipeline(...),
)
```

Simple projects can skip export pipelines entirely and rely on the base exporters provided by `ExporterFactory`.

## Runtime Pipeline Usage

Runtime pipelines receive the `components_cfg` through constructor injection:

```python
pipeline = CenterPointONNXPipeline(
    pytorch_model=model,
    onnx_dir="/path/to/onnx",
    device="cuda:0",
    components_cfg=deploy_cfg["components"],  # Pass component config
)
```

This allows pipelines to resolve artifact paths from the unified config.
