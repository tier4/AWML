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

CenterPoint splits the model into multiple ONNX/TensorRT artifacts:

- The produced filenames are driven by `deploy_cfg.onnx_config.components[*].onnx_file` (ONNX)
- The produced engine filenames are driven by `deploy_cfg.tensorrt_config.components[*].engine_file` (TensorRT, optional)

Export pipelines orchestrate:

- Sequential export of each component.
- Input/output wiring between stages.
- Directory structure management.

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
