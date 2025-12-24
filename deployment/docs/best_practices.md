# Best Practices & Troubleshooting

## Configuration Management

- Keep deployment configs separate from training/model configs.
- Use relative paths for datasets and artifacts when possible.
- Document non-default configuration options in project READMEs.

## Model Export

- Inject wrapper classes (and optional workflows) into project runners; let `ExporterFactory` build exporters lazily.
- Store wrappers under `exporters/{model}/model_wrappers.py` and reuse `IdentityWrapper` when reshaping is unnecessary.
- Add workflow modules only when orchestration beyond single file export is required.
- Always verify ONNX exports before TensorRT conversion.
- Choose TensorRT precision policies (`auto`, `fp16`, `fp32_tf32`, `strongly_typed`) based on deployment targets.

## Unified Architecture Pattern

```
exporters/{model}/
├── model_wrappers.py
├── [optional] onnx_workflow.py
└── [optional] tensorrt_workflow.py
```

- Simple models: use base exporters + wrappers, no subclassing.
- Complex models: compose workflows that call the base exporters multiple times.

## Dependency Injection Pattern

```python
runner = YOLOXOptElanDeploymentRunner(
    ...,
    onnx_wrapper_cls=YOLOXOptElanONNXWrapper,
)
```

- Keeps dependencies explicit.
- Enables lazy exporter construction.
- Simplifies testing via mock wrappers/workflows.

## Verification Tips

- Start with strict tolerances (0.01) and relax only when necessary.
- Verify a representative sample set.
- Ensure preprocessing/postprocessing is consistent across backends.

## Evaluation Tips

- Align evaluation settings across backends.
- Report latency statistics alongside accuracy metrics.
- Compare backend-specific outputs for regressions.

## Pipeline Development

- Inherit from the correct task-specific base pipeline.
- Share preprocessing/postprocessing logic where possible.
- Keep backend-specific implementations focused on inference glue code.

## Troubleshooting

1. **ONNX export fails**
   - Check for unsupported ops and validate input shapes.
   - Try alternative opset versions.
2. **TensorRT build fails**
   - Validate the ONNX model.
   - Confirm input shape/profile configuration.
   - Adjust workspace size if memory errors occur.
3. **Verification fails**
   - Tweak tolerance settings.
   - Confirm identical preprocessing across backends.
   - Verify device assignments.
4. **Evaluation errors**
   - Double-check data loader paths.
   - Ensure model outputs match evaluator expectations.
   - Confirm the correct `task_type` in config.

## Future Enhancements

- Support more task types (segmentation, etc.).
- Automatic precision tuning for TensorRT.
- Distributed evaluation support.
- MLOps pipeline integration.
- Performance profiling tools.
