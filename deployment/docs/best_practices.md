# Best practices and troubleshooting

Supplementary **operational** notes: troubleshooting, tuning, and lessons from real deployment runs. For architecture rules, see [core_contract.md](./core_contract.md); for authoritative config keys, see [configuration.md](./configuration.md). This is not the primary onboarding path — start from [README.md](../README.md) or [getting_started.md](./getting_started.md).

## Configuration

- Keep deploy config separate from training model config; deploy config is loaded as MMEngine `Config` into `BaseDeploymentConfig`.
- Prefer relative paths anchored to a single `export.work_dir` for ONNX/TensorRT outputs.
- Set `deploy_log_path` (e.g. `deployment.log`) if you need a persistent log next to artifacts; use `None` to disable file logging.
- Document non-default verification tolerances—numerics vary by GPU, drivers, and ORT/TRT versions (see comments in `deploy_config.py`).

## Model export

- Pass wrapper classes and optional export pipelines into the project runner; let `ExporterFactory` build exporters inside the runtime.
- Shared wrappers for reuse live under `deployment/exporters/common/model_wrappers.py`; project-specific ONNX subgraphs may live under `deployment/projects/<project>/`.
- Add export-pipeline modules only when you need multi-file or multi-stage orchestration.
- Validate ONNX before TensorRT build; adjust opset or simplify flags if export fails.
- Choose TensorRT `precision_policy` to match deployment targets (`auto`, `fp16`, `fp32_tf32`, `strongly_typed`).

## Architecture

- Simple projects: one `components` entry, base exporters, optional single pipeline factory registration.
- Complex projects: compose export pipelines from generic exporters (CenterPoint pattern).

## Verification

- Start with strict tolerance and relax only when justified.
- Use scenario sets per export mode (`both` / `onnx` / `trt`) so partial export runs do not execute irrelevant pairs.
- Keep preprocessing identical across reference and test backends.

## Evaluation

- Align `num_samples` and devices across backends when comparing latency.
- Set `model_dir` / `engine_dir` for ONNX and TensorRT if artifacts are not freshly registered in-process.
- Rely on logging for metric summaries so `deploy_log_path` captures evaluation output.

## Pipeline development

- Inherit from `BaseInferencePipeline` and keep metric computation out of pipelines (evaluators own metrics).
- Register new projects in `pipeline_registry` via a `BasePipelineFactory` subclass.

## Troubleshooting

1. **ONNX export fails** — Unsupported ops, wrong dynamic axes, or bad trace inputs; try opset change or inspect trace with `verbose` export flags where available.
2. **TensorRT build fails** — Validate ONNX, profiles (min/opt/max), and workspace size.
3. **Verification fails** — Tolerance, device pairing, or preprocessing mismatch; confirm scenario `ref_device` / `test_device` match available hardware.
4. **Evaluation cannot find models** — Check `ArtifactManager` resolution: run export first, or set `evaluation.backends.onnx.model_dir` / `tensorrt.engine_dir`.
5. **Empty log file** — Confirm `deploy_log_path` is non-empty, `export.work_dir` resolves as expected for relative paths, and code uses `logging` rather than `print` for summaries.

## Future enhancements

Ideas only; not a commitment: more task types, automated precision tuning, distributed evaluation, MLOps hooks, profiling tools.
