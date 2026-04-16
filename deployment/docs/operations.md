# Deployment operations

Troubleshooting and practical deployment guidance. This page is for day-to-day operation, not for architecture reference.

Use [runbook.md](./runbook.md) for the execution flow, [configuration.md](./configuration.md) for config fields, and [architecture.md](./architecture.md) for framework boundaries.

## Configuration

- Keep deploy config separate from the training model config.
- Prefer paths rooted under one `export.work_dir` so artifacts and logs stay together.
- Set `deploy_log_path` when you need a persistent deployment log next to the exported artifacts.
- Keep one top-level `devices` map and reuse it in evaluation and verification settings.

## Export

- Start with a valid ONNX export before tuning TensorRT.
- Keep `components` aligned with the actual deployable subgraphs of the model.
- Use project export pipelines only when a project needs multi-stage or multi-file export orchestration.
- Match `precision_policy` to the deployment target instead of treating it as a generic speed knob.

## Verification

- Start with strict tolerance and relax only when there is a clear backend-driven reason.
- Keep preprocessing identical across reference and test backends.
- Group scenarios by `export.mode` so partial runs do not execute irrelevant backend pairs.

## Evaluation

- Compare latency only when devices and sample counts are aligned across backends.
- Set `model_dir` and `engine_dir` explicitly when evaluating previously exported artifacts.
- Log evaluation summaries through `logging`, not `print`, so they are captured by `deploy_log_path`.

## Pipeline development

- Pipelines should only own inference mechanics and tensor shaping.
- Evaluators own metrics, verification input preparation, and result reporting.
- Register project pipeline factories through `pipeline_registry` so all backends are created consistently.

## Troubleshooting

1. ONNX export fails: check unsupported ops, dynamic axes, and representative export inputs first.
2. TensorRT build fails: validate the ONNX graph, shape profiles, and workspace limits.
3. Verification fails: check tolerance, backend pairing, and preprocessing parity before assuming export is broken.
4. Evaluation cannot find artifacts: confirm current export registration or set explicit ONNX and TensorRT directories.
5. Log file is empty: confirm `deploy_log_path` is enabled and summaries use `logging`.
