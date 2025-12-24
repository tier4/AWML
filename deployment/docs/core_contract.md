## Deployment Core Contract

This document defines the responsibilities and boundaries between the primary deployment components. Treat it as the “architecture contract” for contributors.

### BaseDeploymentRunner (and project runners)
- Owns the end-to-end deployment flow: load PyTorch model → export ONNX/TensorRT → verify → evaluate.
- Constructs exporters via `ExporterFactory` and never embeds exporter-specific logic.
- Injects project-provided `BaseDataLoader`, `BaseEvaluator`, model configs, wrappers, and optional workflows.
- Ensures evaluators receive:
  - Loaded PyTorch model (`set_pytorch_model`)
  - Runtime/export artifacts (via `ArtifactManager`)
  - Verification/evaluation requests (via orchestrators)
- Must not contain task-specific preprocessing/postprocessing; defer to evaluators/pipelines.

### BaseEvaluator (and task evaluators)
- The single base class for all task evaluators, integrating `VerificationMixin`.
- Provides the unified evaluation loop: iterate samples → infer → accumulate → compute metrics.
- Requires a `TaskProfile` (task name, class names) and a `BaseMetricsInterface` at construction.
- Responsible for:
  - Creating backend pipelines through `PipelineFactory`
  - Preparing verification inputs from the data loader
  - Computing task metrics using metrics interfaces
  - Printing/reporting evaluation summaries
- Subclasses implement task-specific hooks:
  - `_create_pipeline(model_spec, device)` → create backend pipeline
  - `_prepare_input(sample, data_loader, device)` → extract model input + inference kwargs
  - `_parse_predictions(pipeline_output)` → normalize raw output
  - `_parse_ground_truths(gt_data)` → extract ground truth
  - `_add_to_interface(predictions, ground_truths)` → feed metrics interface
  - `_build_results(latencies, breakdowns, num_samples)` → construct final results dict
  - `print_results(results)` → format and display results
- Inherits `VerificationMixin` automatically; subclasses only need `_get_output_names()` if custom names are desired.
- Provides common utilities: `_ensure_model_on_device()`, `_compute_latency_breakdown()`, `compute_latency_stats()`.

### BaseDeploymentPipeline & PipelineFactory
- `BaseDeploymentPipeline` defines the inference template (`preprocess → run_model → postprocess`).
- Backend-specific subclasses handle only the inference mechanics for their backend.
- `PipelineFactory` is the single entrypoint for creating pipelines per task/backend:
  - Hides backend instantiation details from evaluators.
  - Ensures consistent constructor signatures (PyTorch models vs. ONNX paths vs. TensorRT engines).
  - Central location for future pipeline wiring (new tasks/backends).
- Pipelines must avoid loading artifacts or computing metrics; they only execute inference.

### Metrics Interfaces (Autoware-based interfaces)
- Provide a uniform interface for adding frames and computing summaries regardless of task.
- Encapsulate conversion from model predictions/ground truth to Autoware perception evaluation inputs.
- Return metric dictionaries that evaluators incorporate into `EvalResultDict` results.
- Should not access loaders, runners, or exporters directly; evaluators pass in the data they need.

### Summary of Allowed Dependencies
- **Runner → Evaluator** (injection) ✓
- **Evaluator → PipelineFactory / Pipelines / Metrics Interfaces** ✓
- **PipelineFactory → Pipelines** ✓
- **Pipelines ↔ Metrics Interfaces** ✗ (evaluators mediate)
- **Metrics Interfaces → Runner/PipelineFactory** ✗

Adhering to this contract keeps responsibilities isolated, simplifies testing, and allows independent refactors of runners, evaluators, pipelines, and metrics logic.
