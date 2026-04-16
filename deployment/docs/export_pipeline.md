# Export pipelines

Mental model for **export** in two parts: what a **component** is and how specs drive ONNX/TRT, then how **execution** flows through generic exporters, artifacts, and runtime inference. **Field definitions and snippets** for `components`, `onnx_config`, and `tensorrt_config` are in [configuration.md](./configuration.md) — this page explains how those pieces behave together.

## Part 1 — Export model (mental model)

### What a component is

A **component** is one exportable slice of the model graph with its own ONNX filename, optional TensorRT engine filename, I/O metadata, and TRT shape profiles. Multi-component projects (CenterPoint) export **several** ONNX/TRT artifacts; single-component tasks use a **single** entry in the `components` map.

### Single-component vs multi-component

| Style | When | Typical layout |
| --- | --- | --- |
| Single-component | One graph, one ONNX, one engine | One key under `components` (for example `model`). |
| Multi-component | Staged graphs or separately deployable subgraphs | Multiple keys (for example `pts_voxel_encoder`, `pts_backbone_neck_head`). |

### How component specs drive ONNX and TensorRT export

- **ONNX** — per-component `onnx_file`, merged `onnx_config`, and per-component `io` (inputs, outputs, dynamic axes) drive `torch.onnx.export` and naming under `export.work_dir`.
- **TensorRT** — shared `tensorrt_config` (for example precision policy) plus per-component `tensorrt_profile` and `engine_file` drive engine build and output paths.

### Wrappers, sample adapters, and component builders

- **Wrappers** — align export-time subgraphs with deployment tensor layouts (shared under `deployment/exporters/common/`; project-specific where needed).
- **Sample adapters / loaders** — supply representative inputs for tracing and export.
- **Component builders** (CenterPoint) — split a loaded `nn.Module` into pieces that line up with `components` keys for sequential export.

**Do not** duplicate a nested `name` inside each component dict: the **dict key** is the canonical id (see [configuration.md](./configuration.md#multi-component-export-centerpoint-style)).

## Part 2 — Export execution (flows)

### Generic ONNX flow

1. Load PyTorch checkpoint and optional wrapper for export-time subgraphs.
2. Take a representative sample (often via project `SampleAdapter` / data loader).
3. Run `torch.onnx.export` with merged settings from `onnx_config` and the target `components` entry.
4. Optionally simplify ONNX.
5. Write files under `export.work_dir` (typically `.../onnx/` for multi-component projects).

ONNX filenames come from **`components.<id>.onnx_file`**, not from a global `save_file` in `onnx_config`.

### Generic TensorRT flow

1. Validate ONNX inputs exist.
2. Parse ONNX and build the TensorRT network.
3. Apply `tensorrt_config.precision_policy` (`auto`, `fp16`, `fp32_tf32`, `strongly_typed`, …).
4. Apply per-input min/opt/max shapes from **`components.<id>.tensorrt_profile`**.
5. Build and serialize the engine to **`components.<id>.engine_file`** under the TensorRT output directory.

### Artifact registration and downstream stages

`ArtifactManager` registers paths after export so verification and evaluation resolve ONNX directories and engine directories consistently. Wrappers keep tensor layout aligned across backends.

### Dependency injection (runner)

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

### Runtime inference pipelines

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
