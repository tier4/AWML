# Deployment Framework Cleanup & Refactor Plan

Working checklist for the cleanliness pass across the shared framework and the
`bevfusion_l` / `centerpoint` project bundles. Goal: clear responsibilities, clear
naming, high readability, no smells / hard-code / over-engineering, and BEVFusion aligned
with CenterPoint's clean architecture (fixing the shared layer once where both diverge).

Status legend: `[ ]` todo · `[x]` done · `[~]` intentionally skipped / deferred (documented).

Verification note: the host has no torch/CUDA/mmengine (see project memory), so each change is
verified statically — `ast.parse`, `pyflakes`, targeted `grep`, CLI package discovery, and
`exec()` of the (pure-Python) deploy config. A Docker e2e smoke test (`bevfusion_l` / `centerpoint`
export+eval) is still recommended after the pass and is **not** covered here.

---

## Tier A — safe fixes (dead code / one real bug)

- [x] **A-BUG** `OutputComparator._merge_summaries` `inf * 0 = nan` fixed by skipping zero-element
  children in the weighted mean. Verified: mixed shape-mismatch + valid child now yields finite
  `mean_diff` with `max_diff=inf`. `verification/output_comparator.py`.
- [x] **A1** `VerificationOrchestrator` now logs `verification_results["error"]` (device-validation
  failures) and `continue`s instead of counting the scenario as 0/0.
- [x] **A2** Removed dead `TensorDiffDetail.passed` field + both construction sites.
- [x] **A3** Removed unused module `logger` (and the `logging` import) from `primitives/artifacts.py`.
- [x] **A4** `tensorrt_plugins.py`: dropped dead `loaded_now`, collapsed the duplicated `CDLL`
  branch to one call + conditional log, and now returns the libraries newly loaded by this call
  (matches the docstring) instead of the cumulative global set.

## Tier B — shared consistency (fix once, helps both projects)

- [x] **A5** Added `_enum_from_value` helper in `config/enums.py`; `PrecisionPolicy` / `ExportMode`
  / `Backend` all route through it (consistent None-default + `ValueError`/`TypeError`). Behavior
  smoke-tested.
- [x] **A7** Added `verification/reporting.py` (`BANNER_WIDTH=80`, `banner()`, `format_verdict()`);
  applied in `backend_verifier.py` (was `60` + inline emoji) and `verification_orchestrator.py`
  (was `80`). Also unified the counter noun to "samples" and the `policy`→`scenario` terminology.
- [x] **A8** `load_tensorrt_plugin_libraries` no longer takes an injected `logger` (uses a module
  logger). Both call sites updated: `bevfusion_l` TRT pipeline **and** the shared
  `export/exporters/tensorrt_exporter.py` (the second caller — caught by the grep sweep).
- [x] **A9** Consistent `__str__` on `PrecisionPolicy` / `ExportMode` / `Backend` (all return `.value`).
- [x] **(bonus)** Fixed the garbled `_fmt_finite_diff` docstring in `backend_verifier.py`.

## Tier C — BEVFusion cleanup (align with CenterPoint)

- [x] **P0-2** BEVFusion TRT pipeline now stores `self._engines` / `self._contexts` dicts keyed by
  component name (CenterPoint pattern). 6 attributes + `_split` branching → 2 dicts + a uniform
  `_load_tensorrt_engines` loop and a single-line `_release_gpu_resources`.
- [x] **P1-2** Runner docstring trimmed from ~20 lines to CenterPoint brevity.
- [x] **P1-3** Dropped `_pick_bound_input_name` + its "mAP=0" warning; the single-input dense engine
  now binds `input_names[0]`. `strict=False` output ordering **kept deliberately** (BEVFusion export
  had name drift; aligning to CenterPoint's `strict=True` is unsafe without an e2e run).

## Tier D — config cleanliness

- [x] **P1-1** `deploy_config.py` restructured to CenterPoint's numbered-section layout with hoisted
  single-source `_` literals (`_CUDA`, `_WORK_DIR`/`_ONNX_DIR`/`_TENSORRT_DIR`, `_LIDAR_BEV_SHAPE`,
  voxel-profile literals) and a cleaned/accurate docstring. **All values preserved** — verified by
  `exec()`-ing the file and asserting every resolved value (incl. `engine_dir`) matches the original.
- [~] **P0-3 (DEFERRED)** `_base_` dedup of the two variants. The base bakes computed paths
  (`_TENSORRT_DIR` → `evaluation.backends.tensorrt.engine_dir`), so a child overriding
  `export.work_dir` would silently keep the base's `engine_dir` unless it also overrides that nested
  path — fragile, and unverifiable here (no mmengine to run `Config.fromfile`). Do this in Docker
  where the resolved dicts can be asserted equal, or after refactoring the base to not bake derived
  paths. Left the three explicit configs as-is (they work).

## Tier E — cross-project entrypoint dedup

- [x] **P0-1** Added `runtime/detection3d_entrypoint.py::run_detection3d_deployment(...)`. Both
  `bevfusion_l/entrypoint.py` and `centerpoint/entrypoint.py` are now ~30 lines that inject only
  `pipeline_name` + `config_factory` + `executor_factory` + `runner_factory`. The ~90% duplicated
  wiring lives once.

---

## Intentionally skipped (cosmetic / would be churn, not value)

- [~] `DeviceSpec.to_ort_provider` / `to_torch_device` "leaky primitive" — pragmatic, low ROI.
- [~] 44-line docstring on `resolve_artifact_path`; `# ===` section banners; PEP585-vs-`typing`
  generics mixing; `__post_init__` re-validating a `Literal`; `Artifact.exists` (dir-OK) vs
  resolver `is_file()`.
- [~] model_loader strategy divergence (CenterPoint type-swap vs BEVFusion wrappers), split/merged
  branching, CPU-vs-CUDA load, 2-vs-3 backends, component naming — justified divergences.

---

## Re-audit results

- [x] `ast.parse` clean on all 17 changed `.py`.
- [x] `pyflakes` clean (no unused imports / undefined names) on all changed `.py`.
- [x] CLI package discovery lists exactly `['bevfusion_l', 'centerpoint']`.
- [x] grep: no references to removed symbols (`_pick_bound_input_name`, `_engine_sparse`,
  `_apply_spconv_do_sort`, `_get_num_proposals`, 2-arg `load_tensorrt_plugin_libraries`).
- [x] enum + comparator-nan behavior smoke-tested; deploy_config values `exec()`-verified.
- [ ] **Docker e2e smoke test** (`bevfusion_l` + `centerpoint` export/eval) — REQUIRED, not run here.
