# BEVFusion 2.8.x deployment notes

What changed in BEVFusion **2.8.x** (model release commit `78b66a70`, *"feat(BEVFusion):
release BEVFusion 2.8.x (#217)"*) that matters for the new `deployment/` framework, and
how to run a 2.8 LiDAR model through the split FP16 export → TensorRT → eval pipeline.

> The new `deployment/` framework was ported from the old (2.7-era) framework. This doc
> records the deltas needed to deploy a **2.8** checkpoint, validated end-to-end in Docker
> on 2026-06-30.

---

## 1. How to run (split FP16, 2.8 model)

```bash
# inside the awml-bevfusion container, /workspace = host AWML
python -m deployment.cli.main bevfusion_l \
  deployment/projects/bevfusion_l/config/deploy_config.py \
  projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_8xb16_j6gen2_base_120m_t4metric_v2.py
```

- Checkpoint: `work_dirs/bevfusion/bevfusion_2_8/best_epoch_25.pth` (7-class, set in the deploy config).
- Pair the checkpoint's class count with the model config. A **2.7** checkpoint
  (`best_epoch_28.pth`) is **5-class** (`car, truck, bus, bicycle, pedestrian`); a **2.8**
  checkpoint is **7-class** (adds `traffic_cone, barrier`). Mixing them gives
  `size mismatch ... [5,...] vs [7,...]` on `bbox_head`.

---

## 2. The 2.8 ONNX-export refactor (already in `projects/BEVFusion`)

These changes live in the **model code** (commit `78b66a70`, already in the host repo). The
deployment wrappers call the model directly (`BEVFusionSparseWrapper.forward` →
`model.extract_pts_feat`; `BEVFusionDenseWrapper.forward` → `pts_backbone/neck` +
`bbox_head`), so the exported ONNX **automatically** reflects them. No deployment-side port
is required for these — they are listed so the graph shape is understood.

| Area | 2.7 | 2.8 | File |
|---|---|---|---|
| Voxel mean-pool | `voxelize_reduce` inside `BEVFusion` (sum/`num_points`) | moved into a dedicated voxel encoder | `bevfusion.py` |
| Voxel feature encoding | sin-cos (`num_aug_features`) **inside** `BEVFusionSparseEncoder` | `HardSimpleVoxelSinCosEncoder` does mean-pool + sin-cos (folded `scale*x+bias` via `addcmul`) | `bevfusion_voxel_encoder.py` (new) |
| Sparse → dense | `out.dense()` + permute `(0,1,4,2,3)` | scatter-based `sparse_to_dense()` + permute `(0,4,3,1,2)` (cleaner ONNX, `dense_output_shapes`) | `custom_sparse_conv_tensor.py` (new), `sparse_encoder.py` |
| Head top-k | `argsort(...)[:num_proposals]` | `torch.topk(...)` (single TopK node) | `bevfusion_head.py` |
| Head local-max | in-place slice mutation `local_max[...] = ...` | `F.pad` + `torch.cat` + `local_concat_class_remapping` buffer (no in-place index assignment) | `bevfusion_head.py` |
| `bev_pos` | recomputed `.repeat().to(device)` | registered buffer, `query_pos = bev_pos.squeeze(0)[idx]` | `bevfusion_head.py` |

Consequence: `BEVFusion.__init__` **no longer** has `voxelize_reduce`
(`voxelize_cfg.pop("voxelize_reduce")` and `assert self.voxelize_reduce` were removed). Do
**not** reintroduce them — a 2.7-era `bevfusion.py` will crash against a 2.8 config because
2.8 configs no longer set `voxelize_reduce`.

`BEVFusionSparseEncoder.conv_out` output channels × `dense_output_shapes[2]` define the dense
`lidar_bev` channel count; for the j6gen2 base model this is **256**, with head BEV grid
**180×180** (`grid_size // out_size_factor = 1440 // 8`). These set the `bevfusion_dense`
TensorRT profile (`[1, 256, 180, 180]`).

---

## 3. Deployment-side changes that WERE applied

### 3.1 `max_num_points` 10 → 32 (TensorRT `voxels` profile)
2.8 voxelizes with `max_num_points = 32` (was 10). The `bevfusion_sparse` TensorRT profile's
`voxels` axis-1 must match, else the engine build fails with:

```
Dimension mismatch for tensor voxels and profile 0.
At dimension axis 1, profile has min=10, opt=10, max=10 but tensor has 32.
```

Fixed in `config/deploy_config.py`:
```python
voxels=dict(min_shape=[1, 32, 5], opt_shape=[64000, 32, 5], max_shape=[256000, 32, 5])
```
(feature dim = **5** for the LiDAR+intensity model; the non-intensity model uses 4.)
This mirrors `projects/BEVFusion/deploy/utils.py` (`max_num_points 10 → 32`) and the
`*_tensorrt_dynamic.py` deploy configs in the 2.8 commit.

### 3.2 ONNX opset 17 → 18
2.8 bumped `opset_version` to **18** in
`projects/BEVFusion/configs/deploy/bevfusion_main_body_lidar_only*_tensorrt_dynamic.py`.
Applied to `config/deploy_config.py` (`onnx_config.opset_version = 18`).

### 3.3 What did NOT need porting
- **`purge_mmdeploy_symbolics(["layer_norm"])`** (added to `projects/BEVFusion/deploy/exporter.py`
  in 2.8): only relevant to the **old** mmdeploy `RewriterContext` export path. The new
  framework uses plain `torch.onnx.export` (`_torch_onnx_export_module`), so no mmdeploy
  layer_norm symbolic is registered and `LayerNorm` exports natively. No-op here.
- **`_fix_topk`** (constant-K hardening so TensorRT gets a static `K`): still valid for the
  2.8 native `torch.topk` node — kept as is.

---

## 4. The two `unknown` rows in the eval summary (`traffic_cone`, `barrier`)

The eval summary prints `car, truck, bus, bicycle, pedestrian, unknown, unknown`. The two
`unknown` rows **are** `traffic_cone` and `barrier`.

Cause (not a deployment bug): `perception_eval`'s `AutowareLabel` enum has only
`{UNKNOWN, CAR, TRUCK, BUS, BICYCLE, MOTORBIKE, PEDESTRIAN, ANIMAL, FP, LABEL_TYPE}`. Its
label table maps `movable_object.traffic_cone → UNKNOWN` and `movable_object.barrier →
UNKNOWN`. With `label_prefix="autoware"` (set by the model config's
`evaluation_config_dict`), both classes collapse to `UNKNOWN`, so they cannot be scored
separately. This is identical to what training-time `T4MetricV2` does in this container — it
is an evaluator/label-set limitation, not specific to deployment. The mAP for the 5 supported
classes is valid (e.g. car 0.95 / truck 0.97 / bus 0.99 / bicycle 0.99 / pedestrian 0.78).

To score `traffic_cone`/`barrier` separately you would need a `perception_eval` label set
that includes them (or a non-autoware `label_prefix`); that is a model-repo / evaluator
change, out of scope for the deploy framework.

---

## 5. Other framework-migration fixes (context)

The `init migration` commit added `deployment/` but did not port the model-side deploy hooks
into `projects/`, and renamed some config APIs. To run 2.8 these were also needed:

- `projects/SparseConvolution/sparse_functional.py` — `set_do_sort` / implicit-gemm
  activation+bias fusion / `do_sort_i` export attr (the entrypoint imports `set_do_sort`).
- `projects/BEVFusion/bevfusion/bevfusion.py` — added `_align_lidar_bev_to_head_grid`
  (the dense export wrapper calls it to assert BEV grid == head grid). **Only** this method
  was added; the rest of the 2.8 `bevfusion.py` is unchanged.
- `deployment/projects/bevfusion_l/export/onnx_export_pipeline.py` —
  `config.onnx_config` → `config.deploy_cfg.get("onnx_config", {})` (new framework keeps
  `_onnx_config` private; only `get_onnx_settings(component)` is public).
- deploy config `runtime_io.info_file` → an existing `.pkl`
  (`info/t4dataset_j6gen2_base_infos_test.pkl`).
