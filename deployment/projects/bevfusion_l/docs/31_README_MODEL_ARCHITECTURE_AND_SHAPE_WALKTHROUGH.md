# BEVFusion-L: End-to-End Model Architecture & Shape Walkthrough

> **What this document is.** A complete, first-hand trace of the **BEVFusion-L** (LiDAR-only)
> 3D-detection model as it is deployed by the `bevfusion_l` bundle. Every shape and node fact below
> was captured by actually running **one point-cloud sample end-to-end** — through both the native
> **PyTorch** path and the built **TensorRT** engines — inside the `bevfusion-deployment:latest`
> container, and by parsing the exported ONNX graphs. It documents, per stage: the exact input/output
> tensor shapes, what every PyTorch module does, and what every ONNX node does.

---

## 0. How this was produced (reproducible)

```bash
# Container: bevfusion-deployment:latest (id 173f7f9e…), repo bind-mounted at /workspace,
# data at /workspace/data, spconv plugin at /opt/plugins/libautoware_tensorrt_plugins.so
# The deployment run that built the ONNX + engines:
python -m deployment.cli.main bevfusion_l \
    deployment/projects/bevfusion_l/config/deploy_config.py \
    projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_8xb16_j6gen2_base_120m_t4metric_v2.py
```

Artifacts consumed by this walkthrough (already built):

```
work_dirs/bevfusion_deployment_2_8/
├── onnx/
│   ├── bevfusion_sparse.onnx           # 138 nodes  — voxel encoder + sparse 3D conv → dense BEV
│   ├── bevfusion_dense.onnx            # 242 nodes  — SECOND + FPN + TransFusion head
│   └── bevfusion_lidar_fp16_opt.onnx   # 380 nodes  — merged full graph (= 138 + 242)
└── tensorrt/
    ├── bevfusion_sparse.engine
    ├── bevfusion_dense.engine
    └── bevfusion_lidar_fp16_opt.engine # merged; this is what the TRT backend actually runs
```

The trace/instrumentation scripts used are described in [§11](#11-how-to-reproduce-the-trace). The
sample used is **dataset index 0** of the j6gen2 test split.

---

## 1. TL;DR — the whole pipeline on one sample

```
raw LiDAR points        [460528, 5]          (x, y, z, intensity, time_lag)
      │  (A) voxelization  — hard voxelization, OUTSIDE the graph
      ▼
voxels                  [70747, 32, 5]       up to 32 points/voxel
coors                   [70747, 3]           voxel index (z,y,x at the graph boundary)
num_points_per_voxel    [70747]
      │  ══════════ bevfusion_sparse (ONNX / TRT engine) ══════════
      │  (B) voxel encoder (mean-pool + sin/cos Fourier) → [70747, 50]
      │  (C) sparse 3D conv encoder (spconv, 21 ImplicitGemm layers)
      │  (D) scatter sparse → dense, collapse Z into channels
      ▼
lidar_bev               [1, 256, 180, 180]   dense BEV feature map
      │  ══════════ bevfusion_dense (ONNX / TRT engine) ══════════
      │  (E) SECOND backbone → (F) SECONDFPN neck → [1, 512, 180, 180]
      │  (G) TransFusion head: heatmap → top-500 query selection →
      │      1 transformer decoder layer → per-query regression + scoring
      ▼
bbox_pred [10, 500]   score [500]   label_pred [500]     (500 proposals)
      │  (H) bbox decode + circle-NMS  — OUTSIDE the graph
      ▼
78 detections (PyTorch FP32)  /  77 detections (TRT FP16)     for sample 0
```

Two boundaries are **outside** the exported graph and live in the Python pipeline:
**(A) voxelization** (`preprocess`) and **(H) box decoding + NMS** (`postprocess`). Everything between
`voxels` and `(bbox_pred, score, label_pred)` is the neural network, exported as ONNX/TensorRT.

---

## 2. Model configuration (resolved values)

Resolved by following the `_base_` chain from the model config used on the CLI. Sources:
`projects/BEVFusion/configs/t4dataset/default/pipelines/default_lidar_intensity_120m.py`,
`.../default/models/default_lidar_second_secfpn_120m.py`, and the `_120m.py` base.

| Parameter | Value | Meaning |
|---|---|---|
| `point_cloud_range` | `[-122.4, -122.4, -3.0, 122.4, 122.4, 5.0]` | x/y/z min–max (metres); 120 m range |
| `voxel_size` | `[0.17, 0.17, 0.2]` | metres per voxel (x, y, z) |
| `grid_size` (`sparse_shape`) | `[1440, 1440, 41]` | voxels along x, y, z = range/voxel_size |
| `sparse_dense_output_shapes` | `[180, 180, 2]` | dense grid after sparse encoder (x, y, z) |
| `out_size_factor` | `8` | BEV feature stride: 1440 / 8 = **180** |
| `max_num_points` | `32` | max points per voxel (hard voxelization) |
| `max_voxels` | `[120000, 160000]` | cap (train / test); actual this frame = 70747 |
| `point feature dim` | `5` | (x, y, z, intensity, time_lag) |
| `class_names` | `[car, truck, bus, bicycle, pedestrian, traffic_cone, barrier]` | **7 classes** |
| `num_proposals` | `500` | queries selected by TopK → detections/frame |
| `num_decoder_layers` | `1` | transformer decoder depth |
| `hidden_channel` | `128` | decoder/query embedding dim |
| `num_heads` | `8` | attention heads (128 / 8 = 16 dims/head) |

**Full model definition** (`default_lidar_second_secfpn_120m.py`):

```python
model = dict(
  type="BEVFusion",
  pts_voxel_encoder = HardSimpleVoxelSinCosEncoder(in_channels=5)      # → 5*5*2 = 50 ch
  pts_middle_encoder = BEVFusionSparseEncoder(
      in_channels=50, sparse_shape=[1440,1440,41],
      encoder_channels=((16,16,32),(32,32,64),(64,64,128),(128,128)),
      encoder_paddings=((0,0,1),(0,0,1),(0,0,(1,1,0)),(0,0)),
      block_type="basicblock")
  pts_backbone = SECOND(in_channels=256, out_channels=[128,256],
                        layer_nums=[5,5], layer_strides=[1,2])
  pts_neck    = SECONDFPN(in_channels=[128,256], out_channels=[256,256],
                          upsample_strides=[1,2])                       # concat → 512 ch
  bbox_head   = BEVFusionHead(in_channels=512, hidden_channel=128,
                        num_proposals=500, num_decoder_layers=1,
                        nms_kernel_size=3, num_heads=8,
                        common_heads=dict(center=[2,2], height=[1,2],
                                          dim=[3,2], rot=[2,2], vel=[2,2]),
                        bbox_coder=TransFusionBBoxCoder(out_size_factor=8, code_size=10,
                                          score_threshold=[.015,.010,.010,.020,.030,.040,.020]),
                        test_cfg=dict(nms_type="circle", nms_clusters=[...]))
)
```

---

## 3. Stage A — Voxelization (`preprocess`, outside the graph)

`BEVFusionInferencePipeline.preprocess` runs the model's own hard-voxelization layer
(`pts_voxel_layer`) on the CPU/GPU point tensor. It is **not** part of the ONNX graph — the graph's
first input is `voxels`.

| Tensor | Shape (sample 0) | dtype | Notes |
|---|---|---|---|
| `points` (input) | `[460528, 5]` | float32 | (x, y, z, intensity, time_lag) |
| `voxels` | `[70747, 32, 5]` | float32 | per-voxel point buffer, zero-padded to 32 |
| `coors` | `[70747, 3]` | int32 | voxel grid index. Model-internal is `[x,y,z]`; the **graph input contract is `[z,y,x]`** |
| `num_points_per_voxel` | `[70747]` | int32 | valid points per voxel (1…32) |

> **Coordinate contract.** At the ONNX/TRT boundary `coors` is `[z, y, x]` (no batch column). The
> wrappers flip it back to `[x, y, z]` and prepend a batch column before spconv — see
> `io/voxel_inputs.py` and `export/onnx_models/bevfusion_onnx.py::normalize_sparse_coors_for_autoware`.
> `num_points_per_voxel` is clamped to `>= 1` before the mean-pool so empty voxels never divide by zero
> (a NaN there poisons the whole dense BEV).

---

## 4. Stage B–D — Sparse branch (`bevfusion_sparse`)

**Component:** `bevfusion_sparse.onnx` (138 nodes) / `bevfusion_sparse.engine`.
**Signature:** `(voxels, coors, num_points_per_voxel) → lidar_bev [1,256,180,180]`.

ONNX op histogram: `GetIndicePairsImplicitGemm ×21`, `ImplicitGemm ×21` (the spconv plugin pairs),
`Add ×12`, `Relu ×8`, `Constant ×32`, plus the voxel-encoder (`ReduceSum, Div, Mul, Add, Cos, Sin,
Concat`) and the scatter-to-dense tail (`ScatterElements, Reshape, Transpose`).

### 4.1 Voxel encoder — `HardSimpleVoxelSinCosEncoder`  (nodes 15–32)

Source: `projects/BEVFusion/bevfusion/bevfusion_voxel_encoder.py`. What it computes:

1. **Mean-pool** each voxel over its valid points: `[70747, 32, 5] → [70747, 5]`
   (ONNX `ReduceSum` over the 32-point axis, then `Div` by `num_points_per_voxel`).
2. **Min–max normalize + Fourier expand.** For each of the 5 channels `j` and each frequency
   exponent `i ∈ {0..4}`, form `y[n,i,j] = (mean[n,j]−min_j)/(max_j−min_j) · π · 2^i`.
   This is folded into a single fused multiply-add `y = bias + scale · mean` (ONNX `Mul`+`Add`),
   giving a `[70747, 5, 5]` tensor reshaped to `[70747, 25]`.
   Normalization ranges: `min=[−122.4,−122.4,−3,0,0]`, `max=[122.4,122.4,5,255,0.2]`.
3. **sin/cos** concatenation: `concat(cos(y), sin(y)) → [70747, 50]` (ONNX `Cos`, `Sin`, `Concat`).

Output: **`voxel_features [70747, 50]`** — this is why the sparse encoder's `in_channels = 50`.

### 4.2 Sparse 3D encoder — `BEVFusionSparseEncoder`  (nodes 33–91)

Submanifold/regular sparse 3D convolutions (spconv), each realized in ONNX as a
`GetIndicePairsImplicitGemm` (builds the gather/scatter rulebook) + `ImplicitGemm` (the actual
conv, with the trailing ReLU **fused into the plugin's `act_type`** — that is why there are only 8
explicit `Relu` nodes, not 21). Captured spconv tensor shapes `(features, spatial [x,y,z])`:

| Module | Feature channels | Sparse spatial `[x, y, z]` | Active voxels |
|---|---|---|---|
| `conv_input` (SubMConv3d) | 16 | `[1440, 1440, 41]` | 70747 |
| `encoder_layer1` (basicblock ×2 + downsample) | 32 | `[720, 720, 21]` | 63710 |
| `encoder_layer2` | 64 | `[360, 360, 11]` | 31472 |
| `encoder_layer3` | 128 | `[180, 180, 5]` | 12557 |
| `encoder_layer4` (no downsample) | 128 | `[180, 180, 5]` | 12557 |
| `conv_out` (SparseConv3d, z-stride) | 128 | `[180, 180, 2]` | 9266 |

Each `encoder_layerN.k` is a **basicblock**: `conv1 → conv2 → Add(residual) → ReLU` (visible in the
ONNX as the `ImplicitGemm, ImplicitGemm, Add, Relu` quartets). Layers 1–3 end with a stride-2
downsampling sparse conv (`encoder_layerN.2`), halving x/y/z. `conv_out` compresses Z from 5→2.

### 4.3 Scatter to dense BEV  (nodes 92–137)

The final sparse tensor from `conv_out` (9266 active voxels, 128 ch, grid `[180,180,2]`) must be
turned into a dense `[1,256,180,180]` map for the 2-D backbone. BEVFusion does this with an
**explicit scatter** (`sparse_to_dense` in
`projects/BEVFusion/bevfusion/custom_sparse_conv_tensor.py`) rather than spconv's built-in `.dense()`
— the file header says this exists specifically "to support cleaner ONNX export". The single
`ScatterElements` node in the sparse graph comes from exactly one line of that helper:
`out.scatter(0, scatter_idx, features)` (`torch.Tensor.scatter(dim, index, src)` lowers to ONNX
`ScatterElements`).

Source → ONNX node mapping (sparse graph nodes 95–137):

```python
b, h, w, d = idx.unbind(1)                       # Split + Squeeze×4                 (95, 97–103)
linear_idx = ((b*H + h)*W + w)*D + d             # flatten (b,h,w,d) → 1-D: Mul/Add  (104–112)
out = torch.zeros([num_cells, 128])              # zero dense buffer: ConstantOfShape (120)
                                                 #   num_cells = 1·180·180·2 = 64800
scatter_idx = linear_idx.unsqueeze(1).expand(-1, 128)  # Unsqueeze + Expand          (122, 131)
out = out.scatter(0, scatter_idx, features)      # ★ ScatterElements (dim=0)          (132)
out.view(1, 180, 180, 2, 128)                    # Reshape                            (134)
# then in sparse_encoder.py forward:
.permute(0, 4, 3, 1, 2)                          # Transpose → [1,128,2,180,180]      (135)
.view(1, 256, 180, 180)                          # Reshape → lidar_bev                (137)
```

So `ScatterElements` = **write each of the 9266 active voxels' 128-dim feature vector into row
`linear_idx` of a zeroed `[64800, 128]` dense table** (`64800 = 1×180×180×2` cells); empty cells stay
0. It is `ScatterElements` (not `ScatterND`) precisely because the code uses element-wise
`Tensor.scatter` along `dim=0`, a deliberate choice that keeps the graph clean and the node count
stable. This step lives **inside** the sparse graph — `lidar_bev` is its output.

Finally **Z is folded into the channel dimension** (128 channels × 2 Z-slices = 256):

```
Reshape   → [1, 180, 180, 2, 128]
Transpose → [1, 128, 2, 180, 180]
Reshape   → [1, 256, 180, 180]
```

Output: **`lidar_bev [1, 256, 180, 180]`** (float32). PyTorch↔TRT parity on this tensor is within
FP16 tolerance.

---

## 5. Stage E–G — Dense branch (`bevfusion_dense`)

**Component:** `bevfusion_dense.onnx` (242 nodes) / `bevfusion_dense.engine`.
**Signature:** `lidar_bev [1,256,180,180] → (bbox_pred [10,500], score [500], label_pred [500])`.

### 5.1 SECOND backbone  (ONNX nodes 0–91, interleaved)

`pts_backbone = SECOND`, two blocks of 5 conv layers each:

| Block | Stride | Output |
|---|---|---|
| `blocks.0` | 1 | `[1, 128, 180, 180]` |
| `blocks.1` | 2 | `[1, 256, 90, 90]` |

### 5.2 SECONDFPN neck  (nodes 79–95)

`pts_neck = SECONDFPN` upsamples both scales back to 180×180 and concatenates:

| Deblock | Op | Output |
|---|---|---|
| `deblocks.0` | conv (stride-1) | `[1, 256, 180, 180]` |
| `deblocks.1` | `ConvTranspose` (×2 up) | `[1, 256, 180, 180]` |
| `Concat` | channel concat | **`[1, 512, 180, 180]`** |

`_align_lidar_bev_to_head_grid` is a no-op here (neck already at the 180×180 head grid).

### 5.3 BEVFusion (TransFusion) head — the interesting part

The head turns the `[1,512,180,180]` BEV map into 500 object queries and regresses a box per query.

**(a) Shared conv + dense heatmap** (nodes 96–105):
`shared_conv: 512→128` → `[1,128,180,180]`; `heatmap_head` (Conv-BN-ReLU + Conv) →
`dense_heatmap [1, 7, 180, 180]` → `Sigmoid`. 7 = number of classes.

**(b) Heatmap local-max NMS + TopK query selection** (nodes 106–140):
- A `MaxPool` (kernel `nms_kernel_size=3`) + `Equal`/`Where` keeps only local-peak heatmap cells
  (suppresses neighbours). This local-max pooling is applied **only to the "crowded" classes**
  `dense_heatmap_pooling_classes = [car, truck, bus, barrier]`; the other classes pass through
  un-pooled (see `bevfusion_head.py`).
- Heatmap reshaped `[1,7,180,180] → [1,7,32400] → [1, 226800]` (flatten class × position).
- `TopK` selects the **500** highest scores across *all* classes and positions →
  indices `[1,500]`. **The `K=500` constant is baked in** (a post-export transform pins it, see
  `export/transforms.py::fix_topk_constant_k`).
- `Div` / `Mod` by 32400 decode each flat index into `(class_id, bev_position)`:
  `query_labels [1,500]` and the position used to gather the query's BEV feature and 2-D position.

**(c) Query & positional embeddings** (nodes 130–152):
- `query_feat`: gather the 128-ch BEV feature at each of the 500 positions → `[1,128,500]`.
- `query_pos`: the (x,y) BEV coordinate of each query → embedded by `self_posembed`
  (`PositionEncodingLearned`, Conv1d-BN-ReLU-Conv1d) → `[1,128,500]`.
- `class_encoding`: one-hot class → Conv1d → added into the query feature.
- The **whole BEV** (32400 positions) is embedded once by `cross_posembed` → `[1,128,32400]` to serve
  as the cross-attention key positions.

**(d) Transformer decoder — 1 layer** (`decoder.0`, nodes 153–204):

| Sub-block | ONNX ops | Query tensor |
|---|---|---|
| **Self-attention** (queries↔queries) | `MatMul` Q/K/V, `Mul`(scale), `MatMul`+`Softmax` (8 heads, `[8,500,500]`), `Gemm` out-proj | `[1,500,128]` |
| `norms.0` | `LayerNormalization` | `[1,500,128]` |
| **Cross-attention** (queries↔BEV) | `MatMul` Q/K/V, `Softmax` `[8,500,32400]`, `Gemm` | `[1,500,128]` |
| `norms.1` | `LayerNormalization` | `[1,500,128]` |
| **FFN** | `MatMul 128→256 → ReLU → MatMul 256→128` | `[1,500,128]` |
| `norms.2` | `LayerNormalization` | `[1,500,128]` |

Cross-attention is where each object query reads the BEV features (all 32400 cells) to refine itself.

**(e) Prediction heads — `SeparateHead`** (nodes 205–239): the `[1,128,500]` decoded queries pass
through six Conv1d(128→64→out) branches (`common_heads`), each producing a per-query regression:

| Head | Out channels | Meaning |
|---|---|---|
| `center` | 2 | BEV offset (feature-map units); **`+= query_pos`** (node 224) |
| `height` | 1 | z of gravity centre (metres) |
| `dim` | 3 | log box size (dx, dy, dz) |
| `rot` | 2 | (sin θ, cos θ) |
| `vel` | 2 | (vx, vy) |
| `heatmap` | 7 | per-query class logits |

`Concat` (node 239) stacks center+height+dim+rot+vel → **`bbox_pred [10, 500]`**.

**(f) Scoring** (nodes 137–241, the output contract in `io/head_outputs.py`):

```python
score   = sigmoid(heatmap) * query_heatmap_score * one_hot(query_labels)   # [1,7,500]
score   = score.max(over classes)                                          # → score [500]
label   = query_labels[0]                                                  # → label_pred [500]
```
Implemented in ONNX by `Sigmoid`, `OneHot`, `GatherElements`, `Mul`, `ReduceMax`, `Gather`.

**Dense outputs (sample 0):** `bbox_pred [10,500] f32`, `score [500] f32`, `label_pred [500] i64`.

---

## 6. Stage H — Postprocess (`postprocess`, outside the graph)

`BEVFusionInferencePipeline.postprocess` reproduces the reference-eval selection so PyTorch/TRT match
`test.py`. It calls the model's own `TransFusionBBoxCoder.decode(filter=True)`
(`projects/BEVFusion/bevfusion/utils.py`):

```
center_x_metric = center_x_feat · out_size_factor(8) · voxel_size_x(0.17) + pc_range_x(−122.4)
center_y_metric = center_y_feat · 8 · 0.17 + (−122.4)
dim             = exp(dim_log)                       # log-size → metres
z_bottom        = height − dim_z · 0.5               # gravity centre → bottom centre
yaw             = atan2(rot_sin, rot_cos)
box             = [x, y, z, dx, dy, dz, yaw, vx, vy]
```

Then per-class **score thresholds** (`[.015,.010,.010,.020,.030,.040,.020]`), a `post_center_range`
filter, and **circle-NMS by cluster** (`apply_cluster_nms`; car/truck/bus radius 0.25, others 0.0)
prune the 500 proposals to the final detections.

**Result for sample 0:** PyTorch → **78** detections, TRT (merged FP16) → **77** detections. Example
(PyTorch): `label 0 (car)`, box `[-8.5, -7.74, 0.05, 4.44, 1.71, 1.56, -0.017, …]`, score 0.92.

---

## 7. ONNX graphs at a glance

| Graph | Nodes | Initializers | Inputs → Outputs |
|---|---|---|---|
| `bevfusion_sparse` | **138** | — | `voxels[N,32,5], coors[N,3], num_points[N]` → `lidar_bev[1,256,180,180]` |
| `bevfusion_dense` | **242** | 102 | `lidar_bev[1,256,180,180]` → `bbox_pred[10,500], score[500], label_pred[500]` |
| `bevfusion_lidar_fp16_opt` (merged) | **380** | 144 | `voxels/coors/num_points` → `bbox_pred, score, label_pred` |

The merged graph is exactly `138 + 242 = 380` nodes — it is **composed from the split pair**
post-export (the `bevfusion_merge` finalize hook wires `sparse.lidar_bev → dense.lidar_bev`), not
re-traced. The split pair exists so the sparse (spconv-plugin) and dense parts can be built/profiled
independently; the merged engine is what the TRT backend runs by default.

`opset 18`, `do_constant_folding=True`. The static `lidar_bev` shape `[1,256,180,180]` lets constant
folding drop the head's dynamic shape-glue so the split and merged node counts line up.

---

## 8. TensorRT engines (bindings)

Precision **FP16**; the spconv plugin `libautoware_tensorrt_plugins.so` is loaded before
deserialize. Captured bindings:

**`bevfusion_sparse.engine`**
| I/O | name | shape | dtype |
|---|---|---|---|
| in | `voxels` | `[-1, 32, 5]` | FLOAT |
| in | `coors` | `[-1, 3]` | INT32 |
| in | `num_points_per_voxel` | `[-1]` | INT32 |
| out | `lidar_bev` | `[1, 256, 180, 180]` | FLOAT |

Dynamic voxel-count profile (`voxels`): min `1`, opt `64000`, max `256000`.

**`bevfusion_dense.engine`**: in `lidar_bev [1,256,180,180]` → out `bbox_pred [10,500] f32`,
`score [500] f32`, `label_pred [500] i64`.

**`bevfusion_lidar_fp16_opt.engine`** (merged): in `voxels/coors/num_points` → out
`bbox_pred/score/label_pred` (same as dense).

> ONNXRuntime is **not** a runtime backend for BEVFusion — the sparse graph needs the TRT-only
> `autoware` spconv plugins, so ONNX is an export/interchange format only.

---

## 9. PyTorch module execution trace (captured, sample 0)

Abridged from a forward hook on every submodule (211 modules total), in execution order. Shapes are
the live values for this frame.

```
pts_voxel_encoder            HardSimpleVoxelSinCosEncoder   [70747,32,5] → [70747, 50]

pts_middle_encoder           BEVFusionSparseEncoder
  conv_input.0  SubMConv3d      → sparse(feat[70747,16],  grid[1440,1440,41])
  encoder_layer1               → sparse(feat[63710,32],  grid[720,720,21])
  encoder_layer2               → sparse(feat[31472,64],  grid[360,360,11])
  encoder_layer3               → sparse(feat[12557,128], grid[180,180,5])
  encoder_layer4               → sparse(feat[12557,128], grid[180,180,5])
  conv_out.0    SparseConv3d    → sparse(feat[9266,128],  grid[180,180,2])
  (scatter→dense)              → [1, 256, 180, 180]

pts_backbone   SECOND
  blocks.0                     → [1, 128, 180, 180]
  blocks.1                     → [1, 256,  90,  90]
pts_neck       SECONDFPN
  deblocks.0                   → [1, 256, 180, 180]
  deblocks.1 (ConvTranspose)   → [1, 256, 180, 180]
  (concat)                     → [1, 512, 180, 180]

bbox_head      BEVFusionHead
  shared_conv                  → [1, 128, 180, 180]
  heatmap_head                 → [1,   7, 180, 180]   (dense_heatmap)
  class_encoding  Conv1d       → [1, 128, 500]
  decoder.0  TransformerDecoderLayer
    self_posembed              → [1, 128,   500]
    cross_posembed             → [1, 128, 32400]
    self_attn  (8 heads)       attn map [1,500,500]   → [1, 500, 128]
    cross_attn (8 heads)       attn map [1,500,32400] → [1, 500, 128]
    ffn (128→256→128)          → [1, 500, 128]
  prediction_heads.0  SeparateHead
    center  → [1,2,500]  height → [1,1,500]  dim → [1,3,500]
    rot     → [1,2,500]  vel    → [1,2,500]  heatmap → [1,7,500]
  → (bbox_pred [10,500], score [500], label_pred [500])
```

---

## 10. Where each concern lives (code map)

| Concern | File |
|---|---|
| Voxelize / decode / NMS (outside graph) | `inference/bevfusion_inference_pipeline.py` |
| PyTorch backend seams | `inference/pytorch_inference_pipeline.py` |
| TensorRT backend (split/merged) | `inference/tensorrt_inference_pipeline.py` |
| ONNX export wrappers (what each graph computes) | `export/onnx_models/bevfusion_onnx.py` |
| Split→dense component definitions | `export/component_builder.py` |
| Split→merge ONNX finalize | `export/transforms.py` |
| spconv ReLU→ImplicitGemm fusion | `export/onnx_fuse_implicit_gemm_activation.py` |
| Head output contract (score/label) | `io/head_outputs.py` |
| Voxel-input coordinate contract (`[z,y,x]`↔`[x,y,z]`) | `io/voxel_inputs.py` |
| Head-output triple → detection dicts | `io/head_outputs.py`, pipeline `postprocess` |
| Voxel encoder (Fourier) | `projects/BEVFusion/bevfusion/bevfusion_voxel_encoder.py` |
| bbox decode | `projects/BEVFusion/bevfusion/utils.py` |
| Model + backbone/neck/head config | `projects/BEVFusion/configs/t4dataset/default/models/default_lidar_second_secfpn_120m.py` |
| Range/voxel/grid config | `projects/BEVFusion/configs/t4dataset/default/pipelines/default_lidar_intensity_120m.py` |

---

## 11. How to reproduce the trace

Three small scripts (run inside the container from `/workspace`) produced every number above:

- **`trace_bevfusion.py`** — builds the model + loads sample 0, registers a forward hook on every
  submodule, runs `preprocess → run_sparse_encoder → run_dense → postprocess`, and dumps every
  input/output shape to `trace_report.json`.
- **`trace_trt.py`** — deserializes the three engines and prints their I/O bindings
  (`trt_bindings.json`).
- **`trace_onnx.py`** — loads each ONNX, runs shape inference, and prints the op histogram + a
  node-by-node walk with inferred shapes (`onnx_analysis.json`).
- **`trace_merged.py`** — runs the merged FP16 engine end-to-end and reports the detection count.

```bash
docker exec awml-bevfusion bash -c 'cd /workspace && python _trace/trace_bevfusion.py'
docker exec awml-bevfusion bash -c 'cd /workspace && python _trace/trace_trt.py'
docker exec awml-bevfusion bash -c 'cd /workspace && python _trace/trace_onnx.py'
docker exec awml-bevfusion bash -c 'cd /workspace && python _trace/trace_merged.py'
```

---

### Appendix — key numbers for sample 0

| Quantity | Value |
|---|---|
| Input points | 460 528 |
| Voxels produced | 70 747 |
| BEV feature map | `[1, 256, 180, 180]` |
| Query proposals | 500 |
| Detections (PyTorch FP32) | 78 |
| Detections (TRT FP16 merged) | 77 |
| Sparse graph nodes | 138 |
| Dense graph nodes | 242 |
| Merged graph nodes | 380 |
