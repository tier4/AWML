# BEVFusion-L：操作 ↔ 程式碼對應表

> 本文件是 [`32_README_MODEL_ARCHITECTURE_detailed.md`](deployment/projects/bevfusion_l/docs/32_README_MODEL_ARCHITECTURE_detailed.md) 的**程式碼對照版**。
> 章節編號與 32 號文件一一對應，每個操作都標出實際檔案與行號，讓你能同時看到「數學/流程」與「真正跑的程式碼」。
>
> 程式碼分屬兩層：
> - **模型層（PyTorch model）**：`projects/BEVFusion/bevfusion/` — 定義 voxel encoder、sparse encoder、SECOND/FPN、TransFusion head、bbox coder。這是 train 與 eval 都會跑的原始模型。
> - **部署層（deployment）**：`deployment/projects/bevfusion_l/` — 把模型切成 `sparse` / `dense` 兩個可匯出的元件（ONNX/TensorRT），並在 graph 外做 voxelization 與 decode+NMS。
>
> **核心對應關係**（先記住這張圖，其餘都是細節）：
>
> | 文件流程 | 部署層 wrapper / pipeline | 模型層真正的運算 |
> |---|---|---|
> | Voxelization（graph 外） | `preprocess()` → `pts_voxel_layer` | `bevfusion.py: voxelize()` |
> | `bevfusion_sparse` 元件 | `BEVFusionSparseWrapper.forward` | `extract_pts_feat` = voxel_encoder + middle_encoder |
> | `bevfusion_dense` 元件 | `BEVFusionDenseWrapper.forward` | `pts_backbone` + `pts_neck` + `bbox_head` |
> | 輸出打包成 (bbox,score,label) | `head_dict_to_detection_outputs` | — |
> | Decode + threshold + NMS（graph 外） | `postprocess()` | `TransFusionBBoxCoder.decode` + `apply_cluster_nms` |

---

## §1 模型總覽 — 兩個可匯出元件的切分

整份「Raw Point Cloud → Final 3D Detections」流程，在部署層被切成兩個 wrapper，這是理解一切的骨架：

- Sparse 分支（voxels/coors/num_points → BEV feature map）：
  [`bevfusion_onnx.py:45-64`](deployment/projects/bevfusion_l/export/onnx_models/bevfusion_onnx.py#L45-L64) `BEVFusionSparseWrapper`
- Dense 分支（BEV feature map → bbox/score/label）：
  [`bevfusion_onnx.py:67-83`](deployment/projects/bevfusion_l/export/onnx_models/bevfusion_onnx.py#L67-L83) `BEVFusionDenseWrapper`

執行期（PyTorch / TensorRT）用同一條 pipeline 依序跑這兩段並各自計時：
[`bevfusion_inference_pipeline.py:101-143`](deployment/projects/bevfusion_l/inference/bevfusion_inference_pipeline.py#L101-L143) `run_model()`（`run_sparse_encoder` → `run_dense`）。

---

## §2–3 輸入點雲 & Voxelization（在 graph 之外）

文件重點：`points [460528,5]` → `voxels [70747,32,5]` / `coors [70747,3]` / `num_points_per_voxel [70747]`。

**部署層**：voxelization 刻意放在 ONNX graph 之外，由 pipeline 的 `preprocess()` 做：

[`bevfusion_inference_pipeline.py:67-99`](deployment/projects/bevfusion_l/inference/bevfusion_inference_pipeline.py#L67-L99)
```python
def preprocess(self, points):
    points_tensor = self.to_device_tensor(points).float()
    with torch.no_grad():
        voxel_output = self.pytorch_model.pts_voxel_layer(points_tensor)   # ← hard voxelization
        voxels, coors, num_points_per_voxel = voxel_output                 # 只支援 hard voxelization
    return {"voxels": voxels, "coors": coors, "num_points_per_voxel": num_points_per_voxel}
```

**模型層**：`pts_voxel_layer` 背後的實作在
[`bevfusion.py:262`](projects/BEVFusion/bevfusion/bevfusion.py#L262) `voxelize()`（`@torch.no_grad()`）。

> **文件對應的 §3 平均公式**（除以「實際有效點數」而非固定 32）就是下面 §4 voxel encoder 裡 `features.sum(dim=1) / num_points` 這一行 —— padding 位置是 0，但分母用 `num_points`。

`coors` 的座標軸順序（`[x,y,z]` ↔ graph 的 `[z,y,x]`）契約全部集中在：
[`voxel_inputs.py`](deployment/projects/bevfusion_l/io/voxel_inputs.py#L1-L72)（`graph_input_zyx_to_model_indices_xyz` 等），並在 sparse wrapper 進 spconv 前翻轉：
[`bevfusion_onnx.py:25-42`](deployment/projects/bevfusion_l/export/onnx_models/bevfusion_onnx.py#L25-L42) `normalize_sparse_coors_for_autoware()`。

---

## §4 Stage B：Voxel Feature Encoder（mean + Sin/Cos Fourier）

文件重點：`[70747,32,5] → [70747,5] → [70747,25] → [70747,50]`。

**模型層**：[`bevfusion_voxel_encoder.py:11-77`](projects/BEVFusion/bevfusion/bevfusion_voxel_encoder.py#L11-L77) `HardSimpleVoxelSinCosEncoder`。

§4.1 Mean pooling（分母是有效點數，對應文件 §3 的平均式）：
[`bevfusion_voxel_encoder.py:63-65`](projects/BEVFusion/bevfusion/bevfusion_voxel_encoder.py#L63-L65)
```python
voxel_mean_features = (features.sum(dim=1, keepdim=False)
                       / num_points.type_as(features).view(-1, 1)).contiguous()   # [N,32,5] -> [N,5]
```

§4.2 Sin/Cos Fourier encoding（`u_j·π·2^i` 被折疊成 `scale·x + bias`，一個 FMA 完成）：
- 常數預先算好：[`bevfusion_voxel_encoder.py:39-46`](projects/BEVFusion/bevfusion/bevfusion_voxel_encoder.py#L39-L46)
  ```python
  exponents = (2 ** torch.arange(0, self.in_channels)).float()   # 頻率 2^i, i∈{0..C-1}
  alpha = (torch.pi * exponents).unsqueeze(0)                    # π·2^i  ← 文件的 π 2^i
  scale = alpha / beta                                          # beta = max-min（normalize）
  bias  = -(alpha * min_norm_values.unsqueeze(1)) / beta
  ```
- 前向：[`bevfusion_voxel_encoder.py:69-74`](projects/BEVFusion/bevfusion/bevfusion_voxel_encoder.py#L69-L74)
  ```python
  y = torch.addcmul(self.exponent_bias, self.exponent_scale, voxel_mean_features.unsqueeze(-1))  # [N,C,C]
  y = y.reshape(-1, self.in_channels * self.in_channels)                                          # [N, C*C]=25
  voxel_fourier_features = torch.cat([torch.cos(y), torch.sin(y)], dim=1)                          # [N, C*C*2]=50
  ```

> 精確說明文件的「5 channels × 5 frequencies = 25」：程式碼是 `in_channels × in_channels`。頻率個數 = `in_channels`，此模型 `in_channels=5`（x,y,z,intensity,time_lag），所以 5×5=25，×2(cos,sin)=50。

---

## §5 Stage C：Sparse 3D Encoder

文件重點：active voxels 隨層數下降、xy 下採樣 8 倍、高度 41→2。

**模型層**：[`sparse_encoder.py:22`](projects/BEVFusion/bevfusion/sparse_encoder.py#L22) `BEVFusionSparseEncoder`，前向：
[`sparse_encoder.py:122-152`](projects/BEVFusion/bevfusion/sparse_encoder.py#L122-L152)
```python
input_sp_tensor = SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
x = self.conv_input(input_sp_tensor)          # conv_input: 50 -> 16 ch
for encoder_layer in self.encoder_layers:     # Layer 1..4：文件那張表的每一列
    x = encoder_layer(x)
    encode_features.append(x)
out = self.conv_out(encode_features[-1])       # conv_out: 高度 5 -> 2
```

§5.1 文件說的 `GetIndicePairsImplicitGemm` → `ImplicitGemm` 是 spconv 每一層在 TensorRT 中的拆解；匯出後還會把後面的 ReLU 折進 plugin 的 `act_type`：
[`onnx_fuse_implicit_gemm_activation.py`](deployment/projects/bevfusion_l/export/onnx_fuse_implicit_gemm_activation.py#L1)（`fuse_autoware_implicit_gemm_trailing_relu`）。
匯出前的 SparseConv+BN 融合：[`spconv_bn_fusion.py`](deployment/projects/bevfusion_l/export/spconv_bn_fusion.py#L1)。

**這一整段的入口**是 sparse wrapper：
[`bevfusion_onnx.py:52-64`](deployment/projects/bevfusion_l/export/onnx_models/bevfusion_onnx.py#L52-L64) → `self.mod.extract_pts_feat(...)` →
[`bevfusion.py:200-216`](projects/BEVFusion/bevfusion/bevfusion.py#L200-L216)
```python
def extract_pts_feat(self, feats, coords, sizes, points=None):
    ...
    feats = self.pts_voxel_encoder(feats, sizes, coords)   # §4
    x     = self.pts_middle_encoder(feats, coords, batch_size)  # §5 + §6
    return x
```

---

## §6 Stage D：Sparse → Dense BEV（ScatterElements + Z collapse）

文件重點：`[9266,128]` 稀疏 → dense `[64800,128]` → reshape/transpose → 把 Z=2 併進 channel → `[1,256,180,180]`。

**模型層**：就在 sparse encoder `forward` 的尾巴：
[`sparse_encoder.py:154-161`](projects/BEVFusion/bevfusion/sparse_encoder.py#L154-L161)
```python
spatial_features = sparse_to_dense(out, batch_size, self.dense_output_shapes, self.output_channels)  # scatter
spatial_features = spatial_features.permute(0, 4, 3, 1, 2).contiguous()                              # → [1,C,Z,H,W]
spatial_features = spatial_features.view(batch_size,
    self.output_channels * self.dense_output_shapes[2],   # 128 * 2 = 256 ← Z 併進 channel
    self.dense_output_shapes[0], self.dense_output_shapes[1])  # [1,256,180,180]
```
- `sparse_to_dense`（文件的 `ScatterElements`，空位補 0）定義於
  [`custom_sparse_conv_tensor.py`](projects/BEVFusion/bevfusion/custom_sparse_conv_tensor.py#L1)。

---

## §7–8 Stage E/F：SECOND 2D Backbone + SECONDFPN

文件重點：backbone 產生 `[1,128,180,180]` 與 `[1,256,90,90]`；FPN 上採樣後 concat 成 `[1,512,180,180]`。

**部署層**（dense wrapper 前半段）：
[`bevfusion_onnx.py:74-80`](deployment/projects/bevfusion_l/export/onnx_models/bevfusion_onnx.py#L74-L80)
```python
def forward(self, lidar_bev):
    x = lidar_bev
    if self.mod.pts_backbone is not None: x = self.mod.pts_backbone(x)   # §7 SECOND
    if self.mod.pts_neck    is not None: x = self.mod.pts_neck(x)        # §8 SECONDFPN
    x = self.mod._align_lidar_bev_to_head_grid(x)                        # 嚴格檢查 H,W == 180
```
**模型層**：`pts_backbone` / `pts_neck` 由 config 建立（SECOND / SECONDFPN，來自 mmdet3d）：
[`bevfusion.py:73-74`](projects/BEVFusion/bevfusion/bevfusion.py#L73-L74)。
grid 對齊檢查（確保 BEV 尺寸符合 head 的 `grid_size // out_size_factor = 1440//8 = 180`）：
[`bevfusion.py:218-259`](projects/BEVFusion/bevfusion/bevfusion.py#L218-L259) `_align_lidar_bev_to_head_grid`。

> 註：`projects/BEVFusion/bevfusion/bevfusion_necks.py` 內另有 `GeneralizedLSSFPN`（camera 分支用）；LiDAR-only 這條由 config 指定的 neck 為主，形狀契約以文件 §8 為準。

---

## §9–10 Detection Head 總覽 + Shared BEV Feature

Head 全部集中在一個 forward：
[`bevfusion_head.py:261-381`](projects/BEVFusion/bevfusion/bevfusion_head.py#L261-L381) `forward_single`（入口 `forward` 在
[`bevfusion_head.py:383-397`](projects/BEVFusion/bevfusion/bevfusion_head.py#L383-L397)）。

§10 Shared convolution `[1,512,180,180] → [1,128,180,180]`：
[`bevfusion_head.py:269`](projects/BEVFusion/bevfusion/bevfusion_head.py#L269)
```python
fusion_feat = self.shared_conv(inputs)
fusion_feat_flatten = fusion_feat.view(-1, self.share_conv_out_channels, self.spatial_dim)  # [B,128,32400]
```

---

## §11 Dense Heatmap

`[1,512..] → heatmap_head → [1,7,180,180]`，再 sigmoid：
[`bevfusion_head.py:280-282`](projects/BEVFusion/bevfusion/bevfusion_head.py#L280-L282)
```python
dense_heatmap = self.heatmap_head(fusion_feat.float())   # heatmap_head 定義於 :132
heatmap = dense_heatmap.detach().sigmoid()               # H = σ(logits)
```

---

## §12 Local-Max Filtering（只對 crowded classes）

3×3 max-pool 保留局部極大，且只對特定類別做（文件的 car/truck/bus/barrier）：
[`bevfusion_head.py:283-314`](projects/BEVFusion/bevfusion/bevfusion_head.py#L283-L314)
```python
if self.dense_heatmap_pooling_class_indices is not None:
    selected_heatmap = heatmap[:, self.dense_heatmap_pooling_class_indices, :, :]
    local_max_inner = F.max_pool2d(selected_heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
    local_max = F.pad(local_max_inner, (pad,pad,pad,pad), value=0.0)      # 還原空間尺寸
    if self.dense_heatmap_exclude_pooling_classes:                       # 不做 pooling 的類別直接放回
        excluded_local_max = heatmap[:, self.dense_heatmap_exclude_pooling_classes, :, :]
        local_max = torch.cat([local_max, excluded_local_max], dim=1)[:, self.local_concat_class_remapping, :, :]
else:
    local_max = heatmap
heatmap = heatmap * (heatmap == local_max)   # 只留 H==M 的 peak（文件的 H_peak）
```

---

## §13 Flatten Heatmap `[1,7,180,180] → [1,226800]`

[`bevfusion_head.py:316-319`](projects/BEVFusion/bevfusion/bevfusion_head.py#L316-L319)
```python
heatmap = heatmap.view(-1, self.num_classes, self.spatial_dim)          # [1,7,32400]
flattened_heatmap = heatmap.view(-1, self.num_classes * self.spatial_dim)  # [1, 7*32400=226800]
```

---

## §14 Top-500 Query Selection + index 反解 class/position

[`bevfusion_head.py:322-327`](projects/BEVFusion/bevfusion/bevfusion_head.py#L322-L327)
```python
_, top_proposals = torch.topk(flattened_heatmap, k=self.num_proposals, dim=-1, largest=True, sorted=False)
top_proposals_class = top_proposals // self.spatial_dim   # ← 文件 i // 32400
top_proposals_index = top_proposals %  self.spatial_dim   # ← 文件 i %  32400（BEV 位置）
```
（`position id → (x,y)` 的 `x=p%180, y=p//180` 反解對應到後面 `bev_pos` 的索引。）

---

## §15 建立 Object Query（gather BEV feature）

從 shared feature 對 500 個位置 gather 出 128 維 query content：
[`bevfusion_head.py:328-332`](projects/BEVFusion/bevfusion/bevfusion_head.py#L328-L332)
```python
query_feat = fusion_feat_flatten.gather(
    index=top_proposals_index[:, None, :].expand(-1, self.share_conv_out_channels, -1), dim=-1)  # [B,128,500]
self.query_labels = top_proposals_class
```

---

## §16 Query Position Embedding & §17 Class Embedding

§17 Class embedding（one-hot → 1×1 conv → 加到 query content）：
[`bevfusion_head.py:335-337`](projects/BEVFusion/bevfusion/bevfusion_head.py#L335-L337)
```python
one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1)
query_cat_encoding = self.class_encoding(one_hot.float())   # class_encoding = nn.Conv1d(num_classes,128,1) :133
query_feat += query_cat_encoding
```
§16 Query position（grid 座標，尚未轉 metric）：
[`bevfusion_head.py:340`](projects/BEVFusion/bevfusion/bevfusion_head.py#L340)
```python
query_pos = self.bev_pos.squeeze(0)[top_proposals_index]   # 取 500 個位置的 (x,y)
```
> 文件 §16 的「learned position embedding `φ_self-pos`」實作在 decoder layer 內的 `self_posembed`（見 §20），`bev_pos` 只是提供每個 query 的 (x,y) grid 座標。

---

## §18–25 Transformer Decoder（Self-Attn + Cross-Attn + FFN）

文件 §18 的「BEV memory」= 完整 `fusion_feat_flatten` 當 key/value；§19–25 的 self/cross/FFN 都在一個 decoder layer 裡：

驅動迴圈（`num_decoder_layers=1`）：
[`bevfusion_head.py:345-358`](projects/BEVFusion/bevfusion/bevfusion_head.py#L345-L358)
```python
for i in range(self.num_decoder_layers):
    query_feat = self.decoder[i](
        query_feat, key=fusion_feat_flatten,   # ← key/value = 32400 個 BEV token（§18 memory）
        query_pos=query_pos, key_pos=self.bev_pos)   # §16 self-pos / §18.1 cross-pos
    res_layer = self.prediction_heads[i](query_feat)          # §26 separate heads
    res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1)  # §27 center += query_pos
```

Decoder layer 定義（Self-Attn / Cross-Attn 就在這）：
[`transformer.py:26-104`](projects/BEVFusion/bevfusion/transformer.py#L26-L104) `TransformerDecoderLayer`
```python
self.self_posembed  = PositionEncodingLearned(**pos_encoding_cfg)   # §16  φ_self-pos
self.cross_posembed = PositionEncodingLearned(**pos_encoding_cfg)   # §18.1 φ_cross-pos
...
query = self.self_attn(...)    # §20 self-attention（500×500）
query = self.cross_attn(...)   # §21 cross-attention（500×32400）
# 之後 FFN + residual + LayerNorm 由基底 DetrTransformerDecoderLayer 提供  # §23,§24
```
- position embedding MLP：[`transformer.py:7-24`](projects/BEVFusion/bevfusion/transformer.py#L7-L24) `PositionEncodingLearned`。
- §23 residual + LayerNorm、§24 FFN(128→256→128)：繼承自 mmdet 的 `DetrTransformerDecoderLayer`。

---

## §26 Separate Prediction Heads

每個 branch 是 Conv1d→ReLU→Conv1d，輸出 center/height/dim/rot/vel/heatmap：
[`bevfusion_head.py:353`](projects/BEVFusion/bevfusion/bevfusion_head.py#L353) `self.prediction_heads[i](query_feat)`
（`prediction_heads` 在 `__init__` 依 `common_heads` 建立）。
§14 選到的 proposal 分數（`query_heatmap_score`）在此回填：
[`bevfusion_head.py:360-363`](projects/BEVFusion/bevfusion/bevfusion_head.py#L360-L363)。

---

## §27–28 Center 修正 & Score 計算

§27 center 加回 query position：見上 [`bevfusion_head.py:354`](projects/BEVFusion/bevfusion/bevfusion_head.py#L354)。

§28 最終分數 `s_query · s_proposal · one_hot`，取 max —— 部署層與 reference eval 用**同一個函式**確保一致：
[`head_outputs.py:15-35`](deployment/projects/bevfusion_l/io/head_outputs.py#L15-L35) `head_dict_to_detection_outputs`
```python
score = outputs["heatmap"].sigmoid()                                            # s_query
one_hot = F.one_hot(outputs["query_labels"], num_classes=score.size(1)).permute(0, 2, 1)
score = score * outputs["query_heatmap_score"] * one_hot                         # ×s_proposal ×one_hot
score = score[0].max(dim=0)[0]                                                   # max_c
bbox_pred = torch.cat([center, height, dim, rot, vel], dim=0)                    # [10, num_proposals]
return bbox_pred, score, query_labels[0]
```
> 這正是 §26 表格中 `bbox_pred [10,500] / score [500] / label [500]` 的產生點，也是 ONNX graph 的輸出契約。
> 原模型 eval 端的等價邏輯在 [`bevfusion_head.py:404-422`](projects/BEVFusion/bevfusion/bevfusion_head.py#L404-L422) `predict_by_feat`。

---

## §29–33 Decode：Feature → Metric 座標

文件 §29–32（center/dim/rot/height 的 decode）與 §33 的完整數值範例，全部在 bbox coder 的一個 `decode`：
[`utils.py:126-163`](projects/BEVFusion/bevfusion/utils.py#L126-L163) `TransFusionBBoxCoder.decode`
```python
final_preds  = heatmap.max(1).indices    # label
final_scores = heatmap.max(1).values      # score
# §29 center: feature → metric（out_size_factor=8, voxel_size=0.17, +pc_range）
center[:,0,:] = center[:,0,:] * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
center[:,1,:] = center[:,1,:] * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
dim = dim.exp()                           # §30 log dim → 實際尺寸
height = height - dim[:,2:3,:] * 0.5      # §32 gravity center → bottom center
rot = torch.atan2(rots, rotc)             # §31 (sin,cos) → yaw
```

---

## §34 Postprocess：threshold + post-center-range + Circle NMS

文件的後處理流程在部署層 `postprocess()`，直接呼叫模型的 coder（`filter=True`）與 cluster/circle NMS，複現 test.py：
[`bevfusion_inference_pipeline.py:145-246`](deployment/projects/bevfusion_l/inference/bevfusion_inference_pipeline.py#L145-L246)
```python
# filter=True → coder 內套用 per-class score_threshold + post_center_range
decoded = bbox_coder.decode(heatmap, rot, dim, center, height, vel, filter=True)[0]
boxes3d, scores, labels = apply_cluster_nms(
    decoded["bboxes"], decoded["scores"], decoded["labels"],
    nms_type=bbox_head.test_cfg.get("nms_type"),
    nms_clusters=getattr(bbox_head, "nms_clusters", []), ...)
```
- per-class `score_threshold` / `post_center_range` 的實際套用：
  [`utils.py:176-200`](projects/BEVFusion/bevfusion/utils.py#L176-L200)（`decode` 的 filter 分支）。
- Circle / cluster NMS：`apply_cluster_nms` 定義於
  [`utils.py`](projects/BEVFusion/bevfusion/utils.py#L1)（import 於
  [`bevfusion_inference_pipeline.py:21`](deployment/projects/bevfusion_l/inference/bevfusion_inference_pipeline.py#L21)）。
- 輸出 detection dict（`bbox_3d / score / label`）：
  [`bevfusion_inference_pipeline.py:225-246`](deployment/projects/bevfusion_l/inference/bevfusion_inference_pipeline.py#L225-L246)。

---

## §35–37 完整對照速查

| 文件章節 / 操作 | 檔案:行 | 函式 / 符號 |
|---|---|---|
| §2–3 Voxelization | [inference:85](deployment/projects/bevfusion_l/inference/bevfusion_inference_pipeline.py#L85) / [bevfusion.py:262](projects/BEVFusion/bevfusion/bevfusion.py#L262) | `preprocess` → `pts_voxel_layer` / `voxelize` |
| §4 Voxel encoder | [voxel_encoder:48](projects/BEVFusion/bevfusion/bevfusion_voxel_encoder.py#L48) | `HardSimpleVoxelSinCosEncoder.forward` |
| §5 Sparse encoder | [sparse_encoder:122](projects/BEVFusion/bevfusion/sparse_encoder.py#L122) | `BEVFusionSparseEncoder.forward` |
| §5 sparse 入口(部署) | [onnx:52](deployment/projects/bevfusion_l/export/onnx_models/bevfusion_onnx.py#L52) / [bevfusion.py:200](projects/BEVFusion/bevfusion/bevfusion.py#L200) | `BEVFusionSparseWrapper` / `extract_pts_feat` |
| §6 Scatter + Z collapse | [sparse_encoder:154](projects/BEVFusion/bevfusion/sparse_encoder.py#L154) | `sparse_to_dense` + `view` |
| §7–8 SECOND + FPN | [onnx:74](deployment/projects/bevfusion_l/export/onnx_models/bevfusion_onnx.py#L74) | `BEVFusionDenseWrapper` → `pts_backbone/pts_neck` |
| §10 Shared conv | [head:269](projects/BEVFusion/bevfusion/bevfusion_head.py#L269) | `shared_conv` |
| §11 Dense heatmap | [head:280](projects/BEVFusion/bevfusion/bevfusion_head.py#L280) | `heatmap_head` + `sigmoid` |
| §12 Local-max | [head:283](projects/BEVFusion/bevfusion/bevfusion_head.py#L283) | `F.max_pool2d` + `heatmap==local_max` |
| §13 Flatten | [head:316](projects/BEVFusion/bevfusion/bevfusion_head.py#L316) | `view` |
| §14 Top-500 + index 反解 | [head:322](projects/BEVFusion/bevfusion/bevfusion_head.py#L322) | `torch.topk` / `//`,`%` |
| §15 Gather query | [head:328](projects/BEVFusion/bevfusion/bevfusion_head.py#L328) | `fusion_feat_flatten.gather` |
| §16 Query pos | [head:340](projects/BEVFusion/bevfusion/bevfusion_head.py#L340) | `bev_pos` + `self_posembed` |
| §17 Class embed | [head:335](projects/BEVFusion/bevfusion/bevfusion_head.py#L335) | `class_encoding` |
| §18–25 Transformer | [head:348](projects/BEVFusion/bevfusion/bevfusion_head.py#L348) / [transformer:26](projects/BEVFusion/bevfusion/transformer.py#L26) | `decoder[i]` / `TransformerDecoderLayer` |
| §26 Prediction heads | [head:353](projects/BEVFusion/bevfusion/bevfusion_head.py#L353) | `prediction_heads[i]` |
| §27 Center += pos | [head:354](projects/BEVFusion/bevfusion/bevfusion_head.py#L354) | `center + query_pos` |
| §28 Score | [head_outputs:25](deployment/projects/bevfusion_l/io/head_outputs.py#L25) | `head_dict_to_detection_outputs` |
| §29–33 Decode | [utils:126](projects/BEVFusion/bevfusion/utils.py#L126) | `TransFusionBBoxCoder.decode` |
| §34 threshold+NMS | [inference:212](deployment/projects/bevfusion_l/inference/bevfusion_inference_pipeline.py#L212) | `decode(filter=True)` + `apply_cluster_nms` |

---

### 一句話總結程式碼結構

> **部署層**只做兩件事：graph 外的 `preprocess`(voxelize) 與 `postprocess`(decode+NMS)，中間把模型切成 `Sparse`/`Dense` 兩個 wrapper 匯出。
> **模型層**才是文件描述的真正運算：`extract_pts_feat`（§4–6）→ `pts_backbone/pts_neck`（§7–8）→ `bbox_head.forward_single`（§9–28）→ `TransFusionBBoxCoder.decode`（§29–33）。
