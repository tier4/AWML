# 26: 為什麼 `ScatterND -> SECOND` 在不同 ONNX 不是一模一樣

本文件說明一個常見疑問：

- `original`（舊 main-body 單圖）ONNX 在 `ScatterND` 後常看到  
  `Transpose -> Shape -> Gather -> Unsqueeze -> ...`
- `bevfusion_deployment_2_7_fp16_opt_merged_flip_no_opt`（split export 再 merge）ONNX  
  常只看到短鏈（例如 `Transpose -> Reshape -> Conv`）

這不是單純「誰對誰錯」，而是 **trace 邊界** 與 **圖組合方式** 不同造成的 ONNX 表達差異。

---

## 1. `ScatterND` 後面那串在做什麼

在 sparse encoder，`conv_out.dense()` 之後要把 5D 張量整理成 BEV 4D：

```20:36:projects/BEVFusion/bevfusion/sparse_encoder.py
x = out_tensor.dense()                      # (N, C, H, W, D)
t = x.permute(0, 1, 4, 2, 3).contiguous()  # (N, C, D, H, W)
return t.view(n, c * int(d), int(h), int(w))
```

PyTorch 的 `permute + view` 在 ONNX 常被展開成：

- `Transpose`（維度重排）
- `Shape/Gather`（抽出 N/C/H/W/D）
- `Unsqueeze/Concat`（組新 shape 向量）
- `Reshape`

所以在 `original` 裡看到 `Transpose -> Shape -> Gather -> Unsqueeze` 是合理的。

---

## 2. 為什麼 split trace 後看起來「少很多」

### 2.1 匯出邊界不同（核心）

`original` 是「單次 trace 全主圖」：  
`voxels/coors -> sparse -> dense backbone/neck/head` 一次畫完。

split export 是兩次 trace：

1. sparse 子圖：`voxels/coors -> lidar_bev`
2. dense 子圖：`lidar_bev -> backbone/neck/head`

對應程式：

- sparse wrapper：`BEVFusionSparseWrapper`
- dense wrapper：`BEVFusionDenseWrapper`
- merge：`onnx.compose.merge_models`

```223:231:/home/yihsiangfang/ml_workspace/AWML/deployment/projects/bevfusion_l/export/onnx_export_pipeline.py
def _export_split(...):
    """Export ``bevfusion_sparse.onnx`` and ``bevfusion_dense.onnx``."""
```

### 2.1.1 分開 trace vs 一起 trace：各自用什麼方式

- **一起 trace（single-file）** — ⚠️ 此路徑已從 codebase 移除，僅保留於此作為 trace 差異的對照說明
  - (舊) 包裝：`BEVFusionMainBodyWrapper`（已刪除）
  - (舊) 方法：呼叫一次匯出，直接把  
    `voxels/coors/num_points_per_voxel -> bbox_pred/score/label` 全流程 trace 成同一張 ONNX
  - 特性：shape 推導鏈通常完整保留在同一張圖內（較常看到 `Shape/Gather/Unsqueeze`）
  - 現況：全圖 `bevfusion_merged` 改由 split 的 sparse+dense ONNX 事後 compose 而成（見下方 §2.2）

- **分開 trace（split export）**
  - 入口：`BEVFusionONNXExportPipeline._export_split()`
  - 包裝與方法：
    1. sparse 段：`BEVFusionSparseWrapper` + `self._export_to_onnx(..., wrapper="sparse")`
    2. dense 段：先用 sparse wrapper 跑出 `lidar_bev`，再用 `self._export_dense_to_onnx(...)` 匯出 dense
    3. 合併段：`onnx.compose.add_prefix + onnx.compose.merge_models + cleanup().toposort()`
  - 特性：merge 後常看到更短的 shape 鏈，因為跨子圖的中介 shape plumbing 容易被折疊

### 2.2 graph compose + cleanup 會消掉中介 shape plumbing

split merge 之後會做 `cleanup().toposort()`，一些跨段的中介 shape 節點會被折疊或改寫：

```488:490:/home/yihsiangfang/ml_workspace/AWML/deployment/projects/bevfusion_l/export/onnx_export_pipeline.py
merged_graph.cleanup().toposort()
onnx.save_model(gs.export_onnx(merged_graph), str(merged_path))
```

因此你在 merged 圖常看到更短鏈（例如 `Transpose -> Reshape -> Conv`），而不是單圖 trace 時那種長 shape plumbing。

---

## 3. 為什麼「分開 trace」會比較少 `Shape/Gather/Unsqueeze`

### 3.1 這串節點在做什麼

`Shape -> Gather -> Unsqueeze -> Concat` 是 ONNX 表達「**在 runtime 才讀取某個維度的值**」的方式。

在 `_conv_out_to_bev`（`sparse_encoder.py`）：

```python
x = out_tensor.dense()          # (N, C, H, W, D)
n = int(x.shape[0])             # ← 這行是關鍵：讀取 batch size
t = x.permute(0, 1, 4, 2, 3)
return t.view(n, c * int(d), int(h), int(w))
```

當 `n` 是「要到 runtime 才知道的值」，ONNX 沒辦法寫死 `Reshape(t, [n, 256, 180, 180])`，
必須展開成完整的 shape 讀取鏈：

```
Shape(x)  →  [N, C, H, W, D]
   ↓
Gather(index=0)  →  N
   ↓
Unsqueeze  →  [N]
   ↓
Concat([N], [256], [180], [180])  →  shape vector [N, 256, 180, 180]
   ↓
Reshape(t, shape_vector)
```

### 3.2 dense 子圖：`permute + view` 根本不在 trace 範圍內

`BEVFusionDenseWrapper.forward()` 的入口是 `lidar_bev`：

```python
def forward(self, lidar_bev: torch.Tensor) -> tuple:
    x = lidar_bev   # 已經是 (N, C*D, H, W) 的 BEV feature map
    x = self.mod.pts_backbone(x)
    ...
```

**dense 子圖從一開始就接收已整理好的 BEV tensor，從來不會 trace 到 `ScatterND`、
`dense()`、`permute`、`view` 這些操作。** 因此 dense ONNX 裡根本不存在這串節點，
不是「被刪掉」，而是「從未被畫進去」。

### 3.3 sparse 子圖：trace 時 batch=1 是靜態值，cleanup 再折疊

export 流程是先跑一次 `BEVFusionSparseWrapper` 的 forward pass 拿到 `lidar_bev`（`onnx_export_pipeline.py:307-313`），
再用同樣那份 sample trace sparse ONNX。此時 batch=1 是 Python int，
`n = int(x.shape[0])` = `1`，trace 後 `Reshape` 的 shape 已是靜態常數 `[1, 256, 180, 180]`。

即使 trace 留下一些中介常數節點，`cleanup()` 做 constant folding 後也會把
`Shape -> Gather -> Unsqueeze -> Concat` 這條鏈折成單一常數向量。

對照單圖 trace：整個圖一次畫完，batch 被宣告為 dynamic axis（或 symbolic 追蹤跨越較多節點邊界），
`n` 無法靜態化，整串 shape plumbing 就保留下來。

### 3.4 小結

| | `Shape/Gather/Unsqueeze` 存在嗎？ | 原因 |
|---|---|---|
| 原始 single-trace ONNX | **有** | 整圖 trace，batch 為動態，`n` 要到 runtime 才讀 |
| split export 的 **dense 子圖** | **沒有**（完全不含這段程式） | dense 子圖從 `lidar_bev` 開始，`permute+view` 不在 trace 範圍 |
| split export 的 **sparse 子圖** + merge 後 | **通常沒有或更短** | trace 時 batch=1 為靜態整數；`cleanup()` constant folding |

不是因為模型少算了，而是：
1. dense 子圖把 `lidar_bev` 當作外部輸入張量，`permute+view` 的 trace 根本不在這裡。
2. sparse 子圖 trace 時 batch 維度靜態已知，shape plumbing 可被折疊。
3. `cleanup()` 刪掉已無引用或可靜態化的 shape 節點。

**同一個數學流程，ONNX 可以有不同等價寫法；split trace + compose 通常更短。**

---

## 4. 這會不會造成數值不同？

### 4.1 理論上

僅就 `Transpose/Shape/Gather/Unsqueeze/Reshape` 這類 layout/shape op：

- 應是語義等價變換
- 不涉及乘加，不應引入浮點誤差

### 4.2 實務上（真正會讓 mAP 掉到 0 的通常不是這串）

真正高風險通常是契約不一致，例如：

- `coors` 欄位順序契約（`xyz` vs `zyx`）
- BN fold 與 bias 契約（特別是 shadow export 路徑）
- head grid 尺寸契約（`grid_size // out_size_factor`，常見 180x180）

也就是：  
**節點長相不同通常不是主因；契約對不齊才是主因。**

---

## 5. 如何判斷是不是「只是圖長相差異」

建議同一筆 sample 做分段比對：

1. 比 `lidar_bev`（sparse 輸出）統計
2. 比 `bbox_pred/score/label` 分布
3. 若需要，再比最終 mAP

如果中間張量與最終輸出一致，表示只是 ONNX 表達不同；  
若差異巨大，優先檢查 `coors`/BN/head-grid 契約。

---

## 6. 結論

`ScatterND` 後面不一模一樣，主要是 **export 策略不同**：

- 單圖 trace：保留較多動態 shape plumbing
- split trace + merge：圖更模組化，shape 子圖更容易被折疊

這種差異本身通常可接受；是否正確應以 **契約一致性** 與 **數值驗證** 為準，而不是只看節點數量或節點名稱是否一樣。
