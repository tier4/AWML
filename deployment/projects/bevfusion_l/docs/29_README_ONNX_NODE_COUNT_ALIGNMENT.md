# 29: BEVFusion ONNX 節點數對齊 — commit `78b66a70` 的 clean-export 改動、split 邊界 `dynamic_axes`、以及新舊方法的殘留差異

本文件回答一個具體問題:

> 用**新方法**(deployment CLI,split sparse+dense 再 merge)匯出的 `bevfusion_lidar.onnx`,
> 為什麼節點數跟**舊方法**(`projects/BEVFusion/deploy/torch2onnx.py`,整體單圖匯出)不一樣?
> 這跟 `78b66a70a1e6b394e74c1912c9c78d4258e7d6db`(BEVFusion 2.8.x release, #217)有關嗎?能不能對齊?

結論先講:

- **能對齊。** 關鍵是把 split config 裡 `lidar_bev` 的 `dynamic_axes` 拿掉,讓切點回到靜態 shape。
- commit `78b66a70` **提供了** clean-export 的機制(把動態 `.shape`/`.dense()` 換成 config 來的靜態 shape),
  但這機制**必須配合「切點不標 dynamic」才會生效**;先前 split config 對 `lidar_bev` 標了 dynamic,把 commit 的靜態 shape 又動態化,效果被抵消。
- 修正後仍有**個位數的殘留差異**,那是 split-then-merge 與 monolithic 在切點本質上的接縫差異,不是膨脹,且數值等價。

> ⚠️ 本文件**更正** [`26_README_SCATTERND_TO_SECOND_TRACE_DIFFERENCE.md`](./26_README_SCATTERND_TO_SECOND_TRACE_DIFFERENCE.md) 的一項結論。
> 詳見下方 [§7 更正 doc 26](#7-更正-doc-26)。

---

## 0. 實測數字(apples-to-apples)

三份都是 **LiDAR-only、opset 18、`simplify=False`、無 fusion**,唯一差別是匯出方式與 `lidar_bev` 的 dynamic 設定:

| 版本 | 匯出方式 | `lidar_bev` dynamic | #nodes | #initializers | 輸出 shape |
|---|---|---|---:|---:|---|
| **舊** `torch2onnx.py` | 整體單圖 | (無此邊界) | **423** | 207 | 符號 `Concat.../ReduceMax...` |
| **新** split(**修正前**) | sparse+dense→merge | `batch, H, W` 皆動態 | **524** | 207 | `[10, dyn]` |
| **新** split(**修正後**) | sparse+dense→merge | **全靜態** | **416** | 207 | **全靜態** `[10,500]` / `[500]` / `[500]` |

- 修正前 split 比舊版多 **101** 個節點,幾乎全是 shape-plumbing(`Shape/Gather/Unsqueeze/Concat/Constant`)。
- 修正後掉到 **416**,與舊版 423 幾乎一致(甚至少 7),initializer 三者都是 207。
- 修正後輸出 shape **比舊版更乾淨**(全靜態),因為靜態邊界讓 shape inference 能一路推到底。

測試模型 / 指令見 [§8 重現步驟](#8-重現步驟)。

---

## 1. 背景:兩條匯出路線

BEVFusion LiDAR-only 有兩條 PyTorch→ONNX 路線,產生的圖**在數學上等價、但結構不同**:

### 1.1 舊:整體單圖(monolithic)

`projects/BEVFusion/deploy/torch2onnx.py --module main_body`

- 一次 `torch.onnx.export`,把 `voxels/coors/num_points_per_voxel → (sparse encoder) → (dense backbone/neck/head) → bbox_pred/score/label` 全流程 trace 成**同一張** ONNX。
- 後處理只有:onnx-graphsurgeon 把 TopK 的 `K` 改成常數([`exporter.py:_fix_onnx_graph`](../../../../projects/BEVFusion/deploy/exporter.py))+ `cleanup().toposort()`。
- **沒有** onnx-simplifier。
- `dynamic_axes` 只標了三個 sparse 輸入(`voxels/coors/num_points_per_voxel` 的第 0 維 `voxels_num`)。中間張量(如 sparse 輸出的 BEV feature)是圖內部,shape 在 trace 時是**具體數字**。

### 1.2 新:split sparse+dense 再 merge

`python -m deployment.cli.main bevfusion_l <deploy_cfg> <model_cfg>`

- 把模型拆成兩段,各自 `torch.onnx.export`:
  1. `bevfusion_sparse.onnx`:`voxels/coors/num_points_per_voxel → lidar_bev`(含 spconv 自訂 op `GetIndicePairsImplicitGemm` / `ImplicitGemm`)
  2. `bevfusion_dense.onnx`:`lidar_bev → bbox_pred/score/label`(純標準 ONNX op,可上一般 TensorRT)
- 再用 [`onnx.compose.merge_models`](../export/transforms.py) 加上 `sparse/`、`dense/` 前綴合併成 `bevfusion_lidar.onnx`,最後 `cleanup().toposort()`。
- 拆分的目的:sparse 段(spconv/plugin)與 dense 段(純 TRT)可以分開處理/量化。

**核心差別**:新路線在 `lidar_bev` 這個張量處**切了一刀**,製造出一個真正的圖邊界(sparse 的輸出、dense 的輸入)。這正是節點數差異的來源(見 §5)。

---

## 2. commit `78b66a70` 做了什麼(ONNX 相關)

這個 commit 是 **BEVFusion 2.8.x release**,主題之一就是「**更乾淨的 ONNX 匯出**」。它系統性地把「在 runtime 讀 `.shape`」和「slice 賦值」這類**難匯出**的寫法,換成「用 config 來的靜態常數」和「乾淨的單一 op」。逐項說明:

### 2.1 `sparse_to_dense` — 靜態化 sparse→dense 邊界(最關鍵)

新增檔案 [`custom_sparse_conv_tensor.py`](../../../../projects/BEVFusion/bevfusion/custom_sparse_conv_tensor.py),docstring 明寫:

> *"This customization is used to support cleaner ONNX export of sparse convolutions."*

它取代了 spconv 的 `out.dense()`:

**舊寫法**(`sparse_encoder.py`,commit 前):
```python
spatial_features = out.dense()                       # spconv scatter，動態 shape
N, C, H, W, D = spatial_features.shape               # ← 讀 .shape（動態）
spatial_features = spatial_features.permute(0, 1, 4, 2, 3).contiguous()
spatial_features = spatial_features.view(N, C * D, H, W)   # ← 用讀來的 N/C/H/W/D
```

**新寫法**(`sparse_encoder.py`,commit 後):
```python
spatial_features = sparse_to_dense(out, batch_size, self.dense_output_shapes, self.output_channels)
spatial_features = spatial_features.permute(0, 4, 3, 1, 2).contiguous()
spatial_features = spatial_features.view(
    batch_size,
    self.output_channels * self.dense_output_shapes[2],   # ← config 常數 C*D
    self.dense_output_shapes[0],                           # ← config 常數 H=180
    self.dense_output_shapes[1],                           # ← config 常數 W=180
)
```

`sparse_to_dense` 內部用手算的 linear index 把 features scatter 進一個**靜態大小**的 `torch.zeros([batch*H*W*D, C])`,再 `view`。重點:reshape 的目標維度來自 **config 的 `dense_output_shapes`(靜態 int)**,不再讀 `.shape`。

同時 `BEVFusionSparseEncoder.__init__` 也改了 signature:移除 `aug_features_min_values/max_values/num_aug_features`,新增 `dense_output_shapes`。新舊兩個 model config **都**設了:
```python
# projects/BEVFusion/configs/t4dataset/default/pipelines/default_lidar_120m.py
sparse_dense_output_shapes = [180, 180, 2]
# .../BEVFusion-L/bevfusion_lidar_voxel_..._120m.py
dense_output_shapes=_base_.sparse_dense_output_shapes,
```

### 2.2 `HardSimpleVoxelSinCosEncoder` — 把 sin-cos 編碼折成一個 FMA

新增檔案 [`bevfusion_voxel_encoder.py`](../../../../projects/BEVFusion/bevfusion/bevfusion_voxel_encoder.py)。原本 sparse encoder 裡的特徵增強(sin-cos / Fourier 編碼)被搬出來,並且把
`((x - min) / (max - min)) * pi * exponents` 這串 **sub → div → mul** 代數化簡成
`scale * x + bias` 一個 `torch.addcmul`(FMA),可折成單一 op。ONNX 因此少掉一串 element-wise 節點。

### 2.3 head 重構 — 全面去除動態 `.shape` 與 slice 賦值

[`bevfusion_head.py`](../../../../projects/BEVFusion/bevfusion/bevfusion_head.py)(+116 行)把 `predict` 路徑整個改成 export-friendly:

| 舊寫法(難匯出) | 新寫法(乾淨) |
|---|---|
| `batch_size = inputs.shape[0]`;`fusion_feat.view(batch_size, C, -1)` | `fusion_feat.view(-1, self.share_conv_out_channels, self.spatial_dim)`(靜態 `spatial_dim = H*W`) |
| `bev_pos.repeat(batch_size,...).to(device)` | `bev_pos` 註冊成 buffer(靜態、自動上 GPU) |
| `local_max[:, idx, pad:-pad, pad:-pad] = ...`(**slice 賦值** → ONNX 變 ScatterND 惡夢) | `F.pad(...)` + `torch.cat(...)` + 用 `local_concat_class_remapping` buffer 做 gather 重排 |
| `heatmap.view(batch_size,...).argsort(...)[:num_proposals]` | `torch.topk(k=num_proposals)`(單一 TopK op) |
| `top_proposals // heatmap.shape[-1]` | `top_proposals // self.spatial_dim`(靜態) |

### 2.4 `bevfusion.py` — onnx 路徑固定 `batch_size = 1`

[`bevfusion.py`](../../../../projects/BEVFusion/bevfusion/bevfusion.py) 的 `extract_pts_feat` onnx 分支直接寫死 `batch_size = 1`(靜態 Python int),並把 voxel 特徵編碼交給新的 `pts_voxel_encoder`。`torch.cuda.amp.autocast` 也更新成 `torch.amp.autocast("cuda", ...)`。

### 2.5 `exporter.py` — LayerNorm 匯出成原生 `LayerNormalization`

舊 deploy 路徑的 [`exporter.py`](../../../../projects/BEVFusion/deploy/exporter.py) 新增 `purge_mmdeploy_symbolics(["layer_norm"])`:刪掉 mmdeploy 對 `layer_norm` 的 symbolic,讓它匯出成 opset 17+ 的**原生 `LayerNormalization`**(而非被 mmdeploy 拆解)。這就是為什麼新舊兩張圖都看到 `LayerNormalization: 3`。

### 2.6 `utils.py` — `TransFusionBBoxCoder` 支援 per-class 分數門檻

[`utils.py`](../../../../projects/BEVFusion/bevfusion/utils.py) 的 `decode` 重構了 filter 邏輯,`score_threshold` 可接受 list/tuple(每類不同門檻,`score_threshold[final_preds]`)。與節點數無直接關係,但屬同一 release 的後處理改動。

---

## 3. 這些改動對 ONNX 的整體效果

一句話:**把「要到 runtime 才知道的形狀」變成「trace/編譯期就是常數的形狀」,讓 `do_constant_folding` 能把 shape-plumbing 折掉。**

- `torch.onnx.export` 遇到 `x.view(...)` / `x.shape[i]` 這類操作,會產生一串 `Shape → Gather → Unsqueeze → Concat → Reshape` 在**執行時**算形狀。
- `do_constant_folding=True`(新舊都有開)只在**這串的輸入全是常數**時,才能把它折成一個 Constant 並刪掉。
- commit `78b66a70` 的每一項改動(`dense_output_shapes`、`spatial_dim`、`batch_size=1`、`F.pad`/`cat`/`topk`)都是在**把那些輸入變成常數**,於是整串 shape-plumbing 可以被折掉 → 圖變乾淨。

**但有一個前提**:這些靜態 shape 只有在「該維度沒有被宣告成 dynamic axis」時才留得住。一旦你在匯出時把某個維度列進 `dynamic_axes`,exporter 就**被迫**保留該維度的 shape-plumbing —— 這正是新方法先前的問題。

---

## 4. 為什麼「切一刀」會多出節點(機制)

> 一個張量只要還在圖**內部**,shape 在 trace 時是具體數字 → 相關 shape 運算被折掉;
> 一旦在那裡**切成邊界**(輸出/輸入),shape 變成符號 → 折不掉,留成一堆 `Shape/Gather/Unsqueeze/Concat`。

以切點 `lidar_bev`(shape ≈ `[1, 512, 180, 180]`)為例:

- **整體單圖(舊)**:`lidar_bev` 是圖內部 activation,trace 時 shape 具體 → dense head 的 shape 運算全折疊 → 乾淨。
- **split(新)**:`lidar_bev` 變成 sparse 的 output + dense 的 input。**生產端**(sparse)要動態組出輸出形狀,**消費端**(dense)的 input 形狀是 placeholder → 兩側都得補一套 shape 重建。切一刀 → 兩邊各多一套,節點淨增。

實測子圖 shape-glue(`Shape+Gather+Unsqueeze+Concat+Constant+Cast`)節點:sparse ≈ 50、dense ≈ 177 —— 對應 §0 修正前多出的 ~101 節點。

---

## 5. 真正的病灶:split config 對 `lidar_bev` 標了 `dynamic_axes`

即使 §2.1 的 `sparse_to_dense` 已經用靜態 shape 建好 `lidar_bev`,先前 split config 又把它標成 dynamic,把靜態成果**動態化**回去:

`deployment/projects/bevfusion_l/config/deploy_config.py`(修正前):
```python
# bevfusion_sparse — 輸出
dynamic_axes={
    "voxels": {0: "voxels_num"},
    "coors": {0: "voxels_num"},
    "num_points_per_voxel": {0: "voxels_num"},
    "lidar_bev": {0: "batch", 2: "bev_h", 3: "bev_w"},   # ← 把 batch/H/W 標成動態
},
# bevfusion_dense — 輸入
dynamic_axes={
    "lidar_bev": {0: "batch"},                            # ← 把 batch 標成動態
},
```

一旦 `lidar_bev` 的 batch/H/W 被列進 `dynamic_axes`:

1. **sparse 側**:輸出 shape 變 `['batch','Reshape...','bev_h','bev_w']`(全符號),`sparse_to_dense` 的靜態 view 被迫展開成 `Shape/Gather/Concat/Reshape`。
2. **dense 側**:input batch 動態 → 整個 head 凡是依賴 batch 維的運算(reshape、gather、decode)都折不掉,留下大量 shape-plumbing。

**佐證:這些 dynamic axes 根本沒必要。** dense 的 TRT profile 是
```python
lidar_bev=dict(min_shape=[1,256,180,180], opt_shape=[1,256,180,180], max_shape=[1,256,180,180])
```
`min == opt == max`,batch/H/W 執行期全鎖死;而且 config docstring 自己就寫「H/W 必須固定 180,不能給範圍,否則 `bbox_head` 的 `Reshape/Gather` 會壞掉、mAP 變垃圾」。既然執行期都是定值,標成 dynamic 純屬有害無益。

### 5.1 修正

把兩處 `lidar_bev` 移出 `dynamic_axes`,**保留** `voxels/coors/num_points_per_voxel` 的動態(voxel 數每幀真的會變,且只影響 sparse 前端、不會膨脹 head):

```python
# bevfusion_sparse — 輸出：拿掉 lidar_bev（保留三個 sparse 輸入的動態）
dynamic_axes={
    "voxels": {0: "voxels_num"},
    "coors": {0: "voxels_num"},
    "num_points_per_voxel": {0: "voxels_num"},
}
# bevfusion_dense — 輸入：完全靜態
dynamic_axes={}
```

已同步套用到:
- [`config/deploy_config.py`](../config/deploy_config.py)(正式,fusion 開)
- [`config/deploy_config_without_opt.py`](../config/deploy_config_without_opt.py)(無 fusion / 無 simplify,用於 §0 的 apples-to-apples 比較)

效果:節點 524 → **416**(見 §0)。

---

## 6. 修正後仍存在的差異(良性)

對齊後 `416 vs 423`,op 層級只剩個位數差異(新-修正後 相對 舊):

```
Constant  -6     Slice  -2     Mul  -1     Concat -1     Sub -1      ← 新版更精簡
Reshape   +2     Mod    +1     Transpose +1                          ← sparse→dense 接縫
```

這些是 **monolithic vs split-then-merge 本質上的接縫差異**,不是 shape-glue 膨脹:

1. **接縫重建**:`sparse_to_dense` 的 scatter/linear-index 計算(`Mod`、`Reshape`、`Transpose`)在 split 版被具體化在切點附近;monolithic 版因整圖 trace 而被折進常數路徑。
2. **有幾項新版反而更少**(`Constant/Slice/Mul/Concat/Sub`),因為靜態邊界讓 folding 更徹底。
3. 這種個位數差異已無法再靠改設定消除,除非放棄 split。

其他非 op-count 的差異:

| 面向 | 舊(monolithic) | 新(split-merge) |
|---|---|---|
| producer | `pytorch 2.8.0` | `onnx.compose.merge_models 1.0` |
| opset_import | `ai.onnx 18` + `autoware 1` 各一次 | 兩組(每個子圖一組,merge 後保留;功能相同) |
| 節點名稱 | 無前綴 | `sparse/`、`dense/` 前綴 |
| 輸出 shape | 符號(`Concatbbox_pred_dim_0`...) | 全靜態(`[10,500]`...) |

### 6.1 正式 config 額外的 fusion(與本文的邊界修正正交)

[`deploy_config.py`](../config/deploy_config.py) 比 `deploy_config_without_opt.py` 多開兩個 fusion,會**再**減少節點,但這與 commit / 邊界修正無關,是獨立的優化旋鈕:

- `fuse_spconv_bn = True`:把 sparse 段的 SparseConv+BN 在匯出前 fold(eval-mode Conv-BN 融合)→ 減少 `BatchNormalization`。
- `spconv_fuse_implicit_gemm_relu = True`:把 `ImplicitGemm` 後的 `Relu`(及 `Add(const)+Relu`)烘進 plugin 的 `act_type` → 減少 `Relu`。

> 注意:兩個 config 的 `onnx_config.simplify` **都是 `False`**。本 repo 目前預設**不跑 onnx-simplifier**;§0 的比較也都是 `simplify=False`。若另外開 `simplify=True`,節點會再大幅下降(constant fold 進 initializer、shape 定死),但那是另一層優化,不影響本文結論。

---

## 7. 更正 doc 26

[`26_README_SCATTERND_TO_SECOND_TRACE_DIFFERENCE.md`](./26_README_SCATTERND_TO_SECOND_TRACE_DIFFERENCE.md) §2–§3 主張「split export 的 shape 鏈通常**更短/更少**」。**依本文實測,在目前的 config 下結論相反**:split(524)比 monolithic(423)**更多**節點。原因:

1. doc 26 沒有把 config 對 `lidar_bev` 宣告的 `dynamic_axes` 納入考量。一旦切點被標 dynamic,dense head 的 shape-plumbing **無法折疊**,反而**增加**節點(§5)。
2. doc 26 引用的程式路徑(`out_tensor.dense()`、`_conv_out_to_bev`、`BEVFusionDenseWrapper`、`onnx_export_pipeline.py:_export_split`)是 **refactor 前的舊碼**,現已不存在;現行實作是 `sparse_to_dense`([`custom_sparse_conv_tensor.py`](../../../../projects/BEVFusion/bevfusion/custom_sparse_conv_tensor.py))+ [`export/transforms.py`](../export/transforms.py)+ [`export/component_builder.py`](../export/component_builder.py)。

doc 26 仍然正確的部分:**「節點長相/數量不同 ≠ 數值不同;應以契約一致性 + 數值驗證為準」**(§4–§6)。這點本文完全同意,見 §9。

---

## 8. 重現步驟

於 docker container `awml-bevfusion`(repo bind-mount 在 `/workspace`,torch 2.8.0 / onnx 1.17)內:

```bash
# 新方法(split，會產出 sparse/dense/merged 三個 onnx)
python -m deployment.cli.main bevfusion_l \
  deployment/projects/bevfusion_l/config/deploy_config_without_opt.py \
  projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_8xb16_j6gen2_base_120m_t4metric_v2.py
# → work_dirs/bevfusion_deployment_2_8_no_opt/onnx/bevfusion_lidar.onnx
# 只想匯 ONNX、跳過 TRT 建置：暫時把 export.mode 設成 "onnx"

# 舊方法(monolithic)
python projects/BEVFusion/deploy/torch2onnx.py \
  projects/BEVFusion/configs/deploy/bevfusion_main_body_lidar_only_tensorrt_dynamic.py \
  projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_8xb16_j6gen2_base_120m.py \
  work_dirs/bevfusion/bevfusion_2_8/best_epoch_25.pth --device cuda:0 \
  --work-dir work_dirs/bevfusion/bevfusion_2_8/ --module main_body
# → work_dirs/bevfusion/bevfusion_2_8/bevfusion_lidar.onnx
```

比對節點數 / op 分佈(host 或 container 皆可,只需 `onnx`):

```python
import onnx
from collections import Counter
for p in [OLD_PATH, NEW_PATH]:
    g = onnx.load(p).graph
    print(p, "#nodes", len(g.node), "#init", len(g.initializer))
    print(sorted(Counter(n.op_type for n in g.node).items()))
```

---

## 9. 結論與建議

- **節點數對齊 = 已完成**:移除 `lidar_bev` 的 `dynamic_axes` 後,split 版 416 ≈ monolithic 版 423,差異只剩良性接縫。
- **與 commit `78b66a70` 的關聯**:相關,但不是「新方法沒 apply」。commit 的 `sparse_to_dense`/靜態 head shape **有 apply**;先前是被 split config 的 `dynamic_axes` 抵消。修正後,commit 的 clean-export 才真正落地(輸出甚至比舊版更靜態)。
- **節點數不是等價判準**:如 doc 26 所述,不同 ONNX 表達可語義等價。真正該驗證的是**契約一致性**(`coors` 欄位序、BN/bias、head grid 尺寸)與**數值**(同一 sample 跑 TensorRT 比對 `lidar_bev` / `bbox_pred`/`score`/`label`)。節點數對齊只是讓兩條路線**更好比對、更好維護**,不是正確性的證明。

---

### 附:相關檔案

- 匯出流程:[`export/onnx_export_pipeline.py`](../export/onnx_export_pipeline.py)、[`export/component_builder.py`](../export/component_builder.py)、[`export/transforms.py`](../export/transforms.py)(`merge_split_sparse_dense_onnx`、TopK fix)
- config:[`config/deploy_config.py`](../config/deploy_config.py)、[`config/deploy_config_without_opt.py`](../config/deploy_config_without_opt.py)
- 模型:[`custom_sparse_conv_tensor.py`](../../../../projects/BEVFusion/bevfusion/custom_sparse_conv_tensor.py)、[`sparse_encoder.py`](../../../../projects/BEVFusion/bevfusion/sparse_encoder.py)、[`bevfusion_voxel_encoder.py`](../../../../projects/BEVFusion/bevfusion/bevfusion_voxel_encoder.py)、[`bevfusion_head.py`](../../../../projects/BEVFusion/bevfusion/bevfusion_head.py)
- 舊路徑:[`projects/BEVFusion/deploy/exporter.py`](../../../../projects/BEVFusion/deploy/exporter.py)、[`torch2onnx.py`](../../../../projects/BEVFusion/deploy/torch2onnx.py)
