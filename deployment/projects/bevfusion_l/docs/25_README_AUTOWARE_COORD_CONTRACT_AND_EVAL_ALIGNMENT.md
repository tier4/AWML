# 25: BEVFusion `coors` 契約對齊 Autoware 與 Evaluation 修正

本文件記錄一個關鍵修正：讓 deployment framework 的 ONNX/TRT evaluation，對齊舊版（Autoware 相容）BEVFusion main-body ONNX 的 `coors` 契約，避免出現「PyTorch 正常、TRT mAP 接近 0」的錯誤對照結果。

---

## 1. 問題現象

同一個 checkpoint 與資料集，出現以下不一致：

- `original` ONNX：PyTorch mAP 正常，TRT mAP 幾乎 0
- `opt/new` ONNX：PyTorch mAP 正常，TRT mAP 正常
- 但 `opt/new` 在實際 Autoware 路徑上無有效輸出（mAP~0）

這代表 framework evaluation 與 Autoware 真實執行路徑存在契約不一致。

---

## 2. 根因：`coors` 的順序契約不一致

舊版 deploy（mmdeploy patch）在資料前處理與圖內 wrapper 有明確註解：

```43:50:projects/BEVFusion/deploy/voxel_detection.py
# The original code/model uses [batch, x, y, z]
# but the data_preprocessor used here uses [batch, z, y, x]
# Since this is outside the graph we format it as [z, y, x]
# and convert it to [batch, x, y, z] inside the graph
```

對應到舊 wrapper 的圖內邏輯：

```44:49:projects/BEVFusion/deploy/containers.py
if coors.shape[1] == 3:
    coors = coors.flip(dims=[-1]).contiguous()  # [z,y,x] -> [x,y,z]
    batch_coors = torch.zeros(num_points, 1).to(coors.device)
    coors = torch.cat([batch_coors, coors], dim=1).contiguous()
```

也就是：

- graph 外輸入 `coors`：`[z, y, x]`（無 batch）
- graph 內先 flip 成 `[x, y, z]`，再 prepend batch

若 evaluation runtime 沒遵守同一契約，就會把座標送錯位，TRT 精度會崩。

---

## 3. 這次修正做了什麼

### 3.1 ONNX export：明確保留 legacy contract

在新 framework 的 ONNX wrapper 中加入與舊版一致的正規化。

- 檔案：`deployment/projects/bevfusion_l/export/onnx_export_pipeline.py`
- 函式：`_normalize_sparse_coors_for_autoware()`
- 行為：`[N,3] coors` 先 `flip(-1)`，再補 batch 欄位

目的：讓新匯出的 ONNX 與舊版 Autoware 相容圖在 `coors` 契約上等價。

### 3.2 ONNX / TRT runtime pipeline：餵入前對齊同一契約

在 framework inference backend（非 metric/evaluator）端加入同樣契約對齊。

- `deployment/projects/bevfusion_l/pipelines/onnx.py`
- `deployment/projects/bevfusion_l/pipelines/tensorrt.py`
- 函式：`_normalize_coors_for_legacy_main_body_contract()`
- 行為：餵 backend 前，對 `[N,3]` coors 做 `flip(-1)`

目的：讓 runtime 輸入與 legacy main-body ONNX 的圖內假設一致。

### 3.3 沒有在 evaluator 做 flip（刻意）

`evaluator` 只應負責 metrics，不應修改模型輸入語義。  
flip 必須放在模型邊界（export wrapper / backend preprocess），且只做一次。

### 3.4 `5132ce649385e0a32ddc9b35821b29e5e69eed66` 的具體改動

該 commit（`chore: temp fix`）實際做了以下 5 項：

1. 新增 `deployment/projects/bevfusion_l/io/coors_contract.py`
   - 統一定義兩個轉換函式：
     - `voxel_indices_xyz_to_graph_input_zyx()`
     - `graph_input_zyx_to_model_indices_xyz()`
2. 修改 `deployment/projects/bevfusion_l/export/onnx_export_pipeline.py`
   - `_normalize_sparse_coors_for_autoware()` 改為呼叫 `coors_contract`，而非散落的 `flip`
   - `_voxelize()` 明確把 voxel layer 輸出的 `[x,y,z]` 轉成 graph input `[z,y,x]`
3. 修改 `deployment/projects/bevfusion_l/pipelines/onnx.py`
   - backend 餵入前改用 `coors_contract` 做 `xyz -> zyx`
4. 修改 `deployment/projects/bevfusion_l/pipelines/tensorrt.py`
   - backend 餵入前改用 `coors_contract` 做 `xyz -> zyx`
5. 新增測試型 config
   - `deployment/projects/bevfusion_l/config/deploy_config_split_fp16_opt_on_board_test.py`

---

## 4. `5132ce64` 是否合理（判斷）

### 4.1 合理的部分（建議保留）

- **把契約集中到單一模組 `coors_contract.py` 是正確方向**  
  可避免 export / ORT / TRT 各自手寫 `flip`，降低再度漂移風險。
- **在 `_voxelize()` 明確標註與轉換 `xyz -> zyx` 是關鍵修正**  
  這一步把「voxelization 真實輸出順序」與「legacy graph input 契約」橋接清楚，對消除 `PyTorch OK / TRT ~0` 很重要。
- **ORT/TRT backend 同步使用同一轉換**  
  保證 runtime 行為一致，不會某 backend 正常、某 backend 失真。

### 4.2 需要注意的部分（不一定錯，但要標記）

- commit message 是 `temp fix`，但內容其實是**正式契約修正**；建議後續用更明確訊息（例如 `fix: align BEVFusion coors contract with legacy autoware main_body`）。
- 新增的 `*_on_board_test.py` 是測試配置，建議在文件中註明用途，避免被誤當 production baseline。

### 4.3 總結判斷

`5132ce64` 對核心問題是**合理且必要**的：  
它不是「為了拉高單次指標的 overfit hack」，而是把 `coors` 契約明文化並在 export/runtime 一致落地。

---

## 5. 為什麼這樣改後 `original` 和 `new export` 都能正常

修正後，三者契約一致：

1. 資料前處理輸出 `coors`（framework path）
2. backend 餵入前的 `coors` 正規化
3. ONNX 圖內對 `coors` 的假設（legacy style）

當這三件事一致，就不會再出現 `PyTorch OK / TRT ~0` 的錯位結果。

---

## 6. 這件事跟 ROS `x/y/z` 有沒有關係？

### 5.1 有關的部分

- ROS/Autoware 的座標系通常使用右手系（例如 `base_link`：`x` 前、`y` 左、`z` 上）。
- `coors` 的欄位命名（`x,y,z` 或 `z,y,x`）在語意上會讓人聯想到座標軸方向。

### 5.2 無直接關聯的部分（本次核心）

本次 bug 的核心不是 ROS frame 本身，而是**稀疏體素索引張量的欄位順序契約**：

- 這是 tensor/算子接口契約問題（graph 外是什麼順序，graph 內期望什麼順序）
- 不等同於「把物理世界座標軸定義改掉」

換句話說：

- ROS frame 定義決定「物理座標如何表示」
- 本次修正決定「已經體素化後的 index tensor 欄位如何對齊算子契約」

兩者相關，但不是同一層問題。

---

## 7. 如何驗證契約是否一致

### 6.1 ONNX 檢查（legacy flip 是否存在）

可檢查 `coors` 路徑是否有 reverse-slice（`steps=-1`）或等價 flip。

### 6.2 Backend 檢查（runtime 是否做對齊）

檢查 ONNX/TRT pipeline 在 `run_bevfusion()` 中，`coors` 是否先經過 legacy 正規化再轉 numpy。

### 6.3 指標檢查

同資料、同 checkpoint：

- PyTorch 應維持正常（作為 reference）
- TRT 與 ONNX 不應再是接近 0 的異常值

---

## 8. 設計原則（後續維護）

1. **單一路徑契約**：不要同時維護「evaluation 專用語義」與「Autoware 實車語義」兩套邏輯。
2. **翻轉只在模型邊界做**：不得在 evaluator 裡補救。
3. **文件化契約**：`coors` 的輸入順序、圖內期望順序、batch 拼接位置必須固定記錄。

---

## 9. 結論

本次修正不是單純「讓分數變好」，而是把 evaluation 對齊回 legacy Autoware-compatible ONNX contract。  
核心是 `coors` 欄位順序契約一致化；ROS `x/y/z` 是背景語意，但本次直接修復點是 tensor index order 對齊，而非改動物理座標系定義。
