# BEVFusion-L：完整架構、Shape 與 TransFusion Head 詳解

> 本文件整理 BEVFusion-L（LiDAR-only）從原始點雲、voxelization、sparse encoder、SECOND/SECONDFPN，到 TransFusion-style detection head、box decode 與 NMS 的完整流程。  
> 數學式使用標準 Markdown LaTeX 語法，可在支援 MathJax / KaTeX 的 Markdown 閱讀器中顯示。

---

## 1. 模型總覽

BEVFusion-L 是 LiDAR-only 版本，因此沒有 camera branch，也沒有 camera-to-BEV pooling。

整體流程：

```text
Raw Point Cloud
      │
      ▼
Hard Voxelization
      │
      ▼
Voxel Feature Encoder
      │
      ▼
Sparse 3D Encoder
      │
      ▼
Dense LiDAR BEV
      │
      ▼
SECOND 2D Backbone
      │
      ▼
SECONDFPN
      │
      ▼
TransFusion-style Head
      │
      ▼
500 Proposals
      │
      ▼
Decode + Circle NMS
      │
      ▼
Final 3D Detections
```

此版本只有 LiDAR feature：

$$
F_{\text{BEV}} = F_{\text{LiDAR}}
$$

---

# 2. 輸入點雲

單一 sample 的 LiDAR point cloud：

$$
P \in \mathbb{R}^{460528 \times 5}
$$

每個 point：

$$
p_i = (x, y, z, \text{intensity}, \text{time lag})
$$

實際 shape：

```text
points = [460528, 5]
```

Point cloud range：

$$
x,y \in [-122.4, 122.4]
$$

$$
z \in [-3.0, 5.0]
$$

Voxel size：

$$
(\Delta x, \Delta y, \Delta z)
=
(0.17, 0.17, 0.2)
$$

因此原始 voxel grid 約為：

$$
1440 \times 1440 \times 41
$$

因為：

$$
\frac{244.8}{0.17} = 1440
$$

---

# 3. Stage A：Voxelization

Voxelization 在 ONNX / TensorRT graph 之外執行。

輸入：

```text
points [460528, 5]
```

輸出：

```text
voxels                  [70747, 32, 5]
coors                   [70747, 3]
num_points_per_voxel    [70747]
```

各維度含義：

- `70747`：這一幀產生的 non-empty voxel 數量。
- `32`：每個 voxel 最多保留 32 個 points。
- `5`：每個 point 的 feature dimension。
- `num_points_per_voxel[i]`：第 `i` 個 voxel 實際包含的有效 point 數量。

因此：

$$
\texttt{num\_points\_per\_voxel.shape} = [70747]
$$

但每個元素的值滿足：

$$
1 \leq \texttt{num\_points\_per\_voxel}[i] \leq 32
$$

例如：

```text
num_points_per_voxel = [3, 1, 32, 7, ...]
```

代表：

| Voxel | Buffer 容量 | 實際有效點數 |
|---|---:|---:|
| voxel 0 | 32 | 3 |
| voxel 1 | 32 | 1 |
| voxel 2 | 32 | 32 |
| voxel 3 | 32 | 7 |

Voxel encoder 做平均時使用：

$$
f_i
=
\frac{
\sum_{j=0}^{31} V_{i,j}
}{
\texttt{num\_points\_per\_voxel}[i]
}
$$

padding 部分通常是 0，因此可以固定對 32 個位置求和，但分母必須是實際有效點數，而不是固定除以 32。

---

## 3.1 稀疏程度

完整 dense voxel grid 有：

$$
1440 \times 1440 \times 41
=
85,017,600
$$

個 cells，但 active voxels 只有：

$$
70,747
$$

Active ratio：

$$
\frac{70,747}{85,017,600}
\approx 0.083\%
$$

這也是 sparse convolution 必要的原因。

---

# 4. Stage B：Voxel Feature Encoder

Voxel encoder：

```text
HardSimpleVoxelSinCosEncoder
```

輸入：

$$
[70747, 32, 5]
$$

輸出：

$$
[70747, 50]
$$

---

## 4.1 Mean pooling

假設第 $n$ 個 voxel 有 $m_n$ 個有效 points：

$$
V_n =
\{p_{n,1}, p_{n,2}, \ldots, p_{n,m_n}\}
$$

每個 feature channel 做平均：

$$
\bar{p}_n
=
\frac{1}{m_n}
\sum_{j=1}^{m_n} p_{n,j}
$$

Shape：

$$
[70747, 32, 5]
\rightarrow
[70747, 5]
$$

得到：

$$
\bar{p}_n =
[
\bar{x},
\bar{y},
\bar{z},
\overline{\text{intensity}},
\overline{\text{time lag}}
]
$$

---

## 4.2 Sin/Cos Fourier encoding

每個原始 channel 使用 5 個頻率：

$$
2^i,
\qquad i \in \{0,1,2,3,4\}
$$

先 normalize：

$$
u_j
=
\frac{\bar{p}_j - \min_j}
{\max_j - \min_j}
$$

再乘：

$$
\pi 2^i
$$

得到：

$$
y_{i,j}
=
u_j \pi 2^i
$$

共有：

$$
5 \text{ channels}
\times
5 \text{ frequencies}
=
25
$$

個值。

再計算：

$$
[\cos(y), \sin(y)]
$$

因此：

$$
25 \times 2 = 50
$$

最後：

```text
voxel_features [70747, 50]
```

完整 shape 流程：

$$
\boxed{
[70747,32,5]
\rightarrow
[70747,5]
\rightarrow
[70747,25]
\rightarrow
[70747,50]
}
$$

---

# 5. Stage C：Sparse 3D Encoder

輸入：

```text
features      [70747, 50]
coordinates   [70747, 4]
spatial grid  [1440, 1440, 41]
```

實際 shape 演化：

| Stage | Active voxels | Channels | Sparse spatial shape |
|---|---:|---:|---|
| Input | 70,747 | 50 | $1440 \times 1440 \times 41$ |
| `conv_input` | 70,747 | 16 | $1440 \times 1440 \times 41$ |
| Layer 1 | 63,710 | 32 | $720 \times 720 \times 21$ |
| Layer 2 | 31,472 | 64 | $360 \times 360 \times 11$ |
| Layer 3 | 12,557 | 128 | $180 \times 180 \times 5$ |
| Layer 4 | 12,557 | 128 | $180 \times 180 \times 5$ |
| `conv_out` | 9,266 | 128 | $180 \times 180 \times 2$ |

在 $x,y$ 方向總 downsample 8 倍：

$$
1440 \rightarrow 720 \rightarrow 360 \rightarrow 180
$$

$$
\frac{1440}{8} = 180
$$

高度方向：

$$
41 \rightarrow 21 \rightarrow 11 \rightarrow 5 \rightarrow 2
$$

---

## 5.1 Sparse convolution 的概念

每一層在 TensorRT 中大致拆成：

```text
GetIndicePairsImplicitGemm
             │
             ▼
ImplicitGemm
```

`GetIndicePairsImplicitGemm` 建立 sparse input-output mapping。

`ImplicitGemm` 執行真正的 convolution。

概念上：

$$
y_j
=
\sum_{k \in \mathcal{K}}
W_k x_{M(j,k)}
$$

其中：

- $j$：output active voxel。
- $k$：kernel offset。
- $M(j,k)$：對應的 input active voxel。
- $W_k$：該 kernel offset 的權重。

---

# 6. Stage D：Sparse to Dense BEV

Sparse encoder 最後輸出：

```text
active features  [9266, 128]
spatial shape    [180, 180, 2]
```

完整 dense cells：

$$
180 \times 180 \times 2
=
64800
$$

建立：

```text
dense table [64800, 128]
```

初始全部為 0，再用 `ScatterElements` 把 9,266 個 active features 寫入對應位置：

$$
D[\text{linear index}_i] = f_i
$$

空位置保持 0。

Shape 變化：

$$
[64800,128]
$$

reshape：

$$
[1,180,180,2,128]
$$

transpose：

$$
[1,128,2,180,180]
$$

把 $Z=2$ 合併進 channel：

$$
[1,128,2,180,180]
\rightarrow
[1,256,180,180]
$$

因此：

$$
\boxed{
F_{\text{LiDAR BEV}}
\in
\mathbb{R}^{1 \times 256 \times 180 \times 180}
}
$$

其中：

$$
256
=
128 \text{ feature channels}
\times
2 \text{ height slices}
$$

---

# 7. Stage E：SECOND 2D Backbone

輸入：

```text
lidar_bev [1,256,180,180]
```

Block 0：

$$
[1,256,180,180]
\rightarrow
[1,128,180,180]
$$

Block 1：

$$
[1,128,180,180]
\rightarrow
[1,256,90,90]
$$

因此 backbone 產生兩個尺度：

$$
F_1
\in
\mathbb{R}^{1 \times 128 \times 180 \times 180}
$$

$$
F_2
\in
\mathbb{R}^{1 \times 256 \times 90 \times 90}
$$

---

# 8. Stage F：SECONDFPN

第一個 feature：

$$
[1,128,180,180]
\rightarrow
[1,256,180,180]
$$

第二個 feature：

$$
[1,256,90,90]
\xrightarrow{\text{ConvTranspose}}
[1,256,180,180]
$$

Channel concatenate：

$$
[1,256,180,180]
\oplus
[1,256,180,180]
$$

得到：

$$
\boxed{
F_{\text{neck}}
\in
\mathbb{R}^{1 \times 512 \times 180 \times 180}
}
$$

---

# 9. Detection Head 總覽

Head 輸入：

$$
F_{\text{neck}}
\in
\mathbb{R}^{1 \times 512 \times 180 \times 180}
$$

輸出：

```text
bbox_pred   [10,500]
score       [500]
label_pred  [500]
```

流程：

```text
Shared Conv
    │
    ▼
Dense Class Heatmap
    │
    ▼
Local Maximum Filtering
    │
    ▼
Top-500 Query Selection
    │
    ▼
Query Feature Initialization
    │
    ▼
Transformer Decoder
    │
    ▼
Separate Prediction Heads
    │
    ▼
500 Boxes + Scores + Labels
```

---

# 10. Shared BEV Feature

Shared convolution：

$$
[1,512,180,180]
\rightarrow
[1,128,180,180]
$$

記為：

$$
F_s
\in
\mathbb{R}^{1 \times 128 \times 180 \times 180}
$$

每個 BEV cell 有一個 128 維 feature：

$$
f_{x,y}
\in
\mathbb{R}^{128}
$$

總 cell 數：

$$
180 \times 180 = 32400
$$

可以理解為：

```text
32,400 個 BEV 位置
每個位置有一個 128 維描述向量
```

---

# 11. Dense Heatmap

模型有 7 個類別：

```text
0 car
1 truck
2 bus
3 bicycle
4 pedestrian
5 traffic_cone
6 barrier
```

Heatmap logits：

$$
H_{\text{logit}}
\in
\mathbb{R}^{1 \times 7 \times 180 \times 180}
$$

經 sigmoid：

$$
H = \sigma(H_{\text{logit}})
$$

其中：

$$
H[c,y,x]
$$

表示：

> 模型認為 BEV 位置 $(x,y)$ 是類別 $c$ 的物體中心的可能性。

例如：

```text
H[car, 80, 90]   = 0.92
H[truck, 80, 90] = 0.08
H[bus, 80, 90]   = 0.03
```

---

# 12. Local-Max Filtering

一個物體可能讓相鄰多個 cell 都有高分，例如：

$$
\begin{bmatrix}
0.62 & 0.75 & 0.68 \\
0.71 & 0.92 & 0.74 \\
0.65 & 0.78 & 0.69
\end{bmatrix}
$$

使用 $3 \times 3$ max-pooling：

$$
M = \operatorname{MaxPool}_{3 \times 3}(H)
$$

只保留局部最大值：

$$
H_{\text{peak}}
=
\begin{cases}
H, & H = M \\
0, & H \neq M
\end{cases}
$$

結果：

$$
\begin{bmatrix}
0 & 0 & 0 \\
0 & 0.92 & 0 \\
0 & 0 & 0
\end{bmatrix}
$$

此模型只對以下 crowded classes 做 local pooling：

```text
car
truck
bus
barrier
```

這一步只是 proposal suppression，不是最終 NMS。

---

# 13. Flatten Heatmap

原始 heatmap：

$$
[1,7,180,180]
$$

每個 class 的空間 flatten：

$$
[180,180]
\rightarrow
[32400]
$$

因此：

$$
[1,7,180,180]
\rightarrow
[1,7,32400]
$$

對應線性位置：

$$
p = y \times 180 + x
$$

反解：

$$
x = p \bmod 180
$$

$$
y =
\left\lfloor
\frac{p}{180}
\right\rfloor
$$

再把 class 與 position 合併：

$$
7 \times 32400 = 226800
$$

所以：

$$
[1,7,32400]
\rightarrow
[1,226800]
$$

這 226,800 個元素代表所有：

$$
(\text{class}, \text{BEV position})
$$

組合。

---

# 14. Top-500 Query Selection

TopK 選的是：

> 500 個最高分的「類別 + 位置」組合。

輸出：

```text
topk_score   [1,500]
topk_index   [1,500]
```

對 flattened index $i$：

$$
\text{class id}
=
\left\lfloor
\frac{i}{32400}
\right\rfloor
$$

$$
\text{position id}
=
i \bmod 32400
$$

再解出：

$$
x = \text{position id} \bmod 180
$$

$$
y =
\left\lfloor
\frac{\text{position id}}{180}
\right\rfloor
$$

---

## 14.1 實際例子

假設：

$$
i = 97110
$$

Class：

$$
c
=
\left\lfloor
\frac{97110}{32400}
\right\rfloor
=
2
$$

Class 2 是 bus。

Position：

$$
p
=
97110 \bmod 32400
=
32310
$$

$$
x
=
32310 \bmod 180
=
90
$$

$$
y
=
\left\lfloor
\frac{32310}{180}
\right\rfloor
=
179
$$

因此：

```text
class = bus
BEV position = (x=90, y=179)
```

這只表示：

> Bus heatmap 認為位置 $(90,179)$ 很可能是 bus center，因此將它選為一個 object query proposal。

---

# 15. 建立 Object Query

Top-K 找到 500 個位置後，回到 shared BEV feature：

$$
F_s
\in
\mathbb{R}^{1 \times 128 \times 180 \times 180}
$$

將 spatial flatten：

$$
F_s^{\text{flat}}
\in
\mathbb{R}^{1 \times 128 \times 32400}
$$

對 500 個位置 gather：

$$
Q_{\text{feat}}
\in
\mathbb{R}^{1 \times 128 \times 500}
$$

單一 query：

$$
q_i
=
F_s[:,y_i,x_i]
$$

$$
q_i
\in
\mathbb{R}^{128}
$$

Transformer 常轉置為：

$$
Q
\in
\mathbb{R}^{1 \times 500 \times 128}
$$

---

## 15.1 為什麼不能只使用 heatmap score？

Heatmap score 只有一個 scalar，例如：

$$
0.92
$$

它只表示「這裡可能有物體」。

但 box prediction 還需要：

- 長度。
- 寬度。
- 高度。
- 朝向。
- 速度。
- 類別細節。

因此需要取回該位置完整的 128 維 BEV feature。

---

# 16. Query Position Embedding

每個 query 有一個 grid position：

$$
p_i = (x_i,y_i)
$$

使用 learned position embedding：

$$
e_i^{\text{pos}}
=
\phi_{\text{self-pos}}(x_i,y_i)
$$

輸出：

$$
e_i^{\text{pos}}
\in
\mathbb{R}^{128}
$$

所有 query 的 position embedding：

$$
E_{\text{query-pos}}
\in
\mathbb{R}^{1 \times 128 \times 500}
$$

這讓 transformer 知道：

> 每個 query 在 BEV 空間中的位置。

Position embedding 使用的是 feature-grid coordinate：

$$
x,y \in [0,179]
$$

此時尚未轉換成 metric coordinate。

---

# 17. Class Embedding

每個 query 是從某個 class heatmap 選出的。

例如：

```text
query 0 → car
query 1 → truck
query 2 → pedestrian
```

先轉成 one-hot：

$$
o_i \in \mathbb{R}^{7}
$$

例如 car：

$$
o_{\text{car}}
=
[1,0,0,0,0,0,0]
$$

再經 learned projection：

$$
e_i^{\text{class}}
=
W_{\text{class}} o_i
$$

$$
e_i^{\text{class}}
\in
\mathbb{R}^{128}
$$

加入 query content：

$$
q_i^0
=
q_i + e_i^{\text{class}}
$$

因此 query 帶有：

1. 該位置的 BEV content。
2. 自己在 BEV 中的位置。
3. 初始 class prior。

---

# 18. BEV Memory

初始 query 只 gather 一個中心 cell：

$$
q_i = F_s(x_i,y_i)
$$

但一個物體通常跨越多個 cells。

Head grid 每 cell 約對應：

$$
8 \times 0.17 = 1.36 \text{ m}
$$

一台長 4.5 m 的車大約跨越：

$$
\frac{4.5}{1.36}
\approx
3.3
$$

個 cells。

因此 transformer 保留完整 BEV feature 作為 memory。

原始：

$$
F_s
\in
\mathbb{R}^{1 \times 128 \times 180 \times 180}
$$

Flatten：

$$
M
\in
\mathbb{R}^{1 \times 128 \times 32400}
$$

Transformer 格式：

$$
M
\in
\mathbb{R}^{1 \times 32400 \times 128}
$$

總共有：

$$
32400
$$

個 BEV memory tokens。

---

## 18.1 BEV Position Embedding

每個 BEV memory token 也有位置：

$$
e_j^{\text{BEV-pos}}
=
\phi_{\text{cross-pos}}(x_j,y_j)
$$

輸出：

$$
E_{\text{BEV-pos}}
\in
\mathbb{R}^{1 \times 32400 \times 128}
$$

因此 cross-attention 能同時理解：

- memory feature 的內容。
- memory feature 在 BEV 中的位置。

---

# 19. Transformer Decoder

模型設定：

```text
num_decoder_layers = 1
num_proposals      = 500
hidden_channel     = 128
num_heads          = 8
```

每個 attention head 的 dimension：

$$
d_h
=
\frac{128}{8}
=
16
$$

Transformer decoder 包含：

1. Self-attention。
2. Cross-attention。
3. Feed-forward network。
4. Residual connection。
5. Layer normalization。

---

# 20. Self-Attention

Self-attention 讓 500 個 object queries 彼此交換資訊。

輸入：

$$
Q
\in
\mathbb{R}^{500 \times 128}
$$

線性投影：

$$
Q_s = QW_Q
$$

$$
K_s = QW_K
$$

$$
V_s = QW_V
$$

每個 head：

$$
Q_s^{(h)},
K_s^{(h)},
V_s^{(h)}
\in
\mathbb{R}^{500 \times 16}
$$

Attention matrix：

$$
A_{\text{self}}^{(h)}
=
\operatorname{softmax}
\left(
\frac{
Q_s^{(h)} K_s^{(h)\top}
}{
\sqrt{16}
}
\right)
$$

Shape：

$$
[500,16]
\times
[16,500]
=
[500,500]
$$

8 個 heads：

$$
[8,500,500]
$$

第 $(i,j)$ 個值代表：

> Query $i$ 在更新自己時，要參考 query $j$ 多少。

---

## 20.1 Self-Attention 範例

假設只有 3 個 queries：

```text
q0：car proposal at (50,50)
q1：car proposal at (51,50)
q2：pedestrian proposal at (100,120)
```

Attention 可能是：

$$
A_{\text{self}}
=
\begin{bmatrix}
0.55 & 0.40 & 0.05 \\
0.42 & 0.53 & 0.05 \\
0.05 & 0.05 & 0.90
\end{bmatrix}
$$

$q_0$ 和 $q_1$：

- 距離近。
- feature 相似。
- class 相同。

因此彼此 attention 較強。

Self-attention 不會直接刪掉重複 query，它只是讓 proposals 知道其他 proposals 的存在。

後續模型可能因此：

- 降低其中一個 query 的分數。
- 修改其 box regression。
- 配合 matching 與 NMS 減少重複。

---

# 21. Cross-Attention

Cross-attention 讓每個 object query 回頭查看完整 BEV memory。

Query：

$$
Q
\in
\mathbb{R}^{500 \times 128}
$$

BEV memory：

$$
M
\in
\mathbb{R}^{32400 \times 128}
$$

每個 head：

$$
Q_c^{(h)}
\in
\mathbb{R}^{500 \times 16}
$$

$$
K_c^{(h)},
V_c^{(h)}
\in
\mathbb{R}^{32400 \times 16}
$$

Attention matrix：

$$
A_{\text{cross}}^{(h)}
=
\operatorname{softmax}
\left(
\frac{
Q_c^{(h)} K_c^{(h)\top}
}{
\sqrt{16}
}
\right)
$$

Shape：

$$
[500,16]
\times
[16,32400]
=
[500,32400]
$$

8 heads：

$$
[8,500,32400]
$$

每一列表示：

> 某個 query 對完整 32,400 個 BEV cells 的關注分布。

---

## 21.1 單一 Query 的 Cross-Attention

假設 car query 位於：

$$
(x=90,y=80)
$$

它可能關注：

```text
(90,80) center      0.20
(89,80) left side   0.12
(91,80) right side  0.14
(90,79) front       0.18
(90,81) rear        0.16
other locations     0.20 total
```

更新後：

$$
q_i^{\text{cross}}
=
\sum_{j=1}^{32400}
a_{ij} v_j
$$

其中：

$$
\sum_{j=1}^{32400} a_{ij} = 1
$$

因此 query 不只使用中心 cell，而是使用整張 BEV 的加權摘要。

---

## 21.2 Cross-Attention 的矩陣乘法

Attention score：

$$
[500,16]
\times
[16,32400]
=
[500,32400]
$$

再乘 value：

$$
[500,32400]
\times
[32400,16]
=
[500,16]
$$

8 個 heads 各輸出 16 維，concatenate：

$$
8 \times 16 = 128
$$

最終回到：

$$
[500,128]
$$

---

# 22. Self-Attention 與 Cross-Attention 比較

| 項目 | Self-attention | Cross-attention |
|---|---|---|
| Query 來源 | 500 object queries | 500 object queries |
| Key/Value 來源 | 500 object queries | 32,400 BEV tokens |
| 每 head attention shape | `[500,500]` | `[500,32400]` |
| 主要作用 | 理解 proposals 間關係 | 從完整場景讀取資訊 |
| 問題 | 其他候選跟我有什麼關係？ | BEV 哪些位置支持我的判斷？ |

簡化：

```text
Self-attention：
物體候選彼此看。

Cross-attention：
物體候選回頭看場景。
```

---

# 23. Residual Connection 與 LayerNorm

Self-attention：

$$
Q_1
=
\operatorname{LN}
\left(
Q_0
+
\operatorname{SelfAttn}(Q_0)
\right)
$$

代表：

```text
更新後 query
=
原始 query
+
從其他 queries 取得的新資訊
```

Cross-attention：

$$
Q_2
=
\operatorname{LN}
\left(
Q_1
+
\operatorname{CrossAttn}(Q_1,M)
\right)
$$

代表：

```text
query 原有資訊
+
從完整 BEV 讀到的新資訊
```

Residual connection 保留原始 proposal feature。

LayerNorm 提升訓練與數值穩定性。

---

# 24. Feed-Forward Network

FFN：

$$
128 \rightarrow 256 \rightarrow 128
$$

對每個 query 獨立執行：

$$
q_i'
=
W_2
\operatorname{ReLU}
(W_1 q_i + b_1)
+
b_2
$$

FFN 不負責 query 間資訊交換；資訊交換已經由 attention 完成。

FFN 的作用是：

> 將 attention 收集到的資訊重新轉換成更適合 box prediction 的 representation。

完整 decoder：

$$
Q_1
=
\operatorname{LN}
\left(
Q_0
+
\operatorname{SelfAttn}(Q_0)
\right)
$$

$$
Q_2
=
\operatorname{LN}
\left(
Q_1
+
\operatorname{CrossAttn}(Q_1,M)
\right)
$$

$$
Q_3
=
\operatorname{LN}
\left(
Q_2
+
\operatorname{FFN}(Q_2)
\right)
$$

最終：

$$
Q_3
\in
\mathbb{R}^{1 \times 500 \times 128}
$$

---

# 25. Transformer Decoder 的白話解釋

初始 query：

```text
我是在位置 (90,80) 找到的 car 候選。
這個位置的 BEV feature 是一個 128 維向量。
```

Self-attention 後：

```text
我發現附近還有另一個很相似的 car 候選，
遠方則有其他不相關候選。
```

Cross-attention 後：

```text
我查看完整 BEV，
發現中心前後左右都有支持一台車的 feature，
因此能推斷物體範圍與方向。
```

FFN 後：

```text
我把收集到的資訊整理成適合預測
center、dimension、rotation、velocity 的 128 維特徵。
```

---

# 26. Separate Prediction Heads

Decoded query：

$$
Q_3
\in
\mathbb{R}^{1 \times 500 \times 128}
$$

通常轉回：

$$
[1,128,500]
$$

每個 branch 大致是：

```text
Conv1d 128 → 64
ReLU
Conv1d 64 → output channels
```

輸出：

| Branch | Shape | 含義 |
|---|---|---|
| `center` | `[1,2,500]` | BEV center |
| `height` | `[1,1,500]` | gravity-center z |
| `dim` | `[1,3,500]` | log box dimensions |
| `rot` | `[1,2,500]` | $(\sin\theta,\cos\theta)$ |
| `vel` | `[1,2,500]` | $(v_x,v_y)$ |
| `heatmap` | `[1,7,500]` | refined class logits |

Regression channels：

$$
2 + 1 + 3 + 2 + 2 = 10
$$

所以：

$$
bbox_{\text{pred}}
\in
\mathbb{R}^{10 \times 500}
$$

單一 query：

$$
\hat{b}_i
=
[
c_x,
c_y,
z,
\log d_x,
\log d_y,
\log d_z,
\sin\theta,
\cos\theta,
v_x,
v_y
]
$$

---

# 27. Center Prediction

Center branch 預測：

$$
\Delta c_i
=
(\Delta x_i,\Delta y_i)
$$

再加回 query position：

$$
c_i^{\text{feat}}
=
p_i
+
\Delta c_i
$$

例如：

$$
p_i = (80,100)
$$

$$
\Delta c_i = (0.25,-0.30)
$$

則：

$$
c_i^{\text{feat}}
=
(80.25,99.70)
$$

此時仍是 feature-map coordinate。

---

# 28. Score 計算

Head 有兩套分類資訊。

Proposal score：

$$
s_i^{\text{proposal}}
$$

來自 dense heatmap Top-K。

Transformer 後的 query classification：

$$
L_i \in \mathbb{R}^{7}
$$

經 sigmoid：

$$
s_{i,c}^{\text{query}}
=
\sigma(L_{i,c})
$$

Query 有 initial class：

$$
c_i^{\text{proposal}}
$$

One-hot：

$$
o_{i,c}
=
\begin{cases}
1, & c = c_i^{\text{proposal}} \\
0, & \text{otherwise}
\end{cases}
$$

最終：

$$
s_{i,c}^{\text{final}}
=
s_{i,c}^{\text{query}}
\cdot
s_i^{\text{proposal}}
\cdot
o_{i,c}
$$

最後：

$$
score_i
=
\max_c s_{i,c}^{\text{final}}
$$

Label：

$$
label_i
=
c_i^{\text{proposal}}
$$

---

# 29. Feature Coordinate 轉 Metric Coordinate

Feature map stride：

$$
\text{out size factor} = 8
$$

Voxel size：

$$
0.17 \text{ m}
$$

每個 head BEV cell 對應：

$$
8 \times 0.17
=
1.36 \text{ m}
$$

Center decode：

$$
x_{\text{metric}}
=
x_{\text{feature}}
\cdot
8
\cdot
0.17
+
x_{\min}
$$

$$
y_{\text{metric}}
=
y_{\text{feature}}
\cdot
8
\cdot
0.17
+
y_{\min}
$$

其中：

$$
x_{\min}
=
y_{\min}
=
-122.4
$$

---

## 29.1 Center Decode 範例

假設：

$$
c_x^{\text{feat}} = 80.25
$$

$$
c_y^{\text{feat}} = 99.70
$$

則：

$$
x_{\text{metric}}
=
80.25 \times 1.36 - 122.4
=
-13.26 \text{ m}
$$

$$
y_{\text{metric}}
=
99.70 \times 1.36 - 122.4
=
13.192 \text{ m}
$$

---

# 30. Dimension Decode

Head 預測 log dimension：

$$
[
\log d_x,
\log d_y,
\log d_z
]
$$

實際尺寸：

$$
d_x = \exp(\log d_x)
$$

$$
d_y = \exp(\log d_y)
$$

$$
d_z = \exp(\log d_z)
$$

例如：

$$
[1.435,0.531,0.445]
$$

則：

$$
d_x \approx 4.20
$$

$$
d_y \approx 1.70
$$

$$
d_z \approx 1.56
$$

---

# 31. Rotation Decode

Head 預測：

$$
(r_s,r_c)
=
(\sin\theta,\cos\theta)
$$

Yaw：

$$
\theta
=
\operatorname{atan2}(r_s,r_c)
$$

例如：

$$
r_s = -0.017
$$

$$
r_c = 0.999
$$

則：

$$
\theta
\approx
-0.017 \text{ rad}
$$

---

# 32. Height Decode

Head 預測 gravity center：

$$
z_g
$$

最終 box 使用 bottom center：

$$
z_{\text{bottom}}
=
z_g
-
\frac{d_z}{2}
$$

例如：

$$
z_g = 0.83
$$

$$
d_z = 1.56
$$

則：

$$
z_{\text{bottom}}
=
0.83 - 0.78
=
0.05
$$

---

# 33. 單一 Proposal 完整數值例子

假設：

```text
query position = (80,100)
proposal class = car
proposal score = 0.96
```

Center offset：

$$
\Delta x = 0.25
$$

$$
\Delta y = -0.30
$$

因此：

$$
c_x = 80.25
$$

$$
c_y = 99.70
$$

Height：

$$
z_g = 0.83
$$

Dimension logits：

$$
[1.435,0.531,0.445]
$$

Decode：

$$
[d_x,d_y,d_z]
=
[4.20,1.70,1.56]
$$

Rotation：

$$
[\sin\theta,\cos\theta]
=
[-0.017,0.999]
$$

$$
\theta
=
-0.017
$$

Velocity：

$$
[v_x,v_y]
=
[2.1,0.1]
$$

Query classification car score：

$$
s_{\text{car}}^{\text{query}}
=
0.958
$$

Final score：

$$
s^{\text{final}}
=
0.958 \times 0.96
\approx
0.920
$$

Metric center：

$$
x
=
80.25 \times 1.36 - 122.4
=
-13.26
$$

$$
y
=
99.70 \times 1.36 - 122.4
=
13.19
$$

Bottom z：

$$
z
=
0.83 - \frac{1.56}{2}
=
0.05
$$

最終 box：

$$
\boxed{
[
-13.26,
13.19,
0.05,
4.20,
1.70,
1.56,
-0.017,
2.1,
0.1
]
}
$$

Score：

$$
0.920
$$

Label：

```text
car
```

---

# 34. Postprocess

Network 固定輸出 500 個 proposals。

後處理：

```text
Box Decode
    │
    ▼
Per-Class Score Threshold
    │
    ▼
Post-Center-Range Filtering
    │
    ▼
Circle NMS
    │
    ▼
Final Detections
```

Class thresholds：

```text
car          0.015
truck        0.010
bus          0.010
bicycle      0.020
pedestrian   0.030
traffic_cone 0.040
barrier      0.020
```

Sample 0：

```text
PyTorch FP32 → 78 detections
TensorRT FP16 → 77 detections
```

---

# 35. 完整 Shape Walkthrough

```text
Raw points
[460528,5]
    │
    ▼
Hard voxelization
    │
    ├── voxels [70747,32,5]
    ├── coors [70747,3]
    └── num_points [70747]
    │
    ▼
Mean pooling
[70747,5]
    │
    ▼
Sin/Cos Fourier encoding
[70747,50]
    │
    ▼
Sparse 3D encoder
70747×16
→ 63710×32
→ 31472×64
→ 12557×128
→ 9266×128
    │
    ▼
Scatter + Z collapse
[1,256,180,180]
    │
    ▼
SECOND backbone
├── [1,128,180,180]
└── [1,256,90,90]
    │
    ▼
SECONDFPN
[1,512,180,180]
    │
    ▼
Shared convolution
[1,128,180,180]
    │
    ├───────────────────────────────────┐
    │                                   │
    ▼                                   │
Dense heatmap                           │
[1,7,180,180]                           │
    │                                   │
    ▼                                   │
Local peak filtering                    │
    │                                   │
    ▼                                   │
Flatten                                 │
[1,226800]                              │
    │                                   │
    ▼                                   │
TopK 500                                │
    │                                   │
    ├── class [1,500]                   │
    ├── position [1,500]                │
    └── proposal score [1,500]          │
                                        │
    ┌──────── gather features ──────────┘
    ▼
Initial queries
[1,128,500]
    │
    ├── class embedding [1,128,500]
    └── position embedding [1,128,500]
    │
    ▼
Transformer format
[1,500,128]
    │
    ▼
Self-attention
8 × [500,500]
    │
    ▼
Cross-attention
8 × [500,32400]
    │
    ▼
FFN
128 → 256 → 128
    │
    ▼
Decoded queries
[1,500,128]
    │
    ├── center  [1,2,500]
    ├── height  [1,1,500]
    ├── dim     [1,3,500]
    ├── rot     [1,2,500]
    ├── vel     [1,2,500]
    └── heatmap [1,7,500]
    │
    ▼
bbox_pred [10,500]
score     [500]
label     [500]
    │
    ▼
Decode + threshold + circle NMS
    │
    ▼
77–78 final detections
```

---

# 36. 最精簡數學表示

Voxel 與 sparse encoder：

$$
F_{3D}
=
E_{\text{sparse}}
\left(
E_{\text{voxel}}
(
\operatorname{Voxelize}(P)
)
\right)
$$

轉成 BEV：

$$
F_{\text{BEV}}
=
\operatorname{CollapseZ}
\left(
\operatorname{ScatterDense}(F_{3D})
\right)
$$

2D backbone：

$$
F_{\text{neck}}
=
\operatorname{SECONDFPN}
\left(
\operatorname{SECOND}(F_{\text{BEV}})
\right)
$$

Dense proposal：

$$
H
=
\sigma
\left(
H_{\text{dense}}(F_{\text{neck}})
\right)
$$

Query selection：

$$
(c_i,p_i)
=
\operatorname{DecodeIndex}
\left(
\operatorname{TopK}(H,500)
\right)
$$

Query initialization：

$$
q_i^0
=
F_s(p_i)
+
E_{\text{class}}(c_i)
$$

Transformer：

$$
Q_1
=
\operatorname{LN}
\left(
Q_0
+
\operatorname{SelfAttn}(Q_0)
\right)
$$

$$
Q_2
=
\operatorname{LN}
\left(
Q_1
+
\operatorname{CrossAttn}(Q_1,M)
\right)
$$

$$
Q_3
=
\operatorname{LN}
\left(
Q_2
+
\operatorname{FFN}(Q_2)
\right)
$$

Box prediction：

$$
B
=
H_{\text{box}}(Q_3)
$$

最終 detection：

$$
\mathcal{D}
=
\operatorname{NMS}
\left(
\operatorname{Decode}(B)
\right)
$$

---

# 37. 核心理解

整個 head 並不是：

```text
TopK 找到位置
→ 直接輸出 box
```

而是：

```text
TopK：
從 226,800 個 class-position 候選中，
選出 500 個粗略 object proposals。

Gather：
取出每個 proposal 所在 cell 的 128 維內容。

Class / Position Encoding：
告訴 query 自己可能是哪一類、位於哪裡。

Self-Attention：
讓 500 個 proposals 彼此交換資訊。

Cross-Attention：
讓每個 proposal 從完整 32,400-cell BEV 場景收集資訊。

FFN：
把收集到的資訊轉成適合 3D box regression 的 feature。

Separate Heads：
預測 center、height、dimension、rotation、velocity 與 class score。
```

最核心的一句話：

> Dense heatmap 負責找候選；object queries 負責表示候選；self-attention 負責候選之間的關係；cross-attention 負責從完整 BEV 中收集資訊；prediction heads 負責輸出最終 3D box。
