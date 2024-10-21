## Main topic of next release

We plan for next release v0.3.0 as below.

- BEVFusion-CL

As main topics of this version, we release BEVFusion-CL model.
We evaluate BEVFusion-L (BEVFusion LiDAR-only model) and BEVFusion-CL (BEVFusion Camera-LiDAR fusion model) for T4dataset.
As mAP of NuScenes increases from 64.6pt to 66.1pt between BEVFusion-L and BEVFusion-CL, T4dataset increases from 60.6pt to 63.7pt
In detail, see https://github.com/tier4/autoware-ml/pull/141.

| model                             | range  | mAP  | car  | truck | bus  | bicycle | pedestrian |
| --------------------------------- | ------ | ---- | ---- | ----- | ---- | ------- | ---------- |
| BEVFusion-L t4xx1_120m_v1         | 122.4m | 60.6 | 74.1 | 54.1  | 58.7 | 55.7    | 60.3       |
| BEVFusion-CL t4xx1_fusion_120m_v1 | 122.4m | 63.7 | 72.4 | 56.8  | 71.8 | 62.1    | 55.3       |

- TransFusion-L

We add base model and X2 model.
In detail, see https://github.com/tier4/autoware-ml/pull/125 and https://github.com/tier4/autoware-ml/pull/126.

| TransFusion-L 90m | train        | eval | All  | car  | truck | bus  | bicycle | pedestrian |
| ----------------- | ------------ | ---- | ---- | ---- | ----- | ---- | ------- | ---------- |
| t4base_90m/v1     | XX1 + X2     | XX1  | 67.4 | 80.7 | 56.0  | 77.6 | 57.4    | 65.5       |
| t4xx1_90m/v2      | XX1          | XX1  | 68.1 | 80.5 | 58.0  | 80.8 | 58.0    | 63.2       |
| t4xx1_90m/v3      | XX1 + XX1new | XX1  | 68.5 | 81.7 | 62.4  | 83.5 | 50.9    | 64.1       |
| t4base_90m/v1     | XX1 + X2     | X2   | 66.0 | 82.3 | 47.5  | 83.6 | 55.1    | 61.6       |
| t4x2_90m/v1       | X2           | X2   | 58.5 | 80.5 | 28.1  | 82.4 | 48.0    | 53.7       |

- YOLOX_opt

We add YOLOX_opt and deploy pipeline for fine detector model for traffic light recognition in Autoware.
In detail, see https://github.com/tier4/autoware-ml/pull/143.

- Add scene_selector

As first prototype, we integrate scene_selector to use active learning pipeline.
In detail, see https://github.com/tier4/autoware-ml/pull/165.

## Next release
### Core library

- Update dataset
  - <https://github.com/tier4/autoware-ml/pull/129>
  - <https://github.com/tier4/autoware-ml/pull/147>
  - <https://github.com/tier4/autoware-ml/pull/157>
  - <https://github.com/tier4/autoware-ml/pull/164>
- Add docker environment with ROS2
  - <https://github.com/tier4/autoware-ml/pull/160>
- Move dataset documents into yaml files
  - <https://github.com/tier4/autoware-ml/pull/144>

### Tools

- fix CI of dataset update
  - <https://github.com/tier4/autoware-ml/pull/153>
- Add scene_selector
  - <https://github.com/tier4/autoware-ml/pull/156>
  - <https://github.com/tier4/autoware-ml/pull/161>
  - <https://github.com/tier4/autoware-ml/pull/165>

### Pipelines

### Projects

- TransFusion
  - <https://github.com/tier4/autoware-ml/pull/163>
    - Fix sigmoid calculation
  - <https://github.com/tier4/autoware-ml/pull/125>
    - Add TransFusion-L model t4base_90m/v1
  - <https://github.com/tier4/autoware-ml/pull/126>
    - Add TransFusion-L model t4x2_90m/v1
- BEVFusion
  - <https://github.com/tier4/autoware-ml/pull/141>
    - Add BEVFusion-CL model t4xx1_120m/v1
- SparseConvolution
  - <https://github.com/tier4/autoware-ml/pull/163>
    - Add SparseConvolutions operation
    - When many projects use this operation, we move it to `autoware_ml`.
- YOLOX_opt
  - <https://github.com/tier4/autoware-ml/pull/143>

### Chore

- Update docs
  - <https://github.com/tier4/autoware-ml/pull/155>
