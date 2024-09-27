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

- YOLOX_opt

We add YOLOX_opt and deploy pipeline for fine detector model for traffic light recognition in Autoware.
In detail, see https://github.com/tier4/autoware-ml/pull/143.

## Next release
### Core library

- Update dataset
  - <https://github.com/tier4/autoware-ml/pull/129>
  - <https://github.com/tier4/autoware-ml/pull/147>
- Move dataset documents into yaml files
  - <https://github.com/tier4/autoware-ml/pull/144>

### Tools

- fix CI of dataset update
  - <https://github.com/tier4/autoware-ml/pull/153>
- Add scene_selector
  - <https://github.com/tier4/autoware-ml/pull/156>

### Pipelines

### Projects

- Add TransFusion-L model t4base_90m/v1
  - <https://github.com/tier4/autoware-ml/pull/125>
- Add TransFusion-L model t4x2_90m/v1
  - <https://github.com/tier4/autoware-ml/pull/126>
- Add BEVFusion-CL model t4xx1_120m/v1
  - <https://github.com/tier4/autoware-ml/pull/141>
- Add SparseConvolutions operation
  - <https://github.com/tier4/autoware-ml/pull/152>
  - When many projects use this operation, we move it to `autoware_ml`.

### Chore

- Update docs
  - <https://github.com/tier4/autoware-ml/pull/155>
