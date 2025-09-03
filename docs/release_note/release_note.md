## Release note for old version

- [Release note for v0](/docs/release_note/release_note_v0.md)

## Next release

## v1.1.0

- Publish arxiv paper

https://arxiv.org/abs/2506.00645
We are glad to announce publishing the arXiv paper.
If you are interested in the paper, please read and share it!

![Screenshot from 2025-06-16 14-59-26](https://github.com/user-attachments/assets/3e92f640-7ccc-42d8-810b-db60727417a4)

- Update CenterPoint model from base/1.3 to base/1.7 and fine-tuning for x2/1.7.1
  - Result for J6Gen2
  - CenterPoint base/1.7 (A)  Without feature of distance between of each pointcloud to center of pillar in Z
  - CenterPoint base/1.7 (B)  With feature of distance between of each pointcloud to center of pillar in Z
  - CenterPoint base/1.7.1 (C) Fine-tune from (B) on J6gen2 data without intensity
  - CenterPoint base/1.7.1 (D) Fine-tune from (B) on J6gen2 data with intensity

| eval range: 120m             | mAP       | car <br> (28,002) | truck <br> (1,123) | bus <br> (1,203) | bicycle <br> (223) | pedestrian <br> (4,007) |
| ---------------------------- | --------- | ----------------- | ------------------ | ---------------- | ------------------ | ----------------------- |
| CenterPoint base/1.6         | 72.20     | 86.17             | 52.62              | 85.53            | 74.99              | 61.69                   |
| CenterPoint base/1.7 (A)     | 71.04     | 85.94             | 53.53              | 84.91            | 69.01              | 61.81                   |
| CenterPoint base/1.7 (B)     | 72.29     | 85.96             | 53.52              | 84.21            | 74.96              | 62.79                   |
| CenterPoint J6Gen2/1.7.1 (C) | 72.94     | 86.96             | 57.88              | **85.09**        | 71.62              | 63.17                   |
| CenterPoint J6Gen2/1.7.1 (D) | **74.64** | **87.87**         | **59.87**          | 84.50            | **77.35**          | **63.62**               |

- Add product-release model as model versioning

We add product-release model as model versioning.
If you are interested in the model versioning, please see https://github.com/tier4/AWML/pull/52.

- Add StreamPETR model

We add StreamPETR as new multi-camera 3D object detection.
If you want to know in detail, please see https://github.com/tier4/AWML/pull/57.

![image](https://github.com/user-attachments/assets/4078ce04-5836-49d5-8261-a6fcd20d8e82)

### Core library

- Add 2D detection library
  - https://github.com/tier4/AWML/pull/47
- Fix dataset
  - https://github.com/tier4/AWML/pull/60
  - https://github.com/tier4/AWML/pull/26
- Add libraries for MLflow
  - https://github.com/tier4/AWML/pull/41

### Tools

- detection2d
  - https://github.com/tier4/AWML/pull/46
- auto_labeling_3d
  - https://github.com/tier4/AWML/pull/38
  - https://github.com/tier4/AWML/pull/50
- analysis_3d
  - https://github.com/tier4/AWML/pull/31
- Fix `deploy_to_autoware`
  - https://github.com/tier4/AWML/pull/36

### Pipelines

### Projects

- CenterPoint
  - https://github.com/tier4/AWML/pull/56 base/1.7
  - https://github.com/tier4/AWML/pull/43 base/1.6
  - https://github.com/tier4/AWML/pull/33 base/1.5
  - https://github.com/tier4/AWML/pull/32 base/1.4
  - https://github.com/tier4/AWML/pull/28 base/1.3
  - https://github.com/tier4/AWML/pull/49 Release short range model
  - https://github.com/tier4/AWML/pull/34 Release short range model
  - https://github.com/tier4/AWML/pull/30 Add finetuning to CenterPoint by freezing certain layers
  - https://github.com/tier4/AWML/pull/37 Add docs
  - https://github.com/tier4/AWML/pull/21 fix config
- StreamPETR
  - https://github.com/tier4/AWML/pull/57 add StreamPETR project
- BEVFusion
  - https://github.com/tier4/AWML/pull/42 base/1.0
  - https://github.com/tier4/AWML/pull/40 base/1.0
  - https://github.com/tier4/AWML/pull/22 fix dependency
  - https://github.com/tier4/AWML/pull/21 fix config
- TransFusion
  - https://github.com/tier4/AWML/pull/21 fix config

### Chore

- Change maintainers
  - https://github.com/tier4/AWML/pull/58
- Delete unused files
  - https://github.com/tier4/AWML/pull/51
- Docs
  - https://github.com/tier4/AWML/pull/61
  - https://github.com/tier4/AWML/pull/53
  - https://github.com/tier4/AWML/pull/52
  - https://github.com/tier4/AWML/pull/14

## v1.0.0

- Release major version v1.0.0

We are glad to announce the release of major version v1.0.0 as OSS "AWML".

- Update CenterPoint model from base/1.0 to base/1.2
  - Dataset: test dataset of db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 + db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 (total frames: 3083):
  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m)

| eval range: 120m     | mAP  | car  | truck | bus  | bicycle | pedestrian |
| -------------------- | ---- | ---- | ----- | ---- | ------- | ---------- |
| CenterPoint base/1.2 | 65.7 | 77.2 | 54.7  | 77.9 | 53.7    | 64.9       |
| CenterPoint base/1.1 | 64.2 | 77.0 | 52.8  | 76.7 | 51.9    | 62.7       |
| CenterPoint base/1.0 | 62.6 | 75.2 | 47.4  | 74.7 | 52.0    | 63.9       |

### Core library

- https://github.com/tier4/AWML/pull/1 Release OSS
- Dependency
  - https://github.com/tier4/AWML/pull/2
  - https://github.com/tier4/AWML/pull/5
  - https://github.com/tier4/AWML/pull/7
  - https://github.com/tier4/AWML/pull/11
- Update config
  - https://github.com/tier4/AWML/pull/8

### Tools

- Add latency measurement tools
  - https://github.com/tier4/AWML/pull/10

### Pipelines

### Projects

- CenterPoint
  - https://github.com/tier4/AWML/pull/8 Release CenterPoint base/1.2
- MobileNetv2
  - https://github.com/tier4/AWML/pull/13 Add document
- YOLOX_opt
  - https://github.com/tier4/AWML/pull/13 Add document

### Chore

- Fix docs
  - https://github.com/tier4/AWML/pull/6
  - https://github.com/tier4/AWML/pull/12
  - https://github.com/tier4/AWML/pull/17
