# Deployed model for BEVFusion-CL J6Gen2/2.X
## Summary

### Overview
- Details: https://docs.google.com/spreadsheets/d/1hPTYbqsa9ttYHMWOFLyBiDBgoF1m6em0FtT6-oeqxrE/edit?gid=25380807#gid=25380807


| Eval range: 120m                | mAP  | car  | truck | bus  | bicycle | pedestrian |
| --------------------------------| ---- | ---- | ----- | ---- | ------- | ---------- |
| BEVFusion-CL J6Gen2/2.0.1 (A)     | 77.64 | **84.87** | 68.96 | 88.65 | **79.66** | 66.05                   |
| BEVFusion-CL J6Gen2/2.0.1 (B)     | **80.08** | 84.56 | **69.56** | **89.76** | 79.48 | **77.03**                   |

### Datasets

#### J6gen2
| eval range: 120m         | mAP  | car <br> (44,008) | truck <br> (2,471) | bus <br> (1,464) | bicycle <br> (333) | pedestrian <br> (6,459) |
| -------------------------| ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| BEVFusion-CL J6Gen2/2.0.1 (A)      | 75.98 | 83.63 | 62.88 | 88.84 | 80.13 | 64.37                  |
| BEVFusion-CL J6Gen2/2.0.1 (B)      | 78.48 | 83.46 | 62.34 | 89.80 | 81.01 | 75.78                   |

#### LargeBus
| eval range: 120m         | mAP  | car <br> (13,831)     | truck <br> (2,137) | bus <br> (95) | bicycle <br> (724) | pedestrian <br> (3,916) |
| -------------------------| ---- | -------------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| BEVFusion-CL J6Gen2/2.0.1 (A)      | 79.61 | 89.04 | 76.38 | 83.63 | 79.86 | 69.11                |
| BEVFusion-CL J6Gen2/2.0.1 (B)      | 82.86 | 88.54 | 78.22 | 88.95 | 79.28 | 79.26                   |



## Release

### BEVFusion-CL J6Gen2/2.0.1
- BEVFusion-CL J6Gen2/2.0.1 (A): Based on `BEVFusion-CL base/2.0.0 (A)` with intensity on J6Gen2 data
- BEVFusion-CL J6Gen2/2.0.1 (B): Based on `BEVFusion-Cl base/2.0.0 (B)` with intensity on J6Gen2 data
- We report only `BEVFusion-CL J6Gen2/2.0.1 (B)` since the performance is much better
than `BEVFusion-CL J6Gen2/2.0.1 (A)`

<details>
<summary> The link of data and evaluation result </summary>

- Model
  - Training dataset: DB J6 Gen2 v1.0 + DB J6 Gen2 v2.0 + DB J6 Gen2 V4.0 + DB LargeBus v1.0 (total frames: 20,777)
  - [Config file path](https://github.com/tier4/AWML/blob/3cacf810b70fef2aafab1ffa25eb5e3581922f8a/autoware_ml/configs/detection3d/dataset/t4dataset/base.py)
  - Deployed onnx model and ROS parameter files [[WebAuto (for internal)]](WIP)
  - Deployed onnx and ROS parameter files [[model-zoo]]
    - [detection_class_remapper.param.yaml](WIP)
    - [centerpoint_ml_package.param.yaml](WIP)
    - [deploy_metadata.yaml](WIP)
    - [pts_voxel_encoder_centerpoint.onnx](WIP)
    - [pts_backbone_neck_head_centerpoint.onnx](WIP)
  - Training results [[Google drive (for internal)]](WIP)
  - Training results [model-zoo]
    - [logs.zip](WIP)
    - [checkpoint_best.pth](WIP)
    - [config.py](WIP)
  - Train time: NVIDIA H100 80GB * 4 * 30 epochs = 20 hours
  - Batch size: 4*16 = 64

- Evaluation
  - db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v4 + + db_largebus_v1 (total frames: 1,761):
  - Total mAP (eval range = 120m): 0.8001

| class_name  | Count    | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----        | -------  | ---- | ---- | ---- | ---- | ---- |
| car         |  57,839  | 84.6 | 74.4    | 84.4    | 88.8    | 90.7    |
| truck       |   4,608  | 69.6 | 51.8    | 69.6    | 76.0    | 80.9    |
| bus         |   1,559  | 89.8 | 80.7    | 86.7    | 95.3    | 96.4    |
| bicycle     |   1,057  | 79.5 | 73.4    | 80.6    | 81.8    | 82.1    |
| pedestrian  |  10,375  | 77.0 | 74.6    | 76.9    | 77.9    | 78.7    |

- db_largebus_v1 (total frames: 604):
  - Total mAP (eval range = 120m): 0.829

| class_name | Count    | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | -------  | ---- | ----    | ---- | ---- | ---- |
| car        |  13,831  | 88.5 | 81.3    | 88.4    | 91.8    | 92.7    |
| truck      |   2,137  | 78.2 | 57.2    | 79.7    | 87.0    | 89.0    |
| bus        |     95   | 89.0 | 76.6    | 91.9    | 93.7    | 93.7    |
| bicycle    |    724   | 79.3 | 71.9    | 80.5    | 82.1    | 82.6    |
| pedestrian |  3,916   | 79.3 | 76.7    | 79.2    | 80.2    | 81.0    |

- db_j6gen2_v1 + db_j6gen2_v2 +db_j6gen2_v4 (total frames: 1,157):
  - Total mAP (eval range = 120m): 0.7850

| class_name | Count   | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | ------  | ---- | ---- | ---- | ---- | ---- |
| car        | 44,008  | 83.5 | 72.8    | 83.3    | 87.9    | 89.9    |
| truck      |  2,471  | 62.3 | 47.6    | 61.3    | 66.6    | 73.9    |
| bus        |  1,464  | 89.8 | 80.8    | 86.6    | 95.4    | 96.5    |
| bicycle    |    333  | 81.0 | 77.5    | 82.2    | 82.2    | 82.2    |
| pedestrian |  6,459  | 75.8 | 73.4    | 75.7    | 76.6    | 77.5    |

</details>
