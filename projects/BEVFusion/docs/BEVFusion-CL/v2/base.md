# Deployed model for BEVFusion-CL base/2.X
## Summary

### Overview
- Details: https://docs.google.com/spreadsheets/d/1hPTYbqsa9ttYHMWOFLyBiDBgoF1m6em0FtT6-oeqxrE/edit?gid=25380807#gid=25380807

| Eval range: 120m                | mAP  | car  | truck | bus  | bicycle | pedestrian |
| --------------------------------| ---- | ---- | ----- | ---- | ------- | ---------- |
| BEVFusion-CL base/2.0.0 (A)     | 70.72 | 81.04 | **62.06** | 82.52 | **70.82** | 57.14                   |
| BEVFusion-CL base/2.0.0 (B)     | **75.03** | 79.62 | 61.20 | **86.67** | 69.99 | **77.62**                   |

### Datasets
#### base
| eval range: 120m         | mAP  | car <br> (144,001) | truck <br> (20,823) | bus <br> (5,691) | bicycle <br> (5,007) | pedestrian <br> (42,034) |
| -------------------------| ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| BEVFusion-CL base/2.0.0 (A)      | 70.72 | 81.04 | 62.06 | 82.52 | 70.82 | 57.14                   |
| BEVFusion-CL base/2.0.0 (B)     | 75.03 | 79.62 | 61.20 | 86.67 | 69.99 | 77.62                   |

#### J6gen2
| eval range: 120m         | mAP  | car <br> (44,008) | truck <br> (2,471) | bus <br> (1,464) | bicycle <br> (333) | pedestrian <br> (6,459) |
| -------------------------| ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| BEVFusion-CL base/2.0.0 (A)      | 73.68 | 82.92 | 59.51 | 88.93 | 78.80 | 58.19                   |
| BEVFusion-CL base/2.0.0 (B)     | 75.98 | 83.63 | 62.88 | 88.84 | 80.138| 64.37                   |

#### LargeBus
| eval range: 120m         | mAP  | car <br> (13,831)     | truck <br> (2,137) | bus <br> (95) | bicycle <br> (724) | pedestrian <br> (3,916) |
| -------------------------| ---- | -------------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| BEVFusion-CL base/2.0.0 (A)      | 75.82 | 87.83 | 73.83 | 80.84 | 79.79 | 56.76                  |
| BEVFusion-CL base/2.0.0 (B)     | 80.75 | 87.74 | 74.34 | 87.41 | 78.01 | 76.21                   |

#### JPNTaxi
| eval range: 120m         | mAP     | car <br> (16,142) | truck <br> (4,578) | bus <br> (1,457) | bicycle <br> (1,040) | pedestrian <br> (11,971) |
| -------------------------| ----    | ----------------- | ------------------- | ---------------- | --------------- | ------------------------|
| BEVFusion-CL base/2.0.0 (A)      | 68.29 | 78.23 | 56.18 | 71.96 | 74.84 | 60.23                 |
| BEVFusion-CL base/2.0.0 (B)     | 74.01 | 77.3 | 56.78 | 81.32 | 74.50 | 80.14                  |

#### J6
| eval range: 120m         | mAP     | car <br> (67,551) | truck <br> (10,013) | bus <br> (2,503) | bicycle <br> (2,846) | pedestrian <br> (19,117) |
| -------------------------| ------- | ----------------- | ------------------- | ---------------- | ---------------- | -------------------- |
| BEVFusion-CL base/2.0.0 (A)      | 69.99 | 79.41 | 64.64 | 83.58 | 67.03 | 55.28                |
| BEVFusion-CL base/2.0.0 (B)     | 74.48 | 77.28 | 62.67 | 87.92 | 66.58 | 77.98                  |


## Release

### BEVFusion-CL base/2.0.0
- BEVFusion-CL base/2.0.0 (A): Without intensity and training pedestrians with pooling pedestrians
- BEVFusion-CL base/2.0.0 (B): Same as `BEVFusion-CL base/2.0.0 (A)` without pooling pedestrians
- We report only `BEVFusion-CL base/2.0.0 (B)` since the performance is much better than `BEVFusion-CL base/2.0.0 (A)`, and it is mainly due to it doesn't downsample the dense heatmaps for pedestrians, and thus it has more queries

<details>
<summary> The link of data and evaluation result </summary>

- Model
  - Training dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v4.0 + DB GSM8 v1.0 + DB J6 v1.0 + DB J6 v2.0 + DB J6 v3.0 + DB J6 v5.0 + DB J6 Gen2 v1.0 + DB J6 Gen2 v2.0 + DB J6 Gen2 v4.0 + DB LargeBus v1.0 (total frames: 71,633)
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
  - Train time: NVIDIA H100 80GB * 4 * 50 epochs = 3 days and 20 hours
  - Batch size: 4*16 = 64

- Evaluation
  - db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 + db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 + db_j6gen2_v1 + db_j6gen2_v1 + db_j6gen2_v4 + db_largebus_v1 (total frames: 5,703):
  - Total mAP (eval range = 120m): 0.7503

| class_name |  Count    | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       |  -------  | ---- | ---- | ---- | ---- | ---- |
| car        |   144,001 | 79.6 | 68.3    | 79.5    | 84.1    | 86.6    |
| truck      |   20,823  | 61.2 | 37.2    | 59.5    | 70.3    | 77.9    |
| bus        |    5,691  | 86.7 | 74.6    | 86.3    | 92.1    | 93.7    |
| bicycle    |    5,007  | 70.0 | 67.4    | 70.4    | 70.8    | 71.4    |
| pedestrian |   42,034  | 77.6 | 74.7    | 77.4    | 78.6    | 79.7    |

- db_largebus_v1 (total frames: 604):
  - Total mAP (eval range = 120m): 0.8075

| class_name |  Count    |  mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       |  -------  | ---- | ---- | ---- | ---- | ---- |
| car        |   13,831  | 87.7 | 80.3    | 87.7    | 90.9    | 92.0    |
| truck      |   2,137   | 74.3 | 53.0    | 74.3    | 83.8    | 86.3    |
| bus        |      95   | 87.4 | 76.9    | 89.1    | 91.8    | 91.8    |
| bicycle    |     724   | 78.0 | 71.5    | 79.3    | 80.4    | 80.9    |
| pedestrian |   3,916   | 76.2 | 73.6    | 76.2    | 77.1    | 77.9    |

- db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v2 (total frames: 1,157):
  - Total mAP (eval range = 120m): 0.765

| class_name |  Count   | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       |  ------  | ---- | ---- | ---- | ---- | ---- |
| car        |  44,008  | 82.0 | 71.4    | 81.5    | 86.4    | 88.7    |
| truck      |   2,471  | 59.0 | 45.6    | 57.5    | 62.2    | 70.6    |
| bus        |   1,464  | 90.0 | 80.0    | 88.0    | 95.5    | 96.6    |
| bicycle    |     333  | 78.3 | 74.6    | 79.5    | 79.5    | 79.7    |
| pedestrian |   6,459  | 73.2 | 70.4    | 73.1    | 74.0    | 75.1    |
</details>
