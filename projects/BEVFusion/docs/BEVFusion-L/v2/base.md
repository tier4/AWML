# Deployed model for BEVFusion-CL base/2.X
## Summary

### Overview
- Details: https://docs.google.com/spreadsheets/d/1hPTYbqsa9ttYHMWOFLyBiDBgoF1m6em0FtT6-oeqxrE/edit?gid=25380807#gid=25380807

| Eval range: 120m                | mAP     | car   | truck | bus   | bicycle | pedestrian |
| --------------------------------| ----    | ----  | ----- | ----  | ------- | ---------- |
| BEVFusion-L base/2.0.0          | 74.10   | 80.06 | 61.16 | 80.90 | 70.22   | 78.11      |


### Datasets
#### base
| eval range: 120m         | mAP  | car <br> (144,001) | truck <br> (20,823) | bus <br> (5,691) | bicycle <br> (5,007) | pedestrian <br> (42,034) |
| -------------------------| ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| BEVFusion-L base/2.0.0          | 74.10   | 80.06 | 61.16 | 80.90 | 70.22   | 78.11      |

#### J6gen2
| eval range: 120m         | mAP  | car <br> (44,008) | truck <br> (2,471) | bus <br> (1,464) | bicycle <br> (333) | pedestrian <br> (6,459) |
| -------------------------| ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| BEVFusion-L base/2.0.0     | 76.17 | 81.50 | 60.46 | 88.68 | 76.94 | 73.23                  |


#### LargeBus
| eval range: 120m         | mAP  | car <br> (13,831)     | truck <br> (2,137) | bus <br> (95) | bicycle <br> (724) | pedestrian <br> (3,916) |
| -------------------------| ---- | -------------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| BEVFusion-L base/2.0.0     | 77.59 | 87.23 | 74.47 | 70.87 | 78.32 | 77.04                  |


#### JPNTaxi
| eval range: 120m         | mAP     | car <br> (16,142) | truck <br> (4,578) | bus <br> (1,457) | bicycle <br> (1,040) | pedestrian <br> (11,971) |
| -------------------------| ----    | ----------------- | ------------------- | ---------------- | --------------- | ------------------------|
| BEVFusion-L base/2.0.0     | 70.81 | 77.13 | 55.74 | 70.95 | 70.04 | 80.17                  |


#### J6
| eval range: 120m         | mAP     | car <br> (67,551) | truck <br> (10,013) | bus <br> (2,503) | bicycle <br> (2,846) | pedestrian <br> (19,117) |
| -------------------------| ------- | ----------------- | ------------------- | ---------------- | ---------------- | -------------------- |
| BEVFusion-L base/2.0.0     | 73.98 | 78.48 | 62.53 | 81.98 | 68.07 | 78.81                  |



## Release

### BEVFusion-L base/2.0.0
- `BEVFusion-L base/2.0.0` is BEVFusion based on lidar without intensity
- It doesn't perform pooling in pedestrians since it gives better performance

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
  - Total mAP (eval range = 120m): 0.7410

| class_name | Count     |mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | -------   |---- | ---- | ---- | ---- | ---- |
| car        |  144,001  | 80.1 | 69.1    | 80.1    | 84.4    | 86.7    |
| truck      |   20,823  | 61.2 | 37.2    | 59.7    | 70.0    | 77.7    |
| bus        |    5,691  | 80.9 | 68.9    | 80.8    | 86.0    | 88.0    |
| bicycle    |    5,007  | 70.2 | 66.8    | 70.9    | 71.3    | 71.9    |
| pedestrian |   42,034  | 78.1 | 75.2    | 77.9    | 79.1    | 80.2    |

- db_largebus_v1 (total frames: 604):
  - Total mAP (eval range = 120m): 0.7760

| class_name | Count     | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | -------   | ---- | ---- | ---- | ---- | ---- |
| car        |  13,831   | 87.2 | 80.0    | 87.3    | 90.2    | 91.6    |
| truck      |   2,137   | 74.5 | 53.2    | 75.1    | 83.4    | 86.2    |
| bus        |      95   | 70.9 | 64.0    | 72.5    | 73.5    | 73.5    |
| bicycle    |     724   | 78.3 | 69.9    | 79.8    | 81.7    | 82.0    |
| pedestrian |   3,916   | 77.0 | 74.4    | 77.0    | 77.9    | 78.8    |

- db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v2 (total frames: 1,157):
  - Total mAP (eval range = 120m): 0.762

| class_name |  Count   | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       |  ------  | ---- | ----    | ---- | ---- | ---- |
| car        |  44,008  | 81.5 | 70.9    | 81.1    | 85.8    | 88.2    |
| truck      |   2,471  | 60.5 | 46.3    | 58.5    | 64.0    | 73.1    |
| bus        |   1,464  | 88.7 | 80.2    | 86.1    | 93.5    | 95.0    |
| bicycle    |     333  | 76.9 | 74.6    | 77.7    | 77.7    | 77.7    |
| pedestrian |   6,459  | 73.2 | 70.3    | 73.2    | 74.2    | 75.2    |

</details>
