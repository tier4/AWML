# Deployed model for BEVFusion-L J6Gen2/2.X
## Summary

### Overview
- Details: https://docs.google.com/spreadsheets/d/1hPTYbqsa9ttYHMWOFLyBiDBgoF1m6em0FtT6-oeqxrE/edit?gid=25380807#gid=25380807


| Eval range: 120m             | mAP  | car  | truck | bus  | bicycle | pedestrian |
| ---------------------------- | ---- | ---- | ----- | ---- | ------- | ---------- |
| BEVFusion-L J6Gen2/2.0.1     | 79.73 | 83.16 | 69.32 | 90.02 | 78.60 | 77.51                   |

### Datasets

#### J6gen2
| eval range: 120m         | mAP  | car <br> (44,008) | truck <br> (2,471) | bus <br> (1,464) | bicycle <br> (333) | pedestrian <br> (6,459) |
| -------------------------| ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| BEVFusion-L J6Gen2/2.0.1      | 78.04 | 81.89 | 62.35 | 90.34 | 79.99 | 75.62                   |

#### LargeBus
| eval range: 120m         | mAP  | car <br> (13,831)     | truck <br> (2,137) | bus <br> (95) | bicycle <br> (724) | pedestrian <br> (3,916) |
| -------------------------| ---- | -------------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| BEVFusion-L J6Gen2/2.0.1      | 82.08 | 87.84 | 77.58 | 85.52 | 78.57 | 80.88                   |



## Release

### BEVFusion-CL J6Gen2/2.0.1
- `BEVFusion-L J6Gen2/2.0.1` is a BEVFusion model based on `BEVFusion-L base/2.0.0` by finetuning on `J6Gen2` with intensity

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
  - Train time: NVIDIA H100 80GB * 4 * 30 epochs = 12 hours
  - Batch size: 4*16 = 64

- Evaluation
  - db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v4 + + db_largebus_v1 (total frames: 1,761):
  - Total mAP (eval range = 120m): 0.7970

| class_name | Count    | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | -------  | ---- | ---- | ---- | ---- | ---- |
| car        |  57,839  | 83.2 | 73.5    | 82.8    | 87.0    | 89.4    |
| truck      |   4,608  | 69.3 | 51.9    | 68.6    | 75.3    | 81.4    |
| bus        |   1,559  | 90.0 | 80.5    | 88.4    | 95.0    | 96.1    |
| bicycle    |   1,057  | 78.6 | 73.0    | 79.9    | 80.7    | 80.9    |
| pedestrian |  10,375  | 77.5 | 74.9    | 77.4    | 78.4    | 79.3    |

- db_largebus_v1 (total frames: 604):
  - Total mAP (eval range = 120m): 0.821

| class_name | Count    | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | -------  | ---- | ---- | ---- | ---- | ---- |
| car        |  13,831  | 87.8 | 80.5    | 87.8    | 90.9    | 92.2    |
| truck      |   2,137  | 77.6 | 56.7    | 79.1    | 85.9    | 88.7    |
| bus        |     95   | 85.5 | 74.1    | 86.4    | 90.8    | 90.8    |
| bicycle    |    724   | 78.6 | 72.0    | 79.9    | 81.0    | 81.3    |
| pedestrian |  3,916   | 80.9 | 78.2    | 80.7    | 81.8    | 82.8    |


- db_j6gen2_v1 + db_j6gen2_v2 +db_j6gen2_v4 (total frames: 1,157):
  - Total mAP (eval range = 120m): 0.7800

| class_name | Count   | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | ------  | ---- | ---- | ---- | ---- | ---- |
| car        | 44,008  | 81.9 | 71.9    | 81.4    | 85.8    | 88.5    |
| truck      |  2,471  | 62.4 | 48.3    | 59.8    | 66.2    | 75.1    |
| bus        |  1,464  | 90.3 | 81.3    | 88.5    | 95.3    | 96.4    |
| bicycle    |    333  | 80.0 | 76.4    | 81.2    | 81.2    | 81.2    |
| pedestrian |  6,459  | 75.6 | 73.1    | 75.6    | 76.5    | 77.4  

</details>
