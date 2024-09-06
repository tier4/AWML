# Deployed model for X2
## T4dataset-X2 90m model

- Performance summary
  - Dataset: database_v2_0 + database_v3_0
  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m)

| eval range: 90m            | range  | mAP  | car  | truck | bus  | bicycle | pedestrian |
| -------------------------- | ------ | ---- | ---- | ----- | ---- | ------- | ---------- |
| TransFusion-L t4x2_90m/v1  | 92.16m | 58.5 | 80.5 | 28.1  | 82.4 | 48.0    | 53.7       |

### TransFusion-L t4x2_90m/v1 (pillar 0.24m * grid 768 = 92.16m)

- model
  - Training dataset: database_v2_0 + database_v3_0
  - Eval dataset: database_v2_0 + database_v3_0
  - [PR](https://github.com/tier4/autoware-ml/pull/126)
  - [Config file path](https://github.com/tier4/autoware-ml/blob/e8701f9953be3034776b0de71ecbd03146c03c5f/projects/TransFusion/configs/t4dataset/transfusion_lidar_pillar_second_secfpn_1xb1_90m-768grid-t4x2.py)
  - [Deployed onnx and ROS parameter files](https://evaluation.tier4.jp/evaluation/mlpackages/1800c7f8-a80e-4162-8574-4ee84432e89d/releases/acdd07c5-4a8f-4983-88e7-a8823f7dc672?project_id=zWhWRzei)
  - [Training results](https://drive.google.com/drive/folders/1d_xr8PZq3gB-BkqJ02GMDM5F8mwdEKXx?usp=drive_link)
  - train time: NVIDIA RTX 6000 Ada Generation * 2 * 2 days
  - Total mAP to test dataset (eval range = 90m): 0.585

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 80.5 | 69.2    | 80.7    | 85.4    | 86.6    |
| truck      | 28.1 | 10.3    | 23.0    | 31.9    | 47.2    |
| bus        | 82.4 | 70.6    | 81.7    | 87.9    | 89.4    |
| bicycle    | 48.0 | 46.4    | 47.4    | 48.5    | 49.6    |
| pedestrian | 53.7 | 49.2    | 51.9    | 55.3    | 58.4    |

## T4dataset-X2 120m model

TBD
