# Deployed model for TransFusion-L-offline base/0.X
## Summary

- Performance summary
  - Dataset: database_v1_0 + database_v1_1 + database_v2_0 + database_v3_0
  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m)
  - Eval range: 120m

| TransFusion-L-offline | mAP  | car  | truck | bus  | bicycle | pedestrian |
| --------------------- | ---- | ---- | ----- | ---- | ------- | ---------- |
| base/0.2              | 57.3 | 80.4 | 64.9  | 84.1 | 46.4    | 58.9       |

## Release

TransFusion-L-offline v0 has many breaking changes.

### base/0.2

- Main parameter
  - range = 122.88m
  - voxel_size = [0.32, 0.32, 10]
  - grid_size = [768, 768, 1]
- model
  - Training dataset: database_v1_0 + database_v1_1 + database_v2_0 + database_v3_0
  - [PR](https://github.com/tier4/autoware-ml/pull/146)
  - [Config file path](https://github.com/tier4/autoware-ml/blob/304ba041e85adc2a47d16229ac1768aea156a2a9/projects/TransFusion/configs/t4dataset/transfusion_lidar_pillar_second_secfpn_1xb1_120m-768grid-t4base.py)
  - [Deployed onnx model and ROS parameter files](https://evaluation.tier4.jp/evaluation/mlpackages/1800c7f8-a80e-4162-8574-4ee84432e89d/releases/6d18aacb-c1d8-468e-a9fc-5ef616b3b55a?project_id=zWhWRzei)
  - [Training results](https://drive.google.com/drive/folders/1v4QN696n4iMIlk0j5O_Pxho7zgoBsJa5)
  - train time: NVIDIA RTX 6000 Ada Generation * 2 * 5 days
- Evaluation result with test-dataset of database_v1_0 + database_v1_1 + database_v2_0 + database_v3_0
  - eval range = 120m
  - Total mAP to test dataset : 0.573

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 70.8 | 54.2    | 71.0    | 77.6    | 80.4    |
| truck      | 44.0 | 14.6    | 40.2    | 56.2    | 64.9    |
| bus        | 73.7 | 54.0    | 74.2    | 82.5    | 84.1    |
| bicycle    | 45.2 | 43.3    | 45.2    | 45.9    | 46.4    |
| pedestrian | 53.0 | 46.7    | 51.0    | 55.3    | 58.9    |

- Evaluation result with eval-dataset of database_v1_0 + database_v1_1

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 72.9 | 49.9    | 73.6    | 82.3    | 86.0    |
| truck      | 48.2 | 14.4    | 43.2    | 63.1    | 72.0    |
| bus        | 59.7 | 33.3    | 62.7    | 70.6    | 72.2    |
| bicycle    | 44.2 | 39.3    | 44.6    | 46.0    | 46.8    |
| pedestrian | 56.0 | 47.9    | 54.3    | 59.1    | 62.8    |

- Evaluation result with eval-dataset of database_v2_0 + database_v3_0

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 70.2 | 56.7    | 70.1    | 75.7    | 78.2    |
| truck      | 42.7 | 15.3    | 39.3    | 53.6    | 62.5    |
| bus        | 78.4 | 60.9    | 78.5    | 86.4    | 87.8    |
| bicycle    | 47.3 | 46.4    | 47.3    | 47.6    | 48.1    |
| pedestrian | 51.2 | 45.8    | 49.1    | 53.1    | 56.7    |

### base/0.1

- Main parameter
  - range = 122.88m
  - voxel_size = [0.32, 0.32, 10]
  - grid_size = [768, 768, 1]
- model
  - Training dataset: database_v1_0 + database_v1_1
  - Eval dataset: database_v1_0 + database_v1_1
  - [Config file path](https://github.com/tier4/autoware-ml/blob/3df40a10310dff2d12e4590e26f81017e002a2a0/projects/TransFusion/configs/t4dataset/transfusion_lidar_pillar_second_secfpn_1xb1-cyclic-20e_t4xx1_120m_768grid.py)
  - [Deployed ROS parameter file](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_120m/v1/transfusion_ml_package.param.yaml)
  - [Deployed onnx model](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_120m/v1/transfusion.onnx)
  - [Training results](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_120m/v1/logs.zip)
  - train time: A100 * 2 * 3.5days
  - Total mAP to test dataset (eval range = 120m): 0.518

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 70.9 | 45.9    | 71.7    | 81.3    | 84.6    |
| truck      | 43.8 | 11.5    | 37.6    | 58.7    | 67.3    |
| bus        | 54.7 | 32.2    | 53.8    | 65.6    | 67.1    |
| bicycle    | 38.7 | 33.5    | 37.6    | 40.2    | 43.4    |
| pedestrian | 50.9 | 44.4    | 48.7    | 52.7    | 57.7    |
