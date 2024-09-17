# Deployed model for base model
## T4dataset-base 50m model

TBD

## T4dataset-base 90m model

- Performance summary
  - Dataset: database_v1_0 + database_v1_1 + database_v2_0 + database_v3_0
  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m)

| eval range: 90m             | range  | mAP  | car  | truck | bus  | bicycle | pedestrian |
| --------------------------- | ------ | ---- | ---- | ----- | ---- | ------- | ---------- |
| TransFusion-L t4base_90m/v1 | 92.16m | 65.3 | 81.7 | 46.7  | 81.7 | 54.8    | 61.7       |


### TransFusion-L t4base_90m/v1 (pillar 0.24m * grid 768 = 92.16m)

- model
  - Training dataset: database_v1_0 + database_v1_1 + database_v2_0 + database_v3_0
  - Eval dataset: database_v1_0 + database_v1_1 + database_v2_0 + database_v3_0
  - [PR](https://github.com/tier4/autoware-ml/pull/125)
  - [Config file path](https://github.com/tier4/autoware-ml/blob/5f472170f07251184dc009a1ec02be3b4f3bf98c/autoware_ml/configs/detection3d/dataset/t4dataset/base.py)
  - [Deployed onnx model and ROS parameter files](https://evaluation.tier4.jp/evaluation/mlpackages/1800c7f8-a80e-4162-8574-4ee84432e89d/releases/008b128a-873a-4e4a-afa6-d2583f7fc224?project_id=zWhWRzei&tab=reports)
  - [Training results](https://drive.google.com/drive/folders/1uyUE-ReYARmykG1GsFG2vYsysWwJv3sW)
  - train time: NVIDIA RTX 6000 Ada Generation * 2 * 5 days
  - Total mAP to test dataset (eval range = 90m): 0.653

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 81.6 | 67.7    | 82.3    | 87.4    | 88.8    |
| truck      | 50.1 | 19.4    | 48.6    | 62.8    | 69.6    |
| bus        | 82.0 | 66.1    | 84.0    | 88.8    | 89.2    |
| bicycle    | 54.9 | 53.4    | 54.8    | 55.4    | 55.9    |
| pedestrian | 63.0 | 56.9    | 61.3    | 65.4    | 68.4    |

Evaluation result with eval-dataset database_v1_0 + database_v1_1:

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 80.7 | 60.4    | 82.1    | 89.2    | 91.2    |
| truck      | 56.0 | 23.8    | 54.4    | 69.7    | 75.9    |
| bus        | 77.6 | 55.2    | 80.4    | 87.3    | 87.7    |
| bicycle    | 57.4 | 54.4    | 57.5    | 58.4    | 59.1    |
| pedestrian | 65.5 | 57.9    | 64.1    | 68.7    | 71.1    |

Evaluation result with eval-dataset database_v2_0 + database_v3_0:

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 82.3 | 71.5    | 82.8    | 87.0    | 87.9    |
| truck      | 47.5 | 17.3    | 46.0    | 59.7    | 66.8    |
| bus        | 83.4 | 68.1    | 85.6    | 89.8    | 90.1    |
| bicycle    | 55.1 | 54.1    | 55.0    | 55.3    | 55.8    |
| pedestrian | 61.6 | 56.1    | 59.7    | 63.8    | 66.9    |

## T4dataset-base 120m model

- Performance summary
  - Dataset: database_v1_0 + database_v1_1 + database_v2_0 + database_v3_0
  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m)

| eval range: 120m             | range   | mAP  | car  | truck | bus  | bicycle | pedestrian |
| ---------------------------- | ------- | ---- | ---- | ----- | ---- | ------- | ---------- |
| TransFusion-L t4base_120m/v1 | 122.88m | 57.3 | 80.4 | 64.9  | 84.1 | 46.4    | 58.9       |

### TransFusion-L t4base_120m/v1 (pillar 0.36m * grid 768 = 122.88m)

- model
  - Training dataset: database_v1_0 + database_v1_1 + database_v2_0 + database_v3_0
  - Eval dataset: database_v1_0 + database_v1_1 + database_v2_0 + database_v3_0
  - [PR](https://github.com/tier4/autoware-ml/pull/146)
  - [Config file path](https://github.com/tier4/autoware-ml/blob/304ba041e85adc2a47d16229ac1768aea156a2a9/projects/TransFusion/configs/t4dataset/transfusion_lidar_pillar_second_secfpn_1xb1_120m-768grid-t4base.py)
  - [Deployed onnx model and ROS parameter files](https://evaluation.tier4.jp/evaluation/mlpackages/1800c7f8-a80e-4162-8574-4ee84432e89d/releases/6d18aacb-c1d8-468e-a9fc-5ef616b3b55a?project_id=zWhWRzei)
  - [Training results](https://drive.google.com/drive/folders/1v4QN696n4iMIlk0j5O_Pxho7zgoBsJa5)
  - train time: NVIDIA RTX 6000 Ada Generation * 2 * 5 days
  - Total mAP to test dataset (eval range = 120m): 0.573

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 70.8 | 54.2    | 71.0    | 77.6    | 80.4    |
| truck      | 44.0 | 14.6    | 40.2    | 56.2    | 64.9    |
| bus        | 73.7 | 54.0    | 74.2    | 82.5    | 84.1    |
| bicycle    | 45.2 | 43.3    | 45.2    | 45.9    | 46.4    |
| pedestrian | 53.0 | 46.7    | 51.0    | 55.3    | 58.9    |

Evaluation result with eval-dataset database_v1_0 + database_v1_1:

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 72.9 | 49.9    | 73.6    | 82.3    | 86.0    |
| truck      | 48.2 | 14.4    | 43.2    | 63.1    | 72.0    |
| bus        | 59.7 | 33.3    | 62.7    | 70.6    | 72.2    |
| bicycle    | 44.2 | 39.3    | 44.6    | 46.0    | 46.8    |
| pedestrian | 56.0 | 47.9    | 54.3    | 59.1    | 62.8    |

Evaluation result with eval-dataset database_v2_0 + database_v3_0:

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 70.2 | 56.7    | 70.1    | 75.7    | 78.2    |
| truck      | 42.7 | 15.3    | 39.3    | 53.6    | 62.5    |
| bus        | 78.4 | 60.9    | 78.5    | 86.4    | 87.8    |
| bicycle    | 47.3 | 46.4    | 47.3    | 47.6    | 48.1    |
| pedestrian | 51.2 | 45.8    | 49.1    | 53.1    | 56.7    |
