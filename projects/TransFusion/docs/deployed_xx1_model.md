# Deployed model for XX1
## T4dataset-XX1 90m model

- Performance summary
  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m)
  - Note
    - Eval DB: Evaluation dataset

| eval range: 90m            | Eval DB | range  | mAP  | car  | truck | bus  | bicycle | pedestrian |
| -------------------------- | ------- | ------ | ---- | ---- | ----- | ---- | ------- | ---------- |
| TransFusion-L t4xx1_90m/v3 | 1.0+1.1 | 92.16m | 68.5 | 81.7 | 62.4  | 83.6 | 50.9    | 64.1       |
| TransFusion-L t4xx1_90m/v2 | 1.0+1.1 | 92.16m | 68.1 | 80.5 | 58.0  | 80.8 | 58.0    | 63.2       |

### TransFusion-L t4xx1_90m/v3 (pillar 0.24m * grid 768 = 92.16m)

- model
  - Training dataset: database_v1_0 + database_v1_1 + database_v1_3
  - Eval dataset: database_v1_0 + database_v1_1 + database_v1_3
  - [PR](https://github.com/tier4/autoware-ml/pull/100)
  - [Config file path](https://github.com/tier4/autoware-ml/blob/37cf92a2b4b3d7f80b09c8bd5eaff6229ca18f95/projects/TransFusion/configs/t4dataset/transfusion_lidar_pillar_second_secfpn_1xb1_90m-768grid-t4xx1.py)
  - [Deployed onnx model](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_90m/v3/transfusion.onnx)
  - [Deployed ROS parameter file](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_90m/v3/transfusion.param.yaml)
  - [Deployed ROS param file for remap](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_90m/v3/detection_class_remapper.param.yaml)
  - [Training results](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_90m/v3/logs.zip)
  - train time: (A100 * 4) * 2 days
- Total mAP: 0.685
  - Test dataset: database_v1_0 + database_v1_1
  - Eval range = 90m

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 81.7 | 61.0    | 83.2    | 90.2    | 92.2    |
| truck      | 62.4 | 30.0    | 60.2    | 76.5    | 82.8    |
| bus        | 83.5 | 56.5    | 90.1    | 93.7    | 93.7    |
| bicycle    | 50.9 | 45.0    | 50.8    | 52.8    | 54.9    |
| pedestrian | 64.1 | 56.9    | 62.1    | 66.9    | 70.4    |

- Total mAP: 0.696
  - Test dataset: database_v1_3
  - Eval range = 90m

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 85.9 | 77.8    | 86.4    | 89.5    | 90.1    |
| truck      | 49.9 | 38.0    | 48.3    | 52.7    | 60.7    |
| bus        | 64.2 | 43.7    | 63.2    | 71.7    | 78.2    |
| bicycle    | 85.2 | 80.6    | 86.7    | 86.7    | 86.7    |
| pedestrian | 62.8 | 58.2    | 61.1    | 64.2    | 67.8    |

### TransFusion-L t4xx1_90m/v2 (pillar 0.24m * grid 768 = 92.16m)

- model
  - Training dataset: database_v1_0 + database_v1_1
  - Eval dataset: database_v1_0 + database_v1_1
  - [Config file path](https://github.com/tier4/autoware-ml/blob/fe28c0a7de0579c68406e40c5abfe9afcaed41f6/projects/TransFusion/configs/t4dataset/transfusion_lidar_pillar_second_secfpn_1xb4-cyclic-20e_t4xx1_90m_768grid.py) (Note: eval range is 75m in training time)
  - [Deployed onnx model](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_90m/v2/transfusion.onnx)
  - [Deployed ROS parameter file](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_90m/v2/transfusion.param.yaml)
  - [Deployed ROS param file for remap](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_90m/v2/detection_class_remapper.param.yaml)
  - [Training results](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_90m/v2/logs.zip)
  - train time: RTX 3090 * 1 * 8 days
  - Total mAP to test dataset (eval range = 90m): 0.681

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 80.5 | 59.7    | 81.8    | 89.1    | 91.4    |
| truck      | 58.0 | 25.2    | 56.4    | 72.2    | 78.1    |
| bus        | 80.8 | 60.3    | 83.6    | 89.5    | 89.7    |
| bicycle    | 58.0 | 53.1    | 58.6    | 59.7    | 60.5    |
| pedestrian | 63.2 | 55.6    | 61.4    | 66.3    | 69.6    |

## T4dataset-XX1 50m model

- Performance summary
  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m)
  - Note
    - Eval DB: Evaluation dataset

| eval range: 50m            | Eval DB | range | mAP  | car  | truck | bus  | bicycle | pedestrian |
| -------------------------- | ------- | ----- | ---- | ---- | ----- | ---- | ------- | ---------- |
| TransFusion-L t4xx1_50m/v1 | 1.0+1.1 | 51.2m | 75.1 | 87.2 | 70.7  | 84.7 | 67.9    | 64.9       |

### TransFusion-L t4xx1_50m/v1 (pillar 0.2m * grid 512 = 51.2m)

This model can be used for detector of near objects with high precision.
This model is considered to use with far range model.

- model
  - Training dataset: database_v1_0 + database_v1_1
  - Eval dataset: database_v1_0 + database_v1_1
  - [Config file path](https://github.com/tier4/autoware-ml/blob/3df40a10310dff2d12e4590e26f81017e002a2a0/projects/TransFusion/configs/t4dataset/transfusion_lidar_pillar_second_secfpn_1xb1-cyclic-20e_t4xx1_50m_512grid.py)
  - [Deployed ROS parameter file](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_50m/v1/transfusion_ml_package.param.yaml)
  - [Deployed onnx model](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_50m/v1/transfusion.onnx)
  - [Training results](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_50m/v1/logs.zip)
  - train time: RTX3090 * 1 * 7 days
  - Total mAP to test dataset (eval range = 50m): 0.751

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 87.2 | 70.1    | 89.0    | 94.3    | 95.6    |
| truck      | 70.7 | 39.8    | 70.9    | 82.6    | 89.4    |
| bus        | 84.7 | 64.9    | 87.6    | 93.1    | 93.1    |
| bicycle    | 67.9 | 63.1    | 69.4    | 69.6    | 69.6    |
| pedestrian | 64.9 | 57.0    | 62.6    | 68.1    | 71.9    |

## T4dataset-XX1 120m model

- Performance summary
  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m)
  - Note
    - Eval DB: Evaluation dataset

| eval range: 120m            | Eval DB | range   | mAP  | car  | truck | bus  | bicycle | pedestrian |
| --------------------------- | ------- | ------- | ---- | ---- | ----- | ---- | ------- | ---------- |
| TransFusion-L t4xx1_120m/v1 | 1.0+1.1 | 122.88m | 51.8 | 70.9 | 43.8  | 54.7 | 38.7    | 50.9       |

### TransFusion-L t4xx1_120m/v1 (pillar 0.32m * grid 768 = 122.88m)

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
