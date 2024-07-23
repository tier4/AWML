# Deployed model
## NuScenes model

|                        | mAP  | Car  | Truck | CV   | Bus  | Tra  | Bar  | Mot  | Bic  | Ped  | Cone |
| ---------------------- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| TransFusion-L (pillar) | 55.0 | 85.6 | 54.8  | 14.6 | 71.1 | 33.0 | 58.7 | 49.5 | 32.7 | 79.5 | 56.8 |

- LiDAR only model (pillar)
  - model: TBD
  - logs: TBD

## T4 dataset model for XX1

- Performance summary
  - Dataset: database_v1_0 + database_v1_1 (eval range: 75m)
  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m)
  - Note:
    - e:50m eval range 50m

| eval range: 90m            | range  | mAP  | car  | truck | bus  | bicycle | pedestrian |
| -------------------------- | ------ | ---- | ---- | ----- | ---- | ------- | ---------- |
| TransFusion-L t4xx1_90m/v1 | 92.16m | 57.8 | 74.0 | 48.0  | 72.0 | 42.7    | 52.1       |
| TransFusion-L t4xx1_90m/v2 | 92.16m | 68.1 | 80.5 | 58.0  | 80.8 | 58.0    | 63.2       |

### TransFusion-L t4xx1_90m/v1 (pillar 0.32m * grid 576 = 92.16m)

- model
  - Training dataset: database_v1_0 + database_v1_1
  - Eval dataset: database_v1_0 + database_v1_1
  - [Config file path](https://github.com/tier4/autoware-ml/blob/fe28c0a7de0579c68406e40c5abfe9afcaed41f6/projects/TransFusion/configs/t4dataset/transfusion_lidar_pillar_second_secfpn_1xb6-cyclic-20e_t4xx1_90m_576grid.py) (Note: eval range is 75m in training time)
  - [Deployed onnx model](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_90m/v1/transfusion.onnx)
  - [Deployed ROS parameter file](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_90m/v1/transfusion.param.yaml)
  - [Deployed ROS param file for remap](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_90m/v1/detection_class_remapper.param.yaml)
  - [Training results](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_90m/v1/logs.zip)
  - train time: A100 * 2 * 5 days
  - Total mAP to test dataset (eval range = 90m): 0.578

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 74.0 | 51.2    | 74.8    | 83.4    | 86.7    |
| truck      | 48.0 | 15.0    | 42.6    | 63.1    | 71.3    |
| bus        | 72.0 | 50.5    | 74.2    | 80.9    | 82.3    |
| bicycle    | 42.7 | 37.6    | 42.0    | 44.7    | 46.6    |
| pedestrian | 52.1 | 44.3    | 49.8    | 54.4    | 59.8    |

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

## T4 dataset model for X2

TBD
