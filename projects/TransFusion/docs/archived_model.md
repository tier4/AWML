# Archived results and models

## NuScenes model

- Performance summary

|                        | mAP  | Car  | Truck | CV   | Bus  | Tra  | Bar  | Mot  | Bic  | Ped  | Cone |
| ---------------------- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| TransFusion-L (pillar) | 55.0 | 85.6 | 54.8  | 14.6 | 71.1 | 33.0 | 58.7 | 49.5 | 32.7 | 79.5 | 56.8 |

- LiDAR only model (pillar)
  - model: TBD
  - logs: TBD

## T4dataset for X2

## T4dataset for XX1
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

### TransFusion-L t4xx1_75m_v1 (pillar 0.3m * grid 512 = 76.8m)

- Note that this model is low performance because of autoware-ml bug.
- model
  - Training dataset: database_v1_0
  - Eval dataset: database_v1_1 (Bug fix and [new dataset config is applied](https://github.com/tier4/autoware-ml/pull/31))
  - [Config file](https://github.com/tier4/autoware-ml/blob/17e8944ac2154f1f1042a507a4001ccf057ffe78/projects/TransFusion/configs/t4dataset/transfusion_lidar_pillar02_second_secfpn_1xb4-cyclic-20e_t4xx1.py)
  - [Deployed onnx model](https://awf.ml.dev.web.auto/perception/models/transfusion/v1/transfusion.onnx)
- val
  - Total mAP: 0.485

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 69.8 | 55.1    | 70.7    | 75.5    | 77.7    |
| truck      | 39.5 | 19.8    | 39.4    | 46.8    | 52.0    |
| bus        | 47.6 | 30.3    | 46.3    | 55.1    | 58.8    |
| bicycle    | 33.4 | 30.6    | 33.3    | 34.0    | 35.6    |
| pedestrian | 52.0 | 46.6    | 50.6    | 53.3    | 57.6    |
