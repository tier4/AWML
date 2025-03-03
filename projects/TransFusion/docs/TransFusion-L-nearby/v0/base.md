# Deployed model for TransFusion-L base/0.X
## Summary

This model can be used for detector of near objects with high precision.

- Performance summary with test-dataset of DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB GSM8 v1.0 + DB J6 v1.0
  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m)
  - Eval range: 90m

| TransFusion-L-nearby | mAP | car | truck | bus | bicycle | pedestrian |
| -------------------- | --- | --- | ----- | --- | ------- | ---------- |
|                      |     |     |       |     |         |            |

## Release

TransFusion-L-nearby v0 has many breaking changes.

### base/0.1

- We released first model for TransFusion-L-nearby model.

<details>
<summary> The link of data and evaluation result </summary>

- Main parameter
  - range = 51.2m
  - voxel_size = [0.2, 0.2, 10]
  - grid_size = [512, 512, 1]
- model
  - Training dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0
  - Eval dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0
  - [Config file path](https://github.com/tier4/AWML/blob/3df40a10310dff2d12e4590e26f81017e002a2a0/projects/TransFusion/configs/t4dataset/transfusion_lidar_pillar_second_secfpn_1xb1-cyclic-20e_t4xx1_50m_512grid.py)
  - Deployed ROS parameter file [[AWF model resistry]](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_50m/v1/transfusion_ml_package.param.yaml) [[model-zoo]](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l-nearby/t4base/v0.1/transfusion.param.yaml)
  - Deployed ROS param file for remap [[model-zoo]](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l-nearby/t4base/v0.1/detection_class_remapper.param.yaml)
  - Deployed onnx model [[AWF model resistry]](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_50m/v1/transfusion.onnx) [[model-zoo]](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l-nearby/t4base/v0.1/transfusion.param.yaml)
  - Training results [[AWF model resistry]](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_50m/v1/logs.zip)
  - Training results [model-zoo]
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l-nearby/t4base/v0.1/log.zip)
    - [checkpoint_best.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l-nearby/t4base/v0.1/epoch_43.pth)
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l-nearby/t4base/v0.1/transfusion_lidar_pillar_second_secfpn_1xb1-cyclic-20e_t4xx1_50m_512grid.py)
  - train time: RTX3090 * 1 * 7 days
  - Total mAP to test dataset (eval range = 50m): 0.751

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 87.2 | 70.1    | 89.0    | 94.3    | 95.6    |
| truck      | 70.7 | 39.8    | 70.9    | 82.6    | 89.4    |
| bus        | 84.7 | 64.9    | 87.6    | 93.1    | 93.1    |
| bicycle    | 67.9 | 63.1    | 69.4    | 69.6    | 69.6    |
| pedestrian | 64.9 | 57.0    | 62.6    | 68.1    | 71.9    |

</details>
