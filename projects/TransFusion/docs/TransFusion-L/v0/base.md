# Deployed model for TransFusion-L base/0.X
## Summary

- Performance summary with test-dataset of DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB GSM8 v1.0 + DB J6 v1.0
  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m)
  - Eval range: 90m

| TransFusion-L | mAP  | car  | truck | bus  | bicycle | pedestrian |
| ------------- | ---- | ---- | ----- | ---- | ------- | ---------- |
| base/0.6      | 65.3 | 81.7 | 46.7  | 81.7 | 54.8    | 61.7       |

## Release

TransFusion-L v0 has many breaking changes.

### TransFusion-L base/0.6

- We trained by dataset of all product.
- Note that this model is trained by DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB GSM8 v1.0 + DB J6 v1.0.
- The evaluation in X2 product increased 12% (from 58.5 to 66.0).

| Method                             | mAP-japantaxi | mAP-X2 |
| ---------------------------------- | ------------- | ------ |
| Train model-japantaxi and model-X2 | 68.1          | 58.5   |
| Train base model with All dataset  | 67.4          | 66.0   |

<details>
<summary> The link of data and evaluation result </summary>

- Main parameter
  - range = 92.16m
  - voxel_size = [0.24, 0.24, 10]
  - grid_size = [768, 768, 1]
- model
  - Training dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB GSM8 v1.0 + DB J6 v1.0
  - [PR](https://github.com/tier4/autoware-ml/pull/125)
  - [Config file path](https://github.com/tier4/autoware-ml/blob/5f472170f07251184dc009a1ec02be3b4f3bf98c/autoware_ml/configs/detection3d/dataset/t4dataset/base.py)
  - Deployed onnx and ROS parameter files [[webauto]](https://evaluation.tier4.jp/evaluation/mlpackages/1800c7f8-a80e-4162-8574-4ee84432e89d/releases/008b128a-873a-4e4a-afa6-d2583f7fc224?project_id=zWhWRzei&tab=reports)
  - Deployed onnx and ROS parameter files [[model-zoo]]
    - [detection_class_remapper.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l/t4base/v0.6/detection_class_remapper.param.yaml)
    - [transfusion_base_ml_package.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l/t4base/v0.6/transfusion_base_ml_package.param.yaml)
    - [transfusion_base.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l/t4base/v0.6/transfusion_base.onnx)
  - Training results [[webauto]](https://drive.google.com/drive/folders/1uyUE-ReYARmykG1GsFG2vYsysWwJv3sW)
  - Training results [model-zoo]
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l/t4base/v0.6/logs.zip)
    - [checkpoint_latest.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l/t4base/v0.6/epoch_50.pth)
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l/t4base/v0.6/config.py)
  - train time: NVIDIA RTX 6000 Ada Generation * 2 * 5 days
- Evaluation result with test-dataset of DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB GSM8 v1.0 + DB J6 v1.0
  - Total mAP to test dataset (eval range = 90m): 0.653

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 81.6 | 67.7    | 82.3    | 87.4    | 88.8    |
| truck      | 50.1 | 19.4    | 48.6    | 62.8    | 69.6    |
| bus        | 82.0 | 66.1    | 84.0    | 88.8    | 89.2    |
| bicycle    | 54.9 | 53.4    | 54.8    | 55.4    | 55.9    |
| pedestrian | 63.0 | 56.9    | 61.3    | 65.4    | 68.4    |

- Evaluation result with eval-dataset of DB JPNTAXI v1.0 + DB JPNTAXI v2.0

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 80.7 | 60.4    | 82.1    | 89.2    | 91.2    |
| truck      | 56.0 | 23.8    | 54.4    | 69.7    | 75.9    |
| bus        | 77.6 | 55.2    | 80.4    | 87.3    | 87.7    |
| bicycle    | 57.4 | 54.4    | 57.5    | 58.4    | 59.1    |
| pedestrian | 65.5 | 57.9    | 64.1    | 68.7    | 71.1    |

- Evaluation result with eval-dataset of DB GSM8 v1.0 + DB J6 v1.0

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 82.3 | 71.5    | 82.8    | 87.0    | 87.9    |
| truck      | 47.5 | 17.3    | 46.0    | 59.7    | 66.8    |
| bus        | 83.4 | 68.1    | 85.6    | 89.8    | 90.1    |
| bicycle    | 55.1 | 54.1    | 55.0    | 55.3    | 55.8    |
| pedestrian | 61.6 | 56.1    | 59.7    | 63.8    | 66.9    |

</details>

### TransFusion-L base/0.5

- This model is released for X2 product at first.
- Note that this model is trained by DB GSM8 v1.0 + DB J6 v1.0.

<details>
<summary> The link of data and evaluation result </summary>

- Parameter
  - pillar 0.24m * grid 768 = 92.16m
- model
  - Training dataset: DB GSM8 v1.0 + DB J6 v1.0
  - Eval dataset: DB GSM8 v1.0 + DB J6 v1.0
  - [PR](https://github.com/tier4/autoware-ml/pull/126)
  - [Config file path](https://github.com/tier4/autoware-ml/blob/e8701f9953be3034776b0de71ecbd03146c03c5f/projects/TransFusion/configs/t4dataset/transfusion_lidar_pillar_second_secfpn_1xb1_90m-768grid-t4x2.py)
  - Deployed onnx and ROS parameter files [[webauto]](https://evaluation.tier4.jp/evaluation/mlpackages/1800c7f8-a80e-4162-8574-4ee84432e89d/releases/acdd07c5-4a8f-4983-88e7-a8823f7dc672?project_id=zWhWRzei)
  - Deployed onnx and ROS parameter files [[model-zoo]]
    - [detection_class_remapper.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l/t4base/v0.5/detection_class_remapper.param.yaml)
    - [transfusion_x2_ml_package.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l/t4base/v0.5/transfusion_x2_ml_package.param.yaml)
    - [transfusion_x2.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l/t4base/v0.5/transfusion_x2.onnx)
  - Training results [[webauto]](https://drive.google.com/drive/folders/1d_xr8PZq3gB-BkqJ02GMDM5F8mwdEKXx?usp=drive_link)
  - Training results [model-zoo]
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l/t4base/v0.5/logs.zip)
    - [checkpoint_latest.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l/t4base/v0.5/epoch_50.pth)
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l/t4base/v0.5/config.py)
  - train time: NVIDIA RTX 6000 Ada Generation * 2 * 2 days
  - Total mAP to test dataset (eval range = 90m): 0.585

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 80.5 | 69.2    | 80.7    | 85.4    | 86.6    |
| truck      | 28.1 | 10.3    | 23.0    | 31.9    | 47.2    |
| bus        | 82.4 | 70.6    | 81.7    | 87.9    | 89.4    |
| bicycle    | 48.0 | 46.4    | 47.4    | 48.5    | 49.6    |
| pedestrian | 53.7 | 49.2    | 51.9    | 55.3    | 58.4    |

</details>

### TransFusion-L base/0.4

- We added DB JPNTAXI v3.0 for training.
- mAP of "DB JPNTAXI v1.0 + DB JPNTAXI v2.0 test dataset, eval range 90m" is as same as the model of base/0.3.

|          | mAP  | car  | truck | bus  | bicycle | pedestrian |
| -------- | ---- | ---- | ----- | ---- | ------- | ---------- |
| base/0.4 | 68.5 | 81.7 | 62.4  | 83.5 | 50.9    | 64.1       |
| base/0.3 | 68.1 | 80.5 | 58.0  | 80.8 | 58.0    | 63.2       |

<details>
<summary> The link of data and evaluation result </summary>

- Parameter
  - pillar 0.24m * grid 768 = 92.16m
- model
  - Training dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v3.0
  - Eval dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v3.0
  - [PR](https://github.com/tier4/autoware-ml/pull/100)
  - [Config file path](https://github.com/tier4/autoware-ml/blob/37cf92a2b4b3d7f80b09c8bd5eaff6229ca18f95/projects/TransFusion/configs/t4dataset/transfusion_lidar_pillar_second_secfpn_1xb1_90m-768grid-t4xx1.py)
  - Deployed onnx model [[webauto]](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_90m/v3/transfusion.onnx) [[model-zoo]](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l/t4base/v0.4/transfusion.onnx)
  - Deployed ROS parameter file [[webauto]](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_90m/v3/transfusion.param.yaml) [[model-zoo]](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l/t4base/v0.4/transfusion.param.yaml)
  - Deployed ROS param file for remap [[webauto]](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_90m/v3/detection_class_remapper.param.yaml) [[model-zoo]](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l/t4base/v0.4/detection_class_remapper.param.yaml)
  - Training results [[webauto]](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_90m/v3/logs.zip)
  - Training results [model-zoo]
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l/t4base/v0.4/logs.zip)
    - [checkpoint_best.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l/t4base/v0.4/epoch_44.pth)
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l/t4base/v0.4/transfusion_lidar_pillar_second_secfpn_1xb1_90m-768grid-t4xx1.py)
  - train time: (A100 * 4) * 2 days
- Total mAP: 0.685
  - Test dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0
  - Eval range = 90m

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 81.7 | 61.0    | 83.2    | 90.2    | 92.2    |
| truck      | 62.4 | 30.0    | 60.2    | 76.5    | 82.8    |
| bus        | 83.5 | 56.5    | 90.1    | 93.7    | 93.7    |
| bicycle    | 50.9 | 45.0    | 50.8    | 52.8    | 54.9    |
| pedestrian | 64.1 | 56.9    | 62.1    | 66.9    | 70.4    |

- Total mAP: 0.696
  - Test dataset: DB JPNTAXI v3.0
  - Eval range = 90m

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 85.9 | 77.8    | 86.4    | 89.5    | 90.1    |
| truck      | 49.9 | 38.0    | 48.3    | 52.7    | 60.7    |
| bus        | 64.2 | 43.7    | 63.2    | 71.7    | 78.2    |
| bicycle    | 85.2 | 80.6    | 86.7    | 86.7    | 86.7    |
| pedestrian | 62.8 | 58.2    | 61.1    | 64.2    | 67.8    |

</details>

### TransFusion-L base/0.3

- We used the high resolution to improve the detection performance.
- Note that this model is trained by DB JPNTAXI v1.0 + DB JPNTAXI v2.0.
- mAP of "DB JPNTAXI v1.0 + DB JPNTAXI v2.0 test dataset, eval range 90m" increased 18% (from 0.578 to 0.681).

|          | mAP  | car  | truck | bus  | bicycle | pedestrian |
| -------- | ---- | ---- | ----- | ---- | ------- | ---------- |
| base/0.3 | 68.1 | 80.5 | 58.0  | 80.8 | 58.0    | 63.2       |
| base/0.2 | 57.8 | 74.0 | 48.0  | 72.0 | 42.7    | 52.1       |

<details>
<summary> The link of data and evaluation result </summary>

- Parameter
  - pillar 0.24m * grid 768 = 92.16m
- model
  - Training dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0
  - Eval dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0
  - [Config file path](https://github.com/tier4/autoware-ml/blob/fe28c0a7de0579c68406e40c5abfe9afcaed41f6/projects/TransFusion/configs/t4dataset/transfusion_lidar_pillar_second_secfpn_1xb4-cyclic-20e_t4xx1_90m_768grid.py) (Note: eval range is 75m in training time)
  - Deployed onnx model [[webauto]](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_90m/v2/transfusion.onnx) [[model-zoo]](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l/t4base/v0.3/transfusion.onnx)
  - Deployed ROS parameter file [[webauto]](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_90m/v2/transfusion.param.yaml) [[model-zoo]](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l/t4base/v0.3/transfusion.param.yaml)
  - Deployed ROS param file for remap [[webauto]](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_90m/v2/detection_class_remapper.param.yaml) [[model-zoo]](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l/t4base/v0.3/detection_class_remapper.param.yaml)
  - Training results [[webauto]](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_90m/v2/logs.zip)
  - Training results [model-zoo]
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l/t4base/v0.3/logs.zip)
    - [checkpoint_best.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l/t4base/v0.3/epoch_50.pth)
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l/t4base/v0.3/transfusion_lidar_pillar_second_secfpn_1xb8-cyclic-20e_t4xx1_90m_768grid.py)
  - train time: RTX 3090 * 1 * 8 days
  - Total mAP to test dataset (eval range = 90m): 0.681

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 80.5 | 59.7    | 81.8    | 89.1    | 91.4    |
| truck      | 58.0 | 25.2    | 56.4    | 72.2    | 78.1    |
| bus        | 80.8 | 60.3    | 83.6    | 89.5    | 89.7    |
| bicycle    | 58.0 | 53.1    | 58.6    | 59.7    | 60.5    |
| pedestrian | 63.2 | 55.6    | 61.4    | 66.3    | 69.6    |

</details>

### TransFusion-L base/0.2

- We fixed the bug of `autoware-ml` and improve the performance.
- Note that this model is trained by DB JPNTAXI v1.0 + DB JPNTAXI v2.0.

<details>
<summary> The link of data and evaluation result </summary>

- Parameter
  - pillar 0.32m * grid 576 = 92.16m
- model
  - Training dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0
  - Eval dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0
  - [Config file path](https://github.com/tier4/autoware-ml/blob/fe28c0a7de0579c68406e40c5abfe9afcaed41f6/projects/TransFusion/configs/t4dataset/transfusion_lidar_pillar_second_secfpn_1xb6-cyclic-20e_t4xx1_90m_576grid.py) (Note: eval range is 75m in training time)
  - Deployed onnx model [[webauto]](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_90m/v1/transfusion.onnx) [[model-zoo]](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l/t4base/v0.2/transfusion.onnx)
  - Deployed ROS parameter file [[webauto]](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_90m/v1/transfusion.param.yaml) [[model-zoo]](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l/t4base/v0.2/transfusion.param.yaml)
  - Deployed ROS param file for remap [[webauto]](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_90m/v1/detection_class_remapper.param.yaml) [[model-zoo]](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l/t4base/v0.2/detection_class_remapper.param.yaml)
  - Training results [[webauto]](https://awf.ml.dev.web.auto/perception/models/transfusion/t4xx1_90m/v1/logs.zip)
  - Training results [model-zoo]
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l/t4base/v0.2/logs.zip)
    - [checkpoint_best.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l/t4base/v0.2/epoch_66.pth)
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l/t4base/v0.2/transfusion_lidar_pillar_second_secfpn_1xb1-cyclic-20e_t4xx1_90m_576grid.py)
  - train time: A100 * 2 * 5 days
  - Total mAP to test dataset (eval range = 90m): 0.578

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 74.0 | 51.2    | 74.8    | 83.4    | 86.7    |
| truck      | 48.0 | 15.0    | 42.6    | 63.1    | 71.3    |
| bus        | 72.0 | 50.5    | 74.2    | 80.9    | 82.3    |
| bicycle    | 42.7 | 37.6    | 42.0    | 44.7    | 46.6    |
| pedestrian | 52.1 | 44.3    | 49.8    | 54.4    | 59.8    |

</details>

### TransFusion-L base/0.1

- We released first model of TransFusion-L
- Note that this model is low performance because of `autoware-ml` bug.

<details>
<summary> The link of data and evaluation result </summary>

- Parameter
  - pillar 0.3m * grid 512 = 76.8m
- model
  - Training dataset: DB JPNTAXI v1.0
  - Eval dataset: DB JPNTAXI v2.0 (Bug fix and [new dataset config is applied](https://github.com/tier4/autoware-ml/pull/31))
  - [Config file](https://github.com/tier4/autoware-ml/blob/17e8944ac2154f1f1042a507a4001ccf057ffe78/projects/TransFusion/configs/t4dataset/transfusion_lidar_pillar02_second_secfpn_1xb4-cyclic-20e_t4xx1.py)
  - Deployed onnx model [[webauto]](https://awf.ml.dev.web.auto/perception/models/transfusion/v1/transfusion.onnx) [[model-zoo]](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/transfusion/transfusion-l/t4base/v0.1/transfusion.onnx)
- val
  - Total mAP: 0.485

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 69.8 | 55.1    | 70.7    | 75.5    | 77.7    |
| truck      | 39.5 | 19.8    | 39.4    | 46.8    | 52.0    |
| bus        | 47.6 | 30.3    | 46.3    | 55.1    | 58.8    |
| bicycle    | 33.4 | 30.6    | 33.3    | 34.0    | 35.6    |
| pedestrian | 52.0 | 46.6    | 50.6    | 53.3    | 57.6    |

</details>
