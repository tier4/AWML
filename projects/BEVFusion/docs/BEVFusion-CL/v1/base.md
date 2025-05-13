# Deployed model for BEVFusion-CL base/1.X
## Summary

|                       | mAP  | car  | truck | bus  | bicycle | pedestrian |
| --------------------- | ---- | ---- | ----- | ---- | ------- | ---------- |
| BEVFusion-CL base/1.0 | 66.9 |  80.6 | 65.65 | 59.7 | 68.8    | 60.2       |

## Release
### BEVFusion-CL base/1.0

- We release LiDAR only model trained with base1.0 dataset

|                       | mAP  | car  | truck | bus  | bicycle | pedestrian |
| --------------------- | ---- | ---- | ----- | ---- | ------- | ---------- |
| BEVFusion-CL base/1.0 | 66.9 |  80.6 | 65.65 | 59.7 | 68.8    | 60.2       |
| BEVFusion-L base/1.0  | 63.7 |  72.3 | 58.8  | 70.5 | 70.2    | 56.3       |

<details>
<summary> The link of data and evaluation result </summary>

- model
  - Training dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v3.0 + DB GSM8 v1.0 + DB J6 v1.0 (total frames: 34,137)
  - [Config file path](https://github.com/tier4/AWML/blob/05302ecc9e832f3c988019f5d30fdfc105455027/projects/BEVFusion/configs/t4dataset/bevfusion_camera_lidar_voxel_second_secfpn_1xb1_t4xx1.py)
  - Training results [model-zoo]
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-cl/t4base/v1.0/logs.zip)
    - [checkpoint_best.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-cl/t4base/v1.0/best_NuScenes_metric_T4Metric_mAP_epoch_38.pth)
    - [checkpoint_latest.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-cl/t4base/v1.0/epoch_40.pth)
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-cl/t4base/v1.0/bevfusion_camera_lidar_voxel_second_secfpn_l4_2xb4_base_1.0.py)
  - train time: NVIDIA A100 80GB * 2 * 40 epochs = 4.5 days
  - Evaluation result with test-dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v3.0 + DB GSM8 v1.0 + DB J6 v1.0 (total frames: 1,394):

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---- | ---- | ---- | ---- | ---- | ---- |
| car        | 80.6 | 69.4    | 80.9    | 85.0    | 87.1    |
| truck      | 65.5 | 42.6    | 64.6    | 74.8    | 79.8    |
| bus        | 59.7 | 30.8    | 58.1    | 72.3    | 77.4    |
| bicycle    | 68.8 | 65.8    | 69.5    | 69.8    | 70.2    |
| pedestrian | 60.2 | 51.2    | 58.3    | 63.6    | 67.8    |

</details>
