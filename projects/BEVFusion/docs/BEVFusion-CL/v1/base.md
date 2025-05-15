# Deployed model for BEVFusion-CL base/1.X
## Summary

|                       | mAP  | car  | truck | bus  | bicycle | pedestrian |
| --------------------- | ---- | ---- | ----- | ---- | ------- | ---------- |
| BEVFusion-CL base/1.0 | 66.9 |  80.6 | 65.65 | 59.7 | 68.8    | 60.2       |

## Release
### BEVFusion-CL base/1.0

- We release LiDAR only model trained with base1.0 dataset

|                              | mAP  | car  | truck | bus  | bicycle | pedestrian |
| ---------------------------  | ---- | ---- | ----- | ---- | ------- | ---------- |
| BEVFusion-CL base/1.0 (ALL)  | 66.9 |  80.6 | 65.65 | 59.7 | 68.8    | 60.2       |
| BEVFusion-L base/1.0 (ALL)   | 63.7 |  72.3 | 58.8  | 70.5 | 70.2    | 56.3       |
| BEVFusion-CL base/1.0  (TAXI)| 68.5 |  76.0 | 64.9  | 68.6 | 73.2    | 60.1       |
| BEVFusion-L base/1.0  (TAXI) | 68.2 |  75.6 | 63.2  | 71.5 | 70.0    | 60.6       |
| BEVFusion-CL base/1.0  (BUS) | 63.3 |  70.8 | 55.7  | 64.8 | 69.9    | 55.9       |
| BEVFusion-L base/1.0  (BUS)  | 63.8 |  70.7 | 55.2  | 69.9 | 70.3    | 52.9       |

<details>
<summary> The link of data and evaluation result </summary>

- model
  - Training dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v4.0 + DB GSM8 v1.0 + DB J6 v1.0 (total frames: 34,137)
  - [Config file path](https://github.com/tier4/AWML/blob/05302ecc9e832f3c988019f5d30fdfc105455027/projects/BEVFusion/configs/t4dataset/bevfusion_camera_lidar_voxel_second_secfpn_1xb1_t4xx1.py)
  - Training results [model-zoo]
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-cl/t4base/v1.0/logs.zip)
    - [checkpoint_best.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-cl/t4base/v1.0/best_NuScenes_metric_T4Metric_mAP_epoch_38.pth)
    - [checkpoint_latest.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-cl/t4base/v1.0/epoch_40.pth)
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-cl/t4base/v1.0/bevfusion_camera_lidar_voxel_second_secfpn_l4_2xb4_base_1.0.py)
  - train time: NVIDIA A100 80GB * 2 * 40 epochs = 4.5 days
  - Evaluation result with test-dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v4.0 + DB GSM8 v1.0 + DB J6 v1.0 (total frames: 2787):

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---- | ---- | ---- | ---- | ---- | ---- |
| car        | 80.6 | 69.4    | 80.9    | 85.0    | 87.1    |
| truck      | 65.5 | 42.6    | 64.6    | 74.8    | 79.8    |
| bus        | 59.7 | 30.8    | 58.1    | 72.3    | 77.4    |
| bicycle    | 68.8 | 65.8    | 69.5    | 69.8    | 70.2    |
| pedestrian | 60.2 | 51.2    | 58.3    | 63.6    | 67.8    |

  - Evaluation result with test-dataset (bus only): DB GSM8 v1.0 + DB J6 v1.0 (total frames: 1280):

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---- | ---- | ---- | ---- | ---- | ---- |
| car        | 70.8 | 57.9    | 70.5    | 76.3    | 78.6    |
| truck      | 55.7 | 27.5    | 53.2    | 65.6    | 76.3    |
| bus        | 64.8 | 38.4    | 65.8    | 76.0    | 78.9    |
| bicycle    | 69.4 | 66.9    | 69.8    | 70.2    | 70.8    |
| pedestrian | 55.9 | 45.0    | 51.5    | 59.8    | 67.3    |

  - Evaluation result with test-dataset (taxi only): DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v4.0  (total frames: 1507):

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---- | ---- | ---- | ---- | ---- | ---- |
| car        | 76.0 | 55.9    | 77.3    | 83.9    | 86.7    |
| truck      | 64.9 | 35.2    | 62.1    | 78.8    | 83.4    |
| bus        | 68.6 | 41.7    | 70.8    | 78.7    | 83.2    |
| bicycle    | 73.2 | 68.4    | 74.4    | 74.9    | 75.1    |
| pedestrian | 60.1 | 48.6    | 58.5    | 64.7    | 68.6    |

</details>
