# Deployed model for BEVFusion-L base/1.X
## Summary

|                       | mAP  | car  | truck | bus  | bicycle | pedestrian |
| --------------------- | ---- | ---- | ----- | ---- | ------- | ---------- |
| BEVFusion-L base/1.0 | 63.7 |  72.3 | 58.8  | 70.5 | 70.2    | 56.3       |

## Release

### BEVFusion-L base/1.0

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
  - Training dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v3.0 + DB GSM8 v1.0 + DB J6 v1.0 (total frames: 34,137)
  - [Config file path](https://github.com/tier4/AWML/blob/249ebfe5cff685c0911c664ea1ef2b855cc6b52f/projects/BEVFusion/configs/t4dataset/bevfusion_lidar_voxel_second_secfpn_1xb1_t4xx1.py)
  - Training results [model-zoo]
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-l/t4base/v1.0/logs.zip)
    - [checkpoint_best.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-l/t4base/v1.0/best_NuScenes_metric_T4Metric_mAP_epoch_24.pth)
    - [checkpoint_latest.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-l/t4base/v1.0/epoch_50.pth)
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-l/t4base/v1.0/bevfusion_lidar_voxel_second_secfpn_l4_2xb4_base_1.0.py)
  - train time: NVIDIA A100 80GB * 2 * 40 epochs = 4.5 days
  - Evaluation result with test-dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v4.0 + DB GSM8 v1.0 + DB J6 v1.0 (total frames: 2787):

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---- | ---- | ---- | ---- | ---- | ---- |
| car        | 72.3 | 56.5    | 72.6    | 78.8    | 81.3    |
| truck      | 58.8 | 27.7    | 56.4    | 71.5    | 79.5    |
| bus        | 70.5 | 42.9    | 71.0    | 83.0    | 85.4    |
| bicycle    | 70.2 | 66.4    | 71.0    | 71.4    | 71.9    |
| pedestrian | 56.3 | 46.8    | 53.5    | 59.8    | 65.0    |

  - Evaluation result with test-dataset (bus only): DB GSM8 v1.0 + DB J6 v1.0 (total frames: 1280):

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---- | ---- | ---- | ---- | ---- | ---- |
| car        | 70.7 | 57.4    | 70.4    | 76.4    | 78.5    |
| truck      | 55.2 | 23.2    | 52.9    | 67.2    | 77.5    |
| bus        | 69.9 | 41.9    | 70.8    | 82.7    | 84.3    |
| bicycle    | 70.3 | 68.3    | 70.4    | 70.8    | 71.5    |
| pedestrian | 52.9 | 43.8    | 49.0    | 56.1    | 62.8    |

  - Evaluation result with test-dataset (taxi only): DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v4.0  (total frames: 1507):

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---- | ---- | ---- | ---- | ---- | ---- |
| car        | 75.6 | 55.4    | 77.0    | 83.6    | 86.5    |
| truck      | 63.2 | 33.4    | 60.8    | 76.5    | 82.0    |
| bus        | 71.5 | 44.2    | 72.7    | 82.9    | 86.4    |
| bicycle    | 70.0 | 59.7    | 73.0    | 73.5    | 73.7    |
| pedestrian | 60.6 | 50.8    | 59.4    | 64.4    | 67.9    |

</details>
