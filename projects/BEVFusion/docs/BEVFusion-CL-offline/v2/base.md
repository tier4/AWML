# Deployed model for BEVFusion-CL base/2.X
## Summary

### Overview

| Eval range: 120m                | mAP  | car  | truck | bus  | bicycle | pedestrian |
| --------------------------------| ---- | ---- | ----- | ---- | ------- | ---------- |
| BEVFusion-CL-offline base/2.0.0 | 77.8  | 87.30 | 61.60 | 85.90 | 73.20 | 80.90     |
| BEVFusion-CL         base/2.0.0 | 76.3  | 80.50 | 61.90 | 85.90 | 74.70 | 78.70     |


## Release

### BEVFusion-CL-offline base/2.0.0

<details>
<summary> The link of data and evaluation result </summary>

- Model
  - Training dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v4.0 + DB GSM8 v1.0 + DB J6 v1.0 + DB J6 v2.0 + DB J6 v3.0 + DB J6 v5.0 + DB J6 Gen2 v1.0 + DB J6 Gen2 v2.0 + DB J6 Gen2 v4.0 + DB LargeBus v1.0 (total frames: 71,633)
  - [Config file path](https://github.com/tier4/AWML/blob/50f35a8ae52c4892351be0c7aa5d260c1b310b7e/projects/BEVFusion/configs/t4dataset/BEVFusion-CL-offline/bevfusion_camera_lidar_offline_voxel_second_secfpn_4xb8_base.py)
  - Training results [model-zoo]
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-cl-offline/t4base/v2.0.0/logs.zip)
    - [checkpoint_best.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-cl-offline/t4base/v2.0.0/best_NuScenes_metric_T4Metric_mAP_epoch_30.pth)
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-cl-offline/t4base/v2.0.0/bevfusion_camera_lidar_voxel_second_secfpn_2xb2_t4offline_no_intensity.py)
  - Train time: NVIDIA H100 80GB * 4 * 50 epochs = 3 days and 20 hours
  - Batch size: 4*5 = 20

- Evaluation
  - db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 + db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 + db_j6gen2_v1 + db_j6gen2_v1 + db_j6gen2_v4 + db_largebus_v1 (total frames: 5,703):
  - Total mAP (eval range = 120m): 0.7503

| class_name |  Count    | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       |  -------  | ---- | ---- | ---- | ---- | ---- |
| car        |   144,001 | 87.3 | 77.5    | 87.8    | 91.6    | 92.2    |
| truck      |   20,823  | 61.6 | 41.0    | 61.3    | 69.0    | 74.9    |
| bus        |    5,691  | 85.9 | 75.6    | 85.6    | 90.3    | 92.2    |
| bicycle    |    5,007  | 73.2 | 71.4    | 73.5    | 73.7    | 74.1    |
| pedestrian |   42,034  | 80.9 | 79.5    | 80.5    | 81.3    | 82.3    |

</details>
