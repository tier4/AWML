# Deployed models for BEVFusion-CL-offline
## Summary

- Performance summary
  - Dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v3.0 + DB GSM8 v1.0 + DB J6 v1.0 (total frames: 34,137)
  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m)
  - eval range: 120m

|          | mAP  | car <br> (610,396) | truck <br> (151,672) | bus <br> (37,876) | bicycle <br> (47,739) | pedestrian <br> (367,200) |
| -------- | ---- | ------------------ | -------------------- | ----------------- | --------------------- | ------------------------- |
| base/0.1 | 73.7 | 81.2               | 61.1                 | 84.5              | 70.2                  | 71.6                      |

## Release
### BEVFusion-CL-offline base/0.1

- This is the first BEVFusion camera-lidar offline model with t4base dataset
- We set the hyperparameters exactly the same to `BEVFusion-L-offline base/0.2` except that bounds in `DepthLSSTransform` are adjusted according to the higher resolution in voxelization `(0.075, 0.075, 0.2)` for the offline environment
- Note that *No intensity* was used during the training
- Compared to `BEVFusion-L-offline base/0.2`, the improvement on mAP is marginal (72.2 vs 73.7), specifically, it improves the performance of `truck` (60.5 vs 61.1) and `bus` (72.2 vs 84.5)
- On the other hand, the performance of `pedestrian` (73.9 vs 71.6) and `bicycle` (73.0 vs 70.2) slightly degrade

|                               | mAP  | car <br> (610,396) | truck <br> (151,672) | bus <br> (37,876) | bicycle <br> (47,739) | pedestrian <br> (367,200) |
| ----------------------------- | ---- | ------------------ | -------------------- | ----------------- | --------------------- | ------------------------- |
| BEVFusion-CL-offline base/0.1 | 73.7 | 81.2               | 61.1                 | 84.5              | 70.2                  | 71.6                      |
| BEVFusion-L-offline base/0.2  | 72.2 | 81.2               | 60.5                 | 72.2              | 73.0                  | 73.9                      |

<details>
<summary> The link of data and evaluation result </summary>

- model
  - DB JPNTAXI v3.0 version: https://github.com/tier4/autoware-ml/blob/707908ec2b36fffad2b08e418a5ddfb29f84cb98/autoware_ml/configs/detection3d/dataset/t4dataset/database_v1_3.yaml
  - Training dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v3.0 + DB GSM8 v1.0 + DB J6 v1.0 (total frames: 34,137)
  - Eval dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v3.0 + DB GSM8 v1.0 + DB J6 v1.0 (total frames: 62)
  - [Config file path](https://github.com/tier4/autoware-ml/blob/24b93f3d2f1d649c2fb53b4425afd68102f367ec/projects/BEVFusion/configs/t4dataset/bevfusion_camera_lidar_voxel_second_secfpn_2xb2_t4offline_no_intensity.py)
  - Training results [Google drive (for internal)](https://drive.google.com/drive/folders/1F5ztzuDMfAqolAtvDXPX8dKgfPyEBwj7?usp=drive_link)
  - Training results [model-zoo]
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-cl-offline/t4base/v0.1/bevfusion_camera_lidar_voxel_second_secfpn_2xb2_t4offline_no_intensity.py)
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-cl-offline/t4base/v0.1/logs.zip)
    - [checkpoint.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-cl-offline/t4base/v0.1/epoch_30.pth)


  - train time: NVIDIA A100 80GB * 2 * 30 epochs = 4 days
  - Total mAP to test dataset (eval range = 120m): 0.737
  - Best epoch: epoch_30.pth

| class_name | Count  | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ------ | ---- | ------- | ------- | ------- | ------- |
| car        | 41,133 | 81.2 | 68.5    | 82.0    | 86.6    | 87.9    |
| truck      | 8,890  | 61.1 | 33.4    | 59.3    | 72.6    | 79.0    |
| bus        | 3,275  | 84.5 | 73.5    | 84.6    | 88.8    | 91.3    |
| bicycle    | 3,635  | 70.2 | 68.3    | 70.4    | 70.7    | 71.4    |
| pedestrian | 25,981 | 71.6 | 68.0    | 70.1    | 72.6    | 75.5    |

</details>
