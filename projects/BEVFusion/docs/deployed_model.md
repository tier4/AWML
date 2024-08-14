# Deployed model
## NuScenes model

|                       | mAP  | Car  | Truck | CV   | Bus  | Tra  | Bar  | Mot  | Bic  | Ped  | Cone |
| --------------------- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| BEVFusion-L (spconv)  | 64.6 | 87.7 | 59.7  | 19.3 | 72.8 | 40.9 | 70.0 | 72.0 | 54.7 | 86.5 | 73.9 |
| BEVFusion-CL (spconv) | 66.1 | 88.3 | 53.6  | 21.8 | 73.3 | 38.1 | 72.4 | 76.2 | 61.2 | 87.4 | 78.0 |

- [LiDAR only model (spconv, voxel 0.075)](./configs/nuscenes/bevfusion_lidar_voxel0075_second_secfpn_1xb1-cyclic-20e_nus-3d.py)
  - [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth)
  - [logs](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_20230322_053447.log)
- [Camera-LiDAR model (spconv, voxel 0.075)](./configs/nuscenes/bevfusion_lidar-cam_voxel0075_second_secfpn_1xb1-cyclic-20e_nus-3d.py)
  - [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth)
  - [logs](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_20230524_001539.log)

## T4 dataset model for XX1 (DB1.0 + DB1.1)

- Performance summary
  - Dataset: database_v1_0 + database_v1_1
  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m)

### BEVFusion-L t4xx1_120m/v1

- model
  - Training dataset: database_v1_0 + database_v1_1
  - Eval dataset: database_v1_0 + database_v1_1
  - [Config file path](https://github.com/tier4/autoware-ml/blob/249ebfe5cff685c0911c664ea1ef2b855cc6b52f/projects/BEVFusion/configs/t4dataset/bevfusion_lidar_voxel_second_secfpn_1xb1_t4xx1.py)
  - Results are in internal data.
  - 16 batch A100 * 2 * 4 days
  - Total mAP to test dataset (eval range = 120m): 0.606

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 74.1 | 57.1    | 74.3    | 80.8    | 84.1    |
| truck      | 54.1 | 27.5    | 49.2    | 65.3    | 74.3    |
| bus        | 58.7 | 20.8    | 57.3    | 73.3    | 83.5    |
| bicycle    | 55.7 | 49.4    | 56.3    | 58.1    | 59.0    |
| pedestrian | 60.3 | 52.3    | 58.3    | 62.7    | 68.1    |

### BEVFusion-L t4offline_120m/v1

- model
  - Training dataset: database_v1_0 + database_v1_1
  - Eval dataset: database_v1_0 + database_v1_1
  - [Config file path](https://github.com/tier4/autoware-ml/blob/249ebfe5cff685c0911c664ea1ef2b855cc6b52f/projects/BEVFusion/configs/t4dataset/bevfusion_lidar_voxel_second_secfpn_1xb1_t4offline.py)
  - Results are in internal data.
  - 3 batch A100 * 2 * 6 days
  - Total mAP to test dataset (eval range = 120m): 0.657

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 76.2 | 57.6    | 77.1    | 83.9    | 86.2    |
| truck      | 56.4 | 26.0    | 55.4    | 70.4    | 73.9    |
| bus        | 65.6 | 39.0    | 67.1    | 77.8    | 78.6    |
| bicycle    | 65.0 | 61.1    | 65.4    | 66.4    | 67.0    |
| pedestrian | 65.3 | 59.8    | 64.2    | 67.5    | 69.8    |
