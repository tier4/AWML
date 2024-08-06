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
