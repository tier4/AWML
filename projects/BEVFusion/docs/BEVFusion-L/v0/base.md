# Deployed model for BEVFusion-L base/0.X
## Summary

## Release
### BEVFusion-L base/0.2

- We released BEVFusion-L using T4dataset.
- Note that this model is trained by DB JPNTAXI v1.0 + DB JPNTAXI v2.0.

| model                         | range   | mAP  | car  | truck | bus  | bicycle | pedestrian |
| ----------------------------- | ------- | ---- | ---- | ----- | ---- | ------- | ---------- |
| TransFusion-L t4xx1_120m_v1   | 122.88m | 51.8 | 70.9 | 43.8  | 54.7 | 38.7    | 50.9       |
| BEVFusion-L t4xx1_120m_v1     | 122.4m  | 60.6 | 74.1 | 54.1  | 58.7 | 55.7    | 60.3       |
| BEVFusion-L t4offline_120m_v1 | 122.4m  | 65.7 | 76.2 | 56.4  | 65.6 | 65.0    | 65.3       |

<details>
<summary> The link of data and evaluation result </summary>

- model
  - Training dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0
  - Eval dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0
  - [Config file path](https://github.com/tier4/AWML/blob/249ebfe5cff685c0911c664ea1ef2b855cc6b52f/projects/BEVFusion/configs/t4dataset/bevfusion_lidar_voxel_second_secfpn_1xb1_t4xx1.py)
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

</details>

### BEVFusion-L base/0.1

- We used nuScenes model at first.

<details>
<summary> The link of data and evaluation result </summary>

- [LiDAR only model (spconv, voxel 0.075)](./configs/nuscenes/bevfusion_lidar_voxel0075_second_secfpn_1xb1-cyclic-20e_nus-3d.py)
  - [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth)
  - [logs](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_20230322_053447.log)
- [Camera-LiDAR model (spconv, voxel 0.075)](./configs/nuscenes/bevfusion_lidar-cam_voxel0075_second_secfpn_1xb1-cyclic-20e_nus-3d.py)
  - [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth)
  - [logs](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_20230524_001539.log)

|                       | mAP  | Car  | Truck | CV   | Bus  | Tra  | Bar  | Mot  | Bic  | Ped  | Cone |
| --------------------- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| BEVFusion-L (spconv)  | 64.6 | 87.7 | 59.7  | 19.3 | 72.8 | 40.9 | 70.0 | 72.0 | 54.7 | 86.5 | 73.9 |
| BEVFusion-CL (spconv) | 66.1 | 88.3 | 53.6  | 21.8 | 73.3 | 38.1 | 72.4 | 76.2 | 61.2 | 87.4 | 78.0 |

</details>
