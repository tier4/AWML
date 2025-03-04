# Deployed model for BEVFusion-CL base/0.X
## Summary


## Release
### BEVFusion-CL base/0.1

- We released first Camera-LiDAR fusion model
- The mAP of "DB JPNTAXI v1.0 + DB JPNTAXI v2.0 test dataset, eval 120m" increase 5%.

|                       | mAP  | car  | truck | bus  | bicycle | pedestrian |
| --------------------- | ---- | ---- | ----- | ---- | ------- | ---------- |
| BEVFusion-CL base/0.1 | 63.7 | 72.4 | 56.8  | 71.8 | 62.1    | 55.3       |
| BEVFusion-L base/0.2  | 60.6 | 74.1 | 54.1  | 58.7 | 55.7    | 60.3       |

<details>
<summary> The link of data and evaluation result </summary>

- model
  - Training dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0
  - Eval dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0
  - [Config file path](https://github.com/tier4/AWML/blob/05302ecc9e832f3c988019f5d30fdfc105455027/projects/BEVFusion/configs/t4dataset/bevfusion_camera_lidar_voxel_second_secfpn_1xb1_t4xx1.py)
  - Results are in internal data.
  - 5 batch A100 * 2 * 7 days
  - Total mAP to test dataset (eval range = 120m): 0.637

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 72.4 | 49.3    | 73.6    | 81.6    | 85.0    |
| truck      | 56.8 | 21.4    | 55.1    | 72.3    | 78.4    |
| bus        | 71.8 | 39.4    | 76.1    | 85.1    | 86.5    |
| bicycle    | 62.1 | 53.8    | 62.9    | 65.3    | 66.3    |
| pedestrian | 55.3 | 45.6    | 54.8    | 58.9    | 62.0    |

</details>
