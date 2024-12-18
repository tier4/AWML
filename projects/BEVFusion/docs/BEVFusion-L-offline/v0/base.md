# Deployed models for BEVFusion-L-offline
## Summary

- Performance summary
  - Dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v3.0 + DB GSM8 v1.0 + DB J6 v1.0 (total frames: 34,137)
  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m)
  - eval range: 120m

|          | mAP  | car <br> (610,396) | truck <br> (151,672) | bus <br> (37,876) | bicycle <br> (47,739) | pedestrian <br> (367,200) |
| -------- | ---- | ------------------ | -------------------- | ----------------- | --------------------- | ------------------------- |
| base/0.2 | 72.2 | 81.2               | 60.5                 | 72.2              | 73.0                  | 73.9                      |

## Release
### BEVFusion-L-offline base/0.2

- We use all dataset to make BEVFusion-L-offline model.

<details>
<summary> The link of data and evaluation result </summary>

- model
  - Training dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v3.0 + DB GSM8 v1.0 + DB J6 v1.0 (total frames: 34,137)
  - Eval dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v3.0 + DB GSM8 v1.0 + DB J6 v1.0 (total frames: 62)
  - *No intensity*
  - [PR](https://github.com/tier4/autoware-ml/pull/215)
  - [Config file path](https://github.com/tier4/autoware-ml/blob/5b71e0d4b51c5024e3dbcc64506365bbe68f8f0b/projects/BEVFusion/configs/t4dataset/bevfusion_lidar_voxel_second_secfpn_2xb2_t4offline_no_intensity.py)
  - [Training results](https://drive.google.com/drive/folders/1qUCfjYRaO2v_EePVK2btaxjlrBZqdWhk?usp=drive_link)
  - train time: NVIDIA A100 80GB * 2 * 30 epochs = 4 days
  - Total mAP to test dataset (eval range = 120m): 0.722
  - Best epoch: epoch_30.pth

| class_name | Count  | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ------ | ---- | ------- | ------- | ------- | ------- |
| car        | 41,133 | 81.2 | 68.6    | 82.1    | 86.4    | 87.7    |
| truck      | 8,890  | 60.5 | 33.5    | 57.3    | 72.3    | 78.9    |
| bus        | 3,275  | 72.2 | 58.6    | 73.0    | 77.7    | 79.7    |
| bicycle    | 3,635  | 73.0 | 70.0    | 73.5    | 73.8    | 74.9    |
| pedestrian | 25,981 | 73.9 | 69.8    | 72.2    | 75.3    | 78.2    |

</details>

### BEVFusion-L-offline base/0.1

- We released first BEVFusion-L-offline model to use for auto labeling.

<details>
<summary> The link of data and evaluation result </summary>

- model
  - Training dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0
  - Eval dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0
  - [PR](https://github.com/tier4/autoware-ml/pull/110)
  - [Config file path](https://github.com/tier4/autoware-ml/blob/249ebfe5cff685c0911c664ea1ef2b855cc6b52f/projects/BEVFusion/configs/t4dataset/bevfusion_lidar_voxel_second_secfpn_1xb1_t4offline.py)
  - [Checkpoint](https://drive.google.com/drive/folders/16f-IDF0_qXwEbln6RKKkLolQ3cDkZg35)
  - Results are in internal data
  - 3 batch A100 * 2 * 6 days
  - Total mAP to test dataset (eval range = 120m): 0.657

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 76.2 | 57.6    | 77.1    | 83.9    | 86.2    |
| truck      | 56.4 | 26.0    | 55.4    | 70.4    | 73.9    |
| bus        | 65.6 | 39.0    | 67.1    | 77.8    | 78.6    |
| bicycle    | 65.0 | 61.1    | 65.4    | 66.4    | 67.0    |
| pedestrian | 65.3 | 59.8    | 64.2    | 67.5    | 69.8    |

</details>
