# Deployed models for BevFusion-T4offline
## 120m model

- Performance summary
  - Dataset: database_v1_0 + database_v1_1 + database_v_1_3 + database_v_2_0 + database_v_3_0 (total frames: 34,137)
  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m)

| eval range: 120m                         | range  | mAP  | car <br> (610,396)   | truck <br> (151,672) | bus <br> (37,876)  | bicycle <br> (47,739) | pedestrian <br> (367,200) |
| -----------------------------------------| ------ | ---- | ---------------------| -------------------- | -------------------| --------------------- | ------------------------- |
| BevFusion-L/t4base_offline_120m_v1       | 121.0m | 72.2 | 81.2                 | 60.5                 | 72.2               | 73.0                  | 73.9                      |


### BevFusion-L T4offline/v1

- model
  - Training dataset: database_v1_0 + database_v1_1 + database_v_1_3 + database_v_2_0 + database_v_3_0 (total frames: 34,137)
  - Eval dataset: database_v1_0 + database_v1_1 + database_v_1_3 + database_v_2_0 + database_v_3_0 (total frames: 62)
  - *No intensity*
  - [PR](https://github.com/tier4/autoware-ml/pull/215)
  - [Config file path](https://github.com/tier4/autoware-ml/blob/5b71e0d4b51c5024e3dbcc64506365bbe68f8f0b/projects/BEVFusion/configs/t4dataset/bevfusion_lidar_voxel_second_secfpn_2xb2_t4offline_no_intensity.py)
  - [Training results](https://drive.google.com/drive/folders/1qUCfjYRaO2v_EePVK2btaxjlrBZqdWhk?usp=drive_link)
  - train time: NVIDIA A100 80GB * 2 * 30 epochs = 4 days
  - Total mAP to test dataset (eval range = 120m): 0.722
  - Best epoch: epoch_30.pth 

    | class_name   | Count  | mAP      | AP@0.5m   | AP@1.0m   | AP@2.0m   | AP@4.0m   |
    |------------- |------- |--------- |---------  |---------- |---------- |---------- |
    | car          | 41,133 | 81.2     | 68.6      | 82.1      | 86.4      | 87.7      |
    | truck        |  8,890 | 60.5     | 33.5      | 57.3      | 72.3      | 78.9      |
    | bus          |  3,275 | 72.2     | 58.6      | 73.0      | 77.7      | 79.7      |
    | bicycle      |  3,635 | 73.0     | 70.0      | 73.5      | 73.8      | 74.9      |
    | pedestrian   | 25,981 | 73.9     | 69.8      | 72.2      | 75.3      | 78.2      |
