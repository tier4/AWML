# Deployed model for CenterPoint base/1.X
## Summary

### Overview
- Main parameter
  - range = 121.60m
  - voxel_size = [0.32, 0.32, 8.0]
  - grid_size = [760, 760, 1]
- Detailed comparison
  - [Internal Link](https://docs.google.com/spreadsheets/d/1jkadazpbA2BUYEUdVV8Rpe54-snH1cbdJbbHsuK04-U/edit?usp=sharing)
- Performance summary
  - Dataset: test dataset of db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 + db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 + db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v4 + db_largebus_v1 (total frames: 5,703)
  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m):

| eval range: 120m         | mAP  | car <br> (144,001) | truck <br> (20,823) | bus <br> (5,691) | bicycle <br> (5,007) | pedestrian <br> (42,034) |
| -------------------------| ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| CenterPoint base/2.0     | 68.06 | 82.19            | 55.44               | 79.30         | 57.62                 | 65.74                   |
| CenterPoint base/1.7     | 67.54 | 81.40            | 51.60               | 80.11         | 59.61                 | 64.96                   |


### Datasets

<details>
<summary> LargeBus </summary>

- Test datases: db_largebus_v1 (total frames: 604)

| eval range: 120m         | mAP  | car <br> (13,831)     | truck <br> (2,137) | bus <br> (95) | bicycle <br> (724) | pedestrian <br> (3,916) |
| -------------------------| ---- | -------------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| CenterPoint base/2.0          | 71.09   | 89.01   | 64.11   | 77.75 | 61.04       | 63.56       |
| CenterPoint base/1.7          | 69.15   | 87.87   | 53.81   | 78.40   | 63.08     | 62.58       |

</details>

<details>
<summary> J6Gen2 </summary>

- Test datases: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v4 (total frames: 1,157)

| eval range: 120m         | mAP  | car <br> (44,008) | truck <br> (2,471) | bus <br> (1,464) | bicycle <br> (333) | pedestrian <br> (6,459) |
| -------------------------| ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| CenterPoint base/2.0     | 71.04 | 83.56 | 53.08 | 85.06 | 68.54   | 64.97       |
| CenterPoint base/1.7     | 70.10 | 82.27 | 49.65 | 81.78 | 73.67   | 63.14       |

</details>

<details>
<summary> JPNTaxi </summary>

- Test datases: db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 (total frames: 1,507)

| eval range: 120m         | mAP     | car <br> (16,142) | truck <br> (4,578) | bus <br> (1,457) | bicycle <br> (1,040) | pedestrian <br> (11,971) |
| -------------------------| ----    | ----------------- | ------------------- | ---------------- | --------------- | ------------------------|
| CenterPoint base/2.0     | 66.20   | 76.02             | 52.59               | 71.18            | 63.55           | 67.67                   |
| CenterPoint base/1.7     | 65.86   | 75.46             | 51.65               | 73.10            | 61.25           | 67.82                   |

</details>

<details>
<summary> J6 </summary>

- Test datases: db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 (total frames: 2,435)

| eval range: 120m         | mAP     | car <br> (67,551) | truck <br> (10,013) | bus <br> (2,503) | bicycle <br> (2,846) | pedestrian <br> (19,117) |
| -------------------------| ------- | ----------------- | ------------------- | ---------------- | ---------------- | -------------------- |
| CenterPoint base/2.0     | 67.41   | 82.09             | 56.08               | 80.41            | 53.75      			 | 64.71                |
| CenterPoint base/1.7     | 67.35  | 81.28             | 52.10               | 83.28            | 56.22             | 63.90                |

</details>

## Release

### CenterPoint base/2.0
- Changes:
  - Add new dataset `db_j6gen2_v4`
  - Add more data to `db_j6_v3`, `db_j6_v5`, `db_j6gen2_v1`, `db_j6gen2_v2`, `db_largebus_v1`  

- Overall:
  - Slightly better overall (+0.52 mAP)
  - Car and pedestrian detection remain fairly stable, with small improvements in `base/2.0`.
  - Truck detection shows the largest improvement (+3.84) in `base/2.0`.
  - Bus and bicycle performance slightly drops in `base/2.0`.

<details>
<summary> The link of data and evaluation result </summary>

- Model
  - Training dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v4.0 + DB GSM8 v1.0 + DB J6 v1.0 + DB J6 v2.0 + DB J6 v3.0 + DB J6 v5.0 + DB J6 Gen2 v1.0 + DB J6 Gen2 v2.0 + DB J6 Gen2 v4.0 + DB LargeBus v1.0 (total frames: 71,633)
  - [Config file path](https://github.com/tier4/AWML/blob/c50daa0f941da334a2167a4aa587589f6ab76a85/autoware_ml/configs/detection3d/dataset/t4dataset/base.py)
  - Deployed onnx model and ROS parameter files [[WebAuto (for internal)]](https://evaluation.tier4.jp/evaluation/mlpackages/7156b453-2861-4ae9-b135-e24e48cc9029/releases/4489a6b0-e8f4-4204-a217-2889f18d3b66?project_id=zWhWRzei)
  - Deployed onnx and ROS parameter files [[model-zoo]]
    - [detection_class_remapper.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.0.0/detection_class_remapper.param.yaml)
    - [centerpoint_ml_package.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.0.0/centerpoint_ml_package.param.yaml)
    - [deploy_metadata.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.0.0/deploy_metadata.yaml)
    - [pts_voxel_encoder_centerpoint.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.0.0/pts_voxel_encoder.onnx)
    - [pts_backbone_neck_head_centerpoint.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.0.0/pts_backbone_neck_head.onnx)
  - Training results [[Google drive (for internal)]](https://drive.google.com/drive/folders/1QspUscYcPbWPGAkC321L_s7W70gsZ2Ad?usp=drive_link)
  - Training results [model-zoo]
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.0.0/logs.zip)
    - [checkpoint_best.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.0.0/best_NuScenes_metric_T4Metric_mAP_epoch_49.pth)
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.0.0/second_secfpn_4xb16_121m_base_amp.py)
  - Train time: NVIDIA H100 80GB * 4 * 50 epochs = 2 days and 20 hours
  - Batch size: 4*16 = 64

- Evaluation
  - db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 + db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 + db_j6gen2_v1 + db_j6gen2_v1 + db_j6gen2_v4 + db_largebus_v1 (total frames: 5,703):
  - Total mAP (eval range = 120m): 0.6806

| class_name | Count    | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| -----------| -------  | ----  | ------- | ------- | ------- | ------- |
| car        |  144,001  | 82.19 | 75.15    | 83.01    | 85.17    | 85.45    |
| truck      |  20,823  | 55.44 | 40.06    | 56.20    | 60.07    | 65.44    |
| bus        |   5,691  | 79.30 | 71.94    | 79.83    | 82.51    | 82.96    |
| bicycle    |   5,007  | 57.62 | 55.87    | 58.15    | 58.21    | 58.28    |
| pedestrian |  42,034  | 65.74 | 63.60    | 64.92    | 66.32    | 68.10    |

- db_largebus_v1 (total frames: 604):
  - Total mAP (eval range = 120m): 0.7109

| class_name | Count    | mAP    | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| -----------| -------  | -----  | ------- | ------- | ------- | ------- |
| car        |  13,831   | 89.01  | 83.62    | 89.78    | 90.98    | 91.70    |
| truck      |  2,137   | 64.11  | 51.19    | 65.25    | 68.75    | 71.25    |
| bus        |     95   | 77.75  | 71.04    | 79.99    | 79.99    | 79.99    |
| bicycle    |    724   | 61.04  | 55.98    | 62.13    | 63.04    | 63.04    |
| pedestrian |  3,916   | 63.56  | 61.90    | 63.03    | 63.78    | 65.55    |

- db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v2 (total frames: 1,157):
  - Total mAP (eval range = 120m): 0.7104

| class_name  | Count   | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----------  | ------  | ---- | ------- | ------- | ------- | ------- |
| car         | 44,008  | 83.56 | 77.31    | 83.73    | 86.11    | 87.12    |
| truck       |  2,471  | 53.08 | 43.13    | 52.46    | 54.80    | 61.93    |
| bus         |  1,464  | 85.06 | 79.26    | 83.83    | 88.56    | 88.60    |
| bicycle     |    333  | 68.54 | 67.73    | 68.82    | 68.82    | 68.82    |
| pedestrian  |  6,459  | 64.97 | 63.12    | 64.26    | 65.41    | 67.10    |

</details>
