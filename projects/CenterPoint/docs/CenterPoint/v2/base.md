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
  - This releases add more data to `db_j6gen2_v1`
  - Use `PillarFeatureNet` instead `BackwardPillarFeatureNet`
  - Add new label mapping: `construction_vehicle: truck`
  - Clip velocity in data when it exceeds a threshold, where the velocity can be abnormal

- Overall:
  - Slightly better overall (+0.25 mAP)
  - Car: Almost unchanged
  - Truck: Slight improvement in 1.7
  - Bus: Small gain in 1.7
  - Bicycle: Minor improvement
  - Pedestrian: Slight increase

<details>
<summary> The link of data and evaluation result </summary>

- Model
  - Training dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v4.0 + DB GSM8 v1.0 + DB J6 v1.0 + DB J6 v2.0 + DB J6 v3.0 + DB J6 v5.0 + DB J6 Gen2 v1.0 + DB J6 Gen2 v1.1 + DB J6 Gen2 v2.0 + DB LargeBus v1.0 (total frames: 58,323)
  - [Config file path](https://github.com/tier4/AWML/blob/6db4a553d15b18ac6471d228a236c014f55c8307/autoware_ml/configs/detection3d/dataset/t4dataset/base.py)
  - Deployed onnx model and ROS parameter files [[WebAuto (for internal)]](https://evaluation.tier4.jp/evaluation/mlpackages/7156b453-2861-4ae9-b135-e24e48cc9029/releases/41d44753-739c-430e-b0c3-c6c707b22ad2?project_id=zWhWRzei)
  - Deployed onnx and ROS parameter files [[model-zoo]]
    - [detection_class_remapper.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v1.7/detection_class_remapper.param.yaml)
    - [centerpoint_t4base_ml_package.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v1.7/centerpoint_t4base_ml_package.param.yaml)
    - [deploy_metadata.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v1.7/deploy_metadata.yaml)
    - [pts_voxel_encoder_centerpoint_t4base.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v1.7/pts_voxel_encoder.onnx)
    - [pts_backbone_neck_head_centerpoint_t4base.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v1.7/pts_backbone_neck_head.onnx)
  - Training results [[Google drive (for internal)]](https://drive.google.com/drive/folders/1dVri0Jq9_yobzed0T2Rno-mfChbjPesn?usp=drive_link)
  - Training results [model-zoo]
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v1.7/logs.zip)
    - [checkpoint_best.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v1.7/best_NuScenes_metric_T4Metric_mAP_epoch_49.pth)
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v1.7/second_secfpn_4xb16_121m_base_amp.py)
  - Train time: NVIDIA H100 80GB * 4 * 50 epochs = 2 days and 5 hours
  - Batch size: 4*16 = 64

- Evaluation
  - db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 + db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 + db_j6gen2_v1 + db_largebus_v1 (total frames: 4,199):
  - Total mAP (eval range = 120m): 0.6821

| class_name | Count    | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| -----------| -------  | ----  | ------- | ------- | ------- | ------- |
| car        |  90,242  | 81.13 | 73.12  | 82.26  | 84.41  | 84.75  |
| truck      |  14,910  | 53.95 | 35.21  | 54.71  | 60.02  | 65.86  |
| bus        |   4,992  | 80.97 | 73.91  | 81.00  | 83.87  | 85.14  |
| bicycle    |   4,666  | 59.19 | 73.91  | 81.00  | 83.87  | 85.14  |
| pedestrian |  36,690  | 65.79 | 63.84  | 65.09  | 66.38  | 67.86  |

- db_largebus_v1 (total frames: 315):
  - Total mAP (eval range = 120m): 0.7414

| class_name | Count    | mAP    | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| -----------| -------  | -----  | ------- | ------- | ------- | ------- |
| car        |  5,714   | 87.69  | 81.25  | 88.85  | 90.21  | 90.46  |
| truck      |  1,123   | 59.49  | 51.25  | 59.61  | 63.12  | 64.00  |
| bus        |     51   | 97.70  | 95.04  | 98.51  | 98.65  | 98.65  |
| bicycle    |    504   | 62.59  | 58.57  | 62.58  | 64.61  | 64.61  |
| pedestrian |  2,782   | 63.25  | 61.53  | 62.71  | 63.67  | 65.11  |

- db_j6gen2_v1 + db_j6gen2_v2 (total frames: 801):
  - Total mAP (eval range = 120m): 0.7228

| class_name  | Count   | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----------  | ------  | ---- | ------- | ------- | ------- | ------- |
| car         | 28,002  | 85.96 | 80.38  | 85.95  | 88.26  | 89.26  |
| truck       |  1,123  | 53.52 | 47.83  | 54.24  | 55.64  | 56.38  |
| bus         |  1,203  | 84.21 | 80.39  | 82.48  | 86.97  | 87.00  |
| bicycle     |    223  | 74.96 |  73.43  | 75.21  | 75.21  | 75.99  |
| pedestrian  |   4,407 | 62.78 | 61.46  | 62.05  | 63.12  | 64.52  |

</details>
