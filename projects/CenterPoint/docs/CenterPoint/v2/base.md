# Deployed model for CenterPoint base/1.X
## Summary

### Overview
- Main parameter
  - range = 121.60m
  - voxel_size = [0.32, 0.32, 8.0]
  - grid_size = [760, 760, 1]
- Detailed comparison
  - [Internal Link](https://docs.google.com/spreadsheets/d/13Stt9hdbTER6ugaRMEZscbt_6kD7ld-0b30JyeTGTrs/edit?usp=sharing)
  - Dataset: test dataset of db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 + db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 + db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v4 + db_largebus_v1 (total frames: 5,703)
  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m):

| eval range: 120m         | mAP  | car <br> (152,572) | truck <br> (24,172) | bus <br> (9,839) | bicycle <br> (5,317) | pedestrian <br> (50,699) |
| -------------------------| ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| CenterPoint base/2.2     | 66.00 | 82.60            | 55.50               | 71.90         | 56.40                 | 66.50                   |
| CenterPoint base/2.1     | 65.55 | 82.51            | 53.12               | 68.95         | 56.32                 | 66.81                   |

### Deprecated results
- [Internal Link](https://docs.google.com/spreadsheets/d/1jkadazpbA2BUYEUdVV8Rpe54-snH1cbdJbbHsuK04-U/edit?usp=sharing)

- Performance summary
| eval range: 120m         | mAP  | car <br> (144,001) | truck <br> (20,823) | bus <br> (5,691) | bicycle <br> (5,007) | pedestrian <br> (42,034) |
| -------------------------| ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| CenterPoint base/2.1     | 67.85 | 82.34            | 55.15               | 78.10         | 57.36                 | 66.26                   |
| CenterPoint base/2.0     | 68.06 | 82.19            | 55.44               | 79.30         | 57.62                 | 65.74                   |
| CenterPoint base/1.7     | 67.54 | 81.40            | 51.60               | 80.11         | 59.61                 | 64.96                   |

### Datasets

<details>
<summary> JPNTaxi Gen2 </summary>

- Test datases: db_jpntaxi_gen2 (total frames: 1,479)

| eval range: 120m         | mAP     | car <br> (8,571)     | truck <br> (3,349) | bus <br> (4,148) | bicycle <br> (310) | pedestrian <br> (8,665) |
| -------------------------| ----    | -------------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| CenterPoint base/2.2     | 57.01   | 86.30   | 35.50   | 60.80 | 30.50       | 72.10       |
| CenterPoint base/2.1     | 53.44   | 85.00   | 32.50   | 47.00 | 33.40       | 69.30       |

</details>

<details>
<summary> LargeBus </summary>

- Test datases: db_largebus_v1 (total frames: 604)

| eval range: 120m         | mAP     | car <br> (13,831)     | truck <br> (2,137) | bus <br> (95) | bicycle <br> (724) | pedestrian <br> (3,916) |
| -------------------------| ----    | -------------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| CenterPoint base/2.2     | 71.68   | 88.60   | 63.60   | 78.70 | 64.10       | 63.40       |
| CenterPoint base/2.1     | 71.85   | 88.85   | 64.67   | 80.35 | 62.15       | 63.20       |
| CenterPoint base/2.0     | 71.09   | 89.01   | 64.11   | 77.75 | 61.04       | 63.56       |
| CenterPoint base/1.7     | 69.15   | 87.87   | 53.81   | 78.40   | 63.08     | 62.58       |

</details>

<details>
<summary> J6Gen2 </summary>

- Test datases: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v4 (total frames: 1,157)

| eval range: 120m         | mAP  | car <br> (44,008) | truck <br> (2,471) | bus <br> (1,464) | bicycle <br> (333) | pedestrian <br> (6,459) |
| -------------------------| ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| CenterPoint base/2.2     | 72.40 | 83.60 | 54.40 | 84.00 | 74.30   | 65.70       |
| CenterPoint base/2.1     | 71.83 | 83.51 | 53.01 | 84.62 | 73.42   | 64.58       |
| CenterPoint base/2.0     | 71.04 | 83.56 | 53.08 | 85.06 | 68.54   | 64.97       |
| CenterPoint base/1.7     | 70.10 | 82.27 | 49.65 | 81.78 | 73.67   | 63.14       |

</details>

<details>
<summary> JPNTaxi </summary>

- Test datases: db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 (total frames: 1,507)

| eval range: 120m         | mAP     | car <br> (16,142) | truck <br> (4,578) | bus <br> (1,457) | bicycle <br> (1,040) | pedestrian <br> (11,971) |
| -------------------------| ----    | ----------------- | ------------------- | ---------------- | --------------- | ------------------------|
| CenterPoint base/2.2     | 66.43   | 76.00             | 52.70               | 74.60            | 61.00           | 68.00                   |
| CenterPoint base/2.1     | 65.46   | 75.86             | 52.90               | 71.79            | 59.39           | 67.36                   |
| CenterPoint base/2.0     | 66.20   | 76.02             | 52.59               | 71.18            | 63.55           | 67.67                   |
| CenterPoint base/1.7     | 65.86   | 75.46             | 51.65               | 73.10            | 61.25           | 67.82                   |

</details>

<details>
<summary> J6 </summary>

- Test datases: db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 (total frames: 2,435)

| eval range: 120m         | mAP     | car <br> (67,551) | truck <br> (10,013) | bus <br> (2,503) | bicycle <br> (2,846) | pedestrian <br> (19,117) |
| -------------------------| ------- | ----------------- | ------------------- | ---------------- | ---------------- | -------------------- |
| CenterPoint base/2.2     | 65.08   | 82.20             | 54.10               | 72.40            | 53.10      			 | 63.60                |
| CenterPoint base/2.1     | 67.19   | 82.15             | 55.54               | 77.13            | 54.42      			 | 66.66                |
| CenterPoint base/2.0     | 67.41   | 82.09             | 56.08               | 80.41            | 53.75      			 | 64.71                |
| CenterPoint base/1.7     | 67.35   | 81.28             | 52.10               | 83.28            | 56.22             | 63.90                |

</details>

## Release
### CenterPoint base/2.2
- Add a new training set: `db_jpntaxi_gen2_v1`

<details>
<summary> The link of data and evaluation result </summary>

- Model
  - Training dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v4.0 + DB GSM8 v1.0 + DB J6 v1.0 + DB J6 v2.0 + DB J6 v3.0 + DB J6 v5.0 + DB J6 Gen2 v1.0 + DB J6 Gen2 v2.0 + DB J6 Gen2 v4.0 + DB LargeBus v1.0 + DB JPNTAXI_GEN2 v1.0 (total frames: 88,762)
  - [Config file path](http://github.com/tier4/AWML/blob/81314d29d4efa560952324c48ef7c0ea1e56f1ee/projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp.py)
  - Deployed onnx and ROS parameter files (for internal)
    - [WebAuto](https://evaluation.tier4.jp/evaluation/mlpackages/7156b453-2861-4ae9-b135-e24e48cc9029/releases/83ba5207-44c3-46fc-899e-30863dcf1423?project_id=zWhWRzei)
    - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.2.0/deployment.zip)
    - [Google drive](https://drive.google.com/file/d/1v5rJqrv9vmM3RHD-lDSemcrxYJuhiQla/view?usp=drive_link)
  - Logs (for internal)
    - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.2.0/logs.zip)
    - [Google drive](https://drive.google.com/file/d/1aFFA9WRd2G_eqqwVI92jbKQ-daAIJqD0/view?usp=drive_link)
  - Train time: NVIDIA H100 80GB * 4 * 50 epochs = 4 days
  - Batch size: 4*16 = 64

- Evaluation
  - db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 + db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 + db_j6gen2_v1 + db_j6gen2_v1 + db_j6gen2_v4 + db_largebus_v1 + db_jpntaxi_gen2_v1 (total frames: 7,182):
  - Total mAP (eval range = 120m): 0.66

| class_name | Count     | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | ---       | ---- | ---- | ---- | ---- | ---- |
| car        |  152,572  | 82.6 | 75.3    | 83.7    | 85.3    | 86.2    |
| truck      |   24,172  | 52.5 | 35.3    | 53.0    | 57.5    | 64.4    |
| bus        |    5,691  | 71.9 | 61.2    | 72.3    | 76.4    | 77.6    |
| bicycle    |    5,317  | 56.4 | 54.8    | 56.3    | 57.1    | 57.2    |
| pedestrian |   50,699  | 66.5 | 64.5    | 65.6    | 67.0    | 68.7    |

- db_largebus_v1 (total frames: 604):
  - Total mAP (eval range = 120m): 0.7170

| class_name | Count    | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | -------  | ---- | ---- | ---- | ---- | ---- |
| car        |  13,831  | 88.6 | 83.4    | 89.5    | 90.7    | 90.9    |
| truck      |  2,137   | 63.6 | 51.1    | 64.5    | 68.2    | 70.6    |
| bus        |     95   | 78.7 | 76.3    | 79.5    | 79.5    | 79.5    |
| bicycle    |    724   | 64.1 | 59.8    | 64.9    | 65.8    | 65.8    |
| pedestrian |  3,916   | 63.4 | 61.4    | 63.0    | 63.9    | 65.4    |

- db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v2 (total frames: 1,157):
  - Total mAP (eval range = 120m): 0.7240

| class_name | Count   | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | ------  | ---- | ---- | ---- | ---- | ---- |
| car        | 44,008  | 83.6 | 77.3    | 83.8    | 86.1    | 87.1    |
| truck      |  2,471  | 54.4 | 43.8    | 54.5    | 56.8    | 62.4    |
| bus        |  1,464  | 84.0 | 79.3    | 83.4    | 86.7    | 86.7    |
| bicycle    |    333  | 74.3 | 73.7    | 74.5    | 74.5    | 74.5    |
| pedestrian |  6,459  | 65.7 | 64.3    | 65.1    | 65.8    | 67.6    |

- db_jpntaxi_gen2_v1 (total frames: 1,479):
  - Total mAP (eval range = 120m): 0.5701

| class_name | Count | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | ------| ---- | ---- | ---- | ---- | ---- |
| car        | 8,571 | 86.3 | 80.9    | 87.3    | 88.3    | 88.6    |
| truck      | 3,349 | 35.5 | 24.4    | 32.0    | 34.8    | 50.7    |
| bus        | 4,148 | 60.8 | 39.2    | 59.9    | 71.2    | 72.7    |
| bicycle    |   310 | 30.5 | 30.0    | 30.6    | 30.6    | 30.6    |
| pedestrian | 8,665 | 72.1 | 70.4    | 71.1    | 72.3    | 74.5    |

### CenterPoint base/2.1
- Add more training data to `db_j6gen2_v2` and `db_j6gen2_v4`

- Overall:
  - Slightly worse overall (-0.21 mAP).
	- Main improvement comes from `LargeBus` (71.85 vs 71.09) and `J6Gen2` (71.83 vs 71.04).

<details>
<summary> The link of data and evaluation result </summary>

- Model
  - Training dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v4.0 + DB GSM8 v1.0 + DB J6 v1.0 + DB J6 v2.0 + DB J6 v3.0 + DB J6 v5.0 + DB J6 Gen2 v1.0 + DB J6 Gen2 v2.0 + DB J6 Gen2 v4.0 + DB LargeBus v1.0 (total frames: 75,963)
  - [Config file path](https://github.com/tier4/AWML/blob/69aba0d001fd26282880a7a3e7622b89115042de/autoware_ml/configs/detection3d/dataset/t4dataset/base.py)
  - Deployed onnx model and ROS parameter files [[WebAuto (for internal)]](https://evaluation.tier4.jp/evaluation/mlpackages/7156b453-2861-4ae9-b135-e24e48cc9029/releases/b73806f0-404f-4e9c-8b83-d9beb0c66ebd?project_id=zWhWRzei)
  - Deployed onnx and ROS parameter files [[model-zoo]]
    - [detection_class_remapper.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.1.0/detection_class_remapper.param.yaml)
    - [centerpoint_ml_package.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.1.0/centerpoint_ml_package.param.yaml)
    - [deploy_metadata.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.1.0/deploy_metadata.yaml)
    - [pts_voxel_encoder_centerpoint.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.1.0/pts_voxel_encoder_centerpoint.onnx)
    - [pts_backbone_neck_head_centerpoint.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.1.0/pts_backbone_neck_head_centerpoint.onnx)
  - Training results [[Google drive (for internal)]](https://drive.google.com/drive/u/0/folders/1FNw3bEvM1Z9Igp-uUzvXQFjLObONfwyg)
  - Training results [model-zoo]
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.1.0/logs.zip)
    - [checkpoint_best.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.1.0/best_NuScenes_metric_T4Metric_mAP_epoch_49.pth)
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.1.0/second_secfpn_4xb16_121m_base_amp.py)
  - Train time: NVIDIA H100 80GB * 4 * 50 epochs = 2 days and 23 hours
  - Batch size: 4*16 = 64

- Evaluation
  - db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 + db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 + db_j6gen2_v1 + db_j6gen2_v1 + db_j6gen2_v4 + db_largebus_v1 (total frames: 5,703):
  - Total mAP (eval range = 120m): 0.678

| class_name | Count    | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| -----------| -------  | ----  | ------- | ------- | ------- | ------- |
| car        |  144,001 | 82.3 | 75.1    | 83.0    | 85.2    | 86.1    |
| truck      |  20,823  | 55.2 | 40.0    | 55.5    | 59.8    | 65.4    |
| bus        |   5,691  | 78.1 | 70.7    | 79.0    | 80.8    | 81.8    |
| bicycle    |   5,007  | 57.4 | 56.3    | 57.7    | 57.7    | 57.8    |
| pedestrian |  42,034  | 66.3 | 64.0    | 65.4    | 66.9    | 68.8    |

- db_largebus_v1 (total frames: 604):
  - Total mAP (eval range = 120m): 0.7190

| class_name | Count    | mAP    | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| -----------| -------  | -----  | ------- | ------- | ------- | ------- |
| car        |  13,831   | 88.9 | 83.6    | 89.7    | 91.0    | 91.1    |
| truck      |  2,137   | 64.7 | 51.3    | 65.2    | 69.8    | 72.3    |
| bus        |     95   | 80.4 | 77.9    | 81.2    | 81.2    | 81.2    |
| bicycle    |    724   | 62.2 | 57.8    | 63.0    | 63.9    | 63.9    |
| pedestrian |  3,916   | 63.2 | 61.4    | 62.7    | 63.6    | 65.1    |

- db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v2 (total frames: 1,157):
  - Total mAP (eval range = 120m): 0.7180

| class_name  | Count   | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----------  | ------  | ---- | ------- | ------- | ------- | ------- |
| car         | 44,008  | 83.5 | 77.2    | 83.7    | 86.1    | 87.1    |
| truck       |  2,471  | 53.0 | 42.9    | 52.5    | 55.0    | 61.7    |
| bus         |  1,464  | 84.6 | 81.2    | 83.8    | 86.8    | 86.8    |
| bicycle     |    333  | 73.4 | 72.3    | 73.8    | 73.8    | 73.9    |
| pedestrian  |  6,459  | 64.6 | 63.1    | 64.0    | 64.9    | 66.4    |

</details>

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
