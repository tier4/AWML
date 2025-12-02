# Deployed model for CenterPoint base/1.X
## Summary

### Overview
- Main parameter
  - range = 51.20m
  - voxel_size = [0.16, 0.16, 8.0]
  - grid_size = [640, 640, 1]
- Performance summary
  - Dataset: test dataset of db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 + db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 + db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v4 + db_largebus_v1 (total frames: 5,703)
  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m):

| eval range: 52m                 | mAP  | car <br> (65,073) | truck <br> (7,763) | bus <br> (2,476) | bicycle <br> (2,766) | pedestrian <br> (26,254) |
| ------------------------------- | ---- | ------------------ | -------------------- | ----------------- | --------------------- | ------------------------- |
| CenterPoint-ShortRange base/2.0 | 83.9 | 93.2               | 70.2                 | 93.8              | 84.7                  | 77.5                      |

### Datasets

<details>
<summary> LargeBus </summary>

- Test datases: db_largebus_v1 (total frames: 604)

| eval range: 52m         | mAP  | car <br> (6,232)     | truck <br> (675) | bus <br> (43) | bicycle <br> (463) | pedestrian <br> (2,129) |
| -------------------------| ---- | -------------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| CenterPoint-ShortRange base/2.0     | 86.20   | 94.60   | 82.8   | 97.70 |   80.2  | 75.90 |

</details>

<details>
<summary> J6Gen2 </summary>

- Test datases: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v4 (total frames: 1,157)

| eval range: 52m         | mAP  | car <br> (18,111)     | truck <br> (1,290) | bus <br> (823) | bicycle <br> (226) | pedestrian <br> (4,204) |
| -------------------------| ---- | -------------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| CenterPoint-ShortRange base/2.0     | 86.10   | 95.10   | 71.10   | 94.90 |   91.20  | 78.00 |

</details>

<details>
<summary> JPNTaxi </summary>

- Test datases: db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 (total frames: 1,507)

| eval range: 52m         | mAP     | car <br> (8,401) | truck <br> (2,125) | bus <br> (769) | bicycle <br> (724) | pedestrian <br> (8,333) |
| -------------------------| ----    | ----------------- | ------------------- | ---------------- | --------------- | ------------------------|
| CenterPoint-ShortRange base/2.0     | 78.50   | 88.00   | 61.60   | 92.10 |   74.70  | 75.90 |


</details>

<details>
<summary> J6 </summary>

- Test datases: db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 (total frames: 2,435)

| eval range: 52m         | mAP     | car <br> (32,329) | truck <br> (3,673) | bus <br> (841) | bicycle <br> (1,353) | pedestrian <br> (11,588) |
| -------------------------| ----    | ----------------- | ------------------- | ---------------- | --------------- | ------------------------|
| CenterPoint-ShortRange base/2.0     | 86.10   | 92.90   | 74.90   | 94.00 |   90.00  | 78.60 |


</details>

## Release

### CenterPoint-ShortRange base/2.0
- Release:
  - Same datasets used in `base/2.0.0` with shorter range, higher resolution and stronger backbone

<details>
<summary> The link of data and evaluation result </summary>

- Model
  - Training dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v4.0 + DB GSM8 v1.0 + DB J6 v1.0 + DB J6 v2.0 + DB J6 v3.0 + DB J6 v5.0 + DB J6 Gen2 v1.0 + DB J6 Gen2 v1.1 + DB J6 Gen2 v2.0 + DB LargeBus v1.0 (total frames: 58,323)
  - [Config file path](https://github.com/tier4/AWML/blob/32c7a3d4d7c6b2e6fd6ef56cfc7d305486c5ec1d/projects/CenterPoint/configs/t4dataset/CenterPoint-ShortRange/pillar_016_convnext_secfpn_4xb16_50m_base.py)
  - Deployed onnx model and ROS parameter files [[WebAuto (for internal)]](https://evaluation.tier4.jp/evaluation/mlpackages/7156b453-2861-4ae9-b135-e24e48cc9029/releases/53ff6785-c35f-462c-a913-c436b57b6959?project_id=zWhWRzei)
  - Deployed onnx and ROS parameter files [[model-zoo]]
    - [detection_class_remapper.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/t4base/v2.0.0/detection_class_remapper.param.yaml)
    - [centerpoint_ml_package.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/t4base/v2.0.0/centerpoint_short_range_ml_package.param.yaml)
    - [deploy_metadata.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/t4base/v2.0.0/deploy_metadata.yaml)
    - [pts_voxel_encoder_centerpoint.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/t4base/v2.0.0/pts_voxel_encoder_centerpoint_short_range.onnx)
    - [pts_backbone_neck_head_centerpoint.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/t4base/v2.0.0/pts_backbone_neck_head_centerpoint_short_range.onnx)
  - Training results [[Google drive (for internal)]](https://drive.google.com/drive/folders/1RpEMTdFFbcO_4FQGeCKhiKo_3RL6aePB?usp=drive_link)
  - Training results [model-zoo]
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/t4base/v2.0.0/logs.zip)
    - [checkpoint_best.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/t4base/v2.0.0/best_NuScenes_metric_T4Metric_mAP_epoch_48.pth)
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/t4base/v2.0.0/pillar_016_convnext_secfpn_4xb16_50m_base.py)
  - Train time: NVIDIA H100 80GB * 4 * 50 epochs = 1 days and 20 hours
  - Batch size: 4*16 = 64

- Evaluation
  - db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 + db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 + db_j6gen2_v1 + db_j6gen2_v1 + db_j6gen2_v4 + db_largebus_v1 (total frames: 5,703):
  - Total mAP (eval range = 52m): 0.839

| class_name | Count     | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| -----------| -------   | ----  | ------- | ------- | ------- | ------- |
| car        |  65,073   | 93.2 | 88.7    | 94.2    | 94.6    | 95.4    |
| truck      |   7,763   | 70.2 | 57.1    | 72.2    | 74.7    | 76.9    |
| bus        |   2,476   | 93.8 | 88.7    | 93.8    | 96.1    | 96.6    |
| bicycle    |   2,766   | 84.7 | 83.3    | 84.6    | 85.4    | 85.6    |
| pedestrian |  26,254   | 77.5 | 75.1    | 76.9    | 78.3    | 79.8    |

- db_largebus_v1 (total frames: 604):
  - Total mAP (eval range = 52m): 0.862

| class_name | Count    | mAP    | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| -----------| -------  | -----  | ------- | ------- | ------- | ------- |
| car        |  6,232   | 94.6   | 91.0    | 95.4    | 95.7    | 96.4    |
| truck      |    675   | 82.8   | 73.3    | 84.6    | 86.4    | 86.9    |
| bus        |     43   | 97.7   | 93.1    | 99.2    | 99.2    | 99.2    |
| bicycle    |    463   | 80.2   | 78.0    | 80.9    | 80.9    | 80.9    |
| pedestrian |  2,129   | 75.9   | 73.6    | 75.4    | 76.7    | 78.1    |


- db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v2 (total frames: 1,157):
  - Total mAP (eval range = 120m): 0.861

| class_name  | Count   | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----------  | ------  | ---- | ------- | ------- | ------- | ------- |
| car         | 18,111  | 95.1 | 91.5    | 95.6    | 96.6    | 96.7    |
| truck       |  1,290  | 71.1 | 63.2    | 72.3    | 73.4    | 75.5    |
| bus         |    823  | 94.9 | 91.7    | 92.9    | 97.4    | 97.4    |
| bicycle     |    226  | 91.2 | 90.5    | 91.5    | 91.5    | 91.5    |
| pedestrian  |  4,204  | 78.0 | 76.0    | 77.2    | 78.5    | 80.2    |

</details>
