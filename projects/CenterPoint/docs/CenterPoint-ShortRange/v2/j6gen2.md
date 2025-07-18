# Deployed model for CenterPoint J6Gen2/1.X
## Summary

### Overview
- Main parameter
  - range = 51.20m
  - voxel_size = [0.16, 0.16, 8.0]
  - grid_size = [640, 640, 1]
	- **With Intensity**
- Performance summary
  - Dataset: test dataset of db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v4 + db_largebus_v1 (total frames: 1,761)
  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m):

| eval range: 52m         | mAP  | car <br> (24,343) | truck <br> (1,965) | bus <br> (866) | bicycle <br> (689) | pedestrian <br> (6,333) |
| ---------------------    | ---- | ----------------- | ------------------- | ---------------- | ----------------- | ---------------- |
| CenterPoint-ShortRange J6Gen2/2.0.1 | 86.00 | 95.30            | 76.20               | 94.50         | 85.90                 | 77.90            |


### Datasets

<details>
<summary> LargeBus </summary>

- Test datases: db_largebus_v1 (total frames: 604)

| eval range: 52m         | mAP  | car <br> (6,232)     | truck <br> (675) | bus <br> (43) | bicycle <br> (463) | pedestrian <br> (2,129) |
| -------------------------| ---- | -------------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| CenterPoint-ShortRange J6Gen2/2.0.1     | 87.70   | 95.10   | 86.40   | 98.30 |   81.4  | 77.20 |

</details>

<details>
<summary> J6Gen2 </summary>

- Test datases: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v4 (total frames: 1,157)

| eval range: 52m         | mAP  | car <br> (18,111)     | truck <br> (1,290) | bus <br> (823) | bicycle <br> (226) | pedestrian <br> (4,204) |
| -------------------------| ---- | -------------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| CenterPoint-ShortRange J6Gen2/2.0.1     | 87.00   | 95.30   | 71.00   | 94.30 |   96.00  | 78.20 |

</details>

## Release

### CenterPoint J6Gen2/2.0.1
- Changes:
  - Finetune from `CenterPoint-ShortRange base/2.0.0`
	- Include intensity as an extra feature

<details>
<summary> The link of data and evaluation result </summary>

- Model
  - Training dataset: DB J6 Gen2 v1.0 + DB J6 Gen2 v2.0 + DB J6 Gen2 V4.0 + DB LargeBus v1.0 (total frames: 20,777)
  - [Config file path](https://github.com/tier4/AWML/blob/32c7a3d4d7c6b2e6fd6ef56cfc7d305486c5ec1d/projects/CenterPoint/configs/t4dataset/CenterPoint-ShortRange/pillar_016_convnext_secfpn_4xb16_50m_j6gen2.py)
  - Deployed onnx model and ROS parameter files [[WebAuto (for internal)]](https://evaluation.tier4.jp/evaluation/mlpackages/7156b453-2861-4ae9-b135-e24e48cc9029/releases/c74579f9-e7b3-4956-93a1-af90cc41171a?project_id=zWhWRzei)
  - Deployed onnx and ROS parameter files [[model-zoo]]
    - [detection_class_remapper.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/j6gen2/v2.0.1/detection_class_remapper.param.yaml)
    - [centerpoint_ml_package.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/j6gen2/v2.0.1/centerpoint_short_range_ml_package.param.yaml)
    - [deploy_metadata.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/j6gen2/v2.0.1/deploy_metadata.yaml)
    - [pts_voxel_encoder_centerpoint.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/j6gen2/v2.0.1/pts_voxel_encoder_centerpoint_short_range.onnx)
    - [pts_backbone_neck_head_centerpoint.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/j6gen2/v2.0.1/pts_backbone_neck_head_centerpoint_short_range.onnx)
  - Training results [[Google drive (for internal)]](https://drive.google.com/drive/folders/1OGhyp0RObqiHvejTm2ku1QMc4CnLCrn0?usp=drive_link)
  - Training results [model-zoo]
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/j6gen2/v2.0.1/logs.zip)
    - [checkpoint_best.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/j6gen2/v2.0.1/best_NuScenes_metric_T4Metric_mAP_epoch_28.pth)
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/j6gen2/v2.0.1/pillar_016_convnext_secfpn_4xb16_50m_j6gen2.py)
  - Train time: NVIDIA H100 80GB * 4 * 30 epochs = 12 hours
  - Batch size: 4*16 = 64

- Evaluation
  - db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v4 + + db_largebus_v1 (total frames: 1,761):
  - Total mAP (eval range = 52m): 0.86

| class_name | Count    | mAP    | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| -----------| -------  | ------ | ------- | ------- | ------- | ------- |
| car        |  24,343  | 95.3   | 91.4      | 96.3  | 96.7    | 96.8    |
| truck      |   1,965  | 76.2   | 67.1      | 76.9  | 79.5    | 81.3    |
| bus        |     866  | 94.5   | 90.9      | 92.7  | 97.2    | 97.3    |
| bicycle    |     689  | 85.9   | 82.9      | 87.0  | 87.0    | 87.0    |
| pedestrian |   6,333  | 77.9   | 76.0      | 77.2  | 78.3    | 80.1    |

- db_largebus_v1 (total frames: 604):
  - Total mAP (eval range = 52m): 0.877

| class_name | Count    | mAP    | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| -----------| -------  | -----  | ------- | ------- | ------- | ------- |
| car        |  6,232   | 95.1 | 91.1    | 95.8    | 96.8    | 96.9    |
| truck      |    675   | 86.4 | 77.9    | 87.2    | 90.2    | 90.4    |
| bus        |     43   | 98.3 | 94.9    | 99.4    | 99.4    | 99.4    |
| bicycle    |    463   | 81.4 | 78.3    | 82.1    | 82.1    | 83.0    |
| pedestrian |  2,129   | 77.2 | 74.8    | 76.6    | 77.9    | 79.6    |

- db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v2 (total frames: 1,157):
  - Total mAP (eval range = 52m): 0.87

| class_name  | Count   | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----------  | ------  | ---- | ------- | ------- | ------- | ------- |
| car         | 18,111  | 95.3 | 91.5    | 96.3    | 96.7    | 96.8    |
| truck       |  1,290  | 71.0 | 61.9    | 71.7    | 74.1    | 76.4    |
| bus         |    823  | 94.3 | 90.3    | 92.5    | 97.2    | 97.2    |
| bicycle     |    226  | 96.0 | 95.1    | 96.3    | 96.3    | 96.3    |
| pedestrian  |  4,204  | 78.2 | 76.5    | 77.3    | 78.6    | 80.3    |

</details>
