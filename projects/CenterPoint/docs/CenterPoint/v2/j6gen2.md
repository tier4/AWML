# Deployed model for CenterPoint J6Gen2/1.X
## Summary

### Overview
- Main parameter
  - range = 121.60m
  - voxel_size = [0.32, 0.32, 8.0]
  - grid_size = [760, 760, 1]
	- **With Intensity**
- Detailed comparison
  - [Internal Link](https://docs.google.com/spreadsheets/d/1jkadazpbA2BUYEUdVV8Rpe54-snH1cbdJbbHsuK04-U/edit?usp=sharing)
- Performance summary
  - Dataset: test dataset of db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v4 + db_largebus_v1 (total frames: 1,761)
  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m):

| eval range: 120m         | mAP  | car <br> (57,839) | truck <br> (4,608) | bus <br> (1,559) | bicycle <br> (1,057) | pedestrian <br> (10,375) |
| ---------------------    | ---- | ----------------- | ------------------- | ---------------- | ----------------- | ---------------- |
| CenterPoint J6Gen2/2.1.1 | 74.07 | 85.80            | 61.74               | 84.94         | 70.63                 | 67.24            |
| CenterPoint J6Gen2/2.0.1 | 74.01 | 85.97            | 62.57               | 83.36         | 71.05                 | 67.10            |
| CenterPoint J6Gen2/1.7.1 | 71.77 | 84.80            | 56.64               | 81.78         | 71.26                 | 64.38            |


### Datasets

<details>
<summary> LargeBus </summary>

- Test datases: db_largebus_v1 (total frames: 604)

| eval range: 120m         | mAP  | car <br> (13,831)     | truck <br> (2,137) | bus <br> (95) | bicycle <br> (724) | pedestrian <br> (3,916) |
| -------------------------| ---- | -------------------- | ------------------- | ---------------- | ------------ | ------------------------ |
| CenterPoint J6Gen2/2.1.1   | 74.99   | 90.35   | 70.43   | 79.63 | 67.51       | 66.99       |
| CenterPoint J6Gen2/2.0.1   | 74.73   | 90.53   | 69.55   | 80.06 | 65.72       | 67.82       |
| CenterPoint J6Gen2/1.7.1   | 72.17   | 89.36   | 60.87   | 78.40   | 67.71     | 64.84       |

</details>

<details>
<summary> J6Gen2 </summary>

- Test datases: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v4 (total frames: 1,157)

| eval range: 120m         | mAP  | car <br> (44,008) | truck <br> (2,471) | bus <br> (1,464) | bicycle <br> (333) | pedestrian <br> (6,459) |
| -------------------------| ---- | ----------------- | ------------------- | ---------------- | -------------------- | ---------------------|
| CenterPoint J6Gen2/2.1.1   | 74.24   | 84.65   | 54.66   | 85.45 | 79.07       | 67.35       |
| CenterPoint J6Gen2/2.0.1   | 74.60   | 84.83   | 56.48   | 83.70 | 81.06       | 66.91       |
| CenterPoint J6Gen2/1.7.1   | 72.35   | 83.34   | 53.26   | 82.13 | 79.10     | 63.92       |

</details>

## Release

### CenterPoint J6Gen2/2.1.1
- Changes:
  - Finetune from `CenterPoint base/2.1.0`
	- Include intensity as an extra feature

- Overall:
	- Similar performance compared to `CenterPoint J6Gen2/2.0.1`

<details>
<summary> The link of data and evaluation result </summary>

- Model
  - Training dataset: DB J6 Gen2 v1.0 + DB J6 Gen2 v2.0 + DB J6 Gen2 V4.0 + DB LargeBus v1.0 (total frames: 20,777)
  - [Config file path](https://github.com/tier4/AWML/blob/69aba0d001fd26282880a7a3e7622b89115042de/autoware_ml/configs/detection3d/dataset/t4dataset/gen2_base.py)
  - Deployed onnx model and ROS parameter files [[WebAuto (for internal)]](https://evaluation.tier4.jp/evaluation/mlpackages/7156b453-2861-4ae9-b135-e24e48cc9029/releases/ae4e48e2-d9ed-4db0-8ee9-daf1f566e8f1?project_id=zWhWRzei)
  - Deployed onnx and ROS parameter files [[model-zoo]]
    - [detection_class_remapper.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.1.1/detection_class_remapper.param.yaml)
    - [centerpoint_ml_package.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.1.1/centerpoint_ml_package.param.yaml)
    - [deploy_metadata.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.0.1/deploy_metadata.yaml)
    - [pts_voxel_encoder_centerpoint.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.1.1/pts_voxel_encoder_centerpoint.onnx)
    - [pts_backbone_neck_head_centerpoint.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.1.1/pts_backbone_neck_head_centerpoint.onnx)
  - Training results [[Google drive (for internal)]](https://drive.google.com/drive/u/0/folders/1Kie3hE91QgjemJlv0WlK_WHO37q47Mi1)
  - Training results [model-zoo]
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.1.1/logs.zip)
    - [checkpoint_best.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.1.1/best_NuScenes_metric_T4Metric_mAP_epoch_28.pth)
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.1.1/second_secfpn_4xb16_121m_j6gen2.py)
  - Train time: NVIDIA H100 80GB * 4 * 30 epochs = 12 hours
  - Batch size: 4*16 = 64

- Evaluation
  - db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v4 + + db_largebus_v1 (total frames: 1,761):
  - Total mAP (eval range = 120m): 0.7410

| class_name | Count    | mAP    | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| -----------| -------  | ------ | ------- | ------- | ------- | ------- |
| car        |  57,839  | 85.8 | 79.7    | 86.2    | 88.5    | 88.8    |
| truck      |   4,608  | 61.7 | 49.6    | 62.2    | 65.1    | 70.0    |
| bus        |   1,559  | 84.9 | 80.4    | 84.3    | 87.5    | 87.5    |
| bicycle    |   1,057  | 70.6 | 67.3    | 71.0    | 72.1    | 72.1    |
| pedestrian |  10,375  | 67.2 | 65.6    | 66.7    | 67.5    | 69.1    |

- db_largebus_v1 (total frames: 604):
  - Total mAP (eval range = 120m): 0.75

| class_name | Count    | mAP    | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| -----------| -------  | -----  | ------- | ------- | ------- | ------- |
| car        |  13,831  | 90.4 | 85.1    | 91.2    | 92.4    | 92.6    |
| truck      |   2,137  | 70.4 | 55.9    | 72.1    | 75.3    | 78.5    |
| bus        |     95   | 79.6 | 76.1    | 80.6    | 80.9    | 80.9    |
| bicycle    |    724   | 67.5 | 63.1    | 67.9    | 69.1    | 69.9    |
| pedestrian |  3,916   | 67.0 | 65.2    | 66.6    | 67.3    | 68.9    |

- db_j6gen2_v1 + db_j6gen2_v2 +db_j6gen2_v4 (total frames: 1,157):
  - Total mAP (eval range = 120m): 0.7460

| class_name  | Count   | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----------  | ------  | ---- | ------- | ------- | ------- | ------- |
| car         | 44,008  | 84.7 | 78.2    | 84.8    | 87.2    | 88.3    |
| truck       |  2,471  | 54.7 | 44.5    | 54.4    | 56.8    | 62.9    |
| bus         |  1,464  | 85.5 | 80.8    | 84.6    | 87.9    | 88.6    |
| bicycle     |    333  | 79.1 | 78.2    | 79.4    | 79.4    | 79.4    |
| pedestrian  |  6,459  | 67.4 | 65.7    | 66.7    | 67.7    | 69.4    |

</details>

### CenterPoint J6Gen2/2.0.1
- Changes:
  - Finetune from `CenterPoint base/2.0.0`
	- Include intensity as an extra feature

- Overall:
  - Better than `CenterPoint base/v2.0.0` even when finetuning from `J6Gen2`
  - `CenterPoint J6Gen2/2.0.1` performs better overall, with improvements across most classes
  - The largest improvement is in truck detection (`+5.93 AP`)
  - Pedestrian and car detection also see solid gains.
  - Bicycle detection slightly drops in 2.0.1 but by a negligible margin (`−0.21 AP`).
  - `J6Gen2/2.0.1` is a clear upgrade over `J6Gen2/1.7.1` in terms of detection accuracy, especially for trucks and pedestrians, with only a very minor tradeoff in bicycle detection

<details>
<summary> The link of data and evaluation result </summary>

- Model
  - Training dataset: DB J6 Gen2 v1.0 + DB J6 Gen2 v2.0 + DB J6 Gen2 V4.0 + DB LargeBus v1.0 (total frames: 20,777)
  - [Config file path](https://github.com/tier4/AWML/blob/b1f498a6802f68c36a1d02b9780f72e25a413ee3/autoware_ml/configs/detection3d/dataset/t4dataset/gen2_base.py)
  - Deployed onnx model and ROS parameter files [[WebAuto (for internal)]](https://evaluation.tier4.jp/evaluation/mlpackages/7156b453-2861-4ae9-b135-e24e48cc9029/releases/2ea64514-ad8b-4943-830f-5bd570988828?project_id=zWhWRzei)
  - Deployed onnx and ROS parameter files [[model-zoo]]
    - [detection_class_remapper.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.0.1/detection_class_remapper.param.yaml)
    - [centerpoint_ml_package.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.0.1/centerpoint_ml_package.param.yaml)
    - [deploy_metadata.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.0.1/deploy_metadata.yaml)
    - [pts_voxel_encoder_centerpoint.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.0.1/pts_voxel_encoder.onnx)
    - [pts_backbone_neck_head_centerpoint.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.0.1/pts_backbone_neck_head.onnx)
  - Training results [[Google drive (for internal)]](https://drive.google.com/drive/folders/1x2LUu1hyoeroOdRtTxAPQsKLXDi2TuAc?usp=drive_link)
  - Training results [model-zoo]
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.0.1/logs.zip)
    - [checkpoint_best.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.0.1/best_NuScenes_metric_T4Metric_mAP_epoch_28.pth)
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.0.1/second_secfpn_4xb16_121m_j6gen2.py)
  - Train time: NVIDIA H100 80GB * 4 * 30 epochs = 12 hours
  - Batch size: 4*16 = 64

- Evaluation
  - db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v4 + + db_largebus_v1 (total frames: 1,761):
  - Total mAP (eval range = 120m): 0.7401

| class_name | Count    | mAP    | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| -----------| -------  | ------ | ------- | ------- | ------- | ------- |
| car        |  57,839  | 85.97  | 80.0    | 86.3    | 88.6    | 88.9    |
| truck      |   4,608  | 62.57  | 49.2    | 63.2    | 66.8    | 71.1    |
| bus        |   1,559  | 83.36  | 78.6    | 82.1    | 86.0    | 86.7    |
| bicycle    |   1,057  | 71.05  | 65.6    | 72.2    | 73.2    | 73.2    |
| pedestrian |  10,375  | 67.10  | 65.2    | 66.3    | 67.6    | 69.4    |

- db_largebus_v1 (total frames: 604):
  - Total mAP (eval range = 120m): 0.7473

| class_name | Count    | mAP    | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| -----------| -------  | -----  | ------- | ------- | ------- | ------- |
| car        |  13,831   | 90.53  | 85.4    | 91.3    | 92.6    | 92.8    |
| truck      |   2,137   | 69.55  | 54.1    | 71.0    | 75.4    | 77.6    |
| bus        |     95   | 80.06  | 76.5    | 80.4    | 81.7    | 81.7    |
| bicycle    |    724   | 65.72  | 58.8    | 66.7    | 68.7    | 68.7    |
| pedestrian |  3,916   | 67.82  | 66.1    | 67.3    | 68.0    | 69.9    |

- db_j6gen2_v1 + db_j6gen2_v2 +db_j6gen2_v4 (total frames: 1,157):
  - Total mAP (eval range = 120m): 0.7460

| class_name  | Count   | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----------  | ------  | ---- | ------- | ------- | ------- | ------- |
| car         | 44,008  | 84.83 | 78.6    | 84.9    | 87.4    | 88.4    |
| truck       |  2,471  | 56.48 | 44.7    | 56.5    | 59.1    | 65.7    |
| bus         |  1,464  | 83.70 | 78.7    | 82.2    | 87.0    | 87.0    |
| bicycle     |    333  | 81.06 | 79.3    | 81.3    | 81.8    | 81.8    |
| pedestrian  |  6,459  | 66.91 | 65.1    | 66.0    | 67.5    | 69.1    |

</details>
