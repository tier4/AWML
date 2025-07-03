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
  - Dataset: test dataset of db_j6gen2_v1 + db_largebus_v1 (total frames: 1,116)
  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m):

| eval range: 120m             | mAP  | car <br> (33,716) | truck <br> (1,583) | bus <br> (1,254) | bicycle <br> (727) | pedestrian <br> (6,789) |
| -------------------------    | ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| CenterPoint J6Gen2/1.7.1     | 74.20 | 88.00              | 61.9                | 84.7         | 71.40                 | 64.90                   |


### Datasets

<details>
<summary> LargeBus </summary>

- Test datases: db_largebus_v1 (total frames: 315)

| eval range: 120m                  | mAP     | car <br> (5,714)     | truck <br> (460) | bus <br> (51) | bicycle <br> (504) | pedestrian <br> (2,782) |
| --------------------------------  | ----    | -------------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| CenterPoint J6Gen2/1.7.1          | 77.99   | 89.17   | 67.20   | 97.36   | 69.49     | 66.75       |

</details>

<details>
<summary> J6Gen2 </summary>

- Test datases: db_j6gen2_v1 + db_j6gen2_v2 (total frames: 801)

| eval range: 120m         | mAP  | car <br> (28,002) | truck <br> (1,123) | bus <br> (1,203) | bicycle <br> (223) | pedestrian <br> (4,007) |
| -------------------------| ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| CenterPoint J6Gen2/1.7.1 | 74.64   | 87.86   | 59.87   | 84.49   | 77.35     | 63.62       |

</details>

## Release

### CenterPoint J6Gen2/1.7.1
- Changes:
  - Finetune from `CenterPoint base/1.7`
	- Include intensity as an extra feature

- Overall:
  - Better than `CenterPoint/v1.7` even when finetuning from `J6Gen2`
	- It shows consistent improvement in `LargeBus` across all classes except for a minor trade-off in bus AP, which still remains very high (`97.36`).

<details>
<summary> The link of data and evaluation result </summary>

- Model
  - Training dataset: DB J6 Gen2 v1.0 + DB J6 Gen2 v1.1 + DB J6 Gen2 v2.0 + DB LargeBus v1.0 (total frames: 13,332)
  - [Config file path](https://github.com/tier4/AWML/blob/6db4a553d15b18ac6471d228a236c014f55c8307/autoware_ml/configs/detection3d/dataset/t4dataset/base.py)
  - Deployed onnx model and ROS parameter files [[WebAuto (for internal)]](WIP)
  - Deployed onnx and ROS parameter files [[model-zoo]]
    - [detection_class_remapper.param.yaml](WIP)
    - [centerpoint_t4base_ml_package.param.yaml](WIP)
    - [deploy_metadata.yaml](WIP)
    - [pts_voxel_encoder_centerpoint_t4base.onnx](WIP)
    - [pts_backbone_neck_head_centerpoint_t4base.onnx](WIP)
  - Training results [[Google drive (for internal)]](WIP)
  - Training results [model-zoo]
    - [logs.zip](WIP)
    - [checkpoint_best.pth](WIP)
    - [config.py](WIP)
  - Train time: NVIDIA H100 80GB * 4 * 50 epochs = 9 hours
  - Batch size: 4*16 = 64

- Evaluation
  - db_j6gen2_v1 + db_largebus_v1 (total frames: 1,116):
  - Total mAP (eval range = 120m): 0.7420

| class_name | Count    | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| -----------| -------  | ----  | ------- | ------- | ------- | ------- |
| car        |  33,716  | 88.00 | 81.90  | 88.30  | 90.70  | 91.00  |
| truck      |   1,583  | 61.90 | 54.30  | 62.90  | 64.50  | 66.10  |
| bus        |   1,254  | 84.70 | 81.20  | 82.70  | 87.40  | 87.40  |
| bicycle    |     727  | 71.40 | 67.20  | 71.50  | 73.40  | 73.50  |
| pedestrian |    6,789 | 64.90 | 63.50  | 64.40  | 65.20  | 66.60  |

- db_largebus_v1 (total frames: 315):
  - Total mAP (eval range = 120m): 0.7799

| class_name | Count    | mAP    | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| -----------| -------  | -----  | ------- | ------- | ------- | ------- |
| car        |  5,714   | 82.86  | 89.80  | 91.88  | 92.15  |
| truck      |  1,123   | 56.80  | 67.54  | 71.18  | 73.30  |
| bus        |     51   | 92.53  | 98.68  | 99.12  | 99.12  |
| bicycle    |    504   | 64.71  | 69.69  | 71.75  | 71.83  |
| pedestrian |  2,782   | 64.96  | 66.14  | 67.12  | 68.79  |

- db_j6gen2_v1 + db_j6gen2_v2 (total frames: 801):
  - Total mAP (eval range = 120m): 0.7464

| class_name  | Count   | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----------  | ------  | ---- | ------- | ------- | ------- | ------- |
| car         | 28,002  | 87.86 | 81.88  | 88.18  | 90.54  | 90.87  |
| truck       |  1,123  | 59.87 | 53.38  | 60.82  | 61.93  | 63.36  |
| bus         |  1,203  | 84.49 | 81.01  | 82.43  | 87.26  | 87.28  |
| bicycle     |    223  | 77.35 | 75.77  | 77.88  | 77.88  | 77.88  |
| pedestrian  |   4,407 | 62.52  | 63.16  | 63.76  | 65.05  |

</details>
