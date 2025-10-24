# Deployed model for CenterPoint J6Gen2/1.X
## Summary

### Overview
- Main parameter
  - range = 121.60m
  - voxel_size = [0.32, 0.32, 8.0]
  - grid_size = [760, 760, 1]
	- **With Intensity**
- Detailed comparison
  - [Internal Link](https://docs.google.com/spreadsheets/d/13Stt9hdbTER6ugaRMEZscbt_6kD7ld-0b30JyeTGTrs/edit?gid=1466166341#gid=1466166341)
- Performance summary
  - Datasets (frames: 2,086):
			- j6gen2: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 (1,217 frames)
			- largebus: db_largebus_v1 + db_largebus_v2 (859 frames)

  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m):

| eval range: 120m         | mAP     | car <br> (56,829) | truck <br> (4,633) | bus <br> (1,559) | bicycle <br> (1,166) | pedestrian <br> (10,955) |
| -------------------------    | ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| CenterPoint J6Gen2/2.4.1     | 76.10 | 85.60            | 62.20               | 85.20         | 76.50                 | 70.80                   |
| CenterPoint J6Gen2/2.3.1     | 74.20 | 86.50            | 59.70               | 84.70         | 71.40                 | 68.40                   |


### Deprecated results

<details>
- Performance summary

| eval range: 120m         | mAP  | car <br> (57,839) | truck <br> (4,608) | bus <br> (1,559) | bicycle <br> (1,057) | pedestrian <br> (10,375) |
| ---------------------    | ---- | ----------------- | ------------------- | ---------------- | ----------------- | ---------------- |
| CenterPoint J6Gen2/2.2.1 | 74.50 | 85.80            | 63.00               | 84.90         | 71.10                 | 67.70            |
| CenterPoint J6Gen2/2.1.1 | 74.07 | 85.80            | 61.74               | 84.94         | 70.63                 | 67.24            |
| CenterPoint J6Gen2/2.0.1 | 74.01 | 85.97            | 62.57               | 83.36         | 71.05                 | 67.10            |
| CenterPoint J6Gen2/1.7.1 | 71.77 | 84.80            | 56.64               | 81.78         | 71.26                 | 64.38            |
</details>

### Datasets

<details>
<summary> LargeBus </summary>

- Test datases: db_largebus_v1 + db_largebus_v2 (total frames: 859)

| eval range: 120m         | mAP     | car <br> (16,604)     | truck <br> (1,961) | bus <br> (171) | bicycle <br> (863) | pedestrian <br> (4,659) |
| -------------------------| ----    | -------------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| CenterPoint J6Gen2/2.4.1     | 76.50 | 90.80            | 69.10               | 77.90         | 73.40                 | 71.10                   |
| CenterPoint J6Gen2/2.3.1     | 74.80 | 91.20            | 69.70               | 75.40         | 67.70                 | 69.80                   |

</details>

<details>
<summary> J6Gen2 </summary>

- Test datases: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5  (total frames: 1,217)

| eval range: 120m         | mAP     | car <br> (40,225) | truck <br> (2,672) | bus <br> (1,388) | bicycle <br> (303) | pedestrian <br> (6,296) |
| -------------------------| ----    | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| CenterPoint J6Gen2/2.4.1        | 76.70 | 83.50            | 57.20               | 86.10         | 86.10                 | 70.40                   |
| CenterPoint J6Gen2/2.3.1    | 74.30 | 84.50            | 52.60               | 85.70         | 81.60                 | 67.20                   |


</details>

## Release
### CenterPoint J6Gen2/2.4.1
- Changes:
  - Finetune from `CenterPoint base/2.4.0` with j6gen2 base dataset
  - Include intensity as an extra feature and Repeat Sampling Factor (RFS)

- Overall:
   - Performance is better than `CenterPoint J6Gen2/2.3.1`, especially, `truck`, `bicycle` and `pedestrian`, where `bicycle` improves mAP more than `5.0` in `j6gen2_base` dataset

<details>
<summary> The link of data and evaluation result </summary>

- Model
  - Training Dataset (frames: 30,290):
      - j6gen2: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 (21,077 frames)
			- largebus: db_largebus_v1 + db_largebus_v2 (9,213 frames)
  - [Config file path](https://github.com/tier4/AWML/blob/c0ba7268f110062f71ee80a3469102867a63b740/projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_j6gen2_base.py)
  - Deployed onnx and ROS parameter files (for internal)
    - [WebAuto](https://evaluation.tier4.jp/evaluation/mlpackages/7156b453-2861-4ae9-b135-e24e48cc9029/releases/cb7d790e-4efe-47c2-b2b4-62d9d80aa085?project_id=zWhWRzei)
    - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.3.1/deployment.zip)
    - [Google drive](https://drive.google.com/file/d/1toIlwTYbjIkXVoRdG4e0WSUBwqRKQDNi/view?usp=drive_link)
  - Logs (for internal)
    - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.3.1/logs.zip)
    - [Google drive](https://drive.google.com/file/d/1i4uvbOHsbDuNn0CtF1Dkx4CTrrjCAICt/view?usp=drive_link)
  - Train time: NVIDIA H100 80GB * 4 * 30 epochs = 1 day 3 hours
  - Batch size: 4*16 = 64

- Evaluation

- db_largebus_v1 + db_largebus_v2 (859 frames):
  - Total mAP (eval range = 120m): 0.765

| class_name | Count        | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | -------------| ---- | ---- | ---- | ---- | ---- |
| car        | 16,604       | 90.8 | 85.9    | 91.4    | 92.8    | 93.1    |
| truck      |  1,961       | 69.1 | 56.2    | 69.3    | 74.6    | 76.5    |
| bus        |    171       | 77.9 | 63.0    | 82.8    | 82.9    | 82.9    |
| bicycle    |    863       | 73.4 | 69.4    | 73.9    | 75.1    | 75.3    |
| pedestrian |   4,659      | 71.1 | 69.7    | 70.7    | 71.3    | 72.7    |

- j6gen2: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 (1,217 frames):
  - Total mAP (eval range = 120m): 0.767

| class_name | Count      | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | ---------- | ---- | ---- | ---- | ---- | ---- |
| car        | 40,225     | 83.5 | 77.5    | 83.6    | 85.9    | 86.9    |
| truck      |  2,672     | 57.2 | 46.9    | 56.1    | 59.2    | 66.6    |
| bus        |  1,388     | 86.1 | 81.3    | 85.2    | 88.6    | 89.3    |
| bicycle    |    303     | 86.1 | 85.6    | 86.3    | 86.3    | 86.3    |
| pedestrian |  6,296     | 70.4 | 69.2    | 70.0    | 70.8    | 71.8    |

</details>

### CenterPoint J6Gen2/2.3.1
- Changes:
  - Finetune from `CenterPoint base/2.3.0` with j6gen2 base dataset
  - Include intensity as an extra feature

- Overall:
   - Performance is slightly better than `CenterPoint J6Gen2/2.2.1`

<details>
<summary> The link of data and evaluation result </summary>

- Model
  - Training Dataset (frames: 30,290):
      - j6gen2: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 (21,077 frames)
			- largebus: db_largebus_v1 + db_largebus_v2 (9,213 frames)
  - [Config file path](https://github.com/tier4/AWML/blob/c0ba7268f110062f71ee80a3469102867a63b740/projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_j6gen2_base.py)
  - Deployed onnx and ROS parameter files (for internal)
    - [WebAuto](https://evaluation.tier4.jp/evaluation/mlpackages/7156b453-2861-4ae9-b135-e24e48cc9029/releases/cb7d790e-4efe-47c2-b2b4-62d9d80aa085?project_id=zWhWRzei)
    - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.3.1/deployment.zip)
    - [Google drive](https://drive.google.com/file/d/1toIlwTYbjIkXVoRdG4e0WSUBwqRKQDNi/view?usp=drive_link)
  - Logs (for internal)
    - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.3.1/logs.zip)
    - [Google drive](https://drive.google.com/file/d/1i4uvbOHsbDuNn0CtF1Dkx4CTrrjCAICt/view?usp=drive_link)
  - Train time: NVIDIA H100 80GB * 4 * 30 epochs = 12 hours
  - Batch size: 4*16 = 64

- Evaluation

- db_largebus_v1 + db_largebus_v2 (859 frames):
  - Total mAP (eval range = 120m): 0.748

| class_name | Count        | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | -------------| ---- | ---- | ---- | ---- | ---- |
| car        | 16,604       | 91.2 | 86.0    | 92.0    | 93.2    | 93.5    |
| truck      |  1,961       | 69.7 | 57.1    | 70.3    | 74.9    | 76.5    |
| bus        |    171       | 75.4 | 59.6    | 80.7    | 80.7    | 80.7    |
| bicycle    |    863       | 67.7 | 62.0    | 68.9    | 69.9    | 70.0    |
| pedestrian |   4,659      | 69.8 | 68.1    | 69.2    | 70.1    | 71.8    |

- j6gen2: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 (1,217 frames):
  - Total mAP (eval range = 120m): 0.743

| class_name | Count      | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | ---------- | ---- | ---- | ---- | ---- | ---- |
| car        | 40,225     | 84.5 | 78.4    | 84.9    | 87.2    | 87.5    |
| truck      |  2,672     | 52.6 | 42.1    | 50.8    | 55.1    | 62.5    |
| bus        |  1,388     | 85.7 | 81.9    | 84.2    | 88.0    | 88.7    |
| bicycle    |    303     | 81.6 | 79.9    | 81.8    | 82.3    | 82.3    |
| pedestrian |  6,296     | 67.2 | 65.3    | 66.4    | 67.6    | 69.3    |

</details>

### CenterPoint J6Gen2/2.2.1
- Changes:
  - Finetune from `CenterPoint base/2.2.0` with j6gen2 base dataset
  - Include intensity as an extra feature

  - Overall:
   - Performance is slightly better than `CenterPoint J6Gen2/2.1.1`

<details>
<summary> The link of data and evaluation result </summary>

- Model
  - Training dataset: DB J6 Gen2 v1.0 + DB J6 Gen2 v2.0 + DB J6 Gen2 V4.0 + DB LargeBus v1.0 (total frames: 20,777)
  - [Config file path](https://github.com/tier4/AWML/blob/81314d29d4efa560952324c48ef7c0ea1e56f1ee/projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_j6gen2_base.py)
  - Deployed onnx and ROS parameter files (for internal)
    - [WebAuto](https://evaluation.tier4.jp/evaluation/mlpackages/7156b453-2861-4ae9-b135-e24e48cc9029/releases/115b1a13-84be-4dde-a9d2-8293e2be36ba?project_id=zWhWRzei)
    - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.2.1/deployment.zip)
    - [Google drive](https://drive.google.com/file/d/1AXlkBB1aG7h0kzk5NeU2OIPlRZS2YfwQ/view?usp=drive_link)
  - Logs (for internal)
    - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.2.1/logs.zip)
    - [Google drive](https://drive.google.com/file/d/1i09BfY6LcVjSb4pYHL7IVIRP4Eus2Joo/view?usp=drive_link)
  - Train time: NVIDIA H100 80GB * 4 * 30 epochs = 12 hours
  - Batch size: 4*16 = 64

- Evaluation

- db_largebus_v1 (total frames: 604):
  - Total mAP (eval range = 120m): 0.7467

| class_name | Count    | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | -------  | ---- | ---- | ---- | ---- | ---- |
| car        |  13,831  | 90.2 | 85.1    | 91.1    | 92.3    | 92.5    |
| truck      |  2,137   | 70.6 | 55.1    | 72.7    | 76.1    | 78.4    |
| bus        |     95   | 78.2 | 74.0    | 78.1    | 80.3    | 80.3    |
| bicycle    |    724   | 66.5 | 59.3    | 68.1    | 69.4    | 69.4    |
| pedestrian |  3,916   | 67.8 | 66.1    | 67.3    | 68.2    | 69.6    |

- db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v2 (total frames: 1,157):
  - Total mAP (eval range = 120m): 0.7524

| class_name | Count   | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | ------  | ---- | ---- | ---- | ---- | ---- |
| car        | 44,008  | 84.7 | 78.4    | 85.0    | 87.3    | 88.3    |
| truck      |  2,471  | 57.4 | 46.6    | 57.6    | 60.0    | 65.4    |
| bus        |  1,464  | 85.5 | 81.8    | 84.5    | 87.8    | 87.9    |
| bicycle    |    333  | 80.7 | 80.4    | 80.8    | 80.8    | 80.8    |
| pedestrian |  6,459  | 67.8 | 66.4    | 67.1    | 68.0    | 69.7    |

</details>

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
