# Deployed model for CenterPoint JPNTAXI-GEN2/2.X
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
  - Datasets (frames: 1,709):
			- jpntaxi_gen2_v1: db_jpntaxigen2_v1 (1,409 frames)
			- jpntaxi_gen2_v2: db_jpntaxigen2_v2 (230 frames)
  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m):

| eval range: 120m         | mAP     | car <br> (9,710)     | truck <br> (2,577) | bus <br> (2,569) | bicycle <br> (466) | pedestrian <br> (10,518) |
| ---------------------    | ---- | ----------------- | ------------------- | ---------------- | ----------------- | ---------------- |
| centerpoint jpntaxi_gen2/2.3.2 | 56.70 | 86.60            | 37.00               | 57.60         | 33.80                 | 68.50            |
| centerpoint jpntaxi_gen2/2.3.1 | 57.20 | 86.40            | 37.80               | 56.90         | 34.50                 | 70.20            |

### Datasets


<details>
<summary> JPNTaxi Gen2 </summary>

- Test datases: jpntaxi_gen2: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (1,709 frames)

| eval range: 120m         | mAP     | car <br> (9,710)     | truck <br> (2,577) | bus <br> (2,569) | bicycle <br> (466) | pedestrian <br> (10,518) |
| -------------------------| ----    | -------------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| centerpoint jpntaxi_gen2/2.3.2 | 56.70 | 86.60            | 37.00               | 57.60         | 33.80                 | 68.50            |
| centerpoint jpntaxi_gen2/2.3.1 | 57.20 | 86.40            | 37.80               | 56.90         | 34.50                 | 70.20            |

</details>

## Release
### CenterPoint JPNTaxi_Gen2/2.3.2
- Changes:
  - Finetune from `CenterPoint base/2.3.0` with more data in `db_jpntaxi_gen2`
	- Include intensity as an extra feature

- Overall:
  - Performance is worse than `CenterPoint base/2.3.0`, likely the training and test set still are not diverse

<details>

- Model
  - Training Datasets (frames: 22,971):
			- jpntaxi_gen2: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (22.971 frames)
  - [Config file path](https://github.com/tier4/AWML/blob/834dc15c414be9fa0851187a56d3686fa8f9e126/autoware_ml/configs/detection3d/dataset/t4dataset/jpntaxi_gen2_base.py)
  - Deployed onnx and ROS parameter files (for internal)
    - [WebAuto](https://evaluation.tier4.jp/evaluation/mlpackages/7156b453-2861-4ae9-b135-e24e48cc9029/releases/e2241c36-646b-4893-8fde-486c32375274?project_id=zWhWRzei)
    - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/jpntaxi_gen2/v2.3.2/deployment.zip)
    - [Google drive](https://drive.google.com/drive/u/0/folders/1hD__tIVcQ6UalgJi49jSvRZFgNxERGOA)
  - Logs (for internal)
    - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/jpntaxi_gen2/v2.3.2/logs.zip)
    - [Google drive](https://drive.google.com/file/d/18h0LVXbY919hlk35UOa8MedRqjUplLUt/view?usp=drive_link)
  - Train time: NVIDIA H100 80GB * 4 * 30 epochs = 24 hours
  - Batch size: 4*16 = 64

- Evaluation

- jpntaxi_gen2: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (1,709 frames)
  - Total mAP (eval range = 120m): 0.5620
| class_name | Count      | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | ---------- | ---- | ---- | ---- | ---- | ---- |
| car        |  9,710     | 86.6 | 81.5    | 87.4    | 88.6    | 88.9    |
| truck      |  2,577     | 37.0 | 25.9    | 34.3    | 37.7    | 50.2    |
| bus        |  2,569     | 57.6 | 43.1    | 56.7    | 64.4    | 66.0    |
| bicycle    |    466     | 33.8 | 29.0    | 34.7    | 35.8    | 35.9    |
| pedestrian |  10,518    | 68.5 | 66.8    | 67.9    | 68.9    | 70.4    |

</details>


### CenterPoint JPNTaxi_Gen2/2.3.1
- Changes:
  - Finetune from `CenterPoint base/2.3.0` with `db_jpntaxi_gen2` datasets only
	- Include intensity as an extra feature

- Overall:
  - Performance is worse than `CenterPoint base/2.3.0`, likely the training and test set are not diverse

<details>

- Model
  - Training Datasets (frames: 18,630):
			- jpntaxi_gen2: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (18,630 frames)
  - [Config file path](https://github.com/tier4/AWML/blob/c0ba7268f110062f71ee80a3469102867a63b740/projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_jpntaxi_gen2_base.py)
  - Deployed onnx and ROS parameter files (for internal)
    - [WebAuto](https://evaluation.tier4.jp/evaluation/mlpackages/7156b453-2861-4ae9-b135-e24e48cc9029/releases/0e6b01fe-3f66-4bb3-be5c-4a4ded446191?project_id=zWhWRzei)
    - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/jpntaxi_gen2/v2.3.1/deployment.zip)
    - [Google drive](https://drive.google.com/file/d/1wuZ_HKMMUntCDhrUVRGe6BC0_GJvoppK/view?usp=drive_link)
  - Logs (for internal)
    - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/jpntaxi_gen2/v2.3.1/logs.zip)
    - [Google drive](https://drive.google.com/file/d/18Q5FBb2duFpE3zNABPMyt2ecqjM3rkrF/view?usp=drive_link)
  - Train time: NVIDIA H100 80GB * 4 * 30 epochs = 12 hours
  - Batch size: 4*16 = 64

- Evaluation

- jpntaxi_gen2: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (1,709 frames)
  - Total mAP (eval range = 120m): 0.5720

| class_name | Count      | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | ---------- | ---- | ---- | ---- | ---- | ---- |
| car        |  9,710     | 86.2 | 81.1    | 87.1    | 88.2    | 88.4    |
| truck      |  2,577     | 38.1 | 28.5    | 35.1    | 38.9    | 50.1    |
| bus        |  2,569     | 56.9 | 41.2    | 56.8    | 64.1    | 65.5    |
| bicycle    |    466     | 34.0 | 28.9    | 35.1    | 35.9    | 36.3    |
| pedestrian |  10,518    | 70.7 | 69.4    | 70.3    | 71.1    | 72.1    |

</details>

### CenterPoint JPNTaxi_Gen2/2.2.1
- Changes:
  - Finetune from `CenterPoint base/2.2.0` with `db_jpntaxi_gen2` datasets only
	- Include intensity as an extra feature

- Overall:
  - Performance is worse than `CenterPoint base/2.2.0`, likely the training and test set are not diverse, and it's overffited to the training set

<details>

- Model
  - Training dataset: DB JPNTAXI Gen2 v1.0 (total frames: 12,799)
  - [Config file path](https://github.com/tier4/AWML/blob/81314d29d4efa560952324c48ef7c0ea1e56f1ee/projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_jpntaxi_gen2_base.py)
  - Deployed onnx and ROS parameter files (for internal)
    - [WebAuto](https://evaluation.tier4.jp/evaluation/mlpackages/7156b453-2861-4ae9-b135-e24e48cc9029/releases/7e552a0a-7c31-49f7-85c9-784531172077?project_id=zWhWRzei)
    - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/jpntaxi_gen2/v2.2.1/deployment.zip)
    - [Google drive](https://drive.google.com/file/d/1AzWPmXPEsH2HAMLrHIaoTCzhqaf6hDW5/view?usp=drive_link)
  - Logs (for internal)
    - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/jpntaxi_gen2/v2.2.1/logs.zip)
    - [Google drive](https://drive.google.com/file/d/1MXES2Dbi_ab8VcTRRODFrEzaZaiDRwt5/view?usp=drive_link)
  - Train time: NVIDIA H100 80GB * 4 * 30 epochs = 12 hours
  - Batch size: 4*16 = 64

- Evaluation

- db_jpntaxi_gen2_v1 (total frames: 1,479):
  - Total mAP (eval range = 120m): 0.5090

| class_name | Count | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | ------| ---- | ----    | ---- | ---- | ---- |
| car        | 8,571 | 84.2 | 79.6    | 84.4    | 86.2    | 86.4    |
| truck      | 3,349 | 35.6 | 24.4    | 30.5    | 37.7    | 49.7    |
| bus        | 4,148 | 53.7 | 36.3    | 50.6    | 63.1    | 64.6    |
| bicycle    |   310 | 12.0 | 12.0    | 12.0    | 12.0    | 12.2    |
| pedestrian | 8,665 | 68.9 | 68.0    | 68.5    | 69.1    | 69.9    |

</details>
