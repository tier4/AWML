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
  - Datasets (frames: 7,727):
			- jpntaxi_gen2: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (1,709 frames)
  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m):

| eval range: 120m         | mAP     | car <br> (9,710)     | truck <br> (2,577) | bus <br> (2,569) | bicycle <br> (466) | pedestrian <br> (10,518) |
| ---------------------    | ---- | ----------------- | ------------------- | ---------------- | ----------------- | ---------------- |
| CenterPoint JPNTaxi_Gen2/2.3.1 | 57.20 | 86.20            | 38.10               | 56.90         | 34.00                 | 70.70            |
| CenterPoint JPNTaxi_Gen2/2.2.1 | 52.60 | 83.30            | 37.00               | 55.30         | 21.30                 | 66.10            |

### Datasets


<details>
<summary> JPNTaxi Gen2 </summary>

- Test datases: jpntaxi_gen2: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (1,709 frames)

| eval range: 120m         | mAP     | car <br> (9,710)     | truck <br> (2,577) | bus <br> (2,569) | bicycle <br> (466) | pedestrian <br> (10,518) |
| -------------------------| ----    | -------------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| CenterPoint JPNTaxi_Gen2/2.3.1 | 57.20 | 86.20            | 38.10               | 56.90         | 34.00                 | 70.70            |
| CenterPoint JPNTaxi_Gen2/2.2.1 | 52.60 | 83.30            | 37.00               | 55.30         | 21.30                 | 66.10            |

</details>

## Release
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
