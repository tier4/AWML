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
  - Dataset: test dataset of db_jpntaxi_gen2_v1 (total frames: 1,479)
  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m):

| eval range: 120m         | mAP     | car <br> (8,571)     | truck <br> (3,349) | bus <br> (4,148) | bicycle <br> (310) | pedestrian <br> (8,665) |
| ---------------------    | ---- | ----------------- | ------------------- | ---------------- | ----------------- | ---------------- |
| CenterPoint JPNTaxi_Gen2/2.2.1 | 50.90 | 84.20            | 35.60               | 53.70         | 12.00                 | 68.90            |

### Datasets


<details>
<summary> JPNTaxi Gen2 </summary>

- Test datases: db_jpntaxi_gen2_v1 (total frames: 1,479)

| eval range: 120m         | mAP     | car <br> (8,571)     | truck <br> (3,349) | bus <br> (4,148) | bicycle <br> (310) | pedestrian <br> (8,665) |
| -------------------------| ----    | -------------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| CenterPoint JPNTaxi_Gen2/2.2.1     | 50.90   | 84.20   | 35.60   | 53.70 | 12.00       | 68.90       |

</details>

## Release

### CenterPoint JPNTaxi_Gen2/2.2.1
- Changes:
  - Finetune from `CenterPoint base/2.2.0` with `db_jpntaxi_gen2` datasets only
	- Include intensity as an extra feature

- Overall:
  - Performance is worse than `CenterPoint base/2.2.0`, likely the training and test set are not diverse, and it's overffited to the training set


<details>
<summary> The link of data and evaluation result </summary>

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
