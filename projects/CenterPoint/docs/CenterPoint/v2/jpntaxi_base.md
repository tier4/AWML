
# Deployed model for CenterPoint JPNTaxi-Base/2.X
## Summary
The main difference between `JPNTaxi-Base/2.x` and `JPNTaxi-Gen2/2.x` is that all `JPNTaxi` data from Gen1 and Gen2 are used for training/validation/testing in `JPNTaxi-Base/2.x` to reduce overfitting since they share similar
vehicle and sensor setups.

### Overview
- Main parameter
  - range = 122.40m
  - voxel_size = [0.24, 0.24, 8.0]
  - grid_size = [510, 510, 1]
	- **With Intensity**
- Detailed comparison
- Performance summary
  - Datasets (frames: 3,216):
			- jpntaxi_gen2: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (1,709 frames)
			- jpntaxi: db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 (1,507 frames)
			  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m):

| eval range: 120m                   | mAP     | car <br> (25,852) | truck <br> (7,155) | bus <br> (4,026) | bicycle <br> (1,506) | pedestrian <br> (22,489) |
| -------------------------          | ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| CenterPoint JPNTaxi_Base/2.5.1     | 66.60 | 81.50            | 49.90               | 67.90         | 62.20                 | 71.50                   |
| CenterPoint JPNTaxi_Gen2/2.5.1     | 59.10 | 73.40            | 42.90               | 59.10         | 51.50                 | 68.70                   |
| CenterPoint JPNTaxi_Gen2/2.4.1     | 59.10 | 73.40            | 42.90               | 59.10         | 51.50                 | 68.70                   |

### Datasets

<details>
<summary> JPNTaxi Gen2 </summary>

- Test datases: jpntaxi_gen2: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (1,709 frames)

| eval range: 120m         | mAP     | car <br> (9,710)     | truck <br> (2,577) | bus <br> (2,569) | bicycle <br> (466) | pedestrian <br> (10,518) |
| -------------------------| ----    | -------------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| CenterPoint JPNTaxi_Base/2.5.1   | 63.00 | 88.70            | 42.70               | 64.50         | 45.60                 | 73.30                   |
| CenterPoint JPNTaxi_Gen2/2.5.1   | 60.60 | 87.90            | 40.00               | 62.20         | 41.50                 | 71.60                   |
| CenterPoint JPNTaxi_Gen2/2.4.1   | 58.20 | 86.10            | 38.00               | 56.10         | 39.20                 | 71.70                   |

</details>

<summary> JPNTaxi Gen2_V2 </summary>

- Test datases: db_jpntaxigen2_v2 (total frames: 230)

| eval range: 120m                 | mAP     | car <br> (3,449)     | truck <br> (726) | bus <br> (251) | bicycle <br> (157) | pedestrian <br> (2,42) |
| ---------------------            | ----    | ----------------- | ------------------- | ---------------- | ----------------- | ---------------- |
| CenterPoint JPNTaxi_Base/2.5.1   | 73.33 | 90.00            | 53.80               | 85.40         | 63.60                 | 73.80                   |
| CenterPoint JPNTaxi_Gen2/2.5.1   | 72.20 | 89.50            | 51.60               | 86.00         | 60.10                 | 73.60                   |
| CenterPoint JPNTaxi_Gen2/2.4.1   | 71.10 | 88.10            | 52.80               | 86.70         | 57.50                 | 70.70                   |

</details>

<details>
<summary> JPNTaxi Base </summary>

- Test datases: jpntaxi_gen2: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 + db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 (total frames: 3,216)

| eval range: 120m           | mAP     | car <br> (25,852) | truck <br> (7,155) | bus <br> (4.026) | bicycle <br> (1,506) | pedestrian <br> (22,489) |
| -------------------------  | ----    | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| CenterPoint JPNTaxi_Base/2.5.1   | 66.66 | 81.50            | 49.90               | 67.90         | 62.20                 | 71.50                   |
| CenterPoint JPNTaxi_Gen2/2.4.1   | 59.10 | 73.40            | 42.90               | 59.10         | 51.50                 | 68.70                   |


</details>

## Release
### CenterPoint JPNTaxi_Base/2.5.1
- Changes:
  - Finetune from `CenterPoint base/2.5.0` with `db_jpntaxi_base` datasets, where it includes both JPNTaxi Gen1 and Gen2 data
	- Include intensity as an extra feature and Repeat Sampling Factor (RFS)

- Overall:
  - Performance is better than `CenterPoint base/2.4.0` in `JPNTaxi Gen2` (`60.70`), it's also better than `JPNTaxi_Gen2/2.5.1` in both `JPNTaxi Gen2` and `JPNTaxi Gen2_v2`

<details>
<summary> The link of data and evaluation result </summary>

#### Model
  - Training Datasets (frames: 50,950):
    - jpntaxi: db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 (25,958 frames)
		- jpntaxi_gen2: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (24,992 frames)
  - [Config file path](https://github.com/tier4/AWML/blob/dee55764f5381ef75dcac7a17a303b0bf527d400/projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_jpntaxi_base_amp_rfs.py)
  - Deployed onnx and ROS parameter files (for internal)
    - [WebAuto](https://evaluation.tier4.jp/evaluation/mlpackages/7156b453-2861-4ae9-b135-e24e48cc9029/releases/41e733b4-a778-4281-ba5b-6ee9ab7c89dc?project_id=zWhWRzei)
    - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/jpntaxi_base/v2.5.1/deployment.zip)
    - [Google drive](https://drive.google.com/drive/u/0/folders/1hD__tIVcQ6UalgJi49jSvRZFgNxERGOA)
  - Logs (for internal)
    - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/jpntaxi_base/v2.5.1/logs.zip)
    - [Google drive](https://drive.google.com/file/d/18h0LVXbY919hlk35UOa8MedRqjUplLUt/view?usp=drive_link)
  - Train time: NVIDIA H100 80GB * 4 * 30 epochs = 1 day and 20 hours
  - Batch size: 4*16 = 64

#### Evaluation Summary

##### JPNTaxi Gen2
**Datasets:** db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (1,709 frames)  
**Total mAP (120 m range):** **0.63**

| class_name | Count  | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
|-----------|--------|------|---------|---------|---------|---------|
| car        | 9,710  | 88.7 | 82.9 | 89.6 | 90.7 | 91.4 |
| truck      | 2,577  | 42.7 | 31.5 | 39.6 | 43.7 | 56.0 |
| bus        | 2,569  | 64.5 | 46.9 | 63.5 | 73.1 | 74.6 |
| bicycle    |   466  | 45.6 | 36.1 | 48.4 | 48.8 | 49.1 |
| pedestrian | 10,518 | 73.3 | 71.3 | 72.5 | 74.0 | 75.4 |

---

##### JPNTaxi Gen2_V2
**Dataset:** db_jpntaxigen2_v2 (230 frames)  
**Total mAP (120 m range):** **0.733**

| class_name | Count | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
|-----------|--------|------|---------|---------|---------|---------|
| car        | 3,449 | 90.0 | 82.8 | 91.5 | 92.8 | 93.1 |
| truck      |   726 | 53.8 | 44.7 | 53.7 | 55.7 | 60.9 |
| bus        |   251 | 85.4 | 79.8 | 86.7 | 87.3 | 87.8 |
| bicycle    |   157 | 63.6 | 49.6 | 67.5 | 68.4 | 69.0 |
| pedestrian | 2,443 | 73.8 | 72.0 | 73.5 | 74.3 | 75.2 |

---

##### JPNTaxi Base
**Dataset:** db_jpntaxi_base (3,216 frames)  

**Total mAP (120 m range):** **0.666**

| class_name | Count  | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
|-----------|---------|------|---------|---------|---------|---------|
| car        | 25,882 | 81.5 | 71.5 | 83.4 | 85.4 | 85.9 |
| truck      | 7,155  | 49.9 | 33.6 | 48.9 | 54.9 | 62.2 |
| bus        | 4,026  | 67.9 | 53.2 | 67.7 | 74.5 | 76.2 |
| bicycle    | 1,506  | 62.2 | 55.2 | 64.3 | 64.4 | 65.1 |
| pedestrian | 22,489 | 71.5 | 68.7 | 70.7 | 72.5 | 74.2 |

</details>

---
