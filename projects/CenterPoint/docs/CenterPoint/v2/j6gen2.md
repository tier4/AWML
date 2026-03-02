# Deployed model for CenterPoint J6Gen2/2.X
## Summary

### Main Parameters

  - **Range:** 122.40m
  - **Voxel Size:** [0.24, 0.24, 8.0]
  - **Grid Size:** [510, 510, 1]
  - **With Intensity**

### Testing Datsets

- **Total Frames: 5,179**
  <details>
  <summary> j6gen2 (3,951 frames) </summary>
    - `db_j6gen2_v1`
    - `db_j6gen2_v2`
    - `db_j6gen2_v3`
    - `db_j6gen2_v4`
    - `db_j6gen2_v5`
    - `db_j6gen2_v6`
    - `db_j6gen2_v7`
    - `db_j6gen2_v8`
    - `db_j6gen2_v9`
  </details>

  <details>
  <summary> largebus (1,228 frames) </summary>
    - `db_largebus_v1`
    - `db_largebus_v2`
    - `db_largebus_v3`
  </details>

### mAP
- **Class mAP for BEV Center Distance: 0.5m, 1.0m, 2.0m, 4.0m**

	<details>
  <summary> Eval Range: 0.0 - 50.0m </summary>

  | Model version | mAP  | car <br> (64,520) | truck <br> (6,947) | bus <br> (2,275) | bicycle <br> (1,379) | pedestrian <br> (19,421) |
  | -------------------------| ---- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
  | CenterPoint J6Gen2/2.6.1     | 84.65 | 94.73            | 81.93            | 91.96            | 80.16            | 74.45            |
  | CenterPoint J6Gen2/2.5.1     | 84.20 | 94.24            | 80.77            | 92.59            | 80.16            | 73.25            |

  </details>

	<details>
  <summary> Eval Range: 50.0 - 90.0m </summary>

  | Model version | mAP  | car <br> (58,562) | truck <br> (5,101) | bus <br> (2,078) | bicycle <br> (758) | pedestrian <br> (10,283) |
  | -------------------------| ---- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
  | CenterPoint J6Gen2/2.6.1     | 65.60 | 81.03            | 61.17            | 75.08            | 52.45            | 58.27            |
  | CenterPoint J6Gen2/2.5.1     | 63.43 | 79.61            | 57.96            | 69.85            | 52.76            | 56.99            |

  </details>

	<details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version | mAP  | car <br> (20,371) | truck <br> (3,172) | bus <br> (376) | bicycle <br> (155) | pedestrian <br> (2,794) |
  | -------------------------| ---- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
  | CenterPoint J6Gen2/2.6.1     | 41.52 | 58.37            | 42.35            | 35.37            | 34.32            | 37.19            |
  | CenterPoint J6Gen2/2.5.1     | 38.55 | 55.21            | 40.77            | 27.18            | 33.56            | 36.02            |

  </details>

	<details open>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version | mAP  | car <br> (143,453) | truck <br> (15,220) | bus <br> (4,729) | bicycle <br> (2,292) | pedestrian <br> (32,498) |
  | -------------------------| ---- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
  | CenterPoint J6Gen2/2.6.1     | 74.26 | 85.33            | 68.00            | 81.44            | 69.07            | 67.45            |
  | CenterPoint J6Gen2/2.5.1     | 72.88 | 84.14            | 66.10            | 78.57            | 69.49            | 66.08            |

  </details>

## Datasets

<details>
<summary> LargeBus </summary>

- Datasets (1,228 Testing Frames):
  - `db_largebus_v1`
  - `db_largebus_v2`
  - `db_largebus_v3`

- **Class mAP for BEV Center Distance: 0.5m, 1.0m, 2.0m, 4.0m**

	<details>
  <summary> Eval Range: 0.0 - 50.0m </summary>

  | Model version | mAP  | car <br> (14,883) | truck <br> (1,193) | bus <br> (336) | bicycle <br> (740) | pedestrian <br> (5,059) |
  | -------------------------| ---- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
  | CenterPoint J6Gen2/2.6.1     | 84.00 | 95.89            | 84.14            | 88.03            | 76.93            | 75.03            |
  | CenterPoint J6Gen2/2.5.1     | 83.98 | 95.41            | 83.68            | 90.68            | 76.40            | 73.73            |

  </details>

	<details>
  <summary> Eval Range: 50.0 - 90.0m </summary>

  | Model version | mAP  | car <br> (10,994) | truck <br> (1,011) | bus <br> (143) | bicycle <br> (463) | pedestrian <br> (3,754) |
  | -------------------------| ---- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
  | CenterPoint J6Gen2/2.6.1     | 70.84 | 87.88            | 67.53            | 88.29            | 49.08            | 61.44            |
  | CenterPoint J6Gen2/2.5.1     | 70.47 | 86.57            | 65.90            | 90.25            | 49.54            | 60.10            |

  </details>

	<details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version | mAP  | car <br> (3,018) | truck <br> (602) | bus <br> (60) | bicycle <br> (85) | pedestrian <br> (1,121) |
  | -------------------------| ---- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
  | CenterPoint J6Gen2/2.6.1     | 49.39 | 67.61            | 58.18            | 49.30            | 27.99            | 43.86            |
  | CenterPoint J6Gen2/2.5.1     | 48.52 | 64.05            | 60.15            | 49.12            | 25.80            | 43.50            |

  </details>

	<details>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version | mAP  | car <br> (28,895) | truck <br> (2,806) | bus <br> (539) | bicycle <br> (1,288) | pedestrian <br> (9,934) |
  | -------------------------| ---- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
  | CenterPoint J6Gen2/2.6.1     | 76.01 | 90.83            | 73.45            | 83.80            | 64.63            | 67.36            |
  | CenterPoint J6Gen2/2.5.1     | 75.93 | 89.72            | 72.99            | 86.03            | 64.78            | 66.13            |

  </details>


</details>

<details>
<summary> J6Gen2 </summary>

- Datasets (3,951 Testing Frames):
  - `db_j6gen2_v1`
  - `db_j6gen2_v2`
  - `db_j6gen2_v3`
  - `db_j6gen2_v4`
  - `db_j6gen2_v5`
  - `db_j6gen2_v6`
  - `db_j6gen2_v7`
  - `db_j6gen2_v8`
  - `db_j6gen2_v9`

- **Class mAP for BEV Center Distance: 0.5m, 1.0m, 2.0m, 4.0m**

	<details>
  <summary> Eval Range: 0.0 - 50.0m </summary>

  | Model version | mAP  | car <br> (49,637) | truck <br> (5,754) | bus <br> (1,939) | bicycle <br> (639) | pedestrian <br> (14,362) |
  | -------------------------| ---- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
  | CenterPoint J6Gen2/2.6.1     | 85.34 | 94.26            | 81.40            | 92.57            | 84.10            | 74.38            |
  | CenterPoint J6Gen2/2.5.1     | 84.91 | 93.92            | 80.21            | 92.98            | 84.35            | 73.10            |

  </details>

	<details>
  <summary> Eval Range: 50.0 - 90.0m </summary>

  | Model version | mAP  | car <br> (47,568) | truck <br> (4,090) | bus <br> (1,935) | bicycle <br> (295) | pedestrian <br> (6,529) |
  | -------------------------| ---- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
  | CenterPoint J6Gen2/2.6.1     | 65.73 | 79.41            | 59.58            | 74.45            | 58.90            | 56.30            |
  | CenterPoint J6Gen2/2.5.1     | 63.46 | 77.99            | 55.96            | 68.73            | 59.46            | 55.15            |

  </details>

	<details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version | mAP  | car <br> (17,353) | truck <br> (2,570) | bus <br> (316) | bicycle <br> (70) | pedestrian <br> (1,673) |
  | -------------------------| ---- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
  | CenterPoint J6Gen2/2.6.1     | 40.75 | 56.78            | 38.43            | 32.79            | 43.00            | 32.77            |
  | CenterPoint J6Gen2/2.5.1     | 37.86 | 53.67            | 36.03            | 22.95            | 45.28            | 31.39            |

  </details>

	<details>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version | mAP  | car <br> (114,558) | truck <br> (12,414) | bus <br> (4,190) | bicycle <br> (1,004) | pedestrian <br> (22,564) |
  | -------------------------| ---- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
  | CenterPoint J6Gen2/2.6.1     | 74.80 | 83.83            | 66.80            | 81.25            | 74.70            | 67.44            |
  | CenterPoint J6Gen2/2.5.1     | 73.31 | 82.49            | 64.58            | 77.82            | 75.54            | 66.11            |

  </details>

</details>

## Release

### CenterPoint J6Gen2/2.5.1

<details>
<summary> Changes  </summary>

- Finetune from `CenterPoint base/2.5.0` with j6gen2 base dataset
- Include intensity as an extra feature and Repeat Sampling Factor (RFS)
- Train with new datatasets:
  - `db_j6gen2_v6`
	- `db_j6gen2_v7`
	- `db_j6gen2_v8`
- Around a total frames of `18,000` is added compared to `CenterPoint J6Gen2/2.4.1`
- Overall:
  - Performance is generally better than `CenterPoint base/2.5.0` and `CenterPoint J6Gen2/2.5.1`
	- However, the performance in `largebus` slightly degrades

</details>

<details>
<summary> Artifacts </summary>

- Deployed onnx and ROS parameter files (for internal)
	- [WebAuto](https://evaluation.tier4.jp/evaluation/mlpackages/7156b453-2861-4ae9-b135-e24e48cc9029/releases/5599b337-e151-4f11-abe3-480943d9edec?project_id=zWhWRzei)
	- [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.5.1/deployment.zip)
	- [Google drive](https://drive.google.com/file/d/1PLhgM8vAJCeI0TvOi7wsg4seWzfJTi7m/view?usp=drive_link)
- Logs (for internal)
	- [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.5.1/logs.zip)
	- [Google drive](https://drive.google.com/file/d/1Lw_k5QUs0Wx1W-7mY-KrwtjW-eBc17pv/view?usp=drive_link)

</details>

<details>
<summary> Training configs </summary>

- [Config file path](https://github.com/tier4/AWML/blob/dee55764f5381ef75dcac7a17a303b0bf527d400/projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_j6gen2_base_amp_rfs.py)
- Train time: NVIDIA H100 80GB * 4 * 30 epochs = 1 day 3 hours
- Batch size: 4*16 = 64
- Training Dataset (frames: 48,108):
  - j6gen2: j6gen2: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 + db_j6gen2_v6 + db_j6gen2_v7 + db_j6gen2_v8 (37,002 frames)
	- largebus: db_largebus_v1 + db_largebus_v2 (11,106 frames)

</details>

<details>
<summary> Evaluation </summary>

**LargeBus**: db_largebus_v1 + db_largebus_v2 (859 frames)  
**Total mAP (eval range = 120m): 0.765**

| class_name | Count        | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | -------------| ---- | ---- | ---- | ---- | ---- |
| car        | 16,604       | 90.8 | 85.9 | 91.4 | 92.8 | 93.1 |
| truck      |  1,961       | 69.1 | 56.2 | 69.3 | 74.6 | 76.5 |
| bus        |    171       | 77.9 | 63.0 | 82.8 | 82.9 | 82.9 |
| bicycle    |    863       | 73.4 | 69.4 | 73.9 | 75.1 | 75.3 |
| pedestrian |   4,659      | 71.1 | 69.7 | 70.7 | 71.3 | 72.7 |

---


**J6Gen2_V6**: db_j6gen2_v6 (636 frames)  
**Total mAP (eval range = 120m): 0.734**

| class_name | Count      | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | ---------- | ---- | ---- | ---- | ---- | ---- |
| car        | 23,898     | 85.5 | 79.3 | 85.6 | 88.3 | 88.6 |
| truck      |  1,534     | 59.2 | 48.5 | 56.1 | 60.2 | 72.0 |
| bus        |    957     | 74.0 | 66.6 | 74.3 | 77.4 | 77.6 |
| bicycle    |    163     | 79.3 | 76.4 | 80.3 | 80.3 | 80.3 |
| pedestrian |  4,556     | 69.1 | 67.6 | 68.7 | 69.4 | 70.8 |

---


**J6Gen2:** db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 + db_j6gen2_v6  
**Total mAP (eval range = 120m): 0.752**

| class_name | Count      | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | ---------- | ---- | ---- | ---- | ---- | ---- |
| car        | 66,293     | 84.8 | 78.6 | 84.9 | 87.3 | 88.2 |
| truck      |  4,417     | 56.2 | 46.1 | 54.5 | 57.9 | 66.5 |
| bus        |  2,353     | 80.7 | 75.6 | 80.6 | 83.0 | 83.7 |
| bicycle    |    500     | 84.7 | 83.3 | 85.1 | 85.1 | 85.5 |
| pedestrian |  11,417    | 69.7 | 67.8 | 69.3 | 70.1 | 71.4 |

</details>

---

### CenterPoint J6Gen2/2.4.1

<details>
<summary> Changes  </summary>

- Finetune from `CenterPoint base/2.4.0` with j6gen2 base dataset
  - Include intensity as an extra feature and Repeat Sampling Factor (RFS)
- Overall:
   - Performance is better than `CenterPoint J6Gen2/2.3.1`, especially, `truck`, `bicycle` and `pedestrian`, where `bicycle` improves mAP more than `5.0` in `j6gen2_base` dataset

</details>

<details>
<summary> Artifacts </summary>

- Deployed onnx and ROS parameter files (for internal)
	- [WebAuto](https://evaluation.tier4.jp/evaluation/mlpackages/7156b453-2861-4ae9-b135-e24e48cc9029/releases/cb7d790e-4efe-47c2-b2b4-62d9d80aa085?project_id=zWhWRzei)
	- [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.3.1/deployment.zip)
	- [Google drive](https://drive.google.com/file/d/1toIlwTYbjIkXVoRdG4e0WSUBwqRKQDNi/view?usp=drive_link)
- Logs (for internal)
	- [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.3.1/logs.zip)
	- [Google drive](https://drive.google.com/file/d/1i4uvbOHsbDuNn0CtF1Dkx4CTrrjCAICt/view?usp=drive_link)

</details>

<details>
<summary> Training configs </summary>

- [Config file path](https://github.com/tier4/AWML/blob/c0ba7268f110062f71ee80a3469102867a63b740/projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_j6gen2_base.py)
- Train time: NVIDIA H100 80GB * 4 * 30 epochs = 1 day 3 hours
- Batch size: 4*16 = 64
- Training Dataset (frames: 30,290):
  - j6gen2: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 (21,077 frames)
	- largebus: db_largebus_v1 + db_largebus_v2 (9,213 frames)

</details>

<details>
<summary> Evaluation </summary>

- db_largebus_v1 + db_largebus_v2 (859 frames):
- Total mAP (eval range = 120m): 0.765

| class_name | Count        | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | -------------| ---- | ---- | ---- | ---- | ---- |
| car        | 16,604       | 90.8 | 85.9    | 91.4    | 92.8    | 93.1    |
| truck      |  1,961       | 69.1 | 56.2    | 69.3    | 74.6    | 76.5    |
| bus        |    171       | 77.9 | 63.0    | 82.8    | 82.9    | 82.9    |
| bicycle    |    863       | 73.4 | 69.4    | 73.9    | 75.1    | 75.3    |
| pedestrian |   4,659      | 71.1 | 69.7    | 70.7    | 71.3    | 72.7    |

---

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

---

<details>
<summary> Changes  </summary>

- Finetune from `CenterPoint base/2.5.0` with j6gen2 base dataset
- Include intensity as an extra feature and Repeat Sampling Factor (RFS)
- Train with new datatasets:
  - `db_j6gen2_v6`
	- `db_j6gen2_v7`
	- `db_j6gen2_v8`
- Around a total frames of `18,000` is added compared to `CenterPoint J6Gen2/2.4.1`
- Overall:
  - Performance is generally better than `CenterPoint base/2.5.0` and `CenterPoint J6Gen2/2.5.1`
	- However, the performance in `largebus` slightly degrades

</details>

<details>
<summary> Artifacts </summary>

- Deployed onnx and ROS parameter files (for internal)
	- [WebAuto](https://evaluation.tier4.jp/evaluation/mlpackages/7156b453-2861-4ae9-b135-e24e48cc9029/releases/5599b337-e151-4f11-abe3-480943d9edec?project_id=zWhWRzei)
	- [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.5.1/deployment.zip)
	- [Google drive](https://drive.google.com/file/d/1PLhgM8vAJCeI0TvOi7wsg4seWzfJTi7m/view?usp=drive_link)
- Logs (for internal)
	- [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.5.1/logs.zip)
	- [Google drive](https://drive.google.com/file/d/1Lw_k5QUs0Wx1W-7mY-KrwtjW-eBc17pv/view?usp=drive_link)

</details>

<details>
<summary> Training configs </summary>

- [Config file path](https://github.com/tier4/AWML/blob/dee55764f5381ef75dcac7a17a303b0bf527d400/projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_j6gen2_base_amp_rfs.py)
- Train time: NVIDIA H100 80GB * 4 * 30 epochs = 1 day 3 hours
- Batch size: 4*16 = 64
- Training Dataset (frames: 48,108):
  - j6gen2: j6gen2: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 + db_j6gen2_v6 + db_j6gen2_v7 + db_j6gen2_v8 (37,002 frames)
	- largebus: db_largebus_v1 + db_largebus_v2 (11,106 frames)

</details>

<details>
<summary> Evaluation </summary>

**LargeBus**: db_largebus_v1 + db_largebus_v2 (859 frames)  
**Total mAP (eval range = 120m): 0.765**

| class_name | Count        | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | -------------| ---- | ---- | ---- | ---- | ---- |
| car        | 16,604       | 90.8 | 85.9 | 91.4 | 92.8 | 93.1 |
| truck      |  1,961       | 69.1 | 56.2 | 69.3 | 74.6 | 76.5 |
| bus        |    171       | 77.9 | 63.0 | 82.8 | 82.9 | 82.9 |
| bicycle    |    863       | 73.4 | 69.4 | 73.9 | 75.1 | 75.3 |
| pedestrian |   4,659      | 71.1 | 69.7 | 70.7 | 71.3 | 72.7 |

---


**J6Gen2_V6**: db_j6gen2_v6 (636 frames)  
**Total mAP (eval range = 120m): 0.734**

| class_name | Count      | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | ---------- | ---- | ---- | ---- | ---- | ---- |
| car        | 23,898     | 85.5 | 79.3 | 85.6 | 88.3 | 88.6 |
| truck      |  1,534     | 59.2 | 48.5 | 56.1 | 60.2 | 72.0 |
| bus        |    957     | 74.0 | 66.6 | 74.3 | 77.4 | 77.6 |
| bicycle    |    163     | 79.3 | 76.4 | 80.3 | 80.3 | 80.3 |
| pedestrian |  4,556     | 69.1 | 67.6 | 68.7 | 69.4 | 70.8 |

---

**J6Gen2:** db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 + db_j6gen2_v6  
**Total mAP (eval range = 120m): 0.752**

| class_name | Count      | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | ---------- | ---- | ---- | ---- | ---- | ---- |
| car        | 66,293     | 84.8 | 78.6 | 84.9 | 87.3 | 88.2 |
| truck      |  4,417     | 56.2 | 46.1 | 54.5 | 57.9 | 66.5 |
| bus        |  2,353     | 80.7 | 75.6 | 80.6 | 83.0 | 83.7 |
| bicycle    |    500     | 84.7 | 83.3 | 85.1 | 85.1 | 85.5 |
| pedestrian |  11,417    | 69.7 | 67.8 | 69.3 | 70.1 | 71.4 |

</details>

### CenterPoint J6Gen2/2.3.1

<details>
<summary> Changes  </summary>

- Finetune from `CenterPoint base/2.3.0` with j6gen2 base dataset
- Include intensity as an extra feature

- Overall:
   - Performance is slightly better than `CenterPoint J6Gen2/2.2.1`

</details>

<details>
<summary> Artifacts </summary>

- Deployed onnx and ROS parameter files (for internal)
	- [WebAuto](https://evaluation.tier4.jp/evaluation/mlpackages/7156b453-2861-4ae9-b135-e24e48cc9029/releases/cb7d790e-4efe-47c2-b2b4-62d9d80aa085?project_id=zWhWRzei)
	- [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.3.1/deployment.zip)
	- [Google drive](https://drive.google.com/file/d/1toIlwTYbjIkXVoRdG4e0WSUBwqRKQDNi/view?usp=drive_link)
- Logs (for internal)
	- [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.3.1/logs.zip)
	- [Google drive](https://drive.google.com/file/d/1i4uvbOHsbDuNn0CtF1Dkx4CTrrjCAICt/view?usp=drive_link)

</details>

<details>
<summary> Training configs </summary>

- [Config file path](https://github.com/tier4/AWML/blob/c0ba7268f110062f71ee80a3469102867a63b740/projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_j6gen2_base.py)  
- Train time: NVIDIA H100 80GB * 4 * 30 epochs = 12 hours
- Batch size: 4*16 = 64
- Training Dataset (frames: 30,290):
  - j6gen2: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 (21,077 frames)
	- largebus: db_largebus_v1 + db_largebus_v2 (9,213 frames)

</details>

<details>
<summary> Evaluation </summary>

- db_largebus_v1 + db_largebus_v2 (859 frames):
- Total mAP (eval range = 120m): 0.748

| class_name | Count        | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | -------------| ---- | ---- | ---- | ---- | ---- |
| car        | 16,604       | 91.2 | 86.0    | 92.0    | 93.2    | 93.5    |
| truck      |  1,961       | 69.7 | 57.1    | 70.3    | 74.9    | 76.5    |
| bus        |    171       | 75.4 | 59.6    | 80.7    | 80.7    | 80.7    |
| bicycle    |    863       | 67.7 | 62.0    | 68.9    | 69.9    | 70.0    |
| pedestrian |   4,659      | 69.8 | 68.1    | 69.2    | 70.1    | 71.8    |

---

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

---

### CenterPoint J6Gen2/2.2.1

<details>
<summary> Changes  </summary>

- Finetune from `CenterPoint base/2.2.0` with j6gen2 base dataset
- Include intensity as an extra feature

- Overall:
  - Performance is slightly better than `CenterPoint J6Gen2/2.1.1`

</details>

<details>
<summary> Artifacts </summary>

- Deployed onnx and ROS parameter files (for internal)
	- [WebAuto](https://evaluation.tier4.jp/evaluation/mlpackages/7156b453-2861-4ae9-b135-e24e48cc9029/releases/115b1a13-84be-4dde-a9d2-8293e2be36ba?project_id=zWhWRzei)
	- [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.2.1/deployment.zip)
	- [Google drive](https://drive.google.com/file/d/1AXlkBB1aG7h0kzk5NeU2OIPlRZS2YfwQ/view?usp=drive_link)
- Logs (for internal)
	- [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.2.1/logs.zip)
	- [Google drive](https://drive.google.com/file/d/1i09BfY6LcVjSb4pYHL7IVIRP4Eus2Joo/view?usp=drive_link)

</details>

<details>
<summary> Training configs </summary>

- [Config file path](https://github.com/tier4/AWML/blob/81314d29d4efa560952324c48ef7c0ea1e56f1ee/projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_j6gen2_base.py)  
- Train time: NVIDIA H100 80GB * 4 * 30 epochs = 12 hours
- Batch size: 4*16 = 64
- Training dataset: DB J6 Gen2 v1.0 + DB J6 Gen2 v2.0 + DB J6 Gen2 V4.0 + DB LargeBus v1.0 (total frames: 20,777)

</details>

<details>
<summary> Evaluation </summary>

- db_largebus_v1 (total frames: 604):
- Total mAP (eval range = 120m): 0.7467

| class_name | Count    | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | -------  | ---- | ---- | ---- | ---- | ---- |
| car        |  13,831  | 90.2 | 85.1    | 91.1    | 92.3    | 92.5    |
| truck      |  2,137   | 70.6 | 55.1    | 72.7    | 76.1    | 78.4    |
| bus        |     95   | 78.2 | 74.0    | 78.1    | 80.3    | 80.3    |
| bicycle    |    724   | 66.5 | 59.3    | 68.1    | 69.4    | 69.4    |
| pedestrian |  3,916   | 67.8 | 66.1    | 67.3    | 68.2    | 69.6    |

---

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

---

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
