
# Deployed model for CenterPoint JPNTaxi-base/2.X
## Summary
The main difference between `JPNTaxi-base/2.x` and `JPNTaxi-Gen2/2.x` is that all `JPNTaxi` data from Gen1 and Gen2 are used for training/validation/testing in `JPNTaxi-base/2.x` to reduce overfitting since they share similar
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
| CenterPoint JPNTaxi_base/2.5.1     | 66.60 | 81.50            | 49.90               | 67.90         | 62.20                 | 71.50                   |
| CenterPoint JPNTaxi_Gen2/2.4.1     | 59.10 | 73.40            | 42.90               | 59.10         | 51.50                 | 68.70                   |

### Datasets

<details>
<summary> JPNTaxi Gen2 </summary>

- Test datases: db_largebus_v1 + db_largebus_v2 (total frames: 859)

| eval range: 120m             | mAP     | car <br> (16,604)     | truck <br> (1,961) | bus <br> (171) | bicycle <br> (863) | pedestrian <br> (4,659) |
| ---------------------        | ----    | ----------------- | ------------------- | ---------------- | ----------------- | ---------------- |
| CenterPoint J6Gen2/2.5.1     | 75.40   | 90.80            | 68.70               | 77.60         | 68.60                 | 70.90                   |
| CenterPoint J6Gen2/2.4.1     | 76.50   | 90.80            | 69.10               | 77.90         | 73.40                 | 71.10                   |

</details>

<summary> JPNTaxi Gen2_V2 </summary>

- Test datases: db_largebus_v1 + db_largebus_v2 (total frames: 859)

| eval range: 120m             | mAP     | car <br> (16,604)     | truck <br> (1,961) | bus <br> (171) | bicycle <br> (863) | pedestrian <br> (4,659) |
| ---------------------        | ----    | ----------------- | ------------------- | ---------------- | ----------------- | ---------------- |
| CenterPoint J6Gen2/2.5.1     | 75.40   | 90.80            | 68.70               | 77.60         | 68.60                 | 70.90                   |
| CenterPoint J6Gen2/2.4.1     | 76.50   | 90.80            | 69.10               | 77.90         | 73.40                 | 71.10                   |

</details>

<details>
<summary> JPNTaxi Base </summary>

- Test datases: db_j6gen2_v6  (total frames: 636)

| eval range: 120m           | mAP     | car <br> (23,898) | truck <br> (1,534) | bus <br> (957) | bicycle <br> (163) | pedestrian <br> (4,556) |
| -------------------------  | ----    | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| CenterPoint J6Gen2/2.5.1   | 73.40   | 85.50            | 59.20               | 74.00         | 79.30                 | 69.10                   |
| CenterPoint J6Gen2/2.4.1   | 70.90   | 84.40            | 55.20               | 72.90         | 76.60                 | 65.30                   |


</details>

<details>
<summary> J6Gen2 </summary>

- Test datases: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 + db_j6gen2_v6  (total frames: 1,943)

| eval range: 120m           | mAP     | car <br> (66,293) | truck <br> (4,417) | bus <br> (2,353) | bicycle <br> (500) | pedestrian <br> (11,417) |
| -------------------------  | ----    | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| CenterPoint J6Gen2/2.5.1   | 75.20 | 84.80            | 56.20               | 80.70         | 84.70                 | 69.70                   |
| CenterPoint J6Gen2/2.4.1   | 74.40 | 84.10            | 55.70               | 81.10         | 83.20                 | 67.70                   |


</details>
