# Deployed model for StreamPETR base/2.X
## Summary

### Overview
- Main parameter
  - range: 51.2m
  - image_size: (480, 640)
  - images: `CAM_FRONT, CAM_FRONT_LEFT, CAM_BACK_LEFT, CAM_FRONT_RIGHT, CAM_BACK_RIGHT`

| eval range: 50m          | mAP  | car             | truck              | bus           | bicycle            | pedestrian             |
| -------------------------| ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
| CenterPoint base/2.3     | 80.00 | 92.3            | 67.6               | 88.2         | 78.1                 | 73.8                   |
| StreamPETR base/2.3    | 40.40 | 57.1            | 37.0               | 53.5         | 14.5                 | 40.1                   |
| StreamPETR base/2.5    | 45.00 | 61.7            | 41.7               | 60.7         | 16.7                 | 44.4                   |

## Release
### StreamPETR base/2.5
- Train more data with:
    - `db_largebus_v1`
		- `db_largebus_v2`
		- `db_j6gen2_v1`
		- `db_j6gen2_v2`
		- `db_j6gen2_v4`
		- `db_j6gen2_v5`
		- `db_jpntaxi_gen2_v1`
		- `db_jpntaxi_gen2_v2`
- Train with new datatasets:
    - `db_j6gen2_v6`
		- `db_j6gen2_v7`
		- `db_j6gen2_v8`
<details>
<summary> The link of data and evaluation result </summary>

- Model

  - Training Dataset (frames: 123,708):
  
    - jpntaxi: db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 (25,958 frames)
    - j6: db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 (24,756 frames)
    - j6gen2: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 + db_j6gen2_v6 + db_j6gen2_v7 + db_j6gen2_v8 (37,002 frames)
    - largebus: db_largebus_v1 + db_largebus_v2 (11,106 frames)
    - jpntaxi_gen2: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (24,992 frames)

  - [Config file path](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/streampetr/streampetr-vov99/t4base/v2.5/t4_base_vov_flash_480x640_baseline.py)
  - [Model Checkpoint](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/streampetr/streampetr-vov99/t4base/v2.5/best_NuScenesmetric_T4Metric_mAP_epoch_34.pth)
  - Deployed onnx and ROS parameter files (for internal)
    - [WebAuto](https://evaluation.tier4.jp/evaluation/mlpackages/28c2254f-2d62-417a-bcfa-d5872e331a34/releases/e27c2ba3-d916-48e2-8177-5e7838b57ee2?project_id=zWhWRzei)
    - model-zoo
      - [simplify_extract_img_feat.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/streampetr/streampetr-vov99/t4base/v2.5/simplify_extract_img_feat.onnx)
      - [simplify_position_embedding.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/streampetr/streampetr-vov99/t4base/v2.5/simplify_position_embedding.onnx)
      - [simplify_pts_head_memory.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/streampetr/streampetr-vov99/t4base/v2.5/simplify_pts_head_memory.onnx)
  - Logs (for internal)
    - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/streampetr/streampetr-vov99/t4base/v2.5/logs.zip)
  - Train time: NVIDIA H100 80GB * 4 * 50 epochs = 2 days
  - Batch size: 4*16 = 64

- Evaluation
   - Datasets (8,453 frames):
      - jpntaxi: db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 (1,507 frames)
      - j6: db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 (2,435 frames)
      - j6gen2: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 + db_j6gen2_v6 (1,943 frames)
      - largebus: db_largebus_v1 + db_largebus_v2 (859 frames)
      - jpntaxi_gen2: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (1,709 frames)
  - Total mAP (eval range = 50m): 0.45

| class_name | counts | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| car        | 80782  | 61.7 | 27.1    | 56.5    | 77.7    | 85.3    |
| truck      | 9158 | 41.7 | 10.4    | 34.2    | 55.1    | 67.0    |
| bus        | 3969 | 60.7 | 14.9    | 57.0    | 81.8    | 89.0    |
| bicycle    | 5491 | 16.7 | 2.1     | 12.5    | 23.2    | 29.1    |
| pedestrian |  36097 | 44.4 | 10.9    | 35.0    | 60.1    | 71.5    |
</details>

### StreamPETR base/2.3
- Add a new training set: `db_jpntaxigen2_v2`, `db_j6gen2_v3`, `db_j6gen2_v5`, and `db_largebus_v2`
- Add new data to `db_j6gen2_v4`, `db_largebus_v1`

<details>
<summary> The link of data and evaluation result </summary>

- Model
  - Training Datasets (frames: 99,776):
	    - jpntaxi: db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 (26,100 frames)
			- j6: db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 (24,756 frames)
			- j6gen2: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 (21,077 frames)
			- largebus: db_largebus_v1 + db_largebus_v2 (9,213 frames)
			- jpntaxi_gen2: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (18,630 frames)
  - [Config file path](https://github.com/tier4/AWML/blob/ee1150427900393f815b8df99bf7530f0ec8de1c/projects/StreamPETR/configs/t4dataset/t4_base_vov_flash_480x640_baseline.py)
  - [Model Checkpoint](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/streampetr/streampetr-vov99/t4base/v2.3/best_NuScenes_metric_T4Metric_mAP_epoch_31.pth)
  - Deployed onnx and ROS parameter files (for internal)
    - [WebAuto](https://evaluation.tier4.jp/evaluation/mlpackages/28c2254f-2d62-417a-bcfa-d5872e331a34/releases/1ce03dfa-540e-435d-9573-324590d85d94?project_id=prd_jt)
    - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/streampetr/streampetr-vov99/t4base/v2.3/deployment.zip)
  - Logs (for internal)
    - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/streampetr/streampetr-vov99/t4base/v2.3/logs.zip)
  - Train time: NVIDIA H100 80GB * 4 * 50 epochs = 2 days
  - Batch size: 4*16 = 64

- Evaluation
   - Datasets (frames: 7,727):
	    - jpntaxi: db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 (1,507 frames)
			- j6: db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 (2,435 frames)
			- j6gen2: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 (1,217 frames)
			- largebus: db_largebus_v1 + db_largebus_v2 (859 frames)
			- jpntaxi_gen2: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (1,709 frames)
  - Total mAP (eval range = 50m): 0.41

| class_name | counts | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| car        | 68304 | 57.7 | 22.2    | 51.2    | 74.5    | 82.9    |
| truck      | 8291  | 38.0 | 9.0     | 28.4    | 48.9    | 65.8    |
| bus        | 3522  | 55.8 | 10.4    | 48.9    | 78.7    | 85.4    |
| bicycle    | 5264  | 13.5 | 0.6     | 7.3     | 20.3    | 25.9    |
| pedestrian | 32613 | 39.7 | 6.9     | 28.0    | 53.4    | 70.6    |

</details>
