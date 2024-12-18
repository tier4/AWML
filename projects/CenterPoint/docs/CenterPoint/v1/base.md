# Deployed model for CenterPoint base/1.X
## Summary

- Main parameter
  - range = 121.60m
  - voxel_size = [0.32, 0.32, 8.0]
  - grid_size = [760, 760, 1]
- Performance summary
  - Dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v3.0 + DB GSM8 v1.0 + DB J6 v1.0 (total frames: 34,137)
  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m)

| eval range: 120m     | mAP  | car <br> (610,396) | truck <br> (151,672) | bus <br> (37,876) | bicycle <br> (47,739) | pedestrian <br> (367,200) |
| -------------------- | ---- | ------------------ | -------------------- | ----------------- | --------------------- | ------------------------- |
| CenterPoint base/1.0 | 64.4 | 75.0               | 50.7                 | 78.1              | 53.2                  | 64.8                      |

## Release
### CenterPoint base/1.0

- The first CenterPoint model trained with autoware-ml using TIER IV's in-house dataset. It has been carefully evaluated against the older version, demonstrating comparable performance. This marks a significant step towards lifelong MLOps for object recognition models.
- Trained with data from Robotaxi and Robobus, enhancing robustness across variations in vehicle and sensor configurations.
- Evaluation data for Robobus highlights notable improvements, with mean Average Precision (mAP) increasing by approximately 5% for buses and 20% for trucks, showcasing enhanced performance for large-sized objects.

<details>
<summary> The link of data and evaluation result </summary>

- Model
  - Training dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v3.0 + DB GSM8 v1.0 + DB J6 v1.0 (total frames: 34,137)
  - [PR](https://github.com/tier4/autoware-ml/pull/159)
  - [Config file path](https://github.com/tier4/autoware-ml/blob/5f472170f07251184dc009a1ec02be3b4f3bf98c/autoware_ml/configs/detection3d/dataset/t4dataset/base.py)
  - [Deployed onnx model and ROS parameter files](https://evaluation.tier4.jp/evaluation/mlpackages/7156b453-2861-4ae9-b135-e24e48cc9029/releases/1711f9c5-defa-4af1-b94b-e7978500df89?project_id=zWhWRzei)
  - [Training results](https://drive.google.com/drive/u/0/folders/1bMarMoNQXdF_3nB-BjFx28S5HMIfgeIJ)
  - train time: NVIDIA A100 80GB * 2 * 50 epochs = 4.5 days
- Evaluation result with test-dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v3.0 + DB GSM8 v1.0 + DB J6 v1.0 (total frames: 62):
  - Total mAP (eval range = 120m): 0.644

| class_name | Count  | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ------ | ---- | ------- | ------- | ------- | ------- |
| car        | 41,133 | 75.0 | 64.7    | 76.8    | 79.1    | 79.5    |
| truck      | 8,890  | 50.7 | 27.8    | 50.5    | 59.6    | 65.1    |
| bus        | 3,275  | 78.1 | 69.2    | 79.6    | 81.1    | 82.6    |
| bicycle    | 3,635  | 53.2 | 52.3    | 53.4    | 53.5    | 53.6    |
| pedestrian | 25,981 | 64.8 | 62.4    | 64.0    | 65.4    | 67.4    |

- Evaluation result with eval-dataset DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v3.0 (total frames: 50):
  - Total mAP (eval range = 120m): 0.633

| class_name | Count  | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ------ | ---- | ------- | ------- | ------- | ------- |
| car        | 16,126 | 74.8 | 61.2    | 77.3    | 79.8    | 80.9    |
| truck      | 4,578  | 53.3 | 32.7    | 53.3    | 60.8    | 66.4    |
| bus        | 1,457  | 66.4 | 52.2    | 67.9    | 71.4    | 74.0    |
| bicycle    | 1,040  | 56.3 | 53.9    | 56.6    | 57.3    | 57.4    |
| pedestrian | 11,971 | 65.5 | 62.1    | 64.7    | 66.6    | 68.6    |

- Evaluation result with eval-dataset DB GSM8 v1.0 + DB J6 v1.0 (total frames: 12):
  - Total mAP (eval range = 120m): 0.645

| class_name | Count  | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ------ | ---- | ------- | ------- | ------- | ------- |
| car        | 25,007 | 75.0 | 66.5    | 76.3    | 78.6    | 78.8    |
| truck      | 4,573  | 45.5 | 21.1    | 44.3    | 54.9    | 61.8    |
| bus        | 1,818  | 86.6 | 81.8    | 87.8    | 87.9    | 88.9    |
| bicycle    | 2,567  | 51.2 | 51.2    | 51.2    | 51.2    | 51.3    |
| pedestrian | 14,010 | 63.9 | 63.0    | 63.2    | 63.7    | 65.7    |

</details>
