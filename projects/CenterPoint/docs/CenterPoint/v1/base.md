# Deployed model for CenterPoint base/1.X
## Summary

- Main parameter
  - range = 121.60m
  - voxel_size = [0.32, 0.32, 8.0]
  - grid_size = [760, 760, 1]
- Performance summary
  - Dataset: test dataset of db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 + db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 (total frames: 3026)
  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m)

| eval range: 120m     | mAP  | car <br> (610,396) | truck <br> (151,672) | bus <br> (37,876) | bicycle <br> (47,739) | pedestrian <br> (367,200) |
| -------------------- | ---- | ------------------ | -------------------- | ----------------- | --------------------- | ------------------------- |
| CenterPoint base/1.1 | 64.7 | 76.0               | 53.2                 | 77.6              | 52.4                  | 64.0                      |
| CenterPoint base/1.0 | 62.6 | 74.2               | 48.3                 | 75.4              | 51.6                  | 63.4                      |

## Release

### CenterPoint base/1.1

- Trained with additional data of robobus from a new location.
- Significantly improved mAP on the robobus dataset of the new location, without hampering the performance on the old dataset.
- Compared to base/1.0, the total mAP improved by 27% from 0.386 to 0.492
- The mAP for truck improved by 89% (from 0.294 to 0.541).
- The mAP for car improved by 7.8% (from 0.714 to 0.770).
- The mAP for bus improved by 7.3% (from 0.178 to 0.191).
- The mAP for bicycle improved by 14.1% (from 0.671 to 0.766).
- The mAP for pedestrian improved by 160% (from 0.075 to 0.195).
- But there is still a lot of room for improvement compared to other locations.
- **Note**: The model was trained by setting num_idar_pts for all bboxes to 10 because of a bug in conversion from deepen to t4dataset (which had set the num_idar_pts to 0). So, model should be retrained with the correct num_idar_pts.

<details>
<summary> The link of data and evaluation result </summary>

- Evaluation result with db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 + db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 (total frames: 3026):

| Eval range = 120m             | mAP  | car  | truck | bus  | bicycle | pedestrian |
| ----------------------------- | ---- | ---- | ----- | ---- | ------- | ---------- |
| base/1.1 (122.88m)            | 64.7 | 76.0 | 53.2  | 77.6 | 52.4    | 64.0       |
| base/1.0 (122.88m)            | 62.6 | 74.2 | 48.3  | 75.4 | 51.6    | 63.4       |


- Evaluation result with db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 + db_gsm8_v1 + db_j6_v1 + db_j6_v2 (total frames: 2787):

| Eval range = 120m             | mAP  | car  | truck | bus  | bicycle | pedestrian |
| ----------------------------- | ---- | ---- | ----- | ---- | ------- | ---------- |
| base/1.1 (122.88m)            | 64.8 | 76.1 | 53.2  | 78.7 | 51.8    | 64.4       |
| base/1.0 (122.88m)            | 64.5 | 75.0 | 50.7  | 78.1 | 53.2    | 64.8       |

- Evaluation result with db_j6_v3  (total frames: 239):

| Eval range = 120m             | mAP  | car  | truck | bus  | bicycle | pedestrian |
| ----------------------------- | ---- | ---- | ----- | ---- | ------- | ---------- |
| base/1.1 (122.88m)            | 49.2 | 77.0 | 54.1  | 19.1 | 76.6    | 19.5       |
| base/1.0 (122.88m)            | 38.6 | 71.4 | 29.4  | 17.8 | 67.1    | 7.5        |

- Model
  - Training dataset: db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v3 + db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 (total frames: 40769)
  - [PR](https://github.com/tier4/autoware-ml/pull/389)
  - [Config file path](https://github.com/tier4/autoware-ml/blob/d037b1d511d0ffb6f37f3e4e13460bc8483e2ccf/projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_2xb8_121m_base.py)
  - Deployed onnx model and ROS parameter files [[webauto]](https://evaluation.tier4.jp/evaluation/mlpackages/7156b453-2861-4ae9-b135-e24e48cc9029/releases/48c47d87-5f09-415d-9f69-d9857f513fff?project_id=zWhWRzei&tab=items)
  - Deployed onnx and ROS parameter files [[model-zoo]]
    - [detection_class_remapper.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v1.1/detection_class_remapper.param.yaml)
    - [centerpoint_t4base_ml_package.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v1.1/centerpoint_t4base_ml_package.param.yaml)
    - [deploy_metadata.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v1.1/deploy_metadata.yaml)
    - [pts_voxel_encoder_centerpoint_t4base.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v1.1/pts_voxel_encoder_centerpoint_t4base.onnx)
    - [pts_backbone_neck_head_centerpoint_t4base.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v1.1/pts_backbone_neck_head_centerpoint_t4base.onnx)
  - Training results [model-zoo]
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v1.1/logs.zip)
    - [checkpoint_best.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v1.1/epoch_50.pth)
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v1.1/second_secfpn_121m_2xb8.py)
  - train time: NVIDIA A100 80GB * 2 * 50 epochs = 4.5 days
- Evaluation result with test-dataset: db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 + db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 (total frames: 3026):
  - Total mAP (eval range = 120m): 0.647

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---- | ---- | ---- | ---- | ---- | ---- |
| car        | 76.0 | 66.9    | 77.4    | 79.5    | 80.4    |
| truck      | 53.2 | 32.4    | 53.3    | 60.7    | 66.5    |
| bus        | 77.6 | 66.4    | 79.8    | 81.4    | 82.8    |
| bicycle    | 52.4 | 51.8    | 52.4    | 52.4    | 53.0    |
| pedestrian | 64.0 | 61.9    | 63.2    | 64.6    | 66.4    |

- Evaluation result with eval-dataset db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 + db_gsm8_v1 + db_j6_v1 + db_j6_v2 (total frames: 2787):
  - Total mAP (eval range = 120m): 0.648

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---- | ---- | ---- | ---- | ---- | ---- |
| car        | 76.1 | 66.6    | 77.9    | 79.6    | 80.4    |
| truck      | 53.2 | 31.9    | 53.2    | 60.8    | 67.0    |
| bus        | 78.7 | 67.4    | 80.9    | 82.6    | 84.0    |
| bicycle    | 51.8 | 50.9    | 52.0    | 52.1    | 52.1    |
| pedestrian | 64.4 | 62.2    | 63.5    | 65.0    | 66.8    |

- Evaluation result with eval-dataset db_j6_v3  (total frames: 239):
  - Total mAP (eval range = 120m): 0.492

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---- | ---- | ---- | ---- | ---- | ---- |
| car        | 77.0 | 71.3    | 77.6    | 79.0    | 80.1    |
| truck      | 54.1 | 44.7    | 54.2    | 56.7    | 60.5    |
| bus        | 19.1 | 13.8    | 20.9    | 20.9    | 20.9    |
| bicycle    | 76.6 | 76.6    | 76.6    | 76.6    | 76.6    |
| pedestrian | 19.5 | 18.9    | 19.3    | 19.4    | 20.1    |

- Evaluation result of **base/1.0 ↓** with eval-dataset db_j6_v3  (total frames: 239):
  - Total mAP (eval range = 120m): 0.386

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---- | ---- | ---- | ---- | ---- | ---- |
| car        | 71.4 | 65.6    | 71.5    | 73.7    | 74.9    |
| truck      | 29.4 | 21.8    | 29.6    | 31.9    | 34.4    |
| bus        | 17.8 | 15.3    | 18.6    | 18.6    | 18.6    |
| bicycle    | 67.1 | 67.1    | 67.1    | 67.1    | 67.1    |
| pedestrian | 7.5  | 7.3     | 7.3     | 7.4     | 7.9     |

- Evaluation result of **base/1.0 ↓** with eval-dataset db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 + db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 (total frames: 3026):
  - Total mAP (eval range = 120m): 0.626

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---- | ---- | ---- | ---- | ---- | ---- |
| car        | 74.2 | 64.0    | 75.7    | 78.1    | 79.0    |
| truck      | 48.3 | 26.1    | 48.5    | 56.7    | 62.1    |
| bus        | 75.4 | 65.9    | 76.8    | 78.8    | 80.1    |
| bicycle    | 51.6 | 51.0    | 51.5    | 51.5    | 52.2    |
| pedestrian | 63.4 | 61.2    | 62.5    | 63.9    | 65.9    |

</details>

### CenterPoint base/1.0

- The first CenterPoint model trained with autoware-ml using TIER IV's in-house dataset. It has been carefully evaluated against the older version, demonstrating comparable performance. This marks a significant step towards lifelong MLOps for object recognition models.
- Trained with data from Robotaxi and Robobus, enhancing robustness across variations in vehicle and sensor configurations.
- Evaluation data for Robobus highlights notable improvements, with mean Average Precision (mAP) increasing by approximately 5% for buses and 20% for trucks, showcasing enhanced performance for large-sized objects.

<details>
<summary> The link of data and evaluation result </summary>

We evaluate for T4dataset and compare to old library.
Old library is based on mmdetection3d v0 and we integrate to mmdetection3d v1 based library.

| Eval range = 120m             | mAP  | car  | truck | bus  | bicycle | pedestrian |
| ----------------------------- | ---- | ---- | ----- | ---- | ------- | ---------- |
| Old library version (122.88m) | 62.2 | 74.7 | 43.0  | 75.0 | 54.0    | 64.1       |
| CenterPoint v1.0.0 (122.88m)  | 64.5 | 75.0 | 50.7  | 78.1 | 53.2    | 64.8       |

- Model
  - Training dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v3.0 + DB GSM8 v1.0 + DB J6 v1.0 (total frames: 34,137)
  - [PR](https://github.com/tier4/autoware-ml/pull/159)
  - [Config file path](https://github.com/tier4/autoware-ml/blob/5f472170f07251184dc009a1ec02be3b4f3bf98c/autoware_ml/configs/detection3d/dataset/t4dataset/base.py)
  - Deployed onnx model and ROS parameter files [[webauto]](https://evaluation.tier4.jp/evaluation/mlpackages/7156b453-2861-4ae9-b135-e24e48cc9029/releases/1711f9c5-defa-4af1-b94b-e7978500df89?project_id=zWhWRzei)
  - Deployed onnx and ROS parameter files [[model-zoo]]
    - [detection_class_remapper.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v1.0/detection_class_remapper.param.yaml)
    - [centerpoint_t4base_ml_package.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v1.0/centerpoint_t4base_ml_package.param.yaml)
    - [deploy_metadata.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v1.0/deploy_metadata.yaml)
    - [pts_voxel_encoder_centerpoint_t4base.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v1.0/pts_voxel_encoder_centerpoint_t4base.onnx)
    - [pts_backbone_neck_head_centerpoint_t4base.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v1.0/pts_backbone_neck_head_centerpoint_t4base.onnx)
  - Training results [[GDrive]](https://drive.google.com/drive/u/0/folders/1bMarMoNQXdF_3nB-BjFx28S5HMIfgeIJ)
  - Training results [model-zoo]
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v1.0/logs.zip)
    - [checkpoint_best.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v1.0/epoch_50.pth)
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v1.0/second_secfpn_121m_2xb8.py)
  - train time: NVIDIA A100 80GB * 2 * 50 epochs = 4.5 days
- Evaluation result with test-dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v3.0 + DB GSM8 v1.0 + DB J6 v1.0 (total frames: 1,394):
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
