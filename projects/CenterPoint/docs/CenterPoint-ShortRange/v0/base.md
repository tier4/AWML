# Deployed model for CenterPoint-ConvNeXtPC base/0.X
## Summary

- Main parameter
  - range = 51.20m
- Performance summary
	- Dataset: test dataset of db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 + db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 + db_j6gen2_v1 (total frames: 3804)
  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m)

| eval range: 52m                 | mAP  | car <br> (38,739) | truck <br> (4,764) | bus <br> (2,185) | bicycle <br> (2,049) | pedestrian <br> (20,661) |
| ------------------------------- | ---- | ------------------ | -------------------- | ----------------- | --------------------- | ------------------------- |
| CenterPoint-ShortRange base/1.0 | 81.6 | 91.4               | 65.7                 | 92.4              | 82.3                  | 76.3                      |
| CenterPoint-ShortRange base/0.3 | 78.7 | 90.8               | 59.0                 | 90.5              | 80.8                  | 72.5                      |

## Deprecated Summary
<details>
<summary> Results with previous datasets </summary>

- Dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v3.0 + DB GSM8 v1.0 + DB J6 v1.0 (total frames: 35,292)
- [Comparisons in details](https://docs.google.com/spreadsheets/d/1cOIwmyiXA4Z0uAEl1mkPoaAjqJJ8Mq1O66tzzAOW15I/edit?gid=980227559#gid=980227559)

| eval range: 52m                 | mAP  | car <br> (629,212) | truck <br> (163,402) | bus <br> (39,904) | bicycle <br> (48,043) | pedestrian <br> (383,553) |
| ------------------------------- | ---- | ------------------ | -------------------- | ----------------- | --------------------- | ------------------------- |
| CenterPoint-ShortRange base/0.1 | 77.4 | 83.2               | 67.5                 | 79.4              | 82.3                  | 74.4                      |
| CenterPoint-ShortRange base/0.2 | 76.7 | 87.9               | 64.5                 | 84.3              | 75.5                  | 71.3                      |
| CenterPoint-ShortRange base/0.3 | 81.5 | 89.1               | 71.3                 | 91.3              | 81.0                  | 75.1                      |

- Deployment summary

| Runtime (ms)                    | Min  | P25   | P50   | P75   | P90   | Max   | Mean           |
| ------------------------------- | ---- | ----- | ----- | ----- | ----- | ----- | -------------- |
| CenterPoint-ShortRange base/0.1 | 5.91 | 9.08  | 10.28 | 12.23 | 16.77 | 30.79 | 11.33 (+-3.50) |
| CenterPoint-ShortRange base/0.2 | 5.91 | 9.08  | 10.28 | 12.23 | 16.77 | 30.79 | 11.33 (+-3.50) |
| CenterPoint-ShortRange base/0.3 | 9.96 | 12.29 | 14.64 | 16.32 | 19.03 | 45.82 | 15.35 (+-3.36) |


| GPU Memory (MB)                 | Min     | P25     | P50     | P75     | P90     | Max     | Mean              |
| ------------------------------- | ------- | ------- | ------- | ------- | ------- | ------- | ----------------- |
| CenterPoint-ShortRange base/0.1 | 4486.06 | 4508.69 | 4521.00 | 4524.44 | 4543.19 | 4633.81 | 4519.30 (+-20.38) |
| CenterPoint-ShortRange base/0.2 | 4486.06 | 4508.69 | 4521.00 | 4524.44 | 4543.19 | 4633.81 | 4519.30 (+-20.38) |
| CenterPoint-ShortRange base/0.3 | 3879.88 | 3882.56 | 3884.44 | 4084.62 | 4087.56 | 4218.75 | 3948.19 (+-96.93) |

</details>

## Release
### CenterPoint-ShortRange base/1.0
- This is the first short range model trained with `gen2` data
- The following changes are made as compared to `CenterPoint-ShortRange base/0.3`:
  - `PillarFeatureNet` instead of `BackwardPillarFeatureNet` to include distance of z to pillar center
  - `AMP` training
  - Use `ConvNeXT-PC` as a stronger backbone  
- `CenterPoint-ShortRange base/1.0` achieves a significantly higher mean average precision (mAP) by +2.9%, indicating better general detection performance
  - The biggest gain is in truck detection with a +6.7% boost in AP, which suggests that it handles medium-sized vehicles better, likely due to more training data
  - Pedestrian detection also improves notably by +3.8%, showing better handling of dense or occluded small objects
  - Gains for bus and bicycle are more modest but still consistent
	- Both models perform similarly on cars, with `CenterPoint-ShortRange base/1.0` only slightly better (+0.6%). This is expected since cars have the most samples (38,739), and performance is already saturated

<details>
<summary> The link of data and evaluation result </summary>

- Evaluation result with db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 + db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 + db_j6gen2_v1 (total frames: 3804):

| Eval range = 52m    | mAP  | car  | truck | bus  | bicycle | pedestrian |
| ------------------  | ---- | ---- | ----- | ---- | ------- | ---------- |
| ShortRange base/1.0  | 81.6 | 91.4 | 65.7  | 92.4 | 82.3    | 76.3       |
| ShortRange base/0.3  | 78.7 | 90.8 | 59.0  | 90.5 | 80.8    | 72.5       |

- Model
  - Training dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v4.0 + DB GSM8 v1.0 + DB J6 v1.0 + DB J6 v2.0 + DB J6 v3.0 + DB J6 v5.0 + DB J6 Gen2 v1.0 (total frames: 49,605)
  - [Config file path](hhttps://github.com/tier4/AWML/blob/c625cd745e5ea7b46f7f7818266b3901981ba4f6/autoware_ml/configs/detection3d/dataset/t4dataset/base.py)
  - Deployed onnx model and ROS parameter files [[WebAuto (for internal)]](https://evaluation.tier4.jp/evaluation/mlpackages/7156b453-2861-4ae9-b135-e24e48cc9029/releases/96f8e314-cbbc-47ca-9bc0-b73fc48377f3?project_id=zWhWRzei)
  - Deployed onnx and ROS parameter files [model-zoo]
    - [detection_class_remapper.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/t4base/v1.0/detection_class_remapper.param.yaml)
    - [centerpoint_t4base_ml_package.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/t4base/v1.0/centerpoint_t4base_ml_package.yaml)
    - [deploy_metadata.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/t4base/v1.0/deploy_metadata.yaml)
    - [pts_voxel_encoder_centerpoint_shortrange_t4base.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/t4base/v1.0/pts_voxel_encoder_centerpoint_shortrange_t4base.onnx)
    - [pts_backbone_neck_head_centerpoint_shortrange_t4base.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/t4base/v1.0/pts_backbone_neck_head_centerpoint_shortrange_t4base.onnx)
  - Training results [[Google drive (for internal)]](https://drive.google.com/drive/folders/18CTOTYICBcL56byyHi1E9bhYLJJh8ElX?usp=drive_link)
  - Training results [model-zoo]
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/t4base/v1.0/logs.zip)
    - [checkpoint_best.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/t4base/v1.0/best_epoch.pth)
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/t4base/v1.0/pillar_016_convnext_secfpn_4xb16_50m_base.py)
  - Train time: NVIDIA A100 80GB * 4 * 50 epochs = 2 days and 14 hours
  - Batch size: 4*16 = 64

- Evaluation result with db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 + db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 + db_j6gen2_v1 (total frames: 3804)
  - Total mAP (eval range = 52m): 0.816

| class_name | Count    | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| -----------| -------  | ---- | ------- | ------- | ------- | ------- |
| car        |  38,739  | 91.4 | 85.7    | 92.4    | 93.5    | 93.7    |
| truck      |   4,764  | 65.7 | 50.5    | 67.8    | 70.8    | 73.8    |
| bus        |   2,185  | 92.4 | 84.7    | 92.3    | 95.7    | 95.8    |
| bicycle    |   2,049  | 82.3 | 81.5    | 82.5    | 82.6    | 82.7    |
| pedestrian |  20,661  | 76.3 | 73.8    | 75.6    | 77.2    | 78.7    |

</details>

### CenterPoint-ShortRange base/0.3
- This is exactly the model [CenterPoint-ConvNeXtPC base/0.2](../../CenterPoint-ConvNeXtPC/v0/base.md) except it evaluates in 50m only
- The performance of detection is generally better than `CenterPoint-ShortRange base/0.1` (81.5 vs 77.4) since it also has better performance in 120m, especially, vehicle classes:
  - car (89.1 vs 83.2)
  - truck (71.3 vs 67.5)
  - bus (91.3 vs 79.4)
- However, the following small classes have only limited improvment or even worse performance:
  - bicycle (81.0 vs 82.3)
  - pedestrian (75.1 vs 74.4)

where it is suspected that we should further increase the resolution (e.g., 0.20 -> 0.16) and maintain the output factor by 1:1 (4 -> 1) for the small classes

<details>
<summary> The link of data and evaluation result </summary>

- Model
  - Please refer to this [summary](../../CenterPoint/v1/base.md) for the details  

- Evaluation result with test-dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v3.0 + DB GSM8 v1.0 + DB J6 v1.0 (total frames: 1,394):
  - Total mAP (eval range = 52m): 0.767

| class_name | Count  | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ------ | ---- | ------- | ------- | ------- | ------- |
| car        | 41,133 | 89.1 | 81.3    | 90.7    | 92.9    | 92.4    |
| truck      | 8,890  | 71.3 | 49.9    | 73.7    | 79.6    | 81.8    |
| bus        | 3,275  | 91.3 | 81.1    | 93.5    | 95.2    | 95.3    |
| bicycle    | 3,635  | 81.0 | 79.2    | 81.6    | 81.7    | 81.7    |
| pedestrian | 25,981 | 75.1 | 72.2    | 74.4    | 77.8    | 77.8    |

</details>

### CenterPoint-ShortRange base/0.2
- This is exactly the model [CenterPoint base/1.0](../../CenterPoint/v1/base.md) except it evaluates in 50m only
- The performance of detection is slightly worse compared to `CenterPoint-ShortRange base/0.1` (76.7 vs 77.4), however, the following classes perform better:
  - car (87.9 vs 83.2)
  - bus (84.3 vs 79.4)

where it is suspected that the higher resolution setting ([0.16, 0.16, 8.0]) should generally performs better in small classes, for example, bicycle and pedestrian

<details>
<summary> The link of data and evaluation result </summary>

- Model
  - Please refer to this [summary](../../CenterPoint/v1/base.md) for the details  

- Evaluation result with test-dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v3.0 + DB GSM8 v1.0 + DB J6 v1.0 (total frames: 1,394):
  - Total mAP (eval range = 52m): 0.767

| class_name | Count  | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ------ | ---- | ------- | ------- | ------- | ------- |
| car        | 41,133 | 87.9 | 80.1    | 89.4    | 90.9    | 91.2    |
| truck      | 8,890  | 64.5 | 41.2    | 66.0    | 73.9    | 77.0    |
| bus        | 3,275  | 84.3 | 77.5    | 85.8    | 86.9    | 87.1    |
| bicycle    | 3,635  | 75.5 | 74.7    | 75.7    | 75.8    | 75.8    |
| pedestrian | 25,981 | 71.3 | 68.8    | 70.6    | 72.0    | 73.7    |

</details>

### CenterPoint-ShortRange base/0.1
- This is the CenterPoint model based on `CenterPoint base/1.0` by targeting only objects near to the ego vehicle in 50m and increasing resolution during training  
- Note that it **doesn't** use any pretrained weights
- Specifically, the main changes in hyperparameters are:
	```python
		voxel_size: [0.16, 0.16, 8.0]
  		out_size_factor = 4
  		data_preprocessor: dict(
        	type="Det3DDataPreprocessor",
        	voxel=True,
        	voxel_layer=dict(
            	max_num_points=32,
            	voxel_size=voxel_size,
            	point_cloud_range=point_cloud_range,
            	max_voxels=(32000, 60000),
            	deterministic=True,
        	),
      	)
	```
<details>
<summary> The link of data and evaluation result </summary>

- Model
  - Training dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v3.0 + DB GSM8 v1.0 + DB J6 v1.0 (total frames: 35,392)
  - [Config file path](https://github.com/tier4/AWML/blob/cacf3f3dc282aed5760aeb596094e0652300c113/projects/CenterPoint/configs/t4dataset/pillar_016_second_secfpn_2xb8_50m_base.py)
  - Deployed onnx model and ROS parameter files [[Google drive (for internal)]](https://drive.google.com/drive/folders/18dNXXK0BzgXX3VKkiTMh59cj3b1-HMQP?usp=drive_link)
	- Deployed onnx and ROS parameter files [model-zoo]
    - [detection_class_remapper.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/t4base/v0.1/detection_class_remapper.param.yaml)
    - [centerpoint_x2_ml_package.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/t4base/v0.1/centerpoint_x2_ml_package.param.yaml)
    - [deploy_metadata.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/t4base/v0.1/deploy_metadata.yaml)
    - [pts_voxel_encoder_centerpoint_x2.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/t4base/v0.1/pts_voxel_encoder_centerpoint_x2.onnx)
    - [pts_backbone_neck_head_centerpoint_x2.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/t4base/v0.1/pts_backbone_neck_head_centerpoint_x2.onnx)
  - Training results [[Google drive (for internal)]](https://drive.google.com/drive/folders/1V9QDda9WLo6T-t0IA4A0NZ_xJDgtCrb1?usp=drive_link)
  - Training results [model-zoo]
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/t4base/v0.1/logs.zip)
    - [checkpoint_best.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/t4base/v0.2/epoch_50.pth)
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/t4base/v0.2/pillar_016_second_secfpn_2xb8_50m_base.py)
  - train time: NVIDIA A100 80GB * 2 * 50 epochs = 3.5 days
- Evaluation result with test-dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v3.0 + DB GSM8 v1.0 + DB J6 v1.0 (total frames: 1,394):
  - Total mAP (eval range = 52m): 0.774

| class_name | Count  | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ------ | ---- | ------- | ------- | ------- | ------- |
| car        | 41,133 | 83.2 | 78.0    | 84.2    | 85.3    | 85.5    |
| truck      | 8,890  | 67.5 | 47.2    | 69.5    | 76.0    | 77.3    |
| bus        | 3,275  | 79.4 | 63.7    | 82.2    | 85.0    | 86.7    |
| bicycle    | 3,635  | 82.3 | 81.4    | 82.5    | 82.6    | 82.6    |
| pedestrian | 25,981 | 74.4 | 72.7    | 73.9    | 75.1    | 76.1    |

</details>
