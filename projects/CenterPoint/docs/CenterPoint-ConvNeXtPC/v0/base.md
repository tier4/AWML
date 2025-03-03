# Deployed model for CenterPoint-ConvNeXtPC base/0.X
## Summary

- Main parameter
  - range = 121.60m
  - voxel_size = [0.20, 0.20, 8.0]
	- out_size_factor = 4
  - grid_size = [1216, 1216, 1]
- Performance summary
  - Dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v3.0 + DB GSM8 v1.0 + DB J6 v1.0 (total frames: 35,292)
  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m)

| eval range: 120m                | mAP  | car <br> (629,212) | truck <br> (163,402) | bus <br> (39,904) | bicycle <br> (48,043) | pedestrian <br> (383,553) |
| ------------------------------- | ---- | ------------------ | -------------------- | ----------------- | --------------------- | ------------------------- |
| CenterPoint-ConvNeXtPC base/0.1 | 65.3 | 75.0               | 54.2                 | 78.1              | 52.8                  | 66.2                      |
| CenterPoint-ConvNeXtPC base/0.2 | 68.6 | 77.9               | 58.6                 | 80.9              | 56.4                  | 69.3                      |


## Release
### CenterPoint ConvNeXt base/0.2

- This is same as `CenterPoint-ConvNeXt base/0.1` model except the arhictecture is much bigger by adding more depths and channels
- Specifically, the main changes in hyperparameters are:

  - Architecture:
	```python
  		pts_backbone=dict(
        	_delete_=True,
        	type="ConvNeXt_PC",
        	in_channels=32,
        	out_channels=[32, 192, 192, 192, 192],
        	depths=[3, 3, 2, 1, 1],
        	out_indices=[2, 3, 4],
        	drop_path_rate=0.4,
        	layer_scale_init_value=1.0,
        	gap_before_final_norm=False,
        	with_cp=True,  # We set with_cp to True for svaing gpu memory
    	),
  		pts_neck=dict(
        	type="SECONDFPN",
        	in_channels=[192, 192, 192],
        	out_channels=[128, 128, 128],
        	upsample_strides=[1, 2, 4],
        	norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
        	upsample_cfg=dict(type="deconv", bias=False),
        	use_conv_for_no_stride=True
    	)

- Evaluation data is the same as `CenterPoint base/1.0`, which is TIER IV's in-house dataset. The improvement of mAP in `ConvNeXt` (0.686 vs 0.644) demostrates a promising direction with a stronger backbone
- Based on the performance, it looks promising if we would like to try larger settings
- In deployment of `autoware`:
    - Please set `workspace_size` to more than `2GB` when running tensorrt optimization  
    - SECOND vs ConvNeXt:
      - Max GPU (MB): 4,885 vs 6,087
      - Mean GPU (MB): 4,721 vs 5,926
      - Mean processing time (ms): 18.22 vs 48.19
      - Max processing time (ms): 45.54 vs 106.72
- In conclusion, it generally shows better performance across all categories and thresholds compared to `CenterPoint-ConvNeXt base/0.1` and `CenterPoint base/1.0`, with particularly large improvements in truck and pedestrian categories

<details>
<summary> The link of data and evaluation result </summary>

- Comparison for Base Detection:

| Model (120m)                  | mAP         | Time    | Memory |
| ----------------------------- | ----------- | ------- | ------ |
| CenterPoint base/1.0          | 64.4        | 18.2 ms | 4.7 GB |
| CenterPoint-ConvNeXt base/0.1 | 65.3 (+0.9) | 30.4 ms | 5.6 GB |
| CenterPoint-ConvNeXt base/0.2 | 68.6 (+4.2) | 48.1 ms | 5.9 GB |

- Comparison for Nearby Models:

| Model (50m)                   | Time   | Memory |
| ----------------------------- | ------ | ------ |
| CenterPoint base/1.0          | 5.9 ms | 4.5 GB |
| CenterPoint-ConvNeXt base/0.1 | 7.2 ms | 3.9 GB |
| CenterPoint-ConvNeXt base/0.2 | 9.9 ms | 3.6 GB |

- Model
  - Training dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v3.0 + DB GSM8 v1.0 + DB J6 v1.0 (total frames: 35,392)
  - [Config file path](https://github.com/tier4/AWML/blob/9aead19d7f42b711fba8ffa38ec82ce135300617/projects/CenterPoint/configs/t4dataset/pillar_020_convnext_standard_secfpn_4xb8_121m_base.py)
  - Deployed onnx model and ROS parameter files [[Google drive (for internal)]](https://drive.google.com/drive/folders/1dhthG6LtbZTS91U2OQvd5iQ1q58b2ZEH)
  - Deployed onnx and ROS parameter files [[model-zoo]]
    - [detection_class_remapper.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-convnextpc/t4base/v0.2/detection_class_remapper.param.yaml)
    - [centerpoint_x2_ml_package.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-convnextpc/t4base/v0.2/centerpoint_x2_ml_package.param.yaml)
    - [deploy_metadata.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-convnextpc/t4base/v0.2/deploy_metadata.yaml)
    - [pts_voxel_encoder_centerpoint_x2.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-convnextpc/t4base/v0.2/pts_voxel_encoder_centerpoint_x2.onnx)
    - [pts_backbone_neck_head_centerpoint_x2.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-convnextpc/t4base/v0.2/pts_backbone_neck_head_centerpoint_x2.onnx)
  - Training results [[Google drive (for internal)]](https://drive.google.com/drive/folders/1dhthG6LtbZTS91U2OQvd5iQ1q58b2ZEH)
  - Training results [model-zoo]
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-convnextpc/t4base/v0.2/logs.zip)
    - [checkpoint_best.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-convnextpc/t4base/v0.2/epoch_30.pth)
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-convnextpc/t4base/v0.2/dev_pillar_020_convnetxt_small_secfpn_4xb8_121m_base.py)
  - train time: NVIDIA A100 80GB * 4 * 30 epochs = 3.5 days
- Evaluation result with test-dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v3.0 + DB GSM8 v1.0 + DB J6 v1.0 (total frames: 1,394):
  - Total mAP (eval range = 120m): 0.686

| class_name | Count  | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ------ | ---- | ------- | ------- | ------- | ------- |
| car        | 41,133 | 77.9 | 79.8    | 82.2    | 83.0    | 79.5    |
| truck      | 8,890  | 58.6 | 34.7    | 59.7    | 67.7    | 72.2    |
| bus        | 3,275  | 80.9 | 69.2    | 79.6    | 81.1    | 82.6    |
| bicycle    | 3,635  | 53.2 | 52.3    | 53.4    | 53.5    | 53.6    |
| pedestrian | 25,981 | 64.8 | 62.4    | 64.0    | 65.4    | 67.4    |

- Evaluation for XX1
	- Same as CenterPoint ConvNeXtPC base/0.2
	- Evaluation result with test-dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v4.0 (total frames: 1,507)
  - Total mAP (eval range = 120m): 0.678

| class_name | Count  | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ------ | ---- | ------- | ------- | ------- | ------- |
| car        | 16,126 | 77.1 | 61.3    | 80.1    | 83.0    | 84.0    |
| truck      | 4,578  | 57.4 | 34.3    | 58.0    | 66.0    | 71.2    |
| bus        | 1,457  | 73.9 | 58.3    | 75.0    | 79.7    | 82.5    |
| bicycle    | 1,040  | 61.5 | 56.6    | 62.5    | 63.2    | 63.6    |
| pedestrian | 11,971 | 69.1 | 65.7    | 70.1    | 70.1    | 72.4    |

- Evaluation for X2
	- Evaluation result with test-dataset: DB GSM8 v1.0 + DB J6 v1.0 (total frames: 1,280):
  - Total mAP (eval range = 120m): 0.686

| class_name | Count  | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ------ | ---- | ------- | ------- | ------- | ------- |
| car        | 25,007 | 78.5 | 69.9    | 79.9    | 82.0    | 82.4    |
| truck      | 4,573  | 55.5 | 32.0    | 55.9    | 64.1    | 69.9    |
| bus        | 1,818  | 86.4 | 76.0    | 88.9    | 89.8    | 90.6    |
| bicycle    | 2,567  | 54.3 | 53.6    | 54.5    | 54.5    | 54.6    |
| pedestrian | 14,010 | 69.3 | 67.1    | 68.3    | 69.9    | 72.1    |

</details>

### CenterPoint-ConvNeXt base/0.1
- This the CenterPoint model based on `CenterPoint base/1.0` by replacing the SECOND backbone by [ConvNeXt backbone](https://github.com/WayneMao/PillarNeSt/tree/main)
- It uses higher resolution for voxelization and downsamples the heatmap to reduce gpu memory
- It enables gradient checkpoint to save more gpu memory at the **cost of training time**
- Note that it **doesn't** use any pretrained weights
- Specifically, the main changes in hyperparameters are:
  - General:

	```python
		voxel_size: [0.20, 0.20, 8.0]
  		out_size_factor = 4
  		data_preprocessor: dict(
        	type="Det3DDataPreprocessor",
        	voxel=True,
        	voxel_layer=dict(
            	max_num_points=32,
            	voxel_size=voxel_size,
            	point_cloud_range=point_cloud_range,
            	max_voxels=(48000, 60000),
            	deterministic=True,
        	),
      	)
	```

  - Architecture:
	```python
  		pts_backbone=dict(
        	_delete_=True,
        	type="ConvNeXt_PC",
        	in_channels=32,
        	out_channels=[32, 64, 128, 128, 128],
        	depths=[3, 2, 1, 1, 1],
        	out_indices=[2, 3, 4],
        	drop_path_rate=0.0,	# No drop path
        	layer_scale_init_value=1.0,
        	gap_before_final_norm=False,
        	with_cp=True,  # We set with_cp to True for svaing gpu memory
    	),
  		pts_neck=dict(
        	type="SECONDFPN",
        	in_channels=[128, 128, 128],
        	out_channels=[128, 128, 128],
        	upsample_strides=[1, 2, 4],
        	norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
        	upsample_cfg=dict(type="deconv", bias=False),
        	use_conv_for_no_stride=True
    	)
- The improvement of mAP only slightly better than `CenterPoint base/1.0` (0.653 vs 0.644), with particularly truck and pedestrian categories
- In deployment of `autoware`:
    - Please set `workspace_size` to more than `2GB` when running tensorrt optimization  
    - SECOND vs ConvNeXt:
      - Max GPU (MB): 4,885 vs 5,785
      - Mean GPU (MB): 4,721 vs 5,605
      - Mean processing time (ms): 18.22 vs 30.42
      - Max processing time (ms): 45.54 vs 45.98

<details>
<summary> The link of data and evaluation result </summary>

- Model
  - Training dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v3.0 + DB GSM8 v1.0 + DB J6 v1.0 (total frames: 35,392)
  - [Config file path](https://github.com/tier4/AWML/blob/9aead19d7f42b711fba8ffa38ec82ce135300617/projects/CenterPoint/configs/t4dataset/pillar_020_convnext_small_secfpn_4xb8_121m_base.py)
  - Deployed onnx model and ROS parameter files [[Google drive (for internal)]](https://drive.google.com/drive/folders/1WuP0jo1j6HeAVGf5KehNjb5e9SiLdmgb)
  - Deployed onnx and ROS parameter files [[model-zoo]]
    - [detection_class_remapper.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-convnextpc/t4base/v0.1/detection_class_remapper.param.yaml)
    - [centerpoint_x2_ml_package.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-convnextpc/t4base/v0.1/centerpoint_x2_ml_package.param.yaml)
    - [deploy_metadata.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-convnextpc/t4base/v0.1/deploy_metadata.yaml)
    - [pts_voxel_encoder_centerpoint_x2.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-convnextpc/t4base/v0.1/pts_voxel_encoder_centerpoint_x2.onnx)
    - [pts_backbone_neck_head_centerpoint_x2.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-convnextpc/t4base/v0.1/pts_backbone_neck_head_centerpoint_x2.onnx)
  - Training results [[Google drive (for internal)]](https://drive.google.com/drive/folders/1s4V07dXHO5jayExdho3qnP02yZxvp0gY)
  - Training results [model-zoo]
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-convnextpc/t4base/v0.1/logs.zip)
    - [checkpoint_best.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-convnextpc/t4base/v0.1/epoch_30.pth)
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-convnextpc/t4base/v0.1/pillar_020_convnext_small_secfpn_4xb8_121m_base.py)
  - train time: NVIDIA A100 80GB * 4 * 30 epochs = 2 days
- Evaluation result with test-dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v3.0 + DB GSM8 v1.0 + DB J6 v1.0 (total frames: 1,394):
  - Total mAP (eval range = 120m): 0.653

| class_name | Count  | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ------ | ---- | ------- | ------- | ------- | ------- |
| car        | 41,133 | 75.0 | 63.0    | 76.9    | 79.6    | 80.5    |
| truck      | 8,890  | 54.2 | 29.9    | 55.4    | 63.7    | 68.0    |
| bus        | 3,275  | 78.1 | 58.3    | 81.8    | 85.3    | 87.1    |
| bicycle    | 3,635  | 52.8 | 50.9    | 53.3    | 53.4    | 53.5    |
| pedestrian | 25,981 | 66.2 | 63.5    | 65.4    | 66.8    | 68.9    |

</details>
