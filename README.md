# autoware-ml

This repository is machine learning library for [Autoware](https://github.com/autowarefoundation/autoware).

## Docs

- [Get started for 3D detection](docs/get_started_3d_detection.md)
- [Get started for 2D detection](docs/get_started_2d_detection.md)
- [Design docs](docs/design.md): If you want to know the design of `autoware-ml`, please see this document
- [Contribution](docs/contribution.md): If you want to contribute, please see this document
- [Release note](docs/release_note.md)

## Supported environment

- Tested by [Docker environment](Dockerfile) on Ubuntu 22.04LTS
- NVIDIA dependency: CUDA 12.1 + cuDNN 8
- Library
  - [pytorch v2.2.0](https://github.com/pytorch/pytorch/tree/v2.2.0)
  - [mmcv v2.1.0](https://github.com/open-mmlab/mmcv/tree/v2.1.0)
  - [mmdetection3d v1.4.0](https://github.com/open-mmlab/mmdetection3d/tree/v1.4.0)
  - [mmdetection v3.3.0](https://github.com/open-mmlab/mmdetection/tree/v3.3.0)
  - [mmdeploy v1.3.1](https://github.com/open-mmlab/mmdeploy/tree/v1.3.1)

## Supported model
### 3D detection

- [BEVFusion](projects/BEVFusion)
  - ROS package: TBD
  - Supported model
    - Camera-LiDAR fusion model (spconv)
- [TransFusion](projects/TransFusion)
  - ROS package: TBD
  - Supported model
    - LiDAR-only model (pillar)

|             | T4dataset | NuScenes |
| ----------- | :-------: | :------: |
| BEVFusion   |           |          |
| TransFusion |     ✅     |    ✅     |

### 2D detection

- (TBD) YOLOX-opt
  - ROS package: [tensorrt_yolox](https://github.com/autowarefoundation/autoware.universe/tree/main/perception/tensorrt_yolox)
- (TBD) TwinTransformer
  - ROS package: Not supported
  - Supported model
    - Mask-RCNN with FPN model: This is used for BEVFusion image backbone

|                 | T4dataset | COCO  | NuImages |
| --------------- | :-------: | :---: | :------: |
| YOLOX-opt       |           |       |          |
| TwinTransformer |           |       |          |

### 2D segmentation

- (TBD) SegmentAnything
  - ROS package: Not supported
  - This model is used for evaluation and labeling tools

### 2D classification

- (TBD) EfficientNet
  - ROS package: [traffic_light_classifier](https://github.com/autowarefoundation/autoware.universe/tree/main/perception/traffic_light_classifier)

|              | T4dataset | COCO  | NuImages |
| ------------ | :-------: | :---: | :------: |
| EfficientNet |           |       |          |
