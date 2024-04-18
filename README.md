# autoware-ml

This repository is machine learning library for [Autoware](https://github.com/autowarefoundation/autoware).

## Docs

- [Get started for 3D detection](docs/get_started_3d_detection.md)
- [Get started for 2D detection](docs/get_started_2d_detection.md)
- [Design docs](docs/design.md)
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

## Supported feature
### 3D Detection model

- [BEVFusion-CL (BEVFusion Camera-LiDAR fusion)](projects/BEVFusion)
- TransFusion-L (TransFusion LiDAR only)
- CenterPoint (LiDAR only)
- PointPainted-CenterPoint (Camera-LiDAR fusion)

|                | T4dataset | NuScenes | Argoverse2 | Aimotive |
| -------------- | :-------: | :------: | :--------: | :------: |
| BEVFusion-CL   |           |          |            |          |
| TransFusion-L  |           |          |            |          |
| CenterPoint    |           |          |            |          |
| PP-CenterPoint |           |          |            |          |

### 2D Detection

- YOLOX-opt
- TwinTransformer

|                 | T4dataset | COCO  | NuImages |
| --------------- | :-------: | :---: | :------: |
| YOLOX-opt       |           |       |          |
| TwinTransformer |           |       |          |

### 2D classification

- EfficientNet

|              | T4dataset | COCO  | NuImages |
| ------------ | :-------: | :---: | :------: |
| EfficientNet |           |       |          |
