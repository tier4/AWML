# Autoware-ML

This repository is ML library for Autoware.

## Docs

- [Get started](docs/get_started.md)
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
### Tools

- 3D
  - [ ] Make mp4 files for visualization
  - [ ] Make mcap files for visualization
- 2D
  - [ ] Make mp4 files for visualization
  - [ ] Make mcap files for visualization

### 3D Detection model

- [BEVFusion-CL (BEVFusion Camera-LiDAR fusion)](projects/BEVFusion)
- TransFusion-L (TransFusion LiDAR only)
- CenterPoint (LiDAR only)
- PointPainted-CenterPoint (Camera-LiDAR fusion)

|                | T4dataset | NuScenes | argoverse2 |
| -------------- | :-------: | :------: | :--------: |
| BEVFusion-CL   |           |          |            |
| TransFusion-L  |           |    ✅     |            |
| CenterPoint    |           |          |            |
| PP-CenterPoint |           |          |            |

### 2D Detection

- YOLOX-opt
- TwinTransformer

|                 | T4dataset | NuImages | COCO  |
| --------------- | :-------: | :------: | :---: |
| YOLOX-opt       |           |          |       |
| TwinTransformer |           |          |       |

### 2D classification

- EfficientNet

|              | T4dataset | NuImages | COCO  |
| ------------ | :-------: | :------: | :---: |
| EfficientNet |           |          |       |
