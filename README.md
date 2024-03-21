# Autoware-ML

This repository is ML library for Autoware.

## Docs

- [Get started](docs/get_started.md)

## Supported environment

- Tested by [Docker environment](Dockerfile) on Ubuntu22.04LTS
- NVIDIA dependency: CUDA 11 + cuDNN 8
- Library
  - pytorch: 1.9.0
  - mmcv v2.0.0rc4
  - [mmdetection3d v1.4.0](https://github.com/open-mmlab/mmdetection3d/tree/v1.4.0)
  - [mmdetection v3.3.0](https://github.com/open-mmlab/mmdetection/tree/v3.3.0)

## Supported feature
### Tools

- 3D
  - [x] Train code
  - [ ] Evaluation code
  - [ ] Make mp4 files for visualization
  - [ ] Make mcap files for visualization
- 2D
  - [ ] Train code
  - [ ] Evaluation code
  - [ ] Make mp4 files for visualization
  - [ ] Make mcap files for visualization

### 3D Detection model

- [BEVFusion (Camera-LiDAR fusion)](projects/BEVFusion)
- TransFusion (LiDAR only)
- CenterPoint (LiDAR only)
- PointPainted-CenterPoint (Camera-LiDAR fusion)

|                | Onnx deploy | T4dataset | NuScenes | aimotive |
| -------------- | :---------: | :-------: | :------: | :------: |
| BEVFusion-CL   |             |           |    ✅     |          |
| TransFusion-L  |             |           |          |          |
| CenterPoint    |             |           |          |          |
| PP-CenterPoint |             |           |          |          |

### 2D Detection

- YOLOX-opt
- TwinTransformer

|                 | Onnx deploy | T4dataset | NuImages | COCO  |
| --------------- | :---------: | :-------: | :------: | :---: |
| YOLOX-opt       |             |           |          |       |
| TwinTransformer |             |           |          |       |

### 2D classification

- EfficientNet

|              | Onnx deploy | T4dataset | NuImages | COCO  |
| ------------ | :---------: | :-------: | :------: | :---: |
| EfficientNet |             |           |          |       |

