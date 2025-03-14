# ML pipeline for Autoware (a.k.a. AWML)

This repository is a machine learning library to deploy for [Autoware](https://github.com/autowarefoundation/autoware) to aim for "robotics MLOps".
`AWML` supports training with [T4dataset format](https://github.com/tier4/tier4_perception_dataset) in addition to open datasets and deployment for [Autoware](https://github.com/autowarefoundation/autoware).
In addition to ML model deployment, `AWML` supports active learning framework include auto labeling, semi-auto labeling, and data mining.

![](/docs/fig/AWML.drawio.svg)

`AWML` can deploy following task for now.

- 2D detection for dynamic recognition
- 3D detection for dynamic recognition
- 2D fine detection for traffic light recognition
- 2D classification for traffic light recognition

`AWML` supports following environment.

- All tools are tested by [Docker environment](Dockerfile) on Ubuntu 22.04LTS
- NVIDIA dependency: CUDA 12.1 + cuDNN 8
  - Need > 530.xx.xx NVIDIA device driver

## Docs
### Design documents

If you want to know about the design of `AWML`, you should read following pages.

- [Docs for architecture of dataset pipeline](/docs/design/architecture_dataset.md)
- [Docs for architecture of ML model](/docs/design/architecture_model.md)
- [Docs for architecture of S3 storage](/docs/design/architecture_s3.md)
- [Docs for AWML design](/docs/design/autoware_ml_design.md)

### Contribution

If you want to develop `AWML`, you should read following pages.

- [Docs for contribution](/docs/contribution/contribution.md)

If you want to search the OSS tools around `AWML`, you should read following pages.

- [Community support](/docs/contribution/community_support.md)

### Tips

If you want to know about `AWML`, you should read following pages.

- [Tips for config files](/docs/tips/config.md)
- [Tips for remote development](/docs/tips/remote_development.md)

### Release note

- [Release note](/docs/release_note/release_note.md)

## Supported tools

- [Setting environment for AWML](/tools/setting_environment/)
- Training and evaluation
  - [Training and evaluation for 3D detection and 3D semantic segmentation](/tools/detection3d/)
  - [Training and evaluation for 2D detection](/tools/detection2d/)
  - [Training and evaluation for 2D semantic segmentation](/tools/segmentation2d/)
- Analyze for the dataset and the model
  - [Analysis for 3D detection](/tools/analysis_3d)
  - [Analysis for 2D detection](/tools/analysis_2d)
  - [Visualization with rerun](/tools/rerun_visualization)
- Auto labeling
  - [Auto-label for 3D data](/tools/auto_labeling_3d/)
  - (TBD) [Auto-label for 2D data](/tools/auto_labeling_2d/)
- Data mining
  - [Select scene for rosbag](/tools/scene_selector/)
- ROS2
  - [Deploy to Autoware](/tools/deploy_to_autoware/)
  - (TBD) [ROS2 wrapper for AWML](/tools/autoware_ml_ros2/)

## Supported pipelines

- [Integration test for training and deployment](/pipelines/test_integration/)
- [Use AWML with WebAuto](/pipelines/webauto/)
- (TBD) [Project adaptation](/pipelines/project_adaptation/)

## Supported model
### 3D detection

- Model for Autoware
  - [CenterPoint](/projects/CenterPoint/)
  - [TransFusion](/projects/TransFusion/)
- Model for ML tools
  - [BEVFusion](/projects/BEVFusion/)

### 3D segmentation

- Model for Autoware
  - [FRNet](/projects/FRNet/)

### 2D detection

- Model for Autoware
  - [YOLOX](/projects/YOLOX/)
  - [YOLOX_opt](/projects/YOLOX_opt/)
- Model for ML tools
  - [GLIP](/projects/GLIP/)
  - [SwinTransformer](/projects/SwinTransformer/)

### 2D segmentation

- Model for ML tools
  - (TBD) [SegmentAnything](/projects/SegmentAnything/)

### 2D classification

- Model for Autoware
  - [MobileNetv2](/projects/MobileNetv2/)

### Vision language

- Model for ML tools
  - [BLIP-2](/projects/BLIP-2/)

### Additional operation

- [SparseConvolutions](/projects/SparseConvolution/)
- [ConvNeXt](/projects/ConvNeXt_PC/)

## Reference

- [Autoware](https://github.com/autowarefoundation/autoware)
  - Autoware is a registered trademark of the Autoware Foundation.
- [OpenMMLab](https://github.com/open-mmlab)
