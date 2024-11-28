# autoware-ml

This repository is machine learning library for [Autoware](https://github.com/autowarefoundation/autoware) based on [OpenMMLab library](https://github.com/open-mmlab).
`autoware-ml` support training with [T4dataset format](https://github.com/tier4/tier4_perception_dataset) in addition to open dataset.

![](/docs/fig/autoware-ml.drawio.svg)

- Supported environment
  - All tools are tested by [Docker environment](Dockerfile) on Ubuntu 22.04LTS
  - NVIDIA dependency: CUDA 12.1 + cuDNN 8
    - Need > 530.xx.xx NVIDIA device driver

## Docs
### Design documents

If you want to know about the design of `autoware-ml`, you should read as below pages.

- [Docs for architecture of dataset pipeline](/docs/design/architecture_dataset.md)
- [Docs for architecture of ML model](/docs/design/architecture_model.md)
- [Docs for architecture of S3 storage](/docs/design/architecture_s3.md)
- [Docs for autoware-ml design](/docs/design/autoware_ml_design.md)

### Operation documents

If you want to develop `autoware-ml`, you should read as below pages.

- [Docs for contribution](/docs/operation/contribution.md)
- [Note for next release](/docs/operation/release_note.md)

### Tips

If you want to use `autoware-ml`, you should read as below pages.

- [Tips for config files](/docs/tips/config.md)
- [Tips for remote development](/docs/tips/remote_development.md)

## Supported tools

- [Setting environment for autoware-ml](/tools/setting_environment/)
- Training and evaluation
  - [Training and evaluation for 3D detection and 3D semantic segmentation](/tools/detection3d/)
  - [Training and evaluation for 2D detection](/tools/detection2d/)
  - (TBD) [Training and evaluation for 2D semantic segmentation](/tools/segmentation2d/)
- Analyze for the dataset and the model
  - [Analysis for 3d detection](/tools/analysus_3d)
  - [Visualization with rerun](/tools/rerun_visualization)
- [Select scene for rosbag](/tools/scene_selector/)
- (TBD) [Auto-label for 3D data](/tools/auto_labeling_3d/)
- (TBD) [Auto-label for 2D data](/tools/auto_labeling_2d/)

## Supported pipelines

- [Deploy 3D detection model](/pipelines/deploy_detection3d/)
- [Integration test](/pipelines/test_integration/)
- [Update T4dataset](/pipelines/update_t4dataset/)
- [Use autoware-ml with WebAuto](/pipelines/webauto/)
- (TBD) [Project adaptation](/pipelines/project_adaptation/)

## Supported model
### 3D detection

- Model for Autoware
  - [CenterPoint](projects/CenterPoint/)
  - [TransFusion](projects/TransFusion/)
- Model for ML tools
  - [BEVFusion](projects/BEVFusion/)

### 3D segmentation

- Model for Autoware
  - [FRNet](projects/FRNet/)

### 2D detection

- Model for Autoware
  - [YOLOX](projects/YOLOX/)
  - [YOLOX-opt](projects/YOLOX-opt/)
- Model for ML tools
  - [GLIP](projects/GLIP/)
  - [SwinTransformer](projects/SwinTransformer/)

### 2D segmentation

- Model for ML tools
  - (TBD) [SegmentAnything](projects/SegmentAnything/)

### 2D classification

- Model for Autoware
  - (TBD) [EfficientNet](projects/EfficientNet/)

### Vision language

- Model for ML tools
  - [BLIP-2](projects/BLIP-2/)

### Additional operation

- [SparseConvolutions](projects/SparseConvolution/)
