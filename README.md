# autoware-ml

This repository is machine learning library for [Autoware](https://github.com/autowarefoundation/autoware) based on [OpenMMLab library](https://github.com/open-mmlab).
`autoware-ml` support training with [T4dataset format](https://github.com/tier4/tier4_perception_dataset) in addition to open dataset.

![](/docs/fig/autoware_ml_pipeline.drawio.svg)

- Supported environment
  - All tools are tested by [Docker environment](Dockerfile) on Ubuntu 22.04LTS
  - NVIDIA dependency: CUDA 12.1 + cuDNN 8
    - Need > 530.xx.xx NVIDIA device driver

## Docs
### Design documents

If you want to know about the design of `autoware-ml`, you should read as below pages.

- [Docs for whole architecture](/docs/design/architecture.md)
- [Docs for autoware-ml design](/docs/design/autoware_ml_design.md)
- [Docs for T4dataset](/docs/design/t4dataset.md)

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
- [Training and evaluation for 3D detection and 3D semantic segmentation](/tools/detection3d/)
- [Training and evaluation for 2D detection](/tools/detection2d/)
- (TBD) [Training and evaluation for 2D semantic segmentation](/tools/segmentation2d/)
- [Visualization with rerun](/tools/rerun_visualization)
- (TBD) [Select scene for rosbag](/tools/scene_selector/)
- (TBD) [Pseudo label for T4dataset](/tools/t4dataset_pseudo_label/)

## Supported pipelines

- [Deploy 3D detection model](/pipelines/deploy_detection3d/)
- [Project adaptation for 3D detection model](/pipelines/project_adaptation_detection3d/)
- [Integration test](/pipelines/test_integration/)
- [Update T4dataset](/pipelines/update_t4dataset/)

## Supported model
### 3D detection

- Model for Autoware
  - [TransFusion](projects/TransFusion/)
- Model for ML tools
  - [BEVFusion](projects/BEVFusion/)
- Performance summary

| eval range: 90m            | Eval DB | range  | mAP  | car  | truck | bus  | bicycle | pedestrian |
| -------------------------- | ------- | ------ | ---- | ---- | ----- | ---- | ------- | ---------- |
| TransFusion-L t4xx1_90m/v3 | 1.0+1.1 | 92.16m | 68.5 | 81.7 | 62.4  | 83.6 | 50.9    | 64.1       |

| eval range: 120m                  | Eval DB | range   | mAP  | car  | truck | bus  | bicycle | pedestrian |
| --------------------------------- | ------- | ------- | ---- | ---- | ----- | ---- | ------- | ---------- |
| TransFusion-L t4xx1_120m/v1       | 1.0+1.1 | 122.88m | 51.8 | 70.9 | 43.8  | 54.7 | 38.7    | 50.9       |
| BEVFusion-L t4xx1_120m/v1         | 1.0+1.1 | 122.4m  | 60.6 | 74.1 | 54.1  | 58.7 | 55.7    | 60.3       |
| BEVFusion-L t4offline_120m/v1     | 1.0+1.1 | 122.4m  | 65.7 | 76.2 | 56.4  | 65.6 | 65.0    | 65.3       |
| BEVFusion-CL t4xx1_fusion_120m/v1 | 1.0+1.1 | 122.4m  | 63.7 | 72.4 | 56.8  | 71.8 | 62.1    | 55.3       |

### 3D segmentation

- Model for Autoware
  - [FRNet](projects/FRNet/)

### 2D detection

- Model for Autoware
  - [YOLOX](projects/YOLOX/)
  - (TBD) [YOLOX-opt](projects/YOLOX-opt/)
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
