# autoware-ml

This repository is machine learning library for [Autoware](https://github.com/autowarefoundation/autoware) based on [OpenMMLab library](https://github.com/open-mmlab).
`autoware-ml` support training with [T4dataset format](https://github.com/tier4/tier4_perception_dataset) in addition to open dataset.

## Docs

- Design documents
  - [Docs for whole architecture with software architecture](/docs/design/architecture.md)
  - [Docs for autoware-ml design](/docs/design/autoware_ml_design.md)
  - [Docs for T4dataset](/docs/design/t4dataset.md)
  - [Docs for config files](/docs/design/config.md)
  - [Docs for model deploy pipeline](/docs/design/model_deploy.md)
- Operation documents
  - [Docs for contribution](/docs/operation/contribution.md)
  - [Docs for dataset update](/docs/operation/update_dataset.md)
- [Release note](/docs/release_note.md)

## Environment
### Supported environment

- Tested by [Docker environment](Dockerfile) on Ubuntu 22.04LTS and Ubuntu 20.04LTS
- NVIDIA dependency: CUDA 12.1 + cuDNN 8
  - Need > 530.xx.xx NVIDIA device driver
- Library
  - [pytorch v2.2.0](https://github.com/pytorch/pytorch/tree/v2.2.0)
  - [mmcv v2.1.0](https://github.com/open-mmlab/mmcv/tree/v2.1.0)
  - [mmdetection3d v1.4.0](https://github.com/open-mmlab/mmdetection3d/tree/v1.4.0)
  - [mmdetection v3.3.0](https://github.com/open-mmlab/mmdetection/tree/v3.3.0)
  - [mmdeploy v1.3.1](https://github.com/open-mmlab/mmdeploy/tree/v1.3.1)

### Setup dataset

If you want to use open dataset like nuScenes dataset, you set dataset as [mmdetection3d documents](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/index.html) and [mmdetection documents](https://mmdetection.readthedocs.io/en/latest/user_guides/dataset_prepare.html).
If you want to [T4dataset](https://github.com/tier4/tier4_perception_dataset) and you have data access right of [WebAuto](https://docs.web.auto/en/user-manuals/), you can download T4dataset using [our scripts](/tools/download_t4dataset/).

### Setup environment

- Set environment

```sh
git clone https://github.com/tier4/autoware-ml
ln -s {path_to_dataset} data
```

```sh
├── data
│  └── nuscenes
│  └── t4dataset
│  └── nuimages
│  └── coco
├── Dockerfile
├── projects
├── README.md
└── work_dirs
```

- Build docker
  - Note that this process need for long time.

```sh
DOCKER_BUILDKIT=1 docker build -t autoware-ml .
```

## Supported tools

- [Training and evaluation for 3D detection](/tools/detection3d/)
- [Training and evaluation for 2D detection](/tools/detection2d/)
- [Download T4dataset](/tools/download_t4dataset/)

## Supported pipelines

- [Deploy 3D detection model](/pipelines/deploy_detection3d/)
- [Project adaptation for 3D detection model](/pipelines/project_adaptation_detection3d/)
- [Integration test](/pipelines/test_integration/)

## Supported model
### 3D detection

- Model
  - [BEVFusion](projects/BEVFusion/)
  - [TransFusion](projects/TransFusion/)
- Performance summary
  - Dataset: T4dataset database_v1_0 + database_v1_1 test dataset
  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m)

| model                      | range  | mAP | car | truck | bus | bicycle | pedestrian |
| -------------------------- | ------ | --- | --- | ----- | --- | ------- | ---------- |
| eval range: 50m            |        |     |     |       |     |         |            |
| TransFusion-L t4xx1_50m_v1 |        |     |     |       |     |         |            |
|                            |        |     |     |       |     |         |            |
| eval range: 90m            |        |     |     |       |     |         |            |
| TransFusion-L t4xx1_90m_v1 | 92.16m |     |     |       |     |         |            |
| TransFusion-L t4xx1_90m_v2 | 92.16m |     |     |       |     |         |            |

### 2D detection

- (TBD) YOLOX-opt
  - ROS package: [tensorrt_yolox](https://github.com/autowarefoundation/autoware.universe/tree/main/perception/tensorrt_yolox)
- (TBD) SwinTransformer
  - ROS package: Not supported
  - Supported model
    - Mask-RCNN with FPN model: This is used for BEVFusion image backbone

### 2D segmentation

- (TBD) SegmentAnything
  - ROS package: Not supported
  - This model is used for evaluation and labeling tools

### 2D classification

- (TBD) EfficientNet
  - ROS package: [traffic_light_classifier](https://github.com/autowarefoundation/autoware.universe/tree/main/perception/traffic_light_classifier)
