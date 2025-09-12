# ML pipeline for Autoware (a.k.a. AWML)

|        | URL                              |
| ------ | -------------------------------- |
| Github | https://github.com/tier4/AWML    |
| arXiv  | https://arxiv.org/abs/2506.00645 |

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

If you use this project in your research, please use [the arXiv paper of AWML](https://arxiv.org/abs/2506.00645) and [the arXiv paper for fine-tuning strategy](https://arxiv.org/abs/2509.04711).
And please use consider citation as below.

```
@misc{tanaka2025awmlopensourcemlbasedrobotics,
      title={AWML: An Open-Source ML-based Robotics Perception Framework to Deploy for ROS-based Autonomous Driving Software},
      author={Satoshi Tanaka and Samrat Thapa and Kok Seang Tan and Amadeusz Szymko and Lobos Kenzo and Koji Minoda and Shintaro Tomie and Kotaro Uetake and Guolong Zhang and Isamu Yamashita and Takamasa Horibe},
      year={2025},
      eprint={2506.00645},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.00645},
}

@misc{tanaka2025domainadaptationdifferentsensor,
      title={Domain Adaptation for Different Sensor Configurations in 3D Object Detection},
      author={Satoshi Tanaka and Kok Seang Tan and Isamu Yamashita},
      year={2025},
      eprint={2509.04711},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.04711},
}
```

`AWML` is based on [Autoware Core & Universe strategy](https://autoware.org/autoware-overview/).
We hope that `AWML` promotes the community between Autoware and ML researchers and engineers.

![](/docs/fig/autoware_ml_community.drawio.svg)

|                | URL                                            |
| -------------- | ---------------------------------------------- |
| Autoware       | https://github.com/autowarefoundation/autoware |
| AWMLPrediction | TBD                                            |

## Docs
### Design documents

If you want to know about the design of `AWML`, you should read following pages.

- [Docs for architecture of dataset pipeline](/docs/design/architecture_dataset.md)
- [Docs for architecture of ML model](/docs/design/architecture_model.md)
- [Docs for architecture of S3 storage](/docs/design/architecture_s3.md)
- [Docs for architecture of evaluation](/docs/design/architecture_evaluation.md)
- [Docs for AWML design](/docs/design/autoware_ml_design.md)
- [Docs for deploy flow design](/docs/design/deploy_pipeline_design.md)

### Contribution

If you want to develop `AWML`, you should read following pages.

- [Docs for contribution](/docs/contribution/contribution.md)

If you want to search the OSS tools around `AWML`, you should read following pages.

- [Community support](/docs/contribution/community_support.md)

### Tips

If you want to know about `AWML`, you should read following pages.

- [Tips for how to follow development on AWML](/docs/tips/how_to_follow.md)
- [Tips for config files](/docs/tips/config.md)
- [Tips for remote development](/docs/tips/remote_development.md)

### Release note

- [Release note](/docs/release_note/release_note.md)

## Supported tools

- [Setting environment for AWML](/tools/setting_environment/)
- Training, evaluation, and deployment
  - [Train and evaluate 3D detection model and 3D semantic segmentation model](/tools/detection3d/)
  - [Train and evaluate 2D detection model](/tools/detection2d/)
  - [Train and evaluate 2D semantic segmentation model](/tools/segmentation2d/)
- Analyze for the dataset and the model
  - [Analyze for 3D detection dataset](/tools/analysis_3d)
  - [Analyze for model performance](/tools/performance_tools/)
  - [Visualize in 3D spaces with rerun](/tools/rerun_visualization)
- Auto labeling
  - [Auto-label for 3D data](/tools/auto_labeling_3d/)
  - [Auto-label for 2D data (TBD)](/tools/auto_labeling_2d/)
- Data mining
  - [Calibration classification](/tools/calibration_classification/)
  - [Select scene for rosbag](/tools/scene_selector/)
- ROS2
  - [Deploy to Autoware](/tools/deploy_to_autoware/)
  - [ROS2 wrapper for AWML (TBD)](/tools/autoware_ml_ros2/)

## Supported pipelines

- [Use AWML with WebAuto](/pipelines/webauto/)

## Supported model

- Supported models
  - :star: is recommended to use

| Task              | Model                                         | Use for Autoware   |
| ----------------- | --------------------------------------------- | ------------------ |
| 3D detection      | [CenterPoint](/projects/CenterPoint/)         | :star:             |
| 3D detection      | [TransFusion](/projects/TransFusion/)         | :white_check_mark: |
| 3D detection      | [BEVFusion](/projects/BEVFusion/)             | :white_check_mark: |
| 3D segmentation   | [FRNet](/projects/FRNet/)                     | (Reviewing now)    |
| 2D detection      | [YOLOX](/projects/YOLOX/)                     |                    |
| 2D detection      | [YOLOX_opt](/projects/YOLOX_opt/)             | :star:             |
| 2D detection      | [GLIP](/projects/GLIP/)                       |                    |
| 2D detection      | [SwinTransformer](/projects/SwinTransformer/) |                    |
| 2D classification | [MobileNetv2](/projects/MobileNetv2/)         | :white_check_mark: |
| Vision language   | [BLIP-2](/projects/BLIP-2/)                   |                    |
|                   |                                               |                    |

- Additional plug-ins
  - [SparseConvolutions](/projects/SparseConvolution/)
  - [ConvNeXt](/projects/ConvNeXt_PC/)

## Reference

- [Autoware](https://github.com/autowarefoundation/autoware)
  - Autoware is a registered trademark of the Autoware Foundation.
- [OpenMMLab](https://github.com/open-mmlab)
