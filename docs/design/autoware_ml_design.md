# `autoware-ml` design

## Pipeline design
### Overview diagram

`autoware-ml` is designed for deployment from training and evaluation with Autoware and active learning framework.

![](/docs/fig/autoware_ml_pipeline.drawio.svg)

### The pipeline design of `autoware-ml`

![](/docs/fig/pipeline.drawio.svg)

- Orange line: the deployment flow for ML model.

The deployment pipeline consists as below.

> 1. create_data_info.py

Make info file with annotated data.

> 2. train.py, 3. test.py, 4. visualize.py

From config file, the ML model is trained and evaluated.

> 5. deploy.py

If the model is used for Autoware, the model is deployed to onnx file.

- Green line: the pipeline for active learning

> 6. pseudo_label.py

Make info file from non-annotated T4dataset for 2D and 3D.

> 7. create_pseudo_t4dataset.py

Make pseudo-label T4dataset from the info file which is based on pseudo label.

> 8. choose_annotation.py

Choose annotation from the info file of labels.
For example, it is used for `scene_selector` and `pseudo_label` to tune the parameter of the threshold of confidence with offline model.

## Supported environment

- [pytorch v2.2.0](https://github.com/pytorch/pytorch/tree/v2.2.0)

`autoware-ml` is based on pytorch.

- [mmdetection3d v1.4](https://github.com/open-mmlab/mmdetection3d/tree/v1.4.0).

This is machine learning framework for 3D detection.
`autoware-ml` is strongly based on this framework.

If you want to learn about use of `mmdetection3d`, we recommend to read [user guides](https://mmdetection3d.readthedocs.io/en/latest/user_guides/index.html) at first.
If you want to learn about config files of `mmdetection3d`, we recommend to read [user guides for configs](https://mmdetection3d.readthedocs.io/en/latest/user_guides/config.html).
If you want to learn about info files of  `mmdetection3d`, we recommend to read [nuscenes dataset](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/nuscenes.html?highlight=info).

- [mmdetection v3.2.0](https://github.com/open-mmlab/mmdetection/tree/v3.2.0)

This is machine learning framework for 2D detection.

- [mmcv v2.1.0](https://github.com/open-mmlab/mmcv/tree/v2.1.0)
- [mmdeploy v1.3.1](https://github.com/open-mmlab/mmdeploy/tree/v1.3.1)

These are core library for MMLab libraries.
If you want to develop `autoware-ml`, we recommend to read the documents of these.

## `autoware-ml` architecture
### autoware_ml/

The directory of `autoware_ml` is library for autoware-ml.
This directory can be used as library from other software and this directory doesn't depend on other directories.

- `autoware_ml/detection3d`

It provides the core library of 3D detection.
It contain loader and metrics for T4dataset.

- `autoware_ml/configs`

The config files in `autoware_ml` is used commonly for each projects.

```
- autoware_ml/
  - configs/
    - detection3d/
      - XX1.py
      - X2.py
      - database_v1_0.yaml
      - database_v1_1.yaml
      - database_v1_3.yaml
    - detection2d/
      - XX1.py
      - X2.py
      - database_v1_0.yaml
      - database_v1_1.yaml
      - tlr_v1_0.yaml
```

- dataset configs: `autoware_ml/configs/*.yaml`

The file like `detection3d/database_v1_0.yaml` defines dataset ids of T4dataset.
It contains document about T4dataset as belows.

```yaml
docs: |
  Product: XX1
  Place: Odaiba, Nishi-Shinjuku, and Shiojiri
  Amount: About 12000 frames
  Sensor: Velodyne LiDAR + BFS Camera
  Annotation: All the data are collected at 10Hz and most of them are annotated at 2Hz and DBv1.0_nishi_shinjuku_[0-7]_ are annotated at 10Hz
```

We define T4dataset version as below.

```
- version: major.minor.build
  - major: sensor configuration
  - minor: dataset scenes
  - build: dataset version
```

### docs/

The directory of `docs/` is design documents for `autoware-ml`.
The target of documents is a designer of whole ML pipeline system and developers of `autoware-ml` core library.

### pipelines/

The directory of `pipelines/` manages the pipelines that consist of `tools`.
This directory can depend on `/autoware_ml`, `projects`, `/tools`, and other `/pipelines`.

Each pipeline has `README.md`, a process document to use when you ask someone else to do the work.
The target of `README.md` is a user of `autoware-ml`.

### projects/

The directory of `projects/` manages the model for each tasks.
This directory can depend on `/autoware_ml` and other `projects`.

```
- projects/
  - BEVFusion
  - YOLOX
  - TransFusion
```

Each project has `README.md` for users.
The target of `README.md` is a user of `autoware-ml`.

### tools/

The directory of `tools/` manages the tools for each tasks.
`tools/` scripts are abstracted. For example, `tools/detection3d` can be used for any 3D detection models such as TransFusion and BEVFusion.
This directory can depend on `/autoware_ml` and other `/tools`.


```
- tools/
  - detection3d/
  - detection2d/
  - update_t4dataset/
```

Each tool has `README.md` for developers.
The target of `README.md` is a developer of `autoware-ml`.

## Support priority

We define "support priority" for each tools and projects. Maintainers handle handle the issues according to the priority as below.

- Tier S:
  - It is core function in ML system and it updates frequently from any requests.
  - We strive to make it high maintainability with code quality, unit test and CI/CD.
  - We put highest priority on support and maintenance to it because it leads to fast cycle development for developers.
- Tier A:
  - It is maintenance phase on development, so it's updates is not frequently.
  - We make it maintainability with code quality, unit test and CI/CD, if possible.
  - We put a high priority on support to it.
- Tier B:
  - We fix a broken tool when needed.
  - We put a middle priority on support to it.
- Tier C:
  - We rarely use a tool or a model or it is just prototype version.
  - If it is not used for long time, we delete it.
  - We put a low priority on support to it.

## Versioning strategy for `autoware-ml`

We follow basically [semantic versioning](https://semver.org/).
As our strategy, we follow as below.

- Major version zero (0.y.z) is for initial development.
  - The public API should not be considered stable.
- Major version X (X.y.z | X > 0) is incremented if any backward incompatible changes are introduced to the public API.
- Minor version Y (x.Y.z | x > 0) is incremented if new, backward compatible functionality is introduced to the public API.
  - It is incremented if any public API functionality is marked as deprecated.
  - It is incremented if a new project is added.
- Patch version Z (x.y.Z | x > 0) is incremented if only backward compatible bug fixes are introduced.
  - A bug fix is defined as an internal change that fixes incorrect behavior.
  - It is incremented if a new model is released.
