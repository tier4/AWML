# `autoware-ml` design

## Pipeline design

![](/docs/fig/pipeline.drawio.svg)

## Supported environment

- [pytorch v2.2.0](https://github.com/pytorch/pytorch/tree/v2.2.0)

`autoware-ml` is based on pytorch.

- [mmdetection3d v1.4](https://github.com/open-mmlab/mmdetection3d/tree/v1.4.0).

This is machine learning framework for 3D detection.
`autoware-ml` is strongly based on this framework.

If you want to learn about use of `mmdetection3d`, we recommend to read [user guides](https://mmdetection3d.readthedocs.io/en/latest/user_guides/index.html) at first.
If you want to learn about config files of `mmdetection3d`, we recommend to read [user guides for configs](https://mmdetection3d.readthedocs.io/en/latest/user_guides/config.html).
If you want to learn about info files of  `mmdetection3d`, we recommend to read [nuscenes dataset](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/nuscenes.html?highlight=info).

- [mmdetection v3.3.0](https://github.com/open-mmlab/mmdetection/tree/v3.3.0)
- [mmcv v2.1.0](https://github.com/open-mmlab/mmcv/tree/v2.1.0)
- [mmdeploy v1.3.1](https://github.com/open-mmlab/mmdeploy/tree/v1.3.1)

## `autoware-ml` architecture
### autoware_ml/

The directory of `autoware_ml` is library for autoware-ml.
This directory can be used as library from other software.

- configs

The config files in `autoware_ml` is used commonly for each projects.

```
- autoware_ml/
  - configs/
    - detection3d/
      - XX1.py
      - X2.py
    - detection2d/
      - XX1.py
      - X2.py
```

### docs/

The directory of `docs/` is design documents for `autoware-ml`.
The target of documents is a designer of whole ML pipeline system and developers of `autoware-ml` core library.

### pipelines/

The directory of `pipelines/` manages the pipelines that consist of `tools`.

Each pipeline has `README.md`, a process document to use when you ask someone else to do the work.
The target of `README.md` is a user of `autoware-ml`.

### projects/

The directory of `projects/` manages the model for each tasks.

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
  - We put highest priority on support and maintance to it because it leads to fast cycle development for developers.
- Tier A:
  - It is maintance phase on development, so it's updates is not frequently.
  - We make it maintainability with code quality, unit test and CI/CD, if possible.
  - We put a high prority on support to it.
- Tier B:
  - We fix a broken tool when needed.
  - We put a middle priority on support to it.
- Tier C:
  - We rarely use a tool or a model or it is just prototype version.
  - If it is not used for long time, we delete it.
  - We put a low priority on support to it.
