# Software architecture

The architecture of autoware-ml is based on [mmdetection3d v1.4](https://github.com/open-mmlab/mmdetection3d/tree/v1.4.0).

## autoware_ml/

The directory of `autoware_ml` is library for autoware-ml.
This directory can be used as library from other software.

- configs

The config files in `autoware_ml` is used commonly for each projects.

```
- autoware_ml/
  - configs/
    - detection3d/
    - detection2d/
```

## docs/

The directory of `docs/` is documents for autoware-ml.

## projects/

The directory of `projects/` manages the model for each tasks.

```
- projects/
  - BEVFusion
  - YOLOX
  - TransFusion
```

## tools/

The directory of `tools/` manages tools for each tasks.

- The pipeline of each task for training and evaluation

```
- tools/
  - detection3d/
  - detection2d/
```

- Tools for dataset making

```
- tools/
  - dataset/
    - scene_evaluation
```
