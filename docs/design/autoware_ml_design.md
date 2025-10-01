# `AWML` design
## Pipeline design
### Overview diagram

`AWML` is designed for deployment from training and evaluation with Autoware and active learning framework.

![](/docs/fig/AWML.drawio.svg)

### The pipeline design of `AWML`

![](/docs/fig/pipeline.drawio.svg)

>  Orange line: the deployment flow for ML model.

- (1) create_data_info.py
  - Make info file with annotated data.
- (2) train.py, (3) test.py, (4) visualize.py
  - From config file, the ML model is trained and evaluated.
- (5) deploy.py
  - If the model is used for Autoware, the model is deployed to onnx file.

> Green line: the pipeline for active learning

- (6) pseudo_label.py
  - Make info file from non-annotated T4dataset for 2D and 3D.
- (7) create_pseudo_t4dataset.py
  - Make pseudo-label T4dataset from the info file which is based on pseudo label.
- (8) choose_annotation.py
  - Choose using annotation from raw pseudo label.
  - The info file which is used in [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) by pickle file (`.pkl`) is used in auto labeling  of `AWML` as interface.
  - For example, it is used for `scene_selector` and `pseudo_label` to tune the parameter of the threshold of confidence with offline model.

### The format of infos file

- We use the format of info files based on [the NuScenes format](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/nuscenes.html?highlight=info).

```
- `nuscenes_infos_train.pkl`: training dataset, a dict contains two keys: `metainfo` and `data_list`.
  `metainfo` contains the basic information for the dataset itself, such as `categories`, `dataset` and `info_version`, while `data_list` is a list of dict, each dict (hereinafter referred to as `info`) contains all the detailed information of single sample as follows:
  - info\['sample_idx'\]: The index of this sample in the whole dataset.
  - info\['token'\]: Sample data token.
  - info\['timestamp'\]: Timestamp of the sample data.
  - info\['ego2global'\]: The transformation matrix from the ego vehicle to global coordinates. (4x4 list)
  - info\['lidar_points'\]: A dict containing all the information related to the lidar points.
    - info\['lidar_points'\]\['lidar_path'\]: The filename of the lidar point cloud data.
    - info\['lidar_points'\]\['num_pts_feats'\]: The feature dimension of point.
    - info\['lidar_points'\]\['lidar2ego'\]: The transformation matrix from this lidar sensor to ego vehicle. (4x4 list)
  - info\['lidar_sweeps'\]: A list contains sweeps information (The intermediate lidar frames without annotations)
    - info\['lidar_sweeps'\]\[i\]\['lidar_points'\]\['data_path'\]: The lidar data path of i-th sweep.
    - info\['lidar_sweeps'\]\[i\]\['lidar_points'\]\['lidar2ego'\]: The transformation matrix from this lidar sensor to ego vehicle. (4x4 list)
    - info\['lidar_sweeps'\]\[i\]\['lidar_points'\]\['ego2global'\]: The transformation matrix from the ego vehicle to global coordinates. (4x4 list)
    - info\['lidar_sweeps'\]\[i\]\['lidar2sensor'\]: The transformation matrix from the main lidar sensor to the current sensor (for collecting the sweep data). (4x4 list)
    - info\['lidar_sweeps'\]\[i\]\['timestamp'\]: Timestamp of the sweep data.
    - info\['lidar_sweeps'\]\[i\]\['sample_data_token'\]: The sweep sample data token.
  - info\['images'\]: A dict contains six keys corresponding to each camera: `'CAM_FRONT'`, `'CAM_FRONT_RIGHT'`, `'CAM_FRONT_LEFT'`, `'CAM_BACK'`, `'CAM_BACK_LEFT'`, `'CAM_BACK_RIGHT'`. Each dict contains all data information related to  corresponding camera.
    - info\['images'\]\['CAM_XXX'\]\['img_path'\]: The filename of the image.
    - info\['images'\]\['CAM_XXX'\]\['cam2img'\]: The transformation matrix recording the intrinsic parameters when projecting 3D points to each image plane. (3x3 list)
    - info\['images'\]\['CAM_XXX'\]\['sample_data_token'\]: Sample data token of image.
    - info\['images'\]\['CAM_XXX'\]\['timestamp'\]: Timestamp of the image.
    - info\['images'\]\['CAM_XXX'\]\['cam2ego'\]: The transformation matrix from this camera sensor to ego vehicle. (4x4 list)
    - info\['images'\]\['CAM_XXX'\]\['lidar2cam'\]: The transformation matrix from lidar sensor to this camera. (4x4 list)
  - info\['instances'\]: It is a list of dict. Each dict contains all annotation information of single instance. For the i-th instance:
    - info\['instances'\]\[i\]\['bbox_3d'\]: List of 7 numbers representing the 3D bounding box of the instance, in (x, y, z, l, w, h, yaw) order.
    - info\['instances'\]\[i\]\['bbox_label_3d'\]: A int indicate the label of instance and the -1 indicate ignore.
    - info\['instances'\]\[i\]\['velocity'\]: Velocities of 3D bounding boxes (no vertical measurements due to inaccuracy), a list has shape (2.).
    - info\['instances'\]\[i\]\['num_lidar_pts'\]: Number of lidar points included in each 3D bounding box.
    - info\['instances'\]\[i\]\['num_radar_pts'\]: Number of radar points included in each 3D bounding box.
    - info\['instances'\]\[i\]\['bbox_3d_isvalid'\]: Whether each bounding box is valid. In general, we only take the 3D boxes that include at least one lidar or radar point as valid boxes.
  - info\['cam_instances'\]: It is a dict containing keys `'CAM_FRONT'`, `'CAM_FRONT_RIGHT'`, `'CAM_FRONT_LEFT'`, `'CAM_BACK'`, `'CAM_BACK_LEFT'`, `'CAM_BACK_RIGHT'`. For vision-based 3D object detection task, we split 3D annotations of the whole scenes according to the camera they belong to. For the i-th instance:
    - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['bbox_label'\]: Label of instance.
    - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['bbox_label_3d'\]: Label of instance.
    - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['bbox'\]: 2D bounding box annotation (exterior rectangle of the projected 3D box), a list arrange as \[x1, y1, x2, y2\].
    - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['center_2d'\]: Projected center location on the image, a list has shape (2,), .
    - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['depth'\]: The depth of projected center.
    - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['velocity'\]: Velocities of 3D bounding boxes (no vertical measurements due to inaccuracy), a list has shape (2,).
    - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['attr_label'\]: The attr label of instance. We maintain a default attribute collection and mapping for attribute classification.
    - info\['cam_instances'\]\['CAM_XXX'\]\[i\]\['bbox_3d'\]: List of 7 numbers representing the 3D bounding box of the instance, in (x, y, z, l, h, w, yaw) order.
  - info\['pts_semantic_mask_path'\]：The filename of the lidar point cloud semantic segmentation
```

- We add `pred_instances_3d` and `pred_instances_2d` for this format to use for inference pipeline.
  - For example, the `pred_instances_3d` element uses for auto-labeling.

```
- `nuscenes_infos_train.pkl`: training dataset, a dict contains two keys: `metainfo` and `data_list`.
  - info\['pred_instances_3d'\]: It is a list of dict. Each dict contains all 3D inference information of single instance. For the i-th instance:
    - info\['pred_instances_3d'\]\[i\]\['bbox_3d'\]: List of 7 numbers representing the 3D bounding box of the instance, in (x, y, z, l, w, h, yaw) order.
    - info\['pred_instances_3d'\]\[i\]\['bbox_label_3d'\]: A int indicate the label of instance and the -1 indicate ignore.
    - info\['pred_instances_3d'\]\[i\]\['bbox_score_3d'\]: A float confidence score of instance.
    - info\['pred_instances_3d'\]\[i\]\['velocity'\]: Velocities of 3D bounding boxes (no vertical measurements due to inaccuracy), a list has shape (2.).
    - info\['pred_instances_3d'\]\[i\]\['instance_id_3d'\]: String field of instance id of 3D bounding boxes. It can be used for 3D tracking algorithm.
  - info\['pred_instances_2d'\]: It is a list of dict. Each dict contains all 2D inference information of single instance. For the i-th instance:
    - info\['pred_instances_2d'\]\[i\]\['bbox_2d'\]: List of 7 numbers representing the 2D bounding box of the instance, in (x, y, z, l, w, h, yaw) order.
    - info\['pred_instances_2d'\]\[i\]\['bbox_label_2d'\]: A int indicate the label of instance and the -1 indicate ignore.
    - info\['pred_instances_2d'\]\[i\]\['bbox_score_2d'\]: A float confidence score of instance.
    - info\['pred_instances_2d'\]\[i\]\['instance_id_2d'\]: String field of instance id of 2D bounding boxes. It can be used for 2D tracking algorithm.
```

## Supported environment

- [pytorch v2.2.0](https://github.com/pytorch/pytorch/tree/v2.2.0)

`AWML` is based on pytorch.

- [mmdetection3d v1.4](https://github.com/open-mmlab/mmdetection3d/tree/v1.4.0).

This is machine learning framework for 3D detection.
`AWML` is strongly based on this framework.

If you want to learn about use of `mmdetection3d`, we recommend to read [user guides](https://mmdetection3d.readthedocs.io/en/latest/user_guides/index.html) at first.
If you want to learn about config files of `mmdetection3d`, we recommend to read [user guides for configs](https://mmdetection3d.readthedocs.io/en/latest/user_guides/config.html).
If you want to learn about info files of  `mmdetection3d`, we recommend to read [nuscenes dataset](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/nuscenes.html?highlight=info).

- [mmdetection v3.2.0](https://github.com/open-mmlab/mmdetection/tree/v3.2.0)

This is machine learning framework for 2D detection.

- [mmcv v2.1.0](https://github.com/open-mmlab/mmcv/tree/v2.1.0)
- [mmdeploy v1.3.1](https://github.com/open-mmlab/mmdeploy/tree/v1.3.1)

These are core library for MMLab libraries.
If you want to develop `AWML`, we recommend to read the documents of these.

## `AWML` architecture
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
      - db_jpntaxi_v1.yaml
      - db_jpntaxi_v2.yaml
      - db_jpntaxi_v3.yaml
    - detection2d/
      - XX1.py
      - X2.py
      - db_jpntaxi_v1.yaml
      - db_jpntaxi_v2.yaml
      - tlr_v1_0.yaml
```

- dataset configs: `autoware_ml/configs/*.yaml`

The file like `detection3d/db_jpntaxi_v1.yaml` defines dataset ids of T4dataset.
We define T4dataset version as below.

- version: major.minor.build
  - major: sensor configuration
  - minor: dataset scenes
  - build: dataset version

### docs/

The directory of `docs/` is design documents for `AWML`.
The target of documents is a designer of whole ML pipeline system and developers of `AWML` core library.

### pipelines/

The directory of `pipelines/` manages the pipelines that consist of `tools`.
This directory can depend on `/autoware_ml`, `projects`, `/tools`, and other `/pipelines`.

Each pipeline has `README.md`, a process document to use when you ask someone else to do the work.
The target of `README.md` is a user of `AWML`.

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
The target of `README.md` is a user of `AWML`.

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
The target of `README.md` is a developer of `AWML`.

### data/

`data/` directory is gitignored, and it refers to inputs for `AWML`, for example, datasets and info files.
The design follows `mmdetection3d` as recommended in  https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/nuscenes.html. Please change the directory name for different use cases accordingly.

### work_dirs/

The directory manages any artifacts and files generated from `AWML`, for example, training/evaluation/visualization.
Please change the directory name accordingly.

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

## Versioning strategy for `AWML`

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
