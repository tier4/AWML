# Setting environment for autoware-ml

Tools setting environment for `autoware-ml`.

- [Support priority](https://github.com/tier4/autoware-ml/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier S
- Environment
  - [x] Ubuntu22.04 LTS
    - This scripts do not need docker environment

## 1. Setup dataset
### 1.1. Most open dataset

If you want to use open dataset like nuScenes dataset, you set dataset as [mmdetection3d documents](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/index.html) and [mmdetection documents](https://mmdetection.readthedocs.io/en/latest/user_guides/dataset_prepare.html).

### 1.2. T4dataset

If you want to [T4dataset](https://github.com/tier4/tier4_perception_dataset) and you have data access right of [WebAuto](https://docs.web.auto/en/user-manuals/), you can T4dataset by [the script](/pipelines/webauto/download_t4dataset/).

## 2. Setup environment
### 2.1. Set repository

- Set environment

```sh
git clone https://github.com/tier4/autoware-ml
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

### 2.2 Build docker

- Build docker
  - Note that this process need for long time.
  - You may need `sudo` to use `docker` command.

```sh
DOCKER_BUILDKIT=1 docker build -t autoware-ml .
```

- [Option] If you want to use `autoware-ml` with `ROS2`, you can use other docker environment

```
DOCKER_BUILDKIT=1 docker build -t autoware-ml-ros2 ./tools/setting_environment/ros2/
```
