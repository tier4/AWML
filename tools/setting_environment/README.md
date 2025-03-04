# Setting environment for autoware-ml

Tools setting environment for `AWML`.

- [Support priority](https://github.com/tier4/AWML/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier S
- Environment
  - [x] Ubuntu22.04 LTS
    - This scripts do not need docker environment

## 1. Download dataset
### 1.1. Download T4dataset

If you want to download [T4dataset](https://github.com/tier4/tier4_perception_dataset) and you have data access right for [WebAuto](https://docs.web.auto/en/user-manuals/), you can download T4dataset by a predefined script. Please refer to [the Readme of the script](/pipelines/webauto/download_t4dataset/) for the specific instruction.

If you do not have the access to Web.Auto and still want to use the dataset, please contact Web.Auto team from [the Web.Auto contact form](https://web.auto/contact/). However, please note that these dataset are currently only available for TIER IV members as of September 2024.

### 1.2. Download the other open dataset

If you want to use open dataset like nuScenes dataset, please follow [mmdetection3d documents](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/index.html) and [mmdetection documents](https://mmdetection.readthedocs.io/en/latest/user_guides/dataset_prepare.html).

## 2. Setup environment
### 2.1. Set repository

- Set environment

```sh
git clone https://github.com/tier4/AWML
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

### 2.2 Prepare docker

- Docker pull for base environment
  - See https://github.com/tier4/AWML/pkgs/container/autoware-ml-base

```
docker pull ghcr.io/tier4/autoware-ml-base:latest
```

- Docker pull for `AWML` environment with `ROS2`
  - See https://github.com/tier4/AWML/pkgs/container/autoware-ml-ros2

```
docker pull ghcr.io/tier4/autoware-ml-ros2:latest
```

## Tips
### Build docker on your own

- Build docker
  - Note that this process need for long time.
  - You may need `sudo` to use `docker` command.

```sh
DOCKER_BUILDKIT=1 docker build -t autoware-ml .
```

- If you want to use `AWML` with `ROS2`, you can use other docker environment

```
DOCKER_BUILDKIT=1 docker build -t autoware-ml-ros2 ./tools/setting_environment/ros2/
```

- If you want to do performance testing with tensorrt, you can use tensorrt docker environment.

```
DOCKER_BUILDKIT=1 docker build -t autoware-ml-tensorrt ./tools/setting_environment/tensorrt/
```
