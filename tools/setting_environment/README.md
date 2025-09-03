# Setting environment for AWML

Tools setting environment for `AWML`.

- [Support priority](https://github.com/tier4/AWML/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier S
- Environment
  - [x] Ubuntu22.04 LTS
    - This scripts do not need docker environment

## Setting environment for AWML
### 1.1. Set repository

- Set environment

```sh
git clone https://github.com/tier4/AWML
```

### 1.2 Prepare docker

- Docker pull for base environment
  - See https://github.com/tier4/AWML/pkgs/container/autoware-ml-base

```
docker pull ghcr.io/tier4/autoware-ml-base:latest
```

### 1.3. Download T4dataset

- Get data access right for T4dataset in [WebAuto](https://docs.web.auto/en/user-manuals/).
- Download T4dataset by using [download scripts](/pipelines/webauto/download_t4dataset/)

If you do not have the access to Web.Auto and still want to use the dataset, please contact Web.Auto team from [the Web.Auto contact form](https://web.auto/contact/). However, please note that these dataset are currently only available for TIER IV members as of September 2025.
If you want to use other dataset, please see [this document](setting_other_dataset.md).

After downloading dataset, the directory shows as below.

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

## Other environment
### `AWML` with `ROS2`

- Docker pull for `AWML` environment with `ROS2`
  - See https://github.com/tier4/AWML/pkgs/container/autoware-ml-ros2

```
docker pull ghcr.io/tier4/autoware-ml-ros2:latest
```

- If you want to build in local environment, run below command

```sh
DOCKER_BUILDKIT=1 docker build -t autoware-ml-ros2 ./tools/setting_environment/ros2/
```

### `AWML` with `TensorRT`

- If you want to do performance testing with tensorrt, you can use tensorrt docker environment.

```sh
DOCKER_BUILDKIT=1 docker build -t autoware-ml-tensorrt ./tools/setting_environment/tensorrt/
```
