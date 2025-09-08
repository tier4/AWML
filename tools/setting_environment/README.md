# Setting environment for AWML

Tools setting environment for `AWML`.

- [Support priority](https://github.com/tier4/AWML/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier S
- Environment
  - [x] Ubuntu22.04 LTS
    - This scripts do not need docker environment

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
