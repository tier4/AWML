# (TBD) deploy_ros_parameter_file

Make [the ROS parameter](https://github.com/autowarefoundation/autoware.universe/blob/be8235d785597c41d01782ec35da862ba0906578/perception/autoware_lidar_transfusion/config/transfusion_ml_package.param.yaml) from config file of autoware-ml.

## Get started

- Prepare the ROS config
- Run the script

```sh
tools/deploy_ros_parameter_file/deploy_ros_parameter_file.py {config file of autoware-ml} {config file of deployment} {target_param_config_file}
```

- Example

```sh
tools/deploy_ros_parameter_file/deploy_ros_parameter_file.py \
projects/TransFusion/config/t4dataset/90m-768grid/transfusion_lidar_90m-768grid-t4xx1.py \
projects/TransFusion/config/deploy/transfusion_lidar_tensorrt_dynamic-20x5.py \
projects/TransFusion/config/deploy/target_param_config_file.yaml
```

  - The config file of `projects/TransFusion/config/deploy/target_param_config_file.yaml` is as below.
    - The script fulfill the ROS parameter value of autoware-ml
