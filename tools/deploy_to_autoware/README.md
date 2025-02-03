# deploy_to_autoware

This tools aim to deploy for Autoware.

## Update onnx metadata

This script injects extra meta information.

- Run

```sh
python tools/deploy_to_autoware/update_onnx_metadata.py {path to onnx file} --domain "3D object detection" --version "CenterPoint base/1.0"
```

By this update, logs in Autoware log files or /rosout topic can have model metadata.

```
[autoware_lidar_centerpoint_node-1] [I] [TRT] Input filename:   /home/autoware/autoware_data/lidar_centerpoint/pts_backbone_neck_head_centerpoint_tiny.onnx
[autoware_lidar_centerpoint_node-1] [I] [TRT] ONNX IR version:  0.0.6
[autoware_lidar_centerpoint_node-1] [I] [TRT] Opset version:    11
[autoware_lidar_centerpoint_node-1] [I] [TRT] Producer name:    autoware-ml
[autoware_lidar_centerpoint_node-1] [I] [TRT] Producer version: v0.7.1
[autoware_lidar_centerpoint_node-1] [I] [TRT] Domain:           3D object detection
[autoware_lidar_centerpoint_node-1] [I] [TRT] Model version:    7
[autoware_lidar_centerpoint_node-1] [I] [TRT] Doc string:       v3.2.3
```

## [TBD] Generate ROS parameter

This script makes [the ROS parameter](https://github.com/autowarefoundation/autoware.universe/blob/be8235d785597c41d01782ec35da862ba0906578/perception/autoware_lidar_transfusion/config/transfusion_ml_package.param.yaml) from config file of autoware-ml.

- Run the script

```sh
python tools/deploy_ros_parameter_file/deploy_ros_parameter_file.py {config file of autoware-ml} {config file of deployment} {target_param_config_file}
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

```yaml
/**:
  ros__parameters:
    class_names: {class_names}
    voxels_num: [{backend_config.model_inputs.input_shapes.voxels.min_shape[0]}, {backend_config.model_inputs.input_shapes.voxels.min_shape[1]}, {backend_config.model_inputs.input_shapes.voxels.min_shape[2]}]
    point_cloud_range: {point_cloud_range}
    voxel_size: {vocel_size}
    num_proposals: {num_proposal}
```
