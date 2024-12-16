# autoware_ml_segmentation3d

This tool is ROS2 wrapper of autoware-ml for segmentation3d.

## Node information
### Interface

- Input

| Name                 | Type                            | Description      |
| -------------------- | ------------------------------- | ---------------- |
| `~/input/pointcloud` | `sensor_msgs::msg::PointCloud2` | input pointcloud |

- Output

| Name                       | Type                                    | Description          |
| -------------------------- | --------------------------------------- | -------------------- |
| `~/output/pointcloud`      | `sensor_msgs::msg::PointCloud2`         | output pointcloud    |
| `debug/cyclic_time_ms`     | `tier4_debug_msgs::msg::Float64Stamped` | cyclic time (msg)    |
| `debug/processing_time_ms` | `tier4_debug_msgs::msg::Float64Stamped` | processing time (ms) |

### Node parameters

TBD

## Run
### 1. Setup environment

See [setting environemnt](/tools/setting_environment/)

### 2. Run

TBD
