# autoware_ml_detection2d

This tool is ROS2 wrapper of autoware-ml for detection2d.

## Node information
### Interface

- Input

| Name            | Type                      | Description |
| --------------- | ------------------------- | ----------- |
| `~/input/image` | `sensor_msgs::msg::Image` | input image |

- Output

| Name          | Type                                               | Description                                 |
| ------------- | -------------------------------------------------- | ------------------------------------------- |
| `out/objects` | `tier4_perception_msgs/DetectedObjectsWithFeature` | The detected objects with 2D bounding boxes |

### Node parameters

TBD

## Run
### 1. Setup environment

See [setting environment](/tools/setting_environment/).

### 2. Run

TBD
