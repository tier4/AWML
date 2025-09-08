# autoware_ml_detection3d

This tool is ROS2 wrapper of autoware-ml for detection3d.

## Node information
### Interface

- Input

| Name                 | Type                            | Description      |
| -------------------- | ------------------------------- | ---------------- |
| `~/input/pointcloud` | `sensor_msgs::msg::PointCloud2` | input pointcloud |

- Output

| Name                       | Type                                             | Description          |
| -------------------------- | ------------------------------------------------ | -------------------- |
| `~/output/objects`         | `autoware_perception_msgs::msg::DetectedObjects` | detected objects     |
| `debug/cyclic_time_ms`     | `tier4_debug_msgs::msg::Float64Stamped`          | cyclic time (msg)    |
| `debug/processing_time_ms` | `tier4_debug_msgs::msg::Float64Stamped`          | processing time (ms) |

### Node parameters


| Parameter                     | Description                                    | Default                  |
| ----------------------------- | ---------------------------------------------- | ------------------------ |
| densification_num_past_frames | Number of past frames used fo PC densification | 1                        |
| densification_world_frame_id  | Fixed tf frame used for densification          | map                      |
| tf_timeout_seconds            | Timeout in seconds wheen looking for tfs       | 2.0                      |
| publish_pointcloud            | Publish the densified pointcloud (debug)       | False                    |
| empty_queue_after_inference   | Empty the densification queue after inference  | False                    |
| detection_threshold           | Threshold to accept detections                 | 0.35                     |
| model_config_path             | Path of the autoware-ml model                  | work_dirs/config.py      |
| model_checkpoint_path         | Path of the model's checkpoint                 | work_dirs/checkpoint.pth |

## Run
### 1. Setup environment

See [setting environment](/tools/setting_environment/).

### 2. Run

TBD
