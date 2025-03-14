# YOLOX_opt
## Summary

- [Support priority](https://github.com/tier4/AWML/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier A
- ROS package
  - [tensorrt_yolox](https://github.com/autowarefoundation/autoware.universe/tree/main/perception/tensorrt_yolox)
  - [traffic_light_fine_detector](https://github.com/autowarefoundation/autoware.universe/tree/main/perception/autoware_traffic_light_fine_detector)
- Supported dataset
  - [ ] COCO dataset
  - [x] T4dataset
- Supported model
- Other supported feature
  - [x] Add script to make .onnx file and deploy to Autoware
  - [ ] Add unit test
- Limited feature

## Results and models

- YOLOX_opt-S-TrafficLight
  - v0
    - [YOLOX_opt-S-TrafficLight base/0.X](./docs/YOLOX_opt-S-TrafficLight/v0/base.md)

## Get started
### 1. Setup

- [Run setup environment at first](/tools/setting_environment/)

### 2. Train

- For traffic light recognition of fine detector

```bash
python3 tools/detection2d/train.py /workspace/projects/YOLOX_opt/configs/t4dataset/YOLOX_opt-S-TrafficLight/yolox_s_tlr_416x416_pedcar_t4dataset.py
```

### 3. Evaluate

- For traffic light recognition of fine detector

```bash
python3 tools/detection2d/test.py /workspace/work_dirs/yolox_s_tlr_416x416_pedcar_t4dataset/yolox_s_tlr_416x416_pedcar_t4dataset.py /workspace/work_dirs/yolox_s_tlr_416x416_pedcar_t4dataset/epoch_300.pth
```

### 4. Deploy

- For traffic light recognition of fine detector

```bash
# batch size 1
python3 tools/detection2d/deploy_yolox.py /workspace/work_dirs/yolox_s_tlr_416x416_pedcar_t4dataset/epoch_300.pth --input_size 416 416 --model yolox-s --batch_size 1 --output_onnx_file tlr_car_ped_yolox_s_batch_1.onnx

# batch size 4
python3 tools/detection2d/deploy_yolox.py /workspace/work_dirs/yolox_s_tlr_416x416_pedcar_t4dataset/epoch_300.pth --input_size 416 416 --model yolox-s --batch_size 4 --output_onnx_file tlr_car_ped_yolox_s_batch_4.onnx

# batch size 6
python3 tools/detection2d/deploy_yolox.py /workspace/work_dirs/yolox_s_tlr_416x416_pedcar_t4dataset/epoch_300.pth --input_size 416 416 --model yolox-s --batch_size 6 --output_onnx_file tlr_car_ped_yolox_s_batch_6.onnx
```

## Latency

### YOLOX 224x224 batch_size-6

| Name                                                    | Mean ± Std Dev (ms) | Median (ms) | 80th Percentile (ms) | 90th Percentile (ms) | 95th Percentile (ms) | 99th Percentile (ms)   |  
|---------------------------------------------------------|----------------------|------------|----------------------|----------------------|----------------------|------------------------|  
| traffic_light_classifier Ros2 node (RTX 3090)           | 2.69 ± 1.52          | 2.0        | 3.0                  | 5.0                  | 6.0                  | 8.00                   |  
| tensorrt (A100 80 GB) **(No pre, post processing) fp16**| 1.45 ± 0.07          | 1.44       | 1.45                 | 1.45                 | 1.45                 | 1.46                   |
| pytorch (A100 80 GB) num_workers=16                     | 42.19 ± 10.00        | 40.54      | 47.36                | 51.28                | 55.14                | 68.69                  |  

### YOLOX 224x224 batch_size-4

| Name                                                    | Mean ± Std Dev (ms) | Median (ms) | 80th Percentile (ms) | 90th Percentile (ms) | 95th Percentile (ms) | 99th Percentile (ms)   |  
|---------------------------------------------------------|----------------------|------------|----------------------|----------------------|----------------------|------------------------|  
| traffic_light_classifier Ros2 node (RTX 3090)           | 2.10 ± 1.81          | 1.0        | 4.0                  | 5.0                  | 6.0                  | 8.00                   |  
| tensorrt (A100 80 GB) **(No pre, post processing) fp16**| 1.17 ± 0.05          | 1.17       | 1.17                 | 1.17                 | 1.18                 | 1.18                   |
| pytorch (A100 80 GB) num_workers=16                     | 30.75 ± 5.77         | 29.70      | 35.01                | 37.38                | 39.46                | 46.72                  |  

### YOLOX 224x224 batch_size-1

| Name                                                    | Mean ± Std Dev (ms) | Median (ms) | 80th Percentile (ms) | 90th Percentile (ms) | 95th Percentile (ms) | 99th Percentile (ms)   |  
|---------------------------------------------------------|----------------------|------------|----------------------|----------------------|----------------------|------------------------|  
| traffic_light_classifier Ros2 node (RTX 3090)           | 1.90 ± 1.85          | 1.0        | 2.0                  | 4.0                  | 5.0                  | 6.00                   |  
| tensorrt (A100 80 GB) **(No pre, post processing) fp16**| 0.86 ± 0.05          | 0.86       | 0.86                 | 0.86                 | 0.86                 | 0.87                   |  
| pytorch (A100 80 GB) num_workers=16                     | 19.42 ± 3.96         | 19.37      | 23.89                | 25.30                | 26.37                | 29.27                  |  


## Troubleshooting

## Reference

- Ge Zheng et al. "YOLOX: Exceeding YOLO Series in 2021", arXiv 2021
