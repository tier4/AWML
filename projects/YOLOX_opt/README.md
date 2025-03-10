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

### YOLOX-S 416x416 batch_size-6

| Name                                             | Mean ± Std Dev (ms) | Median (ms) | 80th Percentile (ms) | 90th Percentile (ms) | 95th Percentile (ms) | 99th Percentile (ms) |  
|--------------------------------------------------|----------------------|------------|----------------------|----------------------|----------------------|------------------------|  
| traffic_light_fine_detector Ros2 node (RTX 3090) | 5.27 ± 1.15          | 5.0        | 6.0                  | 6.0                  | 6.0                  | 6.90                   |  
| tensorrt (A100 80 GB) **(No pre, post processing)**  | 5.77 ± 2.03          | 7.2        | 7.4                  | 7.6                  | 7.7                  | 7.9                    |  
| pytorch (A100 80 GB)                             | 123.92 ± 50.41       | 144.35     | 157.64               | 163.84               | 174.32               | 218.65                 |  


## Troubleshooting

## Reference

- Ge Zheng et al. "YOLOX: Exceeding YOLO Series in 2021", arXiv 2021
