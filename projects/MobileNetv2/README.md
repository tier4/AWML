# MobileNet v2
## Summary

- [Support priority](https://github.com/tier4/AWML/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier A
- ROS package: [autoware_traffic_light_classifier](https://github.com/autowarefoundation/autoware.universe/tree/main/perception/autoware_traffic_light_classifier)
- Supported dataset
  - [ ] COCO dataset
  - [x] T4dataset
- Supported model
- Other supported feature
  - [x] Add script to make .onnx file and deploy to Autoware
  - [ ] Add unit test

## Results and models

- MobileNetv2-CarTrafficLight
  - v0
    - [MobileNetv2-CarTrafficLight base/0.X](./docs/MobileNetv2-CarTrafficLight/v0/base.md)
- MobileNetv2-PedestrianTrafficLight
  - v0
    - [MobileNetv2-PedestrianTrafficLight base/0.X](./docs/MobileNetv2-PedestrianTrafficLight/v0/base.md)

## Get started
### 1. Setup

```bash
# Because the annotations for traffic light in t4dataset are are in 2d object detection format, we use the same script as detection2d.
python3 tools/detection2d/create_data_t4dataset.py --config autoware_ml/configs/classification2d/dataset/t4dataset/tlr_classifier_car.py --root_path ./data/tlr/ --data_name tlr -o ./data/info/tlr_classifier_car
python3 tools/detection2d/create_data_t4dataset.py --config autoware_ml/configs/classification2d/dataset/t4dataset/tlr_classifier_pedestrian.py --root_path ./data/tlr/ --data_name tlr -o ./data/info/tlr_classifier_pedestrian
```

### 2. Train

```bash
python3 tools/detection2d/train.py /workspace/projects/MobileNetv2/configs/t4dataset/MobileNetv2-CarTrafficLight/mobilenet-v2_tlr_car_t4dataset.py
python3 tools/detection2d/train.py /workspace/projects/MobileNetv2/configs/t4dataset/MobileNetv2-PedestrianTrafficLight/mobilenet-v2_tlr_ped_t4dataset.py
```

### 3. Evaluate

```bash
python3 tools/classification2d/test.py /workspace/work_dirs/mobilenet-v2_tlr_car_t4dataset/mobilenet-v2_tlr_car_t4dataset.py /workspace/work_dirs/mobilenet-v2_tlr_car_t4dataset/best_multi-label_f1-score_top1_epoch_12.pth

python3 tools/classification2d/test.py /workspace/work_dirs/mobilenet-v2_tlr_ped_t4dataset/mobilenet-v2_tlr_ped_t4dataset.py /workspace/work_dirs/mobilenet-v2_tlr_ped_t4dataset/best_multi-label_f1-score_top1_epoch_54.pth
```

### 4. Deploy

```bash
#car_mobilenet-v2-224x224_batch_1
python3 tools/classification2d/deploy.py /workspace/projects/MobileNetv2/configs/deploy/MobileNetv2-CarTrafficLight/car_mobilenet-v2-224x224_batch_1.py /workspace/projects/MobileNetv2/configs/t4dataset/MobileNetv2-CarTrafficLight/mobilenet-v2_tlr_car_t4dataset.py /workspace/work_dirs/mobilenet-v2_tlr_car_t4dataset/best_multi-label_f1-score_top1_epoch_12.pth /workspace/data/tlr/tlr_v0_1/8bb655ad-e12e-40a1-a7d7-43f279a1bd51/0/data/CAM_TRAFFIC_LIGHT_NEAR/0.jpg 1 --device cuda --work-dir /workspace/work_dirs/mobilenet-v2_tlr_car_t4dataset/

#car_mobilenet-v2-224x224_batch_4
python3 tools/classification2d/deploy.py /workspace/projects/MobileNetv2/configs/deploy/MobileNetv2-CarTrafficLight/car_mobilenet-v2-224x224_batch_4.py /workspace/projects/MobileNetv2/configs/t4dataset/MobileNetv2-CarTrafficLight/mobilenet-v2_tlr_car_t4dataset.py /workspace/work_dirs/mobilenet-v2_tlr_car_t4dataset/best_multi-label_f1-score_top1_epoch_12.pth /workspace/data/tlr/tlr_v0_1/8bb655ad-e12e-40a1-a7d7-43f279a1bd51/0/data/CAM_TRAFFIC_LIGHT_NEAR/0.jpg 4 --device cuda --work-dir /workspace/work_dirs/mobilenet-v2_tlr_car_t4dataset/

#car_mobilenet-v2-224x224_batch_6
python3 tools/classification2d/deploy.py /workspace/projects/MobileNetv2/configs/deploy/MobileNetv2-CarTrafficLight/car_mobilenet-v2-224x224_batch_6.py /workspace/projects/MobileNetv2/configs/t4dataset/MobileNetv2-CarTrafficLight/mobilenet-v2_tlr_car_t4dataset.py /workspace/work_dirs/mobilenet-v2_tlr_car_t4dataset/best_multi-label_f1-score_top1_epoch_12.pth /workspace/data/tlr/tlr_v0_1/8bb655ad-e12e-40a1-a7d7-43f279a1bd51/0/data/CAM_TRAFFIC_LIGHT_NEAR/0.jpg 6 --device cuda --work-dir /workspace/work_dirs/mobilenet-v2_tlr_car_t4dataset/

#ped_mobilenet-v2-224x224_batch_1
python3 tools/classification2d/deploy.py /workspace/projects/MobileNetv2/configs/deploy/MobileNetv2-PedestrianTrafficLight/ped_mobilenet-v2-224x224_batch_1.py /workspace/projects/MobileNetv2/configs/t4dataset/MobileNetv2-PedestrianTrafficLight/mobilenet-v2_tlr_ped_t4dataset.py /workspace/work_dirs/mobilenet-v2_tlr_ped_t4dataset/best_multi-label_f1-score_top1_epoch_54.pth /workspace/data/tlr/tlr_v0_1/8bb655ad-e12e-40a1-a7d7-43f279a1bd51/0/data/CAM_TRAFFIC_LIGHT_NEAR/0.jpg 1 --device cuda --work-dir /workspace/work_dirs/mobilenet-v2_tlr_ped_t4dataset

#ped_mobilenet-v2-224x224_batch_4
python3 tools/classification2d/deploy.py /workspace/projects/MobileNetv2/configs/deploy/MobileNetv2-PedestrianTrafficLight/ped_mobilenet-v2-224x224_batch_4.py /workspace/projects/MobileNetv2/configs/t4dataset/MobileNetv2-PedestrianTrafficLight/mobilenet-v2_tlr_ped_t4dataset.py /workspace/work_dirs/mobilenet-v2_tlr_ped_t4dataset/best_multi-label_f1-score_top1_epoch_54.pth /workspace/data/tlr/tlr_v0_1/8bb655ad-e12e-40a1-a7d7-43f279a1bd51/0/data/CAM_TRAFFIC_LIGHT_NEAR/0.jpg 4 --device cuda --work-dir /workspace/work_dirs/mobilenet-v2_tlr_ped_t4dataset

#ped_mobilenet-v2-224x224_batch_6
python3 tools/classification2d/deploy.py /workspace/projects/MobileNetv2/configs/deploy/MobileNetv2-PedestrianTrafficLight/ped_mobilenet-v2-224x224_batch_6.py /workspace/projects/MobileNetv2/configs/t4dataset/MobileNetv2-PedestrianTrafficLight/mobilenet-v2_tlr_ped_t4dataset.py /workspace/work_dirs/mobilenet-v2_tlr_ped_t4dataset/best_multi-label_f1-score_top1_epoch_54.pth /workspace/data/tlr/tlr_v0_1/8bb655ad-e12e-40a1-a7d7-43f279a1bd51/0/data/CAM_TRAFFIC_LIGHT_NEAR/0.jpg 6 --device cuda --work-dir /workspace/work_dirs/mobilenet-v2_tlr_ped_t4dataset
```

## Latency

### MobileNetV2 224x224 batch_size-6

| Name                                                    | Mean ± Std Dev (ms) | Median (ms) | 80th Percentile (ms) | 90th Percentile (ms) | 95th Percentile (ms) | 99th Percentile (ms)   |  
|---------------------------------------------------------|----------------------|------------|----------------------|----------------------|----------------------|------------------------|  
| traffic_light_classifier Ros2 node (RTX 3090)           | 1.89 ± 1.32          | 1.0        | 3.0                  | 4.0                  | 5.0                  | 6.00                   |  
| tensorrt (A100 80 GB) **(No pre, post processing) fp16**| 0.224 ± 0.035        | 0.223      | 0.225                | 0.226                | 0.227                | 0.228                  |  
| pytorch (A100 80 GB)  num_workers=16                    | 25.61 ± 50.10        | 10.94      | 12.90                | 35.46                | 134.77               | 264.12                 |  

### MobileNetV2 224x224 batch_size-4

| Name                                                    | Mean ± Std Dev (ms) | Median (ms) | 80th Percentile (ms) | 90th Percentile (ms) | 95th Percentile (ms) | 99th Percentile (ms)   |  
|---------------------------------------------------------|----------------------|------------|----------------------|----------------------|----------------------|------------------------|  
| traffic_light_classifier Ros2 node (RTX 3090)           | 1.72 ± 1.45          | 1.0        | 2.0                  | 3.0                  | 4.0                  | 5.00                   |  
| tensorrt (A100 80 GB) **(No pre, post processing) fp16**| 0.202 ± 0.026        | 0.201      | 0.203                | 0.204                | 0.205                | 0.206                  |  
| pytorch (A100 80 GB)  num_workers=16                    | 17.90 ± 28.49        | 10.30      | 11.85                | 14.78                | 90.21                | 147.73                 |  

### MobileNetV2 224x224 batch_size-1

| Name                                                    | Mean ± Std Dev (ms) | Median (ms) | 80th Percentile (ms) | 90th Percentile (ms) | 95th Percentile (ms) | 99th Percentile (ms)   |  
|---------------------------------------------------------|----------------------|------------|----------------------|----------------------|----------------------|------------------------|  
| traffic_light_classifier Ros2 node (RTX 3090)           | 1.65 ± 1.30          | 1.0        | 2.0                  | 3.0                  | 4.0                  | 5.00                   |  
| tensorrt (A100 80 GB) **(No pre, post processing) fp16**| 0.180 ± 0.020        | 0.179      | 0.181                | 0.182                | 0.183                | 0.184                  |  
| pytorch (A100 80 GB)  num_workers=16                    | 9.88 ± 1.85          | 9.68       | 10.85                | 11.62                | 12.56                | 17.99                  |  


## Troubleshooting

## Reference

- Mark Sandler et al. "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018
- [MobileNet v2 of mmpretrain](https://github.com/open-mmlab/mmpretrain/tree/main/configs/mobilenet_v2)
