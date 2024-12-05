# Model name
## Summary

- [Support priority](https://github.com/tier4/autoware-ml/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier A
- ROS package: [tensorrt_yolox](https://github.com/autowarefoundation/autoware.universe/tree/main/perception/autoware_traffic_light_classifier)
- Supported dataset
  - [ ] COCO dataset
  - [x] T4dataset
- Supported model
- Other supported feature
  - [x] Add script to make .onnx file and deploy to Autoware
  - [ ] Add unit test
- Limited feature

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
python3 tools/detection2d/train.py /workspace/projects/MobileNetv2/configs/t4dataset/mobilenet-v2_tlr_car_t4dataset.py
python3 tools/detection2d/train.py /workspace/projects/MobileNetv2/configs/t4dataset/mobilenet-v2_tlr_ped_t4dataset.py
```

### 3. Evaluate

```bash
python3 tools/classification2d/test.py /workspace/work_dirs/mobilenet-v2_tlr_car_t4dataset/mobilenet-v2_tlr_car_t4dataset.py /workspace/work_dirs/mobilenet-v2_tlr_car_t4dataset/best_multi-label_f1-score_top1_epoch_12.pth

python3 tools/classification2d/test.py /workspace/work_dirs/mobilenet-v2_tlr_ped_t4dataset/mobilenet-v2_tlr_ped_t4dataset.py /workspace/work_dirs/mobilenet-v2_tlr_ped_t4dataset/best_multi-label_f1-score_top1_epoch_54.pth
```

### 4. Deploy

```bash
#car_mobilenet-v2-224x224_batch_1
python3 tools/classification2d/deploy.py /workspace/projects/MobileNetv2/configs/deploy/car_mobilenet-v2-224x224_batch_1.py /workspace/projects/MobileNetv2/configs/t4dataset/mobilenet-v2_tlr_car_t4dataset.py /workspace/work_dirs/mobilenet-v2_tlr_car_t4dataset/best_multi-label_f1-score_top1_epoch_12.pth /workspace/data/tlr/tlr_v0_1/8bb655ad-e12e-40a1-a7d7-43f279a1bd51/0/data/CAM_TRAFFIC_LIGHT_NEAR/0.jpg 1 --device cuda --work-dir /workspace/work_dirs/mobilenet-v2_tlr_car_t4dataset/

#car_mobilenet-v2-224x224_batch_4
python3 tools/classification2d/deploy.py /workspace/projects/MobileNetv2/configs/deploy/car_mobilenet-v2-224x224_batch_4.py /workspace/projects/MobileNetv2/configs/t4dataset/mobilenet-v2_tlr_car_t4dataset.py /workspace/work_dirs/mobilenet-v2_tlr_car_t4dataset/best_multi-label_f1-score_top1_epoch_12.pth /workspace/data/tlr/tlr_v0_1/8bb655ad-e12e-40a1-a7d7-43f279a1bd51/0/data/CAM_TRAFFIC_LIGHT_NEAR/0.jpg 4 --device cuda --work-dir /workspace/work_dirs/mobilenet-v2_tlr_car_t4dataset/

#car_mobilenet-v2-224x224_batch_6
python3 tools/classification2d/deploy.py /workspace/projects/MobileNetv2/configs/deploy/car_mobilenet-v2-224x224_batch_6.py /workspace/projects/MobileNetv2/configs/t4dataset/mobilenet-v2_tlr_car_t4dataset.py /workspace/work_dirs/mobilenet-v2_tlr_car_t4dataset/best_multi-label_f1-score_top1_epoch_12.pth /workspace/data/tlr/tlr_v0_1/8bb655ad-e12e-40a1-a7d7-43f279a1bd51/0/data/CAM_TRAFFIC_LIGHT_NEAR/0.jpg 6 --device cuda --work-dir /workspace/work_dirs/mobilenet-v2_tlr_car_t4dataset/

#ped_mobilenet-v2-224x224_batch_1
python3 tools/classification2d/deploy.py /workspace/projects/MobileNetv2/configs/deploy/ped_mobilenet-v2-224x224_batch_1.py /workspace/projects/MobileNetv2/configs/t4dataset/mobilenet-v2_tlr_ped_t4dataset.py /workspace/work_dirs/mobilenet-v2_tlr_ped_t4dataset/best_multi-label_f1-score_top1_epoch_54.pth /workspace/data/tlr/tlr_v0_1/8bb655ad-e12e-40a1-a7d7-43f279a1bd51/0/data/CAM_TRAFFIC_LIGHT_NEAR/0.jpg 1 --device cuda --work-dir /workspace/work_dirs/mobilenet-v2_tlr_ped_t4dataset

#ped_mobilenet-v2-224x224_batch_4
python3 tools/classification2d/deploy.py /workspace/projects/MobileNetv2/configs/deploy/ped_mobilenet-v2-224x224_batch_4.py /workspace/projects/MobileNetv2/configs/t4dataset/mobilenet-v2_tlr_ped_t4dataset.py /workspace/work_dirs/mobilenet-v2_tlr_ped_t4dataset/best_multi-label_f1-score_top1_epoch_54.pth /workspace/data/tlr/tlr_v0_1/8bb655ad-e12e-40a1-a7d7-43f279a1bd51/0/data/CAM_TRAFFIC_LIGHT_NEAR/0.jpg 4 --device cuda --work-dir /workspace/work_dirs/mobilenet-v2_tlr_ped_t4dataset

#ped_mobilenet-v2-224x224_batch_6
python3 tools/classification2d/deploy.py /workspace/projects/MobileNetv2/configs/deploy/ped_mobilenet-v2-224x224_batch_6.py /workspace/projects/MobileNetv2/configs/t4dataset/mobilenet-v2_tlr_ped_t4dataset.py /workspace/work_dirs/mobilenet-v2_tlr_ped_t4dataset/best_multi-label_f1-score_top1_epoch_54.pth /workspace/data/tlr/tlr_v0_1/8bb655ad-e12e-40a1-a7d7-43f279a1bd51/0/data/CAM_TRAFFIC_LIGHT_NEAR/0.jpg 6 --device cuda --work-dir /workspace/work_dirs/mobilenet-v2_tlr_ped_t4dataset
```

## Troubleshooting

## Reference
