# Model name
## Summary

- [Support priority](https://github.com/tier4/autoware-ml/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier A
- ROS package: [tensorrt_yolox](https://github.com/autowarefoundation/autoware.universe/tree/main/perception/tensorrt_yolox)
- Supported dataset
  - [ ] COCO dataset
  - [x] T4dataset
- Supported model
- Other supported feature
  - [x] Add script to make .onnx file and deploy to Autoware
  - [ ] Add unit test
- Limited feature

## Results and models

- [Deployed model](docs/deployed_model.md)
- [Archived model](docs/archived_model.md)

## Get started
### 1. Setup

```bash
python3 tools/detection2d/create_data_t4dataset.py --config autoware_ml/configs/detection2d/dataset/t4dataset/tlr_finedetector.py --root_path ./data/tlr/ --data_name tlr -o ./data/tlr_pedcar
```

### 3. Train

```bash
python3 tools/detection2d/train.py /workspace/projects/YOLOX_opt/configs/t4dataset/yolox_s_tlr_416x416_pedcar_t4dataset.py
```

### 3. Evaluate

```bash
python3 tools/detection2d/test.py /workspace/work_dirs/yolox_s_tlr_416x416_pedcar_t4dataset/yolox_s_tlr_416x416_pedcar_t4dataset.py /workspace/work_dirs/yolox_s_tlr_416x416_pedcar_t4dataset/epoch_300.pth
```

### 4. Deploy

```bash
python3 tools/detection2d/deploy_yolox.py /workspace/work_dirs/yolox_s_tlr_416x416_pedcar_t4dataset/epoch_300.pth --input_size 416 416 --model yolox-s --batch_size 1 --output_onnx_file tlr_car_ped_yolox_s_batch_1.onnx 
python3 tools/detection2d/deploy_yolox.py /workspace/work_dirs/yolox_s_tlr_416x416_pedcar_t4dataset/epoch_300.pth --input_size 416 416 --model yolox-s --batch_size 4 --output_onnx_file tlr_car_ped_yolox_s_batch_4.onnx 
python3 tools/detection2d/deploy_yolox.py /workspace/work_dirs/yolox_s_tlr_416x416_pedcar_t4dataset/epoch_300.pth --input_size 416 416 --model yolox-s --batch_size 6 --output_onnx_file tlr_car_ped_yolox_s_batch_6.onnx 

```

## Troubleshooting

## Reference
