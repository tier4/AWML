# YOLOX_opt_elan
## Summary

- [Support priority](https://github.com/tier4/AWML/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier A
- ROS package
  - [tensorrt_yolox](https://github.com/autowarefoundation/autoware.universe/tree/main/perception/tensorrt_yolox)
- Supported dataset
  - [ ] COCO dataset
  - [ ] T4dataset
- Supported model
- Other supported feature
  - [ ] Add script to make .onnx file and deploy to Autoware
  - [ ] Add unit test
- Limited feature

## Results and models

- YOLOX_opt-S-elan

## Get started
### 1. Setup

- [Run setup environment at first](/tools/setting_environment/)

### 2. Train

- For dynamic recognition

```bash
python3 tools/detection2d/train.py /workspace/projects/YOLOX_opt_elan/configs/t4dataset/YOLOX_opt-S-DynamicRecognition/yolox-s-opt-elan_960x960_300e_t4dataset.py
```

### 3. Evaluate

- For dynamic recognition

```bash
python tools/detection2d/test.py projects/YOLOX_opt_elan/configs/t4dataset/yolox-s-opt-elan_960x960_300e_t4dataset.py /workspace/work_dirs/yolox-s-opt-elan_960x960_300e_t4dataset/epoch_300.pth
```
### 4. Deploy

- For dynamic recognition

```bash
python3 tools/detection2d/deploy_yolox_s_opt.py /workspace/work_dirs/yolox-s-opt-elan_960x960_300e_t4dataset/epoch_300.pth yolox-s+-opt-T4-960x960.py --batch_size 6 --output_onnx_file yolox_s_opt_elan_batch_6.onnx
```

## Latency


## Troubleshooting

## Reference

- Ge Zheng et al. "YOLOX: Exceeding YOLO Series in 2021", arXiv 2021
