# YOLOX_opt_elan
## Summary

- [Support priority](https://github.com/tier4/AWML/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier A
- ROS package
  - [tensorrt_yolox](https://github.com/autowarefoundation/autoware.universe/tree/main/perception/tensorrt_yolox)
- Supported dataset
  - [x] BDD100K dataset
  - [x] T4dataset
- Supported model
- Other supported feature
  - [x] Add script to make .onnx file and deploy to Autoware
  - [ ] Add unit test
- Limited feature

## Results and models

- YOLOX_opt-elan
  - Compared to YOLOX_opt, this model is optimized for computation on Xavier GPU and Xavier DLA.
    - On Xavier GPU, a model comparable to YOLO-l can be executed with an inference time of less than 10ms.
    - On DLA, The optimized YOLOX-s model achieves a latency of 16 milliseconds.
  - The main differences are as below:
    - STEM is simplified since slice processing is slow on Xaiver GPU while not supported on DLA.
    - SPP is removed since pooling kernel size larger than 8 are not supported on DLA. ASPP can be used as a substitute.
    - Activation function Swish is replaced with ReLU or ReLU6. ReLU6 is particularly useful during training for improving quantization robustness.
    - Depth Scaling is increase by a factor of 2 as same as YOLOX-m.
    - ELAN is utilized, as in YOLOv7 and YOLOv8.


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
python projects/YOLOX_opt_elan/scripts/deploy_yolox_s_opt.py work_dirs/yolox-s-opt-elan_960x960_300e_t4dataset/epoch_300.pth projects/YOLOX_opt_elan/scripts/sample/yolox-s+-opt-T4-960x960.py --batch_size 6 --output_onnx_file yolox_s_opt_elan_batch_6.onnx
```

## Latency


## Troubleshooting

## Reference

- Ge Zheng et al. "YOLOX: Exceeding YOLO Series in 2021", arXiv 2021
