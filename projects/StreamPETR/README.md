# Model name
## Summary

- [Support priority](https://github.com/tier4/autoware-ml/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier B
- ROS package: [package_name](https://github.com/autowarefoundation/autoware.universe/tree/main/perception/)
- Supported dataset
  - [ ] NuScenes
  - [ ] T4dataset
- Supported model
- Other supported feature
  - [ ] Add script to make .onnx file and deploy to Autoware
  - [ ] Add unit test
- Limited feature

## Results and models

- [Deployed model](docs/deployed_model.md)
- [Archived model](docs/archived_model.md)

## Get started
### 1. Setup

TBD

### 2. config

### 3. Train


### 4. Evaluation

```bash

python3 tools/detection3d/test.py /workspace/projects/StreamPETR/configs/t4dataset/t4_xx1_vov_flash_320x800_baseline.py "/workspace/work_dirs/t4_xx1_vov_flash_320x800_baseline/best_NuScenes metric_T4Metric_mAP_epoch_33.pth"

```

### 5. Deploy

```bash
python3 projects/StreamPETR/deploy/torch2onnx.py /workspace/projects/StreamPETR/configs/t4dataset/t4_xx1_vov_flash_320x800_baseline.py --section extract_img_feat --checkpoint "/workspace/work_dirs/t4_xx1_vov_flash_320x800_baseline/best_NuScenes metric_T4Metric_mAP_epoch_33.pth" 

python3 projects/StreamPETR/deploy/torch2onnx.py /workspace/projects/StreamPETR/configs/t4dataset/t4_xx1_vov_flash_320x800_baseline.py --section pts_head_memory --checkpoint "/workspace/work_dirs/t4_xx1_vov_flash_320x800_baseline/best_NuScenes metric_T4Metric_mAP_epoch_33.pth" 

python3 projects/StreamPETR/deploy/torch2onnx.py /workspace/projects/StreamPETR/configs/t4dataset/t4_xx1_vov_flash_320x800_baseline.py --section position_embedding --checkpoint "/workspace/work_dirs/t4_xx1_vov_flash_320x800_baseline/best_NuScenes metric_T4Metric_mAP_epoch_33.pth" 


```
## Troubleshooting

## Reference

# NOTES!!!
Currently distortion is not accounted for.
FOr the boundary of sequences, padded using first frame of the minibatch
Since this is camera-only. We need to assert that all data comes from the same sensor suite?
Also, Camera should be in the same order. Or maybe order should not matter, and maybe some cameras can be non-functional too, should I add that as augmentation?
GlobalRotScaleTransImage augmentation removed for now. Cannot understand the logic behind it
Maybe in ROSNode it is good to add a service to reset the memory at times
Input image muse be rectified beforehand. StreamPETR node, or training pipeline wont rectify.
# THINGS TODO

- Experiment with different training strategies (in order of priority)
  - Train with denoising
  - Temporal training with different temporal and losses frame count
  - Image order shuffle augmentation
  - Larger image backbone
  - Larger and smaller image sizes
  