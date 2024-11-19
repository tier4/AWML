# CenterPoint
## Summary

- [Support priority](https://github.com/tier4/autoware-ml/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier S
- ROS package: [auotoware_lidar_centerpoint] (https://github.com/autowarefoundation/autoware.universe/tree/main/perception/autoware_lidar_centerpoint)
- Supported dataset
  - [x] T4dataset
  - [] NuScenes
- Supported model
  - [x] LiDAR-only model
- Other supported feature
  - [x] Add script to make .onnx file and deploy to Autoware
  - [x] Manual visualization of T4dataset
  - [ ] Add unit tests

## Results and models

- [Comparison](https://tier4.atlassian.net/wiki/spaces/~712020b277b703d991438d8305ae640615ecab/pages/3376841030/Training+in+autoware-ml)
- Deployed Models
    - CenterPoint-T4Base v1.0.0
      - [T4datasets - T4Base](./docs/t4datasets/v1.0.0/deployed_t4base_model.md)

## Get started
### 1. Setup

- [Run setup environment](../../tools/setting_environment/README.md)
- Run docker

```sh
docker run -it --rm --gpus all --shm-size=64g --name awml -p 6006:6006 -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml
```

### 2. Implementation details:
To maintain the backward compatibility with `AWMLdetection3d`, which is the previous ML library, we 
modified the original CenterPoint from mmdetection3d v1 such as:
- Exclude voxel center from z-dimension as part of pillar features
- Assume that the rotation system in the deployed ONNX file is in clockwise y-axis, and a bounding box is [x, y, z, w, l, h] for the deployed ONNX file
- Do not use CBGS dataset to align the experiment configuration with the older library

### 3. Train
#### 3.1 Environment set up
Set `CUBLAS_WORKSPACE_CONFIG` for the deterministic behavior, plese check this [nvidia doc](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility) 
for more info
```sh
export CUBLAS_WORKSPACE_CONFIG=:4096:8
``` 
#### 3.2. Train CenterPoint model with T4dataset-base
- [choice] Train with single GPU
  - Rename config file to use for single GPU and batch size
  - Change `train_batch_size` and `train_gpu_size` accordingly

```sh
# T4dataset (121m)
python tools/detection3d/train.py projects/CenterPoint/configs/t4dataset/second_secfpn_2xb8_121m_base.py
```

- [choice] Train with multi GPU

```sh
# T4dataset
bash tools/detection3d/dist_train.sh projects/CenterPoint/configs/t4dataset/second_secfpn_2xb8_121m_base.py 2
```

### 4. Evaluation
- Run evaluation on a test set, please select experiment config accordingly
```sh
# Evaluation for t4dataset
DIR="work_dirs/centerpoint/t4dataset/second_secfpn_2xb8_121m_base/" &&
python tools/detection3d/test.py projects/CenterPoint/configs/t4dataset/second_secfpn_2xb8_121m_base.py $DIR/epoch_50.pth
```

### 5. Inference 
- Run inference and visualize bboxes from a CenterPoint model
```sh
# Inference for t4dataset
DIR="work_dirs/centerpoint/t4dataset/second_secfpn_2xb8_121m_base/" &&
python projects/CenterPoint/scripts/inference.py projects/CenterPoint/configs/t4dataset/second_secfpn_2xb8_121m_base.py $DIR/epoch_50.pth --ann-file-path <info pickle file> --bboxes-score-threshold 0.35 --frame-range 700 1100
```
where `frame-range` represents the range of frames to visualze

### 6. Deploy
- Make an onnx file for a CenterPoint model
```sh
# Deploy for t4dataset
DIR="work_dirs/centerpoint/t4dataset/second_secfpn_2xb8_121m_base/" &&
python projects/CenterPoint/scripts/deploy.py projects/CenterPoint/configs/t4dataset/second_secfpn_2xb8_121m_base.py $DIR/epoch_50.pth --replace_onnx_models --device gpu --rot_y_axis_reference
```
where `rot_y_axis_reference` can be removed if we would like to use the original counterckiwise x-axis rotation system 

## Troubleshooting
- Latest mmdetection3D assumes the lidar coordinate system is in the right-handed x-axis reference, also the dimensionality of a bounding box is [x, y, z, l, w, h], please check [this](https://mmdetection3d.readthedocs.io/en/latest/user_guides/coord_sys_tutorial.html) for more details  

## Reference
- [CenterPoint of mmdetection3d](https://github.com/open-mmlab/mmdetection3d/tree/main/configs/centerpoint)
