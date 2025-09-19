# CenterPoint
## Summary

- [Support priority](https://github.com/tier4/AWML/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier S
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

- CenterPoint
  - v1 (121m range, grid_size = 760)
    - [CenterPoint base/1.X](./docs/CenterPoint/v1/base.md)
    - [CenterPoint x2/1.X](./docs/CenterPoint/v1/x2.md)
    - [CenterPoint-ConvNeXtPC base/0.x](./docs/CenterPoint-ConvNeXtPC/v0/base.md)

## Get started
### 1. Setup

- Please follow the [installation tutorial](/docs/tutorial/tutorial_detection_3d.md)to set up the environment.
- Run docker

```sh
docker run -it --rm --gpus all --shm-size=64g --name awml -p 6006:6006 -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml
```

### 2. Train
#### 2.1 Environment set up

Set `CUBLAS_WORKSPACE_CONFIG` for the deterministic behavior, plese check this [nvidia doc](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility) for more info

```sh
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

#### 2.2. Train CenterPoint model with T4dataset-base

- [choice] Train with a single GPU
  - Rename config file to use for single GPU and batch size
  - Change `train_batch_size` and `train_gpu_size` accordingly

```sh
# T4dataset (121m)
python tools/detection3d/train.py projects/CenterPoint/configs/t4dataset/second_secfpn_2xb8_121m_base.py
```

- [choice] Train with multi GPU

```sh
# Command
bash tools/detection3d/dist_script.sh <config> <number of gpus> train

# Example: T4dataset
bash tools/detection3d/dist_script.sh projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp.py 4 train
```

### 3. Evaluation

- Run evaluation on a test set, please select experiment config accordingly

- [choice] Evaluate with a single GPU

```sh
# Evaluation for t4dataset
DIR="work_dirs/centerpoint/t4dataset/second_secfpn_2xb8_121m_base/" && \
python tools/detection3d/test.py projects/CenterPoint/configs/t4dataset/second_secfpn_2xb8_121m_base.py $DIR/epoch_50.pth
```

- [choice] Evaluate with multiple GPUs
  - Note that if you choose to evaluate with multiple GPUs, you might get slightly different results as compared to single GPU due to differences across GPUs

```sh
# Command
CHECKPOINT_PATH=<checkpoint> && \
bash tools/detection3d/dist_script.sh <config> <number of gpus> test $CHECKPOINT_PATH

# Example: T4dataset
CHECKPOINT_PATH="work_dirs/centerpoint/t4dataset/second_secfpn_2xb8_121m_base/epoch_50.pth" && \
bash tools/detection3d/dist_script.sh projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp.py 4 test $CHECKPOINT_PATH
```

### 4. Visualization

- Run inference and visualize bounding boxes from a CenterPoint model

```sh
# Inference for t4dataset
DIR="work_dirs/centerpoint/t4dataset/second_secfpn_2xb8_121m_base/" &&
python projects/CenterPoint/scripts/inference.py projects/CenterPoint/configs/t4dataset/second_secfpn_2xb8_121m_base.py $DIR/epoch_50.pth --ann-file-path <info pickle file> --bboxes-score-threshold 0.35 --frame-range 700 1100
```

where `frame-range` represents the range of frames to visualize.

### 5. Deploy

- Make an onnx file for a CenterPoint model

```sh
# Deploy for t4dataset
DIR="work_dirs/centerpoint/t4dataset/second_secfpn_2xb8_121m_base/" &&
python projects/CenterPoint/scripts/deploy.py projects/CenterPoint/configs/t4dataset/second_secfpn_2xb8_121m_base.py $DIR/epoch_50.pth --replace_onnx_models --device gpu --rot_y_axis_reference
```

where `rot_y_axis_reference` can be removed if we would like to use the original counterclockwise x-axis rotation system.

## Troubleshooting

- The difference from original CenterPoint from mmdetection3d v1
  - To maintain the backward compatibility with the previous ML library, we modified the original CenterPoint from mmdetection3d v1 such as:
    - Exclude voxel center from z-dimension as part of pillar features
    - Assume that the rotation system in the deployed ONNX file is in clockwise y-axis, and a bounding box is [x, y, z, w, l, h] for the deployed ONNX file
    - Do not use CBGS dataset to align the experiment configuration with the older library
- Latest mmdetection3D assumes the lidar coordinate system is in the right-handed x-axis reference, also the dimensionality of a bounding box is [x, y, z, l, w, h], please check [this](https://mmdetection3d.readthedocs.io/en/latest/user_guides/coord_sys_tutorial.html) for more details

## Reference

- [CenterPoint of mmdetection3d](https://github.com/open-mmlab/mmdetection3d/tree/main/configs/centerpoint)
