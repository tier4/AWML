# BEVFusion
## Summary

- [Support priority](https://github.com/tier4/autoware-ml/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier A
- ROS package: [Link](https://github.com/knzo25/bevfusion_ros2)
- Supported datasets
  - [x] NuScenes
  - [x] T4dataset
- Supported models
  - [x] LiDAR-only model
  - [x] Camera-LiDAR fusion model
- Other supported features
  - [x] ONNX export & TensorRT inference
  - [ ] Camera backbone pretraining
  - [ ] `fp16` inference
  - [ ] Add unit test

## Results and models

- BEVFusion-L
  - v0
    - [BEVFusion-L base/0.X](./docs/BEVFusion-L/v0/base.md)
- BEVFusion-CL
  - v0
    - [BEVFusion-CL base/0.X](./docs/BEVFusion-CL/v0/base.md)
- BEVFusion-L-offline
  - v0
    - [BEVFusion-L-offline base/0.X](./docs/BEVFusion-L-offline/v0/base.md)

## Get started
### 1. Setup

- [Run environment setup first](/tools/setting_environment/)
- Run docker

```sh
docker run -it --rm --gpus all --shm-size=64g --name awml -p 6006:6006 -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml
```

- Build and install dependencies (required only the first time)

```sh
python projects/BEVFusion/setup.py develop
```

- (Choice) Install traveller59's sparse convolutions backend

By default, mmcv's backend will be used, but the commonly adopted backend is traveller59's, which is also includes deployment concerns in its design such as memory allocation. For this reason it is highly recommended to install it:

```bash
pip install spconv-cu120
```

`autoware-ml` will automatically select this implementation if the dependency is installed.

### 2. Train
#### 2.1. Train the LiDAR-only model first

- (Choice) Train with a single GPU

```sh
# nuScenes
python tools/detection3d/train.py projects/BEVFusion/configs/nuscenes/bevfusion_lidar_voxel0075_second_secfpn_1xb1-cyclic-20e_nus-3d.py

# T4dataset
python tools/detection3d/train.py projects/BEVFusion/configs/t4dataset/bevfusion_lidar_voxel0075_second_secfpn_1xb1-cyclic-20e_t4xx1.py
```

- (Choice) Train with multiple GPUs
  - Rename the config file to use multi GPUs and different batch sizes (e.g., `2xb8` means using 2 GPUs and a batch size of 8)

```sh
# T4dataset
bash tools/detection3d/dist_train.sh projects/BEVFusion/configs/t4dataset/bevfusion_lidar_voxel_second_secfpn_1xb1_t4xx1.py 2
```

#### 2.2. [Option] Train the camera backbone

- Download the [Swin pre-trained model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/swint-nuimages-pretrained.pth).
- (Choice) If you want to train the image backbone for fine tuning in the T4dataset, you can do it as follows:

```sh
TBD
```

#### 2.3. [Option] Train the Camera-LiDAR fusion model

- Note that if you want to reduce CUDA memory usage and computational overhead, you can add `--amp` on the tail of the previous commands. The models under this setting will be trained in `fp16` mode.
- (Choice) Train using a single GPU

```sh
# nuScenes
python tools/detection3d/train.py projects/BEVFusion/configs/nuscenes/bevfusion_lidar-cam_voxel0075_second_secfpn_1xb2-cyclic-20e_nus-3d.py --cfg-options load_from=${LIDAR_PRETRAINED_CHECKPOINT} model.img_backbone.init_cfg.checkpoint=${IMAGE_PRETRAINED_BACKBONE}
```

### 3. Deployment

#### 3.2. Sparse convolutions support

Sparse convolutions are not deployable by default. In the [deployment](configs/deploy/bevfusion_lidar_tensorrt_dynamic.py) we follow the instructions found in the [SparseConvolution](../SparseConvolution/README.md) project to enable this feature.

Note: we only support traveller59's backend during deployment, but the model checkpoints can correspond to either backend.

#### 3.3. ONNX export

We provide two general deploy config files:
 - [lidar-only](configs/deploy/bevfusion_lidar_tensorrt_dynamic.py)
 - [camera-lidar](configs/deploy/bevfusion_camera_lidar_tensorrt_dynamic.py)

To export an ONNX, use the following command:

```bash
DEPLOY_CFG=projects/BEVFusion/configs/deploy/bevfusion_lidar_tensorrt_dynamic-20x5.py
MODEL_CFG=...
CHECKPOINT_PATH=...
WORK_DIR=...

python projects/BEVFusion/deploy/export.py \
  ${DEPLOY_CFG} \
  ${MODEL_CFG} \
  ${CHECKPOINT_PATH} \
  --device cuda:0 \
  --work-dir ${WORK_DIR}
```

This will generate two models in the `WORK_DIR` folder. `end2end.onnx` corresponds to the standard exported model ,whereas `end2end_fixed.onnx` contains a fix for the `TopK` operator (compatibility issues between `mmdeploy` and `TensorRT`).

## TODO

 - Fix graph to remove redundant spconv operations
 - `fp16` inference (TensorRT plugin)
 - Fix BEVFusion ROIs for t4dataset
 - Add self-supervised loss

## Reference

- [BEVFusion of mmdetection3d](https://github.com/open-mmlab/mmdetection3d/tree/v1.4.0/projects/BEVFusion)
