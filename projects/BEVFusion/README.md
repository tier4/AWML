# BEVFusion
## Summary

- ROS package: TBD
- Supported dataset
  - [x] NuScenes
  - [x] T4dataset
- Supported model
  - [x] LiDAR-only model (spconv)
  - [ ] Camera-LiDAR fusion model (spconv)
- Other supported feature
  - [ ] Add script to make .onnx file and deploy to Autoware
  - [ ] Add unit test

## Get started
### 1. Setup

- Run docker

```sh
docker run -it --rm --gpus all --shm-size=64g --name awml -p 6006:6006 -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml
```

- Build and install dependency (required only at first run).

```sh
python projects/BEVFusion/setup.py develop
```

### 2. Train
#### 2.1. Train the LiDAR-only model first

- [choice] Train with single GPU

```sh
# nuScenes
python tools/detection3d/train.py projects/BEVFusion/configs/nuscenes/bevfusion_lidar_voxel0075_second_secfpn_1xb1-cyclic-20e_nus-3d.py

# T4dataset
python tools/detection3d/train.py projects/BEVFusion/configs/t4dataset/bevfusion_lidar_voxel0075_second_secfpn_1xb1-cyclic-20e_t4xx1.py
```

- [choice] Train with multi GPU
  - Rename config file to use for multi GPU and batch size

```sh
# T4dataset
bash tools/detection3d/dist_train.sh projects/BEVFusion/configs/t4dataset/bevfusion_lidar_voxel_second_secfpn_1xb1_t4xx1.py 2
```

#### 2.2. [Option] Train the camera backbone

- Download the [Swin pre-trained model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/swint-nuimages-pretrained.pth).
- [choice] If you want to train the image backbone for fine tuning in T4dataset. you train as below

```sh
TBD
```

#### 2.3. [Option] Train with Camera-LiDAR fusion model

- Note that if you want to reduce CUDA memory usage and computational overhead, you could directly add --amp on the tail of the above commands. The model under this setting will be trained in fp16 mode.
- [choice] Train with single GPU

```sh
# nuScenes
python tools/detection3d/train.py projects/BEVFusion/configs/nuscenes/bevfusion_lidar-cam_voxel0075_second_secfpn_1xb2-cyclic-20e_nus-3d.py --cfg-options load_from=${LIDAR_PRETRAINED_CHECKPOINT} model.img_backbone.init_cfg.checkpoint=${IMAGE_PRETRAINED_BACKBONE}
```

### 3. Deploy

TBD

## Results and models
### NuScenes

|                       | mAP  | Car  | Truck | CV   | Bus  | Tra  | Bar  | Mot  | Bic  | Ped  | Cone |
| --------------------- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| BEVFusion-L (spconv)  | 64.6 | 87.7 | 59.7  | 19.3 | 72.8 | 40.9 | 70.0 | 72.0 | 54.7 | 86.5 | 73.9 |
| BEVFusion-CL (spconv) | 66.1 | 88.3 | 53.6  | 21.8 | 73.3 | 38.1 | 72.4 | 76.2 | 61.2 | 87.4 | 78.0 |

- [LiDAR only model (spconv, voxel 0.075)](./configs/bevfusion_lidar_voxel0075_second_secfpn_1xb1-cyclic-20e_nus-3d.py)
  - [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth)
  - [logs](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_20230322_053447.log)
- [Camera-LiDAR model (spconv, voxel 0.075)](./configs/bevfusion_lidar-cam_voxel0075_second_secfpn_1xb1-cyclic-20e_nus-3d.py)
  - [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth)
  - [logs](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_20230524_001539.log)

### T4 dataset

TBD
