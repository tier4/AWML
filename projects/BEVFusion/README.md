# BEVFusion
## Summary

- [Support priority](https://github.com/tier4/autoware-ml/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier A
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

## Results and models

- [Deployed model](docs/deployed_model.md)
- [Archived model](docs/archived_model.md)

## Get started
### 1. Setup

- [Run setup environment at first](/tools/setting_environment/)
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

- (Choice) Train with single GPU

```sh
# nuScenes
python tools/detection3d/train.py projects/BEVFusion/configs/nuscenes/bevfusion_lidar_voxel0075_second_secfpn_1xb1-cyclic-20e_nus-3d.py

# T4dataset
python tools/detection3d/train.py projects/BEVFusion/configs/t4dataset/bevfusion_lidar_voxel0075_second_secfpn_1xb1-cyclic-20e_t4xx1.py
```

- (Choice) Train with multi GPU
  - Rename config file to use for multi GPU and batch size

```sh
# T4dataset
bash tools/detection3d/dist_train.sh projects/BEVFusion/configs/t4dataset/bevfusion_lidar_voxel_second_secfpn_1xb1_t4xx1.py 2
```

#### 2.2. [Option] Train the camera backbone

- Download the [Swin pre-trained model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/swint-nuimages-pretrained.pth).
- (Choice) If you want to train the image backbone for fine tuning in T4dataset. you train as below

```sh
TBD
```

#### 2.3. [Option] Train with Camera-LiDAR fusion model

- Note that if you want to reduce CUDA memory usage and computational overhead, you could directly add --amp on the tail of the above commands. The model under this setting will be trained in fp16 mode.
- (Choice) Train with single GPU

```sh
# nuScenes
python tools/detection3d/train.py projects/BEVFusion/configs/nuscenes/bevfusion_lidar-cam_voxel0075_second_secfpn_1xb2-cyclic-20e_nus-3d.py --cfg-options load_from=${LIDAR_PRETRAINED_CHECKPOINT} model.img_backbone.init_cfg.checkpoint=${IMAGE_PRETRAINED_BACKBONE}
```

### 3. Deploy

TBD

## Results and models

## Troubleshooting

## Reference

- [BEVFusion of mmdetection3d](https://github.com/open-mmlab/mmdetection3d/tree/v1.4.0/projects/BEVFusion)
