## Supported feature

- [x] Train LiDAR-only model
- [ ] Train Camera-LiDAR fusion model
- [x] Train with single GPU
- [ ] Train with multiple GPU
- [ ] Add script to make .onnx file and deploy to Autoware
- [ ] Add unit test

## Usage
### Setup

- Run docker and compile

```sh
python projects/BEVFusion/setup.py develop
```

## Train

1. Train the LiDAR-only model first:

```sh
# for nuScenes with single GPU
python tools/train.py projects/BEVFusion/configs/nuscenes/bevfusion_lidar_voxel0075_second_secfpn_1xb1-cyclic-20e_nus-3d.py

# for nuScenes with multiple GPU
# Rename config file to use for multi GPU and batch size
# bash tools/dist_train.sh projects/BEVFusion/configs/nuscenes/bevfusion_lidar_voxel0075_second_secfpn_2xb2-cyclic-20e_nus-3d.py 2

# for T4dataset with single GPU
python tools/train.py projects/BEVFusion/configs/t4dataset/bevfusion_lidar_voxel0075_second_secfpn_1xb1-cyclic-20e_t4xx1.py
```

2. Train with Camera-LiDAR fusion model
   - Download the [Swin pre-trained model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/swint-nuimages-pretrained.pth).
   - Given the image pre-trained backbone and the lidar-only pre-trained detector, you could train the lidar-camera fusion model.
   - Note that if you want to reduce CUDA memory usage and computational overhead, you could directly add --amp on the tail of the above commands. The model under this setting will be trained in fp16 mode.

```sh
python tools/train.py projects/BEVFusion/configs/nuscenes/bevfusion_lidar-cam_voxel0075_second_secfpn_1xb2-cyclic-20e_nus-3d.py --cfg-options load_from=${LIDAR_PRETRAINED_CHECKPOINT} model.img_backbone.init_cfg.checkpoint=${IMAGE_PRETRAINED_BACKBONE}
```

### Deploy

TBD

## Results and models
### NuScenes

- [LiDAR only model (voxel 0.075)](./configs/bevfusion_lidar_voxel0075_second_secfpn_1xb1-cyclic-20e_nus-3d.py)
  - mAP 64.9
  - [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth)
  - [logs](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_20230322_053447.log)
- [Camera-LiDAR model (voxel 0.075)](./configs/bevfusion_lidar-cam_voxel0075_second_secfpn_1xb1-cyclic-20e_nus-3d.py)
  - mAP 68.6
  - [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth)
  - [logs](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_20230524_001539.log)

### T4 dataset
