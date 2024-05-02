# BEVFusion
## Supported feature

- [x] Train LiDAR-only model
- [ ] Train Camera-LiDAR fusion model
- [x] Train with single GPU
- [ ] Train with multiple GPU
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

1. Train the LiDAR-only model first:

```sh
# for nuScenes with single GPU
python tools/detection3d/train.py projects/BEVFusion/configs/nuscenes/bevfusion_lidar_voxel0075_second_secfpn_1xb1-cyclic-20e_nus-3d.py

# for nuScenes with multiple GPU
# Rename config file to use for multi GPU and batch size
# bash tools/dist_train.sh projects/BEVFusion/configs/nuscenes/bevfusion_lidar_voxel0075_second_secfpn_2xb2-cyclic-20e_nus-3d.py 2

# for T4dataset with single GPU
python tools/detection3d/train.py projects/BEVFusion/configs/t4dataset/bevfusion_lidar_voxel0075_second_secfpn_1xb1-cyclic-20e_t4xx1.py
```

2. Train with Camera-LiDAR fusion model
  - Download the [Swin pre-trained model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/swint-nuimages-pretrained.pth).
  - Given the image pre-trained backbone and the lidar-only pre-trained detector, you could train the lidar-camera fusion model.
  - Note that if you want to reduce CUDA memory usage and computational overhead, you could directly add --amp on the tail of the above commands. The model under this setting will be trained in fp16 mode.

```sh
python tools/detection3d/train.py projects/BEVFusion/configs/nuscenes/bevfusion_lidar-cam_voxel0075_second_secfpn_1xb2-cyclic-20e_nus-3d.py --cfg-options load_from=${LIDAR_PRETRAINED_CHECKPOINT} model.img_backbone.init_cfg.checkpoint=${IMAGE_PRETRAINED_BACKBONE}
```

### 3. Deploy

TBD

## Results and models
### NuScenes

- [LiDAR only model (spconv, voxel 0.075)](./configs/bevfusion_lidar_voxel0075_second_secfpn_1xb1-cyclic-20e_nus-3d.py)
  - [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth)
  - [logs](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_20230322_053447.log)

```
car_AP_dist_1.0: 0.8771
truck_AP_dist_1.0: 0.5971
construction_vehicle_AP_dist_1.0: 0.1929
bus_AP_dist_1.0: 0.7284
trailer_AP_dist_1.0: 0.4088
barrier_AP_dist_1.0: 0.7009
motorcycle_AP_dist_1.0: 0.7197
bicycle_AP_dist_1.0: 0.5468
pedestrian_AP_dist_1.0: 0.8648
traffic_cone_AP_dist_1.0: 0.7388
mAP: 0.6485
```

- [Camera-LiDAR model (spconv, voxel 0.075)](./configs/bevfusion_lidar-cam_voxel0075_second_secfpn_1xb1-cyclic-20e_nus-3d.py)
  - [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth)
  - [logs](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_20230524_001539.log)

```
car_AP_dist_1.0: 0.8828
truck_AP_dist_1.0: 0.5359
construction_vehicle_AP_dist_1.0: 0.2179
bus_AP_dist_1.0: 0.7329
trailer_AP_dist_1.0: 0.3807
barrier_AP_dist_1.0: 0.7235
motorcycle_AP_dist_1.0: 0.7621
bicycle_AP_dist_1.0: 0.6124
pedestrian_AP_dist_1.0: 0.8738
traffic_cone_AP_dist_1.0: 0.7803
mAP: 0.6609
```

### T4 dataset

TBD
