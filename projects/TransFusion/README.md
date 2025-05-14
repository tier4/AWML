# TransFusion
## Summary

- [Support priority](https://github.com/tier4/AWML/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier S
- ROS package: [autoware_lidar_transfusion](https://github.com/autowarefoundation/autoware.universe/tree/main/perception/autoware_lidar_transfusion)
- Supported dataset
  - [x] NuScenes
  - [x] T4dataset
- Supported model
  - [x] LiDAR-only model (pillar)
- Other supported feature
  - [x] Add script to make .onnx file and deploy to Autoware
  - [ ] Add unit test
- Limited feature
  - For now, Camera-LiDAR fusion model is not supported in `AWML` because of performance.

## Results and models

- TransFusion-L
  - v0
    - [TransFusion-L base/0.X](./docs/TransFusion-L/v0/base.md)
- TransFusion-L-offline
  - v0
    - [TransFusion-L-offline base/0.X](./docs/TransFusion-L-offline/v0/base.md)
- TransFusion-L-nearby
  - v0
    - [TransFusion-L-nearby base/0.X](./docs/TransFusion-L-nearby/v0/base.md)

## Get started
### 1. Setup

- [Run setup environment at first](/tools/setting_environment/)
- Docker build for TransFusion

```sh
DOCKER_BUILDKIT=1 docker build -t autoware-ml-transfusion projects/TransFusion/
```

- Run docker

```sh
docker run -it --rm --gpus all --shm-size=64g --name awml -p 6006:6006 -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml-transfusion
```

- Build and install dependency (required only at first run).

```sh
python projects/TransFusion/setup.py develop
```

### 2. config

- Change parameters for your environment by changing [base config file](configs/t4dataset/transfusion_lidar_pillar_second_secfpn_1xb1_t4xx1-base.py).

```py
# user setting
info_directory_path = "info/user_name/"
data_root = "data/t4dataset/"
max_epochs = 50
backend_args = None
lr = 0.0001  # learning rate
```

### 3. Train

- (Choice) Train for nuScenes with single GPU

```sh
python tools/detection3d/train.py projects/TransFusion/configs/nuscenes/transfusion_lidar_pillar02_second_secfpn_1xb8-cyclic-20e_nus-3d.py
```

- (Choice) Train for nuScenes with multi GPU
  - Rename and change [config file](configs/nuscenes/transfusion_lidar_pillar02_second_secfpn_2xb8-cyclic-20e_nus-3d.py) to use for multi GPU and batch size

```sh
bash tools/detection3d/dist_train.sh projects/TransFusion/configs/nuscenes/transfusion_lidar_pillar02_second_secfpn_2xb8-cyclic-20e_nus-3d.py 2
```

- (Choice) Train for T4dataset with single GPU
  - batch size
    - The parameter of batch size can be set by command.
    - If you use RTX3090 GPU, we recommend to set batch size parameter from 4 to 8.
    - If you use A100 GPU, we recommend to set batch size parameter from 16 to 32.

```sh
python tools/detection3d/train.py {config file} \
--cfg-options train_dataloader.batch_size=4 --cfg-options auto_scale_lr.base_batch_size=4
```

- (Choice) Train for T4dataset with multi GPU
  - auto_scale_lr.base_batch_size = batch size * GPU number

```sh
bash ./tools/detection3d/dist_train.sh {config file} 2 \
--cfg-options train_dataloader.batch_size=4 --cfg-options auto_scale_lr.base_batch_size=8
```

### 4. Deploy

- Make onnx file for TransFusion

```sh
# Deploy for nuScenes dataset
python tools/detection3d/deploy.py projects/TransFusion/configs/deploy/transfusion_lidar_tensorrt_dynamic-20x5.py projects/TransFusion/configs/nuscenes/transfusion_lidar_pillar02_second_secfpn_1xb8-cyclic-20e_nus-3d.py work_dirs/transfusion_lidar_pillar02_second_secfpn_1xb8-cyclic-20e_nus-3d/epoch_20.pth data/nuscenes/samples/LIDAR_TOP/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin --device cuda:0 --work-dir /workspace

# Deploy for t4xx1 dataset
DIR="work_dirs/transfusion_lidar_pillar_second_secfpn_1xb1_90m-768grid-t4xx1" && \
python tools/detection3d/deploy.py projects/TransFusion/configs/deploy/transfusion_lidar_tensorrt_dynamic-20x5.py $DIR/transfusion_lidar_pillar_second_secfpn_1xb1_90m-768grid-t4xx1.py $DIR/epoch_50.pth data/t4dataset/db_jpntaxi_v2/0171a378-bf91-420e-9206-d047f6d1139a/0/data/LIDAR_CONCAT/0.pcd.bin --device cuda:0 --work-dir /workspace/$DIR/onnx
```

- Fix the graph

```sh
DIR="work_dirs/transfusion_lidar_pillar_second_secfpn_1xb1_90m-768grid-t4xx1" && \
python projects/TransFusion/scripts/fix_graph.py $DIR/onnx/end2end.onnx
```

- Move onnx file for Autoware data directory and TransFusion can be used in ROS environment by [lidar_transfusion](https://github.com/autowarefoundation/autoware.universe/tree/main/perception/lidar_transfusion).

```sh
DIR="work_dirs/transfusion_lidar_pillar_second_secfpn_1xb1_90m-768grid-t4xx1" && \
mv $DIR/onnx/transfusion.onnx ~/autoware_data/lidar_transfusion
```

## Troubleshooting
### Failed to create TensorRT engine

Currently `mmdeploy` supports CUDA 11 only, where autoware-ml Docker image base on CUDA 12.
You can validate mmdeploy's missing dependencies with command:

```sh
ldd /opt/conda/lib/python3.10/site-packages/mmdeploy/lib/libmmdeploy_tensorrt_ops.so
```

Nevertheless, `onnx` file should be created successfully.

## Reference

- Xuyang Bai, Zeyu Hu, Xinge Zhu, Qingqiu Huang, Yilun Chen, Hongbo Fu and Chiew-Lan Tai. "TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers." arXiv preprint arXiv:2203.11496 (2022).
- https://github.com/open-mmlab/mmdetection3d/pull/2547
