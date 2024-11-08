# FRNet
## Summary

- [Support priority](https://github.com/tier4/autoware-ml/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier B
- ROS package: [lidar_frnet_py](https://github.com/tier4/lidar_frnet_py) (prototype)
- Supported dataset
  - [x] NuScenes
  - [ ] T4dataset
- Other supported feature
  - [x] Add script to make .onnx file (ONNX runtime)
  - [x] Add script to perform ONNX inference
  - [x] Add script to make .engine file (TensorRT runtime)
  - [x] Add script to perform TensorRT inference
  - [ ] Add unit test
- Limited feature

## Results and models

- [Deployed model](docs/deployed_model.md)
- [Archived model](docs/archived_model.md)

## Get started
### 1. Setup

- [Run setup environment at first](/tools/setting_environment/)
- Docker build for FRNet

```sh
DOCKER_BUILDKIT=1 docker build -t autoware-ml-frnet projects/FRNet/
```

- Run docker

```sh
docker run -it --rm --gpus all --shm-size=64g --name awml -p 6006:6006 -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml-frnet
```

### 2. Config

Change parameters for your environment by changing [base config file](configs/nuscenes/frnet_1xb4_nus-seg.py). `TRAIN_BATCH = 2` is appropriate for GPU with 8 GB VRAM.

```py
# user settings
TRAIN_BATCH = 4
ITERATIONS = 50000
VAL_INTERVAL = 1000
```

### 3. Dataset

Make sure you downloaded nuScenes lidar segmentation labels to dataset path `data/nuscenes`. Then run script:

```sh
python projects/FRNet/scripts/create_nuscenes.py
```

### 4. Train

```sh
python tools/detection3d/train.py projects/FRNet/configs/nuscenes/frnet_1xb4_nus-seg.py
```

### 5. Test

```sh
python tools/detection3d/test.py projects/FRNet/configs/nuscenes/frnet_1xb4_nus-seg.py work_dirs/frnet_1xb4_nus-seg/best_miou_iter_<ITER>.pth
```

You can also visualize inference using Torch checkpoint via MMDet3D backend.
```sh
python tools/detection3d/test.py projects/FRNet/configs/nuscenes/frnet_1xb4_nus-seg.py work_dirs/frnet_1xb4_nus-seg/best_miou_iter_<ITER>.pth --show --task lidar_seg
```

For ONNX & TensorRT execution, check the next section.

### 6. Deploy & inference

Provided script allows for deploying at once to ONNX and TensorRT. In addition, it's possible to perform inference on test set with chosen execution method.

```sh
python projects/FRNet/deploy/main.py work_dirs/frnet_1xb4_nus-seg/best_miou_iter_<ITER>.pth --execution tensorrt --verbose
```

For more information:
```sh
python projects/FRNet/deploy/main.py --help
```

## Troubleshooting

* Can't deploy to TensorRT engine - foreign node issue.

  Model uses ScatterElements operation which is available since TensorRT 10.0.0. Update your TensorRT library to 10.0.0 at least.

## Reference

- Xiang Xu, Lingdong Kong, Hui Shuai and Qingshan Liu. "FRNet: Frustum-Range Networks for Scalable LiDAR Segmentation" arXiv preprint arXiv:2312.04484 (2024).
- https://github.com/Xiangxu-0103/FRNet
