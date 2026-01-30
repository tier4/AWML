# FRNet
## Summary

- [Support priority](https://github.com/tier4/AWML/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier B
- ROS package: [lidar_frnet](https://github.com/autowarefoundation/autoware_universe/tree/main/perception/autoware_lidar_frnet)
- Supported dataset
  - [x] NuScenes
  - [x] T4dataset
- Other supported feature
  - [x] Add script to make .onnx file (ONNX runtime)
  - [x] Add script to perform ONNX inference
  - [x] Add script to make .engine file (TensorRT runtime)
  - [x] Add script to perform TensorRT inference
  - [ ] Add unit test
- Limited feature

## Results and models

- FRNet
  - v0
    - [FRNet base/0.X](./docs/FRNet/v0/base.md)

## Get started
### 1. Setup

- Please follow the [installation tutorial](/docs/tutorial/tutorial_detection_3d.md) to set up the environment.
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

- Train for nuScenes
```sh
python tools/detection3d/train.py projects/FRNet/configs/nuscenes/frnet_1xb4_nus-seg.py
```

- Train for T4dataset
```sh
python tools/detection3d/train.py projects/FRNet/configs/t4dataset/frnet_1xb8_t4dataset-seg.py
```

### 5. Test

- Test for nuScenes
```sh
python tools/detection3d/test.py projects/FRNet/configs/nuscenes/frnet_1xb4_nus-seg.py work_dirs/frnet_1xb4_nus-seg/best_miou_iter_<ITER>.pth
```

- Test for T4dataset
```sh
python tools/detection3d/test.py projects/FRNet/configs/t4dataset/frnet_1xb8_t4dataset-seg.py work_dirs/frnet_1xb8_t4dataset-seg/best_miou_iter_<ITER>.pth
```

- Visualize inference for nuScenes
```sh
python tools/detection3d/test.py projects/FRNet/configs/nuscenes/frnet_1xb4_nus-seg.py work_dirs/frnet_1xb4_nus-seg/best_miou_iter_<ITER>.pth --show --task lidar_seg
```

- Visualize inference for T4dataset
```sh
python tools/detection3d/test.py projects/FRNet/configs/t4dataset/frnet_1xb8_t4dataset-seg.py work_dirs/frnet_1xb8_t4dataset-seg/best_miou_iter_<ITER>.pth --show --task lidar_seg
```

For ONNX & TensorRT execution, check the next section.

### 6. Deploy & inference

Provided script allows for deploying at once to ONNX and TensorRT. In addition, it's possible to perform inference on test set with chosen execution method.

- Deploy for nuScenes
```sh
python projects/FRNet/deploy/main.py work_dirs/frnet_1xb4_nus-seg/best_miou_iter_<ITER>.pth --model-cfg projects/FRNet/configs/nuscenes/frnet_1xb4_nus-seg.py --deploy-cfg projects/FRNet/configs/deploy/frnet_tensorrt_dynamic.py --execution tensorrt --verbose
```

- Deploy for T4dataset
```sh
python projects/FRNet/deploy/main.py work_dirs/frnet_1xb8_t4dataset-seg/best_miou_iter_<ITER>.pth --model-cfg projects/FRNet/configs/t4dataset/frnet_1xb8_t4dataset-seg.py --deploy-cfg projects/FRNet/configs/deploy/t4dataset/frnet_tensorrt_dynamic.py --execution tensorrt --verbose
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
