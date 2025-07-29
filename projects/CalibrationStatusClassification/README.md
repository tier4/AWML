# CalibrationStatusClassification
## Summary

- [Support priority](https://github.com/tier4/AWML/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier B
- ROS package: [package_name](https://github.com/autowarefoundation/autoware.universe/tree/main/perception/)
- Supported dataset
  - [ ] NuScenes
  - [ ] T4dataset
- Supported model
  - [x] ResNet18
- Other supported feature
  - [x] Add script to make .onnx file and deploy to Autoware
  - [ ] Add unit test
- Limited feature

## Results and models

- [Deployed model](docs/deployed_model.md)
- [Archived model](docs/archived_model.md)

## Get started
### 1. Setup

- [Run setup environment at first](/tools/setting_environment/)
- Docker build for CalibrationStatusClassification

```sh
DOCKER_BUILDKIT=1 docker build -t autoware-ml-calib projects/CalibrationStatusClassification/
```

- Run docker

```sh
docker run -it --rm --gpus all --shm-size=64g --name awml -p 6006:6006 -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml-calib
```

### 2. Config

- Change parameters for your environment by changing [base config file](configs/t4dataset/resnet18_5ch_1xb8-25e_t4base.py).

```py
batch_size = 8
max_epochs = 25
```

### 3. Dataset

TBD

### 4. Train

Make sure the dataset structure is as follows:

```
|workspace
    └──|data
          └── calibrated_data
              └── training_set
              |  └── data
              |       ├── 0_calibration.npz
              |       ├── 0_image.jpg
              |       ├── 0_pointcloud.npz
              |       ├── 1_calibration.npz
              |       ├── 1_image.jpg
              |       ├── 1_pointcloud.npz
              |       ...
              └── validation_set
                  └── data
                      ├── 0_calibration.npz
                      ├── 0_image.jpg
                      ├── 0_pointcloud.npz
                      ├── 1_calibration.npz
                      ├── 1_image.jpg
                      ├── 1_pointcloud.npz
                      ...
```

Run training:

- Single GPU:
```sh
python tools/classification2d/train.py projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb8-25e_t4base.py
```

- Multi GPU (example with 2 GPUs):
```sh
./tools/classification2d/dist_train.sh projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb8-25e_t4base.py 2
```

### 5. Deploy

Example commands for deployment (modify paths if needed):
- mmdeploy script:
```sh
python3 tools/classification2d/deploy.py projects/CalibrationStatusClassification/configs/deploy/resnet18_5ch.py projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb8-25e_t4base.py work_dirs/resnet18_5ch_1xb8-25e_t4base/epoch_25.pth data/calibrated_data/training_set/data/0_image.jpg 1 --device cuda:0 --work-dir /workspace/work_dirs/
```

- Custom script (with verification):
```sh
python3 projects/CalibrationStatusClassification/deploy/main.py projects/CalibrationStatusClassification/configs/deploy/resnet18_5ch.py projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb8-25e_t4base.py work_dirs/resnet18_5ch_1xb8-25e_t4base/epoch_25.pth data/calibrated_data/validation_set/data/0_image.jpg  --device cuda:0 --work-dir /workspace/work_dirs/ --verify
```

## Troubleshooting

## Reference
