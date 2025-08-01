# CalibrationStatusClassification
## Summary

- [Support priority](https://github.com/tier4/AWML/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier B
- ROS package: [package_name](https://github.com/autowarefoundation/autoware.universe/tree/main/perception/)
- Supported dataset
  - [ ] NuScenes
  - [x] T4dataset
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

- [Run setup environment at first](../../tools/setting_environment/README.md)
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

- [Create dataset](../../tools/calibration_classification/README.md)


### 4. Train


Run training:

- Single GPU:
```sh
python tools/calibration_classification/train.py projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb8-25e_t4base.py
```

- Multi GPU (example with 2 GPUs):
```sh
./tools/calibration_classification/dist_train.sh projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb8-25e_t4base.py 2
```

### 5. Deploy

Example commands for deployment (modify paths if needed):
- Custom script (with verification):
```sh
python projects/CalibrationStatusClassification/deploy/main.py projects/CalibrationStatusClassification/configs/deploy/resnet18_5ch.py projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb8-25e_j6gen2.py checkpoint.pth --info_pkl data/t4dataset/calibration_info/t4dataset_gen2_base_infos_test.pkl --sample_idx 0 --device cuda:0 --work-dir /workspace/work_dirs/ --verify
```

## Troubleshooting

## Reference
