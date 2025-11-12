# StreamPETR
## Summary

- [Support priority](https://github.com/tier4/AWML/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier A
- ROS package: [TODO]()
- Supported dataset
  - [x] T4dataset
  - [x] NuScenes
- Supported model
  - [x] Camera-only model
- Other supported feature
  - [x] Add script to make .onnx file and deploy to Autoware
  - [x] Visualization of detection results
  - [ ] Add unit tests

## Results and models

- StreamPetr
  - TODO
## Get started
### 1. Setup

- Please follow the [installation tutorial](/docs/tutorial/tutorial_detection_3d.md) to set up the environment.
- Run docker

```sh
docker run -it --rm --gpus all --shm-size=64g --name awml -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml
```
- Run setup script

```sh
cd projects/StreamPETR && pip install -e .
```

- Create info files for training

```sh
python tools/detection3d/create_data_t4dataset.py --root_path ./data --config /workspace/autoware_ml/configs/detection3d/dataset/t4dataset/base.py --version base --max_sweeps 1 --out_dir ./data/info/cameraonly/baseline
```
### 2. Train

- Run training on T4 dataset with appropriate configs

```sh
# Single GPU training
python tools/detection3d/train.py projects/StreamPETR/configs/t4dataset/t4_base_vov_flash_480x640_baseline.py
```

```sh
# Multi GPU training

# Command
bash tools/detection3d/dist_script.sh <config> <number of gpus> train

# Example: T4dataset
bash tools/detection3d/dist_script.sh projects/StreamPETR/configs/t4dataset/t4_base_vov_flash_480x640_baseline.py 4 train
```

### 3. Evaluation


- Run evaluation on a test set, please select experiment config accordingly

- [choice] Evaluate with a single GPU

```sh
# Evaluation for t4dataset
python tools/detection3d/test.py projects/StreamPETR/configs/t4dataset/t4_base_vov_flash_480x640_baseline.py work_dirs/t4_base_vov_flash_480x640_baseline/epoch_35.pth
```

- [choice] Evaluate with multiple GPUs
  - Note that if you choose to evaluate with multiple GPUs, you might get slightly different results as compared to single GPU due to differences across GPUs

```sh
# Command
CHECKPOINT_PATH=<checkpoint> && \
bash tools/detection3d/dist_script.sh <config> <number of gpus> test $CHECKPOINT_PATH

# Example: T4dataset
CHECKPOINT_PATH="work_dirs/t4_base_vov_flash_480x640_baseline/epoch_35.pth" && \
bash tools/detection3d/dist_script.sh projects/StreamPETR/configs/t4dataset/t4_base_vov_flash_480x640_baseline.py 4 test $CHECKPOINT_PATH
```

### 4. Visualization

- Run inference and visualize bounding boxes.

```sh
# Inference for t4dataset
python tools/detection3d/visualize_bboxes_cameraonly.py projects/StreamPETR/configs/t4dataset/t4_base_vov_flash_480x640_baseline.py work_dirs/t4_base_vov_flash_480x640_baseline/epoch_35.pth
```
### 5. Deploy

- Make onnx files for a StreamPETR model

```sh
CONFIG_PATH=/path/to/config
CHECKPOINT_PATH=/path/to/checkpoint
python3 projects/StreamPETR/deploy/torch2onnx.py $CONFIG_PATH --section extract_img_feat --checkpoint $CHECKPOINT_PATH
python3 projects/StreamPETR/deploy/torch2onnx.py $CONFIG_PATH --section pts_head_memory --checkpoint $CHECKPOINT_PATH
python3 projects/StreamPETR/deploy/torch2onnx.py $CONFIG_PATH --section position_embedding --checkpoint $CHECKPOINT_PATH
```

## Reference

- [StreamPETR Official](https://github.com/exiawsh/StreamPETR/tree/main)
- [NVIDIDA DL4AGX TensorRT](https://github.com/NVIDIA/DL4AGX/tree/master/AV-Solutions/streampetr-trt)
