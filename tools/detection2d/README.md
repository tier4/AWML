# tools/detection2d

Training and evaluation tools for 2D Detection.

- [Support priority](https://github.com/tier4/AWML/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier A
- Supported dataset
  - [ ] COCO dataset
  - [x] T4dataset
- Other supported feature
  - [ ] Add unit test

## 1. Set environment

- Please follow the [installation tutorial](/docs/tutorial/tutorial_detection_3d.md) to set up the environment.

## 2. Prepare dataset
### 2.1. COCO dataset

TBD

### 2.2. T4dataset

- (Choice) For traffic light recognition of fine detector

```bash
python3 tools/detection2d/create_data_t4dataset.py --config autoware_ml/configs/detection2d/dataset/t4dataset/tlr_finedetector.py --root_path ./data/tlr/ --data_name tlr -o ./data/tlr_pedcar
```

## 3. Train
### 3.1. Change config

- If you need, change parameters for your experiment
- You can change experiment name by file name

### 3.2. Train

- You can use docker command for training as below.
  - See each [projects](projects) for detail command of training and evaluation.

```
python tools/detection2d/train.py {config_file}
```

- You can use docker command for training as below.

```sh
docker run -it --rm --gpus '"device=1"' --name autoware-ml --shm-size=64g -d -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml bash -c '<command for each projects>'
```

## 4. Analyze
### 4.1. Evaluation

- Evaluation

```
python tools/detection2d/test.py {config_file} {checkpoint_file}
```

### 4.2. Visualization

```sh
python tools/detection2d/image_demo.py {image path} {config_file} --weights {pth_file}
```

## 5. Deploy

See each projects
