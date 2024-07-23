# tools/detection2d

Training and evaluation tools for 2D Detection.

- [Support priority](https://github.com/tier4/autoware-ml/blob/main/docs/design/autoware_ml_design.md?#support-priority): Tier A
- Supported dataset
  - [x] COCO dataset
  - [ ] T4dataset
- Other supported feature
  - [ ] Add unit test

## 1. Set environment

See [setting environemnt](/tools/setting_environment/)

## 2. Prepare dataset
### 2.1. COCO dataset

- Run docker

### 2.2. T4dataset

TBD

## 3. Train

- You can use docker command for training as below.
  - See each [projects](projects) for detail command of training and evaluation.

```sh
docker run -it --rm --gpus '"device=1"' --name autoware-ml --shm-size=64g -d -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml bash -c '<command for each projects>'
```

## 4. Analyze
### 4.1. Evaluation

### 4.2. Visualization

```sh
python tools/detection2d/image_demo.py {image path} {config_file} --weights {pth_file}
```

## 5. Deploy

See each projects
