## 1. Set environment

See [README](/README.md)

## 2. Prepare dataset

Prepare the dataset you use.

### 2.1 nuScenes

- Download dataset from official website
- Run docker

```sh
docker run -it --rm --gpus '"device=0"' --shm-size=64g --name awml -p 6006:6006 -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml
```

- Make info files for nuScenes
  - If you want to make own pkl, you should change from "nuscenes" to "custom_name"

```sh
python tools/detection3d/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

### 2.2 T4 dataset

- Run docker

```sh
docker run -it --rm --gpus '"device=0"' --shm-size=64g --name awml -p 6006:6006 -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml
```

- [choice] Make info files for T4dataset XX1

```sh
python tools/detection3d/create_data_t4dataset.py --root_path ./data/t4dataset --config autoware_ml/configs/detection3d/dataset/t4dataset/xx1.py --version xx1 --max_sweeps 2 --out_dir ./data/t4dataset/info/user_name
```

- [choice] Make info files for T4dataset X2

```sh
TBD
```

## 3. Train and evaluation
### 3.1 Change config

- You can change batchsize by file name.
- If you use custom pkl file, you need to change pkl file from `nuscenes_infos_train.pkl`.

### 3.2 Training and evaluation

- Training and evaluation in general by below command.
  - See each [projects](projects) for detail command of training and evaluation.

```sh
# Trainging
python tools/detection3d/train.py {config_file}
# Evaluation
python tools/detection3d/test.py {config_file} {checkpoint_file}
```

- You can use docker command for training as below.

```sh
docker run -it --rm --gpus '"device=0"' --name autoware-ml --shm-size=64g -d -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml bash -c '<command for each projects>'
```

### 3.3. Log analysis by Tensorboard

- Run the TensorBoard and navigate to http://127.0.0.1:6006/

```sh
tensorboard --logdir work_dirs --bind_all
```

### 3.4. Visualization

TBD

## 4. Deploy

See each projects
