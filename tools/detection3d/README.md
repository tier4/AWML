
## 1. Set environment

- Set environment

```sh
git clone  https://github.com/tier4/autoware-ml
ln -s {path_to_dataset} data
```

```sh
├── data
│  └── nuscenes
│  └── t4dataset
├── Dockerfile
├── projects
├── README.md
└── work_dirs
```

- Build docker

```sh
DOCKER_BUILDKIT=1 docker build -t autoware-ml .
```

## 2. Prepare dataset

Prepare the dataset you use.

### 2.1 nuScenes

- Download dataset from official website
- Run docker

```sh
docker run -it --rm --gpus all --shm-size=64g --name awml -p 6006:6006 -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml
```

- Make info files for nuScenes
  - If you want to make own pkl, you should change from "nuscenes" to "custom_name"

```sh
python tools/detection3d/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

### 2.2 T4 dataset

- Run docker

```sh
docker run -it --rm --gpus all --shm-size=64g --name awml -p 6006:6006 -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml
```

- Make info files for T4dataset

```sh
# for XX1
python tools/detection3d/create_data_t4dataset.py t4xx1  --root_path ./data/t4dataset --max_sweeps 2 --dataset_config autoware_ml/configs/detection3d/dataset/t4dataset/database_v1_1.yaml
# for X2
python tools/detection3d/create_data_t4dataset.py t4xx1  --root_path ./data/t4dataset --max_sweeps 2 --dataset_config autoware_ml/configs/detection3d/dataset/t4dataset/database_v3_0.yaml
```

## 3. Train and evaluation
### 3.1 Change config

- You can change batchsize by file name.
- If you use custom pkl file, you need to change pkl file from `nuscenes_infos_train.pkl`.

### 3.2 Training

- You can use docker command for training as below.
  - See each [projects](projects) for detail command of training and evaluation.

```
docker run -it --rm --gpus '"device=1"' --name autoware-ml --shm-size=64g -d -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml bash -c '<command for each projects>'
```

### 3.3. Log analysis by Tensorboard

- Run the TensorBoard and navigate to http://127.0.0.1:6006/

```sh
tensorboard --logdir work_dirs --bind_all
```

## 4. Visualization

TBD

## 5. Deploy

See each projects
