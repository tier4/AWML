
## 1. Set environment

- Set environment

```sh
git clone  https://github.com/tier4/mmdetection3d_bevfusion
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
docker build -t autoware-ml .
```

## 2. Prepare dataset

Prepare the dataset you use.

### 2.1 [Option] nuScenes

- Download dataset from official website
- Run docker

```sh
docker run -it --rm --gpus all --shm-size=64g -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml
```

- Make info files for nuScenes
  - If you want to make own pkl, you should change from "nuscenes" to "custom_name"

```sh
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

### 2.2 [Option] T4 dataset

- [Option] Download dataset

```sh
# download xx1 dataset
python scripts/download_t4dataset.py config/dataset_config/xx1.yaml --project-id prd_jt
# download x2 dataset
python scripts/download_t4dataset.py config/dataset_config/x2.yaml --project-id x2_dev
```

- Run docker

```sh
docker run -it --rm --gpus all --shm-size=64g -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml
```

- Make info files for T4dataset

```sh
# for XX1
python tools/create_data_t4dataset.py t4xx1  --root_path ./data/t4dataset --max_sweeps 2 --dataset_config configs/dataset/xx1.yaml
# for X2
python tools/create_data_t4dataset.py t4xx1  --root_path ./data/t4dataset --max_sweeps 2 --dataset_config configs/dataset/x2.yaml
```

## 3. Train and evaluation

- Change config
  - If you use custom pkl file, you need to change pkl file from `nuscenes_infos_train.pkl`.
- You can use docker command for training as below.
  - See each [projects](projects) for detail command of training and evaluation.

```
docker run -it --rm --gpus '"device=1"' --name autoware-ml --shm-size=64g -d -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml bash -c '<command for each projects>'
```

## 4. Visualization

TBD
