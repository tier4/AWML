
## Set environment

- Set environment

```sh
git clone  https://github.com/tier4/mmdetection3d_bevfusion
ln -s {path_to_dataset} data
```

```sh
├── data
│  └── nuScenes
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

## Make pkl files

- Run docker

```sh
docker run -it --rm --gpus all --shm-size=64g -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml
```

- Make info files for nuScenes
  - If you want to make own pkl, you should change from "nuscenes" to "custom_name"

```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

- Make info files for T4dataset

TBD

## Train and evaluation

- Change config
  - If you use custom pkl file, you need to change pkl file from `nuscenes_infos_train.pkl`.
- You can use docker command for training as below.
  - See each [projects](projects) for detail command of training and evaluation.

```
docker run -it --rm --gpus '"device=1"' --name autoware-ml --shm-size=64g -d -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml bash -c '
python tools/train.py projects/BEVFusion/configs/nuscenes/bevfusion_lidar_voxel0075_second_secfpn_1xb1-cyclic-20e_nus-3d.py'
```

## Visualization

TBD
