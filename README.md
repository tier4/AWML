# Autoware-ML

ML library for Autoware based on [mmdetection3d v1.4.0](https://github.com/open-mmlab/mmdetection3d/tree/v1.4.0) and mmdet v3.3

## Supported model
### 3D detection

- [BEVFusion](projects/BEVFusion/README.md)
- TransFusion
- PointPainting CenterPoint
- CenterPoint

|                           | T4dataset | NuScenes | Waymo | aimotive |
| ------------------------- | :-------: | :------: | :---: | :------: |
| BEVFusion                 |           |    ✅     |       |          |
| TransFusion               |           |          |       |          |
| CenterPoint               |           |          |       |          |
| PointPainting CenterPoint |           |          |       |          |

### 2D detection

- YOLOX-opt
- TwinTransformer

|                 | T4dataset | NuImages | COCO  |
| --------------- | :-------: | :------: | :---: |
| YOLOX-opt       |           |          |       |
| TwinTransformer |           |          |       |

### 2D classification

## Get started
### Set environment

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
docker build -t autoware-ml docker/
```

- Run docker

```sh
docker run -it --rm --gpus all --shm-size=64g -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml
```

- Make info files
  - Change {user_name} to your user name

```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag {user_name}_nuscenes
```

### Train and evaluation

- Change config
  - The name of pkl file is `{user_name}_nuscenes_infos_train.pkl`.

```py
#user_name = ""
user_name = "{user_name}_"
```

- See each [projects](projects) for train and evaluation.
