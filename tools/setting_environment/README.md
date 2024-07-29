# Setting environment for autoware-ml

Tools setting environment for `autoware-ml`.

- [Support priority](https://github.com/tier4/autoware-ml/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier S
- Environment
  - [x] Ubuntu22.04 LTS
    - This scripts do not need docker environment

## 1. Setup dataset
### 1.1. Most open dataset

If you want to use open dataset like nuScenes dataset, you set dataset as [mmdetection3d documents](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/index.html) and [mmdetection documents](https://mmdetection.readthedocs.io/en/latest/user_guides/dataset_prepare.html).

### 1.2. T4dataset

If you want to [T4dataset](https://github.com/tier4/tier4_perception_dataset) and you have data access right of [WebAuto(> v0.33.1)](https://docs.web.auto/en/user-manuals/), you can download by `download_t4dataset.py`.
This script do not need docker environment and is tested by Ubuntu22.04 LTS environment.

- Download for XX1

```sh
python tools/setting_environment/download_t4dataset.py autoware_ml/configs/detection3d/dataset/t4dataset/database_v1_1.yaml --output ./data/t4dataset/ --project-id prd_jt
```

- Download for X2

```sh
python tools/setting_environment/download_t4dataset.py autoware_ml/configs/detection3d/dataset/t4dataset/database_v3_0.yaml --output ./data/t4dataset/ --project-id x2_dev
```

- After download as above command, the directory architecture consists as below.

```
- data/t4dataset
  - /database_v1_1/
    - {t4dataset_id}
    - {t4dataset_id}
    - {t4dataset_id}
    - {t4dataset_id}
```

## 2. Setup environment

- Set environment

```sh
git clone https://github.com/tier4/autoware-ml
```

```sh
├── data
│  └── nuscenes
│  └── t4dataset
│  └── nuimages
│  └── coco
├── Dockerfile
├── projects
├── README.md
└── work_dirs
```

- Build docker
  - Note that this process need for long time.

```sh
DOCKER_BUILDKIT=1 docker build -t autoware-ml .
```
