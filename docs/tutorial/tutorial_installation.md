
## 1. Set up repository

- Prepare the environment

```sh
git clone https://github.com/tier4/AWML
```

## 2. Prepare docker

- Pull the base Docker image
  - See [autoware-ml-base](https://github.com/tier4/AWML/pkgs/container/autoware-ml-base)
.

```
docker pull ghcr.io/tier4/autoware-ml-base:latest
```

## 3. Download T4dataset

- Request access to the T4dataset in [WebAuto](https://docs.web.auto/en/user-manuals/).
- Download T4dataset by using [download scripts](/pipelines/webauto/download_t4dataset/).

If you do not have the access to Web.Auto and still want to use the dataset, please contact Web.Auto team from [the Web.Auto contact form](https://web.auto/contact/).
However, please note that these dataset are currently only available for TIER IV members as of September 2025.

After downloading dataset, the directory shows as below.

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

## Tips
### Set up other dataset

If you want to use open dataset like nuScenes dataset, please follow [mmdetection3d documents](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/index.html) and [mmdetection documents](https://mmdetection.readthedocs.io/en/latest/user_guides/dataset_prepare.html).
