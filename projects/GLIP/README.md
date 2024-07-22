# GLIP
## Summary

- ROS package: Not supported
- Supported dataset for training: Not supported
- Other supported feature
  - [ ] Add unit test
- Limited feature

## Results and models

- See [GLIP of mmdetection](https://github.com/open-mmlab/mmdetection/tree/main/configs/glip/README.md).

## Get started
### 1. Setup

- [Run setup environment at first](/tools/setting_environment/).

```
RUN mim install mmdet[multimodal]==3.3.0
```

- Build docker for GLIP

```
DOCKER_BUILDKIT=1 docker build -t autoware-ml-glip projects/GLIP
```

- Download pretrain weight from [GLIP of mmdetection](https://github.com/open-mmlab/mmdetection/tree/main/configs/glip/README.md).
- Run docker

```
docker run -it --rm --gpus all --shm-size=64g --name awml -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml-glip
```

### 2. Train

Not supported

### 3. Inference

- You can inference GLIP model using an image.

```sh
python tools/detection2d/image_demo.py {image_path} \
projects/GLIP/configs/glip_atss_swin-l_fpn_dyhead_pretrain_mixeddata.py \
--weights work_dirs/pretrain/glip_l_mmdet-abfe026b.pth --texts 'traffic cone. car'
--out-dir work_dirs/glip
```

## Troubleshooting
### mmdetection version

`autoware-ml` support mmdetection==3.2.0, but we use configs of version 3.3.0 for GLIP.
So, we install mmdet[multimodal]==3.3.0 for docker environment for GRIP project.

### Issue with NLTK

When integrating to `autoware-ml`, we face the issue same as https://github.com/open-mmlab/mmdetection/issues/11362.
So we make `GLIP_FIXED` class to fix this.

## Reference

- [GLIP of mmdetection](https://github.com/open-mmlab/mmdetection/tree/main/configs/glip/README.md).
