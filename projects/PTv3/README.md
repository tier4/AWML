# Point Transformer V3 (PTv3)

PTv3 is a lidar segmentation model.
AWML's implementation is a port of the [original code](https://github.com/Pointcept/Pointcept), trimming unused parts of the code base, while also adding support for t4dataset and onnx export.

## Summary

- ROS package: [Link](https://github.com/autowarefoundation/autoware_universe/pull/10600)
- Supported datasets
  - [x] NuScenes
  - [x] T4dataset
- Other supported features
  - [x] ONNX export & TensorRT inference

## Results and models

- TODO


## Get started
### 1. Setup

- This project requires a different docker environment that most other projects.

```sh
DOCKER_BUILDKIT=1 docker build -t autoware-ml-ptv3 -f projects/PTv3/Dockerfile . --progress=plain
```

-Run docker

```sh
docker run -it --rm --gpus '"device=0"' --shm-size=64g --name awml -p 6006:6006 -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml-ptv3
```

### 2. Train

To train the model, use the following commands:

```sh
cd projects/PTv3
python tools/train.py --config-file configs/semseg-pt-v3m1-0-t4dataset.py --num-gpus 1
```

To test the model, use the following commands:

```sh
cd projects/PTv3
python tools/test.py --config-file configs/semseg-pt-v3m1-0-t4dataset.py --num-gpus 1 \
  --options \
  save_path=data/experiment \
  weight=exp/model/model_best.pth
```

### 3. Deployment

To deploy the model, a modified version of spconv is required. To use it,
please add `projects` to the `PYTHONPATH`:

```sh
export PYTHONPATH=${PYTHONPATH}:/workspace/projects
```

and then export the model:

```sh
cd projects/PTv3
python tools/export.py --config-file configs/semseg-pt-v3m1-0-t4dataset.py --num-gpus 1 \
  --options \
  save_path=data/experiment \
  weight=exp/model/model_best.pth
```

which will generate a file called `ptv3.onnx`

## Reference

- [Pointcept's PTv3](https://github.com/Pointcept/Pointcept)
