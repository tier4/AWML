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
DOCKER_BUILDKIT=1 docker build -t awml-ptv3 -f projects/PTv3/Dockerfile . --progress=plain
```

-Run docker

```sh
docker run -it --rm --gpus all --shm-size=64g --name awml-ptv3 -p 6006:6006 -v $PWD/:/workspace -v $PWD/data:/workspace/data awml-ptv3
```

### 2. Train

To train the model, use the following commands:

```sh
python projects/PTv3/tools/train.py --config-file projects/PTv3/configs/semseg-pt-v3m1-0-t4dataset.py --num-gpus 1
```

To test the model, use the following commands:

```sh
python projects/PTv3/tools/test.py --config-file projects/PTv3/configs/semseg-pt-v3m1-0-t4dataset.py --num-gpus 1 \
  --options \
  save_path=work_dirs/experiment \
  weight=work_dirs/ptv3/model/model_best.pth \
  show=True
```

### 3. Deployment

Export the model:

```sh
python projects/PTv3/tools/export.py --config-file projects/PTv3/configs/semseg-pt-v3m1-0-t4dataset.py --num-gpus 1 \
  --options \
  save_path=work_dirs/experiment \
  weight=work_dirs/ptv3/model/model_best.pth
```

which will generate a file called `ptv3.onnx`

### ONNX preprocessing contract

The exported ONNX expects PTv3 serialized-pooling metadata to be generated outside the engine.
For every encoder pooling stage `i`, preprocessing must provide:

| Input | Shape | Meaning |
| --- | --- | --- |
| `serialized_pooling_i_indices` | `[N_i]` | Native ONNX `Gather` indices for features before CSR reduction. |
| `serialized_pooling_i_indptr` | `[M_i + 1]` | CSR row pointer consumed by `autoware::SegmentCSR`. |
| `serialized_pooling_i_cluster` | `[N_i]` | Input voxel to pooled voxel id mapping. |
| `serialized_pooling_i_head_indices` | `[M_i]` | Representative input voxel for each pooled voxel. |
| `serialized_pooling_i_grid_coord` | `[M_i, 3]` | Pooled voxel coordinates for downstream PTv3 blocks. |
| `serialized_pooling_i_serialized_order` | `[O, M_i]` | Serialization order for pooled voxels. |
| `serialized_pooling_i_serialized_inverse` | `[O, M_i]` | Inverse serialization order for pooled voxels. |

`N_i` is the stage input voxel count, `M_i` is the pooled output voxel count, and `O` is the
number of serialization orders. The old data-dependent `Unique`/pooling-shape discovery is not
exported into the ONNX graph; the graph uses native `Gather` plus the existing
`autoware::SegmentCSR` plugin for pooled feature reduction.

## Reference

- [Pointcept's PTv3](https://github.com/Pointcept/Pointcept)
