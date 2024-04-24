# TransFusion
## Supported feature

- [x] Train LiDAR-only model
- [x] Train with single GPU
- [ ] Train with multiple GPU
- [x] Add script to make .onnx file and deploy to Autoware
- [ ] Add unit test

## Limited feature

For now, Camera-LiDAR fusion model is not supported in autoware-ml because of performance.

## Get started
### 1. Setup

- Run docker

```sh
docker run -it --rm --gpus all --shm-size=64g --name awml -p 6006:6006 -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml
```

- Build and install dependency (required only at first run).

```sh
python projects/TransFusion/setup.py develop
```

### 2. Train

- Train

```sh
## for nuScenes with single GPU
mim train mmdet3d projects/TransFusion/configs/nuscenes/transfusion_lidar_pillar02_second_secfpn_1xb8-cyclic-20e_nus-3d.py --launcher pytorch
# or
python tools/train.py projects/TransFusion/configs/nuscenes/transfusion_lidar_pillar02_second_secfpn_1xb8-cyclic-20e_nus-3d.py

## for T4dataset with single GPU
python tools/train.py projects/TransFusion/configs/t4dataset/transfusion_lidar_pillar02_second_secfpn_1xb4-cyclic-20e_t4xx1.py
```

### 3. Deploy

- Docker build for deploy

```
DOCKER_BUILDKIT=1 docker build -t autoware-ml-transfusion-deploy projects/TransFusion/
```

- Make onnx file for TransFusion

```sh
docker run -it --rm --gpus all --shm-size=64g --name awml-deploy -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml-transfusion-deploy

cd /workspace

# for nuScenes dataset
python tools/deploy.py projects/TransFusion/configs/deploy/transfusion_lidar_tensorrt_dynamic-20x5.py projects/TransFusion/configs/nuscenes/transfusion_lidar_pillar02_second_secfpn_1xb8-cyclic-20e_nus-3d.py work_dirs/transfusion_lidar_pillar02_second_secfpn_1xb8-cyclic-20e_nus-3d/epoch_20.pth data/nuscenes/samples/LIDAR_TOP/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin --device cuda:0 --work-dir /workspace

# for t4xx1 dataset
python tools/deploy.py projects/TransFusion/configs/deploy/transfusion_lidar_tensorrt_dynamic-20x5.py projects/TransFusion/configs/t4dataset/transfusion_lidar_pillar02_second_secfpn_1xb4-cyclic-20e_t4xx1.py work_dirs/transfusion_lidar_pillar02_second_secfpn_1xb4-cyclic-20e_t4xx1/epoch_20.pth data/t4dataset/t4xx1/023b4b43-2b00-444c-bb63-4ee602e30779/data/LIDAR_CONCAT/0.pcd.bin --device cuda:0 --work-dir /workspace

# fix the graph
python projects/TransFusion/scripts/fix_graph.py end2end.onnx
```

## Results and models
### NuScenes

- LiDAR only model (pillar)

```
car_AP_dist_1.0: 0.8560
truck_AP_dist_1.0: 0.5484
construction_vehicle_AP_dist_1.0: 0.1463
bus_AP_dist_1.0: 0.7105
trailer_AP_dist_1.0: 0.3302
barrier_AP_dist_1.0: 0.5871
motorcycle_AP_dist_1.0: 0.4949
bicycle_AP_dist_1.0: 0.3265
pedestrian_AP_dist_1.0: 0.7946
traffic_cone_AP_dist_1.0: 0.5678
mAP: 0.5501
```

### T4 dataset

TBD

## Troubleshooting
### Failed to create TensorRT engine

Currently `mmdeploy` supports CUDA 11 only, where autoware-ml Docker image base on CUDA 12.
You can validate mmdeploy's missing dependencies with command:

```sh
ldd /opt/conda/lib/python3.10/site-packages/mmdeploy/lib/libmmdeploy_tensorrt_ops.so
```

Nevertheless, `onnx` file should be created successfully.
