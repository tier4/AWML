
## 1. Setup environment

See [tutorial_installation](/docs/tutorial/tutorial_installation.md) to set up the environment.

## 2. Train and evaluation

In this tutorial, we use [CenterPoint](/projects/CenterPoint/).
If you want to know the tools in detail, please see [detection3d](/tools/detection3d/) and [CenterPoint](/projects/CenterPoint/).

### 2.1. Prepare T4dataset

- Run docker

```sh
docker run -it --rm --gpus '"device=0"' --shm-size=64g --name awml -p 6006:6006 -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml
```

- (Choice) Make info files for T4dataset XX1
  - This process takes time.

```sh
python tools/detection3d/create_data_t4dataset.py --root_path ./data/t4dataset --config autoware_ml/configs/detection3d/dataset/t4dataset/xx1.py --version xx1 --max_sweeps 2 --out_dir ./data/t4dataset/info/user_name
```

- (Choice) Make info files for T4dataset X2
  - This process takes time.

```sh
python tools/detection3d/create_data_t4dataset.py --root_path ./data/t4dataset --config autoware_ml/configs/detection3d/dataset/t4dataset/x2.py --version x2 --max_sweeps 2 --out_dir ./data/t4dataset/info/user_name
```

### 2.2. Train

```sh
python tools/detection3d/train.py projects/CenterPoint/configs/t4dataset/second_secfpn_2xb8_121m_base.py
```

### 2.3. Evaluation

```sh
python tools/detection3d/test.py projects/CenterPoint/configs/t4dataset/second_secfpn_2xb8_121m_base.py work_dirs/centerpoint/t4dataset/second_secfpn_2xb8_121m_base/epoch_50.pth
```

### 2.4. Visualization

- `frame-range`: the range of frames to visualize.

```sh
python projects/CenterPoint/scripts/inference.py projects/CenterPoint/configs/t4dataset/second_secfpn_2xb8_121m_base.py work_dirs/centerpoint/t4dataset/second_secfpn_2xb8_121m_base/epoch_50.pth --ann-file-path <info pickle file> --bboxes-score-threshold 0.35 --frame-range 700 1100
```

### 2.5. Deploy the ONNX file

```sh
python projects/CenterPoint/scripts/deploy.py projects/CenterPoint/configs/t4dataset/second_secfpn_2xb8_121m_base.py work_dirs/centerpoint/t4dataset/second_secfpn_2xb8_121m_base/epoch_50.pth --replace_onnx_models --device gpu --rot_y_axis_reference
```

## Tips for developer
### Build docker on your own

- Build docker
  - Note that this process need for long time.
  - You may need `sudo` to use `docker` command.

```sh
DOCKER_BUILDKIT=1 docker build -t autoware-ml .
```
