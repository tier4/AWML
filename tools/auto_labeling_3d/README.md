# auto_labeling_3d

The pipeline of auto labeling for 3D detection.

- [Support priority](https://github.com/tier4/autoware-ml/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier S

## 1. Setup environment

- See [setting environemnt](/tools/setting_environment/)

## (TBD) 2. Create info file from non-annotated T4dataset

- Set non-annotated T4dataset

```
- data/t4dataset/
  - pseudo_xx1/
    - scene_0/
      - annotation/
        - ..
    - scene_1/
    - ..
```

- Make the info file from non-annotated dataset with offline model.
  - The info file contain the confidence information.

```sh
python tools/auto_labeling_3d/create_info_data/create_info_data.py --root-path {path to directory of non-annotated T4dataset} --out-dir {path to output} --config {model config file to use auto labeling} --ckpt {checkpoint file}
```

- For example, run the following command

```sh
python tools/auto_labeling_3d/create_info_data/create_info_data.py --root-path ./data/t4dataset/pseudo_xx1 --out-dir ./data/info --config projects/BEVFusion/configs/t4dataset/bevfusion_lidar_voxel_second_secfpn_1xb1_t4offline.py --ckpt ./work_dirs/bevfusion_offline/epoch_20.pth
```

- If you want to ensemble for auto labeling, you should create info files for each model.
- As a result, the data is as below

```
- data/t4dataset/
  - pseudo_xx1/
    - scene_0/
      - annotation/
        - ..
    - scene_1/
    - ..
  - info/
    - pseudo_infos_raw_centerpoint.pkl
    - pseudo_infos_raw_bevfusion.pkl
```

- You can check the pseudo labels by [rerun visualization](/tools/rerun_visualization/).

```
TBD
```

## (TBD) 3. Filter objects which do not use for pseudo T4dataset

- Set a config to decide what you want to filter
  - Set threshold to filter objects with low confidence

```py
filter_pipelines = ThresholdFilter(
        info_path="./data/t4dataset/info/pseudo_infos_raw_centerpoint.pkl",
        confidence_threshold=0.40,
        use_label=["car", "track", "bus", "bike", "pedestrian"])
```

- If you want to ensemble model, you set a config as below.

```py
filter_pipelines = EnsembleModel([
    ThresholdFilter(
        info_path="./data/t4dataset/info/pseudo_infos_raw_centerpoint.pkl",
        confidence_threshold=0.40,
        use_label=["car", "track", "bus", "bike", "pedestrian"]),
    ThresholdFilter(
        info_path="./data/t4dataset/info/pseudo_infos_raw_bevfusion.pkl",
        confidence_threshold=0.25,
        use_label=["bike", "pedestrian"]),
])
```

- Make the info file to filter the objects which do not use for pseudo T4dataset

```sh
python tools/auto_labeling_3d/filter_objects/filter_objects.py --config {config_file}
```

- As a result, the data is as below

```
- data/t4dataset/
  - pseudo_xx1/
    - scene_0/
      - annotation/
        - ..
    - scene_1/
    - ..
  - info/
    - pseudo_infos_raw_centerpoint.pkl
    - pseudo_infos_raw_bevfusion.pkl
    - pseudo_infos_filtered.pkl
```

- You can check the pseudo labels by [rerun visualization](/tools/rerun_visualization/).

```
TBD
```

### (TBD) 4. Attach tracking id to info file

- If you do not use for target annotation, you can skip this section.
- Attach tracking id to info

```sh
python tools/auto_labeling_3d/attach_tracking_id/attach_tracking_id.py --input {info file} --output {info_file}
```

- As a result, an info file is made as below.

```
- data/t4dataset/
  - pseudo_xx1/
    - scene_0/
      - annotation/
        - ..
    - scene_1/
    - ..
  - info/
    - pseudo_infos_raw_centerpoint.pkl
    - pseudo_infos_raw_bevfusion.pkl
    - pseudo_infos_filtered.pkl
    - pseudo_infos_tracked.pkl
```

### (TBD) 5. Create pseudo T4dataset

- Run script

```sh
python tools/auto_labeling_3d/create_pseudo_t4dataset.py {yaml config file about T4dataset data} --root-path {path to directory of non-annotated T4dataset} --input {path to pkl file}
```

- As a result, pseudo-label T4dataset is made as below.

```
- data/t4dataset/
  - pseudo_xx1/
    - scene_0/
      - annotation/
        - sample.json
        - ..
    - scene_1/
    - ..
```
