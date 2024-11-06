# scene selector

- [Support priority](https://github.com/tier4/autoware-ml/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier S

## docs

- [Design document](docs/design.md)

## config

- [Object number threshold scene selector with 2D detection](configs/det2d_object_num_selector/)
- [Object number threshold scene selector with 2D open vocabulary](configs/open_vocab_2d_object_num_selector/)
- [(TBD) Select scene with VLM QA](configs/vlm_qa_selector/)
- [(TBD) rosbag selector](configs/rosbag/)

## Get started
### 1. setup environment

```
DOCKER_BUILDKIT=1 docker build -t autoware-ml-ros2 ./tools/setting_environment/ros2/
```

### 2. select scene

Choose from belows.

#### 2.1. Select scene from T4dataset

```sh
python3 tools/scene_selector/image_selector_t4_dataset.py {config_file} --out-dir {output_dir} \
  --dataset-configs {dataset_config} --data-root {data_root} \
  --experiment-name {experiment_name} --true-ratio {true_ratio} \
  --show-visualization --create-symbolic-links
```

Example
```sh
python3 tools/scene_selector/image_selector_t4_dataset.py tools/scene_selector/configs/det2d_object_num_selector/yolox_l_object_number_sum.py \
  --out-dir ./work_dirs/closed_vocab --dataset-configs autoware_ml/configs/detection3d/dataset/t4dataset/database_v1_1_mini.yaml \
  --data-root ./data/t4dataset/ --experiment-name closed_vocab_exp \
  --true-ratio 0.1 --show-visualization --create-symbolic-links
```

#### 2.2. Select scene from images

This is a tool to select images from a set of images.
Note that currently we do not support the evaluation as we do in "t4dataset selector" at this moment.

- Run

```sh
python tools/scene_selector/image_selector.py {config_file} {directory or image_file}
```

- Example

```sh
python3 tools/scene_selector/image_selector.py tools/scene_selector/configs/det2d_object_num_selector/yolox_l_object_number_sum.py \
  "./data/t4dataset/database_v1_1_mini/0338521f-321a-4a9a-9c52-480f1ae1131a/2/data/CAM_FRONT/*.jpg" --out-dir ./work_dirs
```

## Design for new scene selector
### (TBD) Select scene from T4dataset

If you want to use evaluation for scene selector, you should use annotated T4dataset.

- Make `pseudo_label_infos_2d.pkl` by `tools/t4dataset_pseudo_label_2d/t4dataset_pseudo_label.py`

```sh
python3 tools/t4dataset_pseudo_label_2d/t4dataset_pseudo_label.py --input {path to non-annotated T4dataset} --config {config_file} --ckpt {ckpt_file}
```

```
- data/t4dataset/
  - info/
    - pseudo_label_infos_2d.pkl
```

- Run scene selector script to generate `scene_selector_infos.pkl`. This file contains results for which scene to select.

```sh
python3 tools/scene_selector/image_selector_t4_dataset.py {config_file} --out-dir {output_dir} \
  --info-file {info_file_path} --data-root {data_root} \
  --experiment-name {experiment_name} \
  --show-visualization --create-symbolic-links
```

```
- data/t4dataset/
  - info/
    - pseudo_label_infos_2d.pkl
    - scene_selector_infos.pkl
```

- If you want to check the result, you can use the following script.

```
python3 tools/scene_selector/visualize_image_selector.py
```

- If you want to evaluate with fine tuning for selected dataset, you should change the config as below and train by [detection3d tools](/tools/detection3d/)

```py
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="CBGSDataset",
        dataset=dict(
            data_root=data_root,
#            ann_file=info_directory_path + _base_.info_train_file_name,
            ann_file=info_directory_path + t4dataset_selected_infos_train.pkl,
            modality=input_modality,
            type=_base_.dataset_type,
            metainfo=_base_.metainfo,
            class_names=_base_.class_names,
            test_mode=False,
            data_prefix=_base_.data_prefix,
            box_type_3d="LiDAR",
            backend_args=backend_args,
        ),
    ),
)
```

### (TBD) Select scene from rosbag

This tool can be used to test for rosbag casually and cannot be used to evaluate.

```sh
python3 tools/scene_selector/image_selector_rosbag.py {config_file} --out-dir {output_dir} \
  --experiment-name {experiment_name} \
  --show-visualization --create-symbolic-links
```
