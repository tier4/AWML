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

TBD

### 2. Select scene

- [x] Select scene from pictures

```sh
python tools/scene_selector/image_selector.py {config_file} {directory or image_file}
```
Example
```sh
python3 tools/scene_selector/image_selector.py tools/scene_selector/configs/det2d_object_num_selector/yolox_l_object_number_sum.py \
  "./data/t4dataset/database_v1_1_mini/0338521f-321a-4a9a-9c52-480f1ae1131a/2/data/CAM_FRONT/*.jpg" --out-dir ./work_dirs
```

- [x] Select scene from t4 dataset

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
