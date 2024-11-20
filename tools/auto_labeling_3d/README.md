# auto_labeling_3d
## 1. Setup environment

- [Support priority](https://github.com/tier4/autoware-ml/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier S
- See [setting environemnt](/tools/setting_environment/)

## 2. command
### (TBD) 2.1 Create info file from non-annotated T4dataset

- Set non-annotated T4dataset

```
- data/t4dataset/
  - pseudo_xx1/
    - 0/
      - annotation/
        - ..
    - 1/
    - ..
```

- Make the info file from non-annotated dataset with offline model.
  - The info file contain the confidence information.

```
python tools/t4dataset_pseudo_label_3d/create_info_data/create_info_data.py --input {path to list of non-annotated T4dataset} --config {config_file} --ckpt {ckpt_file}
```

- As a result, the data is as below

```
- data/t4dataset/
  - pseudo_xx1/
    - 0/
      - annotation/
        - ..
    - 1/
    - ..
  - info/
    - pseudo_infos_raw.pkl
```

- You can check the pseudo labels by [rerun visualization](/tools/rerun_visualization/).

### (TBD) 2.2 Filter objects which do not use for pseudo T4dataset

- Make the info file to filter the objects which do not use for pseudo T4dataset

```
python tools/t4dataset_pseudo_label_3d/filter_objects/filter_objects.py --config {config_file} --input {info file} --output {info_file}
```

- As a result, the data is as below

```
- data/t4dataset/
  - pseudo_xx1/
    - 0/
      - annotation/
        - ..
    - 1/
    - ..
  - info/
    - pseudo_infos_raw.pkl
    - pseudo_infos.pkl
```

### (TBD) 2.3 Attach tracking id to info file

- If you do not use for target annotation, you can skip this section.
- Attach tracking id to info

```
python tools/t4dataset_pseudo_label_3d/attach_tracking_id/attach_tracking_id.py --config {config_file} --input {info file} --output {info_file}
```

- As a result, an info file is made as below.

```
- data/t4dataset/
  - pseudo_xx1/
    - 0/
      - annotation/
        - ..
    - 1/
    - ..
  - info/
    - pseudo_infos_raw.pkl
    - pseudo_infos.pkl
    - pseudo_infos_tracked.pkl
```

### (TBD) 2.4 Create pseudo T4dataset

- Run script

```
python tools/t4dataset_pseudo_label_3d/create_pseudo_t4dataset.py --input {path to pkl file}
```

- As a result, pseudo-label T4dataset is made as below.

```
- data/t4dataset/
  - pseudo_xx1/
    - 0/
      - annotation/
        - sample.json
        - ..
    - 1/
    - ..
```
