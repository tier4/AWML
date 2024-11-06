# t4dataset_pseudo_label_3d
## 1. Setup environment

- [Support priority](https://github.com/tier4/autoware-ml/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier A
- See [setting environemnt](/tools/setting_environment/)

## 2. command
### (TBD) 2.1 non-annotated dataset to info file

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
python tools/t4dataset_pseudo_label_3d/create_data.py --input {path to non-annotated T4dataset} --config {config_file} --ckpt {ckpt_file}
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
    - pseudo_label_infos.pkl
```

### (TBD) 2.2 info file to pseudo-label T4dataset

- Set non-annotated T4dataset and info file.

```
- data/t4dataset/
  - pseudo_xx1/
    - 0/
      - annotation/
        - ..
    - 1/
    - ..
  - info/
    - pseudo_label_infos.pkl
```

- Run script

```
python tools/t4dataset_pseudo_label_3d/create_pseudo_t4dataset.py --input {path to non-annotated T4dataset} --threshold 0.3
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
