# auto_label_2d
## 1. Setup environment

- [Support priority](https://github.com/tier4/AWML/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier B
- See [setting environemnt](/tools/setting_environment/)

## 2. command
### (TBD) 2.1 T4dataset to info file

This tool can be used for non-annotated T4dataset and annotated T4dataset.
- Set T4dataset

```
- data/t4dataset/
  - pseudo_xx1_2d/
    - 0/
    - 1/
```

- Make the info file from T4dataset with offline model.
  - The info file contain the confidence information.

```
tools/t4dataset_pseudo_label_2d/t4dataset_pseudo_label.py --input {path to non-annotated T4dataset} --config {config_file} --ckpt {ckpt_file}
```

- As a result, the data is as below.

```
- data/t4dataset/
  - pseudo_xx1_2d/
    - 0/
    - 1/
    - ..
  - info/
    - pseudo_label_infos_2d.pkl
```

- The information of pseudo_label_infos_2d.pkl

```
- info\['cam_instances'\]\['CAM_FRONT'\]\[i\]\['bbox_label'\]: Label of instance.
- info\['cam_instances'\]\['CAM_FRONT'\]\[i\]\['bbox_label_3d'\]: Label of instance.
- info\['cam_instances'\]\['CAM_FRONT'\]\[i\]\['bbox'\]: 2D bounding box annotation (exterior rectangle of the projected 3D box), a list arrange as \[x1, y1, x2, y2\].
- info\['cam_instances'\]\['CAM_FRONT'\]\[i\]\['center_2d'\]: Projected center location on the image, a list has shape (2,), .
```

### (TBD) 2.2 info file to pseudo-label T4dataset

- Set non-annotated T4dataset and info file

```
- data/t4dataset/
  - pseudo_xx1_2d/
    - 0/
      - annotation/
        - ..
    - 1/
    - ..
  - info/
    - pseudo_label_infos_2d.pkl
```

- Run script

```
tools/t4dataset_pseudo_label_2d/create_pseudo_t4dataset.py --input {path to non-annotated T4dataset} --threshold 0.3
```

- As a result, pseudo-label T4dataset is made as below.

```
- data/t4dataset/
  - pseudo_xx1_2d/
    - 0/
      - annotation/
        - sample.json
        - ..
    - 1/
    - ..
```
