# Project adaptation

- [Support priority](https://github.com/tier4/autoware-ml/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier A

## Get started
### 1. Prepare non-annotated T4dataset

- Prepare non-annotated T4dataset

### 2. Make pseudo label

- Use [Pseudo label for T4dataset](/tools/t4dataset_pseudo_label_3d/)
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

### 3. Fine tuning

- Fine tuning by pseudo label

```sh
# command
```
