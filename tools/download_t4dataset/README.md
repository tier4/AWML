# Download for T4dataset
## Environment

- Ubuntu22.04 LTS
  - This scripts do not need docker environment
- webauto CLI > v0.33.1

## Download dataset

- Download for XX1

```sh
python tools/download_t4dataset/download_t4dataset.py autoware_ml/configs/detection3d/dataset/t4dataset/database_v1_1.yaml --output ./data/t4dataset/ --project-id prd_jt
```

- Download for X2

```sh
python tools/download_t4dataset/download_t4dataset.py autoware_ml/configs/detection3d/dataset/t4dataset/database_v3_0.yaml --output ./data/t4dataset/ --project-id x2_dev
```

- After download processing, the directory architecture consists as below.

```
- data/t4dataset/
  - database_v1_1/
    - {t4dataset_id}
    - {t4dataset_id}
    - {t4dataset_id}
    - {t4dataset_id}
```
