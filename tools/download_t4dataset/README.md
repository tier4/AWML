# Download for T4dataset
## Download dataset

- Download for XX1

```sh
python tools/download_t4dataset/download_t4dataset.py autoware_ml/configs/detection3d/dataset/t4dataset/database_v1_1.yaml --out-dir ./data/t4dataset/ --project-id prd_jt
```

- Download for X2

```sh
python tools/download_t4dataset/download_t4dataset.py autoware_ml/configs/detection3d/dataset/t4dataset/database_v3_0.yaml --out-dir ./data/t4dataset/ --project-id x2_dev
```

## Set directory architecture

- After download as above command, the directory architecture consists as below.

```
- data/t4dataset/
  - annotation_dataset/
    - {t4dataset_id}
    - {t4dataset_id}
    - {t4dataset_id}
    - {t4dataset_id}
```

- Change name directory for your download dataset name to use autoware-ml.

```sh
cd data/t4dataset

# for XX1
mv annotation_dataset database_v1_1
# for X2
mv annotation_dataset database_v3_0
```

```
- data/t4dataset
  - /database_v1_1/
    - {t4dataset_id}
    - {t4dataset_id}
    - {t4dataset_id}
    - {t4dataset_id}
```
