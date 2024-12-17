# Update T4dataset version

- [Support priority](https://github.com/tier4/autoware-ml/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier S

## 1. Make new vehicle dataset

- 1.1. [Dataset engineer] Make PR adding new config
  - Add new yaml file for [dataset config](/autoware_ml/configs/detection3d/dataset/t4dataset) like `db_j6gen2_v1.yaml` after upload T4dataset.
    - Add document about the dataset
    - Update from X.Y.Z to (X+1).0.0
  - Add new sensor config for [dataset config](/autoware_ml/configs/detection3d/dataset/t4dataset) like `x2_gen2.py`.

```yaml
# db_j6gen2_v1.yaml

version: 1
dataset_version: db-j6gen2-1.0
docs: |
  Product: X2-gen2
  Place: Odaiba
  Amount: About 5000 frames
  Sensor: Hesai LiDAR + C1 Camera + Radar data
  Annotation: All the data are collected at 10Hz and annotated at 2Hz

train:
  - aaaaaaaaaa0 #DB-J6Gen2-v1-odaiba_0
val:
  - aaaaaaaaaa1 #DB-J6Gen2-v1-odaiba_1
test:
  - aaaaaaaaaa2 #DB-J6Gen2-v1-odaiba_2
```

```py
# x2_gen2.py
dataset_version_list = ["db_j6gen2_v1"]
```

- 1.2. [User] Download new dataset by [download_t4dataset](/pipelines/webauto/download_t4dataset/).

```
- t4dataset/
  - db_j6gen2_v1/
    - aaaaaaaaaa0/
    - aaaaaaaaaa1/
    - aaaaaaaaaa2/
```

## 2. Make new dataset for existing vehicle

- 2.1. [Dataset engineer] Make PR adding new config
  - Add new yaml file for [dataset config](/autoware_ml/configs/detection3d/dataset/t4dataset) like `db_j6gen2_v2.yaml` after upload T4dataset.
    - Add document about the dataset
    - Update from X.Y.Z to X.(Y+1).0
  - Fix sensor config for [dataset config](/autoware_ml/configs/detection3d/dataset/t4dataset) like `x2_gen2.py`.

```yaml
# db_j6gen2_v2.yaml

version: 1
dataset_version: db-j6gen2-2.0
docs: |
  Product: X2-gen2
  Place: Odaiba
  Amount: About 5000 frames
  Sensor: Hesai LiDAR + C1 Camera + Radar data
  Annotation: All the data are collected at 10Hz and annotated at 2Hz

train:
  - bbbbbbbbbb0 #DB-J6Gen2-v2-odaiba_0
val:
  - bbbbbbbbbb1 #DB-J6Gen2-v2-odaiba_1
test:
  - bbbbbbbbbb2 #DB-J6Gen2-v2-odaiba_2
```

```py
# x2_gen2.py
dataset_version_list = ["db_j6gen2_v1", "db_j6gen2_v2"]
```

- 2.2. [User] Download new dataset by [download_t4dataset](/pipelines/webauto/download_t4dataset/).

```
- t4dataset/
  - db_j6gen2_v1/
    - aaaaaaaaaa0/
    - aaaaaaaaaa1/
    - aaaaaaaaaa2/
  - db_j6gen2_v2/
    - bbbbbbbbbb0/
    - bbbbbbbbbb1/
    - bbbbbbbbbb2/
```

## 3. Add new dataset for existing database dataset

- 3.1. [github CI] Add T4dataset dataset id to yaml file of [dataset config](/autoware_ml/configs/detection3d/dataset/t4dataset).
  - Update `dataset_version` from X.Y to X.(Y+1)

```yaml
# db_j6gen2_v2.yaml
version: 1
#dataset_version: db-j6gen2-2.0
dataset_version: db-j6gen2-2.1
docs: |
  Product: X2-gen2
  Place: Odaiba
  Amount: About 5000 frames
  Sensor: Hesai LiDAR + C1 Camera + Radar data
  Annotation: All the data are collected at 10Hz and annotated at 2Hz

train:
  - bbbbbbbbbb0 #DB-J6Gen2-v2-odaiba_0
  - cccccccccc0 #DB-J6Gen2-v2-odaiba_3
val:
  - bbbbbbbbbb1 #DB-J6Gen2-v2-odaiba_1
  - cccccccccc1 #DB-J6Gen2-v2-odaiba_4
test:
  - bbbbbbbbbb2 #DB-J6Gen2-v2-odaiba_2
  - cccccccccc2 #DB-J6Gen2-v2-odaiba_5
```

- 3.2. [User] Download new dataset by [download_t4dataset](/pipelines/webauto/download_t4dataset/).

```
- t4dataset/
  - db_j6gen2_v1/
    - aaaaaaaaaa0/
    - aaaaaaaaaa1/
    - aaaaaaaaaa2/
  - db_j6gen2_v2/
    - bbbbbbbbbb0/
    - bbbbbbbbbb1/
    - bbbbbbbbbb2/
    - cccccccccc0/
    - cccccccccc1/
    - cccccccccc2/
```
