# Update T4dataset version

- [Support priority](https://github.com/tier4/autoware-ml/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier S

## 1. Make new vehicle dataset

- 1.1. [Dataset engineer] Make PR adding new config
  - Add new yaml file for [dataset config](/autoware_ml/configs/detection3d/dataset/t4dataset) like `database_v4_0.yaml` after upload T4dataset.
    - Add document about the dataset
    - Update from X.Y.Z to (X+1).0.0
  - Add new sensor config for [dataset config](/autoware_ml/configs/detection3d/dataset/t4dataset) like `x2_gen2.py`.

```yaml
# database/database_v4_0.yaml

version: 1

docs: |
  Dataset version: v4.0.0
  Product: X2-gen2
  Place: Odaiba
  Amount: About 5000 frames
  Sensor: Hesai LiDAR + C1 Camera + Radar data
  Annotation: All the data are collected at 10Hz and annotated at 2Hz

train:
  - aaaaaaaaaa0 #DBv4.0_odaiba_0
val:
  - aaaaaaaaaa1 #DBv4.0_odaiba_1
test:
  - aaaaaaaaaa2 #DBv4.0_odaiba_2
```

```py
# x2_gen2.py
dataset_version_list = ["database_v4_0"]
```

- 1.2. [User] Download new dataset by [download_t4dataset](/pipelines/webauto/download_t4dataset/).

```
- t4dataset/
  - database_v4_0/
    - aaaaaaaaaa0/
    - aaaaaaaaaa1/
    - aaaaaaaaaa2/
```

## 2. Make new dataset for existing vehicle

- 2.1. [Dataset engineer] Make PR adding new config
  - Add new yaml file for [dataset config](/autoware_ml/configs/detection3d/dataset/t4dataset) like `database_v4_1.yaml` after upload T4dataset.
    - Add document about the dataset
    - Update from X.Y.Z to X.(Y+1).0
  - Fix sensor config for [dataset config](/autoware_ml/configs/detection3d/dataset/t4dataset) like `x2_gen2.py`.

```yaml
# database/database_v4_1.yaml

version: 1

docs: |
  Dataset version: v4.1.0
  Product: X2-gen2
  Place: Odaiba
  Amount: About 5000 frames
  Sensor: Hesai LiDAR + C1 Camera + Radar data
  Annotation: All the data are collected at 10Hz and annotated at 2Hz

train:
  - bbbbbbbbbb0 #DBv4.1_odaiba_0
val:
  - bbbbbbbbbb1 #DBv4.1_odaiba_1
test:
  - bbbbbbbbbb2 #DBv4.1_odaiba_2
```

```py
# x2_gen2.py
dataset_version_list = ["database_v4_0", "database_v4_1"]
```

- 2.2. [User] Download new dataset by [download_t4dataset](/pipelines/webauto/download_t4dataset/).

```
- t4dataset/
  - database_v4_0/
    - aaaaaaaaaa0/
    - aaaaaaaaaa1/
    - aaaaaaaaaa2/
  - database_v4_1/
    - bbbbbbbbbb0/
    - bbbbbbbbbb1/
    - bbbbbbbbbb2/
```

## 3. Add new dataset for existing database dataset

- 3.1. [github CI] Add T4dataset dataset id to new yaml file of [dataset config](/autoware_ml/configs/detection3d/dataset/t4dataset).

```yaml
# database/database_v4_1.yaml

version: 1

docs: |
  Dataset version: v4.1.1
  Product: X2-gen2
  Place: Odaiba
  Amount: About 5000 frames
  Sensor: Hesai LiDAR + C1 Camera + Radar data
  Annotation: All the data are collected at 10Hz and annotated at 2Hz

train:
  - bbbbbbbbbb0 #DBv4.1_odaiba_0
  - cccccccccc0 #DBv4.1_odaiba_3
val:
  - bbbbbbbbbb1 #DBv4.1_odaiba_1
  - cccccccccc1 #DBv4.1_odaiba_4
test:
  - bbbbbbbbbb2 #DBv4.1_odaiba_2
  - cccccccccc2 #DBv4.1_odaiba_5
```

- 3.2. [User] Download new dataset by [download_t4dataset](/pipelines/webauto/download_t4dataset/).

```
- t4dataset/
  - database_v4_0/
    - aaaaaaaaaa0/
    - aaaaaaaaaa1/
    - aaaaaaaaaa2/
  - database_v4_1/
    - bbbbbbbbbb0/
    - bbbbbbbbbb1/
    - bbbbbbbbbb2/
    - cccccccccc0/
    - cccccccccc1/
    - cccccccccc2/
```
