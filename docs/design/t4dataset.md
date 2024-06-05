# Dataset docs
## T4dataset version
### XX1

- [database_v1_0](/autoware_ml/configs/detection3d/dataset/t4dataset/database_v1_0.yaml)
  - All the data are collected at 10Hz and most of them are annotated at 2Hz on odaiba, nishi-shinjuku and shiojiri.
  - DBv1.0_nishi_shinjuku_[0-7]_ are annotated at 10Hz.
- [database_v1_1](/autoware_ml/configs/detection3d/dataset/t4dataset/database_v1_1.yaml)

### X2

- [database_v2_0](/autoware_ml/configs/detection3d/dataset/t4dataset/database_v2_0.yaml)
  - All the data are collected and annotated at 10Hz on nishi-shinjuku and GLP-atsugi.
- [database_v3_0](/autoware_ml/configs/detection3d/dataset/t4dataset/database_v3_0.yaml)
  - All the data are collected at 10Hz and annotated at 2Hz on shiojiri.

## Dataset versioning strategy
### dataset version definition

- version: major.minor.build
  - major: sensor configuration
  - minor: dataset scenes
  - build: dataset version

## T4dataset update operation example
### 1. Make new sensor config dataset

- [Dataset engineer] Add new yaml file for [dataset config](https://github.com/tier4/autoware-ml/tree/main/autoware_ml/configs/detection3d/dataset/t4dataset) after upload T4dataset

```yaml
# database/database_v1_0_0.yaml

version: 1

train:
  - aaaaaaaaaa0 #DBv1.0_odaiba_0
val:
  - aaaaaaaaaa1 #DBv1.0_odaiba_1
test:
  - aaaaaaaaaa2 #DBv1.0_odaiba_2
```

- [Dataset engineer] Add new sensor config for [dataset config](https://github.com/tier4/autoware-ml/tree/main/autoware_ml/configs/detection3d/dataset/t4dataset) after upload T4dataset

```py
# xx1_gen2.py
dataset_version_list = ["database_v1_0_0"]
```

- [ML server maintainer] Download dataset by [download_t4dataset](https://github.com/tier4/autoware-ml/tree/main/tools/download_t4dataset)

```
- t4dataset/
  - database_v1_0/
    - aaaaaaaaaa0/
    - aaaaaaaaaa1/
    - aaaaaaaaaa2/
```

### 2. Make new dataset for new scene

- [Dataset engineer] Add yaml file for [dataset config](https://github.com/tier4/autoware-ml/tree/main/autoware_ml/configs/detection3d/dataset/t4dataset) after upload T4dataset

```yaml
# database/database_v1_1_0.yaml

version: 1

train:
  - bbbbbbbbbb0 #DBv1.1_odaiba_0
val:
  - bbbbbbbbbb1 #DBv1.1_odaiba_1
test:
  - bbbbbbbbbb2 #DBv1.1_odaiba_2
```

- [ML engineer] Update new sensor config for [dataset config](https://github.com/tier4/autoware-ml/tree/main/autoware_ml/configs/detection3d/dataset/t4dataset) after upload T4dataset

```py
# xx1_gen2.py
dataset_version_list = ["database_v1_0_0", "database_v1_1_0"]
```

- [ML server maintainer] Download dataset by [download_t4dataset](https://github.com/tier4/autoware-ml/tree/main/tools/download_t4dataset)

```
- t4dataset/
  - database_v1_0/
    - aaaaaaaaaa0/
    - aaaaaaaaaa1/
    - aaaaaaaaaa2/
  - database_v1_1/
    - bbbbbbbbbb0/
    - bbbbbbbbbb1/
    - bbbbbbbbbb2/
```

- [User] Download dataset by [download_t4dataset](https://github.com/tier4/autoware-ml/tree/main/tools/download_t4dataset)

### 3. Fix dataset like fixing annotation or ego pose

- [Dataset engineer] Add yaml file for [dataset config](https://github.com/tier4/autoware-ml/tree/main/autoware_ml/configs/detection3d/dataset/t4dataset) after upload T4dataset

```yaml
# database/database_v1_0_1.yaml

version: 1

train:
  - cccccccccc0 #DBv1.0_odaiba_0
val:
  - cccccccccc1 #DBv1.0_odaiba_1
test:
  - cccccccccc2 #DBv1.0_odaiba_2
```

- [ML engineer] Update new sensor config for [dataset config](https://github.com/tier4/autoware-ml/tree/main/autoware_ml/configs/detection3d/dataset/t4dataset) after upload T4dataset

```py
# xx1_gen2.py
#dataset_version_list = ["database_v1_0_0", "database_v1_1_0"]
dataset_version_list = ["database_v1_0_1", "database_v1_1_0"]
```

- [ML server maintainer] Download dataset by [download_t4dataset](https://github.com/tier4/autoware-ml/tree/main/tools/download_t4dataset)
  - Old scene remain for dataset directory to use train and evaluation for old configuration

```
- t4dataset/
  - database_v1_0/
    - aaaaaaaaaa0/
    - aaaaaaaaaa1/
    - aaaaaaaaaa2/
    - cccccccccc0/
    - cccccccccc1/
    - cccccccccc2/
  - database_v1_1/
    - bbbbbbbbbb0/
    - bbbbbbbbbb1/
    - bbbbbbbbbb2/
```
