# Update T4dataset version

- [Support priority](https://github.com/tier4/autoware-ml/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier S

## 1. Make new sensor config dataset

- 1.1. [Dataset engineer] Add new yaml file for [dataset config](/autoware_ml/configs/detection3d/dataset/t4dataset) like `database_v1_0_0.yaml` after upload T4dataset.

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

- 1.2. [Dataset engineer] Add new sensor config for [dataset config](/autoware_ml/configs/detection3d/dataset/t4dataset) like `xx1.py` after upload T4dataset.

```py
# xx1_gen2.py
dataset_version_list = ["database_v1_0_0"]
```

- 1.3. [ML server maintainer] Download new dataset by [download_t4dataset](/tools/download_t4dataset).

```
- t4dataset/
  - database_v1_0/
    - aaaaaaaaaa0/
    - aaaaaaaaaa1/
    - aaaaaaaaaa2/
```

## 2. Make new dataset for new scene

- 2.1. [Dataset engineer] Add new yaml file for [dataset config](/autoware_ml/configs/detection3d/dataset/t4dataset) like `database_v1_0_0.yaml` after upload T4dataset.

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

- 1.2. [Dataset engineer] Update new sensor config for [dataset config](/autoware_ml/configs/detection3d/dataset/t4dataset) like `xx1.py` after upload T4dataset.

```py
# xx1_gen2.py
dataset_version_list = ["database_v1_0_0", "database_v1_1_0"]
```

- 1.3. [ML server maintainer] Download new dataset by [download_t4dataset](/tools/download_t4dataset).

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

## 3. Fix dataset like fixing annotation or ego pose

- 3.1. [Dataset engineer] Add new yaml file for [dataset config](/autoware_ml/configs/detection3d/dataset/t4dataset) like `database_v1_0_0.yaml` after upload T4dataset.

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

- 3.2. [Dataset engineer] Update new sensor config for [dataset config](/autoware_ml/configs/detection3d/dataset/t4dataset) like `xx1.py` after upload T4dataset.

```py
# xx1_gen2.py
#dataset_version_list = ["database_v1_0_0", "database_v1_1_0"]
dataset_version_list = ["database_v1_0_1", "database_v1_1_0"]
```

- 3.3. [ML server maintainer] Download new dataset by [download_t4dataset](/tools/download_t4dataset).
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
