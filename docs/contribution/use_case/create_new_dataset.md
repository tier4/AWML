# Add a new dataset (a set of T4dataset) for a vehicle with an existing dataset

If you want to add a new dataset for an existing vehicle (e.g. from a different operating area), please increment "X" from the existing dataset. Specifically, please name it as "DB/UC/Pseudo {new vehicle name} v(X+1).0" if the dataset vX.Y already exists.

## 1. [Dataset engineer] Create dataset and upload to WebAuto system

By using WebAuto system, a dataset engineer create dataset.

## 2. [Dataset engineer] Create a PR to add a new config file

- Add a new yaml file for [T4Dataset config](/autoware_ml/configs/t4dataset) like `db_j6gen2_v2.yaml` after uploading T4dataset.
- Fix a sensor config for [detection3d config](/autoware_ml/configs/detection3d/dataset/t4dataset) like `x2_gen2.py`.

```yaml
# db_j6gen2_v2.yaml

version: 1

train:
  - 80b37b8c-ae9d-4641-a921-0b0c2012eee8
val:
  - c8cf2fe3-9097-4f8d-8984-e99c4ddd0ced
test:
  - 9e973a55-3f70-48e0-8b37-a68b66a99686
```

```py
# x2_gen2.py
dataset_version_list = ["db_j6gen2_v1", "db_j6gen2_v2"]
```

For PR review list with code owner

- [ ] Modify the dataset config files
- [ ] Update documentation of dataset

## 3. [User] Download new dataset by [download_t4dataset](/pipelines/webauto/download_t4dataset/).

After download, the dataset directory consists as belows.

```yaml
- t4dataset/
  - db_j6gen2_v1/
    - e6d0237c-274c-4872-acc9-dc7ea2b77943/
      - 0/
    - 3013c354-2492-447b-88ce-70ec0438f494/
      - 0/
    - 13351af0-41cb-4a96-9553-aeb919efb46e/
      - 0/
  - db_j6gen2_v2/
    - 80b37b8c-ae9d-4641-a921-0b0c2012eee8/
      - 0/
    - c8cf2fe3-9097-4f8d-8984-e99c4ddd0ced/
      - 0/
    - 9e973a55-3f70-48e0-8b37-a68b66a99686/
      - 0/
```
