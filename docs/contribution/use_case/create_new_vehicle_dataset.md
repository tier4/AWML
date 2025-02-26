# Create a new vehicle dataset

When you add a new dataset for a certain vehicle, it should begin with "v1.0". Specifically, please name it as "DB/UC/Pseudo {new vehicle name} v(X+1).0".
If you want to update dataset, please modify [the dataset config files](/autoware_ml/configs/t4dataset/).

## 1. [Dataset engineer] Create dataset and upload to WebAuto system

By using WebAuto system, a dataset engineer create dataset.

## 2. [Dataset engineer] Create a PR to add a new config file

- Add a new yaml file for [T4Dataset config](/autoware_ml/configs/t4dataset) like `db_j6gen2_v1.yaml` after uploading T4dataset.
  - Add a document for the dataset
- Add a new sensor config for [detection3d config](/autoware_ml/configs/detection3d/dataset/t4dataset) like `x2_gen2.py`.

```yaml
# db_j6gen2_v1.yaml

version: 1

train:
  - e6d0237c-274c-4872-acc9-dc7ea2b77943
val:
  - 3013c354-2492-447b-88ce-70ec0438f494
test:
  - 13351af0-41cb-4a96-9553-aeb919efb46e
```

```py
# x2_gen2.py
dataset_version_list = ["db_j6gen2_v1"]
```

For PR review list with code owner

- [ ] Modify the dataset config files
- [ ] Update documentation of dataset

## 3. [User] Download the new dataset by [download_t4dataset](/pipelines/webauto/download_t4dataset/)

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
```
