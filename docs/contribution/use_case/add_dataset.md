# Add some new T4datasets for an existing dataset

If you want to add a new T4dataset to an existing dataset config file, please update the version from "DB/UC/Pseudo {new vehicle name} vX.Y" to "DB/UC/Pseudo {new vehicle name} vX.(Y+1)".

- The use cases are following.
  - [Example usecase 1] You want to change the trailer annotation method and modify some of the annotations for "DB/UC/Pseudo {new vehicle name} vX.Y".
    - In this case, please create a new T4dataset with a new T4dataset ID for the T4dataset you modified, since this case leads to a destructive change for T4dataset format. After updating all the T4dataset IDs you have changed, please update the dataset version from "vX.Y" to "vX.(Y+1)".
  - [Example usecase 2] 2D annotation did not exist in version X.Y, so we add it.
    - Same as above. Please update the dataset version from "vX.Y" to "vX.(Y+1)".
  - [Example usecase 3] In version X.Y, we found one vehicle that was not annotated, so we added and modified it by annotating it.
    - Please update `T4dataset WebAuto version`, modify the config file accordingly, and update  "vX.Y" to "vX.(Y+1)". Note that you do not need to create a new T4dataset with different T4dataset ID since this is not a destructive change.
  - [Example usecase 4] For pointcloud topic stored in rosbag of T4dataset, the data arrangement method was changed from XYZI to XYZIRC, and the contents of rosbag were also updated.
    - Please update both T4format and `T4dataset WebAuto version`, and update the dataset version from "vX.Y" to "vX.(Y+1)".

## 1. [Dataset engineer] Create dataset and upload to WebAuto system.

By using WebAuto system, a dataset engineer create dataset.

## 2. [Dataset engineer] Update `T4dataset WebAuto version` if you want to modify dataset.

By using WebAuto system, a dataset engineer modify dataset.

## 3. [github CI] Make PR to add a T4dataset ID to yaml file of [T4dataset config](/autoware_ml/configs/t4dataset).

- Update `dataset_version` from X.Y to X.(Y+1)

```yaml
# db_j6gen2_v2.yaml

version: 1
train:
  - 80b37b8c-ae9d-4641-a921-0b0c2012eee8
  - 4d50abff-427f-4fa8-9c04-99dc13a3a836
val:
  - c8cf2fe3-9097-4f8d-8984-e99c4ddd0ced
  - a1f10b82-6f10-47ab-a253-a12a2f131929
test:
  - 9e973a55-3f70-48e0-8b37-a68b66a99686
  - 54a6cc24-ec9d-47f5-b2bf-813d0da9bf47
```

For PR review list with code owner

- [ ] Modify the dataset config files
- [ ] Update documentation of dataset

## 4. [User] Download new dataset by [download_t4dataset script](/pipelines/webauto/download_t4dataset/).

- If `T4dataset WebAuto version` is updated, the script download new version of T4dataset.
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
      - 1/
    - 4d50abff-427f-4fa8-9c04-99dc13a3a836/
      - 0/
      - 1/
    - c8cf2fe3-9097-4f8d-8984-e99c4ddd0ced/
      - 0/
      - 1/
    - a1f10b82-6f10-47ab-a253-a12a2f131929/
      - 0/
    - 9e973a55-3f70-48e0-8b37-a68b66a99686/
      - 0/
    - 54a6cc24-ec9d-47f5-b2bf-813d0da9bf47/
      - 0/
```
