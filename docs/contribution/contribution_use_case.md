# Use case for contribution

Choose the PR type from the list below.
Note that you need to make as small PR as possible.

## Add/Fix functions to `autoware_ml`

If you want to add/fix functions to use for many projects, you should commit to `autoware_ml/*`.
It is the library used for many projects and needs maintenance, so PR is reviewed on the point of code quality, doc string, type hint.

For PR review list with code owner

- [ ] Write the log of test for training
- [ ] Update documentation
- [ ] Check/Add/Update unit test

## Fix code in `/tools`

If you want to add/fix tools to use for many projects, you should commit to `tools/*`.
It is used for many projects and needs maintenance, so PR is reviewed on the point of code quality, doc string, type hint.

For PR review list with code owner

- [ ] Write the log of test for tools
- [ ] Update documentation

## Fix code in `/pipelines`

If you want to add/fix pipelines to use for many projects, you should commit to `pipelines/*`.
It is used for many deploy projects, so PR is reviewed on the point of code quality, doc string, type hint.

For PR review list with code owner

- [ ] Write the log of test for pipeline
- [ ] Update documentation

## Fix code in `/projects`

You can fix code in a project more casually than fixing codes with `autoware_ml/*` because the area of ​​influence is small.
However, if the model is used for Autoware and if you want to change a model architecture, you need to check deploying to onnx and running at ROS environment.

For PR review list with code owner for the project

- [ ] Write the log of result for the trained model
- [ ] Upload the model and logs
- [ ] Update documentation for the model
- [ ] Check deploying to onnx file and running at Autoware environment (If the model is used for Autoware and you change model architecture)

## Update dataset

If you want to update dataset, please modify [the dataset config files](/autoware_ml/configs/t4dataset/).

For PR review list with code owner

- [ ] Modify the dataset config files
- [ ] Update documentation of dataset

### 1. Create a new vehicle dataset

When you add a new dataset for a certain vehicle, it should begin with "v1.0". Specifically, please name it as "DB/UC/Pseudo {new vehicle name} v(X+1).0".

- 1.1. [Dataset engineer] Create dataset and upload to WebAuto system.
- 1.2. [Dataset engineer] Create a PR to add a new config file
  - Add a new yaml file for [T4Dataset config](/autoware_ml/configs/t4dataset) like `db_j6gen2_v1.yaml` after uploading T4dataset.
    - Add a document for the dataset
  - Add a new sensor config for [detection3d config](/autoware_ml/configs/detection3d/dataset/t4dataset) like `x2_gen2.py`.

```yaml
# db_j6gen2_v1.yaml

version: 1
dataset_version: db-j6gen2-v1.0
docs: |
  Product: J6Gen2
  Place: Odaiba
  Amount: About 5000 frames
  Sensor: Hesai LiDAR + C1 Camera + Radar data
  Annotation: All the data are collected at 10Hz and annotated at 2Hz

train:
  - e6d0237c-274c-4872-acc9-dc7ea2b77943 #DB-J6Gen2-v2-odaiba_0
val:
  - 3013c354-2492-447b-88ce-70ec0438f494 #DB-J6Gen2-v2-odaiba_1
test:
  - 13351af0-41cb-4a96-9553-aeb919efb46e #DB-J6Gen2-v2-odaiba_2
```

```py
# x2_gen2.py
dataset_version_list = ["db_j6gen2_v1"]
```

- 1.3. [User] Download the new dataset by [download_t4dataset](/pipelines/webauto/download_t4dataset/).

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

### 2. Add a new dataset (a set of T4dataset) for a vehicle with an existing dataset

If you want to add a new dataset for an existing vehicle (e.g. from a different operating area), please increment "X" from the existing dataset. Specifically, please name it as "DB/UC/Pseudo {new vehicle name} v(X+1).0" if the dataset vX.Y already exists.

- 2.1. [Dataset engineer] Create dataset and upload to WebAuto system.
- 2.2. [Dataset engineer] Create a PR to add a new config file
  - Add a new yaml file for [T4Dataset config](/autoware_ml/configs/t4dataset) like `db_j6gen2_v2.yaml` after uploading T4dataset.
    - Add a document for the dataset
  - Fix a sensor config for [detection3d config](/autoware_ml/configs/detection3d/dataset/t4dataset) like `x2_gen2.py`.

```yaml
# db_j6gen2_v2.yaml

version: 1
dataset_version: db-j6gen2-v2.0
docs: |
  Product: J6Gen2
  Place: Shiojiri
  Amount: About 5000 frames
  Sensor: Hesai LiDAR + C1 Camera + Radar data
  Annotation: All the data are collected at 10Hz and annotated at 2Hz

train:
  - 80b37b8c-ae9d-4641-a921-0b0c2012eee8 #DB-J6Gen2-v2-odaiba_0
val:
  - c8cf2fe3-9097-4f8d-8984-e99c4ddd0ced #DB-J6Gen2-v2-odaiba_1
test:
  - 9e973a55-3f70-48e0-8b37-a68b66a99686 #DB-J6Gen2-v2-odaiba_2
```

```py
# x2_gen2.py
dataset_version_list = ["db_j6gen2_v1", "db_j6gen2_v2"]
```

- 2.3. [User] Download new dataset by [download_t4dataset](/pipelines/webauto/download_t4dataset/).

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

### 3. Add some new T4datasets for an existing dataset

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
- 3.1. [Dataset engineer] Create dataset and upload to WebAuto system.
- 3.2. [Dataset engineer] Update `T4dataset WebAuto version` if you want to modify dataset.
- 3.3. [github CI] Add a T4dataset ID to yaml file of [T4dataset config](/autoware_ml/configs/t4dataset).
  - Update `dataset_version` from X.Y to X.(Y+1)

```yaml
# db_j6gen2_v2.yaml

version: 1
dataset_version: db-j6gen2-v2.0
docs: |
  Product: J6Gen2
  Place: Shiojiri
  Amount: About 5000 frames
  Sensor: Hesai LiDAR + C1 Camera + Radar data
  Annotation: All the data are collected at 10Hz and annotated at 2Hz

train:
  - 80b37b8c-ae9d-4641-a921-0b0c2012eee8 #DB-J6Gen2-v2-odaiba_0
  - 4d50abff-427f-4fa8-9c04-99dc13a3a836 #DB-J6Gen2-v2-odaiba_3
val:
  - c8cf2fe3-9097-4f8d-8984-e99c4ddd0ced #DB-J6Gen2-v2-odaiba_1
  - a1f10b82-6f10-47ab-a253-a12a2f131929 #DB-J6Gen2-v2-odaiba_4
test:
  - 9e973a55-3f70-48e0-8b37-a68b66a99686 #DB-J6Gen2-v2-odaiba_2
  - 54a6cc24-ec9d-47f5-b2bf-813d0da9bf47 #DB-J6Gen2-v2-odaiba_5
```

- 3.4. [User] Download new dataset by [download_t4dataset script](/pipelines/webauto/download_t4dataset/).
  - If `T4dataset WebAuto version` is updated, the script download new version of T4dataset.

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

## Release new model

If you want to release new model, you may add/fix config files in `projects/{model_name}/configs/*.py`.
After creating the model, update the documentation for release note of models in addition to the PR.
You can refer [the release note of CenterPoint base/1.X](projects/CenterPoint/docs/CenterPoint/v1/base.md).
The release note include

- Explain why you change the config or add dataset, what purpose you make a new model.
- The URL link of model
- The URL link of PR
- The config file
- Evaluation result

This is template for release note.
Please feel free to add a figure, graph, table to explain why you change.

Note that you should use commit hash for config file path after first PR changing configs is merged.

```md
### base/0.4

- We added DB JPNTAXI v3.0 for training.
- mAP of (DB JPNTAXI v1.0 + DB JPNTAXI v2.0 test dataset, eval range 90m) is as same as the model of base/0.3.

|          | mAP  | car  | truck | bus  | bicycle | pedestrian |
| -------- | ---- | ---- | ----- | ---- | ------- | ---------- |
| base/0.4 | 68.5 | 81.7 | 62.4  | 83.5 | 50.9    | 64.1       |
| base/0.3 | 68.1 | 80.5 | 58.0  | 80.8 | 58.0    | 63.2       |

<details>
<summary> The link of data and evaluation result </summary>

- model
  - Training dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v3.0 + DB GSM8 v1.0 + DB J6 v1.0 (total frames: 34,137)
  - [PR]()
  - [Config file path]()
  - [Checkpoint]()
  - [Training log]()
  - [Deployed onnx model]()
  - [Deployed ROS parameter file]()
  - train time: (A100 * 4) * 3 days
- Total mAP: 0.685 for all dataset
  - Test dataset of DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v3.0 + DB GSM8 v1.0 + DB J6 v1.0 (total frames: 62)
  - Eval range = 90m

| class_name | Count | mAP | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ----- | --- | ------- | ------- | ------- | ------- |
| car        |       |     |         |         |         |         |
| truck      |       |     |         |         |         |         |
| bus        |       |     |         |         |         |         |
| bicycle    |       |     |         |         |         |         |
| pedestrian |       |     |         |         |         |         |

- Total mAP: 0.685 for X2 dataset
  - Test dataset of DB GSM8 v1.0 + DB J6 v1.0 (total frames: 62)
  - Eval range = 90m

| class_name | Count | mAP | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ----- | --- | ------- | ------- | ------- | ------- |
| car        |       |     |         |         |         |         |
| truck      |       |     |         |         |         |         |
| bus        |       |     |         |         |         |         |
| bicycle    |       |     |         |         |         |         |
| pedestrian |       |     |         |         |         |         |

</details>

```

For PR review list with code owner

- [ ] Write the log of result for the trained model
- [ ] Upload the model and logs
- [ ] Update documentation for the model
- [ ] Check deploying to onnx file and running at Autoware environment (If the model is used for Autoware and you change model architecture)
- [ ] Write results of training and evaluation including analysis for the model in PR.

## Add a new algorithm / a new tool
### Choice 1. Merge to original MMLab libraries

Note that if you want to new algorithm, basically please make PR for [original MMLab libraries](https://github.com/open-mmlab).
After merged by it for MMLab libraries like [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) and [mmdetection](https://github.com/open-mmlab/mmdetection), we update the version of dependency of MMLab libraries and make our configs in `/projects` for TIER IV products.
If you want to add a config to T4dataset or scripts like onnx deploy for models of MMLab's model, you should add codes to `/projects/{model_name}/`.

For PR review list with code owner

- [ ] Write why you add a new model
- [ ] Add code for `/projects/{model_name}`
- [ ] Add `/projects/{model_name}/README.md`
- [ ] Write the result log for new model

When creating the PR, we recommend writing a PR summary as https://github.com/tier4/autoware-ml/pull/134.
We would you like to write summary of the new model considering the case when some engineers want to catch up.
If someone want to catch up this model, it is best situation that they need to see only github and do not need to search the information in Jira ticket jungle, Confluence ocean, and Slack universe.

### Choice 2. Make on your repository

As another way, which we recommend for especially researcher, you can make a new algorithm or a new tool on your repository.
The repository [mm-project-template](https://github.com/scepter914/mm-project-template) is one example of template repository.
You can start from this template and you can add code of `/tools/*` and `/projects/*` from `autoware-ml` to use for your a new algorithm or a new tool.
We are glad if you want to contribute to `autoware-ml` and the PR to add for the document of [community_support](/docs/tips/community_support.md).
We hope it promotes the community of robotics ML researchers and engineers.

For PR review list with code owner

- [ ] Add your algorithm or your tool for [community_support](/docs/tips/community_support.md)
