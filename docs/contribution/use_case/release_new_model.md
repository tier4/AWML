# Release new model

This document explains the procedure to release new ML model.

## Step 1. Confirm what to do

Before running the model release cycle, please clarify what you want to do.

- Which model are you going to update? (CenterPoint, PointPainting, YOLOX-opt for traffic light recognition, etc)
- Why do you want to update the model? (You've got a new dataset for new vehicle, update model architecture, etc)
- How do you evaluate the model? (e.g. based on mAP for Odaiba data, AP for pedestrians, etc)

## Step 2. Update the configuration

Please select the method to update the ML model from the below choices.
Who does it depends on the tasks.
You will do multiple updates depending on what you want to do.

### 2.1. [Dataset engineer] Update dataset

When you update the dataset, you can release the new model.
Please select from below choices.

- [Create a new vehicle dataset](use_case/create_new_vehicle_dataset.md)
- [Add a new set of T4dataset for a vehicle with an existing dataset](use_case/create_new_dataset.md)
- [Add some new T4datasets for an existing dataset](use_case/add_dataset.md)

### 2.2. [ML engineer] Update parameter

If you want to change hyper parameters to release new model, you may add/fix config files in `projects/{model_name}/configs/*.py`.
For example, you can change parameters to suit your environment.

```py
# user setting
data_root = "data/t4dataset/"
info_directory_path = "info/user_name/"
train_gpu_size = 2
train_batch_size = 8
test_batch_size = 2
max_epochs = 50
work_dir = "work_dirs/centerpoint/" + _base_.dataset_type + "/second_secfpn_2xb8_121m_base/"
```

## Step 3. Train the model

Please see README.md in the project you want to update.
Note that if you train for the model using Autoware, don't forget

- Make onnx file (Please confirm in README.md of each project)
- [Update onnx metadata](/tools/deploy_to_autoware/)
- [Make the yaml file of ROS parameter](/tools/deploy_to_autoware/)

## Step 4. Create a PR to autoware-ml with a release note for the model

If you want to release new model in a project, you may add/fix config files in configs of dataset or `projects/{model_name}/configs/*.py`.
After creating the model, update the documentation for release note of models in addition to the PR.
You can refer [the release note of CenterPoint base/1.X](/projects/CenterPoint/docs/CenterPoint/v1/base.md).

The release note include

- Explain why you change the config or add dataset, what purpose you make a new model.
  - You should explain the model for Autoware user.
  - You should explain by natural language instead of table of metric score.
  - If you need, you can explain by figures.
- In detail, you should write
  - The URL link of PR
  - The path to the config file
    - You should use the commit hash (like `https://github.com/tier4/autoware-ml/blob/5f472170f07251184dc009a1ec02be3b4f3bf98c/autoware_ml/configs/detection3d/dataset/t4dataset/base.py`) instead of branch name (like `https://github.com/tier4/autoware-ml/blob/main/autoware_ml/configs/detection3d/dataset/t4dataset/base.py`)
  - The URL link of the model (checkpoint) and logs
    - When you make a PR at first, you can write a temporary URL (like google drive) of trained model and logs before upload to S3 or Web.Auto ML Packages.
  - Evaluation result
    - You should write the table of metrics.
  - Deployed onnx model and ROS parameter file
    - When you make a PR at first, you can write a temporary URL as the model (checkpoint).

This is template for release note.
Please feel free to add a figure, graph, table to explain why you change.

```md
### base/0.4

- We added DB JPNTAXI v3.0 for training dataset.
- The performance of base/1.1 is better than base/1.0.
  - The performance in mAP of (DB JPNTAXI v1.0 + DB JPNTAXI v2.0 test dataset, eval range 90m) increase from 68.1 to 68.5 than base/0.3.

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

- [ ] Write the experiment results
- [ ] Write results of training and evaluation including analysis for the model in PR.
- [ ] Upload the model and logs
- [ ] Check deploying to onnx file and running at Autoware environment (If the model is used for Autoware and you change model architecture)

## Step 5. Upload to AWS and Web.Auto ML Packages

After PR is reviewed and fixed, the maintainers asks "Please upload the model to AWS S3 model zoo" to person in charge.
Then he/she will upload the model, logs, onnx files, and ROS parameters to AWS S3 model zoo and Web.Auto ML packages.
After that, he/she change the URL for AWS S3 URL link and the maintainer approve for the PR.
