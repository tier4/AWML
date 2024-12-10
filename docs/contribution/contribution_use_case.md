# Use case for contribution

You choose PR type as below.
Note that you need to make as small PR as possible.

## Add/Fix functions to `autoware_ml`

If you want to add/fix functions to use for many projects, you should commit to `autoware_ml/*`.
It is the library used for many projects and need to maintenance, so PR is reviewed on the point of code quality, doc string, type hint.

For PR review list with code owner

- [ ] Write the log of test for training
- [ ] Update docs
- [ ] Check/Add/Update unit test

## Fix code in `/tools`

If you want to add/fix tools to use for many projects, you should commit to `tools/*`.
It is used for many projects and need to maintenance, so PR is reviewed on the point of code quality, doc string, type hint.

For PR review list with code owner

- [ ] Write the log of test for tools
- [ ] Update docs

## Fix code in `/pipelines`

If you want to add/fix pipelines to use for many projects, you should commit to `pipelines/*`.
It is used for many deploy projects, so PR is reviewed on the point of code quality, doc string, type hint.

For PR review list with code owner

- [ ] Write the log of test for pipeline
- [ ] Update docs

## Fix code in `/projects`

You can fix code in a project more casually than fixing codes with `autoware_ml/*` because the area of ​​influence is small.
However, if the model is used for Autoware and if you want to change a model architecture, you need to check deploying to onnx and running at ROS environment.

For PR review list with code owner for the project

- [ ] Write the log of result for the trained model
- [ ] Upload the model and logs
- [ ] Update docs for the model
- [ ] Check deploying to onnx file and running at Autoware environment (If the model is used for Autoware and you change model architecture)

## Update dataset

If you want to update dataset, you change [dataset config](/autoware_ml/configs/detection3d/dataset/t4dataset/).

For PR review list with code owner
- [ ] Change `/autoware_ml/configs/detection3d/dataset/t4dataset/`
- [ ] Update docs of dataset

## Release new model

If you want to release new model, you may add/fix config files in `projects/{model_name}/configs/*.py`.
After making the model, you update docs for release note of models in addition to the PR.
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

- We added DB1.3 for training.
- mAP of (DB1.0 + 1.1 test dataset, eval range 90m) is as same as the model of base/0.3.

|          | mAP  | car  | truck | bus  | bicycle | pedestrian |
| -------- | ---- | ---- | ----- | ---- | ------- | ---------- |
| base/0.4 | 68.5 | 81.7 | 62.4  | 83.5 | 50.9    | 64.1       |
| base/0.3 | 68.1 | 80.5 | 58.0  | 80.8 | 58.0    | 63.2       |

<details>
<summary> The link of data and evaluation result </summary>

- model
  - Training dataset:
  - Eval dataset:
  - [PR]()
  - [Config file path]()
  - [Checkpoint]()
  - [Training log]()
  - [Deployed onnx model]()
  - [Deployed ROS parameter file]()
  - train time: (A100 * 4) * 3 days
- Total mAP: 0.685
  - Dataset: DB1.0 + DB1.1 + DB2.0 L + DB3.0 test dataset
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
- [ ] Update docs for the model
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

When you make the PR, we recommend to write PR summary as https://github.com/tier4/autoware-ml/pull/134.
We would you like to write summary of the new model considering the case when some engineers want to catch up.
If someone want to catch up this model, it is best situation that they need to see only github and do not need to search the information in Jira ticket jungle, Confluence ocean, and Slack universe.

### Choice 2. Make on your repository

As another way, which we recommend for especially researcher, you can make a new algorithm or a new tool on your repository.
The repository [mm-project-template](https://github.com/scepter914/mm-project-template) is one example of template repository.
You can start from this template and you can add code of `/tools/*` and `/projects/*` from `autoware-ml` to use for your a new algorithm or a new tool.
We are glad if you want to contribute to `autoware-ml` and the PR to add for the document of [community_support](/docs/tips/community_support.md).
We hope it leads to promote the community around robotics ML researcher and ML engineer.

For PR review list with code owner

- [ ] Add your algorithm or your tool for [community_support](/docs/tips/community_support.md)
