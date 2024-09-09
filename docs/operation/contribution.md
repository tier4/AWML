# Contribution docs
## Tools for developer

- If you develop `autoware-ml`, we recommend some tools as below.
- [yapf](https://github.com/google/yapf)

```
pip install yapf
```

- [eeyore.yapf](https://marketplace.visualstudio.com/items?itemName=eeyore.yapf)
  - VSCode extension

## Contribute flow
### 1. Report issue

If you want to add/fix some code, comment and show that you want to fix the issue before implementation.

### 2. Implementation

You should fork from autoware-ml to own repository and make new branch.

### 3. Test on integration test

For now, integration test is done on local environment.
See [test_integration](/tools/test_integration) for internal user.

### 4. Make PR

When you make the PR, you check the list as below "Use case for contribute".

## PR rule
### PR Strategy

Because we have few resource to develop `autoware-ml`, we merge the PR only if the code quality reach enough to be maintained.
If you make the PR that make new feature but lack of maintainability, we will not merge.

If you make prototype level code but it is useful feature for `autoware-ml`, please make fork repository and let us by issues.
After considering whether to integration and prioritizing with other developing items, we will integrate to `autoware-ml`.

### PT title

[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) can generate categorized changelogs, for example using [git-cliff](https://github.com/orhun/git-cliff).

```
feat(autoware-ml): add loss function
```

If your change breaks some interfaces, use the ! (breaking changes) mark as follows:

```
feat(autoware-ml)!: change function name from function_a() to function_b()
```

You can use the following definition.

- feat
- fix
- chore
- ! (breaking changes)

### Document

If you add some feature, you need to add document like `README.md`.
The target of document is as below.

- `/docs/*`: Design documents for developers

Design documents aims for developers.
So you should write "why we should do" for documents.

- `/tools/*`: Process documents for engineer users

Process documents aims for engineer users.
So you should write "how we should do" for documents assuming that users know basic command linux around machine learning.

- `/pipelines/*`: Process documents for non-engineer users

Process documents aims for non-engineer users.
So you should write "how we should do" for documents assuming that users do not know basic linux command.

## Use case for contribute

You choose PR type as below.
Note that you need to make as small PR as possible.

### Add/Fix functions to `autoware_ml`

If you want to add/fix functions to use for many projects, you should commit to `autoware_ml/*`.
It is the library used for many projects and need to maintenance, so PR is reviewed on the point of code quality, doc string, type hint.

For PR review list with code owner
- [ ] Performing test for function
- [ ] Update docs
- [ ] Check/Add/Update unit test

### Fix code in `/tools`

If you want to add/fix tools to use for many projects, you should commit to `tools/*`.
It is used for many projects and need to maintenance, so PR is reviewed on the point of code quality, doc string, type hint.

For PR review list with code owner
- [ ] Performing test for tools
- [ ] Update docs

### Fix code in `/pipelines`

If you want to add/fix pipelines to use for many projects, you should commit to `pipelines/*`.
It is used for many deploy projects, so PR is reviewed on the point of code quality, doc string, type hint.

For PR review list with code owner
- [ ] Performing test for pipelines
- [ ] Update docs

### Fix code in `/projects`

You can fix code in a project more casually than fixing codes with `autoware_ml/*` because the area of ​​influence is small.
However, if the model is used for Autoware and if you want to change a model architecture, you need to check deploying to onnx and running at ROS environment.

For PR review list with code owner for the project
- [ ] Upload the model and logs
- [ ] Write the result log for the model
- [ ] Update docs
- [ ] Check deploying to onnx file and running at Autoware environment (If the model is used for Autoware and you change model architecture)

### Add a new algorithm

Note that if you want to new algorithm, basically please make PR for [original MMLab libraries](https://github.com/open-mmlab).
After merged by it for MMLab libraries like [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) and [mmdetection](https://github.com/open-mmlab/mmdetection), we update the version of dependency of MMLab libraries and make our configs in `/projects` for TIER IV products.

If you want to add a config to T4dataset or scripts like onnx deploy for models of MMLab's model, you should add codes to `/projects/{model_name}/`.

For PR review list with code owner
- [ ] Write why you add a new model
- [ ] Add code for `/projects/{model_name}`
- [ ] Add `/projects/{model_name}/README.md`
- [ ] Write the result log for new model

### Update dataset

If you want to update dataset, you change [dataset config](/autoware_ml/configs/detection3d/dataset/t4dataset/).

For PR review list with code owner
- [ ] Change `/autoware_ml/configs/detection3d/dataset/t4dataset/`
- [ ] Update docs

### Release new model

If you want to release new model, you need to make two PRs.
First, you add/fix config files in `projects/{model_name}/configs/{dataset_name}/*.py` for new model like adding new dataset.

Second, you update docs for release note of models.
You can refer [TransFusion release note](/projects/TransFusion/docs/deployed_xx1_model.md)
In addition to docs, you should write results of training and evaluation including analysis for the model in PR.

This is template for release note.
Note that you should use commit hash for config file path after first PR changing configs is merged.

```
- model
  - Training dataset:
  - Eval dataset:
  - [PR]()
  - [Config file path]()
  - [Deployed onnx model]()
  - [Deployed ROS parameter file]()
  - [Training results]()
  - train time: (A100 * 4) * 2 days
- Total mAP:
  - Test dataset:
  - Eval range = 90m

| model | range | mAP | car | truck | bus | bicycle | pedestrian |
| ----- | ----- | --- | --- | ----- | --- | ------- | ---------- |
|       |       |     |     |       |     |         |            |
```

For PR review list with code owner
- [ ] Upload the model and logs
- [ ] Delete unused config file
- [ ] Update `projects/{model_name}/docs/deployed_*_model.md` adding evaluation result for new config.
- [ ] Write results of training and evaluation including analysis for the model in PR.
