# Contribution docs
## Contribute flow
### 1. Report issue

If you want to add/fix some code, comment and show that you want to fix the issue before implementation.

### 2. Implementation

You should fork from autoware-ml to own repository and make new branch.

### 3. Formatting

- If you develop `autoware-ml`, we recommend some tools as below.
- [black](https://github.com/psf/black)

```
pip install black
```

- [isort](https://github.com/PyCQA/isort)

```
pip install isort
```

- We recommend VSCode extension
  - [black-formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)
  - [isort](https://marketplace.visualstudio.com/items?itemName=ms-python.isort)

### 4. Test by CI/CD

For now, integration test is done on local environment.
See [test_integration](/tools/test_integration) for internal user.

### 5. Make PR

Please make PR and write the contents of it in English.
When you make the PR, you check the list as below "Use case for contribute".

### 6. Fix from review

When you get comments for PR, you should fix it.

## PR rule
### PR Strategy

- Basically, the code itself becomes technical debt

Basically we do not merge unused code as a library.
We recommend to manage experimental code in a personal repository.
So if you make PR, please explain in detail why it has to be in the library.

- Reduce maintenance costs as much as possible

Because we have few resource to develop `autoware-ml`, we merge the PR only if the code quality reach enough to be maintained.
If you make the PR of new feature that is lack of maintainability, we will not merge.
If you make prototype level code but it is useful feature for `autoware-ml`, please make fork repository and let us by issues.
After considering whether to integration and prioritizing with other developing items, we will integrate to `autoware-ml`.

- Independent software as much as possible

It is very costly to delete a part of feature from software that has various features.
If tools are separated for each feature, it's easy to erase with just one tool when you no longer use it.

- Architecture design is more important than code itself

Regarding PR review, we review the architecture design more than code itself.
While it is easy to fix something like variable name or function composition, it is difficult to change the architecture of software.
So we often comment as "I don't think we need the option for the tool.", "I recommend you separate the tools for A and B.", or "I recommend to use config format of MMLab library instead of many args."
Of course, please keep the code itself as clean as possible at the point you make a PR.

- For core libraries and tools

When you change core libraries or core tools, we will review PRs strictly.
Please follow the PR template and write the changing point as much as possible.
As you judge how core it is, you can refer [Support priority](https://github.com/tier4/autoware-ml/blob/main/docs/design/autoware_ml_design.md#support-priority).

### Separate PR

If you want to change from core library to your tools, you should separate PR.

At first, you should make PR for core library.
You check the pipelines that already exist for your changes.
This PR is reviewed carefully.

After that, you should make PR for your tools.
This PR is reviewed casually.

### PR title

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

When you make the PR, we recommend to write PR summary as https://github.com/tier4/autoware-ml/pull/134.
We would you like to write summary of the new model considering the case when some engineers want to catch up.
If someone want to catch up this model, it is best situation that they need to see only github and do not need to search the information in Jira ticket jungle, Confluence ocean, and Slack universe.

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
