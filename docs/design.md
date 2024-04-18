# Design
## Software architecture
### Data Pipeline

TBD

### Architecture

The architecture of autoware-ml is based on [mmdetection3d v1.4](https://github.com/open-mmlab/mmdetection3d/tree/v1.4.0).

- /docs

Documents for autoware-ml.

- /configs

Base configs for datasets and train parameters.

- /tools

Tools script like dataset converter.

- /autoware_ml

The core library.

- /projects

The projects directory.
This directory has each model projects.

## Contribute
### Add a new model

If you want to add a new model, you should add codes to `/projects/{model_name}`.
If you want to add a project to improve model performance like some training and data pipeline for pseudo label, you should also add codes to `/projects/{model_name}`.

For PR review list with code owner for Autoware-ML
- [ ] Add `/projects/{model_name}/README.md`
- [ ] Performing test for new model
- [ ] Update [release note](/docs/release_note.md)

### Add/Fix functions to use for many projects

If you want to add/fix functions to use for many projects, you should commit to `autoware_ml/*`.

For PR review list with code owner for Autoware-ML
- [ ] Performing test for function
- [ ] Check/Add/Update/ unit test

### Add/Fix tools

If you want to add/fix tools to use for many projects, you should commit to `tools/*`.

For PR review list with code owner for Autoware-ML
- [ ] Performing test for function
- [ ] Check/Add/Update/ unit test

### Fix code in a project.

You can fix code in a project more casually than fixing codes with `autoware_ml/*`.

For PR review list with code owner for the project
- [ ] Performing test with the model
- [ ] Update [release note](/docs/release_note.md)

### Add/Fix config file in a project

If you want to add/fix config file in a project, you should add/fix `projects/{model_name}/configs/{dataset_name}/*.py`.

For PR review list with code owner for the project
- [ ] Performing test with the model
- [ ] Update `projects/{model_name}/README.md` adding evaluation result for new config.
- [ ] Delete unused config file
- [ ] Update [release note](/docs/release_note.md)
