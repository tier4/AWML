## Main topic of next release

We will release next version of `autoware-ml`.
The main topic is following.

- Replace from nuscenes_devkit to t4_devkit

We replaced from nuscenes_devkit into t4_devkit in data conversion.
From the issue of https://github.com/tier4/autoware-ml/issues/278, if you face error with DB2.0, please update dataset.
Please see [PR](https://github.com/tier4/autoware-ml/pull/238) in detail.

- Add new format for document of release model

In addition to model versioning, We introduced the format of release note.
We ask the developers to write not only evaluation result but also what you change and why you update the model like release note.
We wrote [the document](https://github.com/tier4/autoware-ml/blob/main/docs/contribution/contribution_use_case.md#release-new-model) and [the example in CenterPoint base/v1.0](https://github.com/tier4/autoware-ml/blob/0b60f1a344c5b5ea9c0fdf0c29c6ac227491e15b/projects/CenterPoint/docs/CenterPoint/v1/base.md#centerpoint-base10), please confirm these.

- Change formatter

We changed python formatter from yapf into black because many developers are used to black format.
In addition to it, we add pre-commit for `autoware-ml`.
If you don't install these tools, please follow [the document](https://github.com/tier4/autoware-ml/blob/main/docs/contribution/contribution_flow.md).

### Core library

- Replace t4-devkit instead of nuscenes_devkit
  - https://github.com/tier4/autoware-ml/pull/238
- Add pre-commit
  - https://github.com/tier4/autoware-ml/pull/314
- Add dataset version for dataset config
  - https://github.com/tier4/autoware-ml/pull/301
- Fix CODEOWNERS
  - https://github.com/tier4/autoware-ml/pull/302
- Apply formatter in autoware-ml
  - https://github.com/tier4/autoware-ml/pull/274
  - https://github.com/tier4/autoware-ml/pull/279
  - https://github.com/tier4/autoware-ml/pull/280
  - https://github.com/tier4/autoware-ml/pull/284
  - https://github.com/tier4/autoware-ml/pull/285
  - https://github.com/tier4/autoware-ml/pull/286
  - https://github.com/tier4/autoware-ml/pull/287
  - https://github.com/tier4/autoware-ml/pull/288
  - https://github.com/tier4/autoware-ml/pull/289
  - https://github.com/tier4/autoware-ml/pull/290
  - https://github.com/tier4/autoware-ml/pull/291
  - https://github.com/tier4/autoware-ml/pull/304
  - https://github.com/tier4/autoware-ml/pull/315
  - https://github.com/tier4/autoware-ml/pull/316
  - https://github.com/tier4/autoware-ml/pull/317
  - https://github.com/tier4/autoware-ml/pull/318
  - https://github.com/tier4/autoware-ml/pull/319
- Update github CI for dataset update
  - https://github.com/tier4/autoware-ml/pull/298

### Tools

- add auto_labeling_3d
  - https://github.com/tier4/autoware-ml/pull/275
  - https://github.com/tier4/autoware-ml/pull/276
  - https://github.com/tier4/autoware-ml/pull/277
- test_integration
  - https://github.com/tier4/autoware-ml/pull/311 update config

### Pipelines

### Projects

### Chore

- Update document
  - https://github.com/tier4/autoware-ml/pull/273
  - https://github.com/tier4/autoware-ml/pull/281
  - https://github.com/tier4/autoware-ml/pull/282
  - https://github.com/tier4/autoware-ml/pull/283
  - https://github.com/tier4/autoware-ml/pull/292
  - https://github.com/tier4/autoware-ml/pull/293
  - https://github.com/tier4/autoware-ml/pull/294
  - https://github.com/tier4/autoware-ml/pull/295
  - https://github.com/tier4/autoware-ml/pull/305
  - https://github.com/tier4/autoware-ml/pull/306
  - https://github.com/tier4/autoware-ml/pull/321
  - https://github.com/tier4/autoware-ml/pull/322
