## Main topic of next release

We will release next version of `autoware-ml`.
The main topic is following.

- Integrate auto/semi-auto label pipeline

Please see [auto_labeling_3d](https://github.com/tier4/autoware-ml/tree/main/tools/auto_labeling_3d) in detail.

- Replace from nuscenes_devkit to t4_devkit

Please see [PR](https://github.com/tier4/autoware-ml/pull/238) in detail.

- Integrate model management with S3

Please see [document](https://github.com/tier4/autoware-ml/blob/main/docs/design/architecture_s3.md) in detail.

- Integrate fine tuning strategy

### Core library

- Replace t4-devkit instead of nuscenes_devkit
  - https://github.com/tier4/autoware-ml/pull/238
- Apply black formatter in autoware-ml
  - https://github.com/tier4/autoware-ml/pull/274
  - https://github.com/tier4/autoware-ml/pull/279

### Tools

- add auto_labeling_3d
  - https://github.com/tier4/autoware-ml/pull/275
  - https://github.com/tier4/autoware-ml/pull/276
  - https://github.com/tier4/autoware-ml/pull/277

### Pipelines

### Projects

### Chore

- Update document
  - https://github.com/tier4/autoware-ml/pull/273
