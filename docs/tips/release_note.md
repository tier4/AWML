## Main topic of next release

We will release next version of `autoware-ml`.
The main topic is following.

- Start to dataset versioning

We start dataset versioning.
We changed from `DB v1.0` to `DB JPNTAXI v1.0` and changed config files in autoware-ml according to it.
Please see [document](https://github.com/tier4/autoware-ml/blob/main/docs/design/architecture_dataset.md) in detail.

- Integrated auto/semi-auto label pipeline and started to use pseudo T4dataset

We add the pipeline of auto labeling for 3D detection.
This tool can be used for auto labeling and semi-auto labeling in 3D detection.
Please see [auto_labeling_3d](https://github.com/tier4/autoware-ml/tree/main/tools/auto_labeling_3d) and [document](https://github.com/tier4/autoware-ml/blob/main/docs/design/architecture_dataset.md) in detail.

- Integrated fine tuning strategy

We integrated fine tuning strategy for 3D detection.
Please see [experiment result](https://github.com/tier4/autoware-ml/issues/148) in detail.

- Integrated model management with S3

We start to manage the trained model and data like log file and config file by AWS S3.
Please see [document](https://github.com/tier4/autoware-ml/blob/main/docs/design/architecture_s3.md) in detail.

- Release BEVFusion-CL-offline base/0.1

We release BEVFusion-CL-offline, which is offline model of BEVFusion for Camera-LiDAR fusion input.
Please see [experiment result](https://github.com/tier4/autoware-ml/issues/148) in detail.

### Core library

- Update DB JPNTAXI v3.0
  - https://github.com/tier4/autoware-ml/pull/303

### Tools

### Pipelines

### Projects

- Fix URL in CenterPoint
  - https://github.com/tier4/autoware-ml/pull/326
- Add auto_labeling_3d
  - https://github.com/tier4/autoware-ml/pull/323

### Chore

- Update dataset
  - https://github.com/tier4/autoware-ml/pull/312
  - https://github.com/tier4/autoware-ml/pull/329
  - https://github.com/tier4/autoware-ml/pull/330
