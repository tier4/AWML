## Main topic of next release

We will release next version of `autoware-ml`.
The main topic is following.

- Start to dataset versioning and change config architecture

We started dataset versioning.
We changed from `DB v1.0` to `DB JPNTAXI v1.0` and changed config files in autoware-ml from `database_v1_3.yaml` to `db_jpntaxi_v3.yaml`.
Please see [document](https://github.com/tier4/autoware-ml/blob/main/docs/design/architecture_dataset.md) in detail.

In addition to this, we changed the architecture of configs for T4dataset because the config files for T4dataset is duplicate like `autoware_ml/configs/detection2d/datasset/t4dataset/*.yaml` and `autoware_ml/configs/classification2d/datasset/t4dataset/*.yaml`.
New config directory architecture is following.

```
- configs/
  - t4dataset/
    - db_gsm8_v1.yaml
    - db_j6_v1.yaml
    - db_j6_v2.yaml
    - db_jpntaxi_v1.yaml
    - db_jpntaxi_v2.yaml
    - db_jpntaxi_v3.yaml
    - db_tlr_v1.yaml
    - db_tlr_v2.yaml
    - db_tlr_v3.yaml
    - db_tlr_v4.yaml
    - pseudo_j6_v1.yaml
    - pseudo_j6_v2.yaml
  - classification2d/
    - dataset/
      - t4dataset/
        - tlr_finedetector.py
  - detection2d/
    - dataset/
      - t4dataset/
        - tlr_finedetector.py
  - detection3d/
    - dataset/
      - t4dataset/
        - pretrain.py
        - base.py
        - x2.py
        - xx1.py
```

Please see https://github.com/tier4/autoware-ml/pull/332 and https://github.com/tier4/autoware-ml/pull/334 in detail

- Integrated auto/semi-auto label pipeline and started to use pseudo T4dataset

We add the pipeline of auto labeling for 3D detection.
This tool can be used for auto labeling and semi-auto labeling in 3D detection.
Please see [auto_labeling_3d](https://github.com/tier4/autoware-ml/tree/main/tools/auto_labeling_3d) and [document](https://github.com/tier4/autoware-ml/blob/main/docs/design/architecture_dataset.md) in detail.

- Integrated fine tuning strategy

We integrated fine tuning strategy for 3D detection.
Please see [experiment result](https://github.com/tier4/autoware-ml/issues/148) and [The PR updating configs](https://github.com/tier4/autoware-ml/pull/320) in detail.

- Release BEVFusion-CL-offline base/0.1

We release BEVFusion-CL-offline, which is offline model of BEVFusion for Camera-LiDAR fusion input.
Please see [experiment result](https://github.com/tier4/autoware-ml/issues/148) in detail.

### Core library

- Dataset
  - https://github.com/tier4/autoware-ml/pull/327 Add Pseudo J6 v1.0 and v2.0
  - https://github.com/tier4/autoware-ml/pull/303 Update DB JPNTAXI v3.0
  - https://github.com/tier4/autoware-ml/pull/299 Add DB J6 v2.0
- Change config architecture
  - https://github.com/tier4/autoware-ml/pull/332
  - https://github.com/tier4/autoware-ml/pull/334
- Update pre-commit
  - https://github.com/tier4/autoware-ml/pull/337

### Tools

- Add auto_labeling_3d
  - https://github.com/tier4/autoware-ml/pull/323

### Pipelines

- WebAuto
  - https://github.com/tier4/autoware-ml/pull/336 update document

### Projects

- CenterPoint
  - https://github.com/tier4/autoware-ml/pull/326 Fix URL
- BEVFusion
  - https://github.com/tier4/autoware-ml/pull/162 Add onnx export & tensorrt deployment

### Chore

- Update dataset
  - https://github.com/tier4/autoware-ml/pull/312
  - https://github.com/tier4/autoware-ml/pull/329
  - https://github.com/tier4/autoware-ml/pull/330
  - https://github.com/tier4/autoware-ml/pull/338
