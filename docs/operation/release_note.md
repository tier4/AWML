## Main topic of next release

We plan for next release v0.4.0 as below.

- CenterPoint

As main topics of this version, we release CenterPoint model.
We evaluate for T4dataset.

### Core library

- fix(autoware_ml): fix bicycles without riders (https://github.com/tier4/autoware-ml/pull/176)
- fix(github): update code owners (https://github.com/tier4/autoware-ml/pull/188)
- fix(github): fix CI/CD
  - https://github.com/tier4/autoware-ml/pull/172
- fix: update dataset
  - https://github.com/tier4/autoware-ml/pull/175
  - https://github.com/tier4/autoware-ml/pull/177
  - https://github.com/tier4/autoware-ml/pull/182
  - https://github.com/tier4/autoware-ml/pull/191
- fix: update document
  - https://github.com/tier4/autoware-ml/pull/174
  - https://github.com/tier4/autoware-ml/pull/178
  - https://github.com/tier4/autoware-ml/pull/183
  - https://github.com/tier4/autoware-ml/pull/184

### Tools

- rerun_visualizer
  - feat: color lidar pointcloud based on height in rerun https://github.com/tier4/autoware-ml/pull/190
  - feat: visualize score in rerun https://github.com/tier4/autoware-ml/pull/189

### Pipelines

- feat: add rosbag delete option (https://github.com/tier4/autoware-ml/pull/179)

### Projects

- FRNet
  - feat(FRNet): enable TensorRT deployment & inference (https://github.com/tier4/autoware-ml/pull/150)
  - fix(FRNet): deployment imports (https://github.com/tier4/autoware-ml/pull/187)
- TransFusion
  - fix(TransFusion): use ghcr docker image (https://github.com/tier4/autoware-ml/pull/186)
- BEVFusion
  - chore(BEVFusion): Add checkpoint path to the bevfusion offline model doc (https://github.com/tier4/autoware-ml/pull/192)

### Chore
