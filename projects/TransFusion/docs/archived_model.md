# Archived results and models

## T4dataset for XX1
### TransFusion-L t4xx1_75m_v1 (pillar 0.3m * grid 512 = 76.8m)

- Note that this model is low performance because of autoware-ml bug.
- model
  - Training dataset: database_v1_0
  - Eval dataset: database_v1_1 (Bug fix and [new dataset config is applied](https://github.com/tier4/autoware-ml/pull/31))
  - [Config file](https://github.com/tier4/autoware-ml/blob/17e8944ac2154f1f1042a507a4001ccf057ffe78/projects/TransFusion/configs/t4dataset/transfusion_lidar_pillar02_second_secfpn_1xb4-cyclic-20e_t4xx1.py)
  - [Deployed onnx model](https://awf.ml.dev.web.auto/perception/models/transfusion/v1/transfusion.onnx)
- val
  - Total mAP: 0.485

| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ---- | ------- | ------- | ------- | ------- |
| car        | 69.8 | 55.1    | 70.7    | 75.5    | 77.7    |
| truck      | 39.5 | 19.8    | 39.4    | 46.8    | 52.0    |
| bus        | 47.6 | 30.3    | 46.3    | 55.1    | 58.8    |
| bicycle    | 33.4 | 30.6    | 33.3    | 34.0    | 35.6    |
| pedestrian | 52.0 | 46.6    | 50.6    | 53.3    | 57.6    |
