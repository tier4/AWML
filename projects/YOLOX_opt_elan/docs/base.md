# Deployed model for YOLOX_opt-S-elan base/1.X
## Summary

- Main parameter
  - input_size = (960, 960)
- Performance summary
  - Dataset: val dataset of db_gsm8_v2 + db_jpntaxi_v1
  - iou_threshold: 0.5

|          | mAP    | car    | truck  | bus    | trailer | motorcycle | pedestrian | bicycle | unknown |
| -------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| base/1.0 | 0.539  | 0.723  | 0.533  | 0.678  | 0.000  | 0.000  | 0.632  | 0.131  | 0.000  |

## Release

### base/1.0

- This model is pretrained with BDD100K and finetuned with TIER IV's in-house dataset.
- It achieved higher accuracy with YOLOX-S+ 960x960 INT8 than YOLOX-M 960x608, while maintaining the same execution time as YOLOX-Tiny 960x608 FP16.

<details>
<summary> The link of data and evaluation result </summary>

- Model
  - Training dataset: BDD100K and db_gsm8_v2 + db_jpntaxi_v1 (total frames: 104356)
  - [Config file path](https://github.com/tier4/AWML/blob/main/projects/YOLOX_opt_elan/configs/t4dataset/YOLOX_opt-S-DynamicRecognition/yolox-s-opt-elan_960x960_300e_t4dataset.py)
  - Training results [model-zoo]
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/yolox-opt-elan/yolox-s-roi/t4base/v1.0/logs.zip)
    - [checkpoint_best.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/yolox-opt-elan/yolox-s-roi/t4base/v1.0/epoch_24.pth)
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/yolox-opt-elan/yolox-s-roi/t4base/v1.0/yolox-s-opt-elan_960x960_300e_t4dataset.py)
  - train time: NVIDIA RTX 6000 4GB * 1 * 600 epochs = 14 days
- Evaluation result with DB JPNTAXI v2.0 Nishishinjuku  (total frames: 1200):

```python
---------------iou_thr: 0.5---------------
+------------+------+-------+--------+-------+
| class      | gts  | dets  | recall | ap    |
+------------+------+-------+--------+-------+
| unknown    | 0    | 0     | 0.000  | 0.000 |
| car        | 8749 | 65683 | 0.853  | 0.723 |
| truck      | 3877 | 45562 | 0.702  | 0.533 |
| bus        | 316  | 3812  | 0.747  | 0.678 |
| trailer    | 0    | 11    | 0.000  | 0.000 |
| motorcycle | 0    | 0     | 0.000  | 0.000 |
| pedestrian | 2963 | 30943 | 0.749  | 0.632 |
| bicycle    | 383  | 10958 | 0.178  | 0.131 |
+------------+------+-------+--------+-------+
| mAP        |      |       |        | 0.539 |
+------------+------+-------+--------+-------+

```

</details>
