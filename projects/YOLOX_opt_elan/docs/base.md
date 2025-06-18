# Deployed model for YOLOX_opt-S-elan base/1.X
## Summary

- Main parameter
  - input_size = (960, 960)
- Performance summary
  - Dataset: DB JPNTAXI v2.0 Nishishinjuku (total frames: 1200)
  - iou_threshold: 0.5

|          | mAP    | car    | truck  | bus    | trailer | motorcycle | pedestrian | bicycle | unknown |
| -------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| base/1.0 | 0.622  | 0.799  | 0.777  | 0.818  | 0.000  | 0.000  | 0.715 | 0.000 | 0.000 |

## Release

### base/1.0

- This model is trained with tier4/YOLOX and converted to MMDetection format. It is verified that it maintains the same performance.
- Pretrained with BDD100K and finetuned with dbv1_0 and dbv2_0.
- Achieved higher accuracy with YOLOX-S+ 960x960 INT8 than YOLOX-M 960x608, while maintaining the same execution time as YOLOX-Tiny 960x608 FP16.

<details>
<summary> The link of data and evaluation result </summary>

- Model
  - Training dataset: BDD100K + DB JPNTAXI v1.1 Odaiba + DB JPNTAXI v2.0 Nishishinjuku (total frames: 104356)
  - [Config file path]()
  - Training results [model-zoo]
    - [logs.zip]()
    - [checkpoint_best.pth]()
    - [config.py]()

- Evaluation result with DB JPNTAXI v2.0 Nishishinjuku  (total frames: 1200):

```python

---------------iou_thr: 0.5---------------

+------------+------+-------+--------+-------+
| class      | gts  | dets  | recall | ap    |
+------------+------+-------+--------+-------+
| unknown    | 0    | 2514  | 0.000  | 0.000 |
| car        | 6238 | 16393 | 0.902  | 0.799 |
| truck      | 1880 | 5089  | 0.835  | 0.777 |
| bus        | 245  | 845   | 0.878  | 0.818 |
| trailer    | 0    | 2344  | 0.000  | 0.000 |
| motorcycle | 0    | 107   | 0.000  | 0.000 |
| pedestrian | 2323 | 8306  | 0.802  | 0.715 |
| bicycle    | 239  | 0     | 0.000  | 0.000 |
+------------+------+-------+--------+-------+
| mAP        |      |       |        | 0.622 |
+------------+------+-------+--------+-------+

```

</details>
