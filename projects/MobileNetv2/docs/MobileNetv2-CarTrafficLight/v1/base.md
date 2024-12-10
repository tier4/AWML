# Deployed model for MobileNetv2-CarTrafficLight base/1.X
## Summary

- Performance summary
  - Precision, Recall, F1 score, Counts for each class
  - Note
    - Eval DB: Evaluation dataset

|          | Eval DB           | precision_top1 | recall_top1 | f1-score_top1 | counts |
| -------- | ----------------- | -------------- | ----------- | ------------- | ------ |
| base/1.0 | tlr_v0_1+tlr_v1_2 | 69.10          | 68.24       | 68.68         | 9642   |

## Release
### base/1.0

- The first ML model trained with autoware-ml, carefully evaluated against the older version, demonstrating comparable performance. This is a major step towards lifelong MLOps for traffic light recognition models.
- Introduced the TLRv1.2 dataset, featuring vertical traffic lights in Japan.

<details>
<summary> The link of data and evaluation result </summary>

- model
  - Training dataset: tlr_v0_1 + tlr_v1_0_x2 + tlr_v1_0_xx1 + tlr_v1_2
  - Eval dataset: tlr_v1_2
  - [PR](https://github.com/tier4/autoware-ml/pull/143)
  - [Config file path](https://drive.google.com/drive/folders/17XBZ6AcycliejDvT7nSFRINzUyGmsb2X?usp=drive_link)
  - [Deployed onnx model](https://evaluation.tier4.jp/evaluation/mlpackages/e104265c-2945-4b8a-ae68-13accc1c0af2/releases/d5ce3e03-dd72-4517-b416-7a63a84c9fd3?project_id=zWhWRzei&tab=reports)
  - [Deployed label file](https://evaluation.tier4.jp/evaluation/mlpackages/e104265c-2945-4b8a-ae68-13accc1c0af2/releases/d5ce3e03-dd72-4517-b416-7a63a84c9fd3?project_id=zWhWRzei&tab=reports)
  - [Training results](https://drive.google.com/drive/folders/17XBZ6AcycliejDvT7nSFRINzUyGmsb2X?usp=drive_link)
  - train time: (A40 * 1) * 16 hours ( Used less than 10 GB memory)

- Results

```python
Class-wise Metrics:
----------------------------------------------------------------------------
| Class Name           | Precision  | Recall     | F1-Score   | Counts     |
----------------------------------------------------------------------------
| green                | 99.84      | 99.78      | 99.81      | 4986       |
| left,red             | 98.47      | 94.16      | 96.27      | 137        |
| left,red,straight    | 98.65      | 99.66      | 99.15      | 293        |
| red                  | 99.43      | 99.69      | 99.56      | 3845       |
| red,right            | 97.89      | 95.88      | 96.88      | 194        |
| red,straight         | 100.00     | 100.00     | 100.00     | 11         |
| unknown              | 70.97      | 62.86      | 66.67      | 35         |
| yellow               | 95.86      | 98.58      | 97.20      | 141        |
| red,up_left          | 0.00       | 0.00       | 0.00       | 0          |
| red,right,straight   | 0.00       | 0.00       | 0.00       | 0          |
| red,up_right         | 0.00       | 0.00       | 0.00       | 0          |
----------------------------------------------------------------------------
Overall results:  precision_top1: 69.19     , recall_top1: 68.24     , f1-score_top1: 68.68     , support_top1: 9642.00

```

</details>
