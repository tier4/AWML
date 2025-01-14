# Deployed model for MobileNetv2-CarTrafficLight base/1.X
## Summary

- Performance summary
  - Precision, Recall, F1 score, Counts for each class
  - Note
    - Eval DB: Evaluation dataset

|          | Eval DB           | precision_top1 | recall_top1 | f1-score_top1 | counts |
| -------- | ----------------- | -------------- | ----------- | ------------- | ------ |
| base/1.0 | TLR v1.0 + 4.0    | 69.10          | 68.24       | 68.68         | 9642   |
| base/1.1 | TLR v1.0 + 4.0    | **70.48**      | **68.67**   | **69.49**     | 9642   |

## Release
### base/1.0

- The first ML model trained with autoware-ml, carefully evaluated against the older version, demonstrating comparable performance. This is a major step towards lifelong MLOps for traffic light recognition models.
- Introduced the TLR v4.0 dataset, featuring vertical traffic lights in Japan.

<details>
<summary> The link of data and evaluation result </summary>

- model
  - Training dataset: TLR v1.0 + TLR v2.0 + TLR v3.0 + TLR v4.0
  - Eval dataset: TLR v4.0
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

### base/1.1

- [DB TLR v5.0](../../../../../autoware_ml/configs/t4dataset/db_tlr_v5.yaml) is used for training.

<details>
<summary> The link of data and evaluation result </summary>

- model
  - Training dataset: DB TLR v1.0, 2.0, 3.0, 4.0, 5.0
  - Eval dataset: DB TLR v1.0, 4.0, 5.0
  - [PR](https://github.com/tier4/autoware-ml/pull/354)
  - [Config file path](https://drive.google.com/file/d/1ae1Rpj9xGGBBPmRp8nk9F8ftSOy9uC7J/view?usp=drive_link)
  - [Deployed onnx model](https://evaluation.tier4.jp/evaluation/mlpackages/e104265c-2945-4b8a-ae68-13accc1c0af2/releases/84e9b9b6-6b3b-4a60-8cfe-d410b2af6ba4?project_id=zWhWRzei)
  - Deployed label file: It is the same as [base/1.0](#base10) 
  - [Training results](https://drive.google.com/drive/folders/1ozAAvqQOJKenUx8LE454-Cu83mgYmDVG?usp=drive_link)
  - train time: (NVIDIA A100-SXM4-80GB * 1) * 300 epochs = 22 hours

- Results evaluated with DB TLR v1.0, 4.0, 5.0
  - [Evaluation results](https://drive.google.com/drive/folders/1K8LJ8sCWfK0tLcYVvy0-dhXWfkegp1mj?usp=drive_link)

```python
Class-wise Metrics:
----------------------------------------------------------------------------
| Class Name           | Precision  | Recall     | F1-Score   | Counts     |
----------------------------------------------------------------------------
| green                | 99.94      | 99.88      | 99.91      | 5089       |
| left,red             | 100.00     | 90.51      | 95.02      | 137        |
| left,red,straight    | 98.98      | 99.32      | 99.15      | 294        |
| red                  | 99.39      | 99.82      | 99.60      | 3907       |
| red,right            | 96.96      | 98.29      | 97.62      | 292        |
| red,straight         | 100.00     | 100.00     | 100.00     | 11         |
| unknown              | 85.71      | 63.83      | 73.17      | 47         |
| yellow               | 96.05      | 99.42      | 97.70      | 171        |
| red,up_left          | 0.00       | 0.00       | 0.00       | 0          |
| red,right,straight   | 0.00       | 0.00       | 0.00       | 0          |
| red,up_right         | 0.00       | 0.00       | 0.00       | 0          |
----------------------------------------------------------------------------
Overall results:  precision_top1: 70.64     , recall_top1: 68.28     , f1-score_top1: 69.29     , support_top1: 9948.00
```

- Results evaluated with DB TLR v1.0, 4.0
  - [Evaluation results](https://drive.google.com/drive/folders/10XkxjAPBDShO_NBRGpVJbXxv0QRYa9mo?usp=drive_link)

```python
Class-wise Metrics:
----------------------------------------------------------------------------
| Class Name           | Precision  | Recall     | F1-Score   | Counts     |
----------------------------------------------------------------------------
| green                | 99.94      | 99.88      | 99.91      | 4986       |
| left,red             | 100.00     | 90.51      | 95.02      | 137        |
| left,red,straight    | 99.32      | 99.32      | 99.32      | 293        |
| red                  | 99.40      | 99.82      | 99.61      | 3845       |
| red,right            | 95.96      | 97.94      | 96.94      | 194        |
| red,straight         | 100.00     | 100.00     | 100.00     | 11         |
| unknown              | 82.76      | 68.57      | 75.00      | 35         |
| yellow               | 97.90      | 99.29      | 98.59      | 141        |
| red,up_left          | 0.00       | 0.00       | 0.00       | 0          |
| red,right,straight   | 0.00       | 0.00       | 0.00       | 0          |
| red,up_right         | 0.00       | 0.00       | 0.00       | 0          |
----------------------------------------------------------------------------
Overall results:  precision_top1: 70.48     , recall_top1: 68.67     , f1-score_top1: 69.49     , support_top1: 9642.00   
```

</details>
