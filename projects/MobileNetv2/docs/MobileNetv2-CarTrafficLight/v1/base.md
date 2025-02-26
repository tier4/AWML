# Deployed model for MobileNetv2-CarTrafficLight base/1.X
## Summary

- Performance summary
  - Precision, Recall, F1 score, Counts for each class
  - Note
    - Eval DB: Evaluation dataset

|          | Eval DB            | precision_top1 | recall_top1 | f1-score_top1 | counts |
| -------- | ------------------ | -------------- | ----------- | ------------- | ------ |
| base/1.2 | TLR v1.0, 4.0, 5.0 | **69.75**      | **68.09**   | **68.85**     | 9648   |
| base/1.1 | TLR v1.0, 4.0, 5.0 | **70.64**      | **68.28**   | **69.29**     | 9648   |
| base/1.2 | TLR v6.0           | **21.19**      | **24.17**   | **22.18**     | 957    |
| base/1.1 | TLR v6.0           | **18.21**      | **17.56**   | **12.99**     | 957    |
| base/1.1 | TLR v1.0 + 4.0     | **70.48**      | **68.67**   | **69.49**     | 9642   |
| base/1.0 | TLR v1.0 + 4.0     | 69.10          | 68.24       | 68.68         | 9642   |
| base/1.1 | TLR v5.0           | **48.69**      | **49.91**   | **47.75**     | 306    |
| base/1.0 | TLR v5.0           | 46.35          | 49.08       | 46.39         | 306    |

## Release

### base/1.2
- The car traffic light classifier model trained with X2 gen1.0 BRT dataset([DB TLR v6.0](../../../../../autoware_ml/configs/t4dataset/db_tlr_v6.yaml) dataset), in addition to the dataset used for training base/1.1.
- [DB TLR v6.0](../../../../../autoware_ml/configs/t4dataset/db_tlr_v6.yaml) is dataset of vertical traffic lights, which is different from all other dataset versions.
- The older models base/1.1 performed poorly on db_tlr_v6, whearas the newly trained model has good classification performance across all datasets.

<details>
<summary> The link of data and evaluation result </summary>

- model
  - Training dataset: DB TLR v1.0, 2.0, 3.0, 4.0, 5.0, 6.0
  - Eval dataset: DB TLR v1.0, 4.0, 5.0, 6.0
  - [Config file path](../../../configs/t4dataset/mobilenet-v2_tlr_car_t4dataset.py)
  - Deployed onnx model [[WebAuto (for internal)]](https://evaluation.tier4.jp/evaluation/mlpackages/e104265c-2945-4b8a-ae68-13accc1c0af2/releases/40d09664-0a7b-406e-9578-f65a0707bee3?project_id=zWhWRzei)
  - Deployed onnx model [model-zoo]
    - [traffic_light_classifier_mobilenetv2_batch_1.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/mobilenet-v2/car_traffic_light/t4base/v1.2/traffic_light_classifier_mobilenetv2_batch_1.onnx)
    - [traffic_light_classifier_mobilenetv2_batch_4.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/mobilenet-v2/car_traffic_light/t4base/v1.2/traffic_light_classifier_mobilenetv2_batch_4.onnx)
    - [traffic_light_classifier_mobilenetv2_batch_6.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/mobilenet-v2/car_traffic_light/t4base/v1.2/traffic_light_classifier_mobilenetv2_batch_6.onnx)

  - Deployed [label file](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/mobilenet-v2/car_traffic_light/t4base/v1.2/lamp_labels.txt)
  - Training results [model-zoo]
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/mobilenet-v2/car_traffic_light/t4base/v1.2/mobilenet-v2_tlr_car_t4dataset.py)
    - [checkpoint_best.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/mobilenet-v2/car_traffic_light/t4base/v1.2/best_multi-label_f1-score_top1_epoch_110.pth)
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/mobilenet-v2/car_traffic_light/t4base/v1.2/logs.zip)

  - train time: (NVIDIA A100-SXM4-80GB * 1) * 300 epochs = 22 hours

- Results evaluated with DB TLR v6.0

```python
Class-wise Metrics:
----------------------------------------------------------------------------
| Class Name           | Precision  | Recall     | F1-Score   | Counts     |
----------------------------------------------------------------------------
| green                | 99.77      | 100.00     | 99.89      | 441        |
| left,red             | 0.00       | 0.00       | 0.00       | 0          |
| left,red,straight    | 0.00       | 0.00       | 0.00       | 0          |
| red                  | 100.00     | 99.22      | 99.61      | 513        |
| red,right            | 0.00       | 0.00       | 0.00       | 0          |
| red,straight         | 0.00       | 0.00       | 0.00       | 0          |
| unknown              | 33.33      | 66.67      | 44.44      | 3          |
| yellow               | 0.00       | 0.00       | 0.00       | 0          |
| red,up_left          | 0.00       | 0.00       | 0.00       | 0          |
| red,right,straight   | 0.00       | 0.00       | 0.00       | 0          |
| red,up_right         | 0.00       | 0.00       | 0.00       | 0          |
----------------------------------------------------------------------------
```

- Results evaluated with DB TLR v1.0, 4.0, 5.0

```python
Class-wise Metrics:
----------------------------------------------------------------------------
| Class Name           | Precision  | Recall     | F1-Score   | Counts     |
----------------------------------------------------------------------------
| green                | 99.90      | 99.88      | 99.89      | 5089       |
| left,red             | 100.00     | 89.78      | 94.62      | 137        |
| left,red,straight    | 99.31      | 98.30      | 98.80      | 294        |
| red                  | 99.31      | 99.72      | 99.51      | 3907       |
| red,right            | 96.64      | 98.63      | 97.63      | 292        |
| red,straight         | 100.00     | 100.00     | 100.00     | 11         |
| unknown              | 75.00      | 63.83      | 68.97      | 47         |
| yellow               | 97.13      | 98.83      | 97.97      | 171        |
| red,up_left          | 0.00       | 0.00       | 0.00       | 0          |
| red,right,straight   | 0.00       | 0.00       | 0.00       | 0          |
| red,up_right         | 0.00       | 0.00       | 0.00       | 0          |
----------------------------------------------------------------------------
```

</details>

### base/1.1

- The car traffic light classifier model trained with X2 gen2.0 Odaiba dataset([DB TLR v5.0](../../../../../autoware_ml/configs/t4dataset/db_tlr_v5.yaml) dataset), in addition to the dataset used for training base/1.0.
- In quantitative analysis, the performance of base/1.1 is better than base/1.0.
  - Evaluation data for Robobus(X2 gen2.0, [DB TLR v5.0](../../../../../autoware_ml/configs/t4dataset/db_tlr_v5.yaml) dataset) highlights notable improvements, with overall F1 score increasing by approximately 2.9%.
  - Evaluation data for Robotaxi(XX1, [DB TLR v1](../../../../../autoware_ml/configs/t4dataset/db_tlr_v1.yaml) and [v4](../../../../../autoware_ml/configs/t4dataset/db_tlr_v4.yaml) dataset) highlights improvements, with F1 score increasing by approximately 1.1%.

<details>
<summary> The link of data and evaluation result </summary>

- model
  - Training dataset: DB TLR v1.0, 2.0, 3.0, 4.0, 5.0
  - Eval dataset: DB TLR v1.0, 4.0, 5.0
  - [Config file path](../../../configs/t4dataset/MobileNetv2-CarTrafficLight/mobilenet-v2_tlr_car_t4dataset.py)
  - Deployed onnx model [[WebAuto (for internal)]](https://evaluation.tier4.jp/evaluation/mlpackages/e104265c-2945-4b8a-ae68-13accc1c0af2/releases/84e9b9b6-6b3b-4a60-8cfe-d410b2af6ba4?project_id=zWhWRzei)
  - Deployed onnx model [model-zoo]
    - [traffic_light_classifier_mobilenetv2_batch_1.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/mobilenet-v2/car_traffic_light/t4base/v1.1/traffic_light_classifier_mobilenetv2_batch_1.onnx)
    - [traffic_light_classifier_mobilenetv2_batch_4.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/mobilenet-v2/car_traffic_light/t4base/v1.1/traffic_light_classifier_mobilenetv2_batch_4.onnx)
    - [traffic_light_classifier_mobilenetv2_batch_6.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/mobilenet-v2/car_traffic_light/t4base/v1.1/traffic_light_classifier_mobilenetv2_batch_6.onnx)

  - Deployed label file: It is the same as [base/1.0](#base10)
  - Training results [[Google drive (for internal)]](https://drive.google.com/drive/folders/1ozAAvqQOJKenUx8LE454-Cu83mgYmDVG?usp=drive_link)
  - Training results [model-zoo]
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/mobilenet-v2/car_traffic_light/t4base/v1.1/mobilenet-v2_tlr_car_t4dataset.py)
    - [checkpoint_best.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/mobilenet-v2/car_traffic_light/t4base/v1.1/best_multi-label_f1-score_top1_epoch_224.pth)
    - [checkpoint_last.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/mobilenet-v2/car_traffic_light/t4base/v1.1/epoch_300.pth)
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/mobilenet-v2/car_traffic_light/t4base/v1.1/logs.zip)

  - train time: (NVIDIA A100-SXM4-80GB * 1) * 300 epochs = 22 hours

- Results evaluated with DB TLR v1.0, 4.0, 5.0
  - Evaluation results [[Google drive (for internal)]](https://drive.google.com/drive/folders/1K8LJ8sCWfK0tLcYVvy0-dhXWfkegp1mj?usp=drive_link) [Also in ↑logs.zip file in model zoo training results]

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
  - [Evaluation results](https://drive.google.com/drive/folders/10XkxjAPBDShO_NBRGpVJbXxv0QRYa9mo?usp=drive_link) [Also in ↑logs.zip file in model zoo training results]

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

- Results evaluated with DB TLR v5.0
  - [Evaluation results](https://drive.google.com/drive/folders/1BdPGEn4PuD3TT4LVjrQHloDhhCBpg11q)  [Also in ↑logs.zip file in model zoo training results]

```python
Class-wise Metrics:
----------------------------------------------------------------------------
| Class Name           | Precision  | Recall     | F1-Score   | Counts     |
----------------------------------------------------------------------------
| green                | 100.00     | 100.00     | 100.00     | 103        |
| left,red             | 0.00       | 0.00       | 0.00       | 0          |
| left,red,straight    | 50.00      | 100.00     | 66.67      | 1          |
| red                  | 98.41      | 100.00     | 99.20      | 62         |
| red,right            | 98.98      | 98.98      | 98.98      | 98         |
| red,straight         | 0.00       | 0.00       | 0.00       | 0          |
| unknown              | 100.00     | 50.00      | 66.67      | 12         |
| yellow               | 88.24      | 100.00     | 93.75      | 30         |
| red,up_left          | 0.00       | 0.00       | 0.00       | 0          |
| red,right,straight   | 0.00       | 0.00       | 0.00       | 0          |
| red,up_right         | 0.00       | 0.00       | 0.00       | 0          |
----------------------------------------------------------------------------
Overall results:  precision_top1: 48.69     , recall_top1: 49.91     , f1-score_top1: 47.75     , support_top1: 306.00  
```

</details>

<details>
<summary> The link to baseline(base/1.0) evaluation result </summary>

- Results evaluated with DB TLR v5.0
  - [Baseline evaluation results by base/1.0](https://drive.google.com/drive/folders/1np7lq0OqsZ3szBKE7b6Z9GAXe7eGmAuE) [Also in ↑logs.zip file in model zoo training results]

```python
Class-wise Metrics:
----------------------------------------------------------------------------
| Class Name           | Precision  | Recall     | F1-Score   | Counts     |
----------------------------------------------------------------------------
| green                | 100.00     | 100.00     | 100.00     | 103        |
| left,red             | 0.00       | 0.00       | 0.00       | 0          |
| left,red,straight    | 50.00      | 100.00     | 66.67      | 1          |
| red                  | 98.28      | 91.94      | 95.00      | 62         |
| red,right            | 98.97      | 97.96      | 98.46      | 98         |
| red,straight         | 0.00       | 0.00       | 0.00       | 0          |
| unknown              | 85.71      | 50.00      | 63.16      | 12         |
| yellow               | 76.92      | 100.00     | 86.96      | 30         |
| red,up_left          | 0.00       | 0.00       | 0.00       | 0          |
| red,right,straight   | 0.00       | 0.00       | 0.00       | 0          |
| red,up_right         | 0.00       | 0.00       | 0.00       | 0          |
----------------------------------------------------------------------------
Overall results:  precision_top1: 46.35     , recall_top1: 49.08     , f1-score_top1: 46.39     , support_top1: 306.00  
```

</details>

### base/1.0

- The first ML model trained with autoware-ml, carefully evaluated against the older version, demonstrating comparable performance. This is a major step towards lifelong MLOps for traffic light recognition models.
- Introduced the TLR v4.0 dataset, featuring vertical traffic lights in Japan.

<details>
<summary> The link of data and evaluation result </summary>

- model
  - Training dataset: TLR v1.0 + TLR v2.0 + TLR v3.0 + TLR v4.0
  - Eval dataset: TLR v4.0
  - [Config file path](../../../configs/t4dataset/MobileNetv2-CarTrafficLight/mobilenet-v2_tlr_car_t4dataset.py)
  - Deployed onnx model [[WebAuto (for internal)]](https://evaluation.tier4.jp/evaluation/mlpackages/e104265c-2945-4b8a-ae68-13accc1c0af2/releases/d5ce3e03-dd72-4517-b416-7a63a84c9fd3?project_id=zWhWRzei&tab=reports)
  - Deployed onnx model [model-zoo]
    - [traffic_light_classifier_mobilenetv2_batch_1.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/mobilenet-v2/car_traffic_light/t4base/v1.0/traffic_light_classifier_mobilenetv2_batch_1.onnx)
    - [traffic_light_classifier_mobilenetv2_batch_4.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/mobilenet-v2/car_traffic_light/t4base/v1.0/traffic_light_classifier_mobilenetv2_batch_4.onnx)
    - [traffic_light_classifier_mobilenetv2_batch_6.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/mobilenet-v2/car_traffic_light/t4base/v1.0/traffic_light_classifier_mobilenetv2_batch_6.onnx)
  - Deployed label file [[Google drive (for internal)]](https://evaluation.tier4.jp/evaluation/mlpackages/e104265c-2945-4b8a-ae68-13accc1c0af2/releases/d5ce3e03-dd72-4517-b416-7a63a84c9fd3?project_id=zWhWRzei&tab=reports) [[model-zoo]](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/mobilenet-v2/car_traffic_light/t4base/v1.0/lamp_labels.txt)
  - Training results [[Google drive (for internal)]](https://drive.google.com/drive/folders/17XBZ6AcycliejDvT7nSFRINzUyGmsb2X?usp=drive_link)
  - Training results [model-zoo]
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/mobilenet-v2/car_traffic_light/t4base/v1.0/mobilenet-v2_tlr_car_t4dataset.py)
    - [checkpoint_best.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/mobilenet-v2/car_traffic_light/t4base/v1.0/best_multi-label_f1-score_top1_epoch_12.pth)
    - [checkpoint_last.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/mobilenet-v2/car_traffic_light/t4base/v1.0/epoch_300.pth)
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/mobilenet-v2/car_traffic_light/t4base/v1.0/logs.zip)
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
