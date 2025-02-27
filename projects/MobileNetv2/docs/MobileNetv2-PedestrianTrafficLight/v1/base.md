# Deployed model for MobileNetv2-PedestrianTrafficLight base/1.X
## Summary

- Performance summary
  - Precision, Recall, F1 score, Counts for each class
  - Note
    - Eval DB: Evaluation dataset


|          | Eval DB           | precision_top1 | recall_top1 | f1-score_top1 | counts |
| -------- | ----------------- | -------------- | ----------- | ------------- | ------ |
| base/1.0 | tlr_v0_1+tlr_v1_2 | 69.29          | 72.64       | 68.94         | 1555   |

## Release
### base/1.0

- The first ML model trained with autoware-ml, carefully evaluated against the older version, demonstrating comparable performance. This is a major step towards lifelong MLOps for traffic light recognition models.
- Introduced the TLRv1.2 dataset, featuring vertical traffic lights in Japan.

<details>
<summary> The link of data and evaluation result </summary>

- model
  - Training dataset: tlr_v0_1 + tlr_v1_0_x2 + tlr_v1_0_xx1 + tlr_v1_2
  - Eval dataset: tlr_v1_2
  - [Config file path](../../../configs/t4dataset/MobileNetv2-PedestrianTrafficLight/mobilenet-v2_tlr_ped_t4dataset.py)
  - Deployed onnx model [[Google drive (for internal)]](https://evaluation.tier4.jp/evaluation/mlpackages/e104265c-2945-4b8a-ae68-13accc1c0af2/releases/d5ce3e03-dd72-4517-b416-7a63a84c9fd3?project_id=zWhWRzei&tab=reports)
  - Deployed onnx model [model-zoo]
    - [ped_traffic_light_classifier_mobilenetv2_batch_1.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/mobilenet-v2/pedestrian_traffic_light/t4base/v1.0/ped_traffic_light_classifier_mobilenetv2_batch_1.onnx)
    - [ped_traffic_light_classifier_mobilenetv2_batch_4.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/mobilenet-v2/pedestrian_traffic_light/t4base/v1.0/ped_traffic_light_classifier_mobilenetv2_batch_4.onnx)
    - [ped_traffic_light_classifier_mobilenetv2_batch_6.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/mobilenet-v2/pedestrian_traffic_light/t4base/v1.0/ped_traffic_light_classifier_mobilenetv2_batch_6.onnx)
  - Deployed label file [[WebAuto (for internal)]](https://evaluation.tier4.jp/evaluation/mlpackages/e104265c-2945-4b8a-ae68-13accc1c0af2/releases/d5ce3e03-dd72-4517-b416-7a63a84c9fd3?project_id=zWhWRzei&tab=reports) [[model-zoo]](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/mobilenet-v2/pedestrian_traffic_light/t4base/v1.0/lamp_labels_ped.txt)
  - Training results [[Google drive (for internal)]](https://drive.google.com/drive/folders/1PyyP0jvWpHA2TTBGKzElu4NMoF5HdJNT?usp=drive_link)
  - Training results [model-zoo]
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/mobilenet-v2/pedestrian_traffic_light/t4base/v1.0/mobilenet-v2_tlr_ped_t4dataset.py)
    - [checkpoint_best.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/mobilenet-v2/pedestrian_traffic_light/t4base/v1.0/best_multi-label_f1-score_top1_epoch_54.pth)
    - [checkpoint_last.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/mobilenet-v2/pedestrian_traffic_light/t4base/v1.0/epoch_300.pth)
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/mobilenet-v2/pedestrian_traffic_light/t4base/v1.0/logs.zip)
  - train time: (A40 * 1) * 5 hours ( Used less than 4 GB memory)

- Results
```python
Class-wise Metrics:
----------------------------------------------------------------------------
| Class Name           | Precision  | Recall     | F1-Score   | Counts     |
----------------------------------------------------------------------------
| crosswalk_red        | 95.22      | 95.12      | 95.17      | 942        |
| crosswalk_green      | 98.14      | 84.04      | 90.54      | 564        |
| crosswalk_unknown    | 14.50      | 38.78      | 21.11      | 49         |
----------------------------------------------------------------------------
Overall results:  precision_top1: 69.29     , recall_top1: 72.64     , f1-score_top1: 68.94     , support_top1: 1555.00

```

</details>
