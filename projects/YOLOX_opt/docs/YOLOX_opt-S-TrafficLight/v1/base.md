# Deployed model for YOLOX_opt-S-TrafficLight base/1.X
## Summary

- Performance summary
  - Evaluation dataset: various combinations of TLR v1.0, 4.0, 5.0, 6.0, 7.0
  - mAP, AP for IOUs (50, 60, 70, 80, 90)

|          | Eval DB                      | mAP    | AP50   | AP60   | AP70   | AP80   | AP90   |
| -------- | ---------------------------- | ------ | ------ | ------ | ------ | ------ | ------ |
| base/1.2 | TLR v1.0, 4.0, 5.0, 6.0, 7.0 | 0.3432 | 0.4620 | 0.4520 | 0.4240 | 0.3110 | 0.0670 |
| base/1.1 | TLR v1.0, 4.0, 5.0, 6.0, 7.0 | 0.3446 | 0.4640 | 0.4580 | 0.4260 | 0.3140 | 0.0610 |
| base/1.2 | TLR v1.0, 4.0, 5.0, 6.0      | 0.3654 | 0.4860 | 0.4800 | 0.4560 | 0.3370 | 0.0680 |
| base/1.1 | TLR v1.0, 4.0, 5.0, 6.0      | 0.3683 | 0.4910 | 0.4870 | 0.4620 | 0.3330 | 0.0670 |
| base/1.2 | TLR v7.0                     | 0.5636 | 0.6450 | 0.6450 | 0.6170 | 0.5710 | 0.3400 |
| base/1.1 | TLR v7.0                     | 0.4460 | 0.5260 | 0.5100 | 0.4980 | 0.4520 | 0.2440 |

## Release

### base/1.2

- Adds DB TLR v7.0 dataset to the training data, in addition to the datasets used for training base/1.1.

<details>
<summary> The link of data and evaluation result </summary>
The performance on the new dataset (new location) improved significatly from (0.44->0.56) with slight insignificant performance drop in older datasets.

- model
  - Training dataset: DB TLR v1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0
  - Eval dataset: 
  - [Config file path](../../../configs/t4dataset/YOLOX_opt-S-TrafficLight/yolox_s_tlr_416x416_pedcar_t4dataset.py)
  - Deployed onnx model [[WebAuto (for internal)]](https://evaluation.tier4.jp/evaluation/mlpackages/ac288878-9790-44e3-9fc8-ca246c5cd235/releases/38aece2e-f472-4313-ba66-aaabda71a740?project_id=zWhWRzei)
  - Deployed onnx and labels [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/yolox-opt/yolox-opt-s-trafficlight/t4base/v1.2/deployment.zip)
  - Training and evaluation results [model-zoo]
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/yolox-opt/yolox-opt-s-trafficlight/t4base/v1.2/yolox_s_tlr_416x416_pedcar_t4dataset.py)
    - [checkpoint_last.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/yolox-opt/yolox-opt-s-trafficlight/t4base/v1.2/best_mAP_epoch_300.pth)
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/yolox-opt/yolox-opt-s-trafficlight/t4base/v1.2/logs.zip)
  - train time: (A100 * 1) * 1 days
- Results evaluated on 
  - [tlr_infos_test_crops.json](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/yolox-opt/yolox-opt-s-trafficlight/t4base/v1.2/tlr_infos_test_crops.json)

```python
***************bbox range = (0, inf)***************

---------------iou_thr: 0.5---------------

+--------------------------+-------+-------+--------+-------+
| class                    | gts   | dets  | recall | ap    |
+--------------------------+-------+-------+--------+-------+
| BACKGROUND               | 0     | 0     | 0.000  | 0.000 |
| traffic_light            | 19437 | 20065 | 0.971  | 0.915 |
| pedestrian_traffic_light | 2181  | 2233  | 0.809  | 0.669 |
+--------------------------+-------+-------+--------+-------+
| mAP                      |       |       |        | 0.792 |
+--------------------------+-------+-------+--------+-------+

---------------iou_thr: 0.6---------------

+--------------------------+-------+-------+--------+-------+
| class                    | gts   | dets  | recall | ap    |
+--------------------------+-------+-------+--------+-------+
| BACKGROUND               | 0     | 0     | 0.000  | 0.000 |
| traffic_light            | 19437 | 20065 | 0.962  | 0.898 |
| pedestrian_traffic_light | 2181  | 2233  | 0.796  | 0.645 |
+--------------------------+-------+-------+--------+-------+
| mAP                      |       |       |        | 0.772 |
+--------------------------+-------+-------+--------+-------+

---------------iou_thr: 0.7---------------

+--------------------------+-------+-------+--------+-------+
| class                    | gts   | dets  | recall | ap    |
+--------------------------+-------+-------+--------+-------+
| BACKGROUND               | 0     | 0     | 0.000  | 0.000 |
| traffic_light            | 19437 | 20065 | 0.931  | 0.841 |
| pedestrian_traffic_light | 2181  | 2233  | 0.756  | 0.579 |
+--------------------------+-------+-------+--------+-------+
| mAP                      |       |       |        | 0.710 |
+--------------------------+-------+-------+--------+-------+

---------------iou_thr: 0.8---------------

+--------------------------+-------+-------+--------+-------+
| class                    | gts   | dets  | recall | ap    |
+--------------------------+-------+-------+--------+-------+
| BACKGROUND               | 0     | 0     | 0.000  | 0.000 |
| traffic_light            | 19437 | 20065 | 0.802  | 0.625 |
| pedestrian_traffic_light | 2181  | 2233  | 0.591  | 0.362 |
+--------------------------+-------+-------+--------+-------+
| mAP                      |       |       |        | 0.493 |
+--------------------------+-------+-------+--------+-------+

---------------iou_thr: 0.9---------------

+--------------------------+-------+-------+--------+-------+
| class                    | gts   | dets  | recall | ap    |
+--------------------------+-------+-------+--------+-------+
| BACKGROUND               | 0     | 0     | 0.000  | 0.000 |
| traffic_light            | 19437 | 20065 | 0.351  | 0.122 |
| pedestrian_traffic_light | 2181  | 2233  | 0.208  | 0.046 |
+--------------------------+-------+-------+--------+-------+
| mAP                      |       |       |        | 0.084 |
+--------------------------+-------+-------+--------+-------+

***************bbox range = (0, 33)***************

---------------iou_thr: 0.5---------------

+--------------------------+------+-------+--------+-------+
| class                    | gts  | dets  | recall | ap    |
+--------------------------+------+-------+--------+-------+
| BACKGROUND               | 0    | 0     | 0.000  | 0.000 |
| traffic_light            | 9396 | 20065 | 0.958  | 0.431 |
| pedestrian_traffic_light | 997  | 2233  | 0.821  | 0.301 |
+--------------------------+------+-------+--------+-------+
| mAP                      |      |       |        | 0.366 |
+--------------------------+------+-------+--------+-------+

---------------iou_thr: 0.6---------------

+--------------------------+------+-------+--------+-------+
| class                    | gts  | dets  | recall | ap    |
+--------------------------+------+-------+--------+-------+
| BACKGROUND               | 0    | 0     | 0.000  | 0.000 |
| traffic_light            | 9396 | 20065 | 0.944  | 0.419 |
| pedestrian_traffic_light | 997  | 2233  | 0.802  | 0.287 |
+--------------------------+------+-------+--------+-------+
| mAP                      |      |       |        | 0.353 |
+--------------------------+------+-------+--------+-------+

---------------iou_thr: 0.7---------------

+--------------------------+------+-------+--------+-------+
| class                    | gts  | dets  | recall | ap    |
+--------------------------+------+-------+--------+-------+
| BACKGROUND               | 0    | 0     | 0.000  | 0.000 |
| traffic_light            | 9396 | 20065 | 0.901  | 0.380 |
| pedestrian_traffic_light | 997  | 2233  | 0.757  | 0.256 |
+--------------------------+------+-------+--------+-------+
| mAP                      |      |       |        | 0.318 |
+--------------------------+------+-------+--------+-------+

---------------iou_thr: 0.8---------------

+--------------------------+------+-------+--------+-------+
| class                    | gts  | dets  | recall | ap    |
+--------------------------+------+-------+--------+-------+
| BACKGROUND               | 0    | 0     | 0.000  | 0.000 |
| traffic_light            | 9396 | 20065 | 0.745  | 0.260 |
| pedestrian_traffic_light | 997  | 2233  | 0.582  | 0.151 |
+--------------------------+------+-------+--------+-------+
| mAP                      |      |       |        | 0.206 |
+--------------------------+------+-------+--------+-------+

---------------iou_thr: 0.9---------------

+--------------------------+------+-------+--------+-------+
| class                    | gts  | dets  | recall | ap    |
+--------------------------+------+-------+--------+-------+
| BACKGROUND               | 0    | 0     | 0.000  | 0.000 |
| traffic_light            | 9396 | 20065 | 0.279  | 0.036 |
| pedestrian_traffic_light | 997  | 2233  | 0.173  | 0.013 |
+--------------------------+------+-------+--------+-------+
| mAP                      |      |       |        | 0.025 |
+--------------------------+------+-------+--------+-------+

***************bbox range = (33, inf)***************

---------------iou_thr: 0.5---------------

+--------------------------+-------+-------+--------+-------+
| class                    | gts   | dets  | recall | ap    |
+--------------------------+-------+-------+--------+-------+
| BACKGROUND               | 0     | 0     | 0.000  | 0.000 |
| traffic_light            | 10041 | 20065 | 0.983  | 0.513 |
| pedestrian_traffic_light | 1184  | 2233  | 0.800  | 0.412 |
+--------------------------+-------+-------+--------+-------+
| mAP                      |       |       |        | 0.462 |
+--------------------------+-------+-------+--------+-------+

---------------iou_thr: 0.6---------------

+--------------------------+-------+-------+--------+-------+
| class                    | gts   | dets  | recall | ap    |
+--------------------------+-------+-------+--------+-------+
| BACKGROUND               | 0     | 0     | 0.000  | 0.000 |
| traffic_light            | 10041 | 20065 | 0.979  | 0.508 |
| pedestrian_traffic_light | 1184  | 2233  | 0.791  | 0.396 |
+--------------------------+-------+-------+--------+-------+
| mAP                      |       |       |        | 0.452 |
+--------------------------+-------+-------+--------+-------+

---------------iou_thr: 0.7---------------

+--------------------------+-------+-------+--------+-------+
| class                    | gts   | dets  | recall | ap    |
+--------------------------+-------+-------+--------+-------+
| BACKGROUND               | 0     | 0     | 0.000  | 0.000 |
| traffic_light            | 10041 | 20065 | 0.959  | 0.487 |
| pedestrian_traffic_light | 1184  | 2233  | 0.755  | 0.361 |
+--------------------------+-------+-------+--------+-------+
| mAP                      |       |       |        | 0.424 |
+--------------------------+-------+-------+--------+-------+

---------------iou_thr: 0.8---------------

+--------------------------+-------+-------+--------+-------+
| class                    | gts   | dets  | recall | ap    |
+--------------------------+-------+-------+--------+-------+
| BACKGROUND               | 0     | 0     | 0.000  | 0.000 |
| traffic_light            | 10041 | 20065 | 0.855  | 0.386 |
| pedestrian_traffic_light | 1184  | 2233  | 0.598  | 0.235 |
+--------------------------+-------+-------+--------+-------+
| mAP                      |       |       |        | 0.311 |
+--------------------------+-------+-------+--------+-------+

---------------iou_thr: 0.9---------------

+--------------------------+-------+-------+--------+-------+
| class                    | gts   | dets  | recall | ap    |
+--------------------------+-------+-------+--------+-------+
| BACKGROUND               | 0     | 0     | 0.000  | 0.000 |
| traffic_light            | 10041 | 20065 | 0.419  | 0.095 |
| pedestrian_traffic_light | 1184  | 2233  | 0.238  | 0.039 |
+--------------------------+-------+-------+--------+-------+
| mAP                      |       |       |        | 0.067 |
+--------------------------+-------+-------+--------+-------+
AP50: 0.4620  AP60: 0.4520  AP70: 0.4240  AP80: 0.3110  AP90: 0.0670  mAP: 0.3432 

```

</details>

### base/1.1

- The traffic light fine detector model trained including dataset([DB TLR v5.0](../../../../../autoware_ml/configs/t4dataset/db_tlr_v6.yaml) dataset), in addition to the dataset used for training base/1.0.
- Trained with data from Robotaxi and Robobus, enhancing robustness across variations in vehicle and sensor configurations.
- [DB TLR v6.0](../../../../../autoware_ml/configs/t4dataset/db_tlr_v6.yaml) is dataset of vertical traffic lights, which is different from all other dataset versions.
- The older models base/1.0, x2/1.0.0 performed poorly on db_tlr_v6, whearas the newly trained model has good classification performance across all datasets.

<details>
<summary> The link of data and evaluation result </summary>

- model
  - Training dataset: DB TLR v1.0, 2.0, 3.0, 4.0, 5.0, 6.0
  - Eval dataset: DB TLR v1.0, 4.0, 5.0, 6.0
  - [Config file path](../../../configs/t4dataset/yolox_s_tlr_416x416_pedcar_t4dataset.py)
  - Deployed onnx model [[WebAuto (for internal)]](https://evaluation.tier4.jp/evaluation/mlpackages/ac288878-9790-44e3-9fc8-ca246c5cd235/releases/ec6fd5f1-d1f0-466c-af5d-7a990e6dd5f4?project_id=zWhWRzei)
  - Deployed onnx and labels [model-zoo]
    - [tlr_car_ped_yolox_s_batch_6.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/yolox-opt/yolox-opt-s-trafficlight/t4base/v1.1/tlr_car_ped_yolox_s_batch_6.onnx)
    - [tlr_car_ped_yolox_s_batch_4.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/yolox-opt/yolox-opt-s-trafficlight/t4base/v1.1/tlr_car_ped_yolox_s_batch_4.onnx)
    - [tlr_car_ped_yolox_s_batch_1.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/yolox-opt/yolox-opt-s-trafficlight/t4base/v1.1/tlr_car_ped_yolox_s_batch_1.onnx)
    - [tlr_labels.txt](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/yolox-opt/yolox-opt-s-trafficlight/t4base/v1.1/tlr_labels.txt)
  - Training and evaluation results [model-zoo]
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/yolox-opt/yolox-opt-s-trafficlight/t4base/v1.1/yolox_s_tlr_416x416_pedcar_t4dataset.py)
    - [checkpoint_last.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/yolox-opt/yolox-opt-s-trafficlight/t4base/v1.1/best_mAP_epoch_300.pth)
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/yolox-opt/yolox-opt-s-trafficlight/t4base/v1.1/logs.zip)
  - train time: (NVIDIA A100-SXM4-80GB * 1) * 300 epochs = 31 hours

- Results evaluated on DB TLR v1.0, 4.0, 5.0
  - [tlr_infos_test_crops.json](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/yolox-opt/yolox-opt-s-trafficlight/t4base/v1.1/tlr_infos_test_cropsV1_4_5.json)

```python

***************bbox range = (0, inf)***************
---------------iou_thr: 0.5---------------
+--------------------------+-------+-------+--------+-------+
| class                    | gts   | dets  | recall | ap    |
+--------------------------+-------+-------+--------+-------+
| BACKGROUND               | 0     | 0     | 0.000  | 0.000 |
| traffic_light            | 18052 | 18499 | 0.965  | 0.910 |
| pedestrian_traffic_light | 2183  | 2248  | 0.820  | 0.676 |
+--------------------------+-------+-------+--------+-------+
| mAP                      |       |       |        | 0.793 |
+--------------------------+-------+-------+--------+-------+
---------------iou_thr: 0.6---------------
+--------------------------+-------+-------+--------+-------+
| class                    | gts   | dets  | recall | ap    |
+--------------------------+-------+-------+--------+-------+
| BACKGROUND               | 0     | 0     | 0.000  | 0.000 |
| traffic_light            | 18052 | 18499 | 0.956  | 0.894 |
| pedestrian_traffic_light | 2183  | 2248  | 0.808  | 0.659 |
+--------------------------+-------+-------+--------+-------+
| mAP                      |       |       |        | 0.776 |
+--------------------------+-------+-------+--------+-------+
---------------iou_thr: 0.7---------------
+--------------------------+-------+-------+--------+-------+
| class                    | gts   | dets  | recall | ap    |
+--------------------------+-------+-------+--------+-------+
| BACKGROUND               | 0     | 0     | 0.000  | 0.000 |
| traffic_light            | 18052 | 18499 | 0.927  | 0.840 |
| pedestrian_traffic_light | 2183  | 2248  | 0.761  | 0.594 |
+--------------------------+-------+-------+--------+-------+
| mAP                      |       |       |        | 0.717 |
+--------------------------+-------+-------+--------+-------+
---------------iou_thr: 0.8---------------
+--------------------------+-------+-------+--------+-------+
| class                    | gts   | dets  | recall | ap    |
+--------------------------+-------+-------+--------+-------+
| BACKGROUND               | 0     | 0     | 0.000  | 0.000 |
| traffic_light            | 18052 | 18499 | 0.798  | 0.623 |
| pedestrian_traffic_light | 2183  | 2248  | 0.597  | 0.373 |
+--------------------------+-------+-------+--------+-------+
| mAP                      |       |       |        | 0.498 |
+--------------------------+-------+-------+--------+-------+
---------------iou_thr: 0.9---------------
+--------------------------+-------+-------+--------+-------+
| class                    | gts   | dets  | recall | ap    |
+--------------------------+-------+-------+--------+-------+
| BACKGROUND               | 0     | 0     | 0.000  | 0.000 |
| traffic_light            | 18052 | 18499 | 0.320  | 0.101 |
| pedestrian_traffic_light | 2183  | 2248  | 0.198  | 0.043 |
+--------------------------+-------+-------+--------+-------+
| mAP                      |       |       |        | 0.072 |
+--------------------------+-------+-------+--------+-------+
AP50: 0.4790  AP60: 0.4730  AP70: 0.4480  AP80: 0.3250  AP90: 0.0560  mAP: 0.3562

```

- Results evaluated on DB TLR v6.0
  - [tlr_infos_test_crops.json](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/yolox-opt/yolox-opt-s-trafficlight/t4base/v1.1/tlr_infos_test_crops_v6.json)

```python
***************bbox range = (0, inf)***************
---------------iou_thr: 0.5---------------
+--------------------------+-----+------+--------+-------+
| class                    | gts | dets | recall | ap    |
+--------------------------+-----+------+--------+-------+
| BACKGROUND               | 0   | 0    | 0.000  | 0.000 |
| traffic_light            | 976 | 974  | 0.997  | 0.996 |
| pedestrian_traffic_light | 0   | 2    | 0.000  | 0.000 |
+--------------------------+-----+------+--------+-------+
| mAP                      |     |      |        | 0.996 |
+--------------------------+-----+------+--------+-------+
---------------iou_thr: 0.6---------------
+--------------------------+-----+------+--------+-------+
| class                    | gts | dets | recall | ap    |
+--------------------------+-----+------+--------+-------+
| BACKGROUND               | 0   | 0    | 0.000  | 0.000 |
| traffic_light            | 976 | 974  | 0.997  | 0.996 |
| pedestrian_traffic_light | 0   | 2    | 0.000  | 0.000 |
+--------------------------+-----+------+--------+-------+
| mAP                      |     |      |        | 0.996 |
+--------------------------+-----+------+--------+-------+
---------------iou_thr: 0.7---------------
+--------------------------+-----+------+--------+-------+
| class                    | gts | dets | recall | ap    |
+--------------------------+-----+------+--------+-------+
| BACKGROUND               | 0   | 0    | 0.000  | 0.000 |
| traffic_light            | 976 | 974  | 0.992  | 0.988 |
| pedestrian_traffic_light | 0   | 2    | 0.000  | 0.000 |
+--------------------------+-----+------+--------+-------+
| mAP                      |     |      |        | 0.988 |
+--------------------------+-----+------+--------+-------+
---------------iou_thr: 0.8---------------
+--------------------------+-----+------+--------+-------+
| class                    | gts | dets | recall | ap    |
+--------------------------+-----+------+--------+-------+
| BACKGROUND               | 0   | 0    | 0.000  | 0.000 |
| traffic_light            | 976 | 974  | 0.982  | 0.970 |
| pedestrian_traffic_light | 0   | 2    | 0.000  | 0.000 |
+--------------------------+-----+------+--------+-------+
| mAP                      |     |      |        | 0.970 |
+--------------------------+-----+------+--------+-------+
---------------iou_thr: 0.9---------------
+--------------------------+-----+------+--------+-------+
| class                    | gts | dets | recall | ap    |
+--------------------------+-----+------+--------+-------+
| BACKGROUND               | 0   | 0    | 0.000  | 0.000 |
| traffic_light            | 976 | 974  | 0.791  | 0.638 |
| pedestrian_traffic_light | 0   | 2    | 0.000  | 0.000 |
+--------------------------+-----+------+--------+-------+
| mAP                      |     |      |        | 0.638 |
+--------------------------+-----+------+--------+-------+
AP50: 0.6990  AP60: 0.6990  AP70: 0.6830  AP80: 0.6830  AP90: 0.4830  mAP: 0.6492
```

</details>

### base/1.0

- The first ML model trained with autoware-ml, carefully evaluated against the older version, demonstrating comparable performance. This is a major step towards lifelong MLOps for traffic light recognition models.
- Introduced the TLRv1.2 dataset, featuring vertical traffic lights in Japan.

<details>
<summary> The link of data and evaluation result </summary>

- model name: yolox_s_tlr_416x416_pedcar_t4dataset
- model
  - Training dataset: tlr_v0_1 + tlr_v1_0_x2 + tlr_v1_0_xx1 + tlr_v1_2
  - Eval dataset: tlr_v1_2
  - [Config file path](../../../configs/t4dataset/YOLOX_opt-S-TrafficLight/yolox_s_tlr_416x416_pedcar_t4dataset.py)
  - Deployed onnx model [[WebAuto (for internal)]](https://evaluation.tier4.jp/evaluation/mlpackages/ac288878-9790-44e3-9fc8-ca246c5cd235/releases/e23071aa-1cf9-4837-b71b-2fbbf990748d?project_id=zWhWRzei&tab=items)
  - Deployed label file [[WebAuto (for internal)]](https://evaluation.tier4.jp/evaluation/mlpackages/ac288878-9790-44e3-9fc8-ca246c5cd235/releases/e23071aa-1cf9-4837-b71b-2fbbf990748d?project_id=zWhWRzei&tab=items)
  - Deployed onnx and labels [model-zoo]
    - [tlr_car_ped_yolox_s_batch_6.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/yolox-opt/yolox-opt-s-trafficlight/t4base/v1.0/tlr_car_ped_yolox_s_batch_6.onnx)
    - [tlr_labels.txt](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/yolox-opt/yolox-opt-s-trafficlight/t4base/v1.0/tlr_labels.txt)
  - Training results [[Google drive (for internal)]](https://drive.google.com/drive/folders/1MH5yQT_dqVdk14WRxOQ4DE01TiMH-oIF)
  - Training results [model-zoo]
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/yolox-opt/yolox-opt-s-trafficlight/t4base/v1.0/yolox_s_tlr_416x416_pedcar_t4dataset.py)
    - [checkpoint_last.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/yolox-opt/yolox-opt-s-trafficlight/t4base/v1.0/epoch_300.pth)
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/yolox-opt/yolox-opt-s-trafficlight/t4base/v1.0/logs.zip)
  - train time: (A100 * 1) * 1 days
- Total mAP: 0.3588
  - Test dataset: tlr_v0_1+tlr_v1_2
  - Bbox size range: (0,inf)

```python
---------------iou_thr: 0.5---------------

+--------------------------+-------+-------+--------+-------+
| class                    | gts   | dets  | recall | ap    |
+--------------------------+-------+-------+--------+-------+
| BACKGROUND               | 0     | 0     | 0.000  | 0.000 |
| traffic_light            | 17660 | 18106 | 0.967  | 0.916 |
| pedestrian_traffic_light | 2013  | 2064  | 0.811  | 0.666 |
+--------------------------+-------+-------+--------+-------+
| mAP                      |       |       |        | 0.791 |
+--------------------------+-------+-------+--------+-------+

---------------iou_thr: 0.6---------------

+--------------------------+-------+-------+--------+-------+
| class                    | gts   | dets  | recall | ap    |
+--------------------------+-------+-------+--------+-------+
| BACKGROUND               | 0     | 0     | 0.000  | 0.000 |
| traffic_light            | 17660 | 18106 | 0.959  | 0.901 |
| pedestrian_traffic_light | 2013  | 2064  | 0.800  | 0.649 |
+--------------------------+-------+-------+--------+-------+
| mAP                      |       |       |        | 0.775 |
+--------------------------+-------+-------+--------+-------+

---------------iou_thr: 0.7---------------

+--------------------------+-------+-------+--------+-------+
| class                    | gts   | dets  | recall | ap    |
+--------------------------+-------+-------+--------+-------+
| BACKGROUND               | 0     | 0     | 0.000  | 0.000 |
| traffic_light            | 17660 | 18106 | 0.930  | 0.847 |
| pedestrian_traffic_light | 2013  | 2064  | 0.761  | 0.587 |
+--------------------------+-------+-------+--------+-------+
| mAP                      |       |       |        | 0.717 |
+--------------------------+-------+-------+--------+-------+

---------------iou_thr: 0.8---------------

+--------------------------+-------+-------+--------+-------+
| class                    | gts   | dets  | recall | ap    |
+--------------------------+-------+-------+--------+-------+
| BACKGROUND               | 0     | 0     | 0.000  | 0.000 |
| traffic_light            | 17660 | 18106 | 0.799  | 0.625 |
| pedestrian_traffic_light | 2013  | 2064  | 0.607  | 0.382 |
+--------------------------+-------+-------+--------+-------+
| mAP                      |       |       |        | 0.504 |
+--------------------------+-------+-------+--------+-------+

---------------iou_thr: 0.9---------------

+--------------------------+-------+-------+--------+-------+
| class                    | gts   | dets  | recall | ap    |
+--------------------------+-------+-------+--------+-------+
| BACKGROUND               | 0     | 0     | 0.000  | 0.000 |
| traffic_light            | 17660 | 18106 | 0.323  | 0.103 |
| pedestrian_traffic_light | 2013  | 2064  | 0.234  | 0.064 |
+--------------------------+-------+-------+--------+-------+
| mAP                      |       |       |        | 0.083 |
+--------------------------+-------+-------+--------+-------+

AP50: 0.4810  AP60: 0.4760  AP70: 0.4520  AP80: 0.3250  AP90: 0.0600  mAP: 0.3588
```

- Total mAP: 0.4389
  - Test dataset: tlr_v1_2
  - Bbox size range: (0,inf)

```python
---------------iou_thr: 0.5---------------
+--------------------------+------+------+--------+-------+
| class                    | gts  | dets | recall | ap    |
+--------------------------+------+------+--------+-------+
| BACKGROUND               | 0    | 0    | 0.000  | 0.000 |
| traffic_light            | 804  | 812  | 0.981  | 0.970 |
| pedestrian_traffic_light | 1095 | 869  | 0.770  | 0.748 |
+--------------------------+------+------+--------+-------+
| mAP                      |      |      |        | 0.859 |
+--------------------------+------+------+--------+-------+
---------------iou_thr: 0.6---------------
+--------------------------+------+------+--------+-------+
| class                    | gts  | dets | recall | ap    |
+--------------------------+------+------+--------+-------+
| BACKGROUND               | 0    | 0    | 0.000  | 0.000 |
| traffic_light            | 804  | 812  | 0.969  | 0.947 |
| pedestrian_traffic_light | 1095 | 869  | 0.763  | 0.734 |
+--------------------------+------+------+--------+-------+
| mAP                      |      |      |        | 0.841 |
+--------------------------+------+------+--------+-------+
---------------iou_thr: 0.7---------------
+--------------------------+------+------+--------+-------+
| class                    | gts  | dets | recall | ap    |
+--------------------------+------+------+--------+-------+
| BACKGROUND               | 0    | 0    | 0.000  | 0.000 |
| traffic_light            | 804  | 812  | 0.928  | 0.880 |
| pedestrian_traffic_light | 1095 | 869  | 0.742  | 0.696 |
+--------------------------+------+------+--------+-------+
| mAP                      |      |      |        | 0.788 |
+--------------------------+------+------+--------+-------+
---------------iou_thr: 0.8---------------
+--------------------------+------+------+--------+-------+
| class                    | gts  | dets | recall | ap    |
+--------------------------+------+------+--------+-------+
| BACKGROUND               | 0    | 0    | 0.000  | 0.000 |
| traffic_light            | 804  | 812  | 0.823  | 0.714 |
| pedestrian_traffic_light | 1095 | 869  | 0.598  | 0.455 |
+--------------------------+------+------+--------+-------+
| mAP                      |      |      |        | 0.585 |
+--------------------------+------+------+--------+-------+
---------------iou_thr: 0.9---------------
+--------------------------+------+------+--------+-------+
| class                    | gts  | dets | recall | ap    |
+--------------------------+------+------+--------+-------+
| BACKGROUND               | 0    | 0    | 0.000  | 0.000 |
| traffic_light            | 804  | 812  | 0.323  | 0.128 |
| pedestrian_traffic_light | 1095 | 869  | 0.237  | 0.077 |
+--------------------------+------+------+--------+-------+
| mAP                      |      |      |        | 0.102 |
+--------------------------+------+------+--------+-------+

AP50: 0.5920  AP60: 0.5820  AP70: 0.5460  AP80: 0.4020  AP90: 0.0720  mAP: 0.4389
```
- Total mAP: 0.3524
  - Test dataset: tlr_v0_1
  - Bbox size range: (0,inf)

```python
---------------iou_thr: 0.5---------------

+--------------------------+-------+-------+--------+-------+
| class                    | gts   | dets  | recall | ap    |
+--------------------------+-------+-------+--------+-------+
| BACKGROUND               | 0     | 0     | 0.000  | 0.000 |
| traffic_light            | 16825 | 17339 | 0.968  | 0.911 |
| pedestrian_traffic_light | 926   | 1195  | 0.861  | 0.599 |
+--------------------------+-------+-------+--------+-------+
| mAP                      |       |       |        | 0.755 |
+--------------------------+-------+-------+--------+-------+

---------------iou_thr: 0.6---------------

+--------------------------+-------+-------+--------+-------+
| class                    | gts   | dets  | recall | ap    |
+--------------------------+-------+-------+--------+-------+
| BACKGROUND               | 0     | 0     | 0.000  | 0.000 |
| traffic_light            | 16825 | 17339 | 0.960  | 0.896 |
| pedestrian_traffic_light | 926   | 1195  | 0.839  | 0.568 |
+--------------------------+-------+-------+--------+-------+
| mAP                      |       |       |        | 0.732 |
+--------------------------+-------+-------+--------+-------+

---------------iou_thr: 0.7---------------

+--------------------------+-------+-------+--------+-------+
| class                    | gts   | dets  | recall | ap    |
+--------------------------+-------+-------+--------+-------+
| BACKGROUND               | 0     | 0     | 0.000  | 0.000 |
| traffic_light            | 16825 | 17339 | 0.933  | 0.847 |
| pedestrian_traffic_light | 926   | 1195  | 0.776  | 0.494 |
+--------------------------+-------+-------+--------+-------+
| mAP                      |       |       |        | 0.670 |
+--------------------------+-------+-------+--------+-------+

---------------iou_thr: 0.8---------------

+--------------------------+-------+-------+--------+-------+
| class                    | gts   | dets  | recall | ap    |
+--------------------------+-------+-------+--------+-------+
| BACKGROUND               | 0     | 0     | 0.000  | 0.000 |
| traffic_light            | 16825 | 17339 | 0.797  | 0.618 |
| pedestrian_traffic_light | 926   | 1195  | 0.605  | 0.319 |
+--------------------------+-------+-------+--------+-------+
| mAP                      |       |       |        | 0.468 |
+--------------------------+-------+-------+--------+-------+

---------------iou_thr: 0.9---------------

+--------------------------+-------+-------+--------+-------+
| class                    | gts   | dets  | recall | ap    |
+--------------------------+-------+-------+--------+-------+
| BACKGROUND               | 0     | 0     | 0.000  | 0.000 |
| traffic_light            | 16825 | 17339 | 0.331  | 0.106 |
| pedestrian_traffic_light | 926   | 1195  | 0.216  | 0.048 |
+--------------------------+-------+-------+--------+-------+
| mAP                      |       |       |        | 0.077 |
+--------------------------+-------+-------+--------+-------+

AP50: 0.4710  AP60: 0.4580  AP70: 0.4330  AP80: 0.3320  AP90: 0.0670  mAP: 0.3524
```

</details>
