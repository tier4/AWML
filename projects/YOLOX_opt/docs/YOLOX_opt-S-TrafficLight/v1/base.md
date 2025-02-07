# Deployed model for YOLOX_opt-S-TrafficLight base/1.X
## Summary

- Performance summary
  - Evaluation dataset: Eval dataset of tlr_v0_1 + tlr_v1_2
  - mAP, AP for IOUs (50, 60, 70, 80, 90)

|          | Eval DB            | mAP    | AP50   | AP60   | AP70   | AP80   | AP90   |
| -------- |-----------------   | ------ | ------ | ------ | ------ | ------ | ------ |
| base/1.1 | TLR v1.0, 4.0, 5.0 | 0.3562 | 0.4790 | 0.4730 | 0.4480 | 0.3250 | 0.0560 |
| base/1.1 | TLR v6.0           | 0.6492 | 0.6990 | 0.6990 | 0.6830 | 0.4830 | 0.0330 |
| base/1.0 | TLR v1.0, 4.0      | 0.3588 | 0.4810 | 0.4760 | 0.4520 | 0.3250 | 0.0600 |

## Release

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
  - [PR](https://github.com/tier4/autoware-ml/pull/375)
  - [Config file path](../../../configs/t4dataset/yolox_s_tlr_416x416_pedcar_t4dataset.py)
  - Deployed onnx model [[webauto]](https://evaluation.tier4.jp/evaluation/mlpackages/ac288878-9790-44e3-9fc8-ca246c5cd235/releases/ec6fd5f1-d1f0-466c-af5d-7a990e6dd5f4?project_id=zWhWRzei)
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
  - [PR](https://github.com/tier4/autoware-ml/pull/143)
  - [Config file path](../../../configs/t4dataset/yolox_s_tlr_416x416_pedcar_t4dataset.py)
  - Deployed onnx model [[webauto]](https://evaluation.tier4.jp/evaluation/mlpackages/ac288878-9790-44e3-9fc8-ca246c5cd235/releases/e23071aa-1cf9-4837-b71b-2fbbf990748d?project_id=zWhWRzei&tab=items)
  - Deployed label file [[webauto]](https://evaluation.tier4.jp/evaluation/mlpackages/ac288878-9790-44e3-9fc8-ca246c5cd235/releases/e23071aa-1cf9-4837-b71b-2fbbf990748d?project_id=zWhWRzei&tab=items)
  - Deployed onnx and labels [model-zoo]
    - [tlr_car_ped_yolox_s_batch_6.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/yolox-opt/yolox-opt-s-trafficlight/t4base/v1.0/tlr_car_ped_yolox_s_batch_6.onnx)
    - [tlr_labels.txt](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/yolox-opt/yolox-opt-s-trafficlight/t4base/v1.0/tlr_labels.txt)
  - Training results [[GDrive]](https://drive.google.com/drive/folders/1MH5yQT_dqVdk14WRxOQ4DE01TiMH-oIF)
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
