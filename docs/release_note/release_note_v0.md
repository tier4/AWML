## v0.7.1

- Add new backbone for 3D detection.

We introduced ConvNeXtPC as the new backbone in 3D detection.

Comparison: Base Detection

| Model                      | mAP         | Time    | Memory |
| -------------------------- | ----------- | ------- | ------ |
| base/1.0 (120m)            | 64.4        | 18.2 ms | 4.7 GB |
| ConvNeXtPC-small (120m)    | 65.3 (+0.9) | 30.4 ms | 5.6 GB |
| ConvNeXtPC-standard (120m) | 68.6 (+4.2) | 48.1 ms | 5.9 GB |

Comparison: Nearby Models

| Model                     | Time   | Memory |
| ------------------------- | ------ | ------ |
| base/1.0 (50m)            | 5.9 ms | 4.5 GB |
| ConvNeXtPC-small (50m)    | 7.2 ms | 3.9 GB |
| ConvNeXtPC-standard (50m) | 9.9 ms | 3.6 GB |

## v0.7.0

- Start to dataset versioning and change config architecture

We started dataset versioning.
We changed from `DB v1.0` to `DB JPNTAXI v1.0` and changed config files from `database_v1_3.yaml` to `db_jpntaxi_v3.yaml`.

In addition to this, we changed the architecture of configs for T4dataset.

```
- configs/
  - t4dataset/
    - db_gsm8_v1.yaml
    - db_j6_v1.yaml
    - db_j6_v2.yaml
    - db_jpntaxi_v1.yaml
    - db_jpntaxi_v2.yaml
    - db_jpntaxi_v3.yaml
    - db_tlr_v1.yaml
    - db_tlr_v2.yaml
    - db_tlr_v3.yaml
    - db_tlr_v4.yaml
    - pseudo_j6_v1.yaml
    - pseudo_j6_v2.yaml
  - classification2d/
    - dataset/
      - t4dataset/
        - tlr_finedetector.py
  - detection2d/
    - dataset/
      - t4dataset/
        - tlr_finedetector.py
  - detection3d/
    - dataset/
      - t4dataset/
        - pretrain.py
        - base.py
        - x2.py
        - xx1.py
```

- Integrated auto/semi-auto label pipeline and started to use pseudo T4dataset

We add the pipeline of auto labeling for 3D detection.
This tool can be used for auto labeling and semi-auto labeling in 3D detection.

- Release BEVFusion-CL-offline base/0.1

We release BEVFusion-CL-offline, which is offline model of BEVFusion for Camera-LiDAR fusion input.
Compared to `BEVFusion-L-offline base/0.2`, the improvement on mAP is marginal (72.2 vs 73.7).

|                               | mAP  | car <br> (610,396) | truck <br> (151,672) | bus <br> (37,876) | bicycle <br> (47,739) | pedestrian <br> (367,200) |
| ----------------------------- | ---- | ------------------ | -------------------- | ----------------- | --------------------- | ------------------------- |
| BEVFusion-CL-offline base/0.1 | 73.7 | 81.2               | 61.1                 | 84.5              | 70.2                  | 71.6                      |
| BEVFusion-L-offline base/0.2  | 72.2 | 81.2               | 60.5                 | 72.2              | 73.0                  | 73.9                      |

## v0.6.0

- Replace from nuscenes_devkit to t4_devkit

We replaced from nuscenes_devkit into t4_devkit in data conversion.

- Add new format for document of release model

In addition to model versioning, We introduced the format of release note.
We ask the developers to write not only evaluation result but also what you change and why you update the model like release note.

- Change formatter

We changed python formatter from yapf into black because many developers are used to black format.

## v0.5.0

- Fine tuning strategy with model versioning

We define the fine tuning strategy with model versioning and start to manage the version of model based on it.
Please see [the document](https://github.com/tier4/AWML/blob/main/docs/design/architecture_model.md) in detail.

- Add MobileNet v2 for traffic light recognition

| Class Name         | Precision | Recall | F1-Score | Counts |
| ------------------ | --------- | ------ | -------- | ------ |
| green              | 99.84     | 99.78  | 99.81    | 4986   |
| left,red           | 98.47     | 94.16  | 96.27    | 137    |
| left,red,straight  | 98.65     | 99.66  | 99.15    | 293    |
| red                | 99.43     | 99.69  | 99.56    | 3845   |
| red,right          | 97.89     | 95.88  | 96.88    | 194    |
| red,straight       | 100.00    | 100.00 | 100.00   | 11     |
| unknown            | 70.97     | 62.86  | 66.67    | 35     |
| yellow             | 95.86     | 98.58  | 97.20    | 141    |
| red,up_left        | 0.00      | 0.00   | 0.00     | 0      |
| red,right,straight | 0.00      | 0.00   | 0.00     | 0      |
| red,up_right       | 0.00      | 0.00   | 0.00     | 0      |

## v0.4.0

- Add CenterPoint for 3D detection

As main topics of this version, we release CenterPoint model.
We evaluate for T4dataset and compare to old library.
Old library is based on mmdetection3d v0.

| Eval range = 120m             | mAP  | car  | truck | bus  | bicycle | pedestrian |
| ----------------------------- | ---- | ---- | ----- | ---- | ------- | ---------- |
| Old library version (122.88m) | 62.2 | 74.7 | 43.0  | 75.0 | 54.0    | 64.1       |
| CenterPoint v1.0.0 (122.88m)  | 64.5 | 75.0 | 50.7  | 78.1 | 53.2    | 64.8       |

In the detail result from "CenterPoint v1.0.0" is following table.

| class_name | Count  | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---------- | ------ | ---- | ------- | ------- | ------- | ------- |
| car        | 41,133 | 75.0 | 64.7    | 76.8    | 79.1    | 79.5    |
| truck      | 8,890  | 50.7 | 27.8    | 50.5    | 59.6    | 65.1    |
| bus        | 3,275  | 78.1 | 69.2    | 79.6    | 81.1    | 82.6    |
| bicycle    | 3,635  | 53.2 | 52.3    | 53.4    | 53.5    | 53.6    |
| pedestrian | 25,981 | 64.8 | 62.4    | 64.0    | 65.4    | 67.4    |

## v0.3.0

- BEVFusion-CL

As main topics of this version, we release BEVFusion-CL model.
We evaluate BEVFusion-L (BEVFusion LiDAR-only model) and BEVFusion-CL (BEVFusion Camera-LiDAR fusion model) for T4dataset.
As mAP of NuScenes increases from 64.6pt to 66.1pt between BEVFusion-L and BEVFusion-CL, T4dataset increases from 60.6pt to 63.7pt

| model                             | range  | mAP  | car  | truck | bus  | bicycle | pedestrian |
| --------------------------------- | ------ | ---- | ---- | ----- | ---- | ------- | ---------- |
| BEVFusion-L t4xx1_120m_v1         | 122.4m | 60.6 | 74.1 | 54.1  | 58.7 | 55.7    | 60.3       |
| BEVFusion-CL t4xx1_fusion_120m_v1 | 122.4m | 63.7 | 72.4 | 56.8  | 71.8 | 62.1    | 55.3       |

- TransFusion-L

We add base model and X2 model.

| TransFusion-L 90m | train        | eval | All  | car  | truck | bus  | bicycle | pedestrian |
| ----------------- | ------------ | ---- | ---- | ---- | ----- | ---- | ------- | ---------- |
| t4base_90m/v1     | XX1 + X2     | XX1  | 67.4 | 80.7 | 56.0  | 77.6 | 57.4    | 65.5       |
| t4xx1_90m/v2      | XX1          | XX1  | 68.1 | 80.5 | 58.0  | 80.8 | 58.0    | 63.2       |
| t4xx1_90m/v3      | XX1 + XX1new | XX1  | 68.5 | 81.7 | 62.4  | 83.5 | 50.9    | 64.1       |
| t4base_90m/v1     | XX1 + X2     | X2   | 66.0 | 82.3 | 47.5  | 83.6 | 55.1    | 61.6       |
| t4x2_90m/v1       | X2           | X2   | 58.5 | 80.5 | 28.1  | 82.4 | 48.0    | 53.7       |

- YOLOX_opt

We add YOLOX_opt and deploy pipeline for fine detector model for traffic light recognition in Autoware.

- Add scene_selector

As first prototype, we integrate scene_selector to use active learning pipeline.

## V0.2.0

- Added visualization tools

As main topics of this version, we add [rerun](https://github.com/rerun-io/rerun) based visualization tools

- Released TransFusion-L t4xx1_90m/v3

We release TransFusion-L t4xx1_90m/v3.
This model improve the score of mAP.

| train              | eval         | mAP  |
| ------------------ | ------------ | ---- |
| DB 1.0 + 1.1       | DB 1.0 + 1.1 | 66.4 |
| DB 1.0 + 1.1 + 1.3 | DB 1.0 + 1.1 | 68.5 |
| DB 1.0 + 1.1       | DB 1.3       | 63.4 |
| DB 1.0 + 1.1 + 1.3 | DB 1.3       | 69.6 |

## v0.1.2

As main topics of this version, we released BEVFusion model.
We released 2 model, T4dataset 120m model and T4dataset offline model.

- Model comparison for class mAP with T4dataset
  - Eval range: 90m
  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m)

| model                         | range   | mAP  | car  | truck | bus  | bicycle | pedestrian |
| ----------------------------- | ------- | ---- | ---- | ----- | ---- | ------- | ---------- |
| TransFusion-L t4xx1_120m_v1   | 122.88m | 51.8 | 70.9 | 43.8  | 54.7 | 38.7    | 50.9       |
| BEVFusion-L t4xx1_120m_v1     | 122.4m  | 60.6 | 74.1 | 54.1  | 58.7 | 55.7    | 60.3       |
| BEVFusion-L t4offline_120m_v1 | 122.4m  | 65.7 | 76.2 | 56.4  | 65.6 | 65.0    | 65.3       |

- LiDAR backbone

Whole architecture between TransFusion-L and BEVFusion-L is same.
However, the kind of LiDAR backbone is different.
While our TransFusion-L use pillar for LiDAR backbone, our BEVFusion-L use spconv for LiDAR backbone.
Because of changing from pillar to spconv, mAP of T4dataset increase from 51.8pt to 60.6pt.

- Offline model

We release the offline model of BEVFusion-L, which plan to use pseudo label.
As voxel size changes from 0.17m to 0.075m, mAP increases from 60.6pt to 65.7pt.
Especially, the mAP of bus, bicycle, and pedestrian increase.

## v0.1.1

As main topics of this version, we added vision language model as Machine Learning tools.
We add [GLIP](/projects/GLIP) as open vocabulary tasks, and [BLIP-2](/projects/BLIP-2) as vision language tasks.
For now, we plan to use select scene from rosbag, and use for dataset making.

In addition, we released TransFusion-L XX1 model (50m, 120m).
It expands the availability of 3D detection with Autoware.

| eval range: 50m            | range | mAP  | car  | truck | bus  | bicycle | pedestrian |
| -------------------------- | ----- | ---- | ---- | ----- | ---- | ------- | ---------- |
| TransFusion-L t4xx1_50m_v1 | 51.2m | 75.1 | 87.2 | 70.7  | 84.7 | 67.9    | 64.9       |

| eval range: 120m            | range   | mAP  | car  | truck | bus  | bicycle | pedestrian |
| --------------------------- | ------- | ---- | ---- | ----- | ---- | ------- | ---------- |
| TransFusion-L t4xx1_120m_v1 | 122.88m | 51.8 | 70.9 | 43.8  | 54.7 | 38.7    | 50.9       |

## v0.1.0

As first version, we released 3D detection train and evaluation pipeline.
As main topics of this version, we released TransFusion-L (TransFusion LiDAR only model with pillar base backbone) with autoware.universe.
We train TransFusion-L by nuScenes and T4dataset and you can see evaluation results as below.

- Model comparison for class mAP with nuScenes eval dataset

|               | mAP  | Car  | Truck | CV   | Bus  | Tra  | Bar  | Mot  | Bic  | Ped  | Cone |
| ------------- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| CenterPoint   | 48.7 | 84.5 | 47.9  | 7.9  | 58.9 | 28.1 | 62.5 | 44.1 | 16.0 | 76.3 | 51.9 |
| TransFusion-L | 55.0 | 85.6 | 54.8  | 14.6 | 71.1 | 33.0 | 58.7 | 49.5 | 32.7 | 79.5 | 56.8 |

- Model comparison for class mAP with T4dataset
  - (v): val, (t): test, CenterPoint-PP = PointPainting with CenterPoint
  - We evaluated CenterPoint by our internal old libraries for CenterPoint, CenterPoint-sigma, and CenterPoint-PointPainting.

|                                | grid, range | mAP  | car  | truck | bus  | bicycle | pedestrian |
| ------------------------------ | ----------- | ---- | ---- | ----- | ---- | ------- | ---------- |
| CenterPoint (v)                | 480, 76.8m  | 61.1 | 84.9 | 52.4  | 39.1 | 57.7    | 71.7       |
| CenterPoint-sigma (v)          | 640, 102.4m | 61.8 | 84.6 | 52.8  | 38.8 | 61.2    | 71.6       |
| CenterPoint-PP (v)             | 560, 89.6m  | 62.7 | 81.5 | 57.2  | 53.1 | 48.2    | 73.7       |
| TransFusion-L t4xx1_90m_v1(v)  | 512, 92.16m | 57.0 | 77.7 | 50.6  | 52.1 | 45.7    | 59.1       |
| TransFusion-L t4xx1_90m_v2(v)  | 768, 92.16m | 67.7 | 85.7 | 65.1  | 60.1 | 56.8    | 70.9       |
|                                |             |      |      |       |      |         |            |
| CenterPoint-sigma (t)          | 640, 102.4m | 64.7 | 81.7 | 55.1  | 62.2 | 58.3    | 66.0       |
| TransFusion-L t4xx1_90m_v1 (t) | 512, 92.16m | 61.0 | 75.4 | 49.7  | 77.1 | 48.3    | 54.4       |
| TransFusion-L t4xx1_90m_v2 (t) | 768, 92.16m | 71.1 | 83.1 | 60.2  | 86.4 | 61.5    | 64.4       |
