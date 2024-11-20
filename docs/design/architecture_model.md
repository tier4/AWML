# Architecture for ML model
## ML model

![](/docs/fig/data_pipeline.drawio.svg)

- `autoware-ml` with [WebAuto](https://docs.web.auto/en/)

Download T4dataset by using WebAuto system.

- `autoware-ml` with [T4dataset tools](https://github.com/tier4/tier4_perception_dataset)

`autoware-ml` make pseudo label T4dataset from non-annotated T4dataset.
It lead to make short time to be annotated by using `autoware-ml`.

## T4dataset type

The type of T4dataset is following

- Database dataset

It is used for training and evaluation.
We manage database dataset in [dataset config](/autoware_ml/configs/detection3d/dataset/t4dataset) like `database_v1_0.yaml`.

- Pseudo dataset

It is mainly used to train pre-training model.
We manage database dataset in [dataset config](/autoware_ml/configs/detection3d/dataset/t4dataset) like `pseudo_v1_0.yaml`.
Pseudo T4dataset is created by [t4dataset_pseudo_label_3d](/tools/t4dataset_pseudo_label_3d/).
Note that `autoware-ml` do not manage pseudo T4dataset which is used for domain adaptation.

## ML model type

![](/docs/fig/model_pipeline.drawio.svg)

- `autoware-ml` with [WebAuto](https://docs.web.auto/en/)

Download T4dataset by using WebAuto system.

- `autoware-ml` with [T4dataset tools](https://github.com/tier4/tier4_perception_dataset)

`autoware-ml` make pseudo label T4dataset from non-annotated T4dataset.
It lead to make short time to be annotated by using `autoware-ml`.

## T4dataset

The type of T4dataset is following

- Database dataset

It is used for training and evaluation.
We manage database dataset in [dataset config](/autoware_ml/configs/detection3d/dataset/t4dataset) like `database_v1_0.yaml`.

- Pseudo dataset

It is mainly used to train pre-training model.
We manage database dataset in [dataset config](/autoware_ml/configs/detection3d/dataset/t4dataset) like `pseudo_v1_0.yaml`.
Pseudo T4dataset is created by [t4dataset_pseudo_label_3d](/tools/t4dataset_pseudo_label_3d/).
Note that `autoware-ml` do not manage pseudo T4dataset which is used for domain adaptation.

## ML model

We define 4 types model for deployment and 1 type for active learning.
Here is example for Camera-LIDAR 3D detection model for deploy pipeline.

![](/docs/fig/model_pipeline.drawio.svg)

For now, we deploy only product model in 3D detection with TransFusion LiDAR-only model as product model and BEVFusion LiDAR-only model as offline model.

### 1. Pretrain model

"Pretrain model" is used for training base model to increase generalization performance.
"Base model" is basically trained by public dataset and pseudo label dataset.
"Pretrain model" is managed by `autoware-ml`.

### 2. Base model

"Base model" can be used for a wide range of projects.
"Base model" is based on LiDAR-only model for 3D detection to general purpose.
"Base model" is basically fine-tuned by all of T4dataset from "Pretrain model".
"Base model" is managed by `autoware-ml`.

### 3. Product model

"Product model" can be used for a product defined by reference design like XX1 and X2.
"Product model" can use specific sensor configuration to deploy for sensor fusion model.
"Product model" is basically fine-tuned by all of product dataset from "Base model".
"Product model" is managed by `autoware-ml`.

### 4. Project model

If the performance "product model" is not enough in some reason, "Project model" can be used for specific project.
"Project model" adapts to specific domain, trained by pseudo label using "Offline model".
"Project model" sometimes uses for project-only dataset, which cannot use for other project for some reason.
`autoware-ml` do not manage "product model".

### 5. Offline model

"Offline model" can be used to offline process like pseudo label and cannot be used for real-time autonomous driving application.
"Offline model" is based on LiDAR-only model for 3D detection for generalization performance.
"Base model" is basically trained by all dataset.

## The versioning for ML model

We follow the strategy for ML model release as below.

![](/docs/fig/model_release.drawio.svg)

When new dataset is added, we release the base model at first.
After that we release the product model using the base model as pre-trained model.
If some problem like domain-specific objects, we release the project model by retraining the product model using pseudo label with the offline model.
Note that we do not retrain the product model and project model from the model of before version because it is difficult to trace the model.

## Model management with S3 storage

We prepare S3 storage for models and intermediate product.
In `autoware-ml`, we can use URL path instead of local path as MMLab libraries.
For example, we can run the script as `python tools/detection2d/test.py projects/YOLOX/configs/yolox_l_8xb8-300e_coco.py https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth`.

- Model: {URL}/autoware-ml/models/{algorithm}/{pretrain/base/product}/{version}/{pth, config, log}

```
- {URL}/autoware-ml/models/TransFusion/
  - pretrain/
    - v1/
      - epoch_20.pth
      - config.py
      - log.log
  - base/
    - v1/
      - epoch_20.pth
      - config.py
      - log.log
      - transfusion.onnx
```

- Info file: {URL}/autoware-ml/info/{autoware-ml version}/{pretrain/base/product}/{info_train.pkl, info_val.pkl, info_test.pkl}

```
- {URL}/autoware-ml/info/v0.3.0/product/
  - info_train.pkl
  - info_val.pkl
  - info_test.pkl
```

- Intermediate product for pseudo label: {URL}/autoware-ml/pseudo_label/{day}_{dataset}/info_all.pkl, dataset_id_list.yaml
  - This info file is used to reproduce pseudo T4dataset adjusting confidence threshold as it reduce re-calculation GPU cost.

```
- {URL}/autoware-ml/pseudo_label/20241101_xx1/
  - info_all.pkl
  - dataset_id_list.yaml
```
