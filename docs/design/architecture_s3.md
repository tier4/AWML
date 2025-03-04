## The design of deployment flow with S3

The whole pipeline of deployment flow with S3

![](/docs/fig/model_deployment.drawio.svg)

- 1. Local PC of basic model provider

An engineer makes model data of basic model (any of pretrain model, base model, product model, offline model).
Model data has checkpoint of the model, training log, onnx file for Autoware, ROS parameter for Autoware.
If the model is used for not only `AWML` but also Autoware, we also upload to WebAuto model management system (Basically, we will upload product model).

We plan that we upload model data to S3 from local PC until MLOps system is constructed.

- 2. Local PC of project model provider

We got the issue from projects, we deploy for a model dedicated to that project.
So the engineer who makes project model upload to WebAuto model management system.
Because project models are not managed by `AWML`, he/she upload the model data only to WebAuto model management system.

- 3. S3 model registry

`AWML` manage pretrain model, base model, product model, and offline model.

- 4. WebAuto model management system

WebAuto model management system provide the model deployment.
In detail, please see [WebAuto document](https://docs.web.auto/en/user-manuals/evaluator-ml-pipeline/introduction).

- 5. autoware-ml user

`AWML` users can use S3 model registry as model zoo.
For example, the user of auto labeling can use offline model, and the user of fine-tuning can use base model.

- 6. Autoware user

Autoware user choose the model to use from WebAuto model management system.
In detail, please see [WebAuto document](https://docs.web.auto/en/user-manuals/evaluator-ml-pipeline/introduction).

## Data to manage with S3 storage

We use S3 storage to manage models and intermediate product.
The architecture of directory is following.

```
- {base URL}/autoware-ml/
  - models/
  - info/
  - auto_label/
```

### `/autoware-ml/models/`: Data about models

We use the model registered in S3 storage for fine tuning like model zoo.
In `AWML`, we can use URL path instead of local path as MMLab libraries.
For example, we can run the script as `python tools/detection2d/test.py projects/YOLOX/configs/yolox_l_8xb8-300e_coco.py https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth`.

Model data to manage in S3 are following

- Checkpoint file
  - We use checkpoint file for fine tuning.
  - e.g. `epoch_20.pth`
- Config file
  - We use config file for reproduction of training.
  - e.g. `transfusion_lidar_90m-768grid-t4xx1.py`
- Log file
  - We use log files for reproduction of training.
  - e.g. `train_log.log`, `test_log.log`
- Onnx file
  - e.g. `transfusion.onnx`
- ROS parameter file
  - e.g. `transfusion.param.yaml`

The directory architecture is like `{base URL}/autoware-ml/models/{algorithm}/{model-name}/{version}/*`. For example, we will construct as below.

```
- {base URL}/autoware-ml/models/
  - TransFusion/
    - nuscenes/
      - v1.0.0/
        - epoch_20.pth
        - config.py
        - log.log
    - t4pretrain/
      - v1.0.0/
        - epoch_20.pth
        - config.py
        - log.log
    - t4base/
      - v1.0.0/
        - epoch_20.pth
        - config.py
        - log.log
        - transfusion.onnx
        - transfusion.param.yaml
      - v1.1.0/
        - epoch_20.pth
        - config.py
        - log.log
        - transfusion.onnx
        - transfusion.param.yaml
    - t4x2/
      - v1.0.0/
        - epoch_20.pth
        - config.py
        - log.log
        - transfusion.onnx
        - transfusion.param.yaml
    - t4x2-50m/
      - v1.0.0/
        - epoch_20.pth
        - config.py
        - log.log
        - transfusion.onnx
        - transfusion.param.yaml
  - CenterPoint/
    - ...
```

### `/autoware-ml/info/` Info file using for training

We use info file registered in S3 storage to reduce duplicated process to make info files.

The directory architecture is like `{base URL}/autoware-ml/info/{autoware-ml version}/{pretrain/base/product}/{info_train.pkl, info_val.pkl, info_test.pkl}`.
For example, we will construct as below.

```
- {base URL}/autoware-ml/info/v0.3.0/
  - base/
    - info_train.pkl
    - info_val.pkl
    - info_test.pkl
  - X1/
    - info_train.pkl
    - info_val.pkl
    - info_test.pkl
  - X2/
    - info_train.pkl
    - info_val.pkl
    - info_test.pkl
```

### `/autoware-ml/auto_label/` Info file used by auto labeling

We use info file registered in S3 storage for reproduction of Pseudo T4dataset to reduce GPU calculation cost.

The directory architecture is like `{base URL}/autoware-ml/auto_label/{day}_{dataset}/info_all.pkl, dataset_id_list.yaml`.
For example, we will construct as below.

```
- {base URL}/autoware-ml/auto_label/20241101_xx1/
  - pseudo_infos_raw_centerpoint.pkl
  - pseudo_infos_raw_bevfusion.pkl
  - pseudo_infos_filtered.pkl
  - log.log
```

This info file is used to reproduce pseudo T4dataset adjusting confidence threshold as it reduce re-calculation GPU cost.
