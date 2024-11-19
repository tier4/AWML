# deploy_detection3d

This pipeline can use for deployment of detection3d models.

- [Support priority](https://github.com/tier4/autoware-ml/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier S

## 1. Update info files

You can choose to make info file by local and WebAuto

- [choice 1] Make info file by local
  - You run the script written by ["2. Prepare dataset" on tools/detection3d](https://github.com/tier4/autoware-ml/tree/main/tools/detection3d)

```
python tools/detection3d/create_data_t4dataset.py --root_path ./data/t4dataset --config autoware_ml/configs/detection3d/dataset/t4dataset/base.py --version xx1 --max_sweeps 2 --out_dir ./data/t4dataset/info/user_name
```

- [choice 2] Make info file by WebAuto

```sh
# Make info file by WebAuto
TBD
# Download the info files
TBD
```

- The info files are as below.

```
- work_dirs/info_file/
  - t4dataset_base_info_all.pkl
  - t4dataset_base_info_train.pkl
  - t4dataset_base_info_eval.pkl
  - t4dataset_base_info_test.pkl
```

- Upload to S3

```
aws s3 sync ./work_dirs/info_file s3://autoware-ml/info_file/base/20241031/ --exclude "*" --include "*.pkl"
```

## 2. Deploy base model for 3D detection

- Train base model by WebAuto and download the results.
  - If you need to change parameters, you need to see for each [projects](/projects).

```sh
# Training
TBD
# Download the info files
TBD
```

- The output directory is as below.

```
- work_dirs/{config_name}/
  - checkpoint/
    - epoch_1.pth
    - epoch_2.pth
    - ...
  - upload/
    - checkpoint.pth
    - {yyyymmdd}.log
    - transfusion.onnx
    - transfusion.yaml
```

- Check the result of the trained model.
- If the result of the model is well, you can upload to S3.

```
aws s3 sync ./work_dirs/{config_name}/upload s3://autoware-ml/model/TransFusion/base/v3/
```

## 3. Deploy product model for 3D detection
### Train the model

- Set parameter and run training script

```sh
# Training
TBD
# Download the info files
TBD
```

- The output directory is as below.

```
- work_dirs/{config_name}/
  - checkpoint/
    - epoch_1.pth
    - epoch_2.pth
    - ...
  - upload/
    - checkpoint.pth
    - {yyyymmdd}.log
    - transfusion.onnx
    - transfusion.yaml
```

- Check the result of the trained model.
- If the result of the model is well, you can upload to S3.

```
aws s3 sync ./work_dirs/{config_name}/upload s3://autoware-ml/model/TransFusion/xx1/v3/
```

- Update README for autoware-ml

Update README of the projects.
Please write the results of training score and link for the deployed model.
You can refer [this PR](https://github.com/tier4/autoware-ml/pull/76).

### Check with evaluator

Check the evaluation result by WebAuto Evaluator

- Update model version in `.webauto-ci.yml`.

```sh
TBD
```

- If the evaluation succeed, upload the models to WebAuto ML model registry.

```sh
pipeline/deploy_detection3d/deploy_to_webauto.sh
```

- After uploading to WebAuto, update for pilot-auto of the product.
