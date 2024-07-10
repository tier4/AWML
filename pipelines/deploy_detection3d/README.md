# deploy_detection3d

This pipeline can use for deployment of detection3d models.

## procedure
### 1. Add dataset

- 1.1. Dataset engineer update `autoware-ml/config/dataset` after adding dataset.

### 2. Set environment

- 2.1. Set environment

```sh
git clone https://github.com/tier4/autoware-ml
ln -s {path_to_dataset} data
./pipelines/deploy_detection3d/scripts/init_environment.sh
```

- 2.2. Download new dataset by using [download_t4dataset](/tools/download_t4dataset/README.md).
- 2.3. Update info file

```sh
# XX1
docker run -it --rm --gpus 'all' --name autoware-ml --shm-size=64g -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml bash -c 'python tools/detection3d/create_data_t4dataset.py --root_path ./data/t4dataset --config autoware_ml/configs/detection3d/dataset/t4dataset/xx1.py --version xx1 --max_sweeps 2 --out_dir ./data/t4dataset/info/user_name'
# X2
docker run -it --rm --gpus 'all' --name autoware-ml --shm-size=64g -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml bash -c 'python tools/detection3d/create_data_t4dataset.py --root_path ./data/t4dataset --config autoware_ml/configs/detection3d/dataset/t4dataset/x2.py --version x2 --max_sweeps 2 --out_dir ./data/t4dataset/info/user_name'
```

### 3. Train

- 3.1. Run scripts
  - If you need to change parameters, you need to see for each projects.

```sh
./pipelines/train_models.sh
```

- 3.2 Check results by visualization
  - See [visualization](https://github.com/tier4/autoware-ml/tree/main/tools/detection3d#42-visualization)

### 4. Deploy model

- 4.1 Deploy onnx model

For now, please see for each [projects](/projects).

- 4.2 Deploy parameter files

For now, please see for each [projects](/projects).

- 4.3 Register the deployed files for S3 of AWF version Autoware.

For now, please ask for person in charge.

- 4.4 Update README for autoware-ml

Update README of the projects.
Please write the results of training score and link for the deployed model.
You can refer [this PR](https://github.com/tier4/autoware-ml/pull/76).

### 5. Update Autoware

- 5.1 Update meta repository of Autoware

You can refer [this PR]().

### [option] 6. Update for product version of Pilot.Auto

If you want to use for products, you need to follow this section.

- 6.1. Register the ML model and parameter files of product model for WebAuto

For now, please ask for person in charge.

- 6.2. Update for pilot-auto of the product.

Update model version in `.webauto-ci.yml`.
