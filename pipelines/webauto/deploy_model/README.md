# Deploy model to WebAuto

You can deploy model by shell script using WebAuto-CLI.

- [Support priority](https://github.com/tier4/autoware-ml/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier S

## Procedure

- Set directory

```
- work_dirs/20241118_transfusion_lidar_90m-768grid-t4base/
  - deploy/
    - transfusion.onnx
    - transfusion.yaml (ROS parameter)
    - 20240530_180730.log
    - epoch_20.pth
```

- Run the script

```sh
pipelines/webauto/deploy_model/deploy_model.sh {path to deploy directory}
```

- For example, please run the following command

```sh
pipelines/webauto/deploy_model/deploy_model.sh ./work_dirs/20241118_transfusion_lidar_90m-768grid-t4base/deploy/
```
