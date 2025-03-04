# Deployed model for FRNet
## Summary

## Release
### base/0.1

- We released first model of FRNet.
- We used NuScenes dataset.

<details>
<summary> The link of data and evaluation result </summary>

- Model
  - Training dataset: nuScenes
  - Eval dataset: nuScenes
  - [Config file path](https://github.com/tier4/AWML/blob/2f06bc2a243b6fd44860fed7c77f8fd1e521e89e/projects/FRNet/configs/nuscenes/frnet_1xb4_nus-seg.py)
  - Deployed onnx [[Google drive (for internal)]](https://drive.google.com/file/d/1tJ2qje4sF1_EaHLvMut1JXV-euJx-JJw/view?usp=drive_link) [[model-zoo]](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/frnet/frnet-base/nuscenes/v0.1/frnet.onnx)
  - Training results [[Google drive (for internal)]](https://drive.google.com/file/d/1GBxHcYd9U6mTNaDyTrTh2FGW1WaJHUcR/view?usp=drive_link)
  - Training results [model-zoo]
    - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/frnet/frnet-base/nuscenes/v0.1/logs.zip)
    - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/frnet/frnet-base/nuscenes/v0.1/frnet_1xb4_nus-seg.py)
    - [best_iter.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/frnet/frnet-base/nuscenes/v0.1/best_miou_iter_150000.pth)
    - [last_iter.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/frnet/frnet-base/nuscenes/v0.1/last_checkpoint)



- Performance summary (test set)

| model  | miou | acc  | acc_cls |
| ------ | ---- | ---- | ------- |
| v0.1.0 | 75.9 | 93.7 | 83.7    |

- For each class

| class                | v0.1.0 |
| -------------------- | ------ |
| barrier              | 76.9   |
| bicycle              | 37.7   |
| bus                  | 93.5   |
| car                  | 88.2   |
| construction_vehicle | 54.4   |
| motorcycle           | 81.8   |
| pedestrian           | 74.1   |
| traffic_cone         | 64.6   |
| trailer              | 65.7   |
| truck                | 77.7   |
| driveable_surface    | 96.8   |
| other_flat           | 75.9   |
| sidewalk             | 75.6   |
| terrain              | 76.1   |
| manmade              | 88.3   |
| vegetation           | 86.5   |

</details>
