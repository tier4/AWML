# Deployed model for FRNet
## Summary

## Release
### base/0.1

We use NuScenes dataset.

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

- Model
  - Training dataset: nuScenes
  - Eval dataset: nuScenes
  - [PR](https://github.com/tier4/autoware-ml/pull/150)
  - [Config file path](https://github.com/tier4/autoware-ml/blob/2f06bc2a243b6fd44860fed7c77f8fd1e521e89e/projects/FRNet/configs/nuscenes/frnet_1xb4_nus-seg.py)
  - [Deployed onnx](https://drive.google.com/file/d/1tJ2qje4sF1_EaHLvMut1JXV-euJx-JJw/view?usp=drive_link)
  - [Training results](https://drive.google.com/file/d/1GBxHcYd9U6mTNaDyTrTh2FGW1WaJHUcR/view?usp=drive_link)
