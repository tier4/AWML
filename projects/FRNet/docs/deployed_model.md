# Deployed model
## NuScenes model

- Performance summary (test set)

| classes | barrier | bicycle | bus  | car  | construction_vehicle | motorcycle | pedestrian | traffic_cone | trailer | truck | driveable_surface | other_flat | sidewalk | terrain | manmade | vegetation | miou | acc  | acc_cls |
| ------- | ------- | ------- | ---- | ---- | -------------------- | ---------- | ---------- | ------------ | ------- | ----- | ----------------- | ---------- | -------- | ------- | ------- | ---------- | ---- | ---- | ------- |
| results | 76.9    | 37.7    | 93.5 | 88.2 | 54.4                 | 81.8       | 74.1       | 64.6         | 65.7    | 77.7  | 96.8              | 75.9       | 75.6     | 76.1    | 88.3    | 86.5       | 75.9 | 93.7 | 83.7    |

- Model
  - Training dataset: nuScenes
  - Eval dataset: nuScenes
  - [PR](https://github.com/tier4/autoware-ml/pull/150)
  - [Config file path](https://github.com/tier4/autoware-ml/blob/2f06bc2a243b6fd44860fed7c77f8fd1e521e89e/projects/FRNet/configs/nuscenes/frnet_1xb4_nus-seg.py)
  - [Deployed onnx](https://drive.google.com/file/d/1tJ2qje4sF1_EaHLvMut1JXV-euJx-JJw/view?usp=drive_link)
  - [Training results](https://drive.google.com/file/d/1GBxHcYd9U6mTNaDyTrTh2FGW1WaJHUcR/view?usp=drive_link)
