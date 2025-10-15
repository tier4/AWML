# Deployed model for StreamPETR nuscenes baseline
## Summary

### Overview
- Main parameter
  - range: 51.2m
  - image_size: (320, 800)
  - images: `CAM_FRONT, CAM_BACK, CAM_FRONT_LEFT, CAM_BACK_LEFT, CAM_FRONT_RIGHT, CAM_BACK_RIGHT`

| Object Class (<50m)  | AP     |
|----------------------|--------|
| MEAN                 | 0.469  |
| car                  | 0.636  |
| truck                | 0.421  |
| bus                  | 0.483  |
| trailer              | 0.300  |
| construction_vehicle | 0.145  |
| pedestrian           | 0.541  |
| motorcycle           | 0.456  |
| bicycle              | 0.462  |
| traffic_cone         | 0.646  |
| barrier              | 0.607  |

## Release
### StreamPETR Nuscenes Baseline


<summary> The link of data and evaluation result </summary>

- Model
  - Training Datasets: nuscenes (28130 frames)
  - [Config file path](https://github.com/tier4/AWML/blob/ee1150427900393f815b8df99bf7530f0ec8de1c/projects/StreamPETR/configs/nuscenes/nuscenes_vov_flash_320x800_baseline.py)
  - [model-checkpoint](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/streampetr/streampetr-vov99/nuscenes/v1.0/nuscenes_vov99_baseline_320x800.pth)
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/streampetr/streampetr-vov99/nuscenes/v1.0/logs.zip)
  - Train time: NVIDIA H100 80GB * 4 * 35 epochs = 2 days

<details>

- Evaluation
  - Total mAP (eval range = 50m): 0.4697
  - Frame count: 6019

```
mAP: 0.4697  
mATE: 0.6437
mASE: 0.2618
mAOE: 0.3879
mAVE: 0
mAP: 0.4697  
mATE: 0.6437
mASE: 0.2618
mAOE: 0.3879
mAVE: 0.2885
mAAE: 0.1983
NDS: 0.5568
Eval time: 134.7s

Per-class results:
Object Class            AP      ATE     ASE     AOE     AVE     AAE  
car                     0.636   0.457   0.146   0.065   0.299   0.199
truck                   0.421   0.655   0.201   0.077   0.233   0.207
bus                     0.483   0.728   0.201   0.095   0.529   0.322
trailer                 0.300   0.930   0.224   0.588   0.186   0.153
construction_vehicle    0.145   0.972   0.476   1.072   0.134   0.355
pedestrian              0.541   0.616   0.282   0.424   0.330   0.152
motorcycle              0.456   0.661   0.247   0.428   0.419   0.190
bicycle                 0.462   0.550   0.257   0.608   0.179   0.009
traffic_cone            0.646   0.425   0.312   nan     nan     nan  
barrier                 0.607   0.442   0.272   0.134   nan     nan  
.2885
mAAE: 0.1983
NDS: 0.5568
Eval time: 134.7s

Per-class results:
Object Class            AP      ATE     ASE     AOE     AVE     AAE  
car                     0.636   0.457   0.146   0.065   0.299   0.199
truck                   0.421   0.655   0.201   0.077   0.233   0.207
bus                     0.483   0.728   0.201   0.095   0.529   0.322
trailer                 0.300   0.930   0.224   0.588   0.186   0.153
construction_vehicle    0.145   0.972   0.476   1.072   0.134   0.355
pedestrian              0.541   0.616   0.282   0.424   0.330   0.152
motorcycle              0.456   0.661   0.247   0.428   0.419   0.190
bicycle                 0.462   0.550   0.257   0.608   0.179   0.009
traffic_cone            0.646   0.425   0.312   nan     nan     nan  
barrier                 0.607   0.442   0.272   0.134   nan     nan  
```

</details>
