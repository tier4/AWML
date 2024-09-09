# Dataset docs
## Dataset versioning strategy
### dataset version definition

- version: major.minor.build
  - major: sensor configuration
  - minor: dataset scenes
  - build: dataset version

## T4dataset version
### XX1

- [database_v1_0](/autoware_ml/configs/detection3d/dataset/t4dataset/database_v1_0.yaml)
  - Place: Odaiba, Nishi-Shinjuku, and Shiojiri
  - Amount: About 12000 frames
  - Sensor: Velodyne LiDAR + BFS Camera
  - Annotation: All the data are collected at 10Hz and most of them are annotated at 2Hz
    - DBv1.0_nishi_shinjuku_[0-7]_ are annotated at 10Hz.
- [database_v1_1](/autoware_ml/configs/detection3d/dataset/t4dataset/database_v1_1.yaml)
  - Place: Odaiba
  - Amount: About 2000 frames
  - Sensor: Velodyne LiDAR + BFS Camera
  - Annotation: All the data are collected at 10Hz and annotated at 2Hz.
- [database_v1_3](/autoware_ml/configs/detection3d/dataset/t4dataset/database_v1_3.yaml)
  - Place: Odaiba and Shinagawa
  - Amount: About 6000 frames
  - Sensor: Velodyne LiDAR + C1 Camera + Radar data
  - Annotation: WIP

### X2

- [database_v2_0](/autoware_ml/configs/detection3d/dataset/t4dataset/database_v2_0.yaml)
  - Place: Nishi-Shinjuku and GLP-Atsugi
  - Amount: About 8000 frames
  - Sensor: Hesai LiDAR + BFS Camera
  - Annotation: All the data are collected and annotated at 10Hz.
- [database_v2_1 (Under construction)](/autoware_ml/configs/detection3d/dataset/t4dataset/database_v2_1.yaml)
  - Place: GLP-Sagamihara
  - Amount: About 6000 frames
  - Sensor: Hesai LiDAR + C1 Camera
  - Annotation: All the data are collected and annotated at 10Hz.
- [database_v3_0](/autoware_ml/configs/detection3d/dataset/t4dataset/database_v3_0.yaml)
  - Place: Shiojiri
  - Amount: About 5000 frames
  - Sensor: Hesai LiDAR + C1 Camera + Radar data
  - All the data are collected at 10Hz and annotated at 2Hz.
