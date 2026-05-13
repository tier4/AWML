# Deployed model for BEVFusion-LiDAR JPNTaxi_base/2.X
## Summary

### Main Parameters

  - **Range:** [122.40m, 122.40m, 8.0m]
  - **Voxel Size:** [0.17, 0.17, 0.2]
  - **Grid Size:** [1440, 1440, 40]
  - **With Intensity**

### Testing Datasets

- **Total Frames: 5,179**

	<details>
  <summary> jpntaxi_gen2 (9,975 frames) </summary>
    - `db_jpntaxigen2_v1`
    - `db_jpntaxigen2_v2`

  </details>

### mAP -JPNTaxi_gen2

- **Class mAP for BEV Center Distance: 0.5m, 1.0m, 2.0m, 4.0m**

  <details>
  <summary> Eval Range: 0.0 - 50.0m </summary>

	| Model version | mAP | mAPH | car<br>(42,789) | truck<br>(17,259) | bus<br>(3,437) | bicycle<br>(2,681) | pedestrian<br>(57,948) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR jpntaxi_base/2.7.1 | 0.8862 | 0.8586 | 0.9397 | 0.8591 | 0.8839 | 0.8264 | 0.9218 |

  </details>

  <details>
  <summary> Eval Range: 50.0 - 90.0m </summary>

	| Model version | mAP | mAPH | car<br>(35,518) | truck<br>(22,550) | bus<br>(2,683) | bicycle<br>(1,607) | pedestrian<br>(27,240) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR jpntaxi_base/2.7.1 | 0.7125 | 0.6854 | 0.8453 | 0.6838 | 0.5362 | 0.6969 | 0.8003 |

  </details>

  <details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

	| Model version | mAP | mAPH | car<br>(16,524) | truck<br>(14,587) | bus<br>(2,476) | bicycle<br>(364) | pedestrian<br>(14,297) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR jpntaxi_base/2.7.1 | 0.6030 | 0.5762 | 0.6947 | 0.5260 | 0.5030 | 0.5321 | 0.7591 |

  </details>

  <details open>
  <summary> Eval Range: 0.0 - 121.0m </summary>

	| Model version | mAP | mAPH | car<br>(94,831) | truck<br>(54,396) | bus<br>(8,596) | bicycle<br>(4,652) | pedestrian<br>(99,485) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR jpntaxi_base/2.7.1 | 0.7805 | 0.7527 | 0.8730 | 0.7118 | 0.6785 | 0.7655 | 0.8739 |

  </details>

## Release

### BEVFusion-LiDAR JPNTaxi_base/2.7.1

<details>
<summary> Changes  </summary>

- Finetune from `BEVFusion-LiDAR base/2.7.0` with JPNTaxi_base dataset and intensity.
</details>

<details>
<summary> Artifacts </summary>

- Deployed onnx and ROS parameter files (for internal)
  - [WebAuto](https://evaluation.tier4.jp/evaluation/mlpackages/46f8188d-e3be-4f2f-b989-fd27002610d7/releases/47abcab3-34e1-4971-9bdf-5a2af5d2b2e6?project_id=zWhWRzei)
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-l/jpntaxi_base/v2.7.1/deployment.zip)
  - [Google drive](https://drive.google.com/file/d/1nQlYrnCjlxXbUamEj7MCL_sKxojoU_wk/view?usp=drive_link)
- Logs (for internal)
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-l/jpntaxi_base/v2.7.1/logs.zip)
  - [Google drive](https://drive.google.com/file/d/1q_3zj9nF6mnA5IgyO1QRswS7XqnXqvUH/view?usp=drive_link)
- Pytorch Best checkpoints:
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-l/jpntaxi_base/v2.7.1/best_epoch_30.pth)
  - [Google drive](https://drive.google.com/file/d/1K7rDv7fb8T2haXHxttbZN7FUEoLYESTr/view?usp=drive_link)

</details>

<details>
<summary> Training configs </summary>

- [Config file path](https://github.com/KSeangTan/AWML/blob/07c2e110802ec2537d4c620d9af7f7e1b8120b97/projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_8xb8_jpntaxi_base_120m.py)
- Train time: NVIDIA H100 80GB * 8 * 30 epochs = 20 hours
- Batch size: 8*8 = 64
- Training Dataset (frames: 56,287):
  - jpntaxi: db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 (28,161 frames)
  - jpntaxi_gen2: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (28,126 frames)

</details>

<details>
<summary> Evaluation </summary>

**JPNTaxi_gen2 Datasets (9,975 frames)**:

  - jpntaxi_gen2 (9,975 frames): db_jpntaxigen2_v1 + db_jpntaxigen2_v2

**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.8862**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 42,789 | 0.9397 | 0.891 / 0.943 / 0.960 / 0.965 | 0.918 / 0.946 / 0.953 / 0.954 | 0.284 / 0.175 / 0.175 / 0.164 |
| truck | 17,259 | 0.8591 | 0.701 / 0.842 / 0.935 / 0.958 | 0.792 / 0.882 / 0.932 / 0.946 | 0.409 / 0.321 / 0.241 / 0.241 |
| bus | 3,437 | 0.8839 | 0.796 / 0.888 / 0.925 / 0.927 | 0.853 / 0.897 / 0.910 / 0.910 | 0.296 / 0.184 / 0.104 / 0.104 |
| bicycle | 2,681 | 0.8264 | 0.819 / 0.829 / 0.829 / 0.829 | 0.866 / 0.871 / 0.871 / 0.871 | 0.223 / 0.223 / 0.223 / 0.223 |
| pedestrian | 57,948 | 0.9218 | 0.906 / 0.921 / 0.927 / 0.933 | 0.883 / 0.893 / 0.899 / 0.903 | 0.135 / 0.129 / 0.125 / 0.132 |
| **ALL** | 124,114 | 0.8862 | — | — | — |

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.7125**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 35,518 | 0.8453 | 0.763 / 0.846 / 0.881 / 0.891 | 0.819 / 0.860 / 0.875 / 0.879 | 0.227 / 0.180 / 0.166 / 0.166 |
| truck | 22,550 | 0.6838 | 0.475 / 0.640 / 0.782 / 0.838 | 0.632 / 0.730 / 0.808 / 0.831 | 0.286 / 0.195 / 0.167 / 0.128 |
| bus | 2,683 | 0.5362 | 0.263 / 0.524 / 0.668 / 0.689 | 0.465 / 0.660 / 0.742 / 0.751 | 0.241 / 0.180 / 0.174 / 0.171 |
| bicycle | 1,607 | 0.6969 | 0.656 / 0.709 / 0.710 / 0.713 | 0.745 / 0.770 / 0.771 / 0.772 | 0.145 / 0.138 / 0.138 / 0.138 |
| pedestrian | 27,240 | 0.8003 | 0.782 / 0.798 / 0.807 / 0.814 | 0.782 / 0.790 / 0.795 / 0.799 | 0.163 / 0.163 / 0.163 / 0.164 |
| **ALL** | 89,598 | 0.7125 | — | — | — |

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.6030**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 16,524 | 0.6947 | 0.580 / 0.698 / 0.744 / 0.757 | 0.692 / 0.755 / 0.778 / 0.781 | 0.202 / 0.154 / 0.151 / 0.144 |
| truck | 14,587 | 0.5260 | 0.229 / 0.469 / 0.639 / 0.767 | 0.464 / 0.630 / 0.726 / 0.793 | 0.288 / 0.185 / 0.169 / 0.130 |
| bus | 2,476 | 0.5030 | 0.305 / 0.486 / 0.597 / 0.624 | 0.530 / 0.636 / 0.703 / 0.719 | 0.297 / 0.201 / 0.149 / 0.156 |
| bicycle | 364 | 0.5321 | 0.381 / 0.521 / 0.613 / 0.613 | 0.563 / 0.631 / 0.670 / 0.670 | 0.219 / 0.219 / 0.219 / 0.219 |
| pedestrian | 14,297 | 0.7591 | 0.737 / 0.756 / 0.766 / 0.778 | 0.750 / 0.760 / 0.765 / 0.771 | 0.134 / 0.127 / 0.129 / 0.132 |
| **ALL** | 48,248 | 0.6030 | — | — | — |

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.7805**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 94,831 | 0.8730 | 0.799 / 0.875 / 0.905 / 0.914 | 0.845 / 0.884 / 0.896 / 0.899 | 0.235 / 0.189 / 0.165 / 0.165 |
| truck | 54,396 | 0.7118 | 0.490 / 0.674 / 0.809 / 0.875 | 0.645 / 0.757 / 0.831 / 0.862 | 0.314 / 0.240 / 0.178 / 0.153 |
| bus | 8,596 | 0.6785 | 0.504 / 0.674 / 0.761 / 0.775 | 0.655 / 0.761 / 0.807 / 0.813 | 0.285 / 0.180 / 0.168 / 0.168 |
| bicycle | 4,652 | 0.7655 | 0.736 / 0.770 / 0.778 / 0.778 | 0.800 / 0.816 / 0.819 / 0.820 | 0.194 / 0.159 / 0.159 / 0.159 |
| pedestrian | 99,485 | 0.8739 | 0.857 / 0.872 / 0.880 / 0.887 | 0.835 / 0.845 / 0.850 / 0.854 | 0.142 / 0.137 / 0.135 / 0.137 |
| **ALL** | 261,960 | 0.7805 | — | — | — |

</details>

---
