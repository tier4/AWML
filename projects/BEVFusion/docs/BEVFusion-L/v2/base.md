# Deployed model for BEVFusion-LiDAR base/2.X
## Summary

### Main Parameters

  - **Range:** [122.40m, 122.40m, 8.0m]
  - **Voxel Size:** [0.17, 0.17, 0.2]
  - **Grid Size:** [1440, 1440, 40]

### Testing Datasets

- **Total Frames: 15,154**

  <details>
  <summary> j6gen2 (3,951 frames) </summary>

  - `db_j6gen2_v1`
  - `db_j6gen2_v2`
  - `db_j6gen2_v3`
  - `db_j6gen2_v4`
  - `db_j6gen2_v5`
  - `db_j6gen2_v6`
  - `db_j6gen2_v7`
  - `db_j6gen2_v8`
  - `db_j6gen2_v9`

  </details>

  <details>
  <summary> largebus (1,228 frames) </summary>

  - `db_largebus_v1`
  - `db_largebus_v2`
  - `db_largebus_v3`

  </details>

  <details>
  <summary> jpntaxi_gen2 (9,975 frames) </summary>

  - `db_jpntaxigen2_v1`
  - `db_jpntaxigen2_v2`

  </details>

  <details>
  <summary> base (15,154 frames) </summary>

  - `db_j6gen2_v1`
  - `db_j6gen2_v2`
  - `db_j6gen2_v3`
  - `db_j6gen2_v4`
  - `db_j6gen2_v5`
  - `db_j6gen2_v6`
  - `db_j6gen2_v7`
  - `db_j6gen2_v8`
  - `db_j6gen2_v9`
  - `db_largebus_v1`
  - `db_largebus_v2`
  - `db_largebus_v3`
  - `db_jpntaxigen2_v1`
  - `db_jpntaxigen2_v2`

  </details>

### mAP - Base

- **Class mAP for BEV Center Distance: 0.5m, 1.0m, 2.0m, 4.0m**

  <details>
  <summary> Eval Range: 0.0 - 50.0m </summary>

  | Model version | mAP | car<br>(107,309) | truck<br>(24,206) | bus<br>(5,712) | bicycle<br>(4,060) | pedestrian<br>(77,369) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.6.0 | 0.8774 | 0.9049 | 0.8514 | 0.8824 | 0.8543 | 0.8941 |

  </details>

  <details>
  <summary> Eval Range: 50.0 - 90.0m </summary>

  | Model version | mAP | mAPH | car<br>(94,080) | truck<br>(27,651) | bus<br>(4,761) | bicycle<br>(2,365) | pedestrian<br>(37,523) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.6.0 | 0.6824 | 0.6437 | 0.8005 | 0.6567 | 0.5783 | 0.6322 | 0.7445 |

  </details>

  <details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version | mAP | mAPH | car<br>(36,895) | truck<br>(17,759) | bus<br>(2,852) | bicycle<br>(519) | pedestrian<br>(17,091) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.6.0 | 0.5136 | 0.4788 | 0.6552 | 0.5023 | 0.2849 | 0.4369 | 0.6887 |

  </details>

  <details open>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version | mAP | mAPH | car<br>(238,284) | truck<br>(69,616) | bus<br>(13,325) | bicycle<br>(6,944) | pedestrian<br>(131,983) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.6.0 | 0.7592 | 0.7227 | 0.8398 | 0.6994 | 0.6621 | 0.7595 | 0.8351 |

  </details>

### Insights

**Performance by range (base/2.6.0):**

| Range | mAP | mAPH |
| :---- | ---: | ---: |
| 0-50m | 0.8774 | — |
| 50-90m | 0.6824 | 0.6437 |
| 90-121m | 0.5136 | 0.4788 |
| **0-121m** | **0.7592** | **0.7227** |

- Near-range (0-50m) detection is strong across all classes (mAP 0.8774). Performance degrades significantly with distance: -22.2% at mid-range and -41.4% at far-range relative to near-range.
- Largest far-range drop: **bus** falls from 0.8824 (near) to 0.2849 (far), likely due to sparse LiDAR returns on large vehicles at distance.

**Performance by class (0-121m):**

| Class | mAP | Strongest range | Weakest range |
| :---- | ---: | :---- | :---- |
| car | 0.8398 | 0-50m (0.9049) | 90-121m (0.6552) |
| truck | 0.6994 | 0-50m (0.8514) | 90-121m (0.5023) |
| bus | 0.6621 | 0-50m (0.8824) | 90-121m (0.2849) |
| bicycle | 0.7595 | 0-50m (0.8543) | 90-121m (0.4369) |
| pedestrian | 0.8351 | 0-50m (0.8941) | 90-121m (0.6887) |

- **Pedestrian** retains the best far-range detection (0.6887), benefiting from the large GT count (131,983 total).
- **Bus** is weakest overall (0.6621) due to fewer training samples and poor far-range performance.

**Performance by dataset (0-121m):**

| Dataset | Frames | mAP | mAPH | Best class | Worst class |
| :---- | ---: | ---: | ---: | :---- | :---- |
| LargeBus | 1,228 | 0.7995 | 0.7514 | bus (0.8608) | bicycle (0.7272) |
| J6Gen2 | 3,951 | 0.7712 | 0.7223 | bus (0.8348) | truck (0.7129) |
| JPNTaxi Gen2 | 9,975 | 0.7471 | 0.7176 | pedestrian (0.8606) | bus (0.5446) |
| **Base (all)** | **15,154** | **0.7592** | **0.7227** | | |

- **LargeBus** achieves the highest mAP (0.7995) despite having the fewest frames, likely due to simpler driving scenarios.
- **JPNTaxi Gen2** has the lowest mAP (0.7471), with bus detection (0.5446) lagging significantly behind other datasets, possibly due to different bus appearance characteristics in the jpntaxi environment.

## Datasets

<details>
<summary> JPNTaxi Gen2 </summary>

- Datasets (9,975 Testing Frames):
  - `db_jpntaxigen2_v1`
  - `db_jpntaxigen2_v2`

- **Class mAP for BEV Center Distance: 0.5m, 1.0m, 2.0m, 4.0m**

  <details>
  <summary> Eval Range: 0.0 - 50.0m </summary>

  | Model version | mAP | mAPH | car<br>(42,789) | truck<br>(17,259) | bus<br>(3,437) | bicycle<br>(2,681) | pedestrian<br>(57,948) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.6.0 | 0.8784 | 0.8487 | 0.9436 | 0.8531 | 0.8284 | 0.8546 | 0.9123 |

  </details>

  <details>
  <summary> Eval Range: 50.0 - 90.0m </summary>

  | Model version | mAP | mAPH | car<br>(35,518) | truck<br>(22,550) | bus<br>(2,683) | bicycle<br>(1,607) | pedestrian<br>(27,240) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.6.0 | 0.6692 | 0.6414 | 0.8323 | 0.6571 | 0.4033 | 0.6721 | 0.7812 |

  </details>

  <details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version | mAP | mAPH | car<br>(16,524) | truck<br>(14,587) | bus<br>(2,476) | bicycle<br>(364) | pedestrian<br>(14,297) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.6.0 | 0.5300 | 0.5010 | 0.6692 | 0.5020 | 0.2822 | 0.4586 | 0.7380 |

  </details>

  <details open>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version | mAP | mAPH | car<br>(94,831) | truck<br>(54,396) | bus<br>(8,596) | bicycle<br>(4,652) | pedestrian<br>(99,485) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.6.0 | 0.7471 | 0.7176 | 0.8667 | 0.6928 | 0.5446 | 0.7710 | 0.8606 |

  </details>

</details>

<details>
<summary> LargeBus </summary>

- Datasets (1,228 Testing Frames):
  - `db_largebus_v1`
  - `db_largebus_v2`
  - `db_largebus_v3`

- **Class mAP for BEV Center Distance: 0.5m, 1.0m, 2.0m, 4.0m**

  <details>
  <summary> Eval Range: 0.0 - 50.0m </summary>

  | Model version | mAP | mAPH | car<br>(14,883) | truck<br>(1,193) | bus<br>(336) | bicycle<br>(740) | pedestrian<br>(5,059) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.6.0 | 0.8882 | 0.8475 | 0.9045 | 0.8793 | 0.9482 | 0.8489 | 0.8598 |

  </details>

  <details>
  <summary> Eval Range: 50.0 - 90.0m </summary>

  | Model version | mAP | mAPH | car<br>(10,994) | truck<br>(1,011) | bus<br>(143) | bicycle<br>(463) | pedestrian<br>(3,754) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.6.0 | 0.7132 | 0.6586 | 0.8237 | 0.7245 | 0.7811 | 0.5497 | 0.6871 |

  </details>

  <details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version | mAP | mAPH | car<br>(3,018) | truck<br>(602) | bus<br>(60) | bicycle<br>(85) | pedestrian<br>(1,121) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.6.0 | 0.5202 | 0.4736 | 0.6989 | 0.6297 | 0.4058 | 0.3609 | 0.5056 |

  </details>

  <details open>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version | mAP | mAPH | car<br>(28,895) | truck<br>(2,806) | bus<br>(539) | bicycle<br>(1,288) | pedestrian<br>(9,934) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.6.0 | 0.7995 | 0.7514 | 0.8640 | 0.7788 | 0.8608 | 0.7272 | 0.7669 |

  </details>

</details>

<details>
<summary> J6Gen2 </summary>

- Datasets (3,951 Testing Frames):
  - `db_j6gen2_v1`
  - `db_j6gen2_v2`
  - `db_j6gen2_v3`
  - `db_j6gen2_v4`
  - `db_j6gen2_v5`
  - `db_j6gen2_v6`
  - `db_j6gen2_v7`
  - `db_j6gen2_v8`
  - `db_j6gen2_v9`

- **Class mAP for BEV Center Distance: 0.5m, 1.0m, 2.0m, 4.0m**

  <details>
  <summary> Eval Range: 0.0 - 50.0m </summary>

  | Model version | mAP | mAPH | car<br>(49,637) | truck<br>(5,754) | bus<br>(1,939) | bicycle<br>(639) | pedestrian<br>(14,362) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.6.0 | 0.8702 | 0.8284 | 0.8758 | 0.8410 | 0.9408 | 0.8590 | 0.8344 |

  </details>

  <details>
  <summary> Eval Range: 50.0 - 90.0m </summary>

  | Model version | mAP | mAPH | car<br>(47,568) | truck<br>(4,090) | bus<br>(1,935) | bicycle<br>(295) | pedestrian<br>(6,529) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.6.0 | 0.6708 | 0.6165 | 0.7721 | 0.6421 | 0.7731 | 0.5472 | 0.6192 |

  </details>

  <details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version | mAP | mAPH | car<br>(17,353) | truck<br>(2,570) | bus<br>(316) | bicycle<br>(70) | pedestrian<br>(1,673) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.6.0 | 0.4462 | 0.4042 | 0.6346 | 0.4758 | 0.3215 | 0.4303 | 0.3688 |

  </details>

  <details open>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version | mAP | mAPH | car<br>(114,558) | truck<br>(12,414) | bus<br>(4,190) | bicycle<br>(1,004) | pedestrian<br>(22,564) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.6.0 | 0.7712 | 0.7223 | 0.8110 | 0.7129 | 0.8348 | 0.7458 | 0.7515 |

  </details>

</details>

## Release

### BEVFusion-LiDAR base/2.6.0

<details>
<summary> Changes  </summary>

- Train with more data:
  - `db_j6gen2_v2`
  - `db_j6gen2_v3`
  - `db_j6gen2_v4`
  - `db_j6gen2_v6`
  - `db_j6gen2_v7`
  - `db_largebus_v1`
  - `db_jpntaxigen2_v1`
- Train with new data sets:
  - `db_j6gen2_v9`
  - `db_largebus_v3`
- Train with 8 GPUs instead of 4 GPUs, and thus, it increases the effective batch size from `32` to `64`
- Fixed `BatchNorm` in the DDP environment with `SyncBatchNorm`

</details>

<details>
<summary> Artifacts </summary>

- Deployed onnx and ROS parameter files (for internal)
  - [WebAuto](https://evaluation.tier4.jp/evaluation/mlpackages/46f8188d-e3be-4f2f-b989-fd27002610d7/releases/efc3e923-9fa2-4c18-ad6a-e0eaeed34e71?project_id=zWhWRzei)
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-l/t4base/v2.6.0/deployment.zip)
  - [Google drive](https://drive.google.com/file/d/1Zrxo2qNaVOGCbAEdsUN2pmp2dN5lViDV/view?usp=drive_link)
- Logs (for internal)
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-l/t4base/v2.6.0/logs.zip)
  - [Google drive](https://drive.google.com/file/d/1JMx2ec6cSRlTyV7lwPJrPJAfY7bDbMUT/view?usp=drive_link)
- Pytorch Best checkpoints:
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-l/t4base/v2.6.0/epoch_50.pth)
  - [Google drive](https://drive.google.com/file/d/15XjV2pwm1vTfOQE1cA5hZkHA7k8QMETJ/view?usp=drive_link)

</details>

<details>
<summary> Training configs </summary>

- [Config file path](https://github.com/KSeangTan/AWML/blob/1a9cb6f59e38274fa02aa789e3799652908a3678/projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_50e_8xb8_base_120m.py)
- Train time: NVIDIA H100 80GB * 8 * 50 epochs ~= 4 days
- Batch size: 8*8 = 64
- Training Dataset (frames: 142,196):
  - jpntaxi: db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 (28,161 frames)
  - j6: db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 (29,336 frames)
  - j6gen2: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 + db_j6gen2_v6 + db_j6gen2_v7 + db_j6gen2_v8 + db_j6gen2_v9 (43,968 frames)
  - largebus: db_largebus_v1 + db_largebus_v2 (12,605 frames)
  - jpntaxi_gen2: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (28,126 frames)

</details>

<details>
<summary> Evaluation </summary>

**Base Datasets (15,154 frames)**:

  - j6gen2 (3,951 frames): db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 + db_j6gen2_v6 + db_j6gen2_v7 + db_j6gen2_v8 + db_j6gen2_v9
  - largebus (1,228 frames): db_largebus_v1 + db_largebus_v2 + db_largebus_v3
  - jpntaxi_gen2 (9,975 frames): db_jpntaxigen2_v1 + db_jpntaxigen2_v2

**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.8774**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 107,309 | 0.9049 | 0.851 / 0.902 / 0.924 / 0.942 | 0.897 / 0.929 / 0.937 / 0.942 | 0.247 / 0.195 / 0.159 / 0.141 |
| truck | 24,206 | 0.8514 | 0.701 / 0.841 / 0.919 / 0.945 | 0.799 / 0.875 / 0.920 / 0.934 | 0.297 / 0.196 / 0.169 / 0.165 |
| bus | 5,712 | 0.8824 | 0.781 / 0.878 / 0.934 / 0.937 | 0.805 / 0.864 / 0.898 / 0.900 | 0.027 / 0.024 / 0.024 / 0.024 |
| bicycle | 4,060 | 0.8543 | 0.833 / 0.857 / 0.863 / 0.864 | 0.860 / 0.869 / 0.870 / 0.870 | 0.242 / 0.230 / 0.228 / 0.228 |
| pedestrian | 77,369 | 0.8941 | 0.875 / 0.892 / 0.901 / 0.909 | 0.856 / 0.866 / 0.872 / 0.877 | 0.156 / 0.148 / 0.149 / 0.148 |
| **ALL** | 218,656 | 0.8774 | — | — | — |

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.6824**

| Label | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 94,080 | 0.8005 | 0.683 / 0.798 / 0.848 / 0.873 | 0.771 / 0.833 / 0.859 / 0.869 | 0.230 / 0.179 / 0.158 / 0.141 |
| truck | 27,651 | 0.6567 | 0.430 / 0.620 / 0.760 / 0.817 | 0.600 / 0.718 / 0.794 / 0.820 | 0.239 / 0.193 / 0.162 / 0.155 |
| bus | 4,761 | 0.5783 | 0.321 / 0.551 / 0.705 / 0.736 | 0.472 / 0.623 / 0.721 / 0.739 | 0.255 / 0.069 / 0.068 / 0.068 |
| bicycle | 2,365 | 0.6322 | 0.574 / 0.647 / 0.653 / 0.655 | 0.683 / 0.714 / 0.715 / 0.716 | 0.172 / 0.172 / 0.172 / 0.172 |
| pedestrian | 37,523 | 0.7445 | 0.724 / 0.742 / 0.752 / 0.761 | 0.738 / 0.747 / 0.752 / 0.757 | 0.158 / 0.152 / 0.151 / 0.152 |
| **ALL** | 166,380 | 0.6824 | — | — | — |

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.5136**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 36,895 | 0.6552 | 0.493 / 0.651 / 0.724 / 0.752 | 0.626 / 0.716 / 0.751 / 0.763 | 0.181 / 0.160 / 0.155 / 0.140 |
| truck | 17,759 | 0.5023 | 0.195 / 0.447 / 0.626 / 0.742 | 0.420 / 0.598 / 0.708 / 0.767 | 0.205 / 0.189 / 0.160 / 0.145 |
| bus | 2,852 | 0.2849 | 0.103 / 0.282 / 0.359 / 0.395 | 0.331 / 0.446 / 0.491 / 0.511 | 0.025 / 0.027 / 0.027 / 0.027 |
| bicycle | 519 | 0.4369 | 0.336 / 0.420 / 0.496 / 0.496 | 0.509 / 0.551 / 0.580 / 0.580 | 0.181 / 0.123 / 0.181 / 0.181 |
| pedestrian | 17,091 | 0.6887 | 0.667 / 0.684 / 0.694 / 0.710 | 0.704 / 0.712 / 0.718 / 0.726 | 0.134 / 0.134 / 0.134 / 0.134 |
| **ALL** | 75,116 | 0.5136 | — | — | — |

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.7592**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 238,284 | 0.8398 | 0.744 / 0.838 / 0.878 / 0.900 | 0.809 / 0.862 / 0.881 / 0.888 | 0.230 / 0.177 / 0.159 / 0.157 |
| truck | 69,616 | 0.6994 | 0.475 / 0.666 / 0.797 / 0.859 | 0.632 / 0.749 / 0.823 / 0.853 | 0.269 / 0.199 / 0.163 / 0.155 |
| bus | 13,325 | 0.6621 | 0.478 / 0.650 / 0.749 / 0.771 | 0.567 / 0.673 / 0.732 / 0.743 | 0.228 / 0.044 / 0.044 / 0.044 |
| bicycle | 6,944 | 0.7595 | 0.721 / 0.765 / 0.775 / 0.777 | 0.777 / 0.796 / 0.799 / 0.800 | 0.183 / 0.183 / 0.183 / 0.184 |
| pedestrian | 131,983 | 0.8351 | 0.815 / 0.833 / 0.842 / 0.851 | 0.804 / 0.814 / 0.819 / 0.825 | 0.148 / 0.148 / 0.148 / 0.148 |
| **ALL** | 460,152 | 0.7592 | — | — | — |

---

**LargeBus**: db_largebus_v1 + db_largebus_v2 + db_largebus_v3 (1,228 frames)  

**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.8882**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 14,883 | 0.9045 | 0.854 / 0.902 / 0.922 / 0.940 | 0.905 / 0.930 / 0.936 / 0.942 | 0.213 / 0.195 / 0.153 / 0.124 |
| truck | 1,193 | 0.8793 | 0.749 / 0.895 / 0.927 / 0.947 | 0.822 / 0.907 / 0.918 / 0.923 | 0.270 / 0.167 / 0.167 / 0.167 |
| bus | 336 | 0.9482 | 0.851 / 0.981 / 0.981 / 0.981 | 0.894 / 0.957 / 0.957 / 0.957 | 0.261 / 0.222 / 0.222 / 0.222 |
| bicycle | 740 | 0.8489 | 0.792 / 0.850 / 0.872 / 0.881 | 0.844 / 0.866 / 0.867 / 0.871 | 0.212 / 0.212 / 0.212 / 0.212 |
| pedestrian | 5,059 | 0.8598 | 0.844 / 0.858 / 0.865 / 0.872 | 0.841 / 0.849 / 0.852 / 0.854 | 0.161 / 0.165 / 0.165 / 0.165 |
| **ALL** | 22,211 | 0.8882 | — | — | — |

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.7132**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 10,994 | 0.8237 | 0.716 / 0.823 / 0.866 / 0.890 | 0.792 / 0.852 / 0.873 / 0.882 | 0.213 / 0.181 / 0.158 / 0.147 |
| truck | 1,011 | 0.7245 | 0.521 / 0.729 / 0.813 / 0.834 | 0.661 / 0.796 / 0.836 / 0.840 | 0.212 / 0.169 / 0.169 / 0.143 |
| bus | 143 | 0.7811 | 0.606 / 0.834 / 0.834 / 0.850 | 0.741 / 0.824 / 0.824 / 0.824 | 0.469 / 0.345 / 0.345 / 0.345 |
| bicycle | 463 | 0.5497 | 0.418 / 0.578 / 0.598 / 0.605 | 0.576 / 0.646 / 0.651 / 0.654 | 0.161 / 0.151 / 0.136 / 0.136 |
| pedestrian | 3,754 | 0.6871 | 0.668 / 0.686 / 0.692 / 0.703 | 0.694 / 0.704 / 0.707 / 0.712 | 0.128 / 0.128 / 0.128 / 0.128 |
| **ALL** | 16,365 | 0.7132 | — | — | — |

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.5202**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 3,018 | 0.6989 | 0.552 / 0.696 / 0.765 / 0.783 | 0.661 / 0.741 / 0.775 / 0.784 | 0.191 / 0.179 / 0.162 / 0.162 |
| truck | 602 | 0.6297 | 0.313 / 0.662 / 0.763 / 0.781 | 0.527 / 0.736 / 0.793 / 0.800 | 0.206 / 0.192 / 0.189 / 0.189 |
| bus | 60 | 0.4058 | 0.201 / 0.437 / 0.492 / 0.492 | 0.410 / 0.512 / 0.540 / 0.540 | 0.515 / 0.150 / 0.058 / 0.058 |
| bicycle | 85 | 0.3609 | 0.256 / 0.389 / 0.399 / 0.399 | 0.431 / 0.514 / 0.521 / 0.521 | 0.172 / 0.172 / 0.099 / 0.099 |
| pedestrian | 1,121 | 0.5056 | 0.489 / 0.504 / 0.509 / 0.521 | 0.597 / 0.606 / 0.609 / 0.612 | 0.125 / 0.125 / 0.125 / 0.125 |
| **ALL** | 4,886 | 0.5202 | — | — | — |

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.7995**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 28,895 | 0.8640 | 0.783 / 0.862 / 0.897 / 0.915 | 0.840 / 0.883 / 0.898 / 0.904 | 0.213 / 0.191 / 0.153 / 0.153 |
| truck | 2,806 | 0.7788 | 0.579 / 0.794 / 0.860 / 0.881 | 0.703 / 0.833 / 0.864 / 0.868 | 0.215 / 0.195 / 0.168 / 0.168 |
| bus | 539 | 0.8608 | 0.718 / 0.902 / 0.910 / 0.913 | 0.811 / 0.881 / 0.881 / 0.881 | 0.378 / 0.334 / 0.334 / 0.334 |
| bicycle | 1,288 | 0.7272 | 0.640 / 0.738 / 0.761 / 0.770 | 0.727 / 0.767 / 0.771 / 0.774 | 0.187 / 0.187 / 0.148 / 0.148 |
| pedestrian | 9,934 | 0.7669 | 0.749 / 0.765 / 0.772 / 0.781 | 0.758 / 0.767 / 0.771 / 0.775 | 0.146 / 0.139 / 0.138 / 0.140 |
| **ALL** | 43,462 | 0.7995 | — | — | — |

---

**J6Gen2**: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 + db_j6gen2_v6 + db_j6gen2_v7 + db_j6gen2_v8 + db_j6gen2_v9 (3,951 frames)

**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.8702**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 49,637 | 0.8758 | 0.818 / 0.869 / 0.897 / 0.919 | 0.879 / 0.910 / 0.920 / 0.927 | 0.266 / 0.194 / 0.158 / 0.116 |
| truck | 5,754 | 0.8410 | 0.711 / 0.832 / 0.893 / 0.927 | 0.786 / 0.856 / 0.890 / 0.910 | 0.215 / 0.170 / 0.170 / 0.157 |
| bus | 1,939 | 0.9408 | 0.864 / 0.935 / 0.979 / 0.984 | 0.902 / 0.941 / 0.960 / 0.963 | 0.201 / 0.133 / 0.133 / 0.033 |
| bicycle | 639 | 0.8590 | 0.841 / 0.865 / 0.865 / 0.865 | 0.860 / 0.871 / 0.871 / 0.871 | 0.163 / 0.155 / 0.155 / 0.155 |
| pedestrian | 14,362 | 0.8344 | 0.807 / 0.832 / 0.843 / 0.855 | 0.803 / 0.816 / 0.821 / 0.828 | 0.170 / 0.168 / 0.168 / 0.168 |
| **ALL** | 72,331 | 0.8702 | — | — | — |

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.6708**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 47,568 | 0.7721 | 0.629 / 0.764 / 0.832 / 0.864 | 0.736 / 0.816 / 0.850 / 0.865 | 0.230 / 0.177 / 0.152 / 0.144 |
| truck | 4,090 | 0.6421 | 0.439 / 0.620 / 0.732 / 0.777 | 0.599 / 0.714 / 0.771 / 0.790 | 0.191 / 0.191 / 0.191 / 0.191 |
| bus | 1,935 | 0.7731 | 0.540 / 0.754 / 0.886 / 0.912 | 0.648 / 0.786 / 0.861 / 0.876 | 0.229 / 0.128 / 0.104 / 0.069 |
| bicycle | 295 | 0.5472 | 0.485 / 0.564 / 0.567 / 0.572 | 0.629 / 0.676 / 0.676 / 0.676 | 0.145 / 0.145 / 0.145 / 0.168 |
| pedestrian | 6,529 | 0.6192 | 0.588 / 0.615 / 0.629 / 0.644 | 0.654 / 0.668 / 0.673 / 0.681 | 0.140 / 0.144 / 0.144 / 0.140 |
| **ALL** | 60,417 | 0.6708 | — | — | — |

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.4462**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 17,353 | 0.6346 | 0.437 / 0.619 / 0.722 / 0.761 | 0.595 / 0.698 / 0.750 / 0.770 | 0.182 / 0.157 / 0.155 / 0.140 |
| truck | 2,570 | 0.4758 | 0.184 / 0.409 / 0.609 / 0.701 | 0.401 / 0.569 / 0.690 / 0.739 | 0.195 / 0.138 / 0.137 / 0.130 |
| bus | 316 | 0.3215 | 0.075 / 0.308 / 0.432 / 0.471 | 0.284 / 0.464 / 0.538 / 0.556 | 0.078 / 0.058 / 0.059 / 0.059 |
| bicycle | 70 | 0.4303 | 0.402 / 0.438 / 0.440 / 0.440 | 0.574 / 0.591 / 0.591 / 0.591 | 0.193 / 0.193 / 0.193 / 0.193 |
| pedestrian | 1,673 | 0.3688 | 0.355 / 0.364 / 0.373 / 0.382 | 0.500 / 0.505 / 0.509 / 0.513 | 0.142 / 0.142 / 0.142 / 0.128 |
| **ALL** | 21,982 | 0.4462 | — | — | — |

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.7712**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 114,558 | 0.8110 | 0.699 / 0.803 / 0.857 / 0.886 | 0.780 / 0.842 / 0.869 / 0.880 | 0.230 / 0.180 / 0.158 / 0.141 |
| truck | 12,414 | 0.7129 | 0.522 / 0.688 / 0.795 / 0.847 | 0.654 / 0.757 / 0.815 / 0.841 | 0.215 / 0.191 / 0.162 / 0.155 |
| bus | 4,190 | 0.8348 | 0.669 / 0.822 / 0.915 / 0.933 | 0.750 / 0.841 / 0.888 / 0.897 | 0.231 / 0.137 / 0.104 / 0.113 |
| bicycle | 1,004 | 0.7458 | 0.714 / 0.754 / 0.757 / 0.758 | 0.777 / 0.798 / 0.798 / 0.799 | 0.161 / 0.170 / 0.170 / 0.170 |
| pedestrian | 22,564 | 0.7515 | 0.723 / 0.748 / 0.761 / 0.774 | 0.741 / 0.753 / 0.759 / 0.766 | 0.161 / 0.161 / 0.161 / 0.161 |
| **ALL** | 154,730 | 0.7712 | — | — | — |

---

**JPNTaxi_Gen2**: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (9,975 frames)

**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.8784**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 42,789 | 0.9436 | 0.892 / 0.946 / 0.965 / 0.972 | 0.916 / 0.950 / 0.958 / 0.959 | 0.237 / 0.173 / 0.144 / 0.144 |
| truck | 17,259 | 0.8531 | 0.696 / 0.840 / 0.926 / 0.950 | 0.803 / 0.880 / 0.930 / 0.943 | 0.298 / 0.195 / 0.169 / 0.169 |
| bus | 3,437 | 0.8284 | 0.712 / 0.818 / 0.889 / 0.894 | 0.758 / 0.823 / 0.866 / 0.868 | 0.024 / 0.024 / 0.024 / 0.024 |
| bicycle | 2,681 | 0.8546 | 0.842 / 0.857 / 0.859 / 0.860 | 0.864 / 0.870 / 0.871 / 0.871 | 0.243 / 0.243 / 0.229 / 0.229 |
| pedestrian | 57,948 | 0.9123 | 0.895 / 0.911 / 0.918 / 0.925 | 0.872 / 0.881 / 0.888 / 0.893 | 0.148 / 0.148 / 0.148 / 0.140 |
| **ALL** | 124,114 | 0.8784 | — | — | — |

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.6692**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 35,518 | 0.8323 | 0.747 / 0.835 / 0.867 / 0.880 | 0.810 / 0.852 / 0.866 / 0.871 | 0.232 / 0.167 / 0.159 / 0.157 |
| truck | 22,550 | 0.6571 | 0.424 / 0.616 / 0.763 / 0.824 | 0.598 / 0.715 / 0.797 / 0.825 | 0.235 / 0.193 / 0.155 / 0.126 |
| bus | 2,683 | 0.4033 | 0.129 / 0.358 / 0.545 / 0.583 | 0.303 / 0.484 / 0.601 / 0.624 | 0.042 / 0.044 / 0.044 / 0.044 |
| bicycle | 1,607 | 0.6721 | 0.636 / 0.682 / 0.685 / 0.686 | 0.723 / 0.743 / 0.743 / 0.743 | 0.172 / 0.172 / 0.172 / 0.172 |
| pedestrian | 27,240 | 0.7812 | 0.763 / 0.779 / 0.788 / 0.795 | 0.765 / 0.773 / 0.778 / 0.782 | 0.158 / 0.168 / 0.153 / 0.153 |
| **ALL** | 89,598 | 0.6692 | — | — | — |

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.5300**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 16,524 | 0.6692 | 0.544 / 0.678 / 0.719 / 0.737 | 0.655 / 0.731 / 0.749 / 0.754 | 0.168 / 0.160 / 0.159 / 0.140 |
| truck | 14,587 | 0.5020 | 0.192 / 0.444 / 0.624 / 0.748 | 0.419 / 0.598 / 0.709 / 0.771 | 0.268 / 0.203 / 0.163 / 0.145 |
| bus | 2,476 | 0.2822 | 0.113 / 0.279 / 0.350 / 0.387 | 0.349 / 0.453 / 0.493 / 0.514 | 0.022 / 0.025 / 0.025 / 0.025 |
| bicycle | 364 | 0.4586 | 0.345 / 0.428 / 0.530 / 0.530 | 0.525 / 0.562 / 0.602 / 0.602 | 0.151 / 0.128 / 0.151 / 0.151 |
| pedestrian | 14,297 | 0.7380 | 0.715 / 0.734 / 0.744 / 0.760 | 0.736 / 0.745 / 0.750 / 0.759 | 0.134 / 0.134 / 0.134 / 0.133 |
| **ALL** | 48,248 | 0.5300 | — | — | — |

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.7471**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 94,831 | 0.8667 | 0.788 / 0.870 / 0.898 / 0.910 | 0.835 / 0.879 / 0.891 / 0.894 | 0.214 / 0.168 / 0.160 / 0.159 |
| truck | 54,396 | 0.6928 | 0.460 / 0.655 / 0.794 / 0.862 | 0.624 / 0.744 / 0.823 / 0.855 | 0.285 / 0.199 / 0.173 / 0.155 |
| bus | 8,596 | 0.5446 | 0.351 / 0.528 / 0.637 / 0.662 | 0.489 / 0.598 / 0.660 / 0.673 | 0.027 / 0.027 / 0.027 / 0.028 |
| bicycle | 4,652 | 0.7710 | 0.744 / 0.775 / 0.782 / 0.783 | 0.791 / 0.804 / 0.808 / 0.809 | 0.184 / 0.184 / 0.184 / 0.184 |
| pedestrian | 99,485 | 0.8606 | 0.842 / 0.859 / 0.867 / 0.875 | 0.824 / 0.832 / 0.839 / 0.844 | 0.148 / 0.148 / 0.148 / 0.148 |
| **ALL** | 261,960 | 0.7471 | — | — | — |

</details>

---
