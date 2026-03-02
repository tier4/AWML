# Deployed model for CenterPoint base/2.X
## Summary

### Main Parameters

  - **Range:** 122.40m
  - **Voxel Size:** [0.24, 0.24, 8.0]
  - **Grid Size:** [510, 510, 1]

### Testing Datsets

- **Total Frames: 19,096**
  <details>
  <summary> jpntaxi (1,507 frames)</summary>

    - `db_jpntaxi_v1`
    - `db_jpntaxi_v2`
    - `db_jpntaxi_v4`

  </details>

  <details>
  <summary> j6 (2,435 frames) </summary>

    - `db_gsm8_v1`
    - `db_j6_v1`
    - `db_j6_v2`
    - `db_j6_v3`
    - `db_j6_v5`

  </details>

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
### mAP

- **Class mAP for BEV Center Distance: 0.5m, 1.0m, 2.0m, 4.0m**

  <details>
  <summary> Eval Range: 0.0 - 50.0m </summary>

  | Model version | mAP  | car <br> (145,766) | truck <br> (29,727) | bus <br> (7,196) | bicycle <br> (6,066) | pedestrian <br> (96,613) |
  | -------------------------| ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
  | CenterPoint base/2.6.0     | 82.78 | 93.09            | 80.26               | 83.25         | 81.71                 | 75.57                   |
  | CenterPoint base/2.5.0     | 83.55 | 92.84            | 80.81               | 87.24         | 81.00                 | 75.87                   |

  </details>

  <details>
  <summary>Eval Range: 50.0 - 90.0m </summary>

  | Model version         | mAP  | car <br> (124,669) | truck <br> (32,534) | bus <br> (6,772) | bicycle <br> (3,885) | pedestrian <br> (45,535) |
  | -------------------------| ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
  | CenterPoint base/2.6.0     | 62.37 | 78.88            | 58.26               | 61.21         | 51.57                 | 61.93                   |
  | CenterPoint base/2.5.0     | 62.50 | 78.82            | 57.85               | 62.60         | 51.59                 | 61.65                   |

  </details>

  <details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version         | mAP  | car <br> (46,890) | truck <br> (19,912) | bus <br> (3,159) | bicycle <br> (765) | pedestrian <br> (18,730) |
  | -------------------------| ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
  | CenterPoint base/2.6.0     | 44.42 | 54.25            | 42.15               | 38.53         | 33.66                 | 53.48                   |
  | CenterPoint base/2.5.0     | 45.38 | 54.28            | 42.17               | 40.40         | 35.50                 | 54.52                   |

  </details>

  <details open>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version         | mAP  | car <br> (317,325) | truck <br> (82,173) | bus <br> (17,127) | bicycle <br> (10,716) | pedestrian <br> (160,878) |
  | -------------------------| ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
  | CenterPoint base/2.6.0     | 70.56 | 83.21            | 63.38               | 67.60         | 68.58                 | 70.04                   |
  | CenterPoint base/2.5.0     | 70.99 | 83.10            | 63.45               | 69.98         | 68.20                 | 70.19                   |

  </details>

## Datasets

<details>
<summary> JPNTaxi Gen2 </summary>

- Datasets (9,975 Testing Frames):
  - `db_jpntaxigen2_v1`
  - `db_jpntaxigen2_v2`

- **Class mAP for BEV Center Distance: 0.5m, 1.0m, 2.0m, 4.0m**

  <details>
  <summary> Eval Range: 0.0 - 50.0m </summary>

  | Model version | mAP  | car <br> (42,789) | truck <br> (17,259) | bus <br> (3,437) | bicycle <br> (2,681) | pedestrian <br> (57,948) |
  | -------------------------| ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
  | CenterPoint base/2.6.0     | 81.76 | 92.84            | 82.05               | 76.26         | 80.29                 | 77.37                   |
  | CenterPoint base/2.5.0     | 82.94 | 92.15            | 82.84               | 82.43         | 79.78                 | 77.51                   |

  </details>

  <details>
  <summary>Eval Range: 50.0 - 90.0m </summary>

  | Model version         | mAP  | car <br> (35,518) | truck <br> (22,550) | bus <br> (2,683) | bicycle <br> (1,607) | pedestrian <br> (27,240) |
  | -------------------------| ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
  | CenterPoint base/2.6.0     | 63.63 | 81.94            | 59.84               | 43.03         | 66.74                 | 66.57                   |
  | CenterPoint base/2.5.0     | 64.04 | 81.77            | 59.81               | 44.33         | 68.45                 | 65.84                   |

  </details>

  <details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version         | mAP  | car <br> (16,524) | truck <br> (14,587) | bus <br> (2,476) | bicycle <br> (364) | pedestrian <br> (14,297) |
  | -------------------------| ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
  | CenterPoint base/2.6.0     | 49.21 | 56.59            | 43.87               | 43.69         | 39.66                 | 62.22                   |
  | CenterPoint base/2.5.0     | 50.51 | 56.79            | 43.75               | 45.56         | 43.25                 | 63.19                   |

  </details>

  <details open>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version         | mAP  | car <br> (94,831) | truck <br> (54,396) | bus <br> (8,596) | bicycle <br> (4,652) | pedestrian <br> (99,485) |
  | -------------------------| ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
  | CenterPoint base/2.6.0     | 70.41 | 83.82            | 63.59               | 57.56         | 74.15                 | 72.94                   |
  | CenterPoint base/2.5.0     | 71.01 | 83.46            | 63.85               | 60.51         | 74.26                 | 72.99                   |

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

  | Model version | mAP  | car <br> (14,883) | truck <br> (1,193) | bus <br> (336) | bicycle <br> (740) | pedestrian <br> (5,059) |
  | -------------------------| ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
  | CenterPoint base/2.6.0     | 82.57 | 95.27            | 81.09               | 86.74         | 76.63                 | 73.13                   |
  | CenterPoint base/2.5.0     | 82.61 | 95.25            | 81.86               | 87.57         | 76.37                 | 71.99                   |

  </details>

  <details>
  <summary>Eval Range: 50.0 - 90.0m </summary>

  | Model version         | mAP  | car <br> (10,994) | truck <br> (1,011) | bus <br> (143) | bicycle <br> (463) | pedestrian <br> (3,754) |
  | -------------------------| ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
  | CenterPoint base/2.6.0     | 65.96 | 86.11            | 59.83               | 87.18         | 40.58                 | 56.08                   |
  | CenterPoint base/2.5.0     | 65.84 | 85.70           |  60.40               | 89.16         | 38.85                 | 55.07                   |

  </details>

  <details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version         | mAP  | car <br> (3,018) | truck <br> (602) | bus <br> (60) | bicycle <br> (85) | pedestrian <br> (1,121) |
  | -------------------------| ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
  | CenterPoint base/2.6.0     | 45.26 | 62.34            | 52.57               | 44.14         | 27.94                 | 39.32                   |
  | CenterPoint base/2.5.0     | 45.46 | 60.74            | 55.47               | 45.60         | 25.22                 | 40.24                   |

  </details>

  <details open>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version         | mAP  | car <br> (28,895) | truck <br> (2,806) | bus <br> (539) | bicycle <br> (1,288) | pedestrian <br> (9,934) |
  | -------------------------| ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
  | CenterPoint base/2.6.0     | 73.04 | 89.32            | 68.01               | 82.34         | 61.84                 | 63.70                   |
  | CenterPoint base/2.5.0     | 73.25 | 89.04            | 68.99               | 84.14         | 61.13                 | 62.94                   |

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

  | Model version | mAP  | car <br> (49,637) | truck <br> (5,754) | bus <br> (1,939) | bicycle <br> (639) | pedestrian <br> (14,362) |
  | -------------------------| ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
  | CenterPoint base/2.6.0     | 83.81 | 94.16            | 79.64               | 92.11         | 80.97                 | 72.14                   |
  | CenterPoint base/2.5.0     | 83.62 | 94.15            | 79.17               | 93.26         | 79.43                 | 72.11                   |

  </details>

  <details>
  <summary>Eval Range: 50.0 - 90.0m </summary>

  | Model version         | mAP  | car <br> (47,569) | truck <br> (4,090) | bus <br> (1,935) | bicycle <br> (295) | pedestrian <br> (6,529) |
  | -------------------------| ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
  | CenterPoint base/2.6.0     | 63.22 | 77.72            | 56.82               | 73.15         | 56.58                 | 51.80                   |
  | CenterPoint base/2.5.0     | 63.04 | 77.66            | 54.32               | 72.95         | 57.93                 | 52.33                   |

  </details>

  <details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version         | mAP  | car <br> (17,353) | truck <br> (2,570) | bus <br> (316) | bicycle <br> (70) | pedestrian <br> (1,673) |
  | -------------------------| ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
  | CenterPoint base/2.6.0     | 36.41 | 52.91            | 36.55               | 30.06         | 37.39                 | 25.12                   |
  | CenterPoint base/2.5.0     | 37.12 | 52.98            | 35.78               | 29.73         | 41.69                 | 25.43                   |

  </details>

  <details open>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version         | mAP  | car <br> (114,558) | truck <br> (12,414) | bus <br> (4,190) | bicycle <br> (1,004) | pedestrian <br> (22,564) |
  | -------------------------| ---- | ----------------- | ------------------- | ---------------- | -------------------- | ------------------------ |
  | CenterPoint base/2.6.0     | 72.60 | 82.50            | 64.43               | 80.47         | 71.55                 | 64.05                   |
  | CenterPoint base/2.5.0     | 72.35 | 82.48            | 63.44               | 80.78         | 70.97                 | 64.10                   |

  </details>

</details>

## Release

### CenterPoint base/2.6.0

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
- Train with 8 GPUs instead of 4 GPUs, and thus, it increases the effective batch size from `64` to `128`

</details>

<details>
<summary> Artifacts </summary>

- Deployed onnx and ROS parameter files (for internal)
  - [WebAuto]()
  - [model-zoo]()
  - [Google drive]()
- Logs (for internal)
  - [model-zoo]()
  - [Google drive]()
- Pytorch Best checkpoints:
  - [model-zoo]()
  - [Google drive]()

</details>

<details>
<summary> Training configs </summary>

- [Config file path]()
- Train time: NVIDIA H100 80GB * 8 * 50 epochs = 3 days 15 hours
- Batch size: 8*16 = 128
- Training Dataset (frames: 134,554):
  - jpntaxi: db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 (25,958 frames)
  - j6: db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 (24,756 frames)
  - j6gen2: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 + db_j6gen2_v6 + db_j6gen2_v7 + db_j6gen2_v8 (37,002 frames)
  - largebus: db_largebus_v1 + db_largebus_v2 (11,106 frames)
  - jpntaxi_gen2: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (24,992 frames)

</details>

<details>
<summary> Evaluation </summary>

**Base Datasets (19,096 frames)**:

  - jpntaxi (1,507 frames): db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4
  - j6 (2,435 frames): db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5
  - j6gen2 (3,951 frames): db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 + db_j6gen2_v6 + db_j6gen2_v7 + db_j6gen2_v8 + db_j6gen2_v9
  - largebus (1,228 frames): db_largebus_v1 + db_largebus_v2 + db_largebus_v3
  - jpntaxi_gen2 (9,975 frames): db_jpntaxigen2_v1 + db_jpntaxigen2_v2

**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.8355**

| class_name | Count | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | Optimal_conf@0.5/1.0/2.0/4.0 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **car** | 145,766 | 0.9284 | 0.8876 / 0.9394 / 0.9428 / 0.9439 | 0.9083 / 0.9599 / 0.9604 / 0.9613 | 0.4246 / 0.3990 / 0.3736 / 0.3736 |
| **truck** | 29,727 | 0.8081 | 0.6600 / 0.8219 / 0.8680 / 0.8824 | 0.7375 / 0.8327 / 0.8734 / 0.8863 | 0.4177 / 0.3844 / 0.3802 / 0.3778 |
| **bus** | 7,196 | 0.8724 | 0.8037 / 0.8791 / 0.9021 / 0.9045 | 0.8449 / 0.8930 / 0.9025 / 0.9037 | 0.4656 / 0.4138 / 0.3991 / 0.3991 |
| **bicycle** | 6,066 | 0.8100 | 0.7982 / 0.8135 / 0.8139 / 0.8143 | 0.8661 / 0.8732 / 0.8734 / 0.8734 | 0.5081 / 0.5081 / 0.5081 / 0.5081 |
| **pedestrian** | 99,613 | 0.7587 | 0.7376 / 0.7497 / 0.7645 / 0.7831 | 0.7984 / 0.8030 / 0.8076 / 0.8126 | 0.4285 / 0.4285 / 0.4285 / 0.4139 |

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.6250**

| class_name | Count | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | Optimal_conf@0.5/1.0/2.0/4.0 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **car** | 124,669 | 0.7882 | 0.7089 / 0.7962 / 0.8195 / 0.8283 | 0.7730 / 0.8224 / 0.8357 / 0.8390 | 0.4121 / 0.3790 / 0.3733 / 0.3650 |
| **truck** | 32,534 | 0.5785 | 0.4237 / 0.5729 / 0.6411 / 0.6761 | 0.5733 / 0.6629 / 0.7149 / 0.7398 | 0.3960 / 0.3618 / 0.3614 / 0.3505 |
| **bus** | 6,772 | 0.6260 | 0.4921 / 0.6305 / 0.6810 / 0.7004 | 0.6066 / 0.6930 / 0.7931 / 0.7511 | 0.3791 / 0.3689 / 0.3690 / 0.3690 |
| **bicycle** | 3,885 | 0.5159 | 0.4835 / 0.5239 / 0.5246 / 0.5315 | 0.6807 / 0.6856 / 0.6925 / 0.6987 | 0.4202 / 0.4158 / 0.4158 / 0.4033 |
| **pedestrian** | 45,535 | 0.6165 | 0.5956 / 0.6073 / 0.6233 / 0.639 | 0.6807 / 0.6856 / 0.6925 / 0.6987 | 0.4202 / 0.4158 / 0.4158 / 0.4033 |

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.4538**

| class_name | Count | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | Optimal_conf@0.5/1.0/2.0/4.0 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **car** | 46,890 | 0.5428 | 0.4440 / 0.5522 / 0.5803 / 0.5947 | 0.5791 / 0.6445 / 0.6601 / 0.6654 | 0.3675 / 0.3524 / 0.3478 / 0.3434 |
| **truck** | 19,912 | 0.4217 | 0.2336 / 0.4183 / 0.4929 / 0.5421 | 0.4269 / 0.5485 / 0.6009 / 0.6353 | 0.3657 / 0.3403 / 0.3377 / 0.3309 |
| **bus** | 3,159 | 0.4040 | 0.2489 / 0.4272 / 0.4665 / 0.4736 | 0.4342 / 0.5741 / 0.6104 / 0.6171 | 0.3862 / 0.3786 / 0.3730 / 0.3743 |
| **bicycle** | 765 | 0.3550 | 0.2867 / 0.3424 / 0.3955 / 0.3955 | 0.4933 / 0.5347 / 0.5604 / 0.5604 | 0.3621 / 0.3414 / 0.3414 / 0.3414 |
| **pedestrian** | 18,730 | 0.5452 | 0.5251 / 0.5403 / 0.5490 / 0.5664 | 0.6324 / 0.6397 / 0.6430 / 0.6510 | 0.3590 / 0.3590 / 0.3590 / 0.3519 |

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.7099**

| class_name | Count | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | Optimal_conf@0.5/1.0/2.0/4.0 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **car** | 145,766 | 0.8310 | 0.7645 / 0.8390 / 0.8581 / 0.8626 | 0.8145 / 0.8569 / 0.8671 / 0.8694 | 0.4094 / 0.3802 / 0.3732 / 0.3657 |
| **truck** | 29,727 | 0.6345 | 0.4708 / 0.6372 / 0.6994 / 0.7307 | 0.6079 / 0.7061 / 0.7542 / 0.7767 | 0.4007 / 0.3686 / 0.3590 / 0.3579 |
| **bus** | 7,196 | 0.6998 | 0.5889 / 0.7109 / 0.7446 / 0.7548  | 0.6894 / 0.7651 / 0.7933 / 0.7995 | 0.4339 / 0.3785 / 0.3779 / 0.3778 |
| **bicycle** | 6,066 | 0.6820 | 0.6573 / 0.6879 / 0.6909 / 0.6920 | 0.7710 / 0.7840 / 0.7863 / 0.7872 | 0.4221 / 0.4244 / 0.4244 / 0.4221 |
| **pedestrian** | 99,613 | 0.7019 | 0.6810 / 0.6949 / 0.7070 / 0.7248 | 0.7457 / 0.7507 / 0.7560 / 0.7619 | 0.4141 / 0.4068 / 0.4068 / 0.4068 |

---

**LargeBus**: db_largebus_v1 + db_largebus_v2 + db_largebus_v3 (1,228 frames)  
**Total mAP (eval range = 120m): 0.717**

| class_name | Count  | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | ------ | ---- | ---- | ---- | ---- | ---- |
| car        | 16,604 | 90.1 | 85.0 | 90.5 | 92.3 | 92.6 |
| truck      | 1,961  | 64.0 | 52.2 | 64.3 | 68.6 | 70.7 |
| bus        | 171    | 69.3 | 49.3 | 74.8 | 76.5 | 76.5 |
| bicycle    | 863    | 67.7 | 63.7 | 68.8 | 69.1 | 69.1 |
| pedestrian | 4,659  | 67.7 | 66.0 | 67.3 | 68.0 | 69.5 |

---

**J6Gen2**: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 + db_j6gen2_v6  
**Total mAP (eval range = 120m): 0.7340**

| class_name | Count   | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | ------- | ---- | ---- | ---- | ---- | ---- |
| car        | 66,293  | 85.0 | 78.4 | 85.5 | 87.9 | 88.3 |
| truck      | 4,417   | 54.1 | 43.2 | 52.4 | 56.3 | 64.4 |
| bus        | 2,353   | 82.9 | 75.5 | 82.6 | 86.3 | 87.1 |
| bicycle    | 500     | 76.8 | 75.4 | 77.2 | 77.2 | 77.2 |
| pedestrian | 11,417  | 68.0 | 66.3 | 67.5 | 68.3 | 69.8 |

---

**JPNTaxi_Gen2**: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (1,709 frames)  
**Total mAP (eval range = 120m): 0.607**

| class_name | Count  | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | ------ | ---- | ---- | ---- | ---- | ---- |
| car        | 9,710  | 86.1 | 78.1 | 87.6 | 88.9 | 89.7 |
| truck      | 2,577  | 41.0 | 30.0 | 38.5 | 41.6 | 53.9 |
| bus        | 2,569  | 58.2 | 39.1 | 58.8 | 66.7 | 68.4 |
| bicycle    | 466    | 45.7 | 36.3 | 48.2 | 48.9 | 49.2 |
| pedestrian | 10,518 | 72.3 | 70.0 | 71.3 | 72.8 | 75.0 |

</details>

---

### CenterPoint base/2.5

<details>
<summary> Changes </summary>

- Voxelization increase from `0.20` to `0.24`
- Adjust `[x, y]` range of pointclouds from `[-121.60, 121.60]` to `[-122.40, 122.40]`
- Train with Repeat Sampling Factor (RFS) and low pedestrians (< 1.5m and distance < 50m)
- Reduce the grid size of final heatmaps from `[608, 608]` to `[510, 510]`
- Reduce the maximum size of training and evaluation pointclous from `128000` to `96000`
- Train more data with:
    - `db_largebus_v1`
    - `db_largebus_v2`
    - `db_j6gen2_v1`
    - `db_j6gen2_v2`
    - `db_j6gen2_v4`
    - `db_j6gen2_v5`
    - `db_jpntaxi_gen2_v1`
    - `db_jpntaxi_gen2_v2`
- Train with new datatasets:
    - `db_j6gen2_v6`
    - `db_j6gen2_v7`
    - `db_j6gen2_v8`

</details>

<details>
<summary> Artifacts </summary>

- Deployed onnx and ROS parameter files (for internal)
  - [WebAuto](https://evaluation.tier4.jp/evaluation/mlpackages/7156b453-2861-4ae9-b135-e24e48cc9029/releases/2042f2d3-06f9-48e6-b20e-0ded2843df91?project_id=zWhWRzei)
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.5.0/t4base_deployment.zip)
  - [Google drive](https://drive.google.com/file/d/1o2xroIwhYMTkfPIzdOnzTx72ICCZ3kxT/view?usp=drive_link)
- Logs (for internal)
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.5.0/logs.zip)
  - [Google drive](https://drive.google.com/file/d/1KmbDB5X3fjREoIRhfaBRMo1z7dY915J2/view?usp=drive_link)

</details>

<details>
<summary> Training configs </summary>

- [Config file path](https://github.com/tier4/AWML/blob/e2bc8a2da0ea8db296314efb51d420e550fb7790/projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp_rfs.py)
- Train time: NVIDIA H100 80GB * 4 * 50 epochs = 7 days
- Batch size: 4*16 = 64
- Training Dataset (frames: 123,708):
  - jpntaxi: db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 (25,958 frames)
  - j6: db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 (24,756 frames)
  - j6gen2: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 + db_j6gen2_v6 + db_j6gen2_v7 + db_j6gen2_v8 (37,002 frames)
  - largebus: db_largebus_v1 + db_largebus_v2 (11,106 frames)
  - jpntaxi_gen2: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (24,992 frames)

</details>

<details>
<summary> Evaluation </summary>

**Base Datasets (8,453 frames)**:
   - jpntaxi: db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 (1,507 frames)
   - j6: db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 (2,435 frames)
   - j6gen2: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 + db_j6gen2_v6 (1,943 frames)
   - largebus: db_largebus_v1 + db_largebus_v2 (859 frames)
   - jpntaxi_gen2: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (1,709 frames)

**Total mAP (eval range = 120m): 0.6870**

| class_name | Count    | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | -------- | ---- | ---- | ---- | ---- | ---- |
| car        | 171,648  | 83.9 | 76.9 | 84.7 | 86.8 | 87.2 |
| truck      | 21,415   | 54.2 | 38.3 | 52.8 | 59.3 | 66.2 |
| bus        | 8,895    | 73.3 | 62.2 | 73.9 | 78.0 | 79.2 |
| bicycle    | 5,601    | 63.1 | 60.5 | 63.4 | 64.0 | 64.2 |
| pedestrian | 55,486   | 68.9 | 66.8 | 68.2 | 69.5 | 71.3 |

---

**LargeBus**: db_largebus_v1 + db_largebus_v2 (859 frames)  
**Total mAP (eval range = 120m): 0.717**

| class_name | Count  | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | ------ | ---- | ---- | ---- | ---- | ---- |
| car        | 16,604 | 90.1 | 85.0 | 90.5 | 92.3 | 92.6 |
| truck      | 1,961  | 64.0 | 52.2 | 64.3 | 68.6 | 70.7 |
| bus        | 171    | 69.3 | 49.3 | 74.8 | 76.5 | 76.5 |
| bicycle    | 863    | 67.7 | 63.7 | 68.8 | 69.1 | 69.1 |
| pedestrian | 4,659  | 67.7 | 66.0 | 67.3 | 68.0 | 69.5 |

---

**J6Gen2**: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 + db_j6gen2_v6  
**Total mAP (eval range = 120m): 0.7340**

| class_name | Count   | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | ------- | ---- | ---- | ---- | ---- | ---- |
| car        | 66,293  | 85.0 | 78.4 | 85.5 | 87.9 | 88.3 |
| truck      | 4,417   | 54.1 | 43.2 | 52.4 | 56.3 | 64.4 |
| bus        | 2,353   | 82.9 | 75.5 | 82.6 | 86.3 | 87.1 |
| bicycle    | 500     | 76.8 | 75.4 | 77.2 | 77.2 | 77.2 |
| pedestrian | 11,417  | 68.0 | 66.3 | 67.5 | 68.3 | 69.8 |

---

**JPNTaxi_Gen2**: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (1,709 frames)  
**Total mAP (eval range = 120m): 0.607**

| class_name | Count  | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | ------ | ---- | ---- | ---- | ---- | ---- |
| car        | 9,710  | 86.1 | 78.1 | 87.6 | 88.9 | 89.7 |
| truck      | 2,577  | 41.0 | 30.0 | 38.5 | 41.6 | 53.9 |
| bus        | 2,569  | 58.2 | 39.1 | 58.8 | 66.7 | 68.4 |
| bicycle    | 466    | 45.7 | 36.3 | 48.2 | 48.9 | 49.2 |
| pedestrian | 10,518 | 72.3 | 70.0 | 71.3 | 72.8 | 75.0 |

</details>

---

### CenterPoint base/2.4

<details>
<summary> Main changes </summary>

- Decrease voxelization size from `0.32` to `0.20`
- Train with Repeat Sampling Factor (RFS) and low pedestrians (< 1.5m and distance < 50m)
- Enable `activation_checkpointing` for `backbone` to decrease GPU memory while training

</details>

<details>
<summary> Artifacts </summary>

- Deployed onnx and ROS parameter files (for internal)
  - [WebAuto](https://evaluation.tier4.jp/evaluation/mlpackages/7156b453-2861-4ae9-b135-e24e48cc9029/releases/c140a3f3-2cca-476d-8afa-56f4bad12a5e?project_id=zWhWRzei)
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.3.0/deployment.zip)
  - [Google drive](https://drive.google.com/drive/u/0/folders/1QNJ1Xmz54oPZqHusvBH0-2CMGmKBhEO9)
- Logs (for internal)
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.3.0/logs.zip)
  - [Google drive](https://drive.google.com/drive/u/0/folders/11cUJ883kcHWwQ9DHpizB_d9MvJa5iujr)

</details>

<details>
<summary> Training configs </summary>

- [Config file path](https://github.com/tier4/AWML/blob/eca15d56558c6aba2b6ea337e1e4a3ead028b900/projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp_high_resolution_rfs.py)
- Train time: NVIDIA H100 80GB * 4 * 50 epochs = 6 days
- Batch size: 4*16 = 64
- Training Datasets (frames: 99,776):
  - jpntaxi: db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 (26,100 frames)
  - j6: db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 (24,756 frames)
  - j6gen2: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 (21,077 frames)
  - largebus: db_largebus_v1 + db_largebus_v2 (9,213 frames)
  - jpntaxi_gen2: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (18,630 frames)

</details>

<details>
<summary> Evaluation </summary>

- Datasets (frames: 7,727):
  - jpntaxi: db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 (1,507 frames)
  - j6: db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 (2,435 frames)
  - j6gen2: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 (1,217 frames)
  - largebus: db_largebus_v1 + db_largebus_v2 (859 frames)
  - jpntaxi_gen2: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (1,709 frames)
- Total mAP (eval range = 120m): 0.672

| class_name | Count         | mAP       | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | --------------| --------  |----     | ----    | ----    | ----    |
| car        | 145,580       | 82.5      | 75.6    | 83.4    | 85.3    | 85.8    |
| truck      |  19,670       | 52.7      | 37.1    | 53.1    | 57.4    | 63.3    |
| bus        |   7,930       | 69.5      | 56.5    | 70.3    | 74.8    | 76.6    |
| bicycle    |   5,404       | 62.2      | 60.2    | 62.6    | 62.7    | 63.3    |
| pedestrian |  50,365       | 68.9      | 67.1    | 68.2    | 69.3    | 71.0    |

- db_largebus_v1 + db_largebus_v2 (859 frames):
  - Total mAP (eval range = 120m): 0.713

| class_name | Count        | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | -------------| ---- | ---- | ---- | ---- | ---- |
| car        | 16,604       | 89.3 | 84.8    | 89.9    | 91.2    | 91.5    |
| truck      |  1,961       | 62.4 | 49.8    | 62.6    | 67.5    | 69.7    |
| bus        |    171       | 68.3 | 51.6    | 73.5    | 74.1    | 74.1    |
| bicycle    |    863       | 69.0 | 65.2    | 70.1    | 70.4    | 70.4    |
| pedestrian |   4,659      | 67.4 | 65.9    | 67.0    | 67.7    | 69.0    |

- j6gen2: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 (1,217 frames):
  - Total mAP (eval range = 120m): 0.743

| class_name | Count          | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | ---------------| ---- | ---- | ---- | ---- | ---- |
| car        | 40,225         | 83.0 | 76.9    | 83.1    | 85.5    | 86.5    |
| truck      |  2,672         | 50.9 | 41.5    | 50.1    | 52.9    | 58.9    |
| bus        |  1,388         | 86.1 | 79.5    | 85.4    | 89.6    | 89.7    |
| bicycle    |    303         | 83.8 | 82.9    | 84.0    | 84.0    | 84.4    |
| pedestrian |   6,296        | 67.9 | 66.4    | 67.2    | 68.3    | 69.8    |

- db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (1,709 frames):
  - Total mAP (eval range = 120m): 0.585

| class_name | Count      | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | ---------- | ---- | ---- | ---- | ---- | ---- |
| car        |  9,710     | 85.0 | 78.1    | 86.4    | 87.4    | 88.1    |
| truck      |  2,577     | 40.1 | 31.1    | 37.6    | 40.7    | 50.8    |
| bus        |  2,569     | 53.5 | 34.0    | 53.8    | 62.0    | 64.5    |
| bicycle    |    466     | 42.8 | 38.0    | 44.1    | 44.3    | 44.7    |
| pedestrian |  10,518    | 71.2 | 69.7    | 70.5    | 71.4    | 73.4    |

</details>

---

### CenterPoint base/2.3

<details>
<summary> Changes </summary>

- Add a new training set: `db_jpntaxigen2_v2`, `db_j6gen2_v3`, `db_j6gen2_v5`, and `db_largebus_v2`
- Add new data to `db_j6gen2_v4`, `db_largebus_v1`

</details>

<details>
<summary> Artifacts </summary>

- Deployed onnx and ROS parameter files (for internal)
  - [WebAuto](https://evaluation.tier4.jp/evaluation/mlpackages/7156b453-2861-4ae9-b135-e24e48cc9029/releases/c140a3f3-2cca-476d-8afa-56f4bad12a5e?project_id=zWhWRzei)
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.3.0/deployment.zip)
  - [Google drive](https://drive.google.com/drive/u/0/folders/1QNJ1Xmz54oPZqHusvBH0-2CMGmKBhEO9)
- Logs (for internal)
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.3.0/logs.zip)
  - [Google drive](https://drive.google.com/drive/u/0/folders/11cUJ883kcHWwQ9DHpizB_d9MvJa5iujr)

</details>

<details>
<summary> Training configs </summary>

- [Config file path](https://github.com/tier4/AWML/blob/c0ba7268f110062f71ee80a3469102867a63b740/projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp.py)
- Train time: NVIDIA H100 80GB * 4 * 50 epochs = 4 days
- Batch size: 4*16 = 64
- Training Dataset (frames: 99,776):
  - jpntaxi: db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 (26,100 frames)
  - j6: db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 (24,756 frames)
  - j6gen2: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 (21,077 frames)
  - largebus: db_largebus_v1 + db_largebus_v2 (9,213 frames)
  - jpntaxi_gen2: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (18,630 frames)

</details>

<details>
<summary> Evaluation </summary>

- Datasets (frames: 7,727):
  - jpntaxi: db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 (1,507 frames)
  - j6: db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 (2,435 frames)
  - j6gen2: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 (1,217 frames)
  - largebus: db_largebus_v1 + db_largebus_v2 (859 frames)
  - jpntaxi_gen2: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (1,709 frames)
- Total mAP (eval range = 120m): 0.668

| class_name | Count        | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | ------------ | ---- | ---- | ---- | ---- | ---- |
| car        |  145,580     | 83.0 | 76.1    | 83.9    | 85.4    | 86.3    |
| truck      |   19,670     | 52.6 | 38.5    | 52.5    | 56.7    | 62.9    |
| bus        |    7,930     | 72.1 | 60.9    | 72.9    | 76.6    | 77.8    |
| bicycle    |    5,404     | 58.6 | 56.6    | 58.7    | 59.5    | 59.6    |
| pedestrian |   50,365     | 67.6 | 65.5    | 66.7    | 68.1    | 70.0    |

- db_largebus_v1 + db_largebus_v2 (859 frames):
  - Total mAP (eval range = 120m): 0.703

| class_name | Count        | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | -------------| ---- | ---- | ---- | ---- | ---- |
| car        | 16,604       | 89.5 | 84.4    | 90.3    | 91.6    | 91.8    |
| truck      |  1,961       | 62.9 | 52.0    | 63.6    | 66.8    | 69.3    |
| bus        |    171       | 67.9 | 43.4    | 75.7    | 76.2    | 76.2    |
| bicycle    |    863       | 64.2 | 59.0    | 65.3    | 66.2    | 66.3    |
| pedestrian |   4,659      | 66.8 | 65.1    | 66.4    | 67.3    | 68.6    |

- j6gen2: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 (1,217 frames):
  - Total mAP (eval range = 120m): 0.727

| class_name | Count          | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | ---------------| ---- | ---- | ---- | ---- | ---- |
| car        | 40,225         | 83.6 | 77.3    | 83.8    | 86.1    | 87.1    |
| truck      |  2,672         | 51.6 | 41.5    | 50.1    | 53.7    | 61.1    |
| bus        |  1,388         | 85.6 | 80.6    | 85.4    | 87.9    | 88.7    |
| bicycle    |    303         | 77.5 | 77.0    | 77.5    | 77.5    | 78.1    |
| pedestrian |   6,296        | 65.4 | 63.8    | 64.7    | 65.8    | 67.2    |

- db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (1,709 frames):
  - Total mAP (eval range = 120m): 0.5701

| class_name | Count      | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | ---------- | ---- | ---- | ---- | ---- | ---- |
| car        |  9,710     | 87.0 | 80.8    | 88.3    | 89.2    | 89.6    |
| truck      |  2,577     | 38.8 | 27.9    | 36.9    | 39.8    | 50.6    |
| bus        |  2,569     | 62.4 | 43.5    | 62.9    | 70.6    | 72.6    |
| bicycle    |    466     | 44.7 | 38.3    | 46.1    | 46.8    | 47.4    |
| pedestrian |  10,518    | 70.5 | 68.6    | 69.5    | 70.8    | 73.0    |

</details>

---

### CenterPoint base/2.2

<details>
<summary> Changes </summary>

- Add a new training set: `db_jpntaxi_gen2_v1`

</details>

<details>
<summary> Artifacts </summary>

- Deployed onnx and ROS parameter files (for internal)
  - [WebAuto](https://evaluation.tier4.jp/evaluation/mlpackages/7156b453-2861-4ae9-b135-e24e48cc9029/releases/83ba5207-44c3-46fc-899e-30863dcf1423?project_id=zWhWRzei)
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.2.0/deployment.zip)
  - [Google drive](https://drive.google.com/file/d/1v5rJqrv9vmM3RHD-lDSemcrxYJuhiQla/view?usp=drive_link)
- Logs (for internal)
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.2.0/logs.zip)
  - [Google drive](https://drive.google.com/file/d/1aFFA9WRd2G_eqqwVI92jbKQ-daAIJqD0/view?usp=drive_link)

</details>

<details>
<summary> Training configs </summary>

- [Config file path](http://github.com/tier4/AWML/blob/81314d29d4efa560952324c48ef7c0ea1e56f1ee/projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp.py)
- Train time: NVIDIA H100 80GB * 4 * 50 epochs = 4 days
- Batch size: 4*16 = 64
- Training dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v4.0 + DB GSM8 v1.0 + DB J6 v1.0 + DB J6 v2.0 + DB J6 v3.0 + DB J6 v5.0 + DB J6 Gen2 v1.0 + DB J6 Gen2 v2.0 + DB J6 Gen2 v4.0 + DB LargeBus v1.0 + DB JPNTAXI_GEN2 v1.0 (total frames: 88,762)

</details>

<details>
<summary> Evaluation </summary>

- db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 + db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 + db_j6gen2_v1 + db_j6gen2_v1 + db_j6gen2_v4 + db_largebus_v1 + db_jpntaxi_gen2_v1 (total frames: 7,182)
- Total mAP (eval range = 120m): 0.66

| class_name | Count     | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | ---       | ---- | ---- | ---- | ---- | ---- |
| car        |  152,572  | 82.6 | 75.3    | 83.7    | 85.3    | 86.2    |
| truck      |   24,172  | 52.5 | 35.3    | 53.0    | 57.5    | 64.4    |
| bus        |    5,691  | 71.9 | 61.2    | 72.3    | 76.4    | 77.6    |
| bicycle    |    5,317  | 56.4 | 54.8    | 56.3    | 57.1    | 57.2    |
| pedestrian |   50,699  | 66.5 | 64.5    | 65.6    | 67.0    | 68.7    |

- db_largebus_v1 (total frames: 604):
  - Total mAP (eval range = 120m): 0.7170

| class_name | Count    | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | -------  | ---- | ---- | ---- | ---- | ---- |
| car        |  13,831  | 88.6 | 83.4    | 89.5    | 90.7    | 90.9    |
| truck      |  2,137   | 63.6 | 51.1    | 64.5    | 68.2    | 70.6    |
| bus        |     95   | 78.7 | 76.3    | 79.5    | 79.5    | 79.5    |
| bicycle    |    724   | 64.1 | 59.8    | 64.9    | 65.8    | 65.8    |
| pedestrian |  3,916   | 63.4 | 61.4    | 63.0    | 63.9    | 65.4    |

- db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v2 (total frames: 1,157):
  - Total mAP (eval range = 120m): 0.7240

| class_name | Count   | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | ------  | ---- | ---- | ---- | ---- | ---- |
| car        | 44,008  | 83.6 | 77.3    | 83.8    | 86.1    | 87.1    |
| truck      |  2,471  | 54.4 | 43.8    | 54.5    | 56.8    | 62.4    |
| bus        |  1,464  | 84.0 | 79.3    | 83.4    | 86.7    | 86.7    |
| bicycle    |    333  | 74.3 | 73.7    | 74.5    | 74.5    | 74.5    |
| pedestrian |  6,459  | 65.7 | 64.3    | 65.1    | 65.8    | 67.6    |

- db_jpntaxi_gen2_v1 (total frames: 1,479):
  - Total mAP (eval range = 120m): 0.5701

| class_name | Count | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----       | ------| ---- | ---- | ---- | ---- | ---- |
| car        | 8,571 | 86.3 | 80.9    | 87.3    | 88.3    | 88.6    |
| truck      | 3,349 | 35.5 | 24.4    | 32.0    | 34.8    | 50.7    |
| bus        | 4,148 | 60.8 | 39.2    | 59.9    | 71.2    | 72.7    |
| bicycle    |   310 | 30.5 | 30.0    | 30.6    | 30.6    | 30.6    |
| pedestrian | 8,665 | 72.1 | 70.4    | 71.1    | 72.3    | 74.5    |

</details>

---

### CenterPoint base/2.1

<details>
<summary> Changes </summary>

- Add more training data to `db_j6gen2_v2` and `db_j6gen2_v4`
- Overall:
  - Slightly worse overall (-0.21 mAP).
  - Main improvement comes from `LargeBus` (71.85 vs 71.09) and `J6Gen2` (71.83 vs 71.04).
</details>

<details>
<summary> Artifacts </summary>

- Deployed onnx model and ROS parameter files [[WebAuto (for internal)]](https://evaluation.tier4.jp/evaluation/mlpackages/7156b453-2861-4ae9-b135-e24e48cc9029/releases/b73806f0-404f-4e9c-8b83-d9beb0c66ebd?project_id=zWhWRzei)
- Deployed onnx and ROS parameter files [[model-zoo]]
  - [detection_class_remapper.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.1.0/detection_class_remapper.param.yaml)
  - [centerpoint_ml_package.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.1.0/centerpoint_ml_package.param.yaml)
  - [deploy_metadata.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.1.0/deploy_metadata.yaml)
  - [pts_voxel_encoder_centerpoint.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.1.0/pts_voxel_encoder_centerpoint.onnx)
  - [pts_backbone_neck_head_centerpoint.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.1.0/pts_backbone_neck_head_centerpoint.onnx)
- Training results [[Google drive (for internal)]](https://drive.google.com/drive/u/0/folders/1FNw3bEvM1Z9Igp-uUzvXQFjLObONfwyg)
- Training results [model-zoo]
  - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.1.0/logs.zip)
  - [checkpoint_best.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.1.0/best_NuScenes_metric_T4Metric_mAP_epoch_49.pth)
  - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.1.0/second_secfpn_4xb16_121m_base_amp.py)

</details>

<details>
<summary> Training configs </summary>

- [Config file path](https://github.com/tier4/AWML/blob/69aba0d001fd26282880a7a3e7622b89115042de/autoware_ml/configs/detection3d/dataset/t4dataset/base.py)
- Train time: NVIDIA H100 80GB * 4 * 50 epochs = 2 days and 23 hours
- Batch size: 4*16 = 64
- Training dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v4.0 + DB GSM8 v1.0 + DB J6 v1.0 + DB J6 v2.0 + DB J6 v3.0 + DB J6 v5.0 + DB J6 Gen2 v1.0 + DB J6 Gen2 v2.0 + DB J6 Gen2 v4.0 + DB LargeBus v1.0 (total frames: 75,963)

</details>

<details>
<summary> Evaluation </summary>

- db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 + db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 + db_j6gen2_v1 + db_j6gen2_v1 + db_j6gen2_v4 + db_largebus_v1 (total frames: 5,703):
- Total mAP (eval range = 120m): 0.678

| class_name | Count    | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| -----------| -------  | ----  | ------- | ------- | ------- | ------- |
| car        |  144,001 | 82.3 | 75.1    | 83.0    | 85.2    | 86.1    |
| truck      |  20,823  | 55.2 | 40.0    | 55.5    | 59.8    | 65.4    |
| bus        |   5,691  | 78.1 | 70.7    | 79.0    | 80.8    | 81.8    |
| bicycle    |   5,007  | 57.4 | 56.3    | 57.7    | 57.7    | 57.8    |
| pedestrian |  42,034  | 66.3 | 64.0    | 65.4    | 66.9    | 68.8    |

- db_largebus_v1 (total frames: 604):
  - Total mAP (eval range = 120m): 0.7190

| class_name | Count    | mAP    | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| -----------| -------  | -----  | ------- | ------- | ------- | ------- |
| car        |  13,831   | 88.9 | 83.6    | 89.7    | 91.0    | 91.1    |
| truck      |  2,137   | 64.7 | 51.3    | 65.2    | 69.8    | 72.3    |
| bus        |     95   | 80.4 | 77.9    | 81.2    | 81.2    | 81.2    |
| bicycle    |    724   | 62.2 | 57.8    | 63.0    | 63.9    | 63.9    |
| pedestrian |  3,916   | 63.2 | 61.4    | 62.7    | 63.6    | 65.1    |

- db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v2 (total frames: 1,157):
  - Total mAP (eval range = 120m): 0.7180

| class_name  | Count   | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----------  | ------  | ---- | ------- | ------- | ------- | ------- |
| car         | 44,008  | 83.5 | 77.2    | 83.7    | 86.1    | 87.1    |
| truck       |  2,471  | 53.0 | 42.9    | 52.5    | 55.0    | 61.7    |
| bus         |  1,464  | 84.6 | 81.2    | 83.8    | 86.8    | 86.8    |
| bicycle     |    333  | 73.4 | 72.3    | 73.8    | 73.8    | 73.9    |
| pedestrian  |  6,459  | 64.6 | 63.1    | 64.0    | 64.9    | 66.4    |

</details>

---

### CenterPoint base/2.0

<details>
<summary> Changes </summary>

- Add new dataset `db_j6gen2_v4`
- Add more data to `db_j6_v3`, `db_j6_v5`, `db_j6gen2_v1`, `db_j6gen2_v2`, `db_largebus_v1`
- Overall:
  - Slightly better overall (+0.52 mAP)
  - Car and pedestrian detection remain fairly stable, with small improvements in `base/2.0`.
  - Truck detection shows the largest improvement (+3.84) in `base/2.0`.
  - Bus and bicycle performance slightly drops in `base/2.0`.

</details>

<details>
<summary> Artifacts </summary>

- Deployed onnx model and ROS parameter files [[WebAuto (for internal)]](https://evaluation.tier4.jp/evaluation/mlpackages/7156b453-2861-4ae9-b135-e24e48cc9029/releases/4489a6b0-e8f4-4204-a217-2889f18d3b66?project_id=zWhWRzei)
- Deployed onnx and ROS parameter files [[model-zoo]]
  - [detection_class_remapper.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.0.0/detection_class_remapper.param.yaml)
  - [centerpoint_ml_package.param.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.0.0/centerpoint_ml_package.param.yaml)
  - [deploy_metadata.yaml](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.0.0/deploy_metadata.yaml)
  - [pts_voxel_encoder_centerpoint.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.0.0/pts_voxel_encoder.onnx)
  - [pts_backbone_neck_head_centerpoint.onnx](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.0.0/pts_backbone_neck_head.onnx)
- Training results [[Google drive (for internal)]](https://drive.google.com/drive/folders/1QspUscYcPbWPGAkC321L_s7W70gsZ2Ad?usp=drive_link)
- Training results [model-zoo]
  - [logs.zip](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.0.0/logs.zip)
  - [checkpoint_best.pth](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.0.0/best_NuScenes_metric_T4Metric_mAP_epoch_49.pth)
  - [config.py](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.0.0/second_secfpn_4xb16_121m_base_amp.py)

</details>

<details>
<summary> Training configs </summary>

- [Config file path](https://github.com/tier4/AWML/blob/c50daa0f941da334a2167a4aa587589f6ab76a85/autoware_ml/configs/detection3d/dataset/t4dataset/base.py)
- Train time: NVIDIA H100 80GB * 4 * 50 epochs = 2 days and 20 hours
- Batch size: 4*16 = 64
- Training dataset: DB JPNTAXI v1.0 + DB JPNTAXI v2.0 + DB JPNTAXI v4.0 + DB GSM8 v1.0 + DB J6 v1.0 + DB J6 v2.0 + DB J6 v3.0 + DB J6 v5.0 + DB J6 Gen2 v1.0 + DB J6 Gen2 v2.0 + DB J6 Gen2 v4.0 + DB LargeBus v1.0 (total frames: 71,633)

</details>

<details>
<summary> Evaluation </summary>

- db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 + db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 + db_j6gen2_v1 + db_j6gen2_v1 + db_j6gen2_v4 + db_largebus_v1 (total frames: 5,703):
- Total mAP (eval range = 120m): 0.6806

| class_name | Count    | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| -----------| -------  | ----  | ------- | ------- | ------- | ------- |
| car        |  144,001  | 82.19 | 75.15    | 83.01    | 85.17    | 85.45    |
| truck      |  20,823  | 55.44 | 40.06    | 56.20    | 60.07    | 65.44    |
| bus        |   5,691  | 79.30 | 71.94    | 79.83    | 82.51    | 82.96    |
| bicycle    |   5,007  | 57.62 | 55.87    | 58.15    | 58.21    | 58.28    |
| pedestrian |  42,034  | 65.74 | 63.60    | 64.92    | 66.32    | 68.10    |

- db_largebus_v1 (total frames: 604):
  - Total mAP (eval range = 120m): 0.7109

| class_name | Count    | mAP    | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| -----------| -------  | -----  | ------- | ------- | ------- | ------- |
| car        |  13,831   | 89.01  | 83.62    | 89.78    | 90.98    | 91.70    |
| truck      |  2,137   | 64.11  | 51.19    | 65.25    | 68.75    | 71.25    |
| bus        |     95   | 77.75  | 71.04    | 79.99    | 79.99    | 79.99    |
| bicycle    |    724   | 61.04  | 55.98    | 62.13    | 63.04    | 63.04    |
| pedestrian |  3,916   | 63.56  | 61.90    | 63.03    | 63.78    | 65.55    |

- db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v2 (total frames: 1,157):
  - Total mAP (eval range = 120m): 0.7104

| class_name  | Count   | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ----------  | ------  | ---- | ------- | ------- | ------- | ------- |
| car         | 44,008  | 83.56 | 77.31    | 83.73    | 86.11    | 87.12    |
| truck       |  2,471  | 53.08 | 43.13    | 52.46    | 54.80    | 61.93    |
| bus         |  1,464  | 85.06 | 79.26    | 83.83    | 88.56    | 88.60    |
| bicycle     |    333  | 68.54 | 67.73    | 68.82    | 68.82    | 68.82    |
| pedestrian  |  6,459  | 64.97 | 63.12    | 64.26    | 65.41    | 67.10    |

</details>
