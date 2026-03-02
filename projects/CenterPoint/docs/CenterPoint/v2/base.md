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

- [Config file path](https://github.com/KSeangTan/AWML/blob/0913734976758079c3e2452fa3a866453a00710d/projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_8xb16_121m_base_amp_rfs.py)
- Train time: NVIDIA H100 80GB * 8 * 50 epochs = 3 days 15 hours
- Batch size: 8*16 = 128
- Training Dataset (frames: 134,554):
  - jpntaxi: db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 (25,958 frames)
  - j6: db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 (24,756 frames)
  - j6gen2: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 + db_j6gen2_v6 + db_j6gen2_v7 + db_j6gen2_v8 (43,109 frames)
  - largebus: db_largebus_v1 + db_largebus_v2 (12,605 frames)
  - jpntaxi_gen2: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (28,126 frames)

</details>

<details>
<summary> Evaluation </summary>

**Base Datasets (19,096 frames)**:

  - jpntaxi (1,507 frames): db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4
  - j6 (2,435 frames): db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5
  - j6gen2 (3,951 frames): db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 + db_j6gen2_v6 + db_j6gen2_v7 + db_j6gen2_v8 + db_j6gen2_v9
  - largebus (1,228 frames): db_largebus_v1 + db_largebus_v2 + db_largebus_v3
  - jpntaxi_gen2 (9,975 frames): db_jpntaxigen2_v1 + db_jpntaxigen2_v2

**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.8278**

| class_name | Count | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | Optimal_conf@0.5/1.0/2.0/4.0 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **car** | 145,766 | 0.9309 | 0.8898 / 0.9398 / 0.9428 / 0.9512 | 0.9132 / 0.9402 / 0.9465 / 0.9473 | 0.4244 / 0.3909 / 0.3831 / 0.3831 |
| **truck** | 29.727 | 0.8026 | 0.6518 / 0.8225 / 0.8632 / 0.8730 | 0.7314 / 0.8282 / 0.8659 / 0.8771 | 0.4418 / 0.3869 / 0.3614 / 0.3614 |
| **bus** | 7,196 | 0.8325 | 0.7629 / 0.8401 / 0.8620 / 0.8651 | 0.8263 / 0.8727 / 0.8825 / 0.8836 | 0.4720 / 0.4313 / 0.4335 / 0.4335 |
| **bicycle** | 6,066 | 0.8171 | 0.8027 / 0.8167 / 0.8246 / 0.8246 | 0.8707 / 0.8777 / 0.8784 / 0.8786 | 0.5187 / 0.5187 / 0.5187 / 0.5187 |
| **pedestrian** | 96,613 | 0.7557 | 0.7354 / 0.7476 / 0.7622 / 0.7777 | 0.7968 / 0.8018 / 0.8070 / 0.8130 | 0.4186 / 0.4186 / 0.4166 / 0.4165 |

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.6237**

| class_name | Count | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | Optimal_conf@0.5/1.0/2.0/4.0 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **car** | 124,669 | 0.7888 | 0.7113 / 0.7971 / 0.8206 / 0.8260 | 0.7731 / 0.8213 / 0.8356 / 0.8383 | 0.4049 / 0.3784 / 0.3680 / 0.3665 |
| **truck** | 32,534 | 0.5826 | 0.4364 / 0.5810 / 0.6406 / 0.6723 | 0.5799 / 0.6681 / 0.7162 / 0.7358 | 0.3870 / 0.3514 / 0.3453 / 0.3377 |
| **bus** | 6,772 | 0.6121 | 0.4852 / 0.6151 / 0.6659 / 0.6824 | 0.5933 / 0.6775 / 0.7213 / 0.7313 | 0.3639 / 0.3239 / 0.3457 / 0.3457 |
| **bicycle** | 3,885 | 0.5157 | 0.4833 / 0.5251 / 0.5256 / 0.5289 | 0.6547 / 0.6789 / 0.6795 / 0.6813 | 0.3964 / 0.3964 / 0.3964 / 0.3964 |
| **pedestrian** | 45,535 | 0.6193 | 0.5982 / 0.6110 / 0.6253 / 0.6428 | 0.6822 / 0.6874 / 0.6939 / 0.7016 | 0.4199 / 0.4199 / 0.4199 / 0.4086 |

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.4442**

| class_name | Count | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | Optimal_conf@0.5/1.0/2.0/4.0 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **car** | 46,890 | 0.5425 | 0.4466 / 0.5517 / 0.5797 / 0.5920 | 0.5836 / 0.6459 / 0.6626 / 0.6683 | 0.3634 / 0.3571 / 0.3338 / 0.3332 |
| **truck** | 19,912 | 0.4215 | 0.2117 / 0.4213 / 0.5017 / 0.5512 | 0.4113 / 0.5515 / 0.6124 / 0.6435 | 0.3545 / 0.3457 / 0.3457 / 0.3280 |
| **bus** | 3,159 | 0.3853 | 0.2391 / 0.3890 / 0.4489 / 0.4642 | 0.4441 / 0.5507 / 0.6017 / 0.6130 | 0.3897 / 0.3945 / 0.3586 / 0.3586 |
| **bicycle** | 765 | 0.3366 | 0.2656 / 0.3228 / 0.3791 / 0.3791 | 0.4747 / 0.5195 / 0.5436 / 0.5436 | 0.3693 / 0.3593 / 0.3593 / 0.3593 |
| **pedestrian** | 18,730 | 0.5348 | 0.5153 / 0.5299 / 0.5384 / 0.5557 | 0.6242 / 0.6312 / 0.6348 / 0.6421 | 0.3704 / 0.3704 / 0.3703 / 0.3702 |

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.7056**

| class_name | Count | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | Optimal_conf@0.5/1.0/2.0/4.0 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **car** | 317,325 | 0.8321 | 0.7673 / 0.8393 / 0.8587 / 0.8629 | 0.8173 / 0.8569 / 0.8673 / 0.8694 | 0.4082 / 0.3840 / 0.3728 / 0.3693 |
| **truck** | 82,173 | 0.6338 | 0.4678 / 0.6410 / 0.6989 / 0.7276 | 0.6037 / 0.7069 / 0.7543 / 0.7734 | 0.3962 / 0.3630 / 0.3544 / 0.3537 |
| **bus** | 17,127 | 0.6760 | 0.5704 / 0.6818 / 0.7201 / 0.7316 | 0.6752 / 0.7437 / 0.7749 / 0.7814 | 0.4442 / 0.3654 / 0.3908 / 0.3647 |
| **bicycle** | 10,716 | 0.6858 | 0.6639 / 0.6906 / 0.6940 / 0.6946 | 0.7696 / 0.7844 / 0.7870 / 0.7877 | 0.4327 / 0.4327 / 0.4327 / 0.4327 |
| **pedestrian** | 160,878 | 0.7004 | 0.6794 / 0.6932 / 0.7052 / 0.7237 | 0.7448 / 0.7499 / 0.7554 / 0.7624 | 0.4177 / 0.4177 / 0.4121 / 0.4063 |

---

**LargeBus**: db_largebus_v1 + db_largebus_v2 + db_largebus_v3 (1,228 frames)  

**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.8257**

| class_name | Count | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | Optimal_conf@0.5/1.0/2.0/4.0 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **car** | 14,883 | 0.9527 | 0.9188 / 0.9575 / 0.9668 / 0.9676 | 0.9316 / 0.9563 / 0.9597 / 0.9604 | 0.3665 / 0.3665 / 0.3669 / 0.3669 |
| **truck** | 1,193 | 0.8109 | 0.7249 / 0.8219 / 0.8464 / 0.8506 | 0.7913 / 0.8452 / 0.8652 / 0.8678 | 0.4129 / 0.3772 / 0.3755 / 0.3755 |
| **bus** | 336 | 0.8674 | 0.7440 / 0.9045 / 0.9105 / 0.9105 | 0.8006 / 0.9224 / 0.9274 / 0.9274 | 0.4382 / 0.4382 / 0.4626 / 0.4626 |
| **bicycle** | 740 | 0.7663 | 0.7374 / 0.7687 / 0.7794 / 0.7797 | 0.8397 / 0.8557 / 0.8571 / 0.8586 | 0.4404 / 0.4404 / 0.4404 / 0.4404 |
| **pedestrian** | 5,059 | 0.7313 | 0.7107 / 0.7257 / 0.7354 / 0.7535 | 0.7825 / 0.7875 / 0.7915 / 0.7977 | 0.4783 / 0.4783 / 0.4783 / 0.4550 |

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.6596**

| class_name | Count | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | Optimal_conf@0.5/1.0/2.0/4.0 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **car** | 10,994 | 0.8611 | 0.7918 / 0.8680 / 0.8904 / 0.8944 | 0.8264 / 0.8665 / 0.8787 / 0.8807 | 0.3941 / 0.3752 / 0.3719 / 0.3711 |
| **truck** | 1,011 | 0.5983 | 0.5091 / 0.6065 / 0.6350 / 0.6425 | 0.6431 / 0.7040 / 0.7214 / 0.7226 | 0.3634 / 0.3631 / 0.3567 / 0.3567 |
| **bus** | 143 | 0.8718 | 0.7812 / 0.8966 / 0.8978 / 0.9116 | 0.8140 / 0.8811 / 0.8881 / 0.8881 | 0.5944 / 0.4793 / 0.4793 / 0.4793 |
| **bicycle** | 463 | 0.4058 | 0.3424 / 0.4221 / 0.4295 / 0.4295 | 0.5610 / 0.6043 / 0.6098 / 0.6098 | 0.3950 / 0.3950 / 0.3950 / 0.3950 |
| **pedestrian** | 3,754 | 0.5608 | 0.5432 / 0.5564 / 0.5643 / 0.5791 | 0.6586 / 0.6616 / 0.6646 / 0.6701 | 0.3665 / 0.3665 / 0.3665 / 0.3665 |

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.4526**

| class_name | Count | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | Optimal_conf@0.5/1.0/2.0/4.0 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **car** | 3,018 | 0.6234 | 0.5426 / 0.6306 / 0.6552 / 0.6651 | 0.6565 / 0.7017 / 0.7129 / 0.7178 | 0.3842 / 0.3766 / 0.3336 / 0.3492 |
| **truck** | 602 | 0.5257 | 0.3575 / 0.5413 / 0.5893 / 0.6145 | 0.5316 / 0.6562 / 0.6945 / 0.7077 | 0.3822 / 0.3279 / 0.3286 / 0.3312 |
| **bus** | 60 | 0.4414 | 0.2861 / 0.4745 / 0.5025 / 0.5025 | 0.4835 / 0.6087 / 0.6304 / 0.6304 | 0.4769 / 0.4743 / 0.4743 / 0.4743 |
| **bicycle** | 85 | 0.2794 | 0.1898 / 0.2901 / 0.3189 / 0.3189 | 0.4224 / 0.5101 / 0.5465 / 0.5465 | 0.3571 / 0.4044 / 0.3138 / 0.3138 |
| **pedestrian** | 1,121 | 0.3932 | 0.3795 / 0.3884 / 0.3957 / 0.4090 | 0.5380 / 0.5410 / 0.5451 / 0.5502 | 0.3688 / 0.3688 / 0.3688 / 0.3688 |

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.7304**

| class_name | Count | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | Optimal_conf@0.5/1.0/2.0/4.0 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **car** | 28,895 | 0.8932 | 0.8436 / 0.8988 / 0.9115 / 0.9191 | 0.8678 / 0.8999 / 0.9074 / 0.9088 | 0.3930 / 0.3752 / 0.3742 / 0.3723 |
| **truck** | 2,806 | 0.6801 | 0.5756 / 0.6916 / 0.7215 / 0.7318 | 0.6875 / 0.7577 / 0.7808 / 0.7847 | 0.3936 / 0.3652 / 0.3626 / 0.3626 |
| **bus** | 539 | 0.8234 | 0.7030 / 0.8571 / 0.8639 / 0.8697 | 0.7703 / 0.8794 / 0.8872 / 0.8872 | 0.4743 / 0.4626 / 0.4626 / 0.4626 |
| **bicycle** | 1,288 | 0.6184 | 0.5736 / 0.6262 / 0.6363 / 0.6373 | 0.7221 / 0.7529 / 0.7555 / 0.7564 | 0.3950 / 0.3950 / 0.3950 / 0.3950 |
| **pedestrian** | 9,934 | 0.6370 | 0.6191 / 0.6313 / 0.6402 / 0.6572 | 0.7066 / 0.7109 / 0.7140 / 0.7210 | 0.3954 / 0.3937 / 0.3937 / 0.3937 |

---

**J6Gen2**: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 + db_j6gen2_v6 + db_j6gen2_v7 + db_j6gen2_v8 + db_j6gen2_v9 (3,951 frames)

**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.8381**

| class_name | Count | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | Optimal_conf@0.5/1.0/2.0/4.0 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **car** | 49,637 | 0.9416 | 0.9076 / 0.9454 / 0.9560 / 0.9573 | 0.9235 / 0.9475 / 0.9527 / 0.9535 | 0.4337 / 0.3931 / 0.3921 / 0.3851 |
| **truck** | 5,754 | 0.7964 | 0.6837 / 0.8067 / 0.8392 / 0.8560 | 0.7675 / 0.8287 / 0.8527 / 0.8674 | 0.4261 / 0.3617 / 0.3843 / 0.3560 |
| **bus** | 1,939 | 0.9211 | 0.8648 / 0.9276 / 0.9458 / 0.9463 | 0.8873 / 0.9275 / 0.9446 / 0.9463 | 0.3957 / 0.3937 / 0.3960 / 0.3717 |
| **bicycle** | 639 | 0.8097 | 0.8010 / 0.8126 / 0.8126 / 0.8126 | 0.8650 / 0.8683 / 0.8683 / 0.8683 | 0.5187 / 0.5187 / 0.5187 / 0.5187 |
| **pedestrian** | 14,362 | 0.7214 | 0.6963 / 0.7155 / 0.7277 / 0.7463 | 0.7595 / 0.7674 / 0.7713 / 0.7775 | 0.4524 / 0.4421 / 0.4335 / 0.4335 |

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.6322**

| class_name | Count | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | Optimal_conf@0.5/1.0/2.0/4.0 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **car** | 47,568 | 0.7772 | 0.6957 / 0.7819 / 0.8100 / 0.8213 | 0.7692 / 0.8181 / 0.8370 / 0.8412 | 0.4048 / 0.3684 / 0.3523 / 0.3523 |
| **truck** | 4,090 | 0.5682 | 0.4370 / 0.5681 / 0.6213 / 0.6465 | 0.5883 / 0.6726 / 0.7133 / 0.7313 | 0.3707 / 0.3450 / 0.3450 / 0.3455 |
| **bus** | 1,935 | 0.7315 | 0.5934 / 0.7404 / 0.7879 / 0.8044 | 0.6818 / 0.7570 / 0.7969 / 0.8036 | 0.4226 / 0.3347 / 0.3653 / 0.3675 |
| **bicycle** | 295 | 0.5658 | 0.5431 / 0.5724 / 0.5724 / 0.5755 | 0.6437 / 0.6628 / 0.6628 / 0.6628 | 0.4481 / 0.4481 / 0.4481 / 0.4481 |
| **pedestrian** | 6,529 | 0.5180 | 0.4948 / 0.5118 / 0.5247 / 0.5406 | 0.6028 / 0.6108 / 0.6144 / 0.6203 | 0.4206 / 0.4206 / 0.4184 / 0.4086 |

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.3641**

| class_name | Count | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | Optimal_conf@0.5/1.0/2.0/4.0 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **car** | 17,353 | 0.5291 | 0.4389 / 0.5288 / 0.5663 / 0.5825 | 0.5834 / 0.6395 / 0.6651 / 0.6728 | 0.3405 / 0.3260 / 0.3100 / 0.3100 |
| **truck** | 2,570 | 0.3655 | 0.1810 / 0.3486 / 0.4408 / 0.4915 | 0.3810 / 0.5040 / 0.5687 / 0.6093 | 0.3357 / 0.3344 / 0.3316 / 0.3212 |
| **bus** | 316 | 0.3006 | 0.1851 / 0.2705 / 0.3622 / 0.3844 | 0.3766 / 0.4342 / 0.5059 / 0.5216 | 0.3898 / 0.2941 / 0.3324 / 0.3324 |
| **bicycle** | 70 | 0.3739 | 0.3569 / 0.3634 / 0.3877 / 0.3877 | 0.5524 / 0.5524 / 0.5714 / 0.5714 | 0.4475 / 0.4475 / 0.4475 / 0.4475 |
| **pedestrian** | 1,673 | 0.2512 | 0.2440 / 0.2471 / 0.2532 / 0.2605 | 0.4364 / 0.4377 / 0.4410 / 0.4461 | 0.3852 / 0.3852 / 0.3852 / 0.3814 |

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.7260**

| class_name | Count | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | Optimal_conf@0.5/1.0/2.0/4.0 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **car** | 114,558 | 0.8250 | 0.7621 / 0.8312 / 0.8486 / 0.8582 | 0.8157 / 0.8540 / 0.8674 / 0.8705 | 0.4044 / 0.3702 / 0.3561 / 0.3522 |
| **truck** | 12,414 | 0.6443 | 0.5095 / 0.6479 / 0.6975 / 0.7225 | 0.6394 / 0.7191 / 0.7566 / 0.7767 | 0.3820 / 0.3512 / 0.3556 / 0.3514 |
| **bus** | 4,190 | 0.8047 | 0.7043 / 0.8106 / 0.8470 / 0.8571 | 0.7667 / 0.8217 / 0.8538 / 0.8579 | 0.4518 / 0.3738 / 0.3981 / 0.3644 |
| **bicycle** | 1,004 | 0.7155 | 0.7043 / 0.7172 / 0.7178 / 0.7227 | 0.7797 / 0.7882 / 0.7893 / 0.7893 | 0.4475 / 0.4475 / 0.4475 / 0.4475 |
| **pedestrian** | 22,564 | 0.6405 | 0.6166 / 0.6340 / 0.6470 / 0.6645 | 0.6920 / 0.6998 / 0.7035 / 0.7097 | 0.4238 / 0.4238 / 0.4238 / 0.4238 |

---

**JPNTaxi_Gen2**: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (9,975 frames)

**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.8176**

| class_name | Count | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | Optimal_conf@0.5/1.0/2.0/4.0 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **car** | 42,789 | 0.9284 | 0.8876 / 0.9388 / 0.9429 / 0.9442 | 0.9142 / 0.9410 / 0.9482 / 0.9485 | 0.4245 / 0.3951 / 0.3954 / 0.3954 |
| **truck** | 17,259 | 0.8205 | 0.6726 / 0.8366 / 0.8809 / 0.8918 | 0.7305 / 0.8259 / 0.8745 / 0.8865 | 0.4335 / 0.3543 / 0.3544 / 0.3544 |
| **bus** | 3,437 | 0.7626 | 0.6941 / 0.7637 / 0.7912 / 0.8016 | 0.7833 / 0.8216 / 0.8283 / 0.8305 | 0.4790 / 0.4355 / 0.4355 / 0.3840 |
| **bicycle** | 2,681 | 0.8029 | 0.7993 / 0.8041 / 0.8041 / 0.8041 | 0.8712 / 0.8772 / 0.8772 / 0.8772 | 0.5189 / 0.5189 / 0.5189 / 0.5189 |
| **pedestrian** | 57,948 | 0.7737 | 0.7556 / 0.7658 / 0.7765 / 0.7967 | 0.8108 / 0.8141 / 0.8196 / 0.8263 | 0.4161 / 0.4091 / 0.4077 / 0.4055 |

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.6363**

| class_name | Count | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | Optimal_conf@0.5/1.0/2.0/4.0 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **car** | 35,518 | 0.8194 | 0.7609 / 0.8281 / 0.8420 / 0.8468 | 0.8101 / 0.8469 / 0.8548 / 0.8559 | 0.4079 / 0.3887 / 0.3887 / 0.3887 |
| **truck** | 22,550 | 0.5984 | 0.4651 / 0.5966 / 0.6551 / 0.6769 | 0.5966 / 0.6747 / 0.7238 / 0.7369 | 0.3868 / 0.3523 / 0.3453 / 0.3374 |
| **bus** | 2,683 | 0.4303 | 0.2841 / 0.4127 / 0.5043 / 0.5202 | 0.4427 / 0.5419 / 0.6070 / 0.6213 | 0.2926 / 0.2878 / 0.3341 / 0.3240 |
| **bicycle** | 1,607 | 0.6674 | 0.6315 / 0.6789 / 0.6789 / 0.6804 | 0.7504 / 0.7735 / 0.7735 / 0.7742 | 0.4670 / 0.4670 / 0.4670 / 0.4670 |
| **pedestrian** | 27,240 | 0.6657 | 0.6464 / 0.6578 / 0.6720 / 0.6866 | 0.7146 / 0.7197 / 0.7268 / 0.7333 | 0.4197 / 0.4199 / 0.4199 / 0.4107 |

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.4921**

| class_name | Count | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | Optimal_conf@0.5/1.0/2.0/4.0 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **car** | 16,524 | 0.5659 | 0.4811 / 0.5802 / 0.5964 / 0.6058 | 0.6026 / 0.6611 / 0.6675 / 0.6695 | 0.3993 / 0.3763 / 0.3763 / 0.3599 |
| **truck** | 14,587 | 0.4387 | 0.2182 / 0.4428 / 0.5202 / 0.5738 | 0.4195 / 0.5672 / 0.6270 / 0.6583 | 0.3612 / 0.3536 / 0.3451 / 0.3280 |
| **bus** | 2,476 | 0.4369 | 0.2792 / 0.4486 / 0.5039 / 0.5157 | 0.4756 / 0.5965 / 0.6448 / 0.6545 | 0.3903 / 0.3945 / 0.3832 / 0.3832 |
| **bicycle** | 364 | 0.3966 | 0.2846 / 0.3653 / 0.4682 / 0.4685 | 0.4928 / 0.5493 / 0.5944 / 0.5944 | 0.3800 / 0.3689 / 0.3689 / 0.3689 |
| **pedestrian** | 14,297 | 0.6222 | 0.5999 / 0.6182 / 0.6270 / 0.6436 | 0.6833 / 0.6918 / 0.6951 / 0.7010 | 0.3703 / 0.3703 / 0.3703 / 0.3600 |

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.7041**

| class_name | Count | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | Optimal_conf@0.5/1.0/2.0/4.0 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **car** | 94,831 | 0.8382 | 0.7822 / 0.8467 / 0.8590 / 0.8648 | 0.8286 / 0.8635 / 0.8707 / 0.8717 | 0.4282 / 0.4049 / 0.3906 / 0.3906 |
| **truck** | 54,396 | 0.6359 | 0.4725 / 0.6413 / 0.7018 / 0.7282 | 0.6006 / 0.7032 / 0.7551 / 0.7725 | 0.4091 / 0.3536 / 0.3454 / 0.3418 |
| **bus** | 8,596 | 0.5756 | 0.4630 / 0.5788 / 0.6243 / 0.6360 | 0.5924 / 0.6719 / 0.7127 / 0.7200 | 0.3792 / 0.3627 / 0.3647 / 0.3647 |
| **bicycle** | 4,652 | 0.7415 | 0.7166 / 0.7430 / 0.7530 / 0.7533 | 0.8009 / 0.8138 / 0.8176 / 0.8178 | 0.4788 / 0.4641 / 0.4641 / 0.4641 |
| **pedestrian** | 99,485 | 0.7294 | 0.7100 / 0.7205 / 0.7341 / 0.7530 | 0.7659 / 0.7703 / 0.7761 / 0.7832 | 0.4121 / 0.4121 / 0.4061 / 0.3911 |

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
