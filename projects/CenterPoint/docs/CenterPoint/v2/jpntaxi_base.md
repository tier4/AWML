# Deployed model for CenterPoint JpnTaxi-Base/2.X
## Summary
The main difference between `JPNTaxi-Base/2.x` and `JPNTaxi-Gen2/2.x` is that all `JPNTaxi` data from Gen1 and Gen2 are used for training/validation/testing in `JPNTaxi-Base/2.x` to reduce overfitting since they share similar
vehicle and sensor setups.

### Main Parameters

  - **Range:** 122.40m
  - **Voxel Size:** [0.24, 0.24, 8.0]
  - **Grid Size:** [510, 510, 1]
  - **With Intensity**

### Testing Datsets

- **Total Frames: 11,482**
  <details>
  <summary> jpntaxi (1,507 frames)</summary>

    - `db_jpntaxi_v1`
    - `db_jpntaxi_v2`
    - `db_jpntaxi_v4`

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

  | Model version | mAP  | car <br> (50,801) | truck <br> (19,247) | bus <br> (4,181) | bicycle <br> (3,383) | pedestrian <br> (65,955) |
  | -------------------------| ---- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
  | CenterPoint JPNTaxi_Base/2.6.1     | 83.43 | 91.89            | 83.53            | 84.49            | 79.04            | 78.20            |
  | CenterPoint JPNTaxi_Base/2.5.1     | 83.63 | 91.44            | 83.31            | 85.68            | 79.51            | 78.23            |

  </details>

  <details>
  <summary> Eval Range: 50.0 - 90.0m </summary>

  | Model version | mAP  | car <br> (41,469) | truck <br> (24,337) | bus <br> (3,162) | bicycle <br> (1,902) | pedestrian <br> (30,590) |
  | -------------------------| ---- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
  | CenterPoint JPNTaxi_Base/2.6.1     | 66.45 | 82.56            | 63.30            | 50.24            | 67.38            | 68.79            |
  | CenterPoint JPNTaxi_Base/2.5.1     | 67.27 | 82.61            | 63.46            | 53.34            | 68.17            | 68.76            |

  </details>

  <details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version | mAP  | car <br> (18,703) | truck <br> (15,472) | bus <br> (2,710) | bicycle <br> (407) | pedestrian <br> (14,911) |
  | -------------------------| ---- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
  | CenterPoint JPNTaxi_Base/2.6.1     | 51.16 | 62.09            | 49.33            | 42.80            | 36.99            | 64.59            |
  | CenterPoint JPNTaxi_Base/2.5.1     | 51.35 | 61.33            | 47.02            | 42.91            | 39.91            | 65.61            |

  </details>

  <details open>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version | mAP  | car <br> (110,973) | truck <br> (59,056) | bus <br> (10,053) | bicycle <br> (5,692) | pedestrian <br> (111,456) |
  | -------------------------| ---- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
  | CenterPoint JPNTaxi_Base/2.6.1     | 72.66 | 84.50            | 67.29            | 63.53            | 73.59            | 74.41            |
  | CenterPoint JPNTaxi_Base/2.5.1     | 72.86 | 84.30            | 66.50            | 65.04            | 74.02            | 74.44            |

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
  | -------------------------| ---- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
  | CenterPoint JPNTaxi_Base/2.6.1     | 83.64 | 92.62            | 83.33            | 84.20            | 79.48            | 78.59            |
  | CenterPoint JPNTaxi_Base/2.5.1     | 83.77 | 92.29            | 83.14            | 85.53            | 79.30            | 78.59            |

  </details>

  <details>
  <summary> Eval Range: 50.0 - 90.0m </summary>

  | Model version | mAP  | car <br> (35,518) | truck <br> (22,550) | bus <br> (2,683) | bicycle <br> (1,607) | pedestrian <br> (27,240) |
  | -------------------------| ---- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
  | CenterPoint JPNTaxi_Base/2.6.1     | 66.62 | 84.48            | 63.76            | 45.85            | 69.30            | 69.72            |
  | CenterPoint JPNTaxi_Base/2.5.1     | 67.52 | 84.60            | 63.87            | 49.57            | 69.86            | 69.69            |

  </details>

  <details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version | mAP  | car <br> (16,524) | truck <br> (14,587) | bus <br> (2,476) | bicycle <br> (364) | pedestrian <br> (14,297) |
  | -------------------------| ---- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
  | CenterPoint JPNTaxi_Base/2.6.1     | 52.59 | 63.70            | 50.21            | 45.18            | 37.58            | 66.28            |
  | CenterPoint JPNTaxi_Base/2.5.1     | 52.84 | 63.00            | 47.84            | 45.22            | 40.83            | 67.32            |

  </details>

  <details>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version | mAP  | car <br> (94,831) | truck <br> (54,396) | bus <br> (8,596) | bicycle <br> (4,652) | pedestrian <br> (99,485) |
  | -------------------------| ---- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
  | CenterPoint JPNTaxi_Base/2.6.1     | 72.79 | 85.59            | 67.35            | 61.85            | 74.24            | 74.90            |
  | CenterPoint JPNTaxi_Base/2.5.1     | 73.00 | 85.27            | 66.49            | 63.68            | 74.56            | 74.98            |

  </details>

</details>

</details>

## Release

### CenterPoint JPNTaxi_Base/2.6.1

<details>
<summary> Changes  </summary>

- Finetune from `CenterPoint base/2.5.0` with `db_jpntaxi_base` datasets, where it includes both JPNTaxi Gen1 and Gen2 data
- Include intensity as an extra feature and Repeat Sampling Factor (RFS)
- Overall:
  - Performance is better than `CenterPoint base/2.6.0`, but almost similar compared to `CenterPoint JPJPNTaxi_Base/2.5.1`

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
- Train time: NVIDIA H100 80GB * 8 * 30 epochs = 21 hours
- Batch size: 8*16 = 128
- Training Dataset (frames: 54,084):
  - jpntaxi: db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 (25,958 frames)
  - jpntaxi_gen2: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (28,126 frames)

</details>

<details>
<summary> Evaluation </summary>

**JPNTaxi_Base Datasets (11,482 frames)**:

  - jpntaxi (1,507 frames): db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4
  - jpntaxi_gen2 (9,975 frames): db_jpntaxigen2_v1 + db_jpntaxigen2_v2

**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.8343**

| class_name | Count | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | Optimal_conf@0.5/1.0/2.0/4.0 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **car** | 50,801 | 0.9189 | 0.8641 / 0.9318 / 0.9353 / 0.9442 | 0.9015 / 0.9347 / 0.9419 / 0.9424 | 0.4041 / 0.3617 / 0.3531 / 0.3531 |
| **truck** | 19,247 | 0.8353 | 0.6747 / 0.8462 / 0.9064 / 0.9137 | 0.7484 / 0.8437 / 0.8914 / 0.9013 | 0.4700 / 0.4227 / 0.3803 / 0.3803 |
| **bus** | 4,181 | 0.8449 | 0.7792 / 0.8523 / 0.8701 / 0.8779 | 0.8244 / 0.8652 / 0.8762 / 0.8775 | 0.5134 / 0.3990 / 0.3990 / 0.3917 |
| **bicycle** | 3,383 | 0.7904 | 0.7777 / 0.7945 / 0.7947 / 0.7948 | 0.8481 / 0.8588 / 0.8591 / 0.8594 | 0.4891 / 0.4891 / 0.4891 / 0.4891 |
| **pedestrian** | 65,955 | 0.7820 | 0.7596 / 0.7758 / 0.7870 / 0.8054 | 0.8158 / 0.8201 / 0.8244 / 0.8293 | 0.3969 / 0.3969 / 0.3753 / 0.3753 |

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.6645**

| class_name | Count | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | Optimal_conf@0.5/1.0/2.0/4.0 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **car** | 41,469 | 0.8256 | 0.7660 / 0.8354 / 0.8493 / 0.8517 | 0.8112 / 0.8504 / 0.8606 / 0.8624 | 0.3977 / 0.3722 / 0.3558 / 0.3558 |
| **truck** | 24,337 | 0.6330 | 0.4781 / 0.6366 / 0.6961 / 0.7213 | 0.6090 / 0.7005 / 0.7473 / 0.7657 | 0.4128 / 0.3564 / 0.3481 / 0.3426 |
| **bus** | 3,162 | 0.5024 | 0.3595 / 0.4977 / 0.5686 / 0.5838 | 0.4939 / 0.5972 / 0.6667 / 0.6766 | 0.3875 / 0.3443 / 0.3441 / 0.3441 |
| **bicycle** | 1,902 | 0.6738 | 0.6375 / 0.6856 / 0.6856 / 0.6864 | 0.7520 / 0.7793 / 0.7793 / 0.7799 | 0.4208 / 0.3652 / 0.3652 / 0.3652 |
| **pedestrian** | 30,590 | 0.6879 | 0.6665 / 0.6796 / 0.6960 / 0.7097 | 0.7305 / 0.7360 / 0.7434 / 0.7487 | 0.4106 / 0.4106 / 0.4106 / 0.4106 |

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.5116**

| class_name | Count | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | Optimal_conf@0.5/1.0/2.0/4.0 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **car** | 18,703 | 0.6209 | 0.5305 / 0.6391 / 0.6540 / 0.6599 | 0.6496 / 0.7152 / 0.7250 / 0.7274 | 0.3357 / 0.3112 / 0.3112 / 0.3112 |
| **truck** | 15,472 | 0.4933 | 0.2673 / 0.4967 / 0.5784 / 0.6307 | 0.4592 / 0.6020 / 0.6639 / 0.6995 | 0.3906 / 0.3539 / 0.3462 / 0.3409 |
| **bus** | 2,710 | 0.4280 | 0.3259 / 0.3997 / 0.4844 / 0.5018 | 0.4996 / 0.5430 / 0.6377 / 0.6543 | 0.4675 / 0.3666 / 0.3923 / 0.3881 |
| **bicycle** | 407 | 0.3699 | 0.2659 / 0.3486 / 0.4321 / 0.4332 | 0.5006 / 0.5472 / 0.5885 / 0.5885 | 0.3499 / 0.3287 / 0.3559 / 0.3559 |
| **pedestrian** | 14,911 | 0.6459 | 0.6242 / 0.6403 / 0.6507 / 0.6684 | 0.6947 / 0.7022 / 0.7068 / 0.7134 | 0.3712 / 0.3712 / 0.3712 / 0.3673 |

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.7266**

| class_name | Count | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | Optimal_conf@0.5/1.0/2.0/4.0 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **car** | 110,973 | 0.8450 | 0.7816 / 0.8569 / 0.8695 / 0.8720 | 0.8299 / 0.8701 / 0.8789 / 0.8802 | 0.3914 / 0.3630 / 0.3542 / 0.3542 |
| **truck** | 59,056 | 0.6729 | 0.4945 / 0.6780 / 0.7460 / 0.7730 | 0.6222 / 0.7277 / 0.7806 / 0.8005 | 0.4053 / 0.3780 / 0.3566 / 0.3559 |
| **bus** | 10,053 | 0.6353 | 0.5370 / 0.6329 / 0.6802 / 0.6912 | 0.6485 / 0.7030 / 0.7533 / 0.7617 | 0.4732 / 0.3864 / 0.3801 / 0.3801 |
| **bicycle** | 5,692 | 0.7359 | 0.7077 / 0.7381 / 0.7486 / 0.7492 | 0.7921 / 0.8108 / 0.8140 / 0.8143 | 0.4208 / 0.4034 / 0.4034 / 0.4034 |
| **pedestrian** | 111,456 | 0.7441 | 0.7233 / 0.7357 / 0.7500 / 0.7672 | 0.7761 / 0.7810 / 0.7862 / 0.7914 | 0.4106 / 0.3970 / 0.3970 / 0.3790 |

---

**JPNTaxi_Gen2 Datasets (9,975 frames)**:

  - jpntaxi_gen2 (9,975 frames): db_jpntaxigen2_v1 + db_jpntaxigen2_v2

**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.8364**

| class_name | Count | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | Optimal_conf@0.5/1.0/2.0/4.0 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **car** | 42,789 | 0.9262 | 0.8787 / 0.9346 / 0.9451 / 0.9463 | 0.9118 / 0.9387 / 0.9453 / 0.9456 | 0.3859 / 0.3617 / 0.3532 / 0.3532 |
| **truck** | 17,259 | 0.8333 | 0.6759 / 0.8429 / 0.9022 / 0.9124 | 0.7463 / 0.8414 / 0.8894 / 0.8998 | 0.4700 / 0.4230 / 0.3744 / 0.3805 |
| **bus** | 3,437 | 0.8420 | 0.7797 / 0.8474 / 0.8663 / 0.8745 | 0.8302 / 0.8626 / 0.8724 / 0.8740 | 0.5192 / 0.3990 / 0.3990 / 0.3917 |
| **bicycle** | 2,681 | 0.7948 | 0.7848 / 0.7981 / 0.7981 / 0.7981 | 0.8628 / 0.8688 / 0.8688 / 0.8688 | 0.5179 / 0.5179 / 0.5179 / 0.5179 |
| **pedestrian** | 57,948 | 0.7859 | 0.7653 / 0.7803 / 0.7901 / 0.8079 | 0.8210 / 0.8243 / 0.8278 / 0.8319 | 0.3798 / 0.3753 / 0.3753 / 0.3753 |

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.6662**

| class_name | Count | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | Optimal_conf@0.5/1.0/2.0/4.0 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **car** | 35,518 | 0.8448 | 0.7964 / 0.8521 / 0.8644 / 0.8662 | 0.8320 / 0.8620 / 0.8715 / 0.8727 | 0.3962 / 0.3722 / 0.3704 / 0.3704 |
| **truck** | 22,550 | 0.6376 | 0.4909 / 0.6396 / 0.6977 / 0.7223 | 0.6183 / 0.7045 / 0.7490 / 0.7653 | 0.4128 / 0.3564 / 0.3481 / 0.3478 |
| **bus** | 2,683 | 0.4585 | 0.3127 / 0.4514 / 0.5295 / 0.5402 | 0.4553 / 0.5670 / 0.6366 / 0.6433 | 0.3443 / 0.3054 / 0.3441 / 0.3441 |
| **bicycle** | 1,607 | 0.6930 | 0.6610 / 0.7034 / 0.7034 / 0.7044 | 0.7739 / 0.8019 / 0.8019 / 0.8026 | 0.3695 / 0.3518 / 0.3518 / 0.3518 |
| **pedestrian** | 27,240 | 0.6972 | 0.6748 / 0.6890 / 0.7050 / 0.7201 | 0.7378 / 0.7430 / 0.7510 / 0.7559 | 0.4370 / 0.4297 / 0.4297 / 0.4102 |

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.5259**

| class_name | Count | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | Optimal_conf@0.5/1.0/2.0/4.0 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **car** | 16,524 | 0.6370 | 0.5575 / 0.6523 / 0.6661 / 0.6722 | 0.6659 / 0.7214 / 0.7306 / 0.7330 | 0.3616 / 0.3112 / 0.3112 / 0.3112 |
| **truck** | 14,587 | 0.5021 | 0.2774 / 0.5067 / 0.5860 / 0.6382 | 0.4697 / 0.6112 / 0.6709 / 0.7062 | 0.3906 / 0.3534 / 0.3462 / 0.3462 |
| **bus** | 2,476 | 0.4518 | 0.3543 / 0.4224 / 0.5078 / 0.5226 | 0.5197 / 0.5551 / 0.6553 / 0.6697 | 0.4737 / 0.3666 / 0.3923 / 0.3881 |
| **bicycle** | 364 | 0.3758 | 0.2664 / 0.3505 / 0.4418 / 0.4445 | 0.5029 / 0.5486 / 0.5951 / 0.5951 | 0.3499 / 0.3559 / 0.3559 / 0.3559 |
| **pedestrian** | 14,297 | 0.6628 | 0.6409 / 0.6569 / 0.6677 / 0.6855 | 0.7073 / 0.7148 / 0.7195 / 0.7258 | 0.3712 / 0.3712 / 0.3712 / 0.3673 |

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.7279**

| class_name | Count | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | Optimal_conf@0.5/1.0/2.0/4.0 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **car** | 94,831 | 0.8559 | 0.8003 / 0.8675 / 0.8747 / 0.8812 | 0.8435 / 0.8761 / 0.8842 / 0.8852 | 0.3914 / 0.3644 / 0.3617 / 0.3617 |
| **truck** | 54,396 | 0.6735 | 0.4988 / 0.6786 / 0.7448 / 0.7718 | 0.6248 / 0.7287 / 0.7802 / 0.7996 | 0.4043 / 0.3780 / 0.3539 / 0.3559 |
| **bus** | 8,596 | 0.6185 | 0.5236 / 0.6125 / 0.6641 / 0.6737 | 0.6387 / 0.6881 / 0.7428 / 0.7491 | 0.4734 / 0.3771 / 0.3801 / 0.3801 |
| **bicycle** | 4,652 | 0.7424 | 0.7156 / 0.7476 / 0.7531 / 0.7533 | 0.7998 / 0.8157 / 0.8194 / 0.8198 | 0.4208 / 0.3915 / 0.3915 / 0.3652 |
| **pedestrian** | 99,485 | 0.7490 | 0.7297 / 0.7407 / 0.7544 / 0.7712 | 0.7811 / 0.7853 / 0.7902 / 0.7951 | 0.4070 / 0.3970 / 0.3970 / 0.3790 |

</details>

---

### CenterPoint JPNTaxi_Base/2.5.1

<details>
<summary> Changes  </summary>

- Finetune from `CenterPoint base/2.5.0` with `db_jpntaxi_base` datasets, where it includes both JPNTaxi Gen1 and Gen2 data
- Include intensity as an extra feature and Repeat Sampling Factor (RFS)
- Overall:
  - Performance is better than `CenterPoint base/2.4.0` in `JPNTaxi Gen2` (`60.70`), it's also better than `JPNTaxi_Gen2/2.5.1` in both `JPNTaxi Gen2` and `JPNTaxi Gen2_v2`

</details>

<details>
<summary> Artifacts </summary>

- Deployed onnx and ROS parameter files (for internal)
  - [WebAuto](https://evaluation.tier4.jp/evaluation/mlpackages/7156b453-2861-4ae9-b135-e24e48cc9029/releases/41e733b4-a778-4281-ba5b-6ee9ab7c89dc?project_id=zWhWRzei)
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/jpntaxi_base/v2.5.1/deployment.zip)
  - [Google drive](https://drive.google.com/file/d/1wI5lIv1E4Ysg5d1jWj-e9f8CBFB5cM2m/view?usp=drive_link)
- Logs (for internal)
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/jpntaxi_base/v2.5.1/logs.zip)
  - [Google drive](https://drive.google.com/file/d/1nHuc0u_9UWAJZWIol_xnJWgODIeFJv0g/view?usp=drive_link)

</details>

<details>
<summary> Training configs </summary>

- [Config file path](https://github.com/tier4/AWML/blob/dee55764f5381ef75dcac7a17a303b0bf527d400/projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_jpntaxi_base_amp_rfs.py)
- Train time: NVIDIA H100 80GB * 4 * 30 epochs = 1 day and 20 hours
- Batch size: 4*16 = 64
- Training Datasets (frames: 50,950):
  - jpntaxi: db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 (25,958 frames)
  - jpntaxi_gen2: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (24,992 frames)

</details>

<details>
<summary> Evaluation </summary>

**JPNTaxi Base:** db_jpntaxi_base (3,216 frames)  
**Total mAP (120 m range):** **0.666**

| class_name | Count  | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
|-----------|---------|------|---------|---------|---------|---------|
| car        | 25,882 | 81.5 | 71.5 | 83.4 | 85.4 | 85.9 |
| truck      | 7,155  | 49.9 | 33.6 | 48.9 | 54.9 | 62.2 |
| bus        | 4,026  | 67.9 | 53.2 | 67.7 | 74.5 | 76.2 |
| bicycle    | 1,506  | 62.2 | 55.2 | 64.3 | 64.4 | 65.1 |
| pedestrian | 22,489 | 71.5 | 68.7 | 70.7 | 72.5 | 74.2 |

---

**JPNTaxi Gen2:** db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (1,709 frames)  
**Total mAP (120 m range):** **0.63**

| class_name | Count  | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
|-----------|--------|------|---------|---------|---------|---------|
| car        | 9,710  | 88.7 | 82.9 | 89.6 | 90.7 | 91.4 |
| truck      | 2,577  | 42.7 | 31.5 | 39.6 | 43.7 | 56.0 |
| bus        | 2,569  | 64.5 | 46.9 | 63.5 | 73.1 | 74.6 |
| bicycle    |   466  | 45.6 | 36.1 | 48.4 | 48.8 | 49.1 |
| pedestrian | 10,518 | 73.3 | 71.3 | 72.5 | 74.0 | 75.4 |

---

**JPNTaxi Gen2_V2:** db_jpntaxigen2_v2 (230 frames)  
**Total mAP (120 m range):** **0.733**

| class_name | Count | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
|-----------|--------|------|---------|---------|---------|---------|
| car        | 3,449 | 90.0 | 82.8 | 91.5 | 92.8 | 93.1 |
| truck      |   726 | 53.8 | 44.7 | 53.7 | 55.7 | 60.9 |
| bus        |   251 | 85.4 | 79.8 | 86.7 | 87.3 | 87.8 |
| bicycle    |   157 | 63.6 | 49.6 | 67.5 | 68.4 | 69.0 |
| pedestrian | 2,443 | 73.8 | 72.0 | 73.5 | 74.3 | 75.2 |

</details>

---
