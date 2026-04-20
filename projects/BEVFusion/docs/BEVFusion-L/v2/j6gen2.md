# Deployed model for BEVFusion-LiDAR J6Gen2_base/2.X
## Summary

### Main Parameters

  - **Range:** [122.40m, 122.40m, 8.0m]
  - **Voxel Size:** [0.17, 0.17, 0.2]
  - **Grid Size:** [1440, 1440, 40]
  - **With Intensity**

### Testing Datasets

- **Total Frames: 5,179**

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
  <summary> j6gen2_base (5,179 frames) </summary>

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

  </details>


### mAP - J6Gen2_base

- **Class mAP for BEV Center Distance: 0.5m, 1.0m, 2.0m, 4.0m**

  <details>
  <summary> Eval Range: 0.0 - 50.0m </summary>

  | Model version | mAP | mAPH | car<br>(64,520) | truck<br>(6,947) | bus<br>(2,275) | bicycle<br>(1,379) | pedestrian<br>(19,421) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.6.1 | 0.8810 | 0.8380 | 0.8873 | 0.8586 | 0.9476 | 0.8583 | 0.8534 |

  </details>

  <details>
  <summary> Eval Range: 50.0 - 90.0m </summary>

  | Model version | mAP | mAPH | car<br>(58,562) | truck<br>(5,101) | bus<br>(2,078) | bicycle<br>(758) | pedestrian<br>(10,283) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.6.1 | 0.7032 | 0.6483 | 0.7876 | 0.6830 | 0.7911 | 0.5802 | 0.6741 |

  </details>

  <details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version | mAP | mAPH | car<br>(20,371) | truck<br>(3,172) | bus<br>(376) | bicycle<br>(155) | pedestrian<br>(2,794) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.6.1 | 0.4938 | 0.4494 | 0.6564 | 0.5192 | 0.3777 | 0.4406 | 0.4752 |

  </details>

  <details open>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version | mAP | mAPH | car<br>(143,453) | truck<br>(15,220) | bus<br>(4,729) | bicycle<br>(2,292) | pedestrian<br>(32,498) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.6.1 | 0.7903 | 0.7413 | 0.8266 | 0.7409 | 0.8510 | 0.7541 | 0.7790 |

  </details>

## Datasets

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
  | BEVFusion-LiDAR j6gen2_base/2.6.1 | 0.8985 | 0.8484 | 0.9087 | 0.8974 | 0.9636 | 0.8447 | 0.8780 |

  </details>

  <details>
  <summary> Eval Range: 50.0 - 90.0m </summary>

  | Model version | mAP | mAPH | car<br>(10,994) | truck<br>(1,011) | bus<br>(143) | bicycle<br>(463) | pedestrian<br>(3,754) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.6.1 | 0.7475 | 0.6925 | 0.8317 | 0.7758 | 0.7910 | 0.5959 | 0.7433 |

  </details>

  <details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version | mAP | mAPH | car<br>(3,018) | truck<br>(602) | bus<br>(60) | bicycle<br>(85) | pedestrian<br>(1,121) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.6.1 | 0.5636 | 0.5191 | 0.7125 | 0.6383 | 0.4781 | 0.4293 | 0.5595 |

  </details>

  <details>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version | mAP | mAPH | car<br>(28,895) | truck<br>(2,806) | bus<br>(539) | bicycle<br>(1,288) | pedestrian<br>(9,934) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.6.1 | 0.8198 | 0.7666 | 0.8690 | 0.8052 | 0.8756 | 0.7455 | 0.8036 |

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
  | BEVFusion-LiDAR j6gen2_base/2.6.1 | 0.8788 | 0.8368 | 0.8813 | 0.8505 | 0.9427 | 0.8749 | 0.8448 |

  </details>

  <details>
  <summary> Eval Range: 50.0 - 90.0m </summary>

  | Model version | mAP | mAPH | car<br>(47,568) | truck<br>(4,090) | bus<br>(1,935) | bicycle<br>(295) | pedestrian<br>(6,529) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.6.1 | 0.6864 | 0.6344 | 0.7772 | 0.6609 | 0.7913 | 0.5671 | 0.6357 |

  </details>

  <details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version | mAP | mAPH | car<br>(17,353) | truck<br>(2,570) | bus<br>(316) | bicycle<br>(70) | pedestrian<br>(1,673) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.6.1 | 0.4766 | 0.4309 | 0.6465 | 0.4903 | 0.3618 | 0.4627 | 0.4214 |

  </details>

  <details>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version | mAP | mAPH | car<br>(114,558) | truck<br>(12,414) | bus<br>(4,190) | bicycle<br>(1,004) | pedestrian<br>(22,564) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.6.1 | 0.7851 | 0.7375 | 0.8166 | 0.7262 | 0.8481 | 0.7661 | 0.7687 |

  </details>

</details>

## Release

### BEVFusion-LiDAR J6Gen2_base/2.6.1

<details>
<summary> Changes  </summary>

- Finetune from `BEVFusion-LiDAR base/2.6.0` with j6gen2 base dataset
- Train with new datasets:
  - `db_j6gen2_v9`
  - `db_largebus_v3`
</details>

<details>
<summary> Artifacts </summary>

- Deployed onnx and ROS parameter files (for internal)
  - [WebAuto](https://evaluation.tier4.jp/evaluation/mlpackages/46f8188d-e3be-4f2f-b989-fd27002610d7/releases/c9e6a2c5-b31f-48af-b53c-3ab6a898509e?project_id=zWhWRzei)
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-l/j6gen2_base/v2.6.1/deployment.zip)
  - [Google drive](https://drive.google.com/file/d/1CrFCZaXv5Thnz7qL_f4ftL8PchsO21sW/view?usp=drive_link)
- Logs (for internal)
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-l/j6gen2_base/v2.6.1/logs.zip)
  - [Google drive](https://drive.google.com/file/d/1ejh_49Phev_nnoHC6XywpOFwpL7UICip/view?usp=drive_link)
- Pytorch Best checkpoints:
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-l/j6gen2_base/v2.6.1/epoch_28.pth)
  - [Google drive](https://drive.google.com/file/d/1NHrrcKsG2Hea4ShAE44NUwteHty7-LFL/view?usp=drive_link)

</details>

<details>
<summary> Training configs </summary>

- [Config file path](https://github.com/KSeangTan/AWML/blob/f03f8f474157f11535ee628befc54e34d3087804/projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_8xb8_j6gen2_base_120m.py)
- Train time: NVIDIA H100 80GB * 8 * 30 epochs = 20 hours
- Batch size: 8*8 = 64
- Training Dataset (frames: 55,714):
  - j6gen2: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 + db_j6gen2_v6 + db_j6gen2_v7 + db_j6gen2_v8 (43,109 frames)
  - largebus: db_largebus_v1 + db_largebus_v2 + db_largebus_v3 (12,605 frames)

</details>

<details>
<summary> Evaluation </summary>

**J6Gen2_base Datasets (5,179 frames)**:

  - j6gen2 (3,951 frames): db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 + db_j6gen2_v6 + db_j6gen2_v7 + db_j6gen2_v8 + db_j6gen2_v9
  - largebus (1,228 frames): db_largebus_v1 + db_largebus_v2 + db_largebus_v3

**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.8810**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 64,520 | 0.8873 | 0.831 / 0.880 / 0.907 / 0.931 | 0.889 / 0.916 / 0.924 / 0.933 | 0.261 / 0.183 / 0.164 / 0.107 |
| truck | 6,947 | 0.8586 | 0.735 / 0.853 / 0.907 / 0.940 | 0.804 / 0.869 / 0.900 / 0.917 | 0.242 / 0.194 / 0.165 / 0.165 |
| bus | 2,275 | 0.9476 | 0.879 / 0.946 / 0.982 / 0.983 | 0.914 / 0.954 / 0.969 / 0.970 | 0.188 / 0.137 / 0.137 / 0.137 |
| bicycle | 1,379 | 0.8583 | 0.823 / 0.854 / 0.876 / 0.881 | 0.857 / 0.869 / 0.883 / 0.885 | 0.281 / 0.185 / 0.185 / 0.185 |
| pedestrian | 19,421 | 0.8534 | 0.829 / 0.851 / 0.862 / 0.872 | 0.819 / 0.830 / 0.837 / 0.842 | 0.172 / 0.159 / 0.159 / 0.159 |
| **ALL** | 94,542 | 0.8810 | — | — | — |

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.7032**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 58,562 | 0.7876 | 0.651 / 0.778 / 0.840 / 0.882 | 0.753 / 0.825 / 0.853 / 0.873 | 0.222 / 0.185 / 0.158 / 0.123 |
| truck | 5,101 | 0.6830 | 0.490 / 0.663 / 0.770 / 0.810 | 0.638 / 0.739 / 0.794 / 0.813 | 0.227 / 0.195 / 0.195 / 0.194 |
| bus | 2,078 | 0.7911 | 0.565 / 0.784 / 0.894 / 0.921 | 0.684 / 0.811 / 0.875 / 0.892 | 0.342 / 0.150 / 0.138 / 0.113 |
| bicycle | 758 | 0.5802 | 0.494 / 0.598 / 0.614 / 0.615 | 0.635 / 0.681 / 0.685 / 0.687 | 0.171 / 0.174 / 0.174 / 0.174 |
| pedestrian | 10,283 | 0.6741 | 0.646 / 0.669 / 0.684 / 0.696 | 0.691 / 0.704 / 0.712 / 0.719 | 0.139 / 0.136 / 0.138 / 0.136 |
| **ALL** | 76,782 | 0.7032 | — | — | — |

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.4938**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 20,371 | 0.6564 | 0.465 / 0.641 / 0.735 / 0.785 | 0.615 / 0.716 / 0.761 / 0.787 | 0.180 / 0.164 / 0.151 / 0.144 |
| truck | 3,172 | 0.5192 | 0.229 / 0.467 / 0.655 / 0.726 | 0.445 / 0.608 / 0.717 / 0.755 | 0.199 / 0.187 / 0.137 / 0.122 |
| bus | 376 | 0.3777 | 0.159 / 0.342 / 0.486 / 0.524 | 0.351 / 0.492 / 0.582 / 0.599 | 0.076 / 0.044 / 0.040 / 0.040 |
| bicycle | 155 | 0.4406 | 0.346 / 0.458 / 0.479 / 0.479 | 0.506 / 0.577 / 0.591 / 0.591 | 0.124 / 0.185 / 0.124 / 0.124 |
| pedestrian | 2,794 | 0.4752 | 0.459 / 0.472 / 0.480 / 0.490 | 0.580 / 0.586 / 0.590 / 0.595 | 0.131 / 0.131 / 0.118 / 0.118 |
| **ALL** | 26,868 | 0.4938 | — | — | — |

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.7903**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 143,453 | 0.8266 | 0.721 / 0.817 / 0.867 / 0.901 | 0.798 / 0.853 / 0.874 / 0.889 | 0.223 / 0.191 / 0.162 / 0.135 |
| truck | 15,220 | 0.7409 | 0.556 / 0.720 / 0.821 / 0.867 | 0.680 / 0.777 / 0.831 / 0.853 | 0.242 / 0.191 / 0.165 / 0.163 |
| bus | 4,729 | 0.8510 | 0.697 / 0.844 / 0.924 / 0.940 | 0.780 / 0.862 / 0.903 / 0.912 | 0.335 / 0.150 / 0.113 / 0.113 |
| bicycle | 2,292 | 0.7541 | 0.696 / 0.758 / 0.780 / 0.783 | 0.766 / 0.794 / 0.805 / 0.807 | 0.185 / 0.185 / 0.185 / 0.185 |
| pedestrian | 32,498 | 0.7790 | 0.754 / 0.776 / 0.788 / 0.799 | 0.759 / 0.771 / 0.778 / 0.784 | 0.153 / 0.153 / 0.151 / 0.151 |
| **ALL** | 198,192 | 0.7903 | — | — | — |

---

**LargeBus**: db_largebus_v1 + db_largebus_v2 + db_largebus_v3 (1,228 frames)  

**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.8985**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 14,883 | 0.9087 | 0.866 / 0.903 / 0.923 / 0.943 | 0.908 / 0.932 / 0.937 / 0.944 | 0.225 / 0.164 / 0.164 / 0.098 |
| truck | 1,193 | 0.8974 | 0.779 / 0.908 / 0.944 / 0.959 | 0.846 / 0.912 / 0.927 / 0.930 | 0.350 / 0.167 / 0.167 / 0.166 |
| bus | 336 | 0.9636 | 0.901 / 0.983 / 0.985 / 0.985 | 0.921 / 0.968 / 0.968 / 0.968 | 0.394 / 0.394 / 0.394 / 0.394 |
| bicycle | 740 | 0.8447 | 0.791 / 0.833 / 0.873 / 0.882 | 0.847 / 0.861 / 0.881 / 0.884 | 0.282 / 0.278 / 0.277 / 0.277 |
| pedestrian | 5,059 | 0.8780 | 0.862 / 0.877 / 0.884 / 0.890 | 0.852 / 0.862 / 0.867 / 0.870 | 0.161 / 0.153 / 0.159 / 0.159 |
| **ALL** | 22,211 | 0.8985 | — | — | — |

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.7475**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 10,994 | 0.8317 | 0.721 / 0.828 / 0.873 / 0.905 | 0.798 / 0.860 / 0.878 / 0.893 | 0.221 / 0.177 / 0.177 / 0.127 |
| truck | 1,011 | 0.7758 | 0.630 / 0.768 / 0.843 / 0.862 | 0.742 / 0.818 / 0.855 / 0.859 | 0.207 / 0.158 / 0.158 / 0.171 |
| bus | 143 | 0.7910 | 0.561 / 0.868 / 0.868 / 0.868 | 0.707 / 0.851 / 0.851 / 0.851 | 0.592 / 0.592 / 0.592 / 0.592 |
| bicycle | 463 | 0.5959 | 0.486 / 0.620 / 0.639 / 0.640 | 0.626 / 0.679 / 0.686 / 0.686 | 0.146 / 0.146 / 0.146 / 0.146 |
| pedestrian | 3,754 | 0.7433 | 0.724 / 0.741 / 0.749 / 0.759 | 0.738 / 0.750 / 0.753 / 0.760 | 0.123 / 0.123 / 0.123 / 0.123 |
| **ALL** | 16,365 | 0.7475 | — | — | — |

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.5636**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 3,018 | 0.7125 | 0.571 / 0.707 / 0.771 / 0.801 | 0.680 / 0.752 / 0.778 / 0.792 | 0.229 / 0.196 / 0.160 / 0.160 |
| truck | 602 | 0.6383 | 0.344 / 0.636 / 0.772 / 0.800 | 0.540 / 0.726 / 0.799 / 0.814 | 0.333 / 0.213 / 0.213 / 0.138 |
| bus | 60 | 0.4781 | 0.320 / 0.479 / 0.551 / 0.563 | 0.477 / 0.590 / 0.629 / 0.629 | 0.064 / 0.034 / 0.034 / 0.034 |
| bicycle | 85 | 0.4293 | 0.303 / 0.448 / 0.483 / 0.483 | 0.505 / 0.590 / 0.623 / 0.623 | 0.124 / 0.124 / 0.124 / 0.124 |
| pedestrian | 1,121 | 0.5595 | 0.543 / 0.556 / 0.562 / 0.577 | 0.633 / 0.640 / 0.642 / 0.647 | 0.134 / 0.133 / 0.131 / 0.131 |
| **ALL** | 4,886 | 0.5636 | — | — | — |

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.8198**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 28,895 | 0.8690 | 0.787 / 0.866 / 0.898 / 0.925 | 0.845 / 0.888 / 0.900 / 0.910 | 0.221 / 0.185 / 0.160 / 0.126 |
| truck | 2,806 | 0.8052 | 0.635 / 0.808 / 0.878 / 0.900 | 0.744 / 0.839 / 0.875 / 0.881 | 0.259 / 0.174 / 0.137 / 0.137 |
| bus | 539 | 0.8756 | 0.747 / 0.908 / 0.922 / 0.925 | 0.821 / 0.896 / 0.896 / 0.896 | 0.394 / 0.337 / 0.337 / 0.337 |
| bicycle | 1,288 | 0.7455 | 0.665 / 0.748 / 0.782 / 0.787 | 0.741 / 0.775 / 0.794 / 0.797 | 0.196 / 0.196 / 0.194 / 0.194 |
| pedestrian | 9,934 | 0.8036 | 0.785 / 0.802 / 0.810 / 0.818 | 0.785 / 0.795 / 0.798 / 0.803 | 0.143 / 0.134 / 0.134 / 0.143 |
| **ALL** | 43,462 | 0.8198 | — | — | — |

---

**J6Gen2**: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 + db_j6gen2_v6 + db_j6gen2_v7 + db_j6gen2_v8 + db_j6gen2_v9 (3,951 frames)

**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.8788**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 49,637 | 0.8813 | 0.820 / 0.877 / 0.900 / 0.928 | 0.884 / 0.912 / 0.920 / 0.929 | 0.269 / 0.189 / 0.165 / 0.107 |
| truck | 5,754 | 0.8505 | 0.725 / 0.841 / 0.900 / 0.936 | 0.796 / 0.861 / 0.894 / 0.915 | 0.242 / 0.189 / 0.165 / 0.162 |
| bus | 1,939 | 0.9427 | 0.878 / 0.935 / 0.975 / 0.983 | 0.916 / 0.953 / 0.971 / 0.973 | 0.124 / 0.124 / 0.124 / 0.124 |
| bicycle | 639 | 0.8749 | 0.861 / 0.879 / 0.879 / 0.880 | 0.884 / 0.894 / 0.894 / 0.894 | 0.151 / 0.151 / 0.151 / 0.151 |
| pedestrian | 14,362 | 0.8448 | 0.818 / 0.841 / 0.854 / 0.865 | 0.807 / 0.820 / 0.826 / 0.833 | 0.190 / 0.165 / 0.159 / 0.165 |
| **ALL** | 72,331 | 0.8788 | — | — | — |

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.6864**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 47,568 | 0.7772 | 0.636 / 0.765 / 0.832 / 0.876 | 0.742 / 0.816 / 0.847 / 0.868 | 0.231 / 0.190 / 0.158 / 0.123 |
| truck | 4,090 | 0.6609 | 0.458 / 0.637 / 0.752 / 0.796 | 0.613 / 0.721 / 0.780 / 0.802 | 0.227 / 0.193 / 0.193 / 0.193 |
| bus | 1,935 | 0.7913 | 0.567 / 0.776 / 0.897 / 0.926 | 0.684 / 0.811 / 0.880 / 0.899 | 0.342 / 0.150 / 0.113 / 0.113 |
| bicycle | 295 | 0.5671 | 0.518 / 0.576 / 0.585 / 0.588 | 0.660 / 0.692 / 0.692 / 0.695 | 0.179 / 0.179 / 0.179 / 0.179 |
| pedestrian | 6,529 | 0.6357 | 0.603 / 0.629 / 0.649 / 0.662 | 0.667 / 0.681 / 0.692 / 0.699 | 0.139 / 0.136 / 0.139 / 0.139 |
| **ALL** | 60,417 | 0.6864 | — | — | — |

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.4766**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 17,353 | 0.6465 | 0.446 / 0.629 / 0.728 / 0.783 | 0.605 / 0.710 / 0.758 / 0.786 | 0.180 / 0.163 / 0.144 / 0.143 |
| truck | 2,570 | 0.4903 | 0.201 / 0.425 / 0.627 / 0.709 | 0.423 / 0.579 / 0.698 / 0.742 | 0.199 / 0.185 / 0.122 / 0.122 |
| bus | 316 | 0.3618 | 0.133 / 0.317 / 0.478 / 0.520 | 0.332 / 0.483 / 0.582 / 0.603 | 0.076 / 0.052 / 0.045 / 0.048 |
| bicycle | 70 | 0.4627 | 0.403 / 0.478 / 0.485 / 0.485 | 0.561 / 0.614 / 0.614 / 0.614 | 0.214 / 0.214 / 0.214 / 0.214 |
| pedestrian | 1,673 | 0.4214 | 0.405 / 0.418 / 0.428 / 0.435 | 0.543 / 0.551 / 0.556 / 0.560 | 0.118 / 0.118 / 0.118 / 0.118 |
| **ALL** | 21,982 | 0.4766 | — | — | — |

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.7851**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 114,558 | 0.8166 | 0.704 / 0.808 / 0.858 / 0.897 | 0.786 / 0.844 / 0.868 / 0.884 | 0.223 / 0.190 / 0.158 / 0.136 |
| truck | 12,414 | 0.7262 | 0.538 / 0.700 / 0.808 / 0.860 | 0.666 / 0.763 / 0.821 / 0.847 | 0.242 / 0.192 / 0.163 / 0.163 |
| bus | 4,190 | 0.8481 | 0.690 / 0.836 / 0.924 / 0.942 | 0.775 / 0.859 / 0.906 / 0.916 | 0.309 / 0.150 / 0.113 / 0.113 |
| bicycle | 1,004 | 0.7661 | 0.737 / 0.772 / 0.777 / 0.778 | 0.800 / 0.819 / 0.819 / 0.820 | 0.192 / 0.185 / 0.185 / 0.185 |
| pedestrian | 22,564 | 0.7687 | 0.741 / 0.765 / 0.779 / 0.791 | 0.748 / 0.761 / 0.769 / 0.775 | 0.152 / 0.152 / 0.151 / 0.151 |
| **ALL** | 154,730 | 0.7851 | — | — | — |

</details>

---
