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
  | BEVFusion-LiDAR base/2.6.0 | 0.8774 | 0.9049<br>(preds: 296,476) | 0.8514<br>(preds: 164,380) | 0.8824<br>(preds: 34,104) | 0.8543<br>(preds: 70,454) | 0.8941<br>(preds: 1,428,756) |

  </details>

  <details>
  <summary> Eval Range: 50.0 - 90.0m </summary>

  | Model version | mAP | mAPH | car<br>(94,080) | truck<br>(27,651) | bus<br>(4,761) | bicycle<br>(2,365) | pedestrian<br>(37,523) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.6.0 | 0.6824 | 0.6437 | 0.8005<br>(preds: 558,958) | 0.6567<br>(preds: 364,384) | 0.5783<br>(preds: 97,681) | 0.6322<br>(preds: 108,824) | 0.7445<br>(preds: 1,688,164) |

  </details>

  <details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version | mAP | mAPH | car<br>(36,895) | truck<br>(17,759) | bus<br>(2,852) | bicycle<br>(519) | pedestrian<br>(17,091) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.6.0 | 0.5136 | 0.4788 | 0.6552<br>(preds: 479,002) | 0.5023<br>(preds: 323,541) | 0.2849<br>(preds: 99,503) | 0.4369<br>(preds: 76,680) | 0.6887<br>(preds: 1,084,129) |

  </details>

  <details open>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version | mAP | mAPH | car<br>(238,284) | truck<br>(69,616) | bus<br>(13,325) | bicycle<br>(6,944) | pedestrian<br>(131,983) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.6.0 | 0.7592 | 0.7227 | 0.8398<br>(preds: 1,334,436) | 0.6994<br>(preds: 852,305) | 0.6621<br>(preds: 231,288) | 0.7595<br>(preds: 255,958) | 0.8351<br>(preds: 4,201,049) |

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

  | Model version | mAP | mAPH | car<br>(42,789) | truck<br>(17,259) | bus<br>(3,437) | bicycle<br>(2,681) | pedestrian<br>(57,948) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.6.0 | 0.8784 | 0.8487 | 0.9436<br>(preds: 158,413) | 0.8531<br>(preds: 113,284) | 0.8284<br>(preds: 25,085) | 0.8546<br>(preds: 50,072) | 0.9123<br>(preds: 909,802) |

  </details>

  <details>
  <summary> Eval Range: 50.0 - 90.0m </summary>

  | Model version | mAP | mAPH | car<br>(35,518) | truck<br>(22,550) | bus<br>(2,683) | bicycle<br>(1,607) | pedestrian<br>(27,240) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.6.0 | 0.6692 | 0.6414 | 0.8323<br>(preds: 316,517) | 0.6571<br>(preds: 254,916) | 0.4033<br>(preds: 74,300) | 0.6721<br>(preds: 75,955) | 0.7812<br>(preds: 1,114,768) |

  </details>

  <details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version | mAP | mAPH | car<br>(16,524) | truck<br>(14,587) | bus<br>(2,476) | bicycle<br>(364) | pedestrian<br>(14,297) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.6.0 | 0.5300 | 0.5010 | 0.6692<br>(preds: 290,114) | 0.5020<br>(preds: 232,484) | 0.2822<br>(preds: 76,725) | 0.4586<br>(preds: 51,256) | 0.7380<br>(preds: 741,963) |

  </details>

  <details open>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version | mAP | mAPH | car<br>(94,831) | truck<br>(54,396) | bus<br>(8,596) | bicycle<br>(4,652) | pedestrian<br>(99,485) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.6.0 | 0.7471 | 0.7176 | 0.8667<br>(preds: 765,044) | 0.6928<br>(preds: 600,684) | 0.5446<br>(preds: 176,110) | 0.7710<br>(preds: 177,283) | 0.8606<br>(preds: 2,766,533) |

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
  | BEVFusion-LiDAR base/2.6.0 | 0.8882 | 0.8475 | 0.9045<br>(preds: 35,517) | 0.8793<br>(preds: 13,243) | 0.9482<br>(preds: 1,812) | 0.8489<br>(preds: 6,030) | 0.8598<br>(preds: 123,764) |
  | BEVFusion-LiDAR j6gen2_base/2.6.1 | 0.8985 | 0.8484 | 0.9087<br>(preds: 34,803) | 0.8974<br>(preds: 11,206) | 0.9636<br>(preds: 1,592) | 0.8447<br>(preds: 6,087) | 0.8780<br>(preds: 121,342) |

  </details>

  <details>
  <summary> Eval Range: 50.0 - 90.0m </summary>

  | Model version | mAP | mAPH | car<br>(10,994) | truck<br>(1,011) | bus<br>(143) | bicycle<br>(463) | pedestrian<br>(3,754) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.6.0 | 0.7132 | 0.6586 | 0.8237<br>(preds: 52,451) | 0.7245<br>(preds: 24,337) | 0.7811<br>(preds: 3,950) | 0.5497<br>(preds: 9,742) | 0.6871<br>(preds: 138,030) |
  | BEVFusion-LiDAR j6gen2_base/2.6.1 | 0.7475 | 0.6925 | 0.8317<br>(preds: 54,532) | 0.7758<br>(preds: 22,159) | 0.7910<br>(preds: 3,594) | 0.5959<br>(preds: 8,897) | 0.7433<br>(preds: 130,386) |

  </details>

  <details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version | mAP | mAPH | car<br>(3,018) | truck<br>(602) | bus<br>(60) | bicycle<br>(85) | pedestrian<br>(1,121) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.6.0 | 0.5202 | 0.4736 | 0.6989<br>(preds: 39,277) | 0.6297<br>(preds: 19,133) | 0.4058<br>(preds: 4,162) | 0.3609<br>(preds: 7,974) | 0.5056<br>(preds: 91,114) |
  | BEVFusion-LiDAR j6gen2_base/2.6.1 | 0.5636 | 0.5191 | 0.7125<br>(preds: 45,021) | 0.6383<br>(preds: 19,358) | 0.4781<br>(preds: 3,942) | 0.4293<br>(preds: 7,544) | 0.5595<br>(preds: 92,867) |

  </details>

  <details open>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version | mAP | mAPH | car<br>(28,895) | truck<br>(2,806) | bus<br>(539) | bicycle<br>(1,288) | pedestrian<br>(9,934) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.6.0 | 0.7995 | 0.7514 | 0.8640<br>(preds: 127,245) | 0.7788<br>(preds: 56,713) | 0.8608<br>(preds: 9,924) | 0.7272<br>(preds: 23,746) | 0.7669<br>(preds: 352,908) |
  | BEVFusion-LiDAR j6gen2_base/2.6.1 | 0.8198 | 0.7666 | 0.8690<br>(preds: 134,356) | 0.8052<br>(preds: 52,723) | 0.8756<br>(preds: 9,128) | 0.7455<br>(preds: 22,528) | 0.8036<br>(preds: 344,595) |

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
  | BEVFusion-LiDAR base/2.6.0 | 0.8702 | 0.8284 | 0.8758<br>(preds: 102,540) | 0.8410<br>(preds: 37,854) | 0.9408<br>(preds: 7,207) | 0.8590<br>(preds: 14,352) | 0.8344<br>(preds: 395,185) |
  | BEVFusion-LiDAR j6gen2_base/2.6.1 | 0.8788 | 0.8368 | 0.8813<br>(preds: 102,012) | 0.8505<br>(preds: 36,635) | 0.9427<br>(preds: 6,610) | 0.8749<br>(preds: 14,250) | 0.8448<br>(preds: 376,671) |

  </details>

  <details>
  <summary> Eval Range: 50.0 - 90.0m </summary>

  | Model version | mAP | mAPH | car<br>(47,568) | truck<br>(4,090) | bus<br>(1,935) | bicycle<br>(295) | pedestrian<br>(6,529) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.6.0 | 0.6708 | 0.6165 | 0.7721<br>(preds: 189,993) | 0.6421<br>(preds: 85,129) | 0.7731<br>(preds: 19,431) | 0.5472<br>(preds: 23,124) | 0.6192<br>(preds: 435,365) |
  | BEVFusion-LiDAR j6gen2_base/2.6.1 | 0.6864 | 0.6344 | 0.7772<br>(preds: 192,445) | 0.6609<br>(preds: 82,423) | 0.7913<br>(preds: 18,598) | 0.5671<br>(preds: 21,929) | 0.6357<br>(preds: 410,196) |

  </details>

  <details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version | mAP | mAPH | car<br>(17,353) | truck<br>(2,570) | bus<br>(316) | bicycle<br>(70) | pedestrian<br>(1,673) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.6.0 | 0.4462 | 0.4042 | 0.6346<br>(preds: 149,618) | 0.4758<br>(preds: 71,927) | 0.3215<br>(preds: 18,616) | 0.4303<br>(preds: 17,451) | 0.3688<br>(preds: 251,053) |
  | BEVFusion-LiDAR j6gen2_base/2.6.1 | 0.4766 | 0.4309 | 0.6465<br>(preds: 167,424) | 0.4903<br>(preds: 76,034) | 0.3618<br>(preds: 18,271) | 0.4627<br>(preds: 16,203) | 0.4214<br>(preds: 253,107) |

  </details>

  <details open>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version | mAP | mAPH | car<br>(114,558) | truck<br>(12,414) | bus<br>(4,190) | bicycle<br>(1,004) | pedestrian<br>(22,564) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.6.0 | 0.7712 | 0.7223 | 0.8110<br>(preds: 442,151) | 0.7129<br>(preds: 194,910) | 0.8348<br>(preds: 45,254) | 0.7458<br>(preds: 54,927) | 0.7515<br>(preds: 1,081,603) |
  | BEVFusion-LiDAR j6gen2_base/2.6.1 | 0.7851 | 0.7375 | 0.8166<br>(preds: 461,881) | 0.7262<br>(preds: 195,092) | 0.8481<br>(preds: 43,479) | 0.7661<br>(preds: 52,382) | 0.7687<br>(preds: 1,039,974) |

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

| class_name | GTs | Preds | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | ---: | :---- | :---- | :---- |
| car | 107,309 | 296,476 | 0.9049 | 0.851 / 0.902 / 0.924 / 0.942 | 0.897 / 0.929 / 0.937 / 0.942 | 0.247 / 0.195 / 0.159 / 0.141 |
| truck | 24,206 | 164,380 | 0.8514 | 0.701 / 0.841 / 0.919 / 0.945 | 0.799 / 0.875 / 0.920 / 0.934 | 0.297 / 0.196 / 0.169 / 0.165 |
| bus | 5,712 | 34,104 | 0.8824 | 0.781 / 0.878 / 0.934 / 0.937 | 0.805 / 0.864 / 0.898 / 0.900 | 0.027 / 0.024 / 0.024 / 0.024 |
| bicycle | 4,060 | 70,454 | 0.8543 | 0.833 / 0.857 / 0.863 / 0.864 | 0.860 / 0.869 / 0.870 / 0.870 | 0.242 / 0.230 / 0.228 / 0.228 |
| pedestrian | 77,369 | 1,428,756 | 0.8941 | 0.875 / 0.892 / 0.901 / 0.909 | 0.856 / 0.866 / 0.872 / 0.877 | 0.156 / 0.148 / 0.149 / 0.148 |
| **ALL** | 218,656 | 1,994,170 | 0.8774 | — | — | — |

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.6824**

| Label | GTs | Preds | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | ---: | :---- | :---- | :---- |
| car | 94,080 | 558,958 | 0.8005 | 0.683 / 0.798 / 0.848 / 0.873 | 0.771 / 0.833 / 0.859 / 0.869 | 0.230 / 0.179 / 0.158 / 0.141 |
| truck | 27,651 | 364,384 | 0.6567 | 0.430 / 0.620 / 0.760 / 0.817 | 0.600 / 0.718 / 0.794 / 0.820 | 0.239 / 0.193 / 0.162 / 0.155 |
| bus | 4,761 | 97,681 | 0.5783 | 0.321 / 0.551 / 0.705 / 0.736 | 0.472 / 0.623 / 0.721 / 0.739 | 0.255 / 0.069 / 0.068 / 0.068 |
| bicycle | 2,365 | 108,824 | 0.6322 | 0.574 / 0.647 / 0.653 / 0.655 | 0.683 / 0.714 / 0.715 / 0.716 | 0.172 / 0.172 / 0.172 / 0.172 |
| pedestrian | 37,523 | 1,688,164 | 0.7445 | 0.724 / 0.742 / 0.752 / 0.761 | 0.738 / 0.747 / 0.752 / 0.757 | 0.158 / 0.152 / 0.151 / 0.152 |
| **ALL** | 166,380 | 2,818,011 | 0.6824 | — | — | — |

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.5136**

| class_name | GTs | Preds | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | ---: | :---- | :---- | :---- |
| car | 36,895 | 479,002 | 0.6552 | 0.493 / 0.651 / 0.724 / 0.752 | 0.626 / 0.716 / 0.751 / 0.763 | 0.181 / 0.160 / 0.155 / 0.140 |
| truck | 17,759 | 323,541 | 0.5023 | 0.195 / 0.447 / 0.626 / 0.742 | 0.420 / 0.598 / 0.708 / 0.767 | 0.205 / 0.189 / 0.160 / 0.145 |
| bus | 2,852 | 99,503 | 0.2849 | 0.103 / 0.282 / 0.359 / 0.395 | 0.331 / 0.446 / 0.491 / 0.511 | 0.025 / 0.027 / 0.027 / 0.027 |
| bicycle | 519 | 76,680 | 0.4369 | 0.336 / 0.420 / 0.496 / 0.496 | 0.509 / 0.551 / 0.580 / 0.580 | 0.181 / 0.123 / 0.181 / 0.181 |
| pedestrian | 17,091 | 1,084,129 | 0.6887 | 0.667 / 0.684 / 0.694 / 0.710 | 0.704 / 0.712 / 0.718 / 0.726 | 0.134 / 0.134 / 0.134 / 0.134 |
| **ALL** | 75,116 | 2,062,855 | 0.5136 | — | — | — |

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.7592**

| class_name | GTs | Preds | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | ---: | :---- | :---- | :---- |
| car | 238,284 | 1,334,436 | 0.8398 | 0.744 / 0.838 / 0.878 / 0.900 | 0.809 / 0.862 / 0.881 / 0.888 | 0.230 / 0.177 / 0.159 / 0.157 |
| truck | 69,616 | 852,305 | 0.6994 | 0.475 / 0.666 / 0.797 / 0.859 | 0.632 / 0.749 / 0.823 / 0.853 | 0.269 / 0.199 / 0.163 / 0.155 |
| bus | 13,325 | 231,288 | 0.6621 | 0.478 / 0.650 / 0.749 / 0.771 | 0.567 / 0.673 / 0.732 / 0.743 | 0.228 / 0.044 / 0.044 / 0.044 |
| bicycle | 6,944 | 255,958 | 0.7595 | 0.721 / 0.765 / 0.775 / 0.777 | 0.777 / 0.796 / 0.799 / 0.800 | 0.183 / 0.183 / 0.183 / 0.184 |
| pedestrian | 131,983 | 4,201,049 | 0.8351 | 0.815 / 0.833 / 0.842 / 0.851 | 0.804 / 0.814 / 0.819 / 0.825 | 0.148 / 0.148 / 0.148 / 0.148 |
| **ALL** | 460,152 | 6,875,036 | 0.7592 | — | — | — |

---

**LargeBus**: db_largebus_v1 + db_largebus_v2 + db_largebus_v3 (1,228 frames)  

**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.8882**

| class_name | GTs | Preds | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | ---: | :---- | :---- | :---- |
| car | 14,883 | 35,517 | 0.9045 | 0.854 / 0.902 / 0.922 / 0.940 | 0.905 / 0.930 / 0.936 / 0.942 | 0.213 / 0.195 / 0.153 / 0.124 |
| truck | 1,193 | 13,243 | 0.8793 | 0.749 / 0.895 / 0.927 / 0.947 | 0.822 / 0.907 / 0.918 / 0.923 | 0.270 / 0.167 / 0.167 / 0.167 |
| bus | 336 | 1,812 | 0.9482 | 0.851 / 0.981 / 0.981 / 0.981 | 0.894 / 0.957 / 0.957 / 0.957 | 0.261 / 0.222 / 0.222 / 0.222 |
| bicycle | 740 | 6,030 | 0.8489 | 0.792 / 0.850 / 0.872 / 0.881 | 0.844 / 0.866 / 0.867 / 0.871 | 0.212 / 0.212 / 0.212 / 0.212 |
| pedestrian | 5,059 | 123,764 | 0.8598 | 0.844 / 0.858 / 0.865 / 0.872 | 0.841 / 0.849 / 0.852 / 0.854 | 0.161 / 0.165 / 0.165 / 0.165 |
| **ALL** | 22,211 | 180,366 | 0.8882 | — | — | — |

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.7132**

| class_name | GTs | Preds | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | ---: | :---- | :---- | :---- |
| car | 10,994 | 52,451 | 0.8237 | 0.716 / 0.823 / 0.866 / 0.890 | 0.792 / 0.852 / 0.873 / 0.882 | 0.213 / 0.181 / 0.158 / 0.147 |
| truck | 1,011 | 24,337 | 0.7245 | 0.521 / 0.729 / 0.813 / 0.834 | 0.661 / 0.796 / 0.836 / 0.840 | 0.212 / 0.169 / 0.169 / 0.143 |
| bus | 143 | 3,950 | 0.7811 | 0.606 / 0.834 / 0.834 / 0.850 | 0.741 / 0.824 / 0.824 / 0.824 | 0.469 / 0.345 / 0.345 / 0.345 |
| bicycle | 463 | 9,742 | 0.5497 | 0.418 / 0.578 / 0.598 / 0.605 | 0.576 / 0.646 / 0.651 / 0.654 | 0.161 / 0.151 / 0.136 / 0.136 |
| pedestrian | 3,754 | 138,030 | 0.6871 | 0.668 / 0.686 / 0.692 / 0.703 | 0.694 / 0.704 / 0.707 / 0.712 | 0.128 / 0.128 / 0.128 / 0.128 |
| **ALL** | 16,365 | 228,510 | 0.7132 | — | — | — |

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.5202**

| class_name | GTs | Preds | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | ---: | :---- | :---- | :---- |
| car | 3,018 | 39,277 | 0.6989 | 0.552 / 0.696 / 0.765 / 0.783 | 0.661 / 0.741 / 0.775 / 0.784 | 0.191 / 0.179 / 0.162 / 0.162 |
| truck | 602 | 19,133 | 0.6297 | 0.313 / 0.662 / 0.763 / 0.781 | 0.527 / 0.736 / 0.793 / 0.800 | 0.206 / 0.192 / 0.189 / 0.189 |
| bus | 60 | 4,162 | 0.4058 | 0.201 / 0.437 / 0.492 / 0.492 | 0.410 / 0.512 / 0.540 / 0.540 | 0.515 / 0.150 / 0.058 / 0.058 |
| bicycle | 85 | 7,974 | 0.3609 | 0.256 / 0.389 / 0.399 / 0.399 | 0.431 / 0.514 / 0.521 / 0.521 | 0.172 / 0.172 / 0.099 / 0.099 |
| pedestrian | 1,121 | 91,114 | 0.5056 | 0.489 / 0.504 / 0.509 / 0.521 | 0.597 / 0.606 / 0.609 / 0.612 | 0.125 / 0.125 / 0.125 / 0.125 |
| **ALL** | 4,886 | 161,660 | 0.5202 | — | — | — |

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.7995**

| class_name | GTs | Preds | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | ---: | :---- | :---- | :---- |
| car | 28,895 | 127,245 | 0.8640 | 0.783 / 0.862 / 0.897 / 0.915 | 0.840 / 0.883 / 0.898 / 0.904 | 0.213 / 0.191 / 0.153 / 0.153 |
| truck | 2,806 | 56,713 | 0.7788 | 0.579 / 0.794 / 0.860 / 0.881 | 0.703 / 0.833 / 0.864 / 0.868 | 0.215 / 0.195 / 0.168 / 0.168 |
| bus | 539 | 9,924 | 0.8608 | 0.718 / 0.902 / 0.910 / 0.913 | 0.811 / 0.881 / 0.881 / 0.881 | 0.378 / 0.334 / 0.334 / 0.334 |
| bicycle | 1,288 | 23,746 | 0.7272 | 0.640 / 0.738 / 0.761 / 0.770 | 0.727 / 0.767 / 0.771 / 0.774 | 0.187 / 0.187 / 0.148 / 0.148 |
| pedestrian | 9,934 | 352,908 | 0.7669 | 0.749 / 0.765 / 0.772 / 0.781 | 0.758 / 0.767 / 0.771 / 0.775 | 0.146 / 0.139 / 0.138 / 0.140 |
| **ALL** | 43,462 | 570,536 | 0.7995 | — | — | — |

---

**J6Gen2**: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 + db_j6gen2_v6 + db_j6gen2_v7 + db_j6gen2_v8 + db_j6gen2_v9 (3,951 frames)

**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.8702**

| class_name | GTs | Preds | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | ---: | :---- | :---- | :---- |
| car | 49,637 | 102,540 | 0.8758 | 0.818 / 0.869 / 0.897 / 0.919 | 0.879 / 0.910 / 0.920 / 0.927 | 0.266 / 0.194 / 0.158 / 0.116 |
| truck | 5,754 | 37,854 | 0.8410 | 0.711 / 0.832 / 0.893 / 0.927 | 0.786 / 0.856 / 0.890 / 0.910 | 0.215 / 0.170 / 0.170 / 0.157 |
| bus | 1,939 | 7,207 | 0.9408 | 0.864 / 0.935 / 0.979 / 0.984 | 0.902 / 0.941 / 0.960 / 0.963 | 0.201 / 0.133 / 0.133 / 0.033 |
| bicycle | 639 | 14,352 | 0.8590 | 0.841 / 0.865 / 0.865 / 0.865 | 0.860 / 0.871 / 0.871 / 0.871 | 0.163 / 0.155 / 0.155 / 0.155 |
| pedestrian | 14,362 | 395,185 | 0.8344 | 0.807 / 0.832 / 0.843 / 0.855 | 0.803 / 0.816 / 0.821 / 0.828 | 0.170 / 0.168 / 0.168 / 0.168 |
| **ALL** | 72,331 | 557,138 | 0.8702 | — | — | — |

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.6708**

| class_name | GTs | Preds | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | ---: | :---- | :---- | :---- |
| car | 47,568 | 189,993 | 0.7721 | 0.629 / 0.764 / 0.832 / 0.864 | 0.736 / 0.816 / 0.850 / 0.865 | 0.230 / 0.177 / 0.152 / 0.144 |
| truck | 4,090 | 85,129 | 0.6421 | 0.439 / 0.620 / 0.732 / 0.777 | 0.599 / 0.714 / 0.771 / 0.790 | 0.191 / 0.191 / 0.191 / 0.191 |
| bus | 1,935 | 19,431 | 0.7731 | 0.540 / 0.754 / 0.886 / 0.912 | 0.648 / 0.786 / 0.861 / 0.876 | 0.229 / 0.128 / 0.104 / 0.069 |
| bicycle | 295 | 23,124 | 0.5472 | 0.485 / 0.564 / 0.567 / 0.572 | 0.629 / 0.676 / 0.676 / 0.676 | 0.145 / 0.145 / 0.145 / 0.168 |
| pedestrian | 6,529 | 435,365 | 0.6192 | 0.588 / 0.615 / 0.629 / 0.644 | 0.654 / 0.668 / 0.673 / 0.681 | 0.140 / 0.144 / 0.144 / 0.140 |
| **ALL** | 60,417 | 753,042 | 0.6708 | — | — | — |

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.4462**

| class_name | GTs | Preds | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | ---: | :---- | :---- | :---- |
| car | 17,353 | 149,618 | 0.6346 | 0.437 / 0.619 / 0.722 / 0.761 | 0.595 / 0.698 / 0.750 / 0.770 | 0.182 / 0.157 / 0.155 / 0.140 |
| truck | 2,570 | 71,927 | 0.4758 | 0.184 / 0.409 / 0.609 / 0.701 | 0.401 / 0.569 / 0.690 / 0.739 | 0.195 / 0.138 / 0.137 / 0.130 |
| bus | 316 | 18,616 | 0.3215 | 0.075 / 0.308 / 0.432 / 0.471 | 0.284 / 0.464 / 0.538 / 0.556 | 0.078 / 0.058 / 0.059 / 0.059 |
| bicycle | 70 | 17,451 | 0.4303 | 0.402 / 0.438 / 0.440 / 0.440 | 0.574 / 0.591 / 0.591 / 0.591 | 0.193 / 0.193 / 0.193 / 0.193 |
| pedestrian | 1,673 | 251,053 | 0.3688 | 0.355 / 0.364 / 0.373 / 0.382 | 0.500 / 0.505 / 0.509 / 0.513 | 0.142 / 0.142 / 0.142 / 0.128 |
| **ALL** | 21,982 | 508,665 | 0.4462 | — | — | — |

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.7712**

| class_name | GTs | Preds | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | ---: | :---- | :---- | :---- |
| car | 114,558 | 442,151 | 0.8110 | 0.699 / 0.803 / 0.857 / 0.886 | 0.780 / 0.842 / 0.869 / 0.880 | 0.230 / 0.180 / 0.158 / 0.141 |
| truck | 12,414 | 194,910 | 0.7129 | 0.522 / 0.688 / 0.795 / 0.847 | 0.654 / 0.757 / 0.815 / 0.841 | 0.215 / 0.191 / 0.162 / 0.155 |
| bus | 4,190 | 45,254 | 0.8348 | 0.669 / 0.822 / 0.915 / 0.933 | 0.750 / 0.841 / 0.888 / 0.897 | 0.231 / 0.137 / 0.104 / 0.113 |
| bicycle | 1,004 | 54,927 | 0.7458 | 0.714 / 0.754 / 0.757 / 0.758 | 0.777 / 0.798 / 0.798 / 0.799 | 0.161 / 0.170 / 0.170 / 0.170 |
| pedestrian | 22,564 | 1,081,603 | 0.7515 | 0.723 / 0.748 / 0.761 / 0.774 | 0.741 / 0.753 / 0.759 / 0.766 | 0.161 / 0.161 / 0.161 / 0.161 |
| **ALL** | 154,730 | 1,818,845 | 0.7712 | — | — | — |

---

**JPNTaxi_Gen2**: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (9,975 frames)

**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.8784**

| class_name | GTs | Preds | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | ---: | :---- | :---- | :---- |
| car | 42,789 | 158,413 | 0.9436 | 0.892 / 0.946 / 0.965 / 0.972 | 0.916 / 0.950 / 0.958 / 0.959 | 0.237 / 0.173 / 0.144 / 0.144 |
| truck | 17,259 | 113,284 | 0.8531 | 0.696 / 0.840 / 0.926 / 0.950 | 0.803 / 0.880 / 0.930 / 0.943 | 0.298 / 0.195 / 0.169 / 0.169 |
| bus | 3,437 | 25,085 | 0.8284 | 0.712 / 0.818 / 0.889 / 0.894 | 0.758 / 0.823 / 0.866 / 0.868 | 0.024 / 0.024 / 0.024 / 0.024 |
| bicycle | 2,681 | 50,072 | 0.8546 | 0.842 / 0.857 / 0.859 / 0.860 | 0.864 / 0.870 / 0.871 / 0.871 | 0.243 / 0.243 / 0.229 / 0.229 |
| pedestrian | 57,948 | 909,802 | 0.9123 | 0.895 / 0.911 / 0.918 / 0.925 | 0.872 / 0.881 / 0.888 / 0.893 | 0.148 / 0.148 / 0.148 / 0.140 |
| **ALL** | 124,114 | 1,256,656 | 0.8784 | — | — | — |

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.6692**

| class_name | GTs | Preds | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | ---: | :---- | :---- | :---- |
| car | 35,518 | 316,517 | 0.8323 | 0.747 / 0.835 / 0.867 / 0.880 | 0.810 / 0.852 / 0.866 / 0.871 | 0.232 / 0.167 / 0.159 / 0.157 |
| truck | 22,550 | 254,916 | 0.6571 | 0.424 / 0.616 / 0.763 / 0.824 | 0.598 / 0.715 / 0.797 / 0.825 | 0.235 / 0.193 / 0.155 / 0.126 |
| bus | 2,683 | 74,300 | 0.4033 | 0.129 / 0.358 / 0.545 / 0.583 | 0.303 / 0.484 / 0.601 / 0.624 | 0.042 / 0.044 / 0.044 / 0.044 |
| bicycle | 1,607 | 75,955 | 0.6721 | 0.636 / 0.682 / 0.685 / 0.686 | 0.723 / 0.743 / 0.743 / 0.743 | 0.172 / 0.172 / 0.172 / 0.172 |
| pedestrian | 27,240 | 1,114,768 | 0.7812 | 0.763 / 0.779 / 0.788 / 0.795 | 0.765 / 0.773 / 0.778 / 0.782 | 0.158 / 0.168 / 0.153 / 0.153 |
| **ALL** | 89,598 | 1,836,456 | 0.6692 | — | — | — |

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.5300**

| class_name | GTs | Preds | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | ---: | :---- | :---- | :---- |
| car | 16,524 | 290,114 | 0.6692 | 0.544 / 0.678 / 0.719 / 0.737 | 0.655 / 0.731 / 0.749 / 0.754 | 0.168 / 0.160 / 0.159 / 0.140 |
| truck | 14,587 | 232,484 | 0.5020 | 0.192 / 0.444 / 0.624 / 0.748 | 0.419 / 0.598 / 0.709 / 0.771 | 0.268 / 0.203 / 0.163 / 0.145 |
| bus | 2,476 | 76,725 | 0.2822 | 0.113 / 0.279 / 0.350 / 0.387 | 0.349 / 0.453 / 0.493 / 0.514 | 0.022 / 0.025 / 0.025 / 0.025 |
| bicycle | 364 | 51,256 | 0.4586 | 0.345 / 0.428 / 0.530 / 0.530 | 0.525 / 0.562 / 0.602 / 0.602 | 0.151 / 0.128 / 0.151 / 0.151 |
| pedestrian | 14,297 | 741,963 | 0.7380 | 0.715 / 0.734 / 0.744 / 0.760 | 0.736 / 0.745 / 0.750 / 0.759 | 0.134 / 0.134 / 0.134 / 0.133 |
| **ALL** | 48,248 | 1,392,542 | 0.5300 | — | — | — |

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.7471**

| class_name | GTs | Preds | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | ---: | :---- | :---- | :---- |
| car | 94,831 | 765,044 | 0.8667 | 0.788 / 0.870 / 0.898 / 0.910 | 0.835 / 0.879 / 0.891 / 0.894 | 0.214 / 0.168 / 0.160 / 0.159 |
| truck | 54,396 | 600,684 | 0.6928 | 0.460 / 0.655 / 0.794 / 0.862 | 0.624 / 0.744 / 0.823 / 0.855 | 0.285 / 0.199 / 0.173 / 0.155 |
| bus | 8,596 | 176,110 | 0.5446 | 0.351 / 0.528 / 0.637 / 0.662 | 0.489 / 0.598 / 0.660 / 0.673 | 0.027 / 0.027 / 0.027 / 0.028 |
| bicycle | 4,652 | 177,283 | 0.7710 | 0.744 / 0.775 / 0.782 / 0.783 | 0.791 / 0.804 / 0.808 / 0.809 | 0.184 / 0.184 / 0.184 / 0.184 |
| pedestrian | 99,485 | 2,766,533 | 0.8606 | 0.842 / 0.859 / 0.867 / 0.875 | 0.824 / 0.832 / 0.839 / 0.844 | 0.148 / 0.148 / 0.148 / 0.148 |
| **ALL** | 261,960 | 4,485,654 | 0.7471 | — | — | — |

</details>
