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
  <summary> j6gen2 (4,682 frames) </summary>

  - `db_j6gen2_v1`
  - `db_j6gen2_v2`
  - `db_j6gen2_v3`
  - `db_j6gen2_v4`
  - `db_j6gen2_v5`
  - `db_j6gen2_v6`
  - `db_j6gen2_v7`
  - `db_j6gen2_v8`
  - `db_j6gen2_v9`
  - `db_j6gen2_v10`
  - `db_j6gen2_v11`
  - `db_j6gen2_v12`

  </details>

  <details>
  <summary> largebus (1,228 frames) </summary>

  - `db_largebus_v1`
  - `db_largebus_v2`
  - `db_largebus_v3`

  </details>

  <details>
  <summary> j6gen2_base (5,910 frames) </summary>

  - `db_j6gen2_v1`
  - `db_j6gen2_v2`
  - `db_j6gen2_v3`
  - `db_j6gen2_v4`
  - `db_j6gen2_v5`
  - `db_j6gen2_v6`
  - `db_j6gen2_v7`
  - `db_j6gen2_v8`
  - `db_j6gen2_v9`
  - `db_j6gen2_v10`
  - `db_j6gen2_v11`
  - `db_j6gen2_v12`
  - `db_largebus_v1`
  - `db_largebus_v2`
  - `db_largebus_v3`

  </details>


### mAP - J6Gen2_base

- **Class mAP for BEV Center Distance: 0.5m, 1.0m, 2.0m, 4.0m**

  <details>
  <summary> Eval Range: 0.0 - 50.0m </summary>

  | Model version | mAP | mAPH | map_based_nds (recall @ 0.10) | map_based_nds (recall @ 0.40) | maph_based_nds (recall @ 0.10) | maph_based_nds (recall 0.40) | car<br>(75,589) | truck<br>(8,273) | bus<br>(2,706) | bicycle<br>(2,097) | pedestrian<br>(23,254) | traffic_cone<br>(8,310) | barrier<br>(1,350) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 0.7289 | 0.6808 | 0.6820 | 0.6757 | 0.6579 | 0.6516 | 0.9000 | 0.8398 | 0.9130 | 0.8907 | 0.8535 | 0.4465 | 0.2590 |

  </details>

  <details>
  <summary> Eval Range: 50.0 - 90.0m </summary>

  | Model version | mAP | mAPH | map_based_nds (recall @ 0.10) | map_based_nds (recall @ 0.40) | maph_based_nds (recall @ 0.10) | maph_based_nds (recall 0.40) | car<br>(64,960) | truck<br>(5,922) | bus<br>(2,257) | bicycle<br>(1,298) | pedestrian<br>(12,052) | traffic_cone<br>(2,636) | barrier<br>(622) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 0.5802 | 0.5217 | 0.5876 | 0.5690 | 0.5584 | 0.5398 | 0.8127 | 0.6518 | 0.7926 | 0.6527 | 0.6690 | 0.2760 | 0.2064 |

  </details>

  <details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version | mAP | mAPH | map_based_nds (recall @ 0.10) | map_based_nds (recall @ 0.40) | maph_based_nds (recall @ 0.10) | maph_based_nds (recall 0.40) | car<br>(22,141) | truck<br>(3,506) | bus<br>(544) | bicycle<br>(376) | pedestrian<br>(3,656) | traffic_cone<br>(462) | barrier<br>(145) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 0.4396 | 0.3969 | 0.5002 | 0.4696 | 0.4789 | 0.4483 | 0.7147 | 0.5324 | 0.5445 | 0.4977 | 0.4993 | 0.1329 | 0.1559 |

  </details>

  <details open>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version | mAP | mAPH | map_based_nds (recall @ 0.10) | map_based_nds (recall @ 0.40) | maph_based_nds (recall @ 0.10) | maph_based_nds (recall 0.40) | car<br>(162,690) | truck<br>(17,701) | bus<br>(5,507) | bicycle<br>(3,771) | pedestrian<br>(38,962) | traffic_cone<br>(11,408) | barrier<br>(2,117) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 0.6590 | 0.6053 | 0.6391 | 0.6214 | 0.6122 | 0.5946 | 0.8547 | 0.7285 | 0.8389 | 0.7843 | 0.7789 | 0.3955 | 0.2321 |

  </details>

### Mean TPError - J6Gen2_base

- Recalls: `0.10`, `0.40`, `optimal`

  <details>
  <summary> Eval Range: 0.0 - 50.0m </summary>

  | Model version | mATE (recall @ 0.10) | mAOE (recall @ 0.10) | mASE (recall @ 0.10) | mAVE (recall @ 0.10) | mAAE (recall @ 0.10) | mATE (recall @ 0.40) | mAOE (recall @ 0.40) | mASE (recall @ 0.40) | mAVE (recall @ 0.40) | mAAE (recall @ 0.40) | mATE (optimal) | mAOE (optimal) | mASE (optimal) | mAVE (optimal) | mAAE (optimal) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 0.1699 | 0.1847 | 0.2714 | 0.1985 | 1.0000 | 0.1878 | 0.2040 | 0.2866 | 0.2091 | 1.0000 | 0.2039 | 0.2176 | 0.2883 | 0.2138 | 1.0000 |

  <summary><strong>Num match summary</strong></summary>

  **recall 0.10**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 75,589) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 8,273) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 2,706) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 2,097) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 23,254) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 8,310) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 1,350) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 8,314 / 8,314 / 8,314 / 8,314 | 910 / 910 / 910 / 910 | 297 / 297 / 297 / 297 | 230 / 230 / 230 / 230 | 2,557 / 2,557 / 2,557 / 2,557 | 914 / 914 / 914 / 914 | 148 / 148 / 148 / 148 |

  **recall 0.40**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 75,589) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 8,273) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 2,706) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 2,097) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 23,254) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 8,310) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 1,350) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 30,991 / 30,991 / 30,991 / 30,991 | 3,391 / 3,391 / 3,391 / 3,391 | 1,109 / 1,109 / 1,109 / 1,109 | 859 / 859 / 859 / 859 | 9,534 / 9,534 / 9,534 / 9,534 | 3,407 / 3,407 / 3,407 / 3,407 | 553 / 553 / 553 / 553 |

  **optimal**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 75,589) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 8,273) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 2,706) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 2,097) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 23,254) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 8,310) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 1,350) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 65,297 / 67,749 / 68,945 / 69,400 | 6,298 / 7,033 / 7,266 / 7,405 | 2,282 / 2,494 / 2,565 / 2,589 | 1,789 / 1,815 / 1,822 / 1,825 | 18,701 / 19,099 / 19,305 / 19,357 | 4,529 / 4,841 / 5,046 / 5,561 | 453 / 529 / 550 / 562 |

  </details>

  <details>
  <summary> Eval Range: 50.0 - 90.0m </summary>

  | Model version | mATE (recall @ 0.10) | mAOE (recall @ 0.10) | mASE (recall @ 0.10) | mAVE (recall @ 0.10) | mAAE (recall @ 0.10) | mATE (recall @ 0.40) | mAOE (recall @ 0.40) | mASE (recall @ 0.40) | mAVE (recall @ 0.40) | mAAE (recall @ 0.40) | mATE (optimal) | mAOE (optimal) | mASE (optimal) | mAVE (optimal) | mAAE (optimal) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 0.2380 | 0.2382 | 0.2946 | 0.2537 | 1.0000 | 0.2943 | 0.2886 | 0.3273 | 0.3009 | 1.0000 | 0.2809 | 0.2785 | 0.3078 | 0.2721 | 1.0000 |

  <summary><strong>Num match summary</strong></summary>

  **recall 0.10**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 64,960) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 5,922) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 2,257) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 1,298) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 12,052) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 2,636) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 622) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 7,145 / 7,145 / 7,145 / 7,145 | 651 / 651 / 651 / 651 | 248 / 248 / 248 / 248 | 142 / 142 / 142 / 142 | 1,325 / 1,325 / 1,325 / 1,325 | 289 / 289 / 289 / 289 | 68 / 68 / 68 / 68 |

  **recall 0.40**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 64,960) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 5,922) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 2,257) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 1,298) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 12,052) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 2,636) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 622) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 26,633 / 26,633 / 26,633 / 26,633 | 2,428 / 2,428 / 2,428 / 2,428 | 925 / 925 / 925 / 925 | 532 / 532 / 532 / 532 | 4,941 / 4,941 / 4,941 / 4,941 | 1,080 / 1,080 / 1,080 / 1,080 | 0 / 255 / 255 / 255 |

  **optimal**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 64,960) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 5,922) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 2,257) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 1,298) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 12,052) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 2,636) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 622) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 47,451 / 52,827 / 55,318 / 55,889 | 3,253 / 4,045 / 4,354 / 4,471 | 1,367 / 1,826 / 1,951 / 1,981 | 839 / 846 / 901 / 902 | 8,085 / 8,202 / 8,275 / 8,325 | 1,120 / 1,225 / 1,270 / 1,347 | 156 / 231 / 237 / 268 |

  </details>

  <details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version | mATE (recall @ 0.10) | mAOE (recall @ 0.10) | mASE (recall @ 0.10) | mAVE (recall @ 0.10) | mAAE (recall @ 0.10) | mATE (recall @ 0.40) | mAOE (recall @ 0.40) | mASE (recall @ 0.40) | mAVE (recall @ 0.40) | mAAE (recall @ 0.40) | mATE (optimal) | mAOE (optimal) | mASE (optimal) | mAVE (optimal) | mAAE (optimal) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 0.3252 | 0.2112 | 0.3207 | 0.3389 | 1.0000 | 0.4093 | 0.3020 | 0.3601 | 0.4304 | 1.0000 | 0.3625 | 0.2467 | 0.3279 | 0.3624 | 1.0000 |

  <summary><strong>Num match summary</strong></summary>

  **recall 0.10**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 22,141) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 3,506) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 544) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 376) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 3,656) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 462) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 145) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 2,435 / 2,435 / 2,435 / 2,435 | 385 / 385 / 385 / 385 | 59 / 59 / 59 / 59 | 41 / 41 / 41 / 41 | 402 / 402 / 402 / 402 | 50 / 50 / 50 / 50 | 15 / 15 / 15 / 15 |

  **recall 0.40**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 22,141) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 3,506) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 544) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 376) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 3,656) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 462) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 145) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 9,077 / 9,077 / 9,077 / 9,077 | 1,437 / 1,437 / 1,437 / 1,437 | 223 / 223 / 223 / 223 | 154 / 154 / 154 / 154 | 1,498 / 1,498 / 1,498 / 1,498 | 0 / 189 / 189 / 189 | 0 / 59 / 59 / 59 |

  **optimal**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 22,141) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 3,506) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 544) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 376) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 3,656) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 462) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 145) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 13,838 / 16,236 / 17,440 / 17,774 | 1,483 / 1,981 / 2,484 / 2,599 | 192 / 312 / 394 / 405 | 187 / 218 / 223 / 223 | 2,141 / 2,167 / 2,181 / 2,199 | 132 / 147 / 178 / 186 | 33 / 52 / 72 / 73 |

  </details>

  <details open>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version | mATE (recall @ 0.10) | mAOE (recall @ 0.10) | mASE (recall @ 0.10) | mAVE (recall @ 0.10) | mAAE (recall @ 0.10) | mATE (recall @ 0.40) | mAOE (recall @ 0.40) | mASE (recall @ 0.40) | mAVE (recall @ 0.40) | mAAE (recall @ 0.40) | mATE (optimal) | mAOE (optimal) | mASE (optimal) | mAVE (optimal) | mAAE (optimal) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 0.1972 | 0.2036 | 0.2826 | 0.2208 | 1.0000 | 0.2503 | 0.2473 | 0.3146 | 0.2684 | 1.0000 | 0.2405 | 0.2402 | 0.2976 | 0.2433 | 1.0000 |

  <summary><strong>Num match summary</strong></summary>

  **recall 0.10**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 162,690) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 17,701) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 5,507) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 3,771) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 38,962) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 11,408) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 2,117) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 17,895 / 17,895 / 17,895 / 17,895 | 1,947 / 1,947 / 1,947 / 1,947 | 605 / 605 / 605 / 605 | 414 / 414 / 414 / 414 | 4,285 / 4,285 / 4,285 / 4,285 | 1,254 / 1,254 / 1,254 / 1,254 | 232 / 232 / 232 / 232 |

  **recall 0.40**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 162,690) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 17,701) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 5,507) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 3,771) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 38,962) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 11,408) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 2,117) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 66,702 / 66,702 / 66,702 / 66,702 | 7,257 / 7,257 / 7,257 / 7,257 | 2,257 / 2,257 / 2,257 / 2,257 | 1,546 / 1,546 / 1,546 / 1,546 | 15,974 / 15,974 / 15,974 / 15,974 | 4,677 / 4,677 / 4,677 / 4,677 | 0 / 867 / 867 / 867 |

  **optimal**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 162,690) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 17,701) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 5,507) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 3,771) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 38,962) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 11,408) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 2,117) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 126,514 / 137,022 / 141,450 / 143,300 | 10,859 / 13,045 / 14,027 / 14,439 | 3,844 / 4,623 / 4,831 / 4,974 | 2,732 / 2,840 / 2,856 / 2,859 | 28,970 / 29,422 / 29,647 / 29,660 | 5,644 / 6,197 / 6,356 / 7,069 | 602 / 827 / 850 / 871 |

  </details>

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

  | Model version | mAP | mAPH | map_based_nds (recall @ 0.10) | map_based_nds (recall @ 0.40) | maph_based_nds (recall @ 0.10) | maph_based_nds (recall 0.40) | car<br>(14,872) | truck<br>(1,192) | bus<br>(336) | bicycle<br>(740) | pedestrian<br>(5,055) | traffic_cone<br>(60) | barrier<br>(0) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 0.6313 | 0.6015 | 0.5746 | 0.5493 | 0.5597 | 0.5344 | 0.9156 | 0.8702 | 0.9160 | 0.8586 | 0.8588 | 0.0000 | 0.0000 |

  </details>

  <details>
  <summary> Eval Range: 50.0 - 90.0m </summary>

  | Model version | mAP | mAPH | map_based_nds (recall @ 0.10) | map_based_nds (recall @ 0.40) | maph_based_nds (recall @ 0.10) | maph_based_nds (recall 0.40) | car<br>(10,929) | truck<br>(1,009) | bus<br>(141) | bicycle<br>(460) | pedestrian<br>(3,721) | traffic_cone<br>(4) | barrier<br>(0) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 0.5281 | 0.4877 | 0.4942 | 0.4916 | 0.4740 | 0.4714 | 0.8442 | 0.7108 | 0.8522 | 0.5764 | 0.7129 | 0.0000 | 0.0000 |

  </details>

  <details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version | mAP | mAPH | map_based_nds (recall @ 0.10) | map_based_nds (recall @ 0.40) | maph_based_nds (recall @ 0.10) | maph_based_nds (recall 0.40) | car<br>(2,883) | truck<br>(600) | bus<br>(60) | bicycle<br>(85) | pedestrian<br>(1,092) | traffic_cone<br>(0) | barrier<br>(0) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 0.4172 | 0.3831 | 0.4189 | 0.4104 | 0.4018 | 0.3934 | 0.7548 | 0.6586 | 0.5716 | 0.3759 | 0.5594 | 0.0000 | 0.0000 |

  </details>

  <details>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version | mAP | mAPH | map_based_nds (recall @ 0.10) | map_based_nds (recall @ 0.40) | maph_based_nds (recall @ 0.10) | maph_based_nds (recall 0.40) | car<br>(28,684) | truck<br>(2,801) | bus<br>(537) | bicycle<br>(1,285) | pedestrian<br>(9,868) | traffic_cone<br>(64) | barrier<br>(0) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 0.5779 | 0.5432 | 0.5404 | 0.5154 | 0.5230 | 0.4980 | 0.8813 | 0.7754 | 0.8642 | 0.7410 | 0.7836 | 0.0000 | 0.0000 |

  </details>

- **Mean TPError - LargeBus**

  <details>
  <summary> Eval Range: 0.0 - 50.0m </summary>

  | Model version | mATE (recall @ 0.10) | mAOE (recall @ 0.10) | mASE (recall @ 0.10) | mAVE (recall @ 0.10) | mAAE (recall @ 0.10) | mATE (recall @ 0.40) | mAOE (recall @ 0.40) | mASE (recall @ 0.40) | mAVE (recall @ 0.40) | mAAE (recall @ 0.40) | mATE (optimal) | mAOE (optimal) | mASE (optimal) | mAVE (optimal) | mAAE (optimal) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 0.2732 | 0.4747 | 0.3197 | 0.3430 | 1.0000 | 0.3937 | 0.3922 | 0.3996 | 0.4786 | 1.0000 | 0.1883 | 0.3943 | 0.2227 | 0.2329 | 1.0000 |

  <summary><strong>Num match summary</strong></summary>

  **recall 0.10**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 14,872) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 1,192) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 336) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 740) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 5,055) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 60) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 0) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 1,635 / 1,635 / 1,635 / 1,635 | 131 / 131 / 131 / 131 | 36 / 36 / 36 / 36 | 81 / 81 / 81 / 81 | 556 / 556 / 556 / 556 | 6 / 6 / 6 / 6 | 0 / 0 / 0 / 0 |

  **recall 0.40**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 14,872) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 1,192) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 336) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 740) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 5,055) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 60) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 0) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 6,097 / 6,097 / 6,097 / 6,097 | 488 / 488 / 488 / 488 | 137 / 137 / 137 / 137 | 303 / 303 / 303 / 303 | 2,072 / 2,072 / 2,072 / 2,072 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

  **optimal**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 14,872) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 1,192) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 336) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 740) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 5,055) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 60) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 0) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 13,178 / 13,676 / 13,748 / 13,798 | 925 / 1,041 / 1,064 / 1,073 | 254 / 330 / 333 / 333 | 612 / 628 / 640 / 643 | 4,247 / 4,294 / 4,313 / 4,330 | 19 / 20 / 20 / 21 | 0 / 0 / 0 / 0 |

  </details>

  <details>
  <summary> Eval Range: 50.0 - 90.0m </summary>

  | Model version | mATE (recall @ 0.10) | mAOE (recall @ 0.10) | mASE (recall @ 0.10) | mAVE (recall @ 0.10) | mAAE (recall @ 0.10) | mATE (recall @ 0.40) | mAOE (recall @ 0.40) | mASE (recall @ 0.40) | mAVE (recall @ 0.40) | mAAE (recall @ 0.40) | mATE (optimal) | mAOE (optimal) | mASE (optimal) | mAVE (optimal) | mAAE (optimal) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 0.3091 | 0.6981 | 0.3081 | 0.3833 | 1.0000 | 0.3181 | 0.6966 | 0.3115 | 0.3980 | 1.0000 | 0.2197 | 0.6583 | 0.2015 | 0.3121 | 1.0000 |

  <summary><strong>Num match summary</strong></summary>

  **recall 0.10**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 10,929) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 1,009) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 141) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 460) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 3,721) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 4) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 0) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 1,202 / 1,202 / 1,202 / 1,202 | 110 / 110 / 110 / 110 | 15 / 15 / 15 / 15 | 50 / 50 / 50 / 50 | 409 / 409 / 409 / 409 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

  **recall 0.40**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 10,929) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 1,009) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 141) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 460) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 3,721) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 4) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 0) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 4,480 / 4,480 / 4,480 / 4,480 | 413 / 413 / 413 / 413 | 57 / 57 / 57 / 57 | 188 / 188 / 188 / 188 | 1,525 / 1,525 / 1,525 / 1,525 | 1 / 1 / 1 / 1 | 0 / 0 / 0 / 0 |

  **optimal**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 10,929) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 1,009) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 141) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 460) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 3,721) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 4) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 0) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 8,463 / 9,288 / 9,554 / 9,621 | 617 / 739 / 799 / 804 | 103 / 124 / 124 / 124 | 263 / 289 / 292 / 292 | 2,604 / 2,652 / 2,667 / 2,682 | 2 / 2 / 2 / 2 | 0 / 0 / 0 / 0 |

  </details>

  <details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version | mATE (recall @ 0.10) | mAOE (recall @ 0.10) | mASE (recall @ 0.10) | mAVE (recall @ 0.10) | mAAE (recall @ 0.10) | mATE (recall @ 0.40) | mAOE (recall @ 0.40) | mASE (recall @ 0.40) | mAVE (recall @ 0.40) | mAAE (recall @ 0.40) | mATE (optimal) | mAOE (optimal) | mASE (optimal) | mAVE (optimal) | mAAE (optimal) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 0.4683 | 0.4107 | 0.4227 | 0.5956 | 1.0000 | 0.4839 | 0.4292 | 0.4259 | 0.6425 | 1.0000 | 0.2920 | 0.2122 | 0.1980 | 0.5124 | 1.0000 |

  <summary><strong>Num match summary</strong></summary>

  **recall 0.10**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 2,883) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 600) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 60) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 85) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 1,092) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 0) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 0) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 317 / 317 / 317 / 317 | 66 / 66 / 66 / 66 | 6 / 6 / 6 / 6 | 9 / 9 / 9 / 9 | 120 / 120 / 120 / 120 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

  **recall 0.40**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 2,883) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 600) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 60) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 85) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 1,092) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 0) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 0) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 1,182 / 1,182 / 1,182 / 1,182 | 246 / 246 / 246 / 246 | 24 / 24 / 24 / 24 | 34 / 34 / 34 / 34 | 447 / 447 / 447 / 447 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

  **optimal**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 2,883) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 600) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 60) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 85) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 1,092) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 0) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 0) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 1,929 / 2,144 / 2,236 / 2,337 | 301 / 408 / 466 / 475 | 26 / 32 / 41 / 41 | 39 / 42 / 46 / 46 | 691 / 697 / 700 / 730 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

  </details>

  <details>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version | mATE (recall @ 0.10) | mAOE (recall @ 0.10) | mASE (recall @ 0.10) | mAVE (recall @ 0.10) | mAAE (recall @ 0.10) | mATE (recall @ 0.40) | mAOE (recall @ 0.40) | mASE (recall @ 0.40) | mAVE (recall @ 0.40) | mAAE (recall @ 0.40) | mATE (optimal) | mAOE (optimal) | mASE (optimal) | mAVE (optimal) | mAAE (optimal) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 0.2907 | 0.5134 | 0.3242 | 0.3572 | 1.0000 | 0.4150 | 0.4172 | 0.4068 | 0.4972 | 1.0000 | 0.2100 | 0.4414 | 0.2253 | 0.2736 | 1.0000 |

  <summary><strong>Num match summary</strong></summary>

  **recall 0.10**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 28,684) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 2,801) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 537) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 1,285) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 9,868) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 64) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 0) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 3,155 / 3,155 / 3,155 / 3,155 | 308 / 308 / 308 / 308 | 59 / 59 / 59 / 59 | 141 / 141 / 141 / 141 | 1,085 / 1,085 / 1,085 / 1,085 | 7 / 7 / 7 / 7 | 0 / 0 / 0 / 0 |

  **recall 0.40**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 28,684) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 2,801) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 537) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 1,285) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 9,868) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 64) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 0) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 11,760 / 11,760 / 11,760 / 11,760 | 1,148 / 1,148 / 1,148 / 1,148 | 220 / 220 / 220 / 220 | 526 / 526 / 526 / 526 | 4,045 / 4,045 / 4,045 / 4,045 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

  **optimal**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 28,684) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 2,801) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 537) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 1,285) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 9,868) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 64) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 0) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 23,528 / 24,950 / 25,596 / 25,735 | 1,866 / 2,196 / 2,328 / 2,367 | 379 / 486 / 490 / 490 | 874 / 918 / 941 / 944 | 7,461 / 7,553 / 7,587 / 7,622 | 19 / 22 / 22 / 23 | 0 / 0 / 0 / 0 |

  </details>

</details>

<details>
<summary> J6Gen2 </summary>

- Datasets (4,682 Testing Frames):
  - `db_j6gen2_v1`
  - `db_j6gen2_v2`
  - `db_j6gen2_v3`
  - `db_j6gen2_v4`
  - `db_j6gen2_v5`
  - `db_j6gen2_v6`
  - `db_j6gen2_v7`
  - `db_j6gen2_v8`
  - `db_j6gen2_v9`
  - `db_j6gen2_v10`
  - `db_j6gen2_v11`
  - `db_j6gen2_v12`

- **Class mAP for BEV Center Distance: 0.5m, 1.0m, 2.0m, 4.0m**

  <details>
  <summary> Eval Range: 0.0 - 50.0m </summary>

  | Model version | mAP | mAPH | map_based_nds (recall @ 0.10) | map_based_nds (recall @ 0.40) | maph_based_nds (recall @ 0.10) | maph_based_nds (recall 0.40) | car<br>(60,938) | truck<br>(7,081) | bus<br>(2,370) | bicycle<br>(1,357) | pedestrian<br>(18,202) | traffic_cone<br>(8,250) | barrier<br>(1,350) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 0.7371 | 0.6887 | 0.6863 | 0.6799 | 0.6621 | 0.6558 | 0.8940 | 0.8368 | 0.9124 | 0.9072 | 0.8537 | 0.4940 | 0.2617 |

  </details>

  <details>
  <summary> Eval Range: 50.0 - 90.0m </summary>

  | Model version | mAP | mAPH | map_based_nds (recall @ 0.10) | map_based_nds (recall @ 0.40) | maph_based_nds (recall @ 0.10) | maph_based_nds (recall 0.40) | car<br>(54,217) | truck<br>(4,913) | bus<br>(2,116) | bicycle<br>(838) | pedestrian<br>(8,336) | traffic_cone<br>(2,632) | barrier<br>(622) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 0.5833 | 0.5245 | 0.5890 | 0.5704 | 0.5596 | 0.5410 | 0.8044 | 0.6387 | 0.7893 | 0.6949 | 0.6496 | 0.2967 | 0.2096 |

  </details>

  <details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version | mAP | mAPH | map_based_nds (recall @ 0.10) | map_based_nds (recall @ 0.40) | maph_based_nds (recall @ 0.10) | maph_based_nds (recall 0.40) | car<br>(19,301) | truck<br>(2,906) | bus<br>(484) | bicycle<br>(291) | pedestrian<br>(2,564) | traffic_cone<br>(462) | barrier<br>(145) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 0.4384 | 0.3949 | 0.4973 | 0.4673 | 0.4756 | 0.4456 | 0.7075 | 0.5046 | 0.5412 | 0.5343 | 0.4732 | 0.1509 | 0.1571 |

  </details>

  <details>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version | mAP | mAPH | map_based_nds (recall @ 0.10) | map_based_nds (recall @ 0.40) | maph_based_nds (recall @ 0.10) | maph_based_nds (recall 0.40) | car<br>(134,456) | truck<br>(14,900) | bus<br>(4,970) | bicycle<br>(2,486) | pedestrian<br>(29,102) | traffic_cone<br>(11,344) | barrier<br>(2,117) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 0.6650 | 0.6111 | 0.6420 | 0.6241 | 0.6150 | 0.5972 | 0.8448 | 0.7186 | 0.8363 | 0.8063 | 0.7779 | 0.4361 | 0.2350 |

  </details>

- **Mean TPError - J6Gen2**

  <details>
  <summary> Eval Range: 0.0 - 50.0m </summary>

  | Model version | mATE (recall @ 0.10) | mAOE (recall @ 0.10) | mASE (recall @ 0.10) | mAVE (recall @ 0.10) | mAAE (recall @ 0.10) | mATE (recall @ 0.40) | mAOE (recall @ 0.40) | mASE (recall @ 0.40) | mAVE (recall @ 0.40) | mAAE (recall @ 0.40) | mATE (optimal) | mAOE (optimal) | mASE (optimal) | mAVE (optimal) | mAAE (optimal) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 0.1692 | 0.1845 | 0.2711 | 0.1983 | 1.0000 | 0.1868 | 0.2021 | 0.2864 | 0.2108 | 1.0000 | 0.2026 | 0.2155 | 0.2887 | 0.2184 | 1.0000 |

  <details>
  <summary><strong>Num match summary</strong></summary>

  **recall 0.10**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 60,938) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 7,081) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 2,370) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 1,357) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 18,202) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 8,250) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 1,350) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 6,703 / 6,703 / 6,703 / 6,703 | 778 / 778 / 778 / 778 | 260 / 260 / 260 / 260 | 149 / 149 / 149 / 149 | 2,002 / 2,002 / 2,002 / 2,002 | 907 / 907 / 907 / 907 | 148 / 148 / 148 / 148 |

  **recall 0.40**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 60,938) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 7,081) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 2,370) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 1,357) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 18,202) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 8,250) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 1,350) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 24,984 / 24,984 / 24,984 / 24,984 | 2,903 / 2,903 / 2,903 / 2,903 | 971 / 971 / 971 / 971 | 556 / 556 / 556 / 556 | 7,462 / 7,462 / 7,462 / 7,462 | 3,382 / 3,382 / 3,382 / 3,382 | 553 / 553 / 553 / 553 |

  **optimal**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 60,938) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 7,081) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 2,370) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 1,357) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 18,202) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 8,250) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 1,350) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 52,338 / 54,149 / 55,189 / 55,565 | 5,350 / 5,996 / 6,202 / 6,337 | 2,027 / 2,173 / 2,227 / 2,232 | 1,170 / 1,181 / 1,182 / 1,182 | 14,547 / 14,883 / 15,058 / 15,106 | 4,519 / 4,942 / 5,249 / 5,546 | 453 / 538 / 550 / 562 |

  </details>

  <details>
  <summary> Eval Range: 50.0 - 90.0m </summary>

  | Model version | mATE (recall @ 0.10) | mAOE (recall @ 0.10) | mASE (recall @ 0.10) | mAVE (recall @ 0.10) | mAAE (recall @ 0.10) | mATE (recall @ 0.40) | mAOE (recall @ 0.40) | mASE (recall @ 0.40) | mAVE (recall @ 0.40) | mAAE (recall @ 0.40) | mATE (optimal) | mAOE (optimal) | mASE (optimal) | mAVE (optimal) | mAAE (optimal) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 0.2376 | 0.2394 | 0.2946 | 0.2552 | 1.0000 | 0.2952 | 0.2891 | 0.3267 | 0.3020 | 1.0000 | 0.2820 | 0.2779 | 0.3072 | 0.2730 | 1.0000 |

  <summary><strong>Num match summary</strong></summary>

  **recall 0.10**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 54,217) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 4,913) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 2,116) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 838) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 8,336) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 2,632) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 622) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 5,963 / 5,963 / 5,963 / 5,963 | 540 / 540 / 540 / 540 | 232 / 232 / 232 / 232 | 92 / 92 / 92 / 92 | 916 / 916 / 916 / 916 | 289 / 289 / 289 / 289 | 68 / 68 / 68 / 68 |

  **recall 0.40**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 54,217) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 4,913) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 2,116) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 838) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 8,336) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 2,632) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 622) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 22,228 / 22,228 / 22,228 / 22,228 | 2,014 / 2,014 / 2,014 / 2,014 | 867 / 867 / 867 / 867 | 343 / 343 / 343 / 343 | 3,417 / 3,417 / 3,417 / 3,417 | 1,079 / 1,079 / 1,079 / 1,079 | 0 / 255 / 255 / 255 |

  **optimal**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 54,217) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 4,913) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 2,116) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 838) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 8,336) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 2,632) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 622) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 39,059 / 43,688 / 45,797 / 46,308 | 2,628 / 3,307 / 3,561 / 3,660 | 1,261 / 1,700 / 1,825 / 1,855 | 584 / 579 / 581 / 611 | 5,589 / 5,531 / 5,588 / 5,745 | 1,142 / 1,223 / 1,314 / 1,394 | 156 / 231 / 237 / 268 |

  </details>

  <details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version | mATE (recall @ 0.10) | mAOE (recall @ 0.10) | mASE (recall @ 0.10) | mAVE (recall @ 0.10) | mAAE (recall @ 0.10) | mATE (recall @ 0.40) | mAOE (recall @ 0.40) | mASE (recall @ 0.40) | mAVE (recall @ 0.40) | mAAE (recall @ 0.40) | mATE (optimal) | mAOE (optimal) | mASE (optimal) | mAVE (optimal) | mAAE (optimal) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 0.3293 | 0.2166 | 0.3253 | 0.3476 | 1.0000 | 0.4132 | 0.3083 | 0.3639 | 0.4336 | 1.0000 | 0.3657 | 0.2567 | 0.3324 | 0.3634 | 1.0000 |

  <summary><strong>Num match summary</strong></summary>

  **recall 0.10**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 19,301) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 2,906) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 484) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 291) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 2,564) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 462) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 145) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 2,123 / 2,123 / 2,123 / 2,123 | 319 / 319 / 319 / 319 | 53 / 53 / 53 / 53 | 32 / 32 / 32 / 32 | 282 / 282 / 282 / 282 | 50 / 50 / 50 / 50 | 15 / 15 / 15 / 15 |

  **recall 0.40**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 19,301) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 2,906) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 484) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 291) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 2,564) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 462) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 145) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 7,913 / 7,913 / 7,913 / 7,913 | 1,191 / 1,191 / 1,191 / 1,191 | 198 / 198 / 198 / 198 | 119 / 119 / 119 / 119 | 1,051 / 1,051 / 1,051 / 1,051 | 0 / 189 / 189 / 189 | 0 / 59 / 59 / 59 |

  **optimal**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 19,301) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 2,906) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 484) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 291) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 2,564) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 462) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 145) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 12,130 / 14,043 / 15,222 / 15,426 | 1,162 / 1,568 / 2,007 / 2,112 | 169 / 279 / 355 / 366 | 153 / 180 / 181 / 181 | 1,439 / 1,486 / 1,498 / 1,509 | 155 / 172 / 178 / 186 | 33 / 52 / 72 / 73 |

  </details>

  <details open>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version | mATE (recall @ 0.10) | mAOE (recall @ 0.10) | mASE (recall @ 0.10) | mAVE (recall @ 0.10) | mAAE (recall @ 0.10) | mATE (recall @ 0.40) | mAOE (recall @ 0.40) | mASE (recall @ 0.40) | mAVE (recall @ 0.40) | mAAE (recall @ 0.40) | mATE (optimal) | mAOE (optimal) | mASE (optimal) | mAVE (optimal) | mAAE (optimal) |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 0.1971 | 0.2029 | 0.2829 | 0.2224 | 1.0000 | 0.2506 | 0.2464 | 0.3150 | 0.2717 | 1.0000 | 0.2419 | 0.2401 | 0.2986 | 0.2465 | 1.0000 |

  <summary><strong>Num match summary</strong></summary>

  **recall 0.10**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 134,456) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 14,900) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 4,970) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 2,486) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 29,102) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 11,344) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 2,117) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 14,790 / 14,790 / 14,790 / 14,790 | 1,639 / 1,639 / 1,639 / 1,639 | 546 / 546 / 546 / 546 | 273 / 273 / 273 / 273 | 3,201 / 3,201 / 3,201 / 3,201 | 1,247 / 1,247 / 1,247 / 1,247 | 232 / 232 / 232 / 232 |

  **recall 0.40**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 134,456) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 14,900) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 4,970) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 2,486) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 29,102) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 11,344) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 2,117) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 55,126 / 55,126 / 55,126 / 55,126 | 6,109 / 6,109 / 6,109 / 6,109 | 2,037 / 2,037 / 2,037 / 2,037 | 1,019 / 1,019 / 1,019 / 1,019 | 11,931 / 11,931 / 11,931 / 11,931 | 4,651 / 4,651 / 4,651 / 4,651 | 0 / 867 / 867 / 867 |

  **optimal**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 134,456) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 14,900) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 4,970) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 2,486) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 29,102) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 11,344) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 2,117) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR j6gen2_base/2.8.1 | 102,297 / 112,278 / 116,527 / 117,516 | 9,001 / 10,847 / 11,695 / 12,090 | 3,495 / 4,146 / 4,350 / 4,483 | 1,859 / 1,919 / 1,923 / 1,923 | 21,639 / 21,986 / 21,863 / 22,136 | 5,782 / 6,632 / 6,752 / 7,077 | 602 / 827 / 850 / 872 |

  </details>

</details>

## Release

### BEVFusion-LiDAR J6Gen2_base/2.8.1

<details>
<summary> Changes  </summary>

- Finetune from `BEVFusion-LiDAR base/2.8.0` with j6gen2 base dataset and intensity.
</details>

<details>
<summary> Artifacts </summary>

- Deployed onnx and ROS parameter files (for internal)
  - [WebAuto](https://evaluation.ci.tier4.jp/evaluation/mlpackages/46f8188d-e3be-4f2f-b989-fd27002610d7/releases/fcf081e7-b3a9-4085-82f8-60023df3e854?project_id=zWhWRzei)
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-l/j6gen2_base/v2.8.1/deployment.zip)
  - [Google drive](https://drive.google.com/file/d/1VwFa3BZnDI7WV1i3aq6VYsK3pII2axMb/view?usp=drive_link)
- Logs (for internal)
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-l/j6gen2_base/v2.8.1/logs.zip)
  - [Google drive](https://drive.google.com/file/d/1n1EZUOMF6PKi9SciRQXzoMvCBkMnQaYL/view?usp=drive_link)
- Pytorch Best checkpoints:
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-l/j6gen2_base/v2.8.1/best_epoch_25.zip)
  - [Google drive](https://drive.google.com/file/d/1mOVIs7rUGPumjl3dosuNZqJGZlNOdV-e/view?usp=drive_link)

</details>

<details>
<summary> Training configs </summary>

- [Config file path](https://github.com/KSeangTan/AWML/blob/3d5e2fa3df7ad61d9ae773a3ea3f418f4916e05b/projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_8xb16_j6gen2_base_120m.py)
- Train time: NVIDIA H200 140GB * 8 * 30 epochs = 20 hours
- Batch size: 8*16 = 128
- Training Dataset (frames: 63,813):
  - j6gen2: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 + db_j6gen2_v6 + db_j6gen2_v7 + db_j6gen2_v8 + db_j6gen2_v9 + db_j6gen2_v10 + db_j6gen2_v11 + db_j6gen2_v12 (51,208 frames)
  - largebus: db_largebus_v1 + db_largebus_v2 + db_largebus_v3 (12,605 frames)

</details>

<details>
<summary> Evaluation </summary>

**J6Gen2_base Datasets (5,910 frames)**:

  - j6gen2 (3,951 frames): db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 + db_j6gen2_v6 + db_j6gen2_v7 + db_j6gen2_v8 + db_j6gen2_v9 + db_j6gen2_v10 + db_j6gen2_v11 + db_j6gen2_v12
  - largebus (1,228 frames): db_largebus_v1 + db_largebus_v2 + db_largebus_v3

**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.7289**

| class_name | GTs | num_match@0.5/1.0/2.0/4.0 | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | :---- | :---- | :---- | :---- |
| car | 75,589 | 67,025 / 69,724 / 70,866 / 71,572 | 0.855 / 0.903 / 0.915 / 0.927 | 0.904 / 0.929 / 0.937 / 0.942 | 0.279 / 0.218 / 0.158 / 0.147 |
| truck | 8,273 | 6,615 / 7,329 / 7,646 / 7,835 | 0.715 / 0.838 / 0.889 / 0.917 | 0.802 / 0.875 / 0.903 / 0.920 | 0.263 / 0.191 / 0.186 / 0.184 |
| bus | 2,706 | 2,339 / 2,562 / 2,628 / 2,640 | 0.810 / 0.916 / 0.962 / 0.963 | 0.874 / 0.943 / 0.962 / 0.965 | 0.260 / 0.167 / 0.130 / 0.096 |
| bicycle | 2,097 | 1,950 / 1,979 / 1,992 / 1,996 | 0.866 / 0.894 / 0.898 / 0.905 | 0.877 / 0.889 / 0.893 / 0.894 | 0.158 / 0.157 / 0.157 / 0.157 |
| pedestrian | 23,254 | 21,368 / 21,777 / 21,940 / 22,071 | 0.828 / 0.852 / 0.864 / 0.870 | 0.833 / 0.846 / 0.852 / 0.857 | 0.171 / 0.166 / 0.163 / 0.166 |
| traffic_cone | 8,310 | 5,479 / 5,915 / 6,096 / 6,331 | 0.385 / 0.444 / 0.463 / 0.494 | 0.559 / 0.594 / 0.607 / 0.624 | 0.123 / 0.121 / 0.111 / 0.086 |
| barrier | 1,350 | 572 / 754 / 803 / 843 | 0.174 / 0.267 / 0.289 / 0.306 | 0.409 / 0.462 / 0.472 / 0.483 | 0.283 / 0.260 / 0.248 / 0.248 |

<details>
<summary><strong>TP error — default (recall @0.10)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 8,314 / 8,314 / 8,314 / 8,314 | 0.107 / 0.112 / 0.113 / 0.117 | 0.033 / 0.036 / 0.036 / 0.037 | 0.116 / 0.117 / 0.118 / 0.118 | 0.126 / 0.127 / 0.128 / 0.128 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 910 / 910 / 910 / 910 | 0.144 / 0.165 / 0.181 / 0.200 | 0.028 / 0.030 / 0.031 / 0.031 | 0.127 / 0.132 / 0.135 / 0.137 | 0.297 / 0.306 / 0.311 / 0.310 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 297 / 297 / 297 / 297 | 0.108 / 0.121 / 0.142 / 0.143 | 0.044 / 0.045 / 0.045 / 0.045 | 0.083 / 0.085 / 0.091 / 0.091 | 0.128 / 0.130 / 0.129 / 0.129 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 230 / 230 / 230 / 230 | 0.131 / 0.137 / 0.138 / 0.140 | 0.080 / 0.080 / 0.080 / 0.081 | 0.202 / 0.204 / 0.204 / 0.205 | 0.537 / 0.536 / 0.535 / 0.536 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 2,557 / 2,557 / 2,557 / 2,557 | 0.102 / 0.108 / 0.117 / 0.133 | 0.395 / 0.397 / 0.401 / 0.404 | 0.232 / 0.233 / 0.234 / 0.234 | 0.240 / 0.239 / 0.239 / 0.241 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 914 / 914 / 914 / 914 | 0.176 / 0.198 / 0.219 / 0.297 | 0.328 / 0.325 / 0.327 / 0.329 | 0.644 / 0.648 / 0.649 / 0.650 | 0.026 / 0.026 / 0.026 / 0.026 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 148 / 148 / 148 / 148 | 0.232 / 0.293 / 0.318 / 0.363 | 0.374 / 0.376 / 0.378 / 0.375 | 0.458 / 0.477 / 0.484 / 0.492 | 0.024 / 0.025 / 0.025 / 0.025 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — medium (recall @0.40)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 30,991 / 30,991 / 30,991 / 30,991 | 0.115 / 0.121 / 0.124 / 0.128 | 0.037 / 0.041 / 0.043 / 0.044 | 0.121 / 0.123 / 0.123 / 0.123 | 0.139 / 0.140 / 0.141 / 0.141 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 3,391 / 3,391 / 3,391 / 3,391 | 0.153 / 0.181 / 0.202 / 0.230 | 0.032 / 0.034 / 0.035 / 0.036 | 0.133 / 0.138 / 0.142 / 0.145 | 0.317 / 0.325 / 0.331 / 0.330 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 1,109 / 1,109 / 1,109 / 1,109 | 0.118 / 0.136 / 0.152 / 0.154 | 0.051 / 0.052 / 0.052 / 0.052 | 0.086 / 0.089 / 0.094 / 0.094 | 0.147 / 0.147 / 0.147 / 0.147 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 859 / 859 / 859 / 859 | 0.132 / 0.138 / 0.139 / 0.142 | 0.087 / 0.087 / 0.087 / 0.087 | 0.207 / 0.209 / 0.209 / 0.210 | 0.555 / 0.554 / 0.553 / 0.553 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 9,534 / 9,534 / 9,534 / 9,534 | 0.107 / 0.114 / 0.127 / 0.149 | 0.427 / 0.429 / 0.433 / 0.437 | 0.239 / 0.240 / 0.240 / 0.241 | 0.244 / 0.244 / 0.244 / 0.245 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 3,407 / 3,407 / 3,407 / 3,407 | 0.186 / 0.215 / 0.249 / 0.353 | 0.376 / 0.370 / 0.370 / 0.371 | 0.654 / 0.659 / 0.660 / 0.661 | 0.028 / 0.028 / 0.028 / 0.028 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 553 / 553 / 553 / 553 | 0.251 / 0.333 / 0.368 / 0.443 | 0.422 / 0.406 / 0.410 / 0.403 | 0.542 / 0.541 / 0.547 / 0.554 | 0.025 / 0.025 / 0.025 / 0.025 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — optimal</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 65,297 / 67,749 / 68,945 / 69,400 | 0.130 / 0.146 / 0.159 / 0.175 | 0.052 / 0.063 / 0.072 / 0.076 | 0.131 / 0.133 / 0.135 / 0.135 | 0.149 / 0.152 / 0.154 / 0.154 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 6,298 / 7,033 / 7,266 / 7,405 | 0.164 / 0.211 / 0.246 / 0.296 | 0.041 / 0.053 / 0.054 / 0.058 | 0.142 / 0.153 / 0.158 / 0.163 | 0.313 / 0.324 / 0.331 / 0.331 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 2,282 / 2,494 / 2,565 / 2,589 | 0.138 / 0.181 / 0.205 / 0.217 | 0.065 / 0.069 / 0.073 / 0.077 | 0.095 / 0.102 / 0.106 / 0.107 | 0.173 / 0.166 / 0.166 / 0.170 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 1,789 / 1,815 / 1,822 / 1,825 | 0.136 / 0.144 / 0.149 / 0.153 | 0.097 / 0.097 / 0.097 / 0.097 | 0.214 / 0.216 / 0.217 / 0.218 | 0.550 / 0.548 / 0.547 / 0.547 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 18,701 / 19,099 / 19,305 / 19,357 | 0.113 / 0.125 / 0.144 / 0.177 | 0.453 / 0.457 / 0.462 / 0.467 | 0.244 / 0.246 / 0.247 / 0.247 | 0.250 / 0.249 / 0.250 / 0.252 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 4,529 / 4,841 / 5,046 / 5,561 | 0.187 / 0.217 / 0.256 / 0.403 | 0.385 / 0.383 / 0.382 / 0.387 | 0.654 / 0.659 / 0.662 / 0.669 | 0.028 / 0.028 / 0.028 / 0.029 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 453 / 529 / 550 / 562 | 0.240 / 0.298 / 0.323 / 0.375 | 0.394 / 0.397 / 0.395 / 0.389 | 0.491 / 0.504 / 0.510 / 0.516 | 0.023 / 0.024 / 0.024 / 0.024 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.5802**

| class_name | GTs | num_match@0.5/1.0/2.0/4.0 | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | :---- | :---- | :---- | :---- |
| car | 64,960 | 50,716 / 56,494 / 59,237 / 60,392 | 0.695 / 0.809 / 0.866 / 0.881 | 0.781 / 0.848 / 0.874 / 0.883 | 0.240 / 0.190 / 0.157 / 0.157 |
| truck | 5,922 | 3,638 / 4,443 / 4,919 / 5,132 | 0.459 / 0.634 / 0.737 / 0.777 | 0.625 / 0.731 / 0.786 / 0.805 | 0.249 / 0.165 / 0.164 / 0.159 |
| bus | 2,257 | 1,543 / 1,947 / 2,104 / 2,161 | 0.565 / 0.797 / 0.891 / 0.917 | 0.681 / 0.830 / 0.883 / 0.900 | 0.415 / 0.184 / 0.171 / 0.181 |
| bicycle | 1,298 | 986 / 1,068 / 1,079 / 1,080 | 0.576 / 0.670 / 0.682 / 0.683 | 0.683 / 0.722 / 0.726 / 0.727 | 0.110 / 0.135 / 0.106 / 0.106 |
| pedestrian | 12,052 | 10,341 / 10,570 / 10,667 / 10,768 | 0.642 / 0.664 / 0.679 / 0.692 | 0.694 / 0.705 / 0.711 / 0.716 | 0.145 / 0.145 / 0.145 / 0.146 |
| traffic_cone | 2,636 | 1,308 / 1,442 / 1,510 / 1,614 | 0.214 / 0.260 / 0.291 / 0.339 | 0.436 / 0.477 / 0.493 / 0.523 | 0.085 / 0.085 / 0.084 / 0.084 |
| barrier | 622 | 216 / 296 / 314 / 328 | 0.117 / 0.222 / 0.239 / 0.248 | 0.333 / 0.423 / 0.434 / 0.439 | 0.183 / 0.106 / 0.106 / 0.082 |

<details>
<summary><strong>TP error — default (recall @0.10)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 7,145 / 7,145 / 7,145 / 7,145 | 0.158 / 0.178 / 0.194 / 0.207 | 0.108 / 0.135 / 0.153 / 0.156 | 0.158 / 0.161 / 0.162 / 0.162 | 0.158 / 0.159 / 0.160 / 0.161 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 651 / 651 / 651 / 651 | 0.191 / 0.240 / 0.292 / 0.328 | 0.035 / 0.039 / 0.042 / 0.044 | 0.155 / 0.167 / 0.174 / 0.178 | 0.425 / 0.429 / 0.434 / 0.439 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 248 / 248 / 248 / 248 | 0.156 / 0.206 / 0.236 / 0.248 | 0.149 / 0.137 / 0.139 / 0.146 | 0.114 / 0.123 / 0.128 / 0.129 | 0.146 / 0.153 / 0.153 / 0.153 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 142 / 142 / 142 / 142 | 0.174 / 0.203 / 0.209 / 0.213 | 0.140 / 0.144 / 0.144 / 0.145 | 0.217 / 0.226 / 0.227 / 0.227 | 0.642 / 0.675 / 0.673 / 0.672 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 1,325 / 1,325 / 1,325 / 1,325 | 0.116 / 0.125 / 0.144 / 0.184 | 0.544 / 0.550 / 0.554 / 0.563 | 0.220 / 0.221 / 0.221 / 0.222 | 0.290 / 0.289 / 0.290 / 0.294 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 289 / 289 / 289 / 289 | 0.190 / 0.223 / 0.313 / 0.573 | 0.272 / 0.284 / 0.286 / 0.309 | 0.685 / 0.691 / 0.692 / 0.692 | 0.043 / 0.044 / 0.044 / 0.044 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 68 / 68 / 68 / 68 | 0.247 / 0.336 / 0.361 / 0.416 | 0.375 / 0.360 / 0.359 / 0.359 | 0.456 / 0.473 / 0.481 / 0.485 | 0.032 / 0.033 / 0.034 / 0.034 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — medium (recall @0.40)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 26,633 / 26,633 / 26,633 / 26,633 | 0.169 / 0.196 / 0.217 / 0.235 | 0.129 / 0.164 / 0.187 / 0.190 | 0.164 / 0.167 / 0.169 / 0.169 | 0.167 / 0.168 / 0.169 / 0.170 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 2,428 / 2,428 / 2,428 / 2,428 | 0.202 / 0.264 / 0.329 / 0.375 | 0.045 / 0.048 / 0.052 / 0.055 | 0.165 / 0.178 / 0.186 / 0.191 | 0.441 / 0.450 / 0.455 / 0.462 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 925 / 925 / 925 / 925 | 0.173 / 0.240 / 0.281 / 0.298 | 0.107 / 0.104 / 0.112 / 0.124 | 0.123 / 0.132 / 0.139 / 0.140 | 0.163 / 0.169 / 0.167 / 0.167 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 532 / 532 / 532 / 532 | 0.181 / 0.209 / 0.218 / 0.224 | 0.169 / 0.174 / 0.175 / 0.175 | 0.223 / 0.230 / 0.231 / 0.231 | 0.661 / 0.683 / 0.680 / 0.680 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 4,941 / 4,941 / 4,941 / 4,941 | 0.122 / 0.134 / 0.156 / 0.204 | 0.580 / 0.585 / 0.590 / 0.600 | 0.223 / 0.224 / 0.224 / 0.225 | 0.314 / 0.313 / 0.313 / 0.318 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 1,080 / 1,080 / 1,080 / 1,080 | 0.205 / 0.249 / 0.352 / 0.633 | 0.359 / 0.368 / 0.365 / 0.385 | 0.690 / 0.695 / 0.697 / 0.697 | 0.053 / 0.051 / 0.051 / 0.052 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 255 / 255 / 255 | 1.000 / 0.380 / 0.432 / 0.563 | 1.000 / 0.422 / 0.413 / 0.407 | 1.000 / 0.546 / 0.552 / 0.554 | 1.000 / 0.036 / 0.036 / 0.036 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — optimal</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 47,451 / 52,827 / 55,318 / 55,889 | 0.179 / 0.222 / 0.259 / 0.293 | 0.160 / 0.208 / 0.240 / 0.245 | 0.169 / 0.175 / 0.176 / 0.177 | 0.177 / 0.183 / 0.186 / 0.186 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 3,253 / 4,045 / 4,354 / 4,471 | 0.205 / 0.284 / 0.360 / 0.418 | 0.048 / 0.057 / 0.060 / 0.064 | 0.168 / 0.185 / 0.195 / 0.200 | 0.438 / 0.459 / 0.468 / 0.475 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 1,367 / 1,826 / 1,951 / 1,981 | 0.183 / 0.290 / 0.348 / 0.405 | 0.101 / 0.104 / 0.124 / 0.166 | 0.126 / 0.140 / 0.148 / 0.151 | 0.174 / 0.172 / 0.172 / 0.171 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 839 / 846 / 901 / 902 | 0.183 / 0.209 / 0.219 / 0.225 | 0.190 / 0.180 / 0.192 / 0.194 | 0.225 / 0.232 / 0.233 / 0.233 | 0.663 / 0.692 / 0.686 / 0.686 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 8,085 / 8,202 / 8,275 / 8,325 | 0.123 / 0.135 / 0.157 / 0.205 | 0.588 / 0.593 / 0.598 / 0.607 | 0.223 / 0.224 / 0.224 / 0.225 | 0.321 / 0.320 / 0.322 / 0.326 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 1,120 / 1,225 / 1,270 / 1,347 | 0.202 / 0.247 / 0.346 / 0.630 | 0.354 / 0.373 / 0.372 / 0.396 | 0.687 / 0.695 / 0.697 / 0.699 | 0.049 / 0.050 / 0.050 / 0.051 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 156 / 231 / 237 / 268 | 0.255 / 0.361 / 0.392 / 0.529 | 0.380 / 0.406 / 0.399 / 0.397 | 0.461 / 0.502 / 0.509 / 0.539 | 0.033 / 0.036 / 0.036 / 0.036 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.4396**

| class_name | GTs | num_match@0.5/1.0/2.0/4.0 | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | :---- | :---- | :---- | :---- |
| car | 22,141 | 15,775 / 18,697 / 20,145 / 20,550 | 0.541 / 0.714 / 0.790 / 0.814 | 0.666 / 0.762 / 0.800 / 0.811 | 0.204 / 0.181 / 0.160 / 0.156 |
| truck | 3,506 | 1,627 / 2,289 / 2,818 / 3,016 | 0.257 / 0.472 / 0.668 / 0.733 | 0.464 / 0.620 / 0.736 / 0.770 | 0.159 / 0.159 / 0.111 / 0.111 |
| bus | 544 | 257 / 368 / 432 / 448 | 0.273 / 0.540 / 0.667 / 0.698 | 0.467 / 0.639 / 0.717 / 0.737 | 0.349 / 0.126 / 0.066 / 0.066 |
| bicycle | 376 | 269 / 307 / 317 / 318 | 0.354 / 0.532 / 0.552 / 0.554 | 0.509 / 0.605 / 0.619 / 0.619 | 0.136 / 0.143 / 0.143 / 0.143 |
| pedestrian | 3,656 | 3,001 / 3,053 / 3,081 / 3,122 | 0.482 / 0.496 / 0.505 / 0.515 | 0.591 / 0.598 / 0.602 / 0.607 | 0.135 / 0.135 / 0.135 / 0.135 |
| traffic_cone | 462 | 183 / 207 / 225 / 235 | 0.100 / 0.129 / 0.141 / 0.162 | 0.304 / 0.339 / 0.349 / 0.365 | 0.127 / 0.127 / 0.088 / 0.088 |
| barrier | 145 | 49 / 72 / 90 / 96 | 0.041 / 0.139 / 0.203 / 0.240 | 0.237 / 0.362 / 0.425 / 0.449 | 0.139 / 0.119 / 0.085 / 0.095 |

<details>
<summary><strong>TP error — default (recall @0.10)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 2,435 / 2,435 / 2,435 / 2,435 | 0.199 / 0.241 / 0.275 / 0.305 | 0.198 / 0.246 / 0.274 / 0.284 | 0.180 / 0.184 / 0.185 / 0.186 | 0.299 / 0.294 / 0.293 / 0.294 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 385 / 385 / 385 / 385 | 0.226 / 0.312 / 0.434 / 0.493 | 0.043 / 0.048 / 0.054 / 0.060 | 0.175 / 0.191 / 0.209 / 0.215 | 0.387 / 0.414 / 0.428 / 0.437 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 59 / 59 / 59 / 59 | 0.234 / 0.326 / 0.385 / 0.409 | 0.037 / 0.054 / 0.058 / 0.059 | 0.141 / 0.156 / 0.164 / 0.167 | 0.378 / 0.406 / 0.428 / 0.431 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 41 / 41 / 41 / 41 | 0.237 / 0.297 / 0.311 / 0.314 | 0.101 / 0.094 / 0.095 / 0.095 | 0.249 / 0.264 / 0.266 / 0.266 | 0.777 / 0.771 / 0.776 / 0.779 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 402 / 402 / 402 / 402 | 0.126 / 0.137 / 0.155 / 0.197 | 0.496 / 0.502 / 0.508 / 0.514 | 0.229 / 0.230 / 0.230 / 0.230 | 0.382 / 0.382 / 0.383 / 0.386 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 50 / 50 / 50 / 50 | 0.193 / 0.234 / 0.272 / 0.526 | 0.288 / 0.286 / 0.309 / 0.312 | 0.702 / 0.701 / 0.703 / 0.699 | 0.044 / 0.046 / 0.046 / 0.046 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 15 / 15 / 15 / 15 | 0.301 / 0.435 / 0.535 / 0.997 | 0.250 / 0.220 / 0.216 / 0.212 | 0.487 / 0.511 / 0.530 / 0.530 | 0.045 / 0.045 / 0.046 / 0.046 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — medium (recall @0.40)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 9,077 / 9,077 / 9,077 / 9,077 | 0.208 / 0.259 / 0.301 / 0.339 | 0.248 / 0.296 / 0.327 / 0.336 | 0.186 / 0.190 / 0.191 / 0.191 | 0.306 / 0.302 / 0.301 / 0.302 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 1,437 / 1,437 / 1,437 / 1,437 | 0.240 / 0.346 / 0.494 / 0.573 | 0.067 / 0.062 / 0.068 / 0.075 | 0.187 / 0.201 / 0.220 / 0.229 | 0.457 / 0.479 / 0.483 / 0.490 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 223 / 223 / 223 / 223 | 0.249 / 0.363 / 0.448 / 0.486 | 0.061 / 0.072 / 0.073 / 0.074 | 0.149 / 0.162 / 0.175 / 0.178 | 0.429 / 0.434 / 0.460 / 0.464 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 154 / 154 / 154 / 154 | 0.222 / 0.284 / 0.307 / 0.313 | 0.131 / 0.120 / 0.120 / 0.120 | 0.246 / 0.259 / 0.263 / 0.263 | 0.766 / 0.795 / 0.801 / 0.806 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 1,498 / 1,498 / 1,498 / 1,498 | 0.132 / 0.145 / 0.167 / 0.220 | 0.550 / 0.554 / 0.563 / 0.572 | 0.228 / 0.228 / 0.229 / 0.229 | 0.415 / 0.414 / 0.415 / 0.419 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 0 / 189 / 189 / 189 | 1.000 / 0.274 / 0.365 / 0.629 | 1.000 / 0.405 / 0.430 / 0.444 | 1.000 / 0.718 / 0.719 / 0.714 | 1.000 / 0.060 / 0.059 / 0.055 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 59 / 59 / 59 | 1.000 / 0.438 / 0.604 / 1.056 | 1.000 / 0.237 / 0.228 / 0.223 | 1.000 / 0.567 / 0.578 / 0.581 | 1.000 / 0.045 / 0.047 / 0.047 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — optimal</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 13,838 / 16,236 / 17,440 / 17,774 | 0.211 / 0.272 / 0.327 / 0.375 | 0.275 / 0.339 / 0.383 / 0.394 | 0.188 / 0.193 / 0.195 / 0.196 | 0.306 / 0.304 / 0.306 / 0.308 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 1,483 / 1,981 / 2,484 / 2,599 | 0.238 / 0.351 / 0.530 / 0.622 | 0.055 / 0.061 / 0.083 / 0.093 | 0.186 / 0.205 / 0.230 / 0.238 | 0.465 / 0.493 / 0.490 / 0.496 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 192 / 312 / 394 / 405 | 0.240 / 0.373 / 0.509 / 0.561 | 0.044 / 0.085 / 0.084 / 0.084 | 0.141 / 0.164 / 0.184 / 0.189 | 0.369 / 0.450 / 0.478 / 0.483 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 187 / 218 / 223 / 223 | 0.223 / 0.285 / 0.310 / 0.310 | 0.090 / 0.095 / 0.095 / 0.095 | 0.244 / 0.257 / 0.261 / 0.261 | 0.760 / 0.794 / 0.798 / 0.798 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 2,141 / 2,167 / 2,181 / 2,199 | 0.130 / 0.142 / 0.162 / 0.211 | 0.536 / 0.542 / 0.547 / 0.555 | 0.226 / 0.227 / 0.228 / 0.228 | 0.413 / 0.413 / 0.412 / 0.417 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 132 / 147 / 178 / 186 | 0.201 / 0.255 / 0.313 / 0.600 | 0.338 / 0.322 / 0.401 / 0.422 | 0.704 / 0.702 / 0.710 / 0.705 | 0.047 / 0.048 / 0.055 / 0.054 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 33 / 52 / 72 / 73 | 0.295 / 0.437 / 0.603 / 1.064 | 0.239 / 0.211 / 0.222 / 0.219 | 0.469 / 0.512 / 0.568 / 0.572 | 0.048 / 0.047 / 0.047 / 0.047 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.6590**

| class_name | GTs | num_match@0.5/1.0/2.0/4.0 | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | :---- | :---- | :---- | :---- |
| car | 162,690 | 133,765 / 145,295 / 150,720 / 153,032 | 0.765 / 0.854 / 0.890 / 0.910 | 0.825 / 0.877 / 0.895 / 0.902 | 0.242 / 0.194 / 0.169 / 0.157 |
| truck | 17,701 | 11,903 / 14,102 / 15,459 / 16,076 | 0.550 / 0.709 / 0.807 / 0.848 | 0.683 / 0.781 / 0.835 / 0.856 | 0.266 / 0.176 / 0.165 / 0.157 |
| bus | 5,507 | 4,147 / 4,894 / 5,184 / 5,269 | 0.668 / 0.840 / 0.916 / 0.930 | 0.761 / 0.872 / 0.909 / 0.919 | 0.343 / 0.182 / 0.172 / 0.125 |
| bicycle | 3,771 | 3,210 / 3,360 / 3,393 / 3,399 | 0.732 / 0.795 / 0.802 / 0.808 | 0.775 / 0.805 / 0.809 / 0.810 | 0.158 / 0.156 / 0.156 / 0.156 |
| pedestrian | 38,962 | 34,759 / 35,452 / 35,735 / 36,011 | 0.755 / 0.774 / 0.787 / 0.799 | 0.768 / 0.780 / 0.786 / 0.792 | 0.154 / 0.154 / 0.154 / 0.158 |
| traffic_cone | 11,408 | 6,982 / 7,578 / 7,846 / 8,197 | 0.338 / 0.390 / 0.411 / 0.444 | 0.521 / 0.557 / 0.571 / 0.592 | 0.123 / 0.111 / 0.111 / 0.087 |
| barrier | 2,117 | 839 / 1,125 / 1,212 / 1,272 | 0.144 / 0.241 / 0.263 / 0.280 | 0.367 / 0.438 / 0.450 / 0.459 | 0.274 / 0.185 / 0.185 / 0.183 |

<details>
<summary><strong>TP error — default (recall @0.10)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 17,895 / 17,895 / 17,895 / 17,895 | 0.129 / 0.142 / 0.151 / 0.160 | 0.062 / 0.077 / 0.086 / 0.090 | 0.133 / 0.136 / 0.137 / 0.137 | 0.148 / 0.151 / 0.152 / 0.153 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 1,947 / 1,947 / 1,947 / 1,947 | 0.165 / 0.202 / 0.242 / 0.273 | 0.032 / 0.035 / 0.037 / 0.039 | 0.140 / 0.148 / 0.155 / 0.158 | 0.339 / 0.351 / 0.359 / 0.362 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 605 / 605 / 605 / 605 | 0.127 / 0.157 / 0.182 / 0.188 | 0.079 / 0.078 / 0.080 / 0.083 | 0.094 / 0.100 / 0.106 / 0.106 | 0.144 / 0.150 / 0.151 / 0.152 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 414 / 414 / 414 / 414 | 0.147 / 0.162 / 0.165 / 0.168 | 0.095 / 0.096 / 0.095 / 0.096 | 0.210 / 0.215 / 0.215 / 0.216 | 0.573 / 0.583 / 0.582 / 0.583 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 4,285 / 4,285 / 4,285 / 4,285 | 0.107 / 0.114 / 0.127 / 0.151 | 0.437 / 0.439 / 0.443 / 0.449 | 0.232 / 0.232 / 0.233 / 0.233 | 0.256 / 0.255 / 0.255 / 0.258 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 1,254 / 1,254 / 1,254 / 1,254 | 0.179 / 0.203 / 0.234 / 0.347 | 0.322 / 0.321 / 0.322 / 0.327 | 0.652 / 0.656 / 0.657 / 0.659 | 0.029 / 0.029 / 0.029 / 0.030 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 232 / 232 / 232 / 232 | 0.239 / 0.311 / 0.339 / 0.410 | 0.374 / 0.369 / 0.369 / 0.366 | 0.469 / 0.488 / 0.494 / 0.502 | 0.026 / 0.027 / 0.028 / 0.028 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — medium (recall @0.40)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 66,702 / 66,702 / 66,702 / 66,702 | 0.142 / 0.160 / 0.172 / 0.185 | 0.080 / 0.101 / 0.113 / 0.117 | 0.142 / 0.145 / 0.146 / 0.147 | 0.162 / 0.164 / 0.165 / 0.166 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 7,257 / 7,257 / 7,257 / 7,257 | 0.178 / 0.226 / 0.280 / 0.323 | 0.039 / 0.042 / 0.045 / 0.047 | 0.149 / 0.159 / 0.167 / 0.171 | 0.361 / 0.374 / 0.383 / 0.385 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 2,257 / 2,257 / 2,257 / 2,257 | 0.143 / 0.184 / 0.212 / 0.220 | 0.076 / 0.076 / 0.079 / 0.083 | 0.101 / 0.109 / 0.114 / 0.115 | 0.165 / 0.169 / 0.170 / 0.171 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 1,546 / 1,546 / 1,546 / 1,546 | 0.151 / 0.167 / 0.172 / 0.176 | 0.108 / 0.109 / 0.109 / 0.110 | 0.215 / 0.219 / 0.220 / 0.221 | 0.592 / 0.600 / 0.599 / 0.599 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 15,974 / 15,974 / 15,974 / 15,974 | 0.113 / 0.122 / 0.139 / 0.172 | 0.476 / 0.478 / 0.483 / 0.489 | 0.236 / 0.237 / 0.238 / 0.238 | 0.269 / 0.268 / 0.268 / 0.271 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 4,677 / 4,677 / 4,677 / 4,677 | 0.191 / 0.223 / 0.269 / 0.413 | 0.371 / 0.368 / 0.368 / 0.372 | 0.663 / 0.668 / 0.669 / 0.670 | 0.032 / 0.032 / 0.032 / 0.032 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 867 / 867 / 867 | 1.000 / 0.355 / 0.402 / 0.517 | 1.000 / 0.398 / 0.397 / 0.391 | 1.000 / 0.544 / 0.550 / 0.557 | 1.000 / 0.029 / 0.029 / 0.030 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — optimal</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 126,514 / 137,022 / 141,450 / 143,300 | 0.157 / 0.191 / 0.218 / 0.246 | 0.115 / 0.151 / 0.174 / 0.182 | 0.151 / 0.156 / 0.158 / 0.159 | 0.176 / 0.182 / 0.185 / 0.186 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 10,859 / 13,045 / 14,027 / 14,439 | 0.185 / 0.255 / 0.328 / 0.389 | 0.044 / 0.055 / 0.059 / 0.064 | 0.154 / 0.170 / 0.181 / 0.188 | 0.368 / 0.391 / 0.402 / 0.406 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 3,844 / 4,623 / 4,831 / 4,974 | 0.159 / 0.236 / 0.278 / 0.319 | 0.075 / 0.083 / 0.094 / 0.114 | 0.109 / 0.121 / 0.127 / 0.131 | 0.178 / 0.185 / 0.188 / 0.191 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 2,732 / 2,840 / 2,856 / 2,859 | 0.154 / 0.173 / 0.180 / 0.184 | 0.117 / 0.120 / 0.121 / 0.121 | 0.218 / 0.223 / 0.224 / 0.225 | 0.598 / 0.606 / 0.605 / 0.605 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 28,970 / 29,422 / 29,647 / 29,660 | 0.117 / 0.130 / 0.148 / 0.187 | 0.496 / 0.500 / 0.505 / 0.510 | 0.238 / 0.239 / 0.240 / 0.240 | 0.280 / 0.279 / 0.280 / 0.283 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 5,644 / 6,197 / 6,356 / 7,069 | 0.189 / 0.223 / 0.271 / 0.452 | 0.370 / 0.375 / 0.375 / 0.388 | 0.660 / 0.667 / 0.669 / 0.676 | 0.032 / 0.032 / 0.032 / 0.033 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 602 / 827 / 850 / 871 | 0.245 / 0.329 / 0.356 / 0.433 | 0.378 / 0.383 / 0.380 / 0.377 | 0.478 / 0.506 / 0.509 / 0.515 | 0.026 / 0.027 / 0.028 / 0.028 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

---

**LargeBus**: db_largebus_v1 + db_largebus_v2 + db_largebus_v3 (1,228 frames)  

**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.6313**

| class_name | GTs | num_match@0.5/1.0/2.0/4.0 | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | :---- | :---- | :---- | :---- |
| car | 14,872 | 13,494 / 13,924 / 14,079 / 14,195 | 0.879 / 0.917 / 0.928 / 0.939 | 0.921 / 0.942 / 0.947 / 0.951 | 0.278 / 0.169 / 0.169 / 0.169 |
| truck | 1,192 | 981 / 1,080 / 1,113 / 1,136 | 0.760 / 0.876 / 0.913 / 0.932 | 0.837 / 0.907 / 0.922 / 0.925 | 0.357 / 0.208 / 0.187 / 0.157 |
| bus | 336 | 261 / 332 / 335 / 335 | 0.715 / 0.975 / 0.987 / 0.987 | 0.808 / 0.973 / 0.982 / 0.982 | 0.469 / 0.099 / 0.099 / 0.099 |
| bicycle | 740 | 676 / 694 / 706 / 710 | 0.817 / 0.857 / 0.878 / 0.883 | 0.846 / 0.864 / 0.871 / 0.875 | 0.174 / 0.166 / 0.157 / 0.157 |
| pedestrian | 5,055 | 4,706 / 4,761 / 4,785 / 4,800 | 0.844 / 0.859 / 0.864 / 0.868 | 0.851 / 0.860 / 0.864 / 0.868 | 0.151 / 0.151 / 0.151 / 0.151 |
| traffic_cone | 60 | 20 / 21 / 21 / 22 | 0.000 / 0.000 / 0.000 / 0.000 | 0.038 / 0.040 / 0.040 / 0.042 | 0.065 / 0.065 / 0.065 / 0.065 |
| barrier | 0 | 0 / 0 / 0 / 0 | 0.000 / 0.000 / 0.000 / 0.000 | nan / nan / nan / nan | nan / nan / nan / nan |

<details>
<summary><strong>TP error — default (recall @0.10)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 1,635 / 1,635 / 1,635 / 1,635 | 0.110 / 0.113 / 0.114 / 0.116 | 0.040 / 0.041 / 0.042 / 0.043 | 0.117 / 0.117 / 0.118 / 0.118 | 0.138 / 0.140 / 0.140 / 0.140 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 131 / 131 / 131 / 131 | 0.146 / 0.166 / 0.175 / 0.181 | 0.030 / 0.031 / 0.032 / 0.034 | 0.127 / 0.131 / 0.132 / 0.133 | 0.205 / 0.208 / 0.207 / 0.207 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 36 / 36 / 36 / 36 | 0.141 / 0.172 / 0.174 / 0.174 | 0.205 / 0.198 / 0.198 / 0.198 | 0.084 / 0.090 / 0.090 / 0.090 | 0.185 / 0.180 / 0.181 / 0.181 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 81 / 81 / 81 / 81 | 0.136 / 0.148 / 0.152 / 0.156 | 0.109 / 0.108 / 0.109 / 0.109 | 0.218 / 0.223 / 0.224 / 0.225 | 0.547 / 0.542 / 0.541 / 0.541 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 556 / 556 / 556 / 556 | 0.096 / 0.099 / 0.104 / 0.120 | 0.295 / 0.296 / 0.296 / 0.302 | 0.210 / 0.210 / 0.210 / 0.210 | 0.249 / 0.248 / 0.248 / 0.251 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 6 / 6 / 6 / 6 | 0.155 / 0.194 / 0.194 / 0.312 | 1.696 / 1.646 / 1.646 / 1.590 | 0.453 / 0.468 / 0.468 / 0.485 | 0.077 / 0.082 / 0.082 / 0.083 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — medium (recall @0.40)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 6,097 / 6,097 / 6,097 / 6,097 | 0.118 / 0.122 / 0.124 / 0.127 | 0.044 / 0.047 / 0.048 / 0.049 | 0.121 / 0.122 / 0.122 / 0.123 | 0.150 / 0.151 / 0.152 / 0.152 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 488 / 488 / 488 / 488 | 0.155 / 0.179 / 0.193 / 0.202 | 0.034 / 0.035 / 0.036 / 0.039 | 0.136 / 0.140 / 0.142 / 0.143 | 0.213 / 0.215 / 0.214 / 0.213 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 137 / 137 / 137 / 137 | 0.154 / 0.196 / 0.199 / 0.199 | 0.216 / 0.201 / 0.201 / 0.201 | 0.088 / 0.095 / 0.096 / 0.096 | 0.194 / 0.183 / 0.184 / 0.184 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 303 / 303 / 303 / 303 | 0.140 / 0.150 / 0.156 / 0.162 | 0.122 / 0.121 / 0.122 / 0.121 | 0.221 / 0.225 / 0.227 / 0.228 | 0.547 / 0.544 / 0.541 / 0.542 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 2,072 / 2,072 / 2,072 / 2,072 | 0.100 / 0.104 / 0.110 / 0.130 | 0.334 / 0.335 / 0.335 / 0.342 | 0.216 / 0.216 / 0.217 / 0.217 | 0.255 / 0.255 / 0.255 / 0.258 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — optimal</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 13,178 / 13,676 / 13,748 / 13,798 | 0.133 / 0.147 / 0.154 / 0.166 | 0.062 / 0.074 / 0.078 / 0.080 | 0.132 / 0.134 / 0.135 / 0.135 | 0.155 / 0.157 / 0.158 / 0.158 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 925 / 1,041 / 1,064 / 1,073 | 0.167 / 0.206 / 0.227 / 0.247 | 0.040 / 0.046 / 0.051 / 0.062 | 0.148 / 0.155 / 0.159 / 0.160 | 0.204 / 0.203 / 0.202 / 0.201 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 254 / 330 / 333 / 333 | 0.167 / 0.272 / 0.283 / 0.283 | 0.213 / 0.172 / 0.171 / 0.171 | 0.092 / 0.108 / 0.110 / 0.110 | 0.192 / 0.163 / 0.166 / 0.166 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 612 / 628 / 640 / 643 | 0.145 / 0.156 / 0.171 / 0.182 | 0.136 / 0.135 / 0.136 / 0.135 | 0.223 / 0.228 / 0.232 / 0.234 | 0.529 / 0.525 / 0.517 / 0.516 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 4,247 / 4,294 / 4,313 / 4,330 | 0.108 / 0.117 / 0.126 / 0.152 | 0.374 / 0.377 / 0.379 / 0.388 | 0.222 / 0.222 / 0.223 / 0.223 | 0.257 / 0.255 / 0.256 / 0.260 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 19 / 20 / 20 / 21 | 0.158 / 0.207 / 0.207 / 0.339 | 1.613 / 1.543 / 1.543 / 1.485 | 0.469 / 0.492 / 0.492 / 0.508 | 0.086 / 0.087 / 0.087 / 0.087 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan |

</details>

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.5281**

| class_name | GTs | num_match@0.5/1.0/2.0/4.0 | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | :---- | :---- | :---- | :---- |
| car | 10,929 | 8,919 / 9,721 / 10,089 / 10,240 | 0.747 / 0.843 / 0.887 / 0.899 | 0.816 / 0.875 / 0.895 / 0.901 | 0.216 / 0.166 / 0.149 / 0.149 |
| truck | 1,009 | 664 / 792 / 859 / 874 | 0.548 / 0.705 / 0.789 / 0.802 | 0.701 / 0.790 / 0.844 / 0.849 | 0.283 / 0.180 / 0.155 / 0.155 |
| bus | 141 | 114 / 134 / 135 / 137 | 0.650 / 0.913 / 0.916 / 0.929 | 0.769 / 0.905 / 0.905 / 0.905 | 0.486 / 0.444 / 0.444 / 0.444 |
| bicycle | 460 | 320 / 362 / 371 / 371 | 0.468 / 0.598 / 0.619 / 0.619 | 0.610 / 0.670 / 0.677 / 0.677 | 0.098 / 0.098 / 0.098 / 0.098 |
| pedestrian | 3,721 | 3,190 / 3,254 / 3,271 / 3,297 | 0.691 / 0.713 / 0.718 / 0.729 | 0.728 / 0.740 / 0.744 / 0.749 | 0.125 / 0.124 / 0.124 / 0.124 |
| traffic_cone | 4 | 2 / 2 / 2 / 2 | 0.000 / 0.000 / 0.000 / 0.000 | 0.027 / 0.027 / 0.027 / 0.027 | 0.099 / 0.099 / 0.099 / 0.099 |
| barrier | 0 | 0 / 0 / 0 / 0 | 0.000 / 0.000 / 0.000 / 0.000 | nan / nan / nan / nan | nan / nan / nan / nan |

<details>
<summary><strong>TP error — default (recall @0.10)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 1,202 / 1,202 / 1,202 / 1,202 | 0.157 / 0.172 / 0.180 / 0.188 | 0.079 / 0.094 / 0.103 / 0.104 | 0.147 / 0.151 / 0.152 / 0.152 | 0.204 / 0.209 / 0.213 / 0.213 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 110 / 110 / 110 / 110 | 0.188 / 0.222 / 0.253 / 0.258 | 0.038 / 0.043 / 0.046 / 0.048 | 0.163 / 0.173 / 0.178 / 0.178 | 0.237 / 0.240 / 0.257 / 0.258 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 15 / 15 / 15 / 15 | 0.208 / 0.258 / 0.258 / 0.261 | 0.597 / 0.536 / 0.536 / 0.532 | 0.082 / 0.089 / 0.090 / 0.090 | 0.211 / 0.208 / 0.208 / 0.210 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 50 / 50 / 50 / 50 | 0.185 / 0.236 / 0.248 / 0.249 | 0.182 / 0.184 / 0.185 / 0.185 | 0.243 / 0.258 / 0.262 / 0.262 | 0.621 / 0.730 / 0.726 / 0.726 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 409 / 409 / 409 / 409 | 0.113 / 0.122 / 0.129 / 0.155 | 0.446 / 0.450 / 0.452 / 0.459 | 0.203 / 0.204 / 0.205 / 0.205 | 0.279 / 0.279 / 0.279 / 0.282 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 0 / 0 / 0 / 0 | 0.154 / 0.154 / 0.154 / 0.154 | 2.562 / 2.562 / 2.562 / 2.562 | 0.285 / 0.285 / 0.285 / 0.285 | 0.035 / 0.035 / 0.035 / 0.035 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — medium (recall @0.40)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 4,480 / 4,480 / 4,480 / 4,480 | 0.166 / 0.186 / 0.198 / 0.208 | 0.099 / 0.118 / 0.130 / 0.131 | 0.154 / 0.158 / 0.159 / 0.160 | 0.223 / 0.228 / 0.233 / 0.233 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 413 / 413 / 413 / 413 | 0.199 / 0.241 / 0.285 / 0.293 | 0.045 / 0.051 / 0.054 / 0.057 | 0.169 / 0.180 / 0.187 / 0.187 | 0.271 / 0.270 / 0.281 / 0.283 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 57 / 57 / 57 / 57 | 0.216 / 0.269 / 0.269 / 0.272 | 0.432 / 0.394 / 0.394 / 0.393 | 0.091 / 0.099 / 0.099 / 0.099 | 0.241 / 0.238 / 0.239 / 0.241 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 188 / 188 / 188 / 188 | 0.184 / 0.230 / 0.244 / 0.245 | 0.225 / 0.227 / 0.228 / 0.227 | 0.255 / 0.266 / 0.269 / 0.269 | 0.652 / 0.730 / 0.723 / 0.723 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 1,525 / 1,525 / 1,525 / 1,525 | 0.120 / 0.132 / 0.142 / 0.178 | 0.490 / 0.495 / 0.497 / 0.504 | 0.208 / 0.209 / 0.210 / 0.211 | 0.299 / 0.297 / 0.298 / 0.302 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 1 / 1 / 1 / 1 | 0.157 / 0.157 / 0.157 / 0.157 | 2.579 / 2.579 / 2.579 / 2.579 | 0.270 / 0.270 / 0.270 / 0.270 | 0.034 / 0.034 / 0.034 / 0.034 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — optimal</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 8,463 / 9,288 / 9,554 / 9,621 | 0.178 / 0.214 / 0.240 / 0.263 | 0.140 / 0.176 / 0.194 / 0.196 | 0.162 / 0.168 / 0.170 / 0.171 | 0.241 / 0.261 / 0.269 / 0.269 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 617 / 739 / 799 / 804 | 0.205 / 0.274 / 0.347 / 0.361 | 0.046 / 0.062 / 0.070 / 0.079 | 0.173 / 0.184 / 0.195 / 0.197 | 0.262 / 0.270 / 0.275 / 0.283 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 103 / 124 / 124 / 124 | 0.222 / 0.283 / 0.283 / 0.283 | 0.374 / 0.341 / 0.341 / 0.341 | 0.097 / 0.107 / 0.107 / 0.107 | 0.296 / 0.295 / 0.295 / 0.295 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 263 / 289 / 292 / 292 | 0.183 / 0.227 / 0.237 / 0.237 | 0.249 / 0.257 / 0.255 / 0.255 | 0.256 / 0.268 / 0.271 / 0.271 | 0.631 / 0.726 / 0.721 / 0.721 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 2,604 / 2,652 / 2,667 / 2,682 | 0.124 / 0.139 / 0.152 / 0.191 | 0.521 / 0.526 / 0.528 / 0.535 | 0.211 / 0.212 / 0.213 / 0.214 | 0.310 / 0.308 / 0.310 / 0.313 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 2 / 2 / 2 / 2 | 0.157 / 0.157 / 0.157 / 0.157 | 2.579 / 2.579 / 2.579 / 2.579 | 0.270 / 0.270 / 0.270 / 0.270 | 0.034 / 0.034 / 0.034 / 0.034 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan |

</details>

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.4172**

| class_name | GTs | num_match@0.5/1.0/2.0/4.0 | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | :---- | :---- | :---- | :---- |
| car | 2,883 | 2,173 / 2,512 / 2,669 / 2,713 | 0.616 / 0.762 / 0.812 / 0.829 | 0.705 / 0.783 / 0.806 / 0.814 | 0.203 / 0.202 / 0.192 / 0.162 |
| truck | 600 | 343 / 461 / 519 / 534 | 0.393 / 0.654 / 0.778 / 0.809 | 0.572 / 0.727 / 0.808 / 0.824 | 0.279 / 0.178 / 0.145 / 0.145 |
| bus | 60 | 32 / 44 / 47 / 47 | 0.379 / 0.599 / 0.655 / 0.655 | 0.536 / 0.681 / 0.707 / 0.707 | 0.134 / 0.176 / 0.049 / 0.049 |
| bicycle | 85 | 54 / 61 / 66 / 66 | 0.262 / 0.373 / 0.434 / 0.434 | 0.433 / 0.528 / 0.579 / 0.579 | 0.102 / 0.144 / 0.144 / 0.144 |
| pedestrian | 1,092 | 945 / 960 / 963 / 979 | 0.545 / 0.554 / 0.562 / 0.576 | 0.638 / 0.644 / 0.646 / 0.651 | 0.145 / 0.145 / 0.145 / 0.135 |
| traffic_cone | 0 | 0 / 0 / 0 / 0 | 0.000 / 0.000 / 0.000 / 0.000 | nan / nan / nan / nan | nan / nan / nan / nan |
| barrier | 0 | 0 / 0 / 0 / 0 | 0.000 / 0.000 / 0.000 / 0.000 | nan / nan / nan / nan | nan / nan / nan / nan |

<details>
<summary><strong>TP error — default (recall @0.10)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 317 / 317 / 317 / 317 | 0.195 / 0.225 / 0.242 / 0.263 | 0.082 / 0.107 / 0.125 / 0.131 | 0.178 / 0.184 / 0.185 / 0.185 | 0.449 / 0.461 / 0.470 / 0.476 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 66 / 66 / 66 / 66 | 0.210 / 0.284 / 0.333 / 0.350 | 0.040 / 0.041 / 0.045 / 0.049 | 0.169 / 0.177 / 0.185 / 0.188 | 0.152 / 0.154 / 0.157 / 0.161 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 6 / 6 / 6 / 6 | 0.264 / 0.325 / 0.348 / 0.348 | 0.039 / 0.208 / 0.205 / 0.205 | 0.135 / 0.141 / 0.145 / 0.145 | 0.137 / 0.266 / 0.428 / 0.428 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 9 / 9 / 9 / 9 | 0.203 / 0.257 / 0.320 / 0.320 | 0.120 / 0.110 / 0.108 / 0.108 | 0.270 / 0.272 / 0.282 / 0.282 | 0.796 / 0.879 / 0.888 / 0.888 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 120 / 120 / 120 / 120 | 0.131 / 0.135 / 0.155 / 0.204 | 0.441 / 0.441 / 0.444 / 0.453 | 0.178 / 0.178 / 0.179 / 0.178 | 0.369 / 0.370 / 0.371 / 0.376 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — medium (recall @0.40)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 1,182 / 1,182 / 1,182 / 1,182 | 0.204 / 0.243 / 0.266 / 0.294 | 0.107 / 0.142 / 0.167 / 0.175 | 0.186 / 0.191 / 0.192 / 0.192 | 0.512 / 0.522 / 0.531 / 0.538 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 246 / 246 / 246 / 246 | 0.217 / 0.307 / 0.381 / 0.406 | 0.058 / 0.052 / 0.056 / 0.061 | 0.176 / 0.185 / 0.197 / 0.200 | 0.196 / 0.193 / 0.189 / 0.194 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 24 / 24 / 24 / 24 | 0.236 / 0.355 / 0.393 / 0.393 | 0.039 / 0.269 / 0.255 / 0.255 | 0.134 / 0.146 / 0.152 / 0.152 | 0.234 / 0.402 / 0.674 / 0.674 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 34 / 34 / 34 / 34 | 0.202 / 0.269 / 0.361 / 0.361 | 0.120 / 0.104 / 0.102 / 0.102 | 0.269 / 0.271 / 0.284 / 0.284 | 0.792 / 0.904 / 0.924 / 0.924 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 447 / 447 / 447 / 447 | 0.135 / 0.142 / 0.162 / 0.225 | 0.483 / 0.483 / 0.486 / 0.500 | 0.179 / 0.179 / 0.179 / 0.179 | 0.394 / 0.393 / 0.396 / 0.403 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — optimal</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 1,929 / 2,144 / 2,236 / 2,337 | 0.208 / 0.254 / 0.285 / 0.333 | 0.124 / 0.166 / 0.200 / 0.232 | 0.189 / 0.196 / 0.196 / 0.197 | 0.524 / 0.543 / 0.557 / 0.563 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 301 / 408 / 466 / 475 | 0.217 / 0.320 / 0.431 / 0.466 | 0.048 / 0.062 / 0.071 / 0.076 | 0.177 / 0.193 / 0.209 / 0.213 | 0.195 / 0.228 / 0.220 / 0.222 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 26 / 32 / 41 / 41 | 0.236 / 0.360 / 0.411 / 0.411 | 0.037 / 0.331 / 0.268 / 0.268 | 0.129 / 0.140 / 0.162 / 0.162 | 0.191 / 0.325 / 0.784 / 0.784 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 39 / 42 / 46 / 46 | 0.206 / 0.281 / 0.376 / 0.376 | 0.105 / 0.093 / 0.092 / 0.092 | 0.266 / 0.265 / 0.280 / 0.280 | 0.783 / 0.918 / 0.916 / 0.916 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 691 / 697 / 700 / 730 | 0.136 / 0.142 / 0.161 / 0.230 | 0.492 / 0.492 / 0.490 / 0.504 | 0.175 / 0.176 / 0.176 / 0.178 | 0.393 / 0.393 / 0.393 / 0.402 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 0 / 0 / 0 / 0 | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan |
| barrier | 0 / 0 / 0 / 0 | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan |

</details>

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.5779**

| class_name | GTs | num_match@0.5/1.0/2.0/4.0 | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | :---- | :---- | :---- | :---- |
| car | 28,684 | 24,624 / 26,214 / 26,901 / 27,216 | 0.811 / 0.885 / 0.909 / 0.920 | 0.862 / 0.903 / 0.915 / 0.920 | 0.236 / 0.200 / 0.166 / 0.166 |
| truck | 2,801 | 1,991 / 2,336 / 2,497 / 2,554 | 0.611 / 0.775 / 0.848 / 0.868 | 0.735 / 0.827 / 0.870 / 0.878 | 0.281 / 0.178 / 0.157 / 0.144 |
| bus | 537 | 407 / 510 / 517 / 519 | 0.662 / 0.919 / 0.938 / 0.938 | 0.771 / 0.907 / 0.914 / 0.914 | 0.486 / 0.125 / 0.125 / 0.125 |
| bicycle | 1,285 | 1,053 / 1,121 / 1,146 / 1,150 | 0.667 / 0.750 / 0.772 / 0.775 | 0.739 / 0.775 / 0.784 / 0.787 | 0.169 / 0.166 / 0.156 / 0.156 |
| pedestrian | 9,868 | 8,856 / 8,991 / 9,035 / 9,095 | 0.764 / 0.783 / 0.788 / 0.799 | 0.783 / 0.792 / 0.796 / 0.799 | 0.146 / 0.146 / 0.146 / 0.146 |
| traffic_cone | 64 | 22 / 23 / 23 / 24 | 0.000 / 0.000 / 0.000 / 0.000 | 0.033 / 0.034 / 0.034 / 0.036 | 0.079 / 0.065 / 0.065 / 0.065 |
| barrier | 0 | 0 / 0 / 0 / 0 | 0.000 / 0.000 / 0.000 / 0.000 | nan / nan / nan / nan | nan / nan / nan / nan |

<details>
<summary><strong>TP error — default (recall @0.10)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 3,155 / 3,155 / 3,155 / 3,155 | 0.127 / 0.136 / 0.141 / 0.145 | 0.053 / 0.060 / 0.064 / 0.065 | 0.128 / 0.130 / 0.131 / 0.131 | 0.168 / 0.173 / 0.175 / 0.176 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 308 / 308 / 308 / 308 | 0.169 / 0.202 / 0.224 / 0.232 | 0.034 / 0.036 / 0.038 / 0.041 | 0.145 / 0.152 / 0.156 / 0.157 | 0.204 / 0.206 / 0.211 / 0.212 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 59 / 59 / 59 / 59 | 0.163 / 0.201 / 0.205 / 0.205 | 0.310 / 0.286 / 0.285 / 0.285 | 0.085 / 0.092 / 0.093 / 0.093 | 0.190 / 0.188 / 0.194 / 0.194 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 141 / 141 / 141 / 141 | 0.149 / 0.173 / 0.182 / 0.187 | 0.128 / 0.128 / 0.129 / 0.128 | 0.224 / 0.233 / 0.235 / 0.236 | 0.561 / 0.594 / 0.593 / 0.593 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 1,085 / 1,085 / 1,085 / 1,085 | 0.103 / 0.108 / 0.115 / 0.137 | 0.352 / 0.355 / 0.356 / 0.364 | 0.209 / 0.209 / 0.210 / 0.210 | 0.265 / 0.264 / 0.265 / 0.268 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 7 / 7 / 7 / 7 | 0.155 / 0.190 / 0.190 / 0.300 | 1.777 / 1.717 / 1.717 / 1.666 | 0.439 / 0.454 / 0.454 / 0.471 | 0.073 / 0.078 / 0.078 / 0.079 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — medium (recall @0.40)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 11,760 / 11,760 / 11,760 / 11,760 | 0.139 / 0.151 / 0.157 / 0.163 | 0.063 / 0.073 / 0.078 / 0.080 | 0.135 / 0.138 / 0.139 / 0.139 | 0.186 / 0.191 / 0.194 / 0.195 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 1,148 / 1,148 / 1,148 / 1,148 | 0.179 / 0.221 / 0.253 / 0.263 | 0.040 / 0.042 / 0.045 / 0.049 | 0.154 / 0.161 / 0.167 / 0.168 | 0.222 / 0.222 / 0.226 / 0.227 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 220 / 220 / 220 / 220 | 0.177 / 0.227 / 0.231 / 0.232 | 0.270 / 0.250 / 0.250 / 0.250 | 0.091 / 0.099 / 0.100 / 0.100 | 0.201 / 0.196 / 0.204 / 0.204 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 526 / 526 / 526 / 526 | 0.155 / 0.177 / 0.191 / 0.197 | 0.145 / 0.145 / 0.146 / 0.145 | 0.230 / 0.237 / 0.240 / 0.241 | 0.570 / 0.595 / 0.592 / 0.592 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 4,045 / 4,045 / 4,045 / 4,045 | 0.110 / 0.117 / 0.126 / 0.154 | 0.397 / 0.401 / 0.402 / 0.410 | 0.212 / 0.213 / 0.213 / 0.213 | 0.276 / 0.275 / 0.276 / 0.279 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — optimal</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 23,528 / 24,950 / 25,596 / 25,735 | 0.155 / 0.180 / 0.198 / 0.216 | 0.093 / 0.117 / 0.133 / 0.137 | 0.147 / 0.152 / 0.153 / 0.154 | 0.214 / 0.228 / 0.234 / 0.235 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 1,866 / 2,196 / 2,328 / 2,367 | 0.189 / 0.251 / 0.308 / 0.333 | 0.043 / 0.054 / 0.061 / 0.073 | 0.161 / 0.172 / 0.181 / 0.184 | 0.222 / 0.230 / 0.230 / 0.234 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 379 / 486 / 490 / 490 | 0.187 / 0.278 / 0.288 / 0.288 | 0.246 / 0.230 / 0.229 / 0.229 | 0.095 / 0.110 / 0.112 / 0.112 | 0.215 / 0.212 / 0.234 / 0.234 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 874 / 918 / 941 / 944 | 0.156 / 0.179 / 0.199 / 0.206 | 0.151 / 0.153 / 0.162 / 0.161 | 0.231 / 0.238 / 0.243 / 0.244 | 0.577 / 0.602 / 0.594 / 0.593 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 7,461 / 7,553 / 7,587 / 7,622 | 0.116 / 0.125 / 0.136 / 0.167 | 0.429 / 0.432 / 0.433 / 0.442 | 0.213 / 0.214 / 0.215 / 0.215 | 0.287 / 0.286 / 0.287 / 0.291 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 19 / 22 / 22 / 23 | 0.156 / 0.203 / 0.203 / 0.324 | 1.731 / 1.638 / 1.638 / 1.580 | 0.430 / 0.472 / 0.472 / 0.487 | 0.079 / 0.082 / 0.082 / 0.083 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan |

</details>

---

**J6Gen2**: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 + db_j6gen2_v6 + db_j6gen2_v7 + db_j6gen2_v8 + db_j6gen2_v9 + db_j6gen2_v10 + db_j6gen2_v11 + db_j6gen2_v12 (4,682 frames)

**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.7371**

| class_name | GTs | num_match@0.5/1.0/2.0/4.0 | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | :---- | :---- | :---- | :---- |
| car | 60,938 | 53,537 / 55,808 / 56,797 / 57,392 | 0.844 / 0.892 / 0.914 / 0.925 | 0.898 / 0.925 / 0.933 / 0.938 | 0.251 / 0.220 / 0.157 / 0.147 |
| truck | 7,081 | 5,634 / 6,249 / 6,533 / 6,699 | 0.712 / 0.833 / 0.886 / 0.916 | 0.797 / 0.870 / 0.900 / 0.919 | 0.263 / 0.186 / 0.186 / 0.184 |
| bus | 2,370 | 2,078 / 2,230 / 2,293 / 2,305 | 0.822 / 0.913 / 0.952 / 0.963 | 0.885 / 0.940 / 0.961 / 0.963 | 0.243 / 0.168 / 0.153 / 0.153 |
| bicycle | 1,357 | 1,274 / 1,285 / 1,286 / 1,286 | 0.895 / 0.911 / 0.912 / 0.912 | 0.896 / 0.904 / 0.905 / 0.905 | 0.158 / 0.158 / 0.158 / 0.158 |
| pedestrian | 18,202 | 16,664 / 17,018 / 17,157 / 17,273 | 0.827 / 0.852 / 0.865 / 0.872 | 0.828 / 0.842 / 0.849 / 0.855 | 0.171 / 0.166 / 0.163 / 0.166 |
| traffic_cone | 8,250 | 5,459 / 5,894 / 6,075 / 6,309 | 0.430 / 0.490 / 0.511 / 0.545 | 0.582 / 0.619 / 0.634 / 0.654 | 0.123 / 0.110 / 0.095 / 0.086 |
| barrier | 1,350 | 572 / 754 / 803 / 843 | 0.175 / 0.270 / 0.292 / 0.310 | 0.410 / 0.464 / 0.474 / 0.485 | 0.283 / 0.248 / 0.248 / 0.248 |

<details>
<summary><strong>TP error — default (recall @0.10)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 6,703 / 6,703 / 6,703 / 6,703 | 0.107 / 0.111 / 0.113 / 0.117 | 0.031 / 0.034 / 0.036 / 0.036 | 0.116 / 0.117 / 0.118 / 0.118 | 0.123 / 0.124 / 0.125 / 0.126 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 778 / 778 / 778 / 778 | 0.144 / 0.166 / 0.182 / 0.204 | 0.028 / 0.030 / 0.031 / 0.031 | 0.128 / 0.132 / 0.135 / 0.137 | 0.312 / 0.322 / 0.328 / 0.327 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 260 / 260 / 260 / 260 | 0.105 / 0.115 / 0.137 / 0.139 | 0.026 / 0.027 / 0.028 / 0.028 | 0.083 / 0.085 / 0.091 / 0.091 | 0.122 / 0.125 / 0.123 / 0.124 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 149 / 149 / 149 / 149 | 0.129 / 0.133 / 0.133 / 0.133 | 0.067 / 0.068 / 0.068 / 0.068 | 0.194 / 0.194 / 0.194 / 0.194 | 0.530 / 0.530 / 0.530 / 0.530 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 2,002 / 2,002 / 2,002 / 2,002 | 0.105 / 0.111 / 0.121 / 0.138 | 0.427 / 0.429 / 0.433 / 0.435 | 0.240 / 0.241 / 0.241 / 0.241 | 0.236 / 0.236 / 0.236 / 0.237 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 907 / 907 / 907 / 907 | 0.176 / 0.198 / 0.219 / 0.296 | 0.327 / 0.324 / 0.326 / 0.327 | 0.645 / 0.648 / 0.649 / 0.650 | 0.026 / 0.026 / 0.026 / 0.026 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 148 / 148 / 148 / 148 | 0.232 / 0.293 / 0.318 / 0.363 | 0.374 / 0.376 / 0.378 / 0.375 | 0.458 / 0.477 / 0.484 / 0.492 | 0.024 / 0.025 / 0.025 / 0.025 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — medium (recall @0.40)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 24,984 / 24,984 / 24,984 / 24,984 | 0.115 / 0.121 / 0.124 / 0.130 | 0.036 / 0.040 / 0.042 / 0.043 | 0.122 / 0.123 / 0.123 / 0.124 | 0.136 / 0.137 / 0.138 / 0.139 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 2,903 / 2,903 / 2,903 / 2,903 | 0.153 / 0.181 / 0.204 / 0.235 | 0.031 / 0.034 / 0.035 / 0.036 | 0.132 / 0.138 / 0.142 / 0.145 | 0.336 / 0.345 / 0.352 / 0.350 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 971 / 971 / 971 / 971 | 0.113 / 0.129 / 0.146 / 0.148 | 0.030 / 0.032 / 0.032 / 0.033 | 0.086 / 0.089 / 0.093 / 0.094 | 0.140 / 0.142 / 0.141 / 0.141 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 556 / 556 / 556 / 556 | 0.128 / 0.131 / 0.131 / 0.132 | 0.070 / 0.070 / 0.070 / 0.070 | 0.200 / 0.200 / 0.201 / 0.201 | 0.558 / 0.558 / 0.558 / 0.558 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 7,462 / 7,462 / 7,462 / 7,462 | 0.109 / 0.117 / 0.132 / 0.154 | 0.455 / 0.457 / 0.462 / 0.464 | 0.245 / 0.247 / 0.247 / 0.248 | 0.241 / 0.240 / 0.241 / 0.242 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 3,382 / 3,382 / 3,382 / 3,382 | 0.187 / 0.215 / 0.248 / 0.352 | 0.373 / 0.367 / 0.367 / 0.369 | 0.655 / 0.659 / 0.660 / 0.661 | 0.028 / 0.028 / 0.027 / 0.028 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 553 / 553 / 553 / 553 | 0.251 / 0.333 / 0.368 / 0.443 | 0.422 / 0.406 / 0.410 / 0.403 | 0.542 / 0.541 / 0.547 / 0.554 | 0.025 / 0.025 / 0.025 / 0.025 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — optimal</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 52,338 / 54,149 / 55,189 / 55,565 | 0.130 / 0.146 / 0.161 / 0.177 | 0.050 / 0.061 / 0.071 / 0.074 | 0.131 / 0.133 / 0.134 / 0.135 | 0.148 / 0.151 / 0.153 / 0.153 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 5,350 / 5,996 / 6,202 / 6,337 | 0.164 / 0.212 / 0.249 / 0.305 | 0.041 / 0.055 / 0.055 / 0.058 | 0.141 / 0.153 / 0.158 / 0.164 | 0.333 / 0.345 / 0.353 / 0.353 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 2,027 / 2,173 / 2,227 / 2,232 | 0.134 / 0.168 / 0.193 / 0.202 | 0.049 / 0.054 / 0.059 / 0.061 | 0.095 / 0.101 / 0.106 / 0.106 | 0.171 / 0.166 / 0.166 / 0.168 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 1,170 / 1,181 / 1,182 / 1,182 | 0.131 / 0.137 / 0.137 / 0.137 | 0.076 / 0.077 / 0.077 / 0.077 | 0.209 / 0.210 / 0.210 / 0.210 | 0.563 / 0.562 / 0.563 / 0.563 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 14,547 / 14,883 / 15,058 / 15,106 | 0.115 / 0.128 / 0.149 / 0.185 | 0.478 / 0.482 / 0.488 / 0.491 | 0.251 / 0.253 / 0.254 / 0.254 | 0.248 / 0.248 / 0.249 / 0.250 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 4,519 / 4,942 / 5,249 / 5,546 | 0.187 / 0.219 / 0.265 / 0.403 | 0.382 / 0.379 / 0.382 / 0.383 | 0.654 / 0.661 / 0.666 / 0.670 | 0.028 / 0.028 / 0.028 / 0.028 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 453 / 538 / 550 / 562 | 0.240 / 0.301 / 0.323 / 0.375 | 0.394 / 0.397 / 0.395 / 0.389 | 0.491 / 0.508 / 0.510 / 0.516 | 0.023 / 0.024 / 0.024 / 0.024 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.5833**

| class_name | GTs | num_match@0.5/1.0/2.0/4.0 | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | :---- | :---- | :---- | :---- |
| car | 54,217 | 41,798 / 46,774 / 49,152 / 50,157 | 0.682 / 0.803 / 0.855 / 0.877 | 0.772 / 0.841 / 0.868 / 0.877 | 0.242 / 0.189 / 0.157 / 0.157 |
| truck | 4,913 | 2,974 / 3,651 / 4,060 / 4,258 | 0.440 / 0.619 / 0.724 / 0.772 | 0.610 / 0.720 / 0.775 / 0.797 | 0.249 / 0.164 / 0.164 / 0.164 |
| bus | 2,116 | 1,429 / 1,813 / 1,969 / 2,024 | 0.559 / 0.788 / 0.892 / 0.918 | 0.675 / 0.828 / 0.885 / 0.904 | 0.415 / 0.184 / 0.171 / 0.181 |
| bicycle | 838 | 666 / 706 / 708 / 709 | 0.642 / 0.708 / 0.713 / 0.716 | 0.723 / 0.752 / 0.755 / 0.756 | 0.110 / 0.136 / 0.136 / 0.110 |
| pedestrian | 8,336 | 7,155 / 7,320 / 7,400 / 7,475 | 0.621 / 0.643 / 0.660 / 0.674 | 0.681 / 0.691 / 0.698 / 0.704 | 0.145 / 0.155 / 0.155 / 0.148 |
| traffic_cone | 2,632 | 1,306 / 1,440 / 1,508 / 1,612 | 0.231 / 0.280 / 0.312 / 0.364 | 0.450 / 0.492 / 0.509 / 0.540 | 0.079 / 0.085 / 0.075 / 0.075 |
| barrier | 622 | 216 / 296 / 314 / 328 | 0.118 / 0.226 / 0.242 / 0.252 | 0.335 / 0.427 / 0.438 / 0.445 | 0.183 / 0.106 / 0.106 / 0.082 |

<details>
<summary><strong>TP error — default (recall @0.10)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 5,963 / 5,963 / 5,963 / 5,963 | 0.158 / 0.180 / 0.197 / 0.212 | 0.113 / 0.144 / 0.163 / 0.167 | 0.160 / 0.163 / 0.164 / 0.164 | 0.148 / 0.149 / 0.149 / 0.150 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 540 / 540 / 540 / 540 | 0.192 / 0.244 / 0.299 / 0.344 | 0.034 / 0.037 / 0.041 / 0.043 | 0.153 / 0.166 / 0.173 / 0.178 | 0.472 / 0.474 / 0.475 / 0.481 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 232 / 232 / 232 / 232 | 0.151 / 0.201 / 0.235 / 0.248 | 0.115 / 0.107 / 0.111 / 0.119 | 0.117 / 0.125 / 0.131 / 0.132 | 0.140 / 0.147 / 0.147 / 0.147 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 92 / 92 / 92 / 92 | 0.169 / 0.186 / 0.190 / 0.195 | 0.123 / 0.127 / 0.127 / 0.127 | 0.208 / 0.212 / 0.212 / 0.212 | 0.647 / 0.643 / 0.642 / 0.641 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 916 / 916 / 916 / 916 | 0.117 / 0.127 / 0.152 / 0.197 | 0.592 / 0.598 / 0.603 / 0.612 | 0.229 / 0.230 / 0.230 / 0.230 | 0.295 / 0.295 / 0.296 / 0.299 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 289 / 289 / 289 / 289 | 0.190 / 0.223 / 0.313 / 0.573 | 0.271 / 0.283 / 0.285 / 0.308 | 0.686 / 0.691 / 0.692 / 0.692 | 0.043 / 0.044 / 0.044 / 0.044 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 68 / 68 / 68 / 68 | 0.247 / 0.336 / 0.361 / 0.416 | 0.375 / 0.360 / 0.359 / 0.359 | 0.456 / 0.473 / 0.481 / 0.485 | 0.032 / 0.033 / 0.034 / 0.034 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — medium (recall @0.40)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 22,228 / 22,228 / 22,228 / 22,228 | 0.169 / 0.199 / 0.221 / 0.242 | 0.135 / 0.174 / 0.198 / 0.203 | 0.166 / 0.169 / 0.170 / 0.171 | 0.155 / 0.156 / 0.156 / 0.157 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 2,014 / 2,014 / 2,014 / 2,014 | 0.202 / 0.268 / 0.337 / 0.393 | 0.045 / 0.047 / 0.051 / 0.054 | 0.164 / 0.177 / 0.186 / 0.192 | 0.482 / 0.489 / 0.493 / 0.501 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 867 / 867 / 867 / 867 | 0.170 / 0.238 / 0.284 / 0.303 | 0.082 / 0.083 / 0.092 / 0.105 | 0.126 / 0.135 / 0.143 / 0.144 | 0.156 / 0.161 / 0.160 / 0.160 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 343 / 343 / 343 / 343 | 0.179 / 0.198 / 0.204 / 0.212 | 0.147 / 0.150 / 0.150 / 0.150 | 0.209 / 0.213 / 0.213 / 0.213 | 0.663 / 0.657 / 0.656 / 0.655 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 3,417 / 3,417 / 3,417 / 3,417 | 0.123 / 0.135 / 0.161 / 0.214 | 0.621 / 0.626 / 0.632 / 0.642 | 0.230 / 0.231 / 0.231 / 0.232 | 0.321 / 0.319 / 0.320 / 0.324 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 1,079 / 1,079 / 1,079 / 1,079 | 0.205 / 0.249 / 0.352 / 0.633 | 0.355 / 0.365 / 0.362 / 0.382 | 0.690 / 0.696 / 0.697 / 0.698 | 0.053 / 0.051 / 0.051 / 0.052 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 255 / 255 / 255 | 1.000 / 0.380 / 0.432 / 0.563 | 1.000 / 0.422 / 0.413 / 0.407 | 1.000 / 0.546 / 0.552 / 0.554 | 1.000 / 0.036 / 0.036 / 0.036 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — optimal</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 39,059 / 43,688 / 45,797 / 46,308 | 0.180 / 0.225 / 0.264 / 0.299 | 0.165 / 0.216 / 0.249 / 0.256 | 0.171 / 0.176 / 0.177 / 0.178 | 0.164 / 0.167 / 0.169 / 0.169 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 2,628 / 3,307 / 3,561 / 3,660 | 0.205 / 0.286 / 0.364 / 0.427 | 0.047 / 0.056 / 0.059 / 0.060 | 0.167 / 0.185 / 0.195 / 0.201 | 0.480 / 0.500 / 0.512 / 0.518 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 1,261 / 1,700 / 1,825 / 1,855 | 0.179 / 0.290 / 0.353 / 0.413 | 0.078 / 0.086 / 0.109 / 0.153 | 0.129 / 0.143 / 0.151 / 0.154 | 0.164 / 0.163 / 0.163 / 0.163 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 584 / 579 / 581 / 611 | 0.182 / 0.202 / 0.207 / 0.216 | 0.167 / 0.158 / 0.157 / 0.168 | 0.213 / 0.217 / 0.216 / 0.216 | 0.672 / 0.661 / 0.659 / 0.664 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 5,589 / 5,531 / 5,588 / 5,745 | 0.123 / 0.135 / 0.159 / 0.216 | 0.625 / 0.626 / 0.632 / 0.647 | 0.229 / 0.230 / 0.230 / 0.231 | 0.326 / 0.323 / 0.325 / 0.332 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 1,142 / 1,223 / 1,314 / 1,394 | 0.203 / 0.248 / 0.349 / 0.635 | 0.349 / 0.369 / 0.370 / 0.395 | 0.689 / 0.696 / 0.697 / 0.699 | 0.051 / 0.050 / 0.053 / 0.054 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 156 / 231 / 237 / 268 | 0.255 / 0.361 / 0.392 / 0.529 | 0.380 / 0.406 / 0.399 / 0.397 | 0.461 / 0.502 / 0.509 / 0.539 | 0.033 / 0.036 / 0.036 / 0.036 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.4384**

| class_name | GTs | num_match@0.5/1.0/2.0/4.0 | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | :---- | :---- | :---- | :---- |
| car | 19,301 | 13,605 / 16,188 / 17,481 / 17,847 | 0.528 / 0.703 / 0.787 / 0.811 | 0.660 / 0.759 / 0.799 / 0.810 | 0.188 / 0.181 / 0.156 / 0.156 |
| truck | 2,906 | 1,284 / 1,828 / 2,299 / 2,482 | 0.229 / 0.431 / 0.643 / 0.716 | 0.442 / 0.598 / 0.722 / 0.760 | 0.158 / 0.159 / 0.111 / 0.111 |
| bus | 484 | 225 / 324 / 385 / 401 | 0.261 / 0.534 / 0.668 / 0.702 | 0.460 / 0.635 / 0.719 / 0.742 | 0.349 / 0.126 / 0.066 / 0.066 |
| bicycle | 291 | 215 / 246 / 251 / 252 | 0.381 / 0.577 / 0.588 / 0.590 | 0.535 / 0.629 / 0.633 / 0.633 | 0.136 / 0.136 / 0.136 / 0.136 |
| pedestrian | 2,564 | 2,056 / 2,093 / 2,118 / 2,143 | 0.455 / 0.470 / 0.480 / 0.488 | 0.572 / 0.580 / 0.585 / 0.589 | 0.133 / 0.128 / 0.128 / 0.128 |
| traffic_cone | 462 | 183 / 207 / 225 / 235 | 0.114 / 0.146 / 0.161 / 0.183 | 0.324 / 0.359 / 0.372 / 0.388 | 0.088 / 0.088 / 0.088 / 0.088 |
| barrier | 145 | 49 / 72 / 90 / 96 | 0.042 / 0.140 / 0.204 / 0.242 | 0.237 / 0.362 / 0.427 / 0.452 | 0.139 / 0.119 / 0.085 / 0.095 |

<details>
<summary><strong>TP error — default (recall @0.10)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 2,123 / 2,123 / 2,123 / 2,123 | 0.199 / 0.243 / 0.281 / 0.312 | 0.217 / 0.268 / 0.299 / 0.309 | 0.180 / 0.184 / 0.186 / 0.186 | 0.274 / 0.267 / 0.265 / 0.265 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 319 / 319 / 319 / 319 | 0.232 / 0.319 / 0.462 / 0.532 | 0.044 / 0.048 / 0.057 / 0.062 | 0.178 / 0.194 / 0.215 / 0.222 | 0.461 / 0.495 / 0.507 / 0.513 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 53 / 53 / 53 / 53 | 0.230 / 0.326 / 0.388 / 0.414 | 0.036 / 0.034 / 0.039 / 0.040 | 0.143 / 0.158 / 0.167 / 0.170 | 0.416 / 0.429 / 0.428 / 0.430 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 32 / 32 / 32 / 32 | 0.245 / 0.306 / 0.312 / 0.316 | 0.096 / 0.091 / 0.093 / 0.092 | 0.247 / 0.264 / 0.265 / 0.264 | 0.779 / 0.757 / 0.761 / 0.763 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 282 / 282 / 282 / 282 | 0.124 / 0.137 / 0.155 / 0.193 | 0.525 / 0.533 / 0.542 / 0.546 | 0.255 / 0.256 / 0.256 / 0.257 | 0.389 / 0.389 / 0.389 / 0.391 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 50 / 50 / 50 / 50 | 0.193 / 0.234 / 0.272 / 0.526 | 0.288 / 0.286 / 0.309 / 0.312 | 0.702 / 0.701 / 0.703 / 0.699 | 0.044 / 0.046 / 0.046 / 0.046 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 15 / 15 / 15 / 15 | 0.301 / 0.435 / 0.535 / 0.997 | 0.250 / 0.220 / 0.216 / 0.212 | 0.487 / 0.511 / 0.530 / 0.530 | 0.045 / 0.045 / 0.046 / 0.046 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — medium (recall @0.40)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 7,913 / 7,913 / 7,913 / 7,913 | 0.208 / 0.262 / 0.308 / 0.347 | 0.271 / 0.321 / 0.353 / 0.363 | 0.186 / 0.189 / 0.191 / 0.191 | 0.272 / 0.266 / 0.265 / 0.265 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 1,191 / 1,191 / 1,191 / 1,191 | 0.247 / 0.356 / 0.521 / 0.613 | 0.072 / 0.063 / 0.071 / 0.078 | 0.191 / 0.206 / 0.226 / 0.235 | 0.513 / 0.557 / 0.558 / 0.562 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 198 / 198 / 198 / 198 | 0.250 / 0.363 / 0.453 / 0.493 | 0.063 / 0.047 / 0.051 / 0.053 | 0.151 / 0.164 / 0.177 / 0.181 | 0.456 / 0.444 / 0.437 / 0.441 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 119 / 119 / 119 / 119 | 0.227 / 0.289 / 0.299 / 0.306 | 0.129 / 0.119 / 0.120 / 0.119 | 0.241 / 0.257 / 0.258 / 0.258 | 0.760 / 0.771 / 0.776 / 0.781 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 1,051 / 1,051 / 1,051 / 1,051 | 0.131 / 0.147 / 0.171 / 0.217 | 0.581 / 0.588 / 0.599 / 0.605 | 0.252 / 0.252 / 0.253 / 0.253 | 0.427 / 0.425 / 0.425 / 0.427 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 0 / 189 / 189 / 189 | 1.000 / 0.274 / 0.365 / 0.629 | 1.000 / 0.405 / 0.430 / 0.444 | 1.000 / 0.718 / 0.719 / 0.714 | 1.000 / 0.060 / 0.058 / 0.055 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 59 / 59 / 59 | 1.000 / 0.438 / 0.604 / 1.056 | 1.000 / 0.237 / 0.228 / 0.223 | 1.000 / 0.567 / 0.578 / 0.581 | 1.000 / 0.045 / 0.047 / 0.047 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — optimal</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 12,130 / 14,043 / 15,222 / 15,426 | 0.212 / 0.274 / 0.334 / 0.381 | 0.306 / 0.364 / 0.410 / 0.419 | 0.188 / 0.193 / 0.195 / 0.195 | 0.272 / 0.267 / 0.270 / 0.270 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 1,162 / 1,568 / 2,007 / 2,112 | 0.243 / 0.360 / 0.552 / 0.656 | 0.052 / 0.061 / 0.085 / 0.094 | 0.188 / 0.209 / 0.235 / 0.244 | 0.533 / 0.561 / 0.554 / 0.558 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 169 / 279 / 355 / 366 | 0.240 / 0.375 / 0.521 / 0.578 | 0.045 / 0.057 / 0.062 / 0.063 | 0.143 / 0.167 / 0.187 / 0.193 | 0.407 / 0.458 / 0.445 / 0.451 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 153 / 180 / 181 / 181 | 0.228 / 0.284 / 0.290 / 0.290 | 0.086 / 0.094 / 0.094 / 0.094 | 0.240 / 0.255 / 0.256 / 0.256 | 0.747 / 0.757 / 0.760 / 0.760 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 1,439 / 1,486 / 1,498 / 1,509 | 0.127 / 0.142 / 0.164 / 0.206 | 0.558 / 0.566 / 0.576 / 0.581 | 0.250 / 0.251 / 0.252 / 0.253 | 0.425 / 0.424 / 0.423 / 0.425 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 155 / 172 / 178 / 186 | 0.209 / 0.261 / 0.313 / 0.600 | 0.413 / 0.390 / 0.401 / 0.422 | 0.714 / 0.709 / 0.710 / 0.705 | 0.056 / 0.055 / 0.055 / 0.054 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 33 / 52 / 72 / 73 | 0.295 / 0.437 / 0.603 / 1.064 | 0.239 / 0.211 / 0.222 / 0.219 | 0.469 / 0.512 / 0.568 / 0.572 | 0.048 / 0.047 / 0.047 / 0.047 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.6650**

| class_name | GTs | num_match@0.5/1.0/2.0/4.0 | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | :---- | :---- | :---- | :---- |
| car | 134,456 | 109,151 / 119,093 / 123,838 / 125,849 | 0.751 / 0.842 / 0.886 / 0.900 | 0.816 / 0.870 / 0.890 / 0.897 | 0.260 / 0.189 / 0.157 / 0.157 |
| truck | 14,900 | 9,912 / 11,766 / 12,962 / 13,522 | 0.538 / 0.695 / 0.797 / 0.845 | 0.673 / 0.772 / 0.828 / 0.851 | 0.264 / 0.176 / 0.166 / 0.157 |
| bus | 4,970 | 3,740 / 4,384 / 4,667 / 4,750 | 0.669 / 0.836 / 0.909 / 0.931 | 0.761 / 0.869 / 0.909 / 0.919 | 0.314 / 0.182 / 0.172 / 0.126 |
| bicycle | 2,486 | 2,157 / 2,239 / 2,247 / 2,249 | 0.763 / 0.818 / 0.821 / 0.822 | 0.795 / 0.821 / 0.823 / 0.823 | 0.152 / 0.152 / 0.152 / 0.152 |
| pedestrian | 29,102 | 25,909 / 26,467 / 26,706 / 26,922 | 0.753 / 0.773 / 0.787 / 0.799 | 0.764 / 0.777 / 0.784 / 0.790 | 0.154 / 0.154 / 0.162 / 0.159 |
| traffic_cone | 11,344 | 6,960 / 7,555 / 7,823 / 8,173 | 0.372 / 0.429 / 0.451 / 0.493 | 0.542 / 0.580 / 0.595 / 0.619 | 0.112 / 0.086 / 0.089 / 0.086 |
| barrier | 2,117 | 839 / 1,125 / 1,212 / 1,272 | 0.145 / 0.244 / 0.267 / 0.284 | 0.367 / 0.441 / 0.453 / 0.462 | 0.274 / 0.185 / 0.185 / 0.182 |

<details>
<summary><strong>TP error — default (recall @0.10)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 14,790 / 14,790 / 14,790 / 14,790 | 0.129 / 0.143 / 0.154 / 0.163 | 0.065 / 0.081 / 0.092 / 0.094 | 0.134 / 0.137 / 0.138 / 0.139 | 0.143 / 0.145 / 0.147 / 0.147 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 1,639 / 1,639 / 1,639 / 1,639 | 0.164 / 0.202 / 0.244 / 0.282 | 0.031 / 0.034 / 0.037 / 0.039 | 0.138 / 0.147 / 0.154 / 0.159 | 0.368 / 0.381 / 0.390 / 0.392 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 546 / 546 / 546 / 546 | 0.123 / 0.153 / 0.179 / 0.187 | 0.055 / 0.055 / 0.058 / 0.061 | 0.095 / 0.101 / 0.107 / 0.108 | 0.139 / 0.146 / 0.146 / 0.147 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 273 / 273 / 273 / 273 | 0.145 / 0.157 / 0.158 / 0.160 | 0.080 / 0.082 / 0.082 / 0.082 | 0.203 / 0.206 / 0.206 / 0.206 | 0.575 / 0.575 / 0.575 / 0.575 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 3,201 / 3,201 / 3,201 / 3,201 | 0.109 / 0.116 / 0.131 / 0.156 | 0.468 / 0.470 / 0.474 / 0.479 | 0.240 / 0.241 / 0.241 / 0.242 | 0.252 / 0.252 / 0.252 / 0.254 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 1,247 / 1,247 / 1,247 / 1,247 | 0.179 / 0.203 / 0.234 / 0.349 | 0.321 / 0.319 / 0.321 / 0.326 | 0.652 / 0.656 / 0.657 / 0.659 | 0.029 / 0.029 / 0.029 / 0.030 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 232 / 232 / 232 / 232 | 0.239 / 0.311 / 0.339 / 0.410 | 0.374 / 0.369 / 0.369 / 0.366 | 0.469 / 0.488 / 0.494 / 0.502 | 0.026 / 0.027 / 0.028 / 0.028 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — medium (recall @0.40)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 55,126 / 55,126 / 55,126 / 55,126 | 0.143 / 0.162 / 0.177 / 0.189 | 0.084 / 0.107 / 0.122 / 0.124 | 0.143 / 0.146 / 0.148 / 0.148 | 0.156 / 0.157 / 0.158 / 0.159 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 6,109 / 6,109 / 6,109 / 6,109 | 0.178 / 0.227 / 0.284 / 0.336 | 0.038 / 0.041 / 0.044 / 0.047 | 0.148 / 0.158 / 0.167 / 0.172 | 0.391 / 0.406 / 0.414 / 0.417 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 2,037 / 2,037 / 2,037 / 2,037 | 0.139 / 0.180 / 0.208 / 0.220 | 0.054 / 0.056 / 0.059 / 0.065 | 0.103 / 0.110 / 0.115 / 0.117 | 0.162 / 0.167 / 0.166 / 0.167 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 1,019 / 1,019 / 1,019 / 1,019 | 0.148 / 0.162 / 0.164 / 0.166 | 0.091 / 0.092 / 0.092 / 0.093 | 0.208 / 0.211 / 0.211 / 0.211 | 0.602 / 0.602 / 0.602 / 0.602 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 11,931 / 11,931 / 11,931 / 11,931 | 0.114 / 0.124 / 0.144 / 0.178 | 0.503 / 0.505 / 0.510 / 0.516 | 0.245 / 0.245 / 0.246 / 0.247 | 0.266 / 0.265 / 0.265 / 0.268 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 4,651 / 4,651 / 4,651 / 4,651 | 0.191 / 0.223 / 0.268 / 0.415 | 0.368 / 0.365 / 0.365 / 0.370 | 0.664 / 0.668 / 0.669 / 0.671 | 0.032 / 0.032 / 0.032 / 0.032 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 867 / 867 / 867 | 1.000 / 0.355 / 0.402 / 0.517 | 1.000 / 0.398 / 0.397 / 0.391 | 1.000 / 0.544 / 0.550 / 0.557 | 1.000 / 0.029 / 0.029 / 0.030 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — optimal</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 102,297 / 112,278 / 116,527 / 117,516 | 0.157 / 0.194 / 0.225 / 0.252 | 0.117 / 0.160 / 0.186 / 0.191 | 0.152 / 0.157 / 0.159 / 0.160 | 0.168 / 0.172 / 0.175 / 0.175 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 9,001 / 10,847 / 11,695 / 12,090 | 0.184 / 0.255 / 0.332 / 0.401 | 0.043 / 0.055 / 0.059 / 0.063 | 0.153 / 0.170 / 0.182 / 0.189 | 0.399 / 0.423 / 0.437 / 0.439 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 3,495 / 4,146 / 4,350 / 4,483 | 0.158 / 0.232 / 0.278 / 0.322 | 0.056 / 0.066 / 0.079 / 0.102 | 0.110 / 0.122 / 0.129 / 0.133 | 0.174 / 0.182 / 0.183 / 0.187 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 1,859 / 1,919 / 1,923 / 1,923 | 0.153 / 0.169 / 0.172 / 0.173 | 0.100 / 0.101 / 0.101 / 0.101 | 0.212 / 0.215 / 0.215 / 0.215 | 0.608 / 0.609 / 0.610 / 0.610 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 21,639 / 21,986 / 21,863 / 22,136 | 0.118 / 0.131 / 0.152 / 0.194 | 0.521 / 0.525 / 0.528 / 0.535 | 0.246 / 0.248 / 0.248 / 0.249 | 0.278 / 0.277 / 0.277 / 0.280 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 5,782 / 6,632 / 6,752 / 7,077 | 0.190 / 0.231 / 0.284 / 0.453 | 0.370 / 0.381 / 0.379 / 0.385 | 0.662 / 0.674 / 0.674 / 0.676 | 0.032 / 0.033 / 0.033 / 0.033 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 602 / 827 / 850 / 872 | 0.245 / 0.329 / 0.356 / 0.433 | 0.378 / 0.383 / 0.380 / 0.377 | 0.478 / 0.506 / 0.509 / 0.515 | 0.026 / 0.027 / 0.028 / 0.028 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>
</details>

---

### BEVFusion-LiDAR J6Gen2_base/2.7.1

<details>
<summary> Changes  </summary>

- Finetune from `BEVFusion-LiDAR base/2.7.0` with j6gen2 base dataset and intensity.
</details>

<details>
<summary> Artifacts </summary>

- Deployed onnx and ROS parameter files (for internal)
  - [WebAuto](https://evaluation.tier4.jp/evaluation/mlpackages/46f8188d-e3be-4f2f-b989-fd27002610d7/releases/ab0f33f5-2c8e-4adf-b122-f8f0c229c91e?project_id=zWhWRzei)
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-l/j6gen2_base/v2.7.1/deployment.zip)
  - [Google drive](https://drive.google.com/file/d/1Sw2UkqsoOP_YhoPpLqaBvHFnBapBV1kw/view?usp=drive_link)
- Logs (for internal)
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-l/j6gen2_base/v2.7.1/logs.zip)
  - [Google drive](https://drive.google.com/file/d/1M_Ae0rQ9L1I4NbzSL9tlJ8D0KVGvunKF/view?usp=drive_link)
- Pytorch Best checkpoints:
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-l/j6gen2_base/v2.7.1/best_epoch_28.pth)
  - [Google drive](https://drive.google.com/file/d/1xsFKCIkqVnt273o2SKjjCayuh_4IV-Vd/view?usp=drive_link)

</details>

<details>
<summary> Training configs </summary>

- [Config file path](https://github.com/KSeangTan/AWML/blob/07c2e110802ec2537d4c620d9af7f7e1b8120b97/projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_30e_8xb8_j6gen2_base_120m.py)
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

**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.8828**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 64,520 | 0.9022 | 0.853 / 0.901 / 0.921 / 0.933 | 0.904 / 0.931 / 0.937 / 0.939 | 0.260 / 0.193 / 0.180 / 0.172 |
| truck | 6,947 | 0.8627 | 0.736 / 0.863 / 0.910 / 0.942 | 0.800 / 0.877 / 0.903 / 0.920 | 0.244 / 0.191 / 0.188 / 0.166 |
| bus | 2,275 | 0.9440 | 0.866 / 0.940 / 0.983 / 0.986 | 0.912 / 0.958 / 0.978 / 0.980 | 0.203 / 0.177 / 0.163 / 0.138 |
| bicycle | 1,379 | 0.8483 | 0.802 / 0.849 / 0.869 / 0.874 | 0.847 / 0.867 / 0.876 / 0.879 | 0.205 / 0.191 / 0.172 / 0.172 |
| pedestrian | 19,421 | 0.8569 | 0.834 / 0.854 / 0.865 / 0.875 | 0.822 / 0.833 / 0.838 / 0.844 | 0.163 / 0.152 / 0.152 / 0.152 |
| **ALL** | 94,542 | 0.8828 | — | — | — |

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.7193**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 58,562 | 0.8197 | 0.694 / 0.818 / 0.873 / 0.893 | 0.782 / 0.853 / 0.879 / 0.886 | 0.228 / 0.173 / 0.164 / 0.164 |
| truck | 5,101 | 0.6856 | 0.484 / 0.670 / 0.773 / 0.815 | 0.633 / 0.743 / 0.798 / 0.816 | 0.213 / 0.206 / 0.184 / 0.164 |
| bus | 2,078 | 0.8249 | 0.626 / 0.815 / 0.918 / 0.941 | 0.730 / 0.846 / 0.904 / 0.919 | 0.342 / 0.211 / 0.210 / 0.160 |
| bicycle | 758 | 0.5862 | 0.495 / 0.603 / 0.622 / 0.624 | 0.637 / 0.679 / 0.683 / 0.683 | 0.183 / 0.155 / 0.155 / 0.183 |
| pedestrian | 10,283 | 0.6801 | 0.650 / 0.676 / 0.691 / 0.703 | 0.692 / 0.705 / 0.713 / 0.720 | 0.136 / 0.136 / 0.136 / 0.136 |
| **ALL** | 76,782 | 0.7193 | — | — | — |

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.5223**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 20,371 | 0.6814 | 0.493 / 0.674 / 0.763 / 0.796 | 0.638 / 0.737 / 0.781 / 0.795 | 0.193 / 0.159 / 0.151 / 0.151 |
| truck | 3,172 | 0.5181 | 0.227 / 0.454 / 0.652 / 0.738 | 0.447 / 0.601 / 0.715 / 0.762 | 0.206 / 0.206 / 0.162 / 0.140 |
| bus | 376 | 0.5381 | 0.272 / 0.557 / 0.643 / 0.680 | 0.462 / 0.669 / 0.714 / 0.731 | 0.217 / 0.151 / 0.115 / 0.115 |
| bicycle | 155 | 0.4165 | 0.316 / 0.419 / 0.466 / 0.466 | 0.487 / 0.553 / 0.589 / 0.589 | 0.199 / 0.166 / 0.190 / 0.190 |
| pedestrian | 2,794 | 0.4573 | 0.443 / 0.452 / 0.462 / 0.472 | 0.564 / 0.569 / 0.573 / 0.578 | 0.120 / 0.120 / 0.120 / 0.120 |
| **ALL** | 26,868 | 0.5223 | — | — | — |

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.7990**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 143,453 | 0.8508 | 0.752 / 0.849 / 0.891 / 0.910 | 0.820 / 0.874 / 0.894 / 0.900 | 0.232 / 0.189 / 0.174 / 0.164 |
| truck | 15,220 | 0.7435 | 0.555 / 0.725 / 0.824 / 0.871 | 0.677 / 0.780 / 0.834 / 0.858 | 0.234 / 0.206 / 0.186 / 0.165 |
| bus | 4,729 | 0.8711 | 0.726 / 0.865 / 0.939 / 0.954 | 0.804 / 0.890 / 0.928 / 0.937 | 0.408 / 0.211 / 0.177 / 0.161 |
| bicycle | 2,292 | 0.7487 | 0.682 / 0.754 / 0.777 / 0.781 | 0.760 / 0.789 / 0.799 / 0.801 | 0.191 / 0.189 / 0.189 / 0.190 |
| pedestrian | 32,498 | 0.7809 | 0.756 / 0.777 / 0.790 / 0.801 | 0.760 / 0.772 / 0.778 / 0.784 | 0.151 / 0.136 / 0.136 / 0.136 |
| **ALL** | 198,192 | 0.7990 | — | — | — |

---

**LargeBus**: db_largebus_v1 + db_largebus_v2 + db_largebus_v3 (1,228 frames)  

**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.8947**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 14,883 | 0.9231 | 0.884 / 0.925 / 0.937 / 0.946 | 0.923 / 0.947 / 0.952 / 0.953 | 0.234 / 0.178 / 0.178 / 0.178 |
| truck | 1,193 | 0.8893 | 0.754 / 0.905 / 0.938 / 0.961 | 0.832 / 0.922 / 0.939 / 0.945 | 0.269 / 0.201 / 0.188 / 0.116 |
| bus | 336 | 0.9564 | 0.872 / 0.983 / 0.985 / 0.986 | 0.904 / 0.962 / 0.965 / 0.965 | 0.419 / 0.174 / 0.174 / 0.174 |
| bicycle | 740 | 0.8264 | 0.749 / 0.825 / 0.862 / 0.870 | 0.824 / 0.854 / 0.867 / 0.872 | 0.249 / 0.247 / 0.198 / 0.198 |
| pedestrian | 5,059 | 0.8782 | 0.862 / 0.876 / 0.883 / 0.891 | 0.849 / 0.857 / 0.861 / 0.866 | 0.148 / 0.148 / 0.139 / 0.140 |
| **ALL** | 22,211 | 0.8947 | — | — | — |

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.7679**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 10,994 | 0.8567 | 0.759 / 0.860 / 0.897 / 0.911 | 0.824 / 0.881 / 0.898 / 0.901 | 0.210 / 0.164 / 0.160 / 0.160 |
| truck | 1,011 | 0.7666 | 0.593 / 0.770 / 0.843 / 0.860 | 0.710 / 0.818 / 0.851 / 0.854 | 0.234 / 0.219 / 0.166 / 0.150 |
| bus | 143 | 0.8723 | 0.698 / 0.921 / 0.932 / 0.939 | 0.788 / 0.904 / 0.911 / 0.911 | 0.294 / 0.498 / 0.498 / 0.498 |
| bicycle | 463 | 0.5955 | 0.472 / 0.616 / 0.647 / 0.648 | 0.625 / 0.685 / 0.692 / 0.692 | 0.151 / 0.151 / 0.151 / 0.151 |
| pedestrian | 3,754 | 0.7485 | 0.726 / 0.747 / 0.755 / 0.766 | 0.740 / 0.749 / 0.755 / 0.761 | 0.124 / 0.124 / 0.121 / 0.121 |
| **ALL** | 16,365 | 0.7679 | — | — | — |

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.5924**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 3,018 | 0.7238 | 0.573 / 0.728 / 0.789 / 0.806 | 0.688 / 0.765 / 0.792 / 0.801 | 0.221 / 0.228 / 0.158 / 0.158 |
| truck | 602 | 0.6616 | 0.381 / 0.676 / 0.780 / 0.809 | 0.575 / 0.756 / 0.811 / 0.822 | 0.216 / 0.208 / 0.176 / 0.176 |
| bus | 60 | 0.6305 | 0.434 / 0.626 / 0.730 / 0.732 | 0.608 / 0.745 / 0.793 / 0.793 | 0.217 / 0.217 / 0.087 / 0.087 |
| bicycle | 85 | 0.3964 | 0.298 / 0.382 / 0.452 / 0.453 | 0.468 / 0.544 / 0.595 / 0.595 | 0.166 / 0.166 / 0.166 / 0.166 |
| pedestrian | 1,121 | 0.5497 | 0.536 / 0.546 / 0.552 / 0.565 | 0.624 / 0.629 / 0.633 / 0.638 | 0.120 / 0.118 / 0.118 / 0.118 |
| **ALL** | 4,886 | 0.5924 | — | — | — |

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.8267**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 28,895 | 0.8888 | 0.815 / 0.891 / 0.919 / 0.930 | 0.864 / 0.905 / 0.917 / 0.919 | 0.230 / 0.180 / 0.180 / 0.176 |
| truck | 2,806 | 0.8055 | 0.623 / 0.816 / 0.879 / 0.903 | 0.736 / 0.851 / 0.882 / 0.888 | 0.233 / 0.207 / 0.183 / 0.169 |
| bus | 539 | 0.9009 | 0.783 / 0.929 / 0.945 / 0.948 | 0.838 / 0.921 / 0.929 / 0.929 | 0.430 / 0.208 / 0.208 / 0.208 |
| bicycle | 1,288 | 0.7334 | 0.637 / 0.738 / 0.776 / 0.783 | 0.730 / 0.774 / 0.793 / 0.796 | 0.186 / 0.161 / 0.161 / 0.161 |
| pedestrian | 9,934 | 0.8051 | 0.787 / 0.803 / 0.811 / 0.820 | 0.782 / 0.790 / 0.796 / 0.801 | 0.149 / 0.135 / 0.128 / 0.135 |
| **ALL** | 43,462 | 0.8267 | — | — | — |

---

**J6Gen2**: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 + db_j6gen2_v6 + db_j6gen2_v7 + db_j6gen2_v8 + db_j6gen2_v9 (3,951 frames)

**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.8836**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 49,637 | 0.8942 | 0.843 / 0.891 / 0.912 / 0.931 | 0.899 / 0.926 / 0.933 / 0.935 | 0.277 / 0.202 / 0.189 / 0.172 |
| truck | 5,754 | 0.8569 | 0.732 / 0.854 / 0.905 / 0.937 | 0.794 / 0.867 / 0.896 / 0.915 | 0.244 / 0.191 / 0.189 / 0.180 |
| bus | 1,939 | 0.9393 | 0.864 / 0.932 / 0.975 / 0.986 | 0.916 / 0.958 / 0.981 / 0.984 | 0.203 / 0.187 / 0.139 / 0.138 |
| bicycle | 639 | 0.8780 | 0.868 / 0.881 / 0.881 / 0.882 | 0.881 / 0.888 / 0.888 / 0.888 | 0.172 / 0.172 / 0.172 / 0.172 |
| pedestrian | 14,362 | 0.8494 | 0.824 / 0.846 / 0.858 / 0.869 | 0.813 / 0.825 / 0.831 / 0.837 | 0.163 / 0.161 / 0.155 / 0.155 |
| **ALL** | 72,331 | 0.8836 | — | — | — |

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.7040**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 47,568 | 0.8118 | 0.679 / 0.810 / 0.868 / 0.890 | 0.772 / 0.846 / 0.874 / 0.883 | 0.228 / 0.173 / 0.164 / 0.163 |
| truck | 4,090 | 0.6662 | 0.459 / 0.645 / 0.757 / 0.804 | 0.614 / 0.724 / 0.785 / 0.807 | 0.213 / 0.206 / 0.184 / 0.164 |
| bus | 1,935 | 0.8221 | 0.621 / 0.806 / 0.919 / 0.943 | 0.727 / 0.842 / 0.904 / 0.921 | 0.413 / 0.211 / 0.206 / 0.160 |
| bicycle | 295 | 0.5781 | 0.542 / 0.588 / 0.590 / 0.592 | 0.674 / 0.686 / 0.686 / 0.690 | 0.215 / 0.206 / 0.206 / 0.206 |
| pedestrian | 6,529 | 0.6417 | 0.608 / 0.636 / 0.655 / 0.668 | 0.666 / 0.682 / 0.692 / 0.699 | 0.136 / 0.136 / 0.136 / 0.136 |
| **ALL** | 60,417 | 0.7040 | — | — | — |

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.5030**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 17,353 | 0.6739 | 0.479 / 0.664 / 0.759 / 0.794 | 0.631 / 0.732 / 0.780 / 0.794 | 0.193 / 0.159 / 0.146 / 0.146 |
| truck | 2,570 | 0.4847 | 0.194 / 0.401 / 0.621 / 0.723 | 0.414 / 0.562 / 0.692 / 0.751 | 0.206 / 0.179 / 0.130 / 0.128 |
| bus | 316 | 0.5186 | 0.238 / 0.541 / 0.625 / 0.670 | 0.433 / 0.657 / 0.703 / 0.724 | 0.218 / 0.151 / 0.115 / 0.115 |
| bicycle | 70 | 0.4430 | 0.340 / 0.465 / 0.483 / 0.483 | 0.513 / 0.584 / 0.602 / 0.602 | 0.199 / 0.199 / 0.199 / 0.199 |
| pedestrian | 1,673 | 0.3948 | 0.381 / 0.389 / 0.401 / 0.408 | 0.524 / 0.528 / 0.532 / 0.535 | 0.125 / 0.125 / 0.125 / 0.125 |
| **ALL** | 21,982 | 0.5030 | — | — | — |

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.7958**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 114,558 | 0.8408 | 0.737 / 0.837 / 0.882 / 0.906 | 0.809 / 0.866 / 0.888 / 0.895 | 0.236 / 0.189 / 0.164 / 0.164 |
| truck | 12,414 | 0.7294 | 0.539 / 0.704 / 0.811 / 0.863 | 0.664 / 0.764 / 0.823 / 0.851 | 0.244 / 0.206 / 0.183 / 0.164 |
| bus | 4,190 | 0.8673 | 0.719 / 0.856 / 0.939 / 0.956 | 0.800 / 0.886 / 0.928 / 0.939 | 0.342 / 0.211 / 0.161 / 0.161 |
| bicycle | 1,004 | 0.7710 | 0.747 / 0.778 / 0.780 / 0.780 | 0.801 / 0.813 / 0.814 / 0.815 | 0.191 / 0.191 / 0.191 / 0.191 |
| pedestrian | 22,564 | 0.7706 | 0.743 / 0.766 / 0.781 / 0.792 | 0.751 / 0.764 / 0.771 / 0.778 | 0.152 / 0.146 / 0.136 / 0.146 |
| **ALL** | 154,730 | 0.7958 | — | — | — |

</details>

---

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
