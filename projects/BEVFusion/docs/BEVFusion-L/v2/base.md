# Deployed model for BEVFusion-LiDAR base/2.X
## Summary

### Main Parameters

  - **Range:** [122.40m, 122.40m, 8.0m]
  - **Voxel Size:** [0.17, 0.17, 0.2]
  - **Grid Size:** [1440, 1440, 40]

### Testing Datasets

- **Total Frames: 16,597**

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
  <summary> jpntaxi_gen2 (10,687 frames) </summary>

  - `db_jpntaxigen2_v1`
  - `db_jpntaxigen2_v2`

  </details>

### mAP - Base
- Note that the metrics reported in `traffic_cone/barrier` might not be accurate since some of the evaluation dataset doesn't have annotations for the two classes.

- **Class mAP for BEV Center Distance: 0.5m, 1.0m, 2.0m, 4.0m**

  <details>
  <summary> Eval Range: 0.0 - 50.0m </summary>

  | Model version | mAP | mAPH | map_based_nds (recall @ 0.10) | map_based_nds (recall @ 0.40) | maph_based_nds (recall @ 0.10) | maph_based_nds (recall 0.40) | car<br>(126,168) | truck<br>(26,897) | bus<br>(6,559) | bicycle<br>(5,865) | pedestrian<br>(93,520) | traffic_cone<br>(20,835) | barrier<br>(3,359) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.8.0 | 0.6763 | 0.6381 | 0.6507 | 0.6062 | 0.6316 | 0.5871 | 0.9065 | 0.8566 | 0.8705 | 0.8157 | 0.8913 | 0.3417 | 0.0521 |

	</details>

  <details>
  <summary> Eval Range: 50.0 - 90.0m </summary>

  | Model version | mAP | mAPH | map_based_nds (recall @ 0.10) | map_based_nds (recall @ 0.40) | maph_based_nds (recall @ 0.10) | maph_based_nds (recall 0.40) | car<br>(105,914) | truck<br>(28,864) | bus<br>(5,290) | bicycle<br>(3,608) | pedestrian<br>(48,637) | traffic_cone<br>(9,819) | barrier<br>(2,469) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.8.0 | 0.5447 | 0.5067 | 0.5590 | 0.5236 | 0.5400 | 0.5046 | 0.8132 | 0.6652 | 0.6404 | 0.6241 | 0.7502 | 0.3184 | 0.0017 |

  </details>

  <details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version | mAP | mAPH | map_based_nds (recall @ 0.10) | map_based_nds (recall @ 0.40) | maph_based_nds (recall @ 0.10) | maph_based_nds (recall 0.40) | car<br>(39,577) | truck<br>(18,213) | bus<br>(3,541) | bicycle<br>(942) | pedestrian<br>(20,134) | traffic_cone<br>(1,231) | barrier<br>(711) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.8.0 | 0.3779 | 0.3496 | 0.4428 | 0.3903 | 0.4287 | 0.3762 | 0.6979 | 0.5143 | 0.3860 | 0.3610 | 0.6588 | 0.0272 | 0.0002 |

  </details>

  <details open>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version | mAP | mAPH | map_based_nds (recall @ 0.10) | map_based_nds (recall @ 0.40) | maph_based_nds (recall @ 0.10) | maph_based_nds (recall 0.40) | car<br>(271,659) | truck<br>(73,974) | bus<br>(15,390) | bicycle<br>(10,415) | pedestrian<br>(162,291) | traffic_cone<br>(31,885) | barrier<br>(6,539) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.8.0 | 0.5936 | 0.5554 | 0.6017 | 0.5555 | 0.5826 | 0.5365 | 0.8534 | 0.7110 | 0.6992 | 0.7185 | 0.8315 | 0.3204 | 0.0209 |

  </details>

### Mean TPError - Base
- Recalls: `0.10`, `0.40`, `optimal`

  <details>
    <summary> Eval Range: 0.0 - 50.0m </summary>

    | Model version | mATE (recall @ 0.10) | mAOE (recall @ 0.10) | mASE (recall @ 0.10) | mAVE (recall @ 0.10) | mAAE (recall @ 0.10) | mATE (recall @ 0.40) | mAOE (recall @ 0.40) | mASE (recall @ 0.40) | mAVE (recall @ 0.40) | mAAE (recall @ 0.40) | mATE (optimal) | mAOE (optimal) | mASE (optimal) | mAVE (optimal) | mAAE (optimal) |
    | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | BEVFusion-LiDAR base/2.8.0 | 0.1796 | 0.1993 | 0.2024 | 0.2937 | 1.0000 | 0.2857 | 0.2916 | 0.2957 | 0.4466 | 1.0000 | 0.2149 | 0.2196 | 0.2175 | 0.3260 | 1.0000 |

    <summary><strong>Num match summary</strong></summary>

    **recall 0.10**

    | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 126,168) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 26,897) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 6,559) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 5,865) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 93,520) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 20,835) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 3,359) |
    | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
    | BEVFusion-LiDAR base/2.8.0 | 13,878 / 13,878 / 13,878 / 13,878 | 2,958 / 2,958 / 2,958 / 2,958 | 721 / 721 / 721 / 721 | 645 / 645 / 645 / 645 | 10,287 / 10,287 / 10,287 / 10,287 | 2,291 / 2,291 / 2,291 / 2,291 | 369 / 369 / 369 / 369 |

    **recall 0.40**

    | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 126,168) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 26,897) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 6,559) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 5,865) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 93,520) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 20,835) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 3,359) |
    | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
    | BEVFusion-LiDAR base/2.8.0 | 51,728 / 51,728 / 51,728 / 51,728 | 11,027 / 11,027 / 11,027 / 11,027 | 2,689 / 2,689 / 2,689 / 2,689 | 2,404 / 2,404 / 2,404 / 2,404 | 38,343 / 38,343 / 38,343 / 38,343 | 8,542 / 8,542 / 8,542 / 8,542 | 0 / 0 / 0 / 0 |

    **optimal**

    | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 126,168) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 26,897) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 6,559) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 5,865) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 93,520) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 20,835) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 3,359) |
    | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
    | BEVFusion-LiDAR base/2.8.0 | 109,035 / 113,613 / 115,502 / 116,446 | 20,578 / 23,058 / 24,234 / 24,631 | 5,398 / 5,809 / 5,952 / 5,967 | 4,573 / 4,739 / 4,611 / 4,616 | 78,245 / 79,717 / 80,219 / 80,761 | 10,168 / 11,012 / 11,411 / 11,947 | 452 / 629 / 667 / 715 |

  </details>

  <details>

    <summary> Eval Range: 50.0 - 90.0m </summary>

    | Model version | mATE (recall @ 0.10) | mAOE (recall @ 0.10) | mASE (recall @ 0.10) | mAVE (recall @ 0.10) | mAAE (recall @ 0.10) | mATE (recall @ 0.40) | mAOE (recall @ 0.40) | mASE (recall @ 0.40) | mAVE (recall @ 0.40) | mAAE (recall @ 0.40) | mATE (optimal) | mAOE (optimal) | mASE (optimal) | mAVE (optimal) | mAAE (optimal) |
    | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | BEVFusion-LiDAR base/2.8.0 | 0.2878 | 0.2286 | 0.2487 | 0.3686 | 1.0000 | 0.3596 | 0.3138 | 0.3151 | 0.4991 | 1.0000 | 0.2897 | 0.2224 | 0.2274 | 0.3779 | 1.0000 |

    <summary><strong>Num match summary</strong></summary>

    **recall 0.10**

    | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 105,914) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 28,864) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 5,290) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 3,608) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 48,637) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 9,819) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 2,469) |
    | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
    | BEVFusion-LiDAR base/2.8.0 | 11,650 / 11,650 / 11,650 / 11,650 | 3,175 / 3,175 / 3,175 / 3,175 | 581 / 581 / 581 / 581 | 396 / 396 / 396 / 396 | 5,350 / 5,350 / 5,350 / 5,350 | 1,080 / 1,080 / 1,080 / 1,080 | 0 / 271 / 271 / 271 |

    **recall 0.40**

    | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 105,914) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 28,864) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 5,290) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 3,608) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 48,637) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 9,819) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 2,469) |
    | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
    | BEVFusion-LiDAR base/2.8.0 | 43,424 / 43,424 / 43,424 / 43,424 | 11,834 / 11,834 / 11,834 / 11,834 | 2,168 / 2,168 / 2,168 / 2,168 | 1,479 / 1,479 / 1,479 / 1,479 | 19,941 / 19,941 / 19,941 / 19,941 | 4,025 / 4,025 / 4,025 / 4,025 | 0 / 0 / 0 / 0 |

    **optimal**

    | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 105,914) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 28,864) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 5,290) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 3,608) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 48,637) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 9,819) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 2,469) |
    | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
    | BEVFusion-LiDAR base/2.8.0 | 76,795 / 84,883 / 88,518 / 89,420 | 15,896 / 19,085 / 21,454 / 22,235 | 2,428 / 3,407 / 3,812 / 3,890 | 2,186 / 2,306 / 2,311 / 2,327 | 34,885 / 35,834 / 35,898 / 36,174 | 4,370 / 4,522 / 4,871 / 4,902 | 140 / 222 / 238 / 244 |

  </details>

  <details>

    <summary> Eval Range: 90.0 - 121.0m </summary>

    | Model version | mATE (recall @ 0.10) | mAOE (recall @ 0.10) | mASE (recall @ 0.10) | mAVE (recall @ 0.10) | mAAE (recall @ 0.10) | mATE (recall @ 0.40) | mAOE (recall @ 0.40) | mASE (recall @ 0.40) | mAVE (recall @ 0.40) | mAAE (recall @ 0.40) | mATE (optimal) | mAOE (optimal) | mASE (optimal) | mAVE (optimal) | mAAE (optimal) |
    | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | BEVFusion-LiDAR base/2.8.0 | 0.3878 | 0.2914 | 0.3071 | 0.4752 | 1.0000 | 0.5015 | 0.3966 | 0.4158 | 0.6724 | 1.0000 | 0.3567 | 0.2630 | 0.2646 | 0.4419 | 1.0000 |

    <summary><strong>Num match summary</strong></summary>

    **recall 0.10**

    | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 39,577) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 18,213) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 3,541) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 942) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 20,134) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 1,231) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 711) |
    | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
    | BEVFusion-LiDAR base/2.8.0 | 4,353 / 4,353 / 4,353 / 4,353 | 2,003 / 2,003 / 2,003 / 2,003 | 389 / 389 / 389 / 389 | 103 / 103 / 103 / 103 | 2,214 / 2,214 / 2,214 / 2,214 | 135 / 135 / 135 / 135 | 0 / 0 / 78 / 78 |

    **recall 0.40**

    | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 39,577) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 18,213) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 3,541) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 942) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 20,134) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 1,231) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 711) |
    | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
    | BEVFusion-LiDAR base/2.8.0 | 16,226 / 16,226 / 16,226 / 16,226 | 7,467 / 7,467 / 7,467 / 7,467 | 1,451 / 1,451 / 1,451 / 1,451 | 386 / 386 / 386 / 386 | 8,254 / 8,254 / 8,254 / 8,254 | 0 / 0 / 0 / 504 | 0 / 0 / 0 / 0 |

    **optimal**

    | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 39,577) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 18,213) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 3,541) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 942) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 20,134) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 1,231) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 711) |
    | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
    | BEVFusion-LiDAR base/2.8.0 | 23,637 / 27,796 / 30,068 / 31,051 | 6,996 / 9,963 / 12,146 / 13,138 | 1,277 / 1,704 / 1,857 / 1,963 | 379 / 421 / 434 / 439 | 13,121 / 13,270 / 13,350 / 13,493 | 247 / 251 / 262 / 280 | 28 / 54 / 67 / 70 |

  </details>

  <details open>

    <summary> Eval Range: 0.0 - 121.0m </summary>

    | Model version | mATE (recall @ 0.10) | mAOE (recall @ 0.10) | mASE (recall @ 0.10) | mAVE (recall @ 0.10) | mAAE (recall @ 0.10) | mATE (recall @ 0.40) | mAOE (recall @ 0.40) | mASE (recall @ 0.40) | mAVE (recall @ 0.40) | mAAE (recall @ 0.40) | mATE (optimal) | mAOE (optimal) | mASE (optimal) | mAVE (optimal) | mAAE (optimal) |
    | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | BEVFusion-LiDAR base/2.8.0 | 0.2162 | 0.2040 | 0.2147 | 0.3160 | 1.0000 | 0.3255 | 0.3034 | 0.3079 | 0.4756 | 1.0000 | 0.2567 | 0.2269 | 0.2265 | 0.3571 | 1.0000 |

    <summary><strong>Num match summary</strong></summary>

    **recall 0.10**

    | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 271,659) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 73,974) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 15,390) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 10,415) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 162,291) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 31,885) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 6,539) |
    | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
    | BEVFusion-LiDAR base/2.8.0 | 29,883 / 29,882 / 29,882 / 29,882 | 8,137 / 8,137 / 8,137 / 8,137 | 1,692 / 1,692 / 1,692 / 1,692 | 1,145 / 1,145 / 1,145 / 1,145 | 17,852 / 17,852 / 17,852 / 17,852 | 3,507 / 3,507 / 3,507 / 3,507 | 719 / 719 / 719 / 719 |

    **recall 0.40**

    | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 271,659) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 73,974) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 15,390) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 10,415) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 162,291) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 31,885) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 6,539) |
    | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
    | BEVFusion-LiDAR base/2.8.0 | 111,380 / 111,380 / 111,380 / 111,380 | 30,329 / 30,329 / 30,329 / 30,329 | 6,309 / 6,309 / 6,309 / 6,309 | 4,270 / 4,270 / 4,270 / 4,270 | 66,539 / 66,539 / 66,539 / 66,539 | 13,072 / 13,072 / 13,072 / 13,072 | 0 / 0 / 0 / 0 |

    **optimal**

    | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 271,659) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 73,974) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 15,390) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 10,415) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 162,291) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 31,885) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 6,539) |
    | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
    | BEVFusion-LiDAR base/2.8.0 | 209,064 / 225,972 / 233,684 / 235,583 | 43,131 / 52,358 / 57,620 / 60,504 | 9,224 / 10,854 / 11,562 / 11,707 | 7,044 / 7,224 / 7,242 / 7,260 | 125,762 / 128,271 / 129,072 / 130,130 | 14,442 / 15,921 / 16,500 / 17,198 | 624 / 886 / 950 / 1,028 |

  </details>


## Datasets

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
  | BEVFusion-LiDAR base/2.8.0 | 0.7246 | 0.6765 | 0.6874 | 0.6712 | 0.6633 | 0.6471 | 0.8849 | 0.8325 | 0.9034 | 0.9004 | 0.8381 | 0.4459 | 0.2671 |

  </details>

  <details>
  <summary> Eval Range: 50.0 - 90.0m </summary>

  | Model version | mAP | mAPH | map_based_nds (recall @ 0.10) | map_based_nds (recall @ 0.40) | maph_based_nds (recall @ 0.10) | maph_based_nds (recall 0.40) | car<br>(54,217) | truck<br>(4,913) | bus<br>(2,116) | bicycle<br>(838) | pedestrian<br>(8,336) | traffic_cone<br>(2,632) | barrier<br>(622) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.8.0 | 0.5590 | 0.5053 | 0.5849 | 0.5656 | 0.5581 | 0.5387 | 0.7864 | 0.6212 | 0.7611 | 0.6674 | 0.6253 | 0.2711 | 0.1807 |

  </details>

  <details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version | mAP | mAPH | map_based_nds (recall @ 0.10) | map_based_nds (recall @ 0.40) | maph_based_nds (recall @ 0.10) | maph_based_nds (recall 0.40) | car<br>(19,301) | truck<br>(2,906) | bus<br>(484) | bicycle<br>(291) | pedestrian<br>(2,564) | traffic_cone<br>(462) | barrier<br>(145) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.8.0 | 0.4021 | 0.3638 | 0.4870 | 0.4675 | 0.4679 | 0.4484 | 0.6848 | 0.4894 | 0.4972 | 0.4913 | 0.4232 | 0.1266 | 0.1024 |

  </details>

  <details open>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version | mAP | mAPH | map_based_nds (recall @ 0.10) | map_based_nds (recall @ 0.40) | maph_based_nds (recall @ 0.10) | maph_based_nds (recall 0.40) | car<br>(134,456) | truck<br>(14,900) | bus<br>(4,970) | bicycle<br>(2,486) | pedestrian<br>(29,102) | traffic_cone<br>(11,344) | barrier<br>(2,117) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.8.0 | 0.6463 | 0.5953 | 0.6403 | 0.6221 | 0.6148 | 0.5966 | 0.8310 | 0.7078 | 0.8174 | 0.7884 | 0.7558 | 0.3971 | 0.2263 |

  </details>

- **Mean TPError**

  <details>
  <summary> Eval Range: 0.0 - 50.0m </summary>

  | Model version | mATE (recall @ 0.10) | mAOE (recall @ 0.10) | mASE (recall @ 0.10) | mAVE (recall @ 0.10) | mAAE (recall @ 0.10) | mATE (recall @ 0.40) | mAOE (recall @ 0.40) | mASE (recall @ 0.40) | mAVE (recall @ 0.40) | mAAE (recall @ 0.40) | mATE (optimal) | mAOE (optimal) | mASE (optimal) | mAVE (optimal) | mAAE (optimal) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.8.0 | 0.1731 | 0.1809 | 0.1966 | 0.1987 | 1.0000 | 0.2178 | 0.2153 | 0.2319 | 0.2464 | 1.0000 | 0.2080 | 0.2074 | 0.2153 | 0.2185 | 1.0000 |

  <summary><strong>Num match summary</strong></summary>

  **recall 0.10**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 60,938) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 7,081) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 2,370) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 1,357) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 18,202) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 8,250) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 1,350) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 6,703 / 6,703 / 6,703 / 6,703 | 778 / 778 / 778 / 778 | 261 / 261 / 260 / 260 | 149 / 149 / 149 / 149 | 2,002 / 2,002 / 2,002 / 2,002 | 907 / 907 / 907 / 907 | 148 / 148 / 148 / 148 |

  **recall 0.40**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 60,938) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 7,081) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 2,370) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 1,357) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 18,202) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 8,250) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 1,350) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 24,984 / 24,984 / 24,984 / 24,984 | 2,903 / 2,903 / 2,903 / 2,903 | 971 / 971 / 971 / 971 | 556 / 556 / 556 / 556 | 7,462 / 7,462 / 7,462 / 7,462 | 3,382 / 3,382 / 3,382 / 3,382 | 0 / 553 / 553 / 553 |

  **optimal**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 60,938) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 7,081) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 2,370) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 1,357) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 18,202) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 8,250) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 1,350) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 51,545 / 53,474 / 54,439 / 55,024 | 5,252 / 5,935 / 6,142 / 6,261 | 1,963 / 2,101 / 2,217 / 2,227 | 1,141 / 1,146 / 1,146 / 1,146 | 14,108 / 14,410 / 14,530 / 14,796 | 4,463 / 4,943 / 5,097 / 5,196 | 452 / 587 / 622 / 638 |

  </details>

  <details>
  <summary> Eval Range: 50.0 - 90.0m </summary>

  | Model version | mATE (recall @ 0.10) | mAOE (recall @ 0.10) | mASE (recall @ 0.10) | mAVE (recall @ 0.10) | mAAE (recall @ 0.10) | mATE (recall @ 0.40) | mAOE (recall @ 0.40) | mASE (recall @ 0.40) | mAVE (recall @ 0.40) | mAAE (recall @ 0.40) | mATE (optimal) | mAOE (optimal) | mASE (optimal) | mAVE (optimal) | mAAE (optimal) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.8.0 | 0.2509 | 0.2223 | 0.2170 | 0.2557 | 1.0000 | 0.3081 | 0.2757 | 0.2529 | 0.3025 | 1.0000 | 0.2876 | 0.2507 | 0.2251 | 0.2699 | 1.0000 |

  <summary><strong>Num match summary</strong></summary>

  **recall 0.10**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 54,217) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 4,913) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 2,116) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 838) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 8,336) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 2,632) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 622) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 5,963 / 5,963 / 5,963 / 5,963 | 540 / 540 / 540 / 540 | 232 / 232 / 232 / 232 | 92 / 92 / 92 / 92 | 916 / 916 / 916 / 916 | 289 / 289 / 289 / 289 | 68 / 68 / 68 / 68 |

  **recall 0.40**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 54,217) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 4,913) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 2,116) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 838) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 8,336) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 2,632) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 622) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 22,228 / 22,228 / 22,228 / 22,228 | 2,014 / 2,014 / 2,014 / 2,014 | 867 / 867 / 867 / 867 | 343 / 343 / 343 / 343 | 3,417 / 3,417 / 3,417 / 3,417 | 1,079 / 1,079 / 1,079 / 1,079 | 0 / 255 / 255 / 255 |

  **optimal**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 54,217) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 4,913) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 2,116) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 838) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 8,336) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 2,632) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 622) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 37,866 / 42,472 / 44,630 / 45,417 | 2,598 / 3,145 / 3,407 / 3,593 | 1,175 / 1,604 / 1,740 / 1,815 | 524 / 551 / 552 / 557 | 5,196 / 5,378 / 5,448 / 5,510 | 1,014 / 1,077 / 1,237 / 1,290 | 136 / 209 / 227 / 233 |

  </details>

  <details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version | mATE (recall @ 0.10) | mAOE (recall @ 0.10) | mASE (recall @ 0.10) | mAVE (recall @ 0.10) | mAAE (recall @ 0.10) | mATE (recall @ 0.40) | mAOE (recall @ 0.40) | mASE (recall @ 0.40) | mAVE (recall @ 0.40) | mAAE (recall @ 0.40) | mATE (optimal) | mAOE (optimal) | mASE (optimal) | mAVE (optimal) | mAAE (optimal) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.8.0 | 0.3294 | 0.2250 | 0.2534 | 0.3325 | 1.0000 | 0.3858 | 0.2797 | 0.2836 | 0.3859 | 1.0000 | 0.3505 | 0.2499 | 0.2570 | 0.3449 | 1.0000 |

  <summary><strong>Num match summary</strong></summary>

  **recall 0.10**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 19,301) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 2,906) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 484) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 291) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 2,564) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 462) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 145) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 2,123 / 2,123 / 2,123 / 2,123 | 319 / 319 / 319 / 319 | 53 / 53 / 53 / 53 | 32 / 32 / 32 / 32 | 282 / 282 / 282 / 282 | 50 / 50 / 50 / 50 | 15 / 15 / 15 / 15 |

  **recall 0.40**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 19,301) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 2,906) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 484) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 291) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 2,564) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 462) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 145) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 7,913 / 7,913 / 7,913 / 7,913 | 1,191 / 1,191 / 1,191 / 1,191 | 198 / 198 / 198 / 198 | 119 / 119 / 119 / 119 | 1,051 / 1,051 / 1,051 / 1,051 | 189 / 189 / 189 / 189 | 0 / 59 / 59 / 59 |

  **optimal**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 19,301) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 2,906) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 484) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 291) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 2,564) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 462) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 145) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 11,411 / 13,677 / 14,775 / 15,007 | 1,065 / 1,541 / 1,834 / 2,047 | 164 / 233 / 291 / 296 | 147 / 164 / 164 / 164 | 1,358 / 1,329 / 1,371 / 1,342 | 135 / 135 / 133 / 148 | 26 / 40 / 50 / 53 |

  </details>

  <details>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version | mATE (recall @ 0.10) | mAOE (recall @ 0.10) | mASE (recall @ 0.10) | mAVE (recall @ 0.10) | mAAE (recall @ 0.10) | mATE (recall @ 0.40) | mAOE (recall @ 0.40) | mASE (recall @ 0.40) | mAVE (recall @ 0.40) | mAAE (recall @ 0.40) | mATE (optimal) | mAOE (optimal) | mASE (optimal) | mAVE (optimal) | mAAE (optimal) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.8.0 | 0.2028 | 0.1964 | 0.2072 | 0.2220 | 1.0000 | 0.2571 | 0.2377 | 0.2438 | 0.2713 | 1.0000 | 0.2468 | 0.2298 | 0.2238 | 0.2435 | 1.0000 |

  <summary><strong>Num match summary</strong></summary>

  **recall 0.10**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 134,456) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 14,900) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 4,970) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 2,486) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 29,102) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 11,344) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 2,117) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 14,790 / 14,790 / 14,790 / 14,790 | 1,639 / 1,639 / 1,639 / 1,639 | 546 / 546 / 546 / 546 | 273 / 273 / 273 / 273 | 3,201 / 3,201 / 3,201 / 3,201 | 1,247 / 1,247 / 1,247 / 1,247 | 232 / 232 / 232 / 232 |

  **recall 0.40**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 134,456) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 14,900) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 4,970) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 2,486) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 29,102) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 11,344) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 2,117) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 55,126 / 55,126 / 55,126 / 55,126 | 6,109 / 6,109 / 6,109 / 6,109 | 2,037 / 2,037 / 2,037 / 2,037 | 1,019 / 1,019 / 1,019 / 1,019 | 11,931 / 11,931 / 11,931 / 11,931 | 4,651 / 4,651 / 4,651 / 4,651 | 0 / 867 / 867 / 867 |

  **optimal**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 134,456) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 14,900) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 4,970) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 2,486) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 29,102) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 11,344) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 2,117) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 100,294 / 109,159 / 113,989 / 115,141 | 8,931 / 10,558 / 11,357 / 11,896 | 3,256 / 3,944 / 4,259 / 4,307 | 1,785 / 1,879 / 1,880 / 1,887 | 20,949 / 21,293 / 21,452 / 21,637 | 5,511 / 6,135 / 6,340 / 6,540 | 605 / 817 / 935 / 960 |

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

  | Model version | mAP | mAPH | map_based_nds (recall @ 0.10) | map_based_nds (recall @ 0.40) | maph_based_nds (recall @ 0.10) | maph_based_nds (recall 0.40) | car<br>(14,872) | truck<br>(1,192) | bus<br>(336) | bicycle<br>(740) | pedestrian<br>(5,055) | traffic_cone<br>(60) | barrier<br>(0) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.8.0 | 0.6292 | 0.5987 | 0.5796 | 0.5491 | 0.5644 | 0.5339 | 0.9088 | 0.8625 | 0.9253 | 0.8660 | 0.8414 | 0.0000 | 0.0000 |

  </details>

  <details>
  <summary> Eval Range: 50.0 - 90.0m </summary>

  | Model version | mAP | mAPH | map_based_nds (recall @ 0.10) | map_based_nds (recall @ 0.40) | maph_based_nds (recall @ 0.10) | maph_based_nds (recall 0.40) | car<br>(10,929) | truck<br>(1,009) | bus<br>(141) | bicycle<br>(460) | pedestrian<br>(3,721) | traffic_cone<br>(4) | barrier<br>(0) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.8.0 | 0.5080 | 0.4699 | 0.4842 | 0.4820 | 0.4652 | 0.4630 | 0.8284 | 0.6953 | 0.8101 | 0.5551 | 0.6672 | 0.0000 | 0.0000 |

  </details>

  <details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version | mAP | mAPH | map_based_nds (recall @ 0.10) | map_based_nds (recall @ 0.40) | maph_based_nds (recall @ 0.10) | maph_based_nds (recall 0.40) | car<br>(2,883) | truck<br>(600) | bus<br>(60) | bicycle<br>(85) | pedestrian<br>(1,092) | traffic_cone<br>(0) | barrier<br>(0) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.8.0 | 0.3869 | 0.3535 | 0.4036 | 0.3922 | 0.3870 | 0.3755 | 0.7338 | 0.6045 | 0.5314 | 0.3490 | 0.4896 | 0.0000 | 0.0000 |

  </details>

  <details open>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version | mAP | mAPH | map_based_nds (recall @ 0.10) | map_based_nds (recall @ 0.40) | maph_based_nds (recall @ 0.10) | maph_based_nds (recall 0.40) | car<br>(28,684) | truck<br>(2,801) | bus<br>(537) | bicycle<br>(1,285) | pedestrian<br>(9,868) | traffic_cone<br>(64) | barrier<br>(0) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.8.0 | 0.5663 | 0.5318 | 0.5398 | 0.5103 | 0.5226 | 0.4931 | 0.8718 | 0.7543 | 0.8572 | 0.7306 | 0.7502 | 0.0000 | 0.0000 |

  </details>

- **Mean TPError**

  <details>
  <summary> Eval Range: 0.0 - 50.0m </summary>

  | Model version | mATE (recall @ 0.10) | mAOE (recall @ 0.10) | mASE (recall @ 0.10) | mAVE (recall @ 0.10) | mAAE (recall @ 0.10) | mATE (recall @ 0.40) | mAOE (recall @ 0.40) | mASE (recall @ 0.40) | mAVE (recall @ 0.40) | mAAE (recall @ 0.40) | mATE (optimal) | mAOE (optimal) | mASE (optimal) | mAVE (optimal) | mAAE (optimal) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.8.0 | 0.2738 | 0.4304 | 0.3040 | 0.3416 | 1.0000 | 0.3927 | 0.3902 | 0.3987 | 0.4730 | 1.0000 | 0.1903 | 0.3709 | 0.2019 | 0.2298 | 1.0000 |

  <summary><strong>Num match summary</strong></summary>

  **recall 0.10**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 14,872) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 1,192) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 336) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 740) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 5,055) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 60) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 0) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 1,635 / 1,635 / 1,635 / 1,635 | 131 / 131 / 131 / 131 | 36 / 36 / 36 / 36 | 81 / 81 / 81 / 81 | 556 / 556 / 556 / 556 | 6 / 6 / 6 / 6 | 0 / 0 / 0 / 0 |

  **recall 0.40**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 14,872) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 1,192) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 336) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 740) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 5,055) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 60) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 0) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 6,097 / 6,097 / 6,097 / 6,097 | 488 / 488 / 488 / 488 | 137 / 137 / 137 / 137 | 303 / 303 / 303 / 303 | 2,072 / 2,072 / 2,072 / 2,072 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

  **optimal**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 14,872) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 1,192) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 336) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 740) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 5,055) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 60) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 0) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 13,062 / 13,479 / 13,554 / 13,743 | 932 / 1,043 / 1,066 / 1,074 | 275 / 321 / 324 / 324 | 602 / 607 / 608 / 613 | 4,140 / 4,178 / 4,198 / 4,214 | 23 / 13 / 13 / 13 | 0 / 0 / 0 / 0 |

  </details>

  <details>
  <summary> Eval Range: 50.0 - 90.0m </summary>

  | Model version | mATE (recall @ 0.10) | mAOE (recall @ 0.10) | mASE (recall @ 0.10) | mAVE (recall @ 0.10) | mAAE (recall @ 0.10) | mATE (recall @ 0.40) | mAOE (recall @ 0.40) | mASE (recall @ 0.40) | mAVE (recall @ 0.40) | mAAE (recall @ 0.40) | mATE (optimal) | mAOE (optimal) | mASE (optimal) | mAVE (optimal) | mAAE (optimal) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.8.0 | 0.3215 | 0.6553 | 0.3435 | 0.3779 | 1.0000 | 0.3296 | 0.6554 | 0.3494 | 0.3854 | 1.0000 | 0.2287 | 0.6009 | 0.2438 | 0.2891 | 1.0000 |

  <summary><strong>Num match summary</strong></summary>

  **recall 0.10**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 10,929) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 1,009) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 141) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 460) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 3,721) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 4) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 0) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 1,202 / 1,202 / 1,202 / 1,202 | 110 / 110 / 110 / 110 | 15 / 15 / 15 / 15 | 50 / 50 / 50 / 50 | 409 / 409 / 409 / 409 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

  **recall 0.40**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 10,929) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 1,009) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 141) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 460) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 3,721) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 4) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 0) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 4,480 / 4,480 / 4,480 / 4,480 | 413 / 413 / 413 / 413 | 57 / 57 / 57 / 57 | 188 / 188 / 188 / 188 | 1,525 / 1,525 / 1,525 / 1,525 | 1 / 1 / 1 / 1 | 0 / 0 / 0 / 0 |

  **optimal**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 10,929) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 1,009) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 141) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 460) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 3,721) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 4) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 0) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 8,159 / 8,918 / 9,145 / 9,193 | 584 / 735 / 782 / 787 | 97 / 115 / 115 / 115 | 243 / 263 / 265 / 265 | 2,464 / 2,492 / 2,508 / 2,524 | 2 / 2 / 2 / 2 | 0 / 0 / 0 / 0 |

  </details>

  <details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version | mATE (recall @ 0.10) | mAOE (recall @ 0.10) | mASE (recall @ 0.10) | mAVE (recall @ 0.10) | mAAE (recall @ 0.10) | mATE (recall @ 0.40) | mAOE (recall @ 0.40) | mASE (recall @ 0.40) | mAVE (recall @ 0.40) | mAAE (recall @ 0.40) | mATE (optimal) | mAOE (optimal) | mASE (optimal) | mAVE (optimal) | mAAE (optimal) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.8.0 | 0.4740 | 0.4126 | 0.4223 | 0.5892 | 1.0000 | 0.4917 | 0.4497 | 0.4298 | 0.6415 | 1.0000 | 0.2930 | 0.2332 | 0.2012 | 0.4842 | 1.0000 |

  <summary><strong>Num match summary</strong></summary>

  **recall 0.10**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 2,883) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 600) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 60) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 85) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 1,092) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 0) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 0) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 317 / 317 / 317 / 317 | 66 / 66 / 66 / 66 | 6 / 6 / 6 / 6 | 9 / 9 / 9 / 9 | 120 / 120 / 120 / 120 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

  **recall 0.40**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 2,883) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 600) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 60) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 85) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 1,092) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 0) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 0) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 1,182 / 1,182 / 1,182 / 1,182 | 246 / 246 / 246 / 246 | 24 / 24 / 24 / 24 | 34 / 34 / 34 / 34 | 447 / 447 / 447 / 447 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

  **optimal**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 2,883) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 600) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 60) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 85) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 1,092) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 0) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 0) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 1,763 / 2,103 / 2,227 / 2,243 | 254 / 358 / 448 / 456 | 24 / 38 / 31 / 31 | 38 / 44 / 46 / 46 | 613 / 619 / 623 / 626 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
  </details>

  <details open>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version | mATE (recall @ 0.10) | mAOE (recall @ 0.10) | mASE (recall @ 0.10) | mAVE (recall @ 0.10) | mAAE (recall @ 0.10) | mATE (recall @ 0.40) | mAOE (recall @ 0.40) | mASE (recall @ 0.40) | mAVE (recall @ 0.40) | mAAE (recall @ 0.40) | mATE (optimal) | mAOE (optimal) | mASE (optimal) | mAVE (optimal) | mAAE (optimal) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.8.0 | 0.2926 | 0.4708 | 0.3148 | 0.3550 | 1.0000 | 0.4159 | 0.4150 | 0.4074 | 0.4899 | 1.0000 | 0.2138 | 0.4027 | 0.2131 | 0.2586 | 1.0000 |

  <summary><strong>Num match summary</strong></summary>

  **recall 0.10**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 28,684) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 2,801) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 537) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 1,285) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 9,868) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 64) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 0) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 3,155 / 3,155 / 3,155 / 3,155 | 308 / 308 / 308 / 308 | 59 / 59 / 59 / 59 | 141 / 141 / 141 / 141 | 1,085 / 1,085 / 1,085 / 1,085 | 7 / 7 / 7 / 7 | 0 / 0 / 0 / 0 |

  **recall 0.40**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 28,684) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 2,801) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 537) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 1,285) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 9,868) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 64) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 0) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 11,760 / 11,760 / 11,760 / 11,760 | 1,148 / 1,148 / 1,148 / 1,148 | 220 / 220 / 220 / 220 | 526 / 526 / 526 / 526 | 4,045 / 4,045 / 4,045 / 4,045 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

  **optimal**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 28,684) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 2,801) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 537) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 1,285) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 9,868) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 64) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 0) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 23,001 / 24,465 / 24,972 / 25,071 | 1,757 / 2,148 / 2,295 / 2,313 | 391 / 442 / 445 / 445 | 839 / 875 / 938 / 944 | 7,078 / 7,166 / 7,204 / 7,247 | 13 / 15 / 15 / 15 | 0 / 0 / 0 / 0 |
  </details>

</details>

<details>
<summary> JPNTaxi Gen2 </summary>

- Datasets (10,687 Testing Frames):
  - `db_jpntaxigen2_v1`
  - `db_jpntaxigen2_v2`

- **Class mAP for BEV Center Distance: 0.5m, 1.0m, 2.0m, 4.0m**

  <details>
  <summary> Eval Range: 0.0 - 50.0m </summary>

  | Model version | mAP | mAPH | map_based_nds (recall @ 0.10) | map_based_nds (recall @ 0.40) | maph_based_nds (recall @ 0.10) | maph_based_nds (recall 0.40) | car<br>(50,954) | truck<br>(18,624) | bus<br>(3,853) | bicycle<br>(3,768) | pedestrian<br>(70,699) | traffic_cone<br>(12,525) | barrier<br>(2,009) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.8.0 | 0.6595 | 0.6258 | 0.5925 | 0.5885 | 0.5757 | 0.5717 | 0.9193 | 0.8663 | 0.8424 | 0.7784 | 0.9038 | 0.3064 | 0.0000 |

  </details>

  <details>
  <summary> Eval Range: 50.0 - 90.0m </summary>

  | Model version | mAP | mAPH | map_based_nds (recall @ 0.10) | map_based_nds (recall @ 0.40) | maph_based_nds (recall @ 0.10) | maph_based_nds (recall 0.40) | car<br>(41,196) | truck<br>(22,942) | bus<br>(3,033) | bicycle<br>(2,310) | pedestrian<br>(36,881) | traffic_cone<br>(7,183) | barrier<br>(1,847) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.8.0 | 0.5440 | 0.5111 | 0.5263 | 0.5188 | 0.5099 | 0.5024 | 0.8350 | 0.6741 | 0.5382 | 0.6234 | 0.7829 | 0.3548 | 0.0000 |

  </details>

  <details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version | mAP | mAPH | map_based_nds (recall @ 0.10) | map_based_nds (recall @ 0.40) | maph_based_nds (recall @ 0.10) | maph_based_nds (recall 0.40) | car<br>(17,510) | truck<br>(14,707) | bus<br>(2,997) | bicycle<br>(566) | pedestrian<br>(16,580) | traffic_cone<br>(769) | barrier<br>(566) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.8.0 | 0.3700 | 0.3469 | 0.4109 | 0.3757 | 0.3994 | 0.3641 | 0.7043 | 0.5157 | 0.3679 | 0.2959 | 0.7000 | 0.0063 | 0.0000 |

  </details>

  <details open>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version | mAP | mAPH | map_based_nds (recall @ 0.10) | map_based_nds (recall @ 0.40) | maph_based_nds (recall @ 0.10) | maph_based_nds (recall 0.40) | car<br>(109,660) | truck<br>(56,273) | bus<br>(9,883) | bicycle<br>(6,644) | pedestrian<br>(124,160) | traffic_cone<br>(20,477) | barrier<br>(4,422) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.8.0 | 0.5785 | 0.5444 | 0.5467 | 0.5405 | 0.5296 | 0.5234 | 0.8675 | 0.7091 | 0.6251 | 0.6924 | 0.8516 | 0.3040 | 0.0000 |

  </details>

- **Mean TPError**

  <details>
  <summary> Eval Range: 0.0 - 50.0m </summary>

  | Model version | mATE (recall @ 0.10) | mAOE (recall @ 0.10) | mASE (recall @ 0.10) | mAVE (recall @ 0.10) | mAAE (recall @ 0.10) | mATE (recall @ 0.40) | mAOE (recall @ 0.40) | mASE (recall @ 0.40) | mAVE (recall @ 0.40) | mAAE (recall @ 0.40) | mATE (optimal) | mAOE (optimal) | mASE (optimal) | mAVE (optimal) | mAAE (optimal) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.8.0 | 0.2759 | 0.3051 | 0.2927 | 0.4984 | 1.0000 | 0.2896 | 0.3123 | 0.2968 | 0.5135 | 1.0000 | 0.1903 | 0.2037 | 0.1876 | 0.4654 | 1.0000 |

  <summary><strong>Num match summary</strong></summary>

  **recall 0.10**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 50,954) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 18,624) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 3,853) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 3,768) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 70,699) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 12,525) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 2,009) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 5,604 / 5,604 / 5,604 / 5,604 | 2,048 / 2,048 / 2,048 / 2,048 | 423 / 423 / 423 / 423 | 414 / 414 / 414 / 414 | 7,776 / 7,776 / 7,776 / 7,776 | 1,377 / 1,377 / 1,377 / 1,377 | 0 / 0 / 0 / 0 |

  **recall 0.40**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 50,954) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 18,624) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 3,853) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 3,768) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 70,699) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 12,525) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 2,009) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 20,891 / 20,891 / 20,891 / 20,891 | 7,635 / 7,635 / 7,635 / 7,635 | 1,579 / 1,579 / 1,579 / 1,579 | 1,544 / 1,544 / 1,544 / 1,544 | 28,986 / 28,986 / 28,986 / 28,986 | 5,135 / 5,135 / 5,135 / 5,135 | 0 / 0 / 0 / 0 |

  **optimal**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 50,954) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 18,624) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 3,853) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 3,768) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 70,699) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 12,525) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 2,009) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 44,376 / 46,850 / 47,216 / 47,606 | 14,516 / 16,104 / 17,017 / 17,294 | 3,080 / 3,354 / 3,439 / 3,443 | 2,874 / 2,897 / 2,900 / 2,900 | 59,982 / 60,714 / 61,271 / 61,531 | 5,720 / 6,079 / 6,376 / 6,515 | 0 / 0 / 0 / 0 |

  </details>

  <details>
  <summary> Eval Range: 50.0 - 90.0m </summary>

  | Model version | mATE (recall @ 0.10) | mAOE (recall @ 0.10) | mASE (recall @ 0.10) | mAVE (recall @ 0.10) | mAAE (recall @ 0.10) | mATE (recall @ 0.40) | mAOE (recall @ 0.40) | mASE (recall @ 0.40) | mAVE (recall @ 0.40) | mAAE (recall @ 0.40) | mATE (optimal) | mAOE (optimal) | mASE (optimal) | mAVE (optimal) | mAAE (optimal) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.8.0 | 0.3424 | 0.2851 | 0.3099 | 0.5197 | 1.0000 | 0.3636 | 0.2926 | 0.3147 | 0.5610 | 1.0000 | 0.2700 | 0.1826 | 0.2042 | 0.5180 | 1.0000 |

  <summary><strong>Num match summary</strong></summary>

  **recall 0.10**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 41,196) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 22,942) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 3,033) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 2,310) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 36,881) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 7,183) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 1,847) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 4,531 / 4,531 / 4,531 / 4,531 | 2,523 / 2,523 / 2,523 / 2,523 | 333 / 333 / 333 / 333 | 254 / 254 / 254 / 254 | 4,056 / 4,056 / 4,056 / 4,056 | 790 / 790 / 790 / 790 | 0 / 0 / 0 / 0 |

  **recall 0.40**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 41,196) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 22,942) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 3,033) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 2,310) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 36,881) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 7,183) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 1,847) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 16,890 / 16,890 / 16,890 / 16,890 | 9,406 / 9,406 / 9,406 / 9,406 | 1,243 / 1,243 / 1,243 / 1,243 | 947 / 947 / 947 / 947 | 15,121 / 15,121 / 15,121 / 15,121 | 2,945 / 2,945 / 2,945 / 2,945 | 0 / 0 / 0 / 0 |

  **optimal**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 41,196) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 22,942) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 3,033) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 2,310) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 36,881) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 7,183) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 1,847) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 30,777 / 33,945 / 34,775 / 34,956 | 12,711 / 15,140 / 17,099 / 18,052 | 1,191 / 1,652 / 1,886 / 1,924 | 1,384 / 1,483 / 1,484 / 1,496 | 27,185 / 28,060 / 28,214 / 28,437 | 3,298 / 3,377 / 3,633 / 3,756 | 0 / 0 / 0 / 0 |

  </details>

  <details>
  <summary> Eval Range: 90.0 - 121.0m </summary>

  | Model version | mATE (recall @ 0.10) | mAOE (recall @ 0.10) | mASE (recall @ 0.10) | mAVE (recall @ 0.10) | mAAE (recall @ 0.10) | mATE (recall @ 0.40) | mAOE (recall @ 0.40) | mASE (recall @ 0.40) | mAVE (recall @ 0.40) | mAAE (recall @ 0.40) | mATE (optimal) | mAOE (optimal) | mASE (optimal) | mAVE (optimal) | mAAE (optimal) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.8.0 | 0.4063 | 0.3864 | 0.3484 | 0.5995 | 1.0000 | 0.5077 | 0.3923 | 0.4395 | 0.7535 | 1.0000 | 0.3267 | 0.2998 | 0.2415 | 0.5890 | 1.0000 |

  <summary><strong>Num match summary</strong></summary>

  **recall 0.10**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 17,510) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 14,707) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 2,997) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 566) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 16,580) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 769) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 566) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 1,926 / 1,926 / 1,926 / 1,926 | 1,617 / 1,617 / 1,617 / 1,617 | 329 / 329 / 329 / 329 | 62 / 62 / 62 / 62 | 1,823 / 1,823 / 1,823 / 1,823 | 84 / 84 / 84 / 84 | 0 / 0 / 0 / 0 |

  **recall 0.40**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 17,510) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 14,707) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 2,997) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 566) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 16,580) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 769) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 566) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 7,179 / 7,179 / 7,179 / 7,179 | 6,029 / 6,029 / 6,029 / 6,029 | 1,228 / 1,228 / 1,228 / 1,228 | 232 / 232 / 232 / 232 | 6,797 / 6,797 / 6,797 / 6,797 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

  **optimal**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 17,510) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 14,707) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 2,997) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 566) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 16,580) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 769) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 566) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 10,800 / 12,644 / 13,151 / 13,260 | 5,665 / 8,075 / 9,757 / 10,776 | 1,062 / 1,422 / 1,542 / 1,587 | 221 / 245 / 257 / 261 | 11,283 / 11,409 / 11,480 / 11,627 | 167 / 177 / 127 / 133 | 0 / 0 / 0 / 0 |

  </details>

  <details open>
  <summary> Eval Range: 0.0 - 121.0m </summary>

  | Model version | mATE (recall @ 0.10) | mAOE (recall @ 0.10) | mASE (recall @ 0.10) | mAVE (recall @ 0.10) | mAAE (recall @ 0.10) | mATE (recall @ 0.40) | mAOE (recall @ 0.40) | mASE (recall @ 0.40) | mAVE (recall @ 0.40) | mAAE (recall @ 0.40) | mATE (optimal) | mAOE (optimal) | mASE (optimal) | mAVE (optimal) | mAAE (optimal) |
  | :---- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BEVFusion-LiDAR base/2.8.0 | 0.3080 | 0.2997 | 0.3015 | 0.5162 | 1.0000 | 0.3316 | 0.3074 | 0.3083 | 0.5404 | 1.0000 | 0.2360 | 0.2001 | 0.1988 | 0.4973 | 1.0000 |

  <summary><strong>Num match summary</strong></summary>

  **recall 0.10**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 109,660) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 56,273) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 9,883) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 6,644) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 124,160) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 20,477) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 4,422) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 12,062 / 12,062 / 12,062 / 12,062 | 6,190 / 6,190 / 6,190 / 6,190 | 1,087 / 1,087 / 1,087 / 1,087 | 730 / 730 / 730 / 730 | 13,657 / 13,657 / 13,657 / 13,657 | 2,252 / 2,252 / 2,252 / 2,252 | 0 / 0 / 0 / 0 |

  **recall 0.40**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 109,660) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 56,273) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 9,883) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 6,644) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 124,160) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 20,477) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 4,422) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 44,960 / 44,960 / 44,960 / 44,960 | 23,071 / 23,071 / 23,071 / 23,071 | 4,052 / 4,052 / 4,052 / 4,052 | 2,724 / 2,724 / 2,724 / 2,724 | 50,905 / 50,905 / 50,905 / 50,905 | 8,395 / 8,395 / 8,395 / 8,395 | 0 / 0 / 0 / 0 |

  **optimal**

  | Model version | car<br>0.5/1.0/2.0/4.0<br>(GTs: 109,660) | truck<br>0.5/1.0/2.0/4.0<br>(GTs: 56,273) | bus<br>0.5/1.0/2.0/4.0<br>(GTs: 9,883) | bicycle<br>0.5/1.0/2.0/4.0<br>(GTs: 6,644) | pedestrian<br>0.5/1.0/2.0/4.0<br>(GTs: 124,160) | traffic_cone<br>0.5/1.0/2.0/4.0<br>(GTs: 20,477) | barrier<br>0.5/1.0/2.0/4.0<br>(GTs: 4,422) |
  | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
  | BEVFusion-LiDAR base/2.8.0 | 85,327 / 93,303 / 94,826 / 95,316 | 32,549 / 39,457 / 43,979 / 46,340 | 5,296 / 6,439 / 6,895 / 6,991 | 4,446 / 4,536 / 4,537 / 4,547 | 98,679 / 99,889 / 100,492 / 101,325 | 9,207 / 9,578 / 10,010 / 10,328 | 0 / 0 / 0 / 0 |

  </details>

</details>

## Release

### BEVFusion-LiDAR base/2.8.0

<details>
<summary> Changes </summary>
- Add datasets: `db_j6gen2_v10`, `db_j6gen2_v11`, and `db_j6gen2_v12`.
- Update training batch size from `8` to `16` per gpu.
- Update number of max points per voxel from `10` to `32`.
- Implement 1D-flatten sparse to dense to reduce ONNX ops (projects/BEVFusion/bevfusion/custom_sparse_conv_tensor.py).
- Purse custom LayerNorm in mmdeploy to support ONNX LayerNorm ops (purge_mmdeploy_symbolics).
- Add two classes: `traffic_cone` and `barrier` to the model.
- Do not max-pooling of `bicycle`.
- Filter 3D bboxes in the train set by distance (< 60m, >= 3) and (60m <= x < 130m, >= 2).
- Filter 3D bboxes in the test set by number of points (>= 2).


</details>

<details>
<summary> Artifacts </summary>

- Deployed onnx and ROS parameter files (for internal)
  - [WebAuto](https://evaluation.ci.tier4.jp/evaluation/mlpackages/46f8188d-e3be-4f2f-b989-fd27002610d7/releases/6c8bc393-9cff-413c-bfc7-52c1fcd8ba8a?project_id=zWhWRzei)
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-l/t4base/v2.8.0/deployment.zip)
  - [Google drive](https://drive.google.com/file/d/16dh2UQg4w46WQu0Dbmai9BtD43nz7hLv/view?usp=drive_link)
- Logs (for internal)
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-l/t4base/v2.8.0/logs.zip)
  - [Google drive](https://drive.google.com/file/d/1kQVufXiB_K9JYTL3DSUhGUIGW5fCbbwc/view?usp=drive_link)
- Pytorch Best checkpoints:
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-l/t4base/v2.8.0/best_epoch_47.zip)
  - [Google drive](https://drive.google.com/file/d/1wJjdG1dCbOjfmTCaOFWw9-2xNvlfYDza/view?usp=drive_link)

</details>

<details>
<summary> Training configs </summary>

- [Config file path](https://github.com/KSeangTan/AWML/blob/179ca256a165fd483801bec0a2a95c24866edf70/projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_50e_8xb16_base_120m.py)
- Train time: NVIDIA H200 140GB * 8 * 50 epochs ~= 4 days
- Batch size: 8*16 = 128
- Training Dataset (frames: 151,478):
  - jpntaxi: db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 (28,161 frames)
  - j6: db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 (29,336 frames)
  - j6gen2: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 + db_j6gen2_v6 + db_j6gen2_v7 + db_j6gen2_v8 + db_j6gen2_v9 + db_j6gen2_v10 + db_j6gen2_v11 + db_j6gen2_v12 (51,208 frames)
  - largebus: db_largebus_v1 + db_largebus_v2 + db_largebus_v3 (12,605 frames)
  - jpntaxi_gen2: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (30,168 frames)

</details>

<details>
<summary> Evaluation </summary>

**Base Datasets (16,597 frames)**:

  - j6gen2 (4,682 frames): db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 + db_j6gen2_v6 + db_j6gen2_v7 + db_j6gen2_v8 + db_j6gen2_v9 + db_j6gen2_v10 + db_j6gen2_v11 + db_j6gen2_v12
  - largebus (1,228 frames): db_largebus_v1 + db_largebus_v2 + db_largebus_v3
  - jpntaxi_gen2 (10,687 frames): db_jpntaxigen2_v1 + db_jpntaxigen2_v2

**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.6763**

| class_name | GTs | num_match@0.5/1.0/2.0/4.0 | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | :---- | :---- | :---- | :---- |
| car | 126,168 | 112,018 / 117,488 / 119,486 / 120,557 | 0.852 / 0.913 / 0.925 / 0.936 | 0.900 / 0.932 / 0.939 / 0.942 | 0.249 / 0.213 / 0.168 / 0.148 |
| truck | 26,897 | 21,675 / 24,088 / 25,410 / 25,929 | 0.719 / 0.849 / 0.916 / 0.942 | 0.810 / 0.884 / 0.922 / 0.937 | 0.293 / 0.202 / 0.176 / 0.176 |
| bus | 6,559 | 5,520 / 5,973 / 6,143 / 6,163 | 0.789 / 0.876 / 0.908 / 0.909 | 0.828 / 0.887 / 0.909 / 0.911 | 0.047 / 0.044 / 0.044 / 0.044 |
| bicycle | 5,865 | 5,319 / 5,410 / 5,419 / 5,428 | 0.799 / 0.820 / 0.821 / 0.823 | 0.819 / 0.826 / 0.827 / 0.828 | 0.234 / 0.202 / 0.236 / 0.236 |
| pedestrian | 93,520 | 87,053 / 88,369 / 88,873 / 89,317 | 0.873 / 0.889 / 0.899 / 0.905 | 0.861 / 0.872 / 0.878 / 0.883 | 0.154 / 0.148 / 0.148 / 0.148 |
| traffic_cone | 20,835 | 13,926 / 14,948 / 15,514 / 15,991 | 0.293 / 0.330 / 0.358 / 0.386 | 0.484 / 0.509 / 0.527 / 0.548 | 0.153 / 0.143 / 0.143 / 0.140 |
| barrier | 3,359 | 518 / 711 / 772 / 804 | 0.017 / 0.055 / 0.066 / 0.071 | 0.199 / 0.263 / 0.279 / 0.285 | 0.151 / 0.107 / 0.107 / 0.083 |

<details>
<summary><strong>TP error — default (recall @0.10)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 13,878 / 13,878 / 13,878 / 13,878 | 0.106 / 0.112 / 0.114 / 0.119 | 0.029 / 0.031 / 0.032 / 0.033 | 0.108 / 0.110 / 0.110 / 0.110 | 0.210 / 0.214 / 0.215 / 0.216 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 2,958 / 2,958 / 2,958 / 2,958 | 0.137 / 0.163 / 0.183 / 0.198 | 0.019 / 0.019 / 0.020 / 0.020 | 0.117 / 0.123 / 0.127 / 0.129 | 0.381 / 0.390 / 0.398 / 0.401 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 721 / 721 / 721 / 721 | 0.137 / 0.148 / 0.159 / 0.161 | 0.045 / 0.046 / 0.046 / 0.047 | 0.096 / 0.098 / 0.101 / 0.101 | 0.220 / 0.226 / 0.226 / 0.226 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 645 / 645 / 645 / 645 | 0.127 / 0.131 / 0.132 / 0.134 | 0.089 / 0.090 / 0.090 / 0.091 | 0.187 / 0.188 / 0.188 / 0.188 | 0.758 / 0.757 / 0.758 / 0.758 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 10,287 / 10,287 / 10,287 / 10,287 | 0.106 / 0.109 / 0.114 / 0.126 | 0.282 / 0.284 / 0.286 / 0.288 | 0.228 / 0.228 / 0.228 / 0.228 | 0.380 / 0.380 / 0.380 / 0.382 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 2,291 / 2,291 / 2,291 / 2,291 | 0.147 / 0.173 / 0.236 / 0.364 | 0.505 / 0.505 / 0.509 / 0.503 | 0.291 / 0.292 / 0.292 / 0.292 | 0.063 / 0.063 / 0.063 / 0.062 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 369 / 369 / 369 / 369 | 0.243 / 0.334 / 0.381 / 0.436 | 0.448 / 0.413 / 0.404 / 0.405 | 0.356 / 0.378 / 0.381 / 0.390 | 0.023 / 0.024 / 0.025 / 0.025 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — medium (recall @0.40)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 51,728 / 51,728 / 51,728 / 51,728 | 0.113 / 0.122 / 0.124 / 0.130 | 0.032 / 0.036 / 0.037 / 0.038 | 0.113 / 0.115 / 0.115 / 0.115 | 0.236 / 0.239 / 0.241 / 0.241 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 11,027 / 11,027 / 11,027 / 11,027 | 0.145 / 0.177 / 0.203 / 0.224 | 0.020 / 0.021 / 0.022 / 0.023 | 0.121 / 0.128 / 0.133 / 0.135 | 0.439 / 0.445 / 0.455 / 0.457 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 2,689 / 2,689 / 2,689 / 2,689 | 0.143 / 0.159 / 0.171 / 0.173 | 0.044 / 0.046 / 0.046 / 0.047 | 0.099 / 0.102 / 0.105 / 0.105 | 0.257 / 0.262 / 0.261 / 0.261 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 2,404 / 2,404 / 2,404 / 2,404 | 0.131 / 0.136 / 0.136 / 0.140 | 0.092 / 0.093 / 0.093 / 0.094 | 0.194 / 0.195 / 0.195 / 0.196 | 0.719 / 0.721 / 0.721 / 0.722 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 38,343 / 38,343 / 38,343 / 38,343 | 0.110 / 0.115 / 0.123 / 0.139 | 0.291 / 0.292 / 0.295 / 0.299 | 0.231 / 0.231 / 0.231 / 0.232 | 0.392 / 0.391 / 0.392 / 0.394 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 8,542 / 8,542 / 8,542 / 8,542 | 0.159 / 0.196 / 0.282 / 0.450 | 0.553 / 0.551 / 0.555 / 0.546 | 0.297 / 0.297 / 0.298 / 0.297 | 0.065 / 0.064 / 0.064 / 0.064 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — optimal</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 109,035 / 113,613 / 115,502 / 116,446 | 0.128 / 0.147 / 0.159 / 0.175 | 0.046 / 0.055 / 0.061 / 0.064 | 0.121 / 0.125 / 0.127 / 0.127 | 0.267 / 0.274 / 0.279 / 0.281 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 20,578 / 23,058 / 24,234 / 24,631 | 0.157 / 0.207 / 0.257 / 0.300 | 0.024 / 0.028 / 0.031 / 0.033 | 0.129 / 0.141 / 0.150 / 0.154 | 0.469 / 0.490 / 0.515 / 0.523 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 5,398 / 5,809 / 5,952 / 5,967 | 0.157 / 0.193 / 0.219 / 0.227 | 0.045 / 0.051 / 0.052 / 0.055 | 0.107 / 0.117 / 0.123 / 0.123 | 0.307 / 0.306 / 0.306 / 0.308 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 4,573 / 4,739 / 4,611 / 4,616 | 0.134 / 0.140 / 0.140 / 0.144 | 0.098 / 0.102 / 0.099 / 0.099 | 0.198 / 0.200 / 0.199 / 0.200 | 0.710 / 0.712 / 0.710 / 0.710 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 78,245 / 79,717 / 80,219 / 80,761 | 0.117 / 0.127 / 0.141 / 0.174 | 0.312 / 0.316 / 0.321 / 0.328 | 0.236 / 0.237 / 0.238 / 0.238 | 0.401 / 0.401 / 0.402 / 0.405 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 10,168 / 11,012 / 11,411 / 11,947 | 0.156 / 0.190 / 0.271 / 0.440 | 0.543 / 0.550 / 0.552 / 0.546 | 0.295 / 0.297 / 0.297 / 0.297 | 0.064 / 0.064 / 0.064 / 0.064 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 452 / 629 / 667 / 715 | 0.247 / 0.353 / 0.409 / 0.509 | 0.452 / 0.430 / 0.423 / 0.430 | 0.363 / 0.405 / 0.412 / 0.433 | 0.023 / 0.025 / 0.025 / 0.025 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

---

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.5447**

| class_name | GTs | num_match@0.5/1.0/2.0/4.0 | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | :---- | :---- | :---- | :---- |
| car | 105,914 | 83,239 / 92,799 / 97,175 / 98,872 | 0.700 / 0.813 / 0.860 / 0.880 | 0.780 / 0.843 / 0.865 / 0.872 | 0.227 / 0.188 / 0.164 / 0.161 |
| truck | 28,864 | 17,414 / 21,372 / 24,178 / 25,356 | 0.467 / 0.634 / 0.757 / 0.803 | 0.624 / 0.729 / 0.800 / 0.823 | 0.223 / 0.182 / 0.149 / 0.140 |
| bus | 5,290 | 3,038 / 4,048 / 4,566 / 4,671 | 0.393 / 0.623 / 0.760 / 0.785 | 0.537 / 0.695 / 0.778 / 0.793 | 0.154 / 0.077 / 0.077 / 0.077 |
| bicycle | 3,608 | 2,849 / 3,030 / 3,047 / 3,063 | 0.575 / 0.635 / 0.640 / 0.647 | 0.662 / 0.691 / 0.693 / 0.698 | 0.147 / 0.142 / 0.142 / 0.142 |
| pedestrian | 48,637 | 43,259 / 43,978 / 44,223 / 44,525 | 0.729 / 0.748 / 0.756 / 0.768 | 0.750 / 0.759 / 0.764 / 0.770 | 0.153 / 0.146 / 0.148 / 0.148 |
| traffic_cone | 9,819 | 6,361 / 6,625 / 6,770 / 7,010 | 0.276 / 0.301 / 0.327 / 0.369 | 0.462 / 0.478 / 0.496 / 0.518 | 0.135 / 0.135 / 0.125 / 0.135 |
| barrier | 2,469 | 180 / 276 / 305 / 321 | 0.000 / 0.001 / 0.002 / 0.004 | 0.093 / 0.143 / 0.154 / 0.157 | 0.116 / 0.097 / 0.097 / 0.097 |

<details>
<summary><strong>TP error — default (recall @0.10)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 11,650 / 11,650 / 11,650 / 11,650 | 0.151 / 0.171 / 0.184 / 0.197 | 0.078 / 0.099 / 0.113 / 0.116 | 0.148 / 0.151 / 0.152 / 0.153 | 0.231 / 0.241 / 0.246 / 0.249 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 3,175 / 3,175 / 3,175 / 3,175 | 0.195 / 0.237 / 0.297 / 0.330 | 0.026 / 0.030 / 0.034 / 0.037 | 0.148 / 0.156 / 0.165 / 0.169 | 0.492 / 0.538 / 0.572 / 0.589 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 581 / 581 / 581 / 581 | 0.196 / 0.267 / 0.331 / 0.356 | 0.088 / 0.084 / 0.088 / 0.094 | 0.124 / 0.134 / 0.142 / 0.144 | 0.190 / 0.199 / 0.202 / 0.202 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 396 / 396 / 396 / 396 | 0.183 / 0.200 / 0.203 / 0.235 | 0.107 / 0.110 / 0.111 / 0.110 | 0.223 / 0.226 / 0.226 / 0.225 | 0.786 / 0.801 / 0.801 / 0.801 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 5,350 / 5,350 / 5,350 / 5,350 | 0.116 / 0.122 / 0.134 / 0.166 | 0.339 / 0.343 / 0.346 / 0.352 | 0.253 / 0.254 / 0.254 / 0.254 | 0.429 / 0.429 / 0.430 / 0.434 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 1,080 / 1,080 / 1,080 / 1,080 | 0.164 / 0.184 / 0.300 / 0.551 | 0.429 / 0.435 / 0.436 / 0.439 | 0.249 / 0.250 / 0.250 / 0.250 | 0.089 / 0.088 / 0.088 / 0.087 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 271 / 271 / 271 | 1.000 / 0.398 / 0.488 / 0.703 | 1.000 / 0.327 / 0.314 / 0.313 | 1.000 / 0.414 / 0.417 / 0.433 | 1.000 / 0.036 / 0.036 / 0.035 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — medium (recall @0.40)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 43,424 / 43,424 / 43,424 / 43,424 | 0.163 / 0.188 / 0.207 / 0.224 | 0.095 / 0.121 / 0.139 / 0.144 | 0.154 / 0.157 / 0.159 / 0.159 | 0.260 / 0.271 / 0.277 / 0.280 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 11,834 / 11,834 / 11,834 / 11,834 | 0.201 / 0.263 / 0.343 / 0.390 | 0.034 / 0.039 / 0.044 / 0.047 | 0.153 / 0.165 / 0.176 / 0.182 | 0.581 / 0.629 / 0.666 / 0.685 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 2,168 / 2,168 / 2,168 / 2,168 | 0.219 / 0.317 / 0.401 / 0.439 | 0.077 / 0.076 / 0.085 / 0.094 | 0.129 / 0.142 / 0.153 / 0.154 | 0.225 / 0.222 / 0.223 / 0.223 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 1,479 / 1,479 / 1,479 / 1,479 | 0.187 / 0.208 / 0.211 / 0.253 | 0.132 / 0.135 / 0.136 / 0.133 | 0.229 / 0.232 / 0.232 / 0.231 | 0.812 / 0.825 / 0.825 / 0.828 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 19,941 / 19,941 / 19,941 / 19,941 | 0.121 / 0.129 / 0.144 / 0.186 | 0.363 / 0.368 / 0.371 / 0.379 | 0.254 / 0.254 / 0.255 / 0.255 | 0.449 / 0.449 / 0.450 / 0.455 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 4,025 / 4,025 / 4,025 / 4,025 | 0.172 / 0.195 / 0.320 / 0.588 | 0.437 / 0.444 / 0.444 / 0.448 | 0.248 / 0.249 / 0.249 / 0.250 | 0.086 / 0.086 / 0.085 / 0.085 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — optimal</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 76,795 / 84,883 / 88,518 / 89,420 | 0.174 / 0.215 / 0.247 / 0.276 | 0.122 / 0.158 / 0.183 / 0.189 | 0.160 / 0.165 / 0.167 / 0.168 | 0.288 / 0.310 / 0.324 / 0.327 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 15,896 / 19,085 / 21,454 / 22,235 | 0.205 / 0.280 / 0.387 / 0.461 | 0.036 / 0.048 / 0.060 / 0.064 | 0.156 / 0.171 / 0.190 / 0.199 | 0.611 / 0.683 / 0.745 / 0.776 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 2,428 / 3,407 / 3,812 / 3,890 | 0.217 / 0.335 / 0.437 / 0.496 | 0.072 / 0.077 / 0.091 / 0.109 | 0.128 / 0.146 / 0.158 / 0.160 | 0.192 / 0.236 / 0.245 / 0.244 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 2,186 / 2,306 / 2,311 / 2,327 | 0.186 / 0.206 / 0.209 / 0.255 | 0.129 / 0.133 / 0.133 / 0.133 | 0.230 / 0.233 / 0.234 / 0.233 | 0.807 / 0.817 / 0.817 / 0.821 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 34,885 / 35,834 / 35,898 / 36,174 | 0.124 / 0.134 / 0.151 / 0.200 | 0.374 / 0.381 / 0.385 / 0.393 | 0.254 / 0.255 / 0.255 / 0.255 | 0.460 / 0.464 / 0.463 / 0.468 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 4,370 / 4,522 / 4,871 / 4,902 | 0.167 / 0.187 / 0.315 / 0.563 | 0.434 / 0.441 / 0.443 / 0.447 | 0.249 / 0.250 / 0.250 / 0.251 | 0.087 / 0.087 / 0.086 / 0.086 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 140 / 222 / 238 / 244 | 0.243 / 0.402 / 0.475 / 0.566 | 0.329 / 0.297 / 0.288 / 0.282 | 0.319 / 0.369 / 0.379 / 0.384 | 0.035 / 0.034 / 0.034 / 0.034 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

---

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.3779**

| class_name | GTs | num_match@0.5/1.0/2.0/4.0 | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | :---- | :---- | :---- | :---- |
| car | 39,577 | 28,566 / 33,503 / 35,851 / 36,643 | 0.545 / 0.697 / 0.765 / 0.785 | 0.653 / 0.737 / 0.769 / 0.778 | 0.197 / 0.165 / 0.142 / 0.129 |
| truck | 18,213 | 7,767 / 11,422 / 14,150 / 15,707 | 0.210 / 0.458 / 0.641 / 0.748 | 0.437 / 0.617 / 0.728 / 0.779 | 0.186 / 0.178 / 0.139 / 0.130 |
| bus | 3,541 | 1,576 / 2,234 / 2,516 / 2,627 | 0.174 / 0.385 / 0.475 / 0.511 | 0.408 / 0.544 / 0.593 / 0.611 | 0.050 / 0.050 / 0.050 / 0.046 |
| bicycle | 942 | 651 / 733 / 753 / 757 | 0.279 / 0.363 / 0.397 / 0.405 | 0.459 / 0.509 / 0.525 / 0.531 | 0.159 / 0.159 / 0.159 / 0.159 |
| pedestrian | 20,134 | 17,375 / 17,647 / 17,776 / 17,948 | 0.639 / 0.654 / 0.663 / 0.678 | 0.685 / 0.693 / 0.697 / 0.704 | 0.136 / 0.136 / 0.136 / 0.136 |
| traffic_cone | 1,231 | 454 / 474 / 484 / 524 | 0.022 / 0.024 / 0.028 / 0.036 | 0.194 / 0.197 / 0.206 / 0.220 | 0.143 / 0.143 / 0.144 / 0.143 |
| barrier | 711 | 38 / 65 / 83 / 88 | 0.000 / 0.000 / 0.000 / 0.001 | 0.061 / 0.106 / 0.132 / 0.137 | 0.082 / 0.059 / 0.059 / 0.059 |

<details>
<summary><strong>TP error — default (recall @0.10)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 4,353 / 4,353 / 4,353 / 4,353 | 0.196 / 0.233 / 0.264 / 0.291 | 0.138 / 0.178 / 0.201 / 0.209 | 0.181 / 0.185 / 0.187 / 0.188 | 0.432 / 0.437 / 0.441 / 0.442 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 2,003 / 2,003 / 2,003 / 2,003 | 0.236 / 0.343 / 0.443 / 0.588 | 0.032 / 0.035 / 0.040 / 0.047 | 0.178 / 0.193 / 0.203 / 0.222 | 0.578 / 0.584 / 0.604 / 0.616 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 389 / 389 / 389 / 389 | 0.255 / 0.363 / 0.444 / 0.496 | 0.034 / 0.039 / 0.041 / 0.046 | 0.138 / 0.147 / 0.160 / 0.167 | 0.215 / 0.244 / 0.272 / 0.278 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 103 / 103 / 103 / 103 | 0.210 / 0.255 / 0.293 / 0.338 | 0.192 / 0.184 / 0.180 / 0.190 | 0.238 / 0.250 / 0.262 / 0.260 | 0.831 / 0.866 / 0.881 / 0.878 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 2,214 / 2,214 / 2,214 / 2,214 | 0.132 / 0.139 / 0.151 / 0.205 | 0.274 / 0.275 / 0.280 / 0.289 | 0.276 / 0.276 / 0.276 / 0.276 | 0.538 / 0.538 / 0.539 / 0.544 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 135 / 135 / 135 / 135 | 0.180 / 0.203 / 0.304 / 0.671 | 0.686 / 0.693 / 0.690 / 0.687 | 0.343 / 0.343 / 0.343 / 0.339 | 0.110 / 0.110 / 0.112 / 0.114 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 78 / 78 | 1.000 / 1.000 / 0.652 / 0.972 | 1.000 / 1.000 / 0.257 / 0.243 | 1.000 / 1.000 / 0.479 / 0.489 | 1.000 / 1.000 / 0.050 / 0.051 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — medium (recall @0.40)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 16,226 / 16,226 / 16,226 / 16,226 | 0.206 / 0.253 / 0.292 / 0.326 | 0.175 / 0.216 / 0.241 / 0.250 | 0.186 / 0.191 / 0.192 / 0.193 | 0.460 / 0.466 / 0.472 / 0.473 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 7,467 / 7,467 / 7,467 / 7,467 | 0.243 / 0.374 / 0.510 / 0.666 | 0.070 / 0.050 / 0.054 / 0.063 | 0.190 / 0.202 / 0.215 / 0.234 | 0.696 / 0.686 / 0.693 / 0.706 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 1,451 / 1,451 / 1,451 / 1,451 | 0.271 / 0.376 / 0.474 / 0.535 | 0.039 / 0.044 / 0.048 / 0.055 | 0.144 / 0.154 / 0.170 / 0.177 | 0.229 / 0.283 / 0.316 / 0.322 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 386 / 386 / 386 / 386 | 0.206 / 0.257 / 0.290 / 0.341 | 0.225 / 0.206 / 0.202 / 0.212 | 0.232 / 0.243 / 0.253 / 0.251 | 0.866 / 0.932 / 0.946 / 0.944 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 8,254 / 8,254 / 8,254 / 8,254 | 0.138 / 0.148 / 0.163 / 0.231 | 0.317 / 0.319 / 0.324 / 0.336 | 0.272 / 0.272 / 0.273 / 0.273 | 0.551 / 0.551 / 0.552 / 0.559 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 0 / 0 / 0 / 504 | 1.000 / 1.000 / 1.000 / 0.742 | 1.000 / 1.000 / 1.000 / 0.658 | 1.000 / 1.000 / 1.000 / 0.328 | 1.000 / 1.000 / 1.000 / 0.124 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — optimal</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 23,637 / 27,796 / 30,068 / 31,051 | 0.208 / 0.265 / 0.316 / 0.362 | 0.187 / 0.242 / 0.276 / 0.289 | 0.187 / 0.193 / 0.195 / 0.196 | 0.468 / 0.484 / 0.494 / 0.497 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 6,996 / 9,963 / 12,146 / 13,138 | 0.240 / 0.374 / 0.533 / 0.705 | 0.046 / 0.049 / 0.062 / 0.073 | 0.184 / 0.204 / 0.222 / 0.242 | 0.662 / 0.697 / 0.744 / 0.767 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 1,277 / 1,704 / 1,857 / 1,963 | 0.257 / 0.364 / 0.451 / 0.512 | 0.039 / 0.043 / 0.046 / 0.050 | 0.136 / 0.148 / 0.162 / 0.170 | 0.221 / 0.269 / 0.302 / 0.305 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 379 / 421 / 434 / 439 | 0.202 / 0.248 / 0.279 / 0.329 | 0.175 / 0.169 / 0.165 / 0.180 | 0.237 / 0.248 / 0.257 / 0.256 | 0.864 / 0.904 / 0.922 / 0.916 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 13,121 / 13,270 / 13,350 / 13,493 | 0.139 / 0.149 / 0.163 / 0.233 | 0.313 / 0.316 / 0.320 / 0.333 | 0.272 / 0.272 / 0.272 / 0.273 | 0.551 / 0.551 / 0.552 / 0.560 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 247 / 251 / 262 / 280 | 0.180 / 0.195 / 0.301 / 0.651 | 0.723 / 0.731 / 0.722 / 0.717 | 0.347 / 0.346 / 0.345 / 0.341 | 0.110 / 0.111 / 0.115 / 0.117 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 28 / 54 / 67 / 70 | 0.268 / 0.452 / 0.633 / 0.979 | 0.376 / 0.247 / 0.242 / 0.235 | 0.374 / 0.427 / 0.450 / 0.454 | 0.048 / 0.046 / 0.048 / 0.048 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

---

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.5936**

| class_name | GTs | num_match@0.5/1.0/2.0/4.0 | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | :---- | :---- | :---- | :---- |
| car | 271,659 | 224,244 / 244,411 / 253,295 / 256,954 | 0.763 / 0.852 / 0.893 / 0.905 | 0.821 / 0.872 / 0.888 / 0.894 | 0.229 / 0.192 / 0.165 / 0.161 |
| truck | 73,974 | 46,967 / 57,070 / 64,081 / 67,424 | 0.504 / 0.682 / 0.801 / 0.856 | 0.652 / 0.765 / 0.833 / 0.860 | 0.246 / 0.182 / 0.166 / 0.139 |
| bus | 15,390 | 10,156 / 12,295 / 13,277 / 13,514 | 0.528 / 0.694 / 0.779 / 0.796 | 0.634 / 0.745 / 0.794 / 0.804 | 0.058 / 0.057 / 0.057 / 0.057 |
| bicycle | 10,415 | 8,835 / 9,191 / 9,237 / 9,265 | 0.685 / 0.726 / 0.730 / 0.733 | 0.738 / 0.755 / 0.757 / 0.759 | 0.202 / 0.198 / 0.198 / 0.198 |
| pedestrian | 162,291 | 147,838 / 150,126 / 151,000 / 151,937 | 0.812 / 0.828 / 0.839 / 0.846 | 0.808 / 0.818 / 0.823 / 0.829 | 0.153 / 0.148 / 0.148 / 0.148 |
| traffic_cone | 31,885 | 20,767 / 22,074 / 22,797 / 23,566 | 0.276 / 0.308 / 0.333 / 0.365 | 0.465 / 0.487 / 0.505 / 0.527 | 0.154 / 0.138 / 0.138 / 0.138 |
| barrier | 6,539 | 737 / 1,055 / 1,164 / 1,216 | 0.001 / 0.021 / 0.029 / 0.033 | 0.147 / 0.204 / 0.218 / 0.224 | 0.125 / 0.107 / 0.107 / 0.082 |

<details>
<summary><strong>TP error — default (recall @0.10)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 29,883 / 29,882 / 29,882 / 29,882 | 0.126 / 0.140 / 0.149 / 0.157 | 0.049 / 0.060 / 0.068 / 0.070 | 0.125 / 0.128 / 0.129 / 0.130 | 0.236 / 0.242 / 0.247 / 0.249 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 8,137 / 8,137 / 8,137 / 8,137 | 0.169 / 0.214 / 0.262 / 0.307 | 0.022 / 0.025 / 0.028 / 0.031 | 0.134 / 0.144 / 0.152 / 0.158 | 0.446 / 0.472 / 0.496 / 0.509 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 1,692 / 1,692 / 1,692 / 1,692 | 0.159 / 0.199 / 0.236 / 0.250 | 0.055 / 0.057 / 0.059 / 0.061 | 0.106 / 0.113 / 0.119 / 0.120 | 0.227 / 0.235 / 0.240 / 0.241 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 1,145 / 1,145 / 1,145 / 1,145 | 0.143 / 0.152 / 0.155 / 0.165 | 0.096 / 0.098 / 0.098 / 0.099 | 0.197 / 0.200 / 0.201 / 0.201 | 0.747 / 0.753 / 0.753 / 0.753 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 17,852 / 17,852 / 17,852 / 17,852 | 0.111 / 0.115 / 0.123 / 0.143 | 0.297 / 0.298 / 0.301 / 0.305 | 0.237 / 0.237 / 0.237 / 0.238 | 0.402 / 0.402 / 0.403 / 0.405 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 3,507 / 3,507 / 3,507 / 3,507 | 0.153 / 0.177 / 0.255 / 0.423 | 0.488 / 0.491 / 0.493 / 0.489 | 0.281 / 0.282 / 0.282 / 0.282 | 0.070 / 0.070 / 0.070 / 0.069 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 719 / 719 / 719 / 719 | 0.248 / 0.365 / 0.428 / 0.532 | 0.434 / 0.387 / 0.377 / 0.375 | 0.385 / 0.392 / 0.395 / 0.406 | 0.028 / 0.028 / 0.028 / 0.028 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — medium (recall @0.40)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 111,380 / 111,380 / 111,380 / 111,380 | 0.138 / 0.157 / 0.169 / 0.180 | 0.061 / 0.077 / 0.088 / 0.090 | 0.133 / 0.136 / 0.138 / 0.138 | 0.264 / 0.271 / 0.276 / 0.278 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 30,329 / 30,329 / 30,329 / 30,329 | 0.181 / 0.241 / 0.307 / 0.367 | 0.027 / 0.031 / 0.034 / 0.038 | 0.142 / 0.154 / 0.164 / 0.171 | 0.519 / 0.545 / 0.570 / 0.584 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 6,309 / 6,309 / 6,309 / 6,309 | 0.179 / 0.238 / 0.288 / 0.309 | 0.053 / 0.056 / 0.060 / 0.065 | 0.113 / 0.123 / 0.130 / 0.132 | 0.259 / 0.261 / 0.265 / 0.266 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 4,270 / 4,270 / 4,270 / 4,270 | 0.151 / 0.163 / 0.166 / 0.182 | 0.108 / 0.110 / 0.110 / 0.111 | 0.207 / 0.210 / 0.211 / 0.211 | 0.741 / 0.749 / 0.750 / 0.750 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 66,539 / 66,539 / 66,539 / 66,539 | 0.116 / 0.122 / 0.133 / 0.162 | 0.312 / 0.314 / 0.318 / 0.323 | 0.241 / 0.242 / 0.242 / 0.242 | 0.420 / 0.420 / 0.420 / 0.424 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 13,072 / 13,072 / 13,072 / 13,072 | 0.164 / 0.198 / 0.296 / 0.504 | 0.526 / 0.528 / 0.530 / 0.525 | 0.284 / 0.285 / 0.286 / 0.285 | 0.072 / 0.072 / 0.071 / 0.071 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — optimal</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 209,064 / 225,972 / 233,684 / 235,583 | 0.154 / 0.187 / 0.212 / 0.234 | 0.088 / 0.115 / 0.133 / 0.138 | 0.143 / 0.148 / 0.150 / 0.151 | 0.297 / 0.313 / 0.324 / 0.326 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 43,131 / 52,358 / 57,620 / 60,504 | 0.187 / 0.266 / 0.362 / 0.453 | 0.030 / 0.040 / 0.047 / 0.054 | 0.147 / 0.164 / 0.179 / 0.191 | 0.550 / 0.602 / 0.646 / 0.675 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 9,224 / 10,854 / 11,562 / 11,707 | 0.189 / 0.266 / 0.329 / 0.363 | 0.054 / 0.060 / 0.066 / 0.074 | 0.117 / 0.130 / 0.140 / 0.142 | 0.272 / 0.282 / 0.289 / 0.291 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 7,044 / 7,224 / 7,242 / 7,260 | 0.152 / 0.163 / 0.167 / 0.183 | 0.109 / 0.111 / 0.111 / 0.111 | 0.209 / 0.211 / 0.212 / 0.212 | 0.743 / 0.750 / 0.752 / 0.752 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 125,762 / 128,271 / 129,072 / 130,130 | 0.121 / 0.131 / 0.146 / 0.188 | 0.328 / 0.333 / 0.338 / 0.347 | 0.245 / 0.246 / 0.246 / 0.247 | 0.432 / 0.433 / 0.434 / 0.438 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 14,442 / 15,921 / 16,500 / 17,198 | 0.158 / 0.190 / 0.285 / 0.485 | 0.515 / 0.523 / 0.525 / 0.522 | 0.283 / 0.285 / 0.284 / 0.285 | 0.072 / 0.071 / 0.071 / 0.071 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 624 / 886 / 950 / 1,028 | 0.246 / 0.367 / 0.435 / 0.567 | 0.424 / 0.390 / 0.381 / 0.385 | 0.355 / 0.393 / 0.403 / 0.423 | 0.026 / 0.028 / 0.028 / 0.029 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

---

**LargeBus**: db_largebus_v1 + db_largebus_v2 + db_largebus_v3 (1,228 frames)  

**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.6292**

| class_name | GTs | num_match@0.5/1.0/2.0/4.0 | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | :---- | :---- | :---- | :---- |
| car | 14,872 | 13,343 / 13,829 / 14,017 / 14,145 | 0.867 / 0.906 / 0.926 / 0.937 | 0.914 / 0.938 / 0.943 / 0.946 | 0.228 / 0.194 / 0.193 / 0.145 |
| truck | 1,192 | 960 / 1,091 / 1,118 / 1,135 | 0.726 / 0.886 / 0.910 / 0.928 | 0.823 / 0.910 / 0.919 / 0.923 | 0.247 / 0.208 / 0.157 / 0.152 |
| bus | 336 | 283 / 329 / 332 / 332 | 0.793 / 0.962 / 0.973 / 0.973 | 0.859 / 0.950 / 0.959 / 0.959 | 0.286 / 0.065 / 0.065 / 0.065 |
| bicycle | 740 | 694 / 713 / 717 / 724 | 0.825 / 0.870 / 0.877 / 0.891 | 0.837 / 0.855 / 0.856 / 0.863 | 0.201 / 0.211 / 0.211 / 0.211 |
| pedestrian | 5,055 | 4,641 / 4,701 / 4,718 / 4,731 | 0.825 / 0.839 / 0.848 / 0.854 | 0.840 / 0.847 / 0.851 / 0.855 | 0.160 / 0.160 / 0.160 / 0.160 |
| traffic_cone | 60 | 24 / 24 / 24 / 24 | 0.000 / 0.000 / 0.000 / 0.000 | 0.028 / 0.029 / 0.029 / 0.029 | 0.077 / 0.153 / 0.153 / 0.153 |
| barrier | 0 | 0 / 0 / 0 / 0 | 0.000 / 0.000 / 0.000 / 0.000 | nan / nan / nan / nan | nan / nan / nan / nan |

<details>
<summary><strong>TP error — default (recall @0.10)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 1,635 / 1,635 / 1,635 / 1,635 | 0.110 / 0.114 / 0.116 / 0.119 | 0.040 / 0.042 / 0.044 / 0.045 | 0.112 / 0.113 / 0.113 / 0.113 | 0.141 / 0.142 / 0.143 / 0.143 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 131 / 131 / 131 / 131 | 0.140 / 0.171 / 0.175 / 0.180 | 0.035 / 0.039 / 0.040 / 0.043 | 0.128 / 0.132 / 0.134 / 0.134 | 0.229 / 0.223 / 0.223 / 0.222 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 36 / 36 / 36 / 36 | 0.144 / 0.164 / 0.166 / 0.166 | 0.176 / 0.173 / 0.173 / 0.173 | 0.097 / 0.099 / 0.099 / 0.099 | 0.188 / 0.183 / 0.182 / 0.182 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 81 / 81 / 81 / 81 | 0.131 / 0.142 / 0.145 / 0.156 | 0.092 / 0.092 / 0.093 / 0.093 | 0.206 / 0.211 / 0.211 / 0.214 | 0.517 / 0.511 / 0.511 / 0.510 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 556 / 556 / 556 / 556 | 0.096 / 0.099 / 0.105 / 0.123 | 0.290 / 0.291 / 0.294 / 0.299 | 0.204 / 0.204 / 0.205 / 0.205 | 0.240 / 0.240 / 0.240 / 0.244 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 6 / 6 / 6 / 6 | 0.154 / 0.251 / 0.251 / 0.251 | 1.295 / 1.395 / 1.395 / 1.395 | 0.369 / 0.370 / 0.370 / 0.370 | 0.087 / 0.088 / 0.088 / 0.088 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — medium (recall @0.40)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 6,097 / 6,097 / 6,097 / 6,097 | 0.118 / 0.124 / 0.126 / 0.130 | 0.045 / 0.048 / 0.050 / 0.051 | 0.117 / 0.118 / 0.119 / 0.119 | 0.155 / 0.156 / 0.157 / 0.157 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 488 / 488 / 488 / 488 | 0.151 / 0.188 / 0.195 / 0.201 | 0.041 / 0.047 / 0.048 / 0.052 | 0.140 / 0.144 / 0.146 / 0.147 | 0.231 / 0.224 / 0.224 / 0.223 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 137 / 137 / 137 / 137 | 0.157 / 0.184 / 0.186 / 0.186 | 0.206 / 0.196 / 0.195 / 0.195 | 0.096 / 0.099 / 0.099 / 0.099 | 0.192 / 0.184 / 0.183 / 0.183 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 303 / 303 / 303 / 303 | 0.135 / 0.146 / 0.151 / 0.167 | 0.107 / 0.107 / 0.108 / 0.108 | 0.214 / 0.218 / 0.219 / 0.222 | 0.499 / 0.496 / 0.495 / 0.494 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 2,072 / 2,072 / 2,072 / 2,072 | 0.100 / 0.104 / 0.112 / 0.134 | 0.326 / 0.327 / 0.330 / 0.336 | 0.211 / 0.211 / 0.212 / 0.212 | 0.246 / 0.246 / 0.247 / 0.252 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — optimal</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 13,062 / 13,479 / 13,554 / 13,743 | 0.134 / 0.148 / 0.155 / 0.171 | 0.065 / 0.081 / 0.083 / 0.090 | 0.128 / 0.130 / 0.131 / 0.132 | 0.156 / 0.158 / 0.159 / 0.160 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 932 / 1,043 / 1,066 / 1,074 | 0.169 / 0.218 / 0.232 / 0.248 | 0.057 / 0.063 / 0.067 / 0.081 | 0.151 / 0.158 / 0.161 / 0.162 | 0.223 / 0.215 / 0.218 / 0.217 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 275 / 321 / 324 / 324 | 0.176 / 0.238 / 0.247 / 0.247 | 0.192 / 0.169 / 0.167 / 0.167 | 0.097 / 0.105 / 0.105 / 0.105 | 0.184 / 0.184 / 0.182 / 0.182 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 602 / 607 / 608 / 613 | 0.141 / 0.152 / 0.154 / 0.174 | 0.122 / 0.120 / 0.120 / 0.120 | 0.222 / 0.226 / 0.227 / 0.231 | 0.485 / 0.483 / 0.483 / 0.482 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 4,140 / 4,178 / 4,198 / 4,214 | 0.108 / 0.116 / 0.127 / 0.158 | 0.362 / 0.364 / 0.367 / 0.377 | 0.217 / 0.218 / 0.219 / 0.219 | 0.248 / 0.248 / 0.248 / 0.254 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 23 / 13 / 13 / 13 | 0.147 / 0.302 / 0.302 / 0.302 | 1.599 / 1.355 / 1.355 / 1.355 | 0.378 / 0.374 / 0.374 / 0.374 | 0.085 / 0.087 / 0.087 / 0.087 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan |

</details>

---

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.5080**

| class_name | GTs | num_match@0.5/1.0/2.0/4.0 | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | :---- | :---- | :---- | :---- |
| car | 10,929 | 8,721 / 9,616 / 10,038 / 10,208 | 0.724 / 0.828 / 0.872 / 0.891 | 0.802 / 0.862 / 0.880 / 0.884 | 0.216 / 0.185 / 0.177 / 0.175 |
| truck | 1,009 | 649 / 789 / 850 / 868 | 0.527 / 0.699 / 0.770 / 0.785 | 0.661 / 0.785 / 0.824 / 0.829 | 0.229 / 0.154 / 0.134 / 0.134 |
| bus | 141 | 112 / 130 / 131 / 134 | 0.632 / 0.861 / 0.864 / 0.883 | 0.752 / 0.881 / 0.881 / 0.881 | 0.483 / 0.431 / 0.431 / 0.431 |
| bicycle | 460 | 321 / 363 / 373 / 375 | 0.461 / 0.573 / 0.593 / 0.593 | 0.602 / 0.652 / 0.657 / 0.657 | 0.143 / 0.143 / 0.143 / 0.143 |
| pedestrian | 3,721 | 3,077 / 3,133 / 3,157 / 3,180 | 0.646 / 0.667 / 0.673 / 0.683 | 0.696 / 0.707 / 0.712 / 0.717 | 0.119 / 0.122 / 0.122 / 0.122 |
| traffic_cone | 4 | 2 / 2 / 2 / 2 | 0.000 / 0.000 / 0.000 / 0.000 | 0.043 / 0.043 / 0.043 / 0.043 | 0.227 / 0.227 / 0.227 / 0.227 |
| barrier | 0 | 0 / 0 / 0 / 0 | 0.000 / 0.000 / 0.000 / 0.000 | nan / nan / nan / nan | nan / nan / nan / nan |

<details>
<summary><strong>TP error — default (recall @0.10)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 1,202 / 1,202 / 1,202 / 1,202 | 0.157 / 0.173 / 0.183 / 0.193 | 0.091 / 0.111 / 0.122 / 0.126 | 0.147 / 0.150 / 0.152 / 0.153 | 0.224 / 0.228 / 0.232 / 0.233 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 110 / 110 / 110 / 110 | 0.187 / 0.225 / 0.251 / 0.259 | 0.038 / 0.045 / 0.048 / 0.051 | 0.164 / 0.175 / 0.180 / 0.181 | 0.241 / 0.245 / 0.259 / 0.263 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 15 / 15 / 15 / 15 | 0.225 / 0.269 / 0.271 / 0.278 | 0.493 / 0.444 / 0.443 / 0.439 | 0.100 / 0.114 / 0.115 / 0.115 | 0.155 / 0.185 / 0.188 / 0.191 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 50 / 50 / 50 / 50 | 0.191 / 0.235 / 0.247 / 0.251 | 0.178 / 0.184 / 0.189 / 0.188 | 0.237 / 0.254 / 0.258 / 0.258 | 0.611 / 0.693 / 0.689 / 0.689 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 409 / 409 / 409 / 409 | 0.114 / 0.124 / 0.132 / 0.161 | 0.415 / 0.417 / 0.418 / 0.426 | 0.203 / 0.204 / 0.205 / 0.205 | 0.280 / 0.279 / 0.279 / 0.282 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 0 / 0 / 0 / 0 | 0.219 / 0.219 / 0.219 / 0.219 | 2.371 / 2.371 / 2.371 / 2.371 | 0.513 / 0.513 / 0.513 / 0.513 | 0.034 / 0.034 / 0.034 / 0.034 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — medium (recall @0.40)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 4,480 / 4,480 / 4,480 / 4,480 | 0.167 / 0.188 / 0.202 / 0.215 | 0.114 / 0.138 / 0.153 / 0.157 | 0.154 / 0.158 / 0.160 / 0.161 | 0.240 / 0.245 / 0.249 / 0.250 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 413 / 413 / 413 / 413 | 0.197 / 0.253 / 0.289 / 0.300 | 0.045 / 0.054 / 0.058 / 0.064 | 0.173 / 0.187 / 0.193 / 0.194 | 0.248 / 0.252 / 0.266 / 0.272 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 57 / 57 / 57 / 57 | 0.230 / 0.274 / 0.276 / 0.288 | 0.405 / 0.359 / 0.358 / 0.356 | 0.104 / 0.117 / 0.118 / 0.118 | 0.172 / 0.201 / 0.207 / 0.210 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 188 / 188 / 188 / 188 | 0.192 / 0.236 / 0.251 / 0.257 | 0.219 / 0.227 / 0.232 / 0.232 | 0.246 / 0.261 / 0.265 / 0.265 | 0.610 / 0.685 / 0.681 / 0.681 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 1,525 / 1,525 / 1,525 / 1,525 | 0.123 / 0.136 / 0.149 / 0.189 | 0.463 / 0.464 / 0.465 / 0.475 | 0.209 / 0.210 / 0.211 / 0.212 | 0.297 / 0.295 / 0.295 / 0.298 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 1 / 1 / 1 / 1 | 0.205 / 0.205 / 0.205 / 0.205 | 2.328 / 2.328 / 2.328 / 2.328 | 0.517 / 0.517 / 0.517 / 0.517 | 0.034 / 0.034 / 0.034 / 0.034 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — optimal</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 8,159 / 8,918 / 9,145 / 9,193 | 0.179 / 0.214 / 0.238 / 0.253 | 0.166 / 0.202 / 0.219 / 0.223 | 0.162 / 0.168 / 0.169 / 0.170 | 0.252 / 0.272 / 0.279 / 0.280 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 584 / 735 / 782 / 787 | 0.203 / 0.288 / 0.342 / 0.356 | 0.043 / 0.076 / 0.081 / 0.086 | 0.176 / 0.196 / 0.204 / 0.206 | 0.252 / 0.242 / 0.257 / 0.271 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 97 / 115 / 115 / 115 | 0.232 / 0.284 / 0.284 / 0.284 | 0.381 / 0.326 / 0.326 / 0.326 | 0.105 / 0.118 / 0.118 / 0.118 | 0.224 / 0.244 / 0.244 / 0.244 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 243 / 263 / 265 / 265 | 0.192 / 0.233 / 0.240 / 0.247 | 0.198 / 0.205 / 0.204 / 0.203 | 0.244 / 0.257 / 0.259 / 0.259 | 0.588 / 0.651 / 0.648 / 0.648 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 2,464 / 2,492 / 2,508 / 2,524 | 0.127 / 0.141 / 0.156 / 0.198 | 0.474 / 0.473 / 0.475 / 0.484 | 0.211 / 0.212 / 0.213 / 0.214 | 0.302 / 0.300 / 0.300 / 0.302 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 2 / 2 / 2 / 2 | 0.199 / 0.199 / 0.199 / 0.199 | 2.313 / 2.313 / 2.313 / 2.313 | 0.518 / 0.518 / 0.518 / 0.518 | 0.034 / 0.034 / 0.034 / 0.034 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan |

</details>

---

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.3869**

| class_name | GTs | num_match@0.5/1.0/2.0/4.0 | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | :---- | :---- | :---- | :---- |
| car | 2,883 | 2,115 / 2,486 / 2,667 / 2,711 | 0.586 / 0.739 / 0.796 / 0.814 | 0.688 / 0.769 / 0.794 / 0.801 | 0.231 / 0.180 / 0.162 / 0.163 |
| truck | 600 | 314 / 447 / 515 / 531 | 0.307 / 0.600 / 0.741 / 0.769 | 0.497 / 0.688 / 0.776 / 0.790 | 0.257 / 0.234 / 0.120 / 0.120 |
| bus | 60 | 30 / 45 / 47 / 48 | 0.337 / 0.572 / 0.607 / 0.610 | 0.516 / 0.650 / 0.667 / 0.667 | 0.098 / 0.044 / 0.098 / 0.098 |
| bicycle | 85 | 54 / 62 / 65 / 65 | 0.246 / 0.365 / 0.392 / 0.392 | 0.452 / 0.537 / 0.561 / 0.561 | 0.120 / 0.122 / 0.122 / 0.122 |
| pedestrian | 1,092 | 881 / 895 / 900 / 918 | 0.474 / 0.485 / 0.492 / 0.507 | 0.584 / 0.590 / 0.594 / 0.597 | 0.125 / 0.125 / 0.125 / 0.125 |
| traffic_cone | 0 | 0 / 0 / 0 / 0 | 0.000 / 0.000 / 0.000 / 0.000 | nan / nan / nan / nan | nan / nan / nan / nan |
| barrier | 0 | 0 / 0 / 0 / 0 | 0.000 / 0.000 / 0.000 / 0.000 | nan / nan / nan / nan | nan / nan / nan / nan |

<details>
<summary><strong>TP error — default (recall @0.10)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 317 / 317 / 317 / 317 | 0.195 / 0.229 / 0.248 / 0.272 | 0.108 / 0.142 / 0.165 / 0.172 | 0.179 / 0.186 / 0.187 / 0.187 | 0.521 / 0.529 / 0.535 / 0.536 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 66 / 66 / 66 / 66 | 0.222 / 0.321 / 0.382 / 0.402 | 0.045 / 0.050 / 0.057 / 0.062 | 0.180 / 0.191 / 0.200 / 0.203 | 0.177 / 0.168 / 0.166 / 0.173 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 6 / 6 / 6 / 6 | 0.273 / 0.344 / 0.364 / 0.371 | 0.168 / 0.213 / 0.210 / 0.211 | 0.140 / 0.162 / 0.165 / 0.166 | 0.184 / 0.307 / 0.440 / 0.450 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 9 / 9 / 9 / 9 | 0.183 / 0.260 / 0.290 / 0.290 | 0.091 / 0.084 / 0.086 / 0.086 | 0.236 / 0.248 / 0.253 / 0.253 | 0.721 / 0.705 / 0.729 / 0.729 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 120 / 120 / 120 / 120 | 0.131 / 0.137 / 0.152 / 0.207 | 0.398 / 0.398 / 0.400 / 0.407 | 0.172 / 0.172 / 0.172 / 0.173 | 0.354 / 0.355 / 0.355 / 0.364 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — medium (recall @0.40)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 1,182 / 1,182 / 1,182 / 1,182 | 0.206 / 0.249 / 0.276 / 0.309 | 0.146 / 0.189 / 0.219 / 0.227 | 0.186 / 0.193 / 0.194 / 0.195 | 0.579 / 0.582 / 0.587 / 0.585 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 246 / 246 / 246 / 246 | 0.231 / 0.345 / 0.430 / 0.459 | 0.072 / 0.066 / 0.075 / 0.081 | 0.185 / 0.196 / 0.209 / 0.212 | 0.215 / 0.199 / 0.190 / 0.199 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 24 / 24 / 24 / 24 | 0.291 / 0.384 / 0.416 / 0.425 | 0.269 / 0.312 / 0.301 / 0.298 | 0.146 / 0.175 / 0.178 / 0.179 | 0.315 / 0.475 / 0.689 / 0.695 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 34 / 34 / 34 / 34 | 0.183 / 0.261 / 0.315 / 0.315 | 0.110 / 0.109 / 0.110 / 0.110 | 0.257 / 0.269 / 0.275 / 0.275 | 0.757 / 0.750 / 0.789 / 0.789 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 447 / 447 / 447 / 447 | 0.136 / 0.145 / 0.162 / 0.231 | 0.472 / 0.470 / 0.472 / 0.483 | 0.176 / 0.176 / 0.176 / 0.177 | 0.387 / 0.389 / 0.389 / 0.401 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — optimal</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 1,763 / 2,103 / 2,227 / 2,243 | 0.207 / 0.261 / 0.299 / 0.338 | 0.155 / 0.220 / 0.266 / 0.271 | 0.188 / 0.196 / 0.197 / 0.198 | 0.571 / 0.603 / 0.613 / 0.613 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 254 / 358 / 448 / 456 | 0.225 / 0.356 / 0.481 / 0.514 | 0.051 / 0.061 / 0.095 / 0.097 | 0.182 / 0.199 / 0.221 / 0.224 | 0.192 / 0.187 / 0.206 / 0.217 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 24 / 38 / 31 / 31 | 0.304 / 0.380 / 0.407 / 0.407 | 0.305 / 0.282 / 0.341 / 0.341 | 0.155 / 0.172 / 0.176 / 0.176 | 0.329 / 0.487 / 0.633 / 0.633 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 38 / 44 / 46 / 46 | 0.178 / 0.253 / 0.304 / 0.304 | 0.113 / 0.102 / 0.102 / 0.102 | 0.252 / 0.263 / 0.267 / 0.267 | 0.725 / 0.699 / 0.717 / 0.717 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 613 / 619 / 623 / 626 | 0.134 / 0.141 / 0.157 / 0.208 | 0.439 / 0.441 / 0.439 / 0.442 | 0.173 / 0.172 / 0.173 / 0.174 | 0.383 / 0.385 / 0.385 / 0.389 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 0 / 0 / 0 / 0 | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan |
| barrier | 0 / 0 / 0 / 0 | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan |

</details>

---

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.5663**

| class_name | GTs | num_match@0.5/1.0/2.0/4.0 | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | :---- | :---- | :---- | :---- |
| car | 28,684 | 24,224 / 25,996 / 26,796 / 27,149 | 0.797 / 0.872 / 0.903 / 0.914 | 0.852 / 0.894 / 0.906 / 0.910 | 0.225 / 0.192 / 0.177 / 0.177 |
| truck | 2,801 | 1,924 / 2,330 / 2,488 / 2,543 | 0.570 / 0.767 / 0.831 / 0.850 | 0.700 / 0.818 / 0.856 / 0.863 | 0.250 / 0.177 / 0.138 / 0.138 |
| bus | 537 | 425 / 504 / 510 / 514 | 0.707 / 0.897 / 0.909 / 0.916 | 0.795 / 0.889 / 0.895 / 0.895 | 0.337 / 0.284 / 0.284 / 0.284 |
| bicycle | 1,285 | 1,072 / 1,141 / 1,158 / 1,167 | 0.666 / 0.740 / 0.755 / 0.762 | 0.727 / 0.759 / 0.764 / 0.769 | 0.200 / 0.200 / 0.150 / 0.150 |
| pedestrian | 9,868 | 8,613 / 8,743 / 8,786 / 8,842 | 0.733 / 0.748 / 0.757 / 0.764 | 0.759 / 0.769 / 0.773 / 0.777 | 0.146 / 0.146 / 0.146 / 0.146 |
| traffic_cone | 64 | 26 / 26 / 26 / 26 | 0.000 / 0.000 / 0.000 / 0.000 | 0.024 / 0.027 / 0.027 / 0.027 | 0.157 / 0.153 / 0.153 / 0.153 |
| barrier | 0 | 0 / 0 / 0 / 0 | 0.000 / 0.000 / 0.000 / 0.000 | nan / nan / nan / nan | nan / nan / nan / nan |

<details>
<summary><strong>TP error — default (recall @0.10)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 3,155 / 3,155 / 3,155 / 3,155 | 0.127 / 0.137 / 0.143 / 0.148 | 0.057 / 0.067 / 0.072 / 0.074 | 0.125 / 0.127 / 0.128 / 0.129 | 0.177 / 0.182 / 0.185 / 0.186 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 308 / 308 / 308 / 308 | 0.165 / 0.209 / 0.229 / 0.237 | 0.038 / 0.043 / 0.046 / 0.050 | 0.147 / 0.155 / 0.159 / 0.160 | 0.223 / 0.219 / 0.223 / 0.225 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 59 / 59 / 59 / 59 | 0.165 / 0.195 / 0.197 / 0.199 | 0.250 / 0.239 / 0.238 / 0.238 | 0.098 / 0.103 / 0.103 / 0.104 | 0.186 / 0.190 / 0.193 / 0.193 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 141 / 141 / 141 / 141 | 0.146 / 0.167 / 0.175 / 0.186 | 0.114 / 0.115 / 0.118 / 0.117 | 0.215 / 0.223 / 0.226 / 0.227 | 0.538 / 0.554 / 0.553 / 0.553 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 1,085 / 1,085 / 1,085 / 1,085 | 0.103 / 0.109 / 0.116 / 0.140 | 0.338 / 0.339 / 0.342 / 0.348 | 0.205 / 0.206 / 0.206 / 0.207 | 0.257 / 0.257 / 0.257 / 0.262 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 7 / 7 / 7 / 7 | 0.160 / 0.246 / 0.246 / 0.246 | 1.421 / 1.507 / 1.507 / 1.507 | 0.389 / 0.391 / 0.391 / 0.391 | 0.081 / 0.082 / 0.082 / 0.082 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — medium (recall @0.40)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 11,760 / 11,760 / 11,760 / 11,760 | 0.139 / 0.152 / 0.160 / 0.167 | 0.069 / 0.082 / 0.090 / 0.092 | 0.133 / 0.136 / 0.137 / 0.137 | 0.197 / 0.202 / 0.205 / 0.206 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 1,148 / 1,148 / 1,148 / 1,148 | 0.179 / 0.235 / 0.263 / 0.275 | 0.046 / 0.052 / 0.055 / 0.061 | 0.159 / 0.168 / 0.173 / 0.174 | 0.229 / 0.225 / 0.229 / 0.232 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 220 / 220 / 220 / 220 | 0.181 / 0.217 / 0.220 / 0.223 | 0.254 / 0.238 / 0.237 / 0.237 | 0.099 / 0.106 / 0.106 / 0.106 | 0.184 / 0.190 / 0.194 / 0.195 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 526 / 526 / 526 / 526 | 0.154 / 0.176 / 0.187 / 0.200 | 0.137 / 0.138 / 0.143 / 0.141 | 0.226 / 0.234 / 0.236 / 0.239 | 0.525 / 0.543 / 0.542 / 0.542 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 4,045 / 4,045 / 4,045 / 4,045 | 0.111 / 0.118 / 0.129 / 0.159 | 0.383 / 0.383 / 0.386 / 0.393 | 0.209 / 0.210 / 0.210 / 0.211 | 0.268 / 0.268 / 0.268 / 0.273 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — optimal</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 23,001 / 24,465 / 24,972 / 25,071 | 0.155 / 0.182 / 0.199 / 0.212 | 0.107 / 0.136 / 0.149 / 0.152 | 0.145 / 0.149 / 0.151 / 0.151 | 0.222 / 0.237 / 0.243 / 0.243 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 1,757 / 2,148 / 2,295 / 2,313 | 0.188 / 0.266 / 0.319 / 0.340 | 0.051 / 0.067 / 0.078 / 0.087 | 0.163 / 0.178 / 0.188 / 0.190 | 0.228 / 0.223 / 0.227 / 0.235 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 391 / 442 / 445 / 445 | 0.196 / 0.242 / 0.248 / 0.248 | 0.248 / 0.230 / 0.228 / 0.228 | 0.102 / 0.110 / 0.110 / 0.110 | 0.191 / 0.192 / 0.209 / 0.209 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 839 / 875 / 938 / 944 | 0.154 / 0.176 / 0.199 / 0.216 | 0.135 / 0.135 / 0.157 / 0.157 | 0.226 / 0.234 / 0.243 / 0.246 | 0.519 / 0.542 / 0.536 / 0.535 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 7,078 / 7,166 / 7,204 / 7,247 | 0.115 / 0.125 / 0.138 / 0.175 | 0.398 / 0.400 / 0.402 / 0.413 | 0.210 / 0.211 / 0.212 / 0.213 | 0.272 / 0.272 / 0.272 / 0.278 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 13 / 15 / 15 / 15 | 0.174 / 0.289 / 0.289 / 0.289 | 1.258 / 1.483 / 1.483 / 1.483 | 0.389 / 0.393 / 0.393 / 0.393 | 0.079 / 0.080 / 0.080 / 0.080 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan |

</details>

---

**J6Gen2**: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 + db_j6gen2_v6 + db_j6gen2_v7 + db_j6gen2_v8 + db_j6gen2_v9 + db_j6gen2_v10 + db_j6gen2_v11 + db_j6gen2_v12 (4,682 frames)


**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.7246**

| class_name | GTs | num_match@0.5/1.0/2.0/4.0 | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | :---- | :---- | :---- | :---- |
| car | 134,456 | 107,307 / 117,848 / 123,019 / 125,254 | 0.730 / 0.828 / 0.873 / 0.893 | 0.804 / 0.858 / 0.879 / 0.887 | 0.240 / 0.200 / 0.165 / 0.162 |
| truck | 14,900 | 9,814 / 11,683 / 12,829 / 13,364 | 0.527 / 0.690 / 0.786 / 0.828 | 0.667 / 0.766 / 0.817 / 0.841 | 0.225 / 0.182 / 0.170 / 0.145 |
| bus | 4,970 | 3,654 / 4,321 / 4,639 / 4,711 | 0.643 / 0.809 / 0.902 / 0.916 | 0.740 / 0.842 / 0.892 / 0.901 | 0.318 / 0.140 / 0.103 / 0.102 |
| bicycle | 2,486 | 2,153 / 2,228 / 2,239 / 2,239 | 0.755 / 0.796 / 0.801 / 0.802 | 0.789 / 0.809 / 0.809 / 0.812 | 0.186 / 0.164 / 0.164 / 0.164 |
| pedestrian | 29,102 | 25,516 / 26,078 / 26,291 / 26,512 | 0.728 / 0.752 / 0.765 / 0.778 | 0.748 / 0.761 / 0.766 / 0.773 | 0.153 / 0.153 / 0.153 / 0.153 |
| traffic_cone | 11,344 | 7,980 / 8,516 / 8,749 / 9,099 | 0.342 / 0.388 / 0.410 / 0.448 | 0.509 / 0.538 / 0.551 / 0.575 | 0.177 / 0.156 / 0.153 / 0.157 |
| barrier | 2,117 | 737 / 1,055 / 1,164 / 1,216 | 0.104 / 0.236 / 0.275 / 0.290 | 0.323 / 0.439 / 0.467 / 0.479 | 0.151 / 0.155 / 0.115 / 0.115 |

<details>
<summary><strong>TP error — default (recall @0.10)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 14,790 / 14,790 / 14,790 / 14,790 | 0.129 / 0.145 / 0.156 / 0.167 | 0.064 / 0.082 / 0.095 / 0.099 | 0.129 / 0.133 / 0.134 / 0.135 | 0.151 / 0.153 / 0.154 / 0.155 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 1,639 / 1,639 / 1,639 / 1,639 | 0.161 / 0.202 / 0.244 / 0.276 | 0.029 / 0.033 / 0.037 / 0.039 | 0.134 / 0.144 / 0.150 / 0.154 | 0.360 / 0.375 / 0.382 / 0.385 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 546 / 546 / 546 / 546 | 0.132 / 0.163 / 0.196 / 0.203 | 0.057 / 0.059 / 0.061 / 0.064 | 0.094 / 0.100 / 0.107 / 0.107 | 0.126 / 0.130 / 0.131 / 0.131 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 273 / 273 / 273 / 273 | 0.146 / 0.154 / 0.155 / 0.158 | 0.079 / 0.081 / 0.082 / 0.083 | 0.195 / 0.197 / 0.197 / 0.197 | 0.593 / 0.594 / 0.595 / 0.596 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 3,201 / 3,201 / 3,201 / 3,201 | 0.110 / 0.118 / 0.133 / 0.160 | 0.455 / 0.459 / 0.464 / 0.469 | 0.238 / 0.239 / 0.239 / 0.240 | 0.245 / 0.245 / 0.246 / 0.248 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 1,247 / 1,247 / 1,247 / 1,247 | 0.152 / 0.181 / 0.225 / 0.389 | 0.292 / 0.291 / 0.290 / 0.293 | 0.283 / 0.283 / 0.283 / 0.284 | 0.027 / 0.027 / 0.027 / 0.027 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 232 / 232 / 232 / 232 | 0.235 / 0.339 / 0.389 / 0.459 | 0.390 / 0.353 / 0.349 / 0.349 | 0.319 / 0.353 / 0.362 / 0.370 | 0.026 / 0.028 / 0.028 / 0.028 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — medium (recall @0.40)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 55,126 / 55,126 / 55,126 / 55,126 | 0.143 / 0.165 / 0.181 / 0.196 | 0.084 / 0.109 / 0.126 / 0.131 | 0.139 / 0.143 / 0.144 / 0.145 | 0.164 / 0.165 / 0.166 / 0.166 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 6,109 / 6,109 / 6,109 / 6,109 | 0.177 / 0.231 / 0.287 / 0.332 | 0.037 / 0.042 / 0.047 / 0.049 | 0.144 / 0.155 / 0.164 / 0.169 | 0.386 / 0.403 / 0.410 / 0.414 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 2,037 / 2,037 / 2,037 / 2,037 | 0.149 / 0.192 / 0.229 / 0.239 | 0.056 / 0.060 / 0.064 / 0.069 | 0.102 / 0.108 / 0.115 / 0.115 | 0.149 / 0.150 / 0.149 / 0.150 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 1,019 / 1,019 / 1,019 / 1,019 | 0.151 / 0.161 / 0.163 / 0.168 | 0.090 / 0.093 / 0.094 / 0.096 | 0.201 / 0.204 / 0.204 / 0.204 | 0.618 / 0.619 / 0.621 / 0.622 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 11,931 / 11,931 / 11,931 / 11,931 | 0.115 / 0.126 / 0.145 / 0.183 | 0.487 / 0.491 / 0.496 / 0.503 | 0.243 / 0.244 / 0.245 / 0.245 | 0.261 / 0.260 / 0.261 / 0.264 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 4,651 / 4,651 / 4,651 / 4,651 | 0.161 / 0.197 / 0.257 / 0.463 | 0.318 / 0.316 / 0.315 / 0.318 | 0.288 / 0.287 / 0.287 / 0.288 | 0.028 / 0.028 / 0.028 / 0.028 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 867 / 867 / 867 | 1.000 / 0.374 / 0.445 / 0.567 | 1.000 / 0.395 / 0.385 / 0.383 | 1.000 / 0.407 / 0.414 / 0.425 | 1.000 / 0.028 / 0.029 / 0.029 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — optimal</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 100,294 / 109,159 / 113,989 / 115,141 | 0.159 / 0.194 / 0.227 / 0.255 | 0.120 / 0.160 / 0.190 / 0.196 | 0.148 / 0.154 / 0.156 / 0.157 | 0.172 / 0.175 / 0.178 / 0.178 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 8,931 / 10,558 / 11,357 / 11,896 | 0.184 / 0.254 / 0.327 / 0.403 | 0.044 / 0.054 / 0.063 / 0.072 | 0.150 / 0.167 / 0.177 / 0.186 | 0.393 / 0.411 / 0.422 / 0.428 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 3,256 / 3,944 / 4,259 / 4,307 | 0.162 / 0.238 / 0.301 / 0.332 | 0.056 / 0.067 / 0.080 / 0.096 | 0.106 / 0.120 / 0.130 / 0.131 | 0.153 / 0.154 / 0.158 / 0.159 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 1,785 / 1,879 / 1,880 / 1,887 | 0.153 / 0.168 / 0.168 / 0.179 | 0.096 / 0.105 / 0.105 / 0.108 | 0.202 / 0.207 / 0.207 / 0.207 | 0.628 / 0.634 / 0.635 / 0.637 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 20,949 / 21,293 / 21,452 / 21,637 | 0.119 / 0.132 / 0.153 / 0.200 | 0.497 / 0.502 / 0.508 / 0.516 | 0.245 / 0.246 / 0.247 / 0.247 | 0.270 / 0.269 / 0.270 / 0.274 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 5,511 / 6,135 / 6,340 / 6,540 | 0.156 / 0.191 / 0.247 / 0.443 | 0.303 / 0.308 / 0.309 / 0.312 | 0.284 / 0.286 / 0.286 / 0.286 | 0.028 / 0.028 / 0.028 / 0.028 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 605 / 817 / 935 / 960 | 0.247 / 0.359 / 0.432 / 0.527 | 0.422 / 0.386 / 0.380 / 0.378 | 0.351 / 0.379 / 0.397 / 0.407 | 0.026 / 0.027 / 0.028 / 0.028 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

---

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.5590**

| class_name | GTs | num_match@0.5/1.0/2.0/4.0 | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | :---- | :---- | :---- | :---- |
| car | 54,217 | 40,773 / 46,131 / 48,671 / 49,739 | 0.657 / 0.787 / 0.840 / 0.863 | 0.755 / 0.828 / 0.855 / 0.866 | 0.227 / 0.189 / 0.163 / 0.155 |
| truck | 4,913 | 2,938 / 3,603 / 3,991 / 4,182 | 0.433 / 0.605 / 0.700 / 0.747 | 0.606 / 0.710 / 0.757 / 0.781 | 0.223 / 0.182 / 0.165 / 0.141 |
| bus | 2,116 | 1,374 / 1,777 / 1,950 / 1,994 | 0.516 / 0.753 / 0.876 / 0.900 | 0.645 / 0.798 / 0.865 / 0.881 | 0.329 / 0.140 / 0.140 / 0.110 |
| bicycle | 838 | 666 / 708 / 711 / 711 | 0.617 / 0.681 / 0.684 / 0.688 | 0.693 / 0.729 / 0.730 / 0.737 | 0.164 / 0.164 / 0.164 / 0.164 |
| pedestrian | 8,336 | 6,988 / 7,142 / 7,202 / 7,290 | 0.599 / 0.621 / 0.634 / 0.647 | 0.667 / 0.677 / 0.683 / 0.690 | 0.153 / 0.145 / 0.144 / 0.144 |
| traffic_cone | 2,632 | 1,574 / 1,688 / 1,748 / 1,850 | 0.215 / 0.254 / 0.277 / 0.338 | 0.405 / 0.430 / 0.449 / 0.484 | 0.153 / 0.153 / 0.126 / 0.134 |
| barrier | 622 | 180 / 276 / 305 / 321 | 0.057 / 0.195 / 0.227 / 0.244 | 0.268 / 0.401 / 0.429 / 0.440 | 0.140 / 0.121 / 0.116 / 0.116 |

<details>
<summary><strong>TP error — default (recall @0.10)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 5,963 / 5,963 / 5,963 / 5,963 | 0.159 / 0.185 / 0.203 / 0.219 | 0.112 / 0.145 / 0.167 / 0.171 | 0.157 / 0.161 / 0.162 / 0.162 | 0.158 / 0.158 / 0.158 / 0.158 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 540 / 540 / 540 / 540 | 0.191 / 0.240 / 0.292 / 0.338 | 0.034 / 0.040 / 0.046 / 0.048 | 0.152 / 0.163 / 0.170 / 0.175 | 0.438 / 0.449 / 0.453 / 0.458 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 232 / 232 / 232 / 232 | 0.168 / 0.223 / 0.268 / 0.281 | 0.117 / 0.109 / 0.113 / 0.119 | 0.119 / 0.126 / 0.133 / 0.133 | 0.131 / 0.131 / 0.130 / 0.130 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 92 / 92 / 92 / 92 | 0.172 / 0.187 / 0.188 / 0.200 | 0.114 / 0.121 / 0.120 / 0.123 | 0.199 / 0.201 / 0.201 / 0.201 | 0.691 / 0.685 / 0.686 / 0.689 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 916 / 916 / 916 / 916 | 0.119 / 0.129 / 0.148 / 0.192 | 0.569 / 0.574 / 0.579 / 0.587 | 0.230 / 0.231 / 0.231 / 0.231 | 0.291 / 0.290 / 0.292 / 0.297 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 289 / 289 / 289 / 289 | 0.165 / 0.201 / 0.289 / 0.670 | 0.269 / 0.283 / 0.286 / 0.308 | 0.290 / 0.290 / 0.291 / 0.292 | 0.038 / 0.038 / 0.038 / 0.039 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 68 / 68 / 68 / 68 | 0.240 / 0.385 / 0.440 / 0.537 | 0.297 / 0.260 / 0.257 / 0.255 | 0.309 / 0.350 / 0.357 / 0.361 | 0.034 / 0.034 / 0.033 / 0.033 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — medium (recall @0.40)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 22,228 / 22,228 / 22,228 / 22,228 | 0.171 / 0.204 / 0.228 / 0.250 | 0.138 / 0.177 / 0.205 / 0.210 | 0.163 / 0.167 / 0.168 / 0.169 | 0.164 / 0.164 / 0.163 / 0.163 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 2,014 / 2,014 / 2,014 / 2,014 | 0.204 / 0.269 / 0.334 / 0.393 | 0.046 / 0.054 / 0.061 / 0.064 | 0.163 / 0.174 / 0.182 / 0.189 | 0.465 / 0.481 / 0.484 / 0.492 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 867 / 867 / 867 / 867 | 0.187 / 0.263 / 0.322 / 0.341 | 0.089 / 0.087 / 0.097 / 0.108 | 0.125 / 0.134 / 0.142 / 0.143 | 0.141 / 0.137 / 0.135 / 0.135 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 343 / 343 / 343 / 343 | 0.180 / 0.202 / 0.204 / 0.223 | 0.142 / 0.149 / 0.148 / 0.153 | 0.199 / 0.202 / 0.202 / 0.201 | 0.711 / 0.704 / 0.704 / 0.709 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 3,417 / 3,417 / 3,417 / 3,417 | 0.124 / 0.136 / 0.159 / 0.215 | 0.590 / 0.596 / 0.602 / 0.612 | 0.230 / 0.231 / 0.231 / 0.232 | 0.315 / 0.313 / 0.314 / 0.320 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 1,079 / 1,079 / 1,079 / 1,079 | 0.175 / 0.218 / 0.328 / 0.718 | 0.348 / 0.355 / 0.359 / 0.379 | 0.297 / 0.297 / 0.296 / 0.297 | 0.038 / 0.038 / 0.038 / 0.038 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 255 / 255 / 255 | 1.000 / 0.398 / 0.489 / 0.692 | 1.000 / 0.332 / 0.312 / 0.307 | 1.000 / 0.409 / 0.415 / 0.424 | 1.000 / 0.036 / 0.035 / 0.035 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — optimal</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 37,866 / 42,472 / 44,630 / 45,417 | 0.182 / 0.229 / 0.269 / 0.309 | 0.167 / 0.219 / 0.256 / 0.266 | 0.168 / 0.173 / 0.175 / 0.176 | 0.167 / 0.169 / 0.170 / 0.170 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 2,598 / 3,145 / 3,407 / 3,593 | 0.206 / 0.281 / 0.355 / 0.432 | 0.047 / 0.061 / 0.072 / 0.081 | 0.165 / 0.181 / 0.190 / 0.200 | 0.477 / 0.495 / 0.499 / 0.513 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 1,175 / 1,604 / 1,740 / 1,815 | 0.193 / 0.302 / 0.379 / 0.440 | 0.087 / 0.090 / 0.118 / 0.151 | 0.126 / 0.142 / 0.151 / 0.153 | 0.136 / 0.136 / 0.133 / 0.139 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 524 / 551 / 552 / 557 | 0.179 / 0.201 / 0.203 / 0.230 | 0.141 / 0.147 / 0.147 / 0.154 | 0.197 / 0.202 / 0.202 / 0.202 | 0.701 / 0.690 / 0.692 / 0.696 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 5,196 / 5,378 / 5,448 / 5,510 | 0.124 / 0.136 / 0.159 / 0.216 | 0.574 / 0.584 / 0.590 / 0.601 | 0.230 / 0.230 / 0.231 / 0.231 | 0.316 / 0.319 / 0.321 / 0.327 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 1,014 / 1,077 / 1,237 / 1,290 | 0.166 / 0.203 / 0.318 / 0.694 | 0.293 / 0.300 / 0.335 / 0.360 | 0.291 / 0.291 / 0.293 / 0.294 | 0.038 / 0.038 / 0.038 / 0.039 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 136 / 209 / 227 / 233 | 0.241 / 0.390 / 0.465 / 0.548 | 0.322 / 0.295 / 0.284 / 0.278 | 0.319 / 0.355 / 0.365 / 0.371 | 0.035 / 0.034 / 0.034 / 0.034 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

---

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.4021**

| class_name | GTs | num_match@0.5/1.0/2.0/4.0 | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | :---- | :---- | :---- | :---- |
| car | 19,301 | 13,277 / 15,990 / 17,394 / 17,838 | 0.500 / 0.678 / 0.767 / 0.794 | 0.640 / 0.736 / 0.776 / 0.789 | 0.200 / 0.169 / 0.153 / 0.153 |
| truck | 2,906 | 1,247 / 1,799 / 2,243 / 2,429 | 0.212 / 0.427 / 0.622 / 0.697 | 0.430 / 0.589 / 0.701 / 0.743 | 0.191 / 0.144 / 0.144 / 0.107 |
| bus | 484 | 215 / 322 / 390 / 403 | 0.225 / 0.483 / 0.628 / 0.652 | 0.407 / 0.579 / 0.671 / 0.683 | 0.098 / 0.098 / 0.073 / 0.073 |
| bicycle | 291 | 211 / 238 / 242 / 242 | 0.395 / 0.517 / 0.525 / 0.529 | 0.575 / 0.641 / 0.641 / 0.641 | 0.182 / 0.182 / 0.182 / 0.182 |
| pedestrian | 2,564 | 1,978 / 2,019 / 2,045 / 2,066 | 0.407 / 0.420 / 0.429 / 0.437 | 0.540 / 0.547 / 0.549 / 0.552 | 0.121 / 0.129 / 0.123 / 0.129 |
| traffic_cone | 462 | 218 / 223 / 228 / 251 | 0.114 / 0.117 / 0.124 / 0.151 | 0.322 / 0.322 / 0.332 / 0.353 | 0.133 / 0.133 / 0.144 / 0.133 |
| barrier | 145 | 38 / 65 / 83 / 88 | 0.014 / 0.084 / 0.147 / 0.165 | 0.198 / 0.304 / 0.375 / 0.397 | 0.112 / 0.112 / 0.110 / 0.110 |

<details>
<summary><strong>TP error — default (recall @0.10)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 2,123 / 2,123 / 2,123 / 2,123 | 0.202 / 0.250 / 0.292 / 0.326 | 0.216 / 0.280 / 0.318 / 0.330 | 0.180 / 0.184 / 0.186 / 0.186 | 0.295 / 0.284 / 0.280 / 0.279 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 319 / 319 / 319 / 319 | 0.225 / 0.328 / 0.457 / 0.534 | 0.043 / 0.051 / 0.066 / 0.073 | 0.172 / 0.193 / 0.210 / 0.219 | 0.442 / 0.467 / 0.477 / 0.484 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 53 / 53 / 53 / 53 | 0.265 / 0.372 / 0.446 / 0.471 | 0.028 / 0.030 / 0.034 / 0.036 | 0.147 / 0.167 / 0.179 / 0.180 | 0.345 / 0.342 / 0.343 / 0.342 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 32 / 32 / 32 / 32 | 0.252 / 0.288 / 0.293 / 0.303 | 0.091 / 0.096 / 0.098 / 0.099 | 0.246 / 0.259 / 0.259 / 0.259 | 0.754 / 0.761 / 0.765 / 0.765 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 282 / 282 / 282 / 282 | 0.123 / 0.134 / 0.153 / 0.197 | 0.505 / 0.506 / 0.511 / 0.518 | 0.250 / 0.250 / 0.251 / 0.252 | 0.389 / 0.390 / 0.390 / 0.393 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 50 / 50 / 50 / 50 | 0.146 / 0.153 / 0.222 / 0.594 | 0.289 / 0.290 / 0.290 / 0.310 | 0.316 / 0.315 / 0.315 / 0.314 | 0.035 / 0.035 / 0.035 / 0.035 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 15 / 15 / 15 / 15 | 0.270 / 0.426 / 0.570 / 0.932 | 0.370 / 0.281 / 0.272 / 0.268 | 0.363 / 0.395 / 0.423 / 0.426 | 0.046 / 0.045 / 0.047 / 0.047 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — medium (recall @0.40)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 7,913 / 7,913 / 7,913 / 7,913 | 0.211 / 0.267 / 0.318 / 0.361 | 0.274 / 0.331 / 0.371 / 0.383 | 0.185 / 0.189 / 0.191 / 0.191 | 0.284 / 0.276 / 0.273 / 0.272 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 1,191 / 1,191 / 1,191 / 1,191 | 0.243 / 0.362 / 0.521 / 0.621 | 0.091 / 0.076 / 0.093 / 0.099 | 0.189 / 0.205 / 0.222 / 0.232 | 0.501 / 0.524 / 0.522 / 0.529 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 198 / 198 / 198 / 198 | 0.278 / 0.417 / 0.521 / 0.558 | 0.050 / 0.043 / 0.046 / 0.048 | 0.160 / 0.179 / 0.195 / 0.195 | 0.394 / 0.358 / 0.356 / 0.354 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 119 / 119 / 119 / 119 | 0.234 / 0.274 / 0.282 / 0.298 | 0.122 / 0.127 / 0.129 / 0.131 | 0.237 / 0.250 / 0.250 / 0.250 | 0.763 / 0.799 / 0.804 / 0.805 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 1,051 / 1,051 / 1,051 / 1,051 | 0.131 / 0.145 / 0.169 / 0.219 | 0.569 / 0.570 / 0.576 / 0.584 | 0.247 / 0.248 / 0.248 / 0.249 | 0.421 / 0.420 / 0.421 / 0.425 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 189 / 189 / 189 / 189 | 0.160 / 0.178 / 0.270 / 0.709 | 0.332 / 0.328 / 0.332 / 0.362 | 0.317 / 0.316 / 0.315 / 0.313 | 0.040 / 0.039 / 0.039 / 0.039 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 59 / 59 / 59 | 1.000 / 0.450 / 0.638 / 0.970 | 1.000 / 0.271 / 0.250 / 0.246 | 1.000 / 0.444 / 0.461 / 0.465 | 1.000 / 0.048 / 0.050 / 0.050 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — optimal</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 11,411 / 13,677 / 14,775 / 15,007 | 0.213 / 0.279 / 0.340 / 0.390 | 0.299 / 0.372 / 0.416 / 0.429 | 0.186 / 0.192 / 0.194 / 0.194 | 0.278 / 0.273 / 0.273 / 0.272 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 1,065 / 1,541 / 1,834 / 2,047 | 0.234 / 0.364 / 0.529 / 0.666 | 0.057 / 0.073 / 0.097 / 0.114 | 0.180 / 0.206 / 0.224 / 0.240 | 0.494 / 0.516 / 0.515 / 0.550 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 164 / 233 / 291 / 296 | 0.279 / 0.412 / 0.546 / 0.582 | 0.044 / 0.044 / 0.050 / 0.052 | 0.155 / 0.178 / 0.198 / 0.198 | 0.340 / 0.321 / 0.363 / 0.358 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 147 / 164 / 164 / 164 | 0.233 / 0.270 / 0.270 / 0.270 | 0.110 / 0.119 / 0.119 / 0.119 | 0.239 / 0.252 / 0.252 / 0.252 | 0.756 / 0.786 / 0.786 / 0.786 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 1,358 / 1,329 / 1,371 / 1,342 | 0.128 / 0.139 / 0.157 / 0.196 | 0.544 / 0.532 / 0.546 / 0.545 | 0.249 / 0.249 / 0.248 / 0.249 | 0.417 / 0.409 / 0.417 / 0.417 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 135 / 135 / 133 / 148 | 0.142 / 0.142 / 0.216 / 0.596 | 0.300 / 0.300 / 0.303 / 0.308 | 0.313 / 0.313 / 0.312 / 0.313 | 0.035 / 0.035 / 0.035 / 0.035 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 26 / 40 / 50 / 53 | 0.274 / 0.409 / 0.604 / 0.937 | 0.312 / 0.258 / 0.259 / 0.276 | 0.370 / 0.387 / 0.427 / 0.429 | 0.046 / 0.045 / 0.048 / 0.048 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.6463**

| class_name | GTs | num_match@0.5/1.0/2.0/4.0 | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | :---- | :---- | :---- | :---- |
| car | 134,456 | 107,307 / 117,848 / 123,019 / 125,254 | 0.730 / 0.828 / 0.873 / 0.893 | 0.804 / 0.858 / 0.879 / 0.887 | 0.240 / 0.200 / 0.165 / 0.162 |
| truck | 14,900 | 9,814 / 11,683 / 12,829 / 13,364 | 0.527 / 0.690 / 0.786 / 0.828 | 0.667 / 0.766 / 0.817 / 0.841 | 0.225 / 0.182 / 0.170 / 0.145 |
| bus | 4,970 | 3,654 / 4,321 / 4,639 / 4,711 | 0.643 / 0.809 / 0.902 / 0.916 | 0.740 / 0.842 / 0.892 / 0.901 | 0.318 / 0.140 / 0.103 / 0.102 |
| bicycle | 2,486 | 2,153 / 2,228 / 2,239 / 2,239 | 0.755 / 0.796 / 0.801 / 0.802 | 0.789 / 0.809 / 0.809 / 0.812 | 0.186 / 0.164 / 0.164 / 0.164 |
| pedestrian | 29,102 | 25,516 / 26,078 / 26,291 / 26,512 | 0.728 / 0.752 / 0.765 / 0.778 | 0.748 / 0.761 / 0.766 / 0.773 | 0.153 / 0.153 / 0.153 / 0.153 |
| traffic_cone | 11,344 | 7,980 / 8,516 / 8,749 / 9,099 | 0.342 / 0.388 / 0.410 / 0.448 | 0.509 / 0.538 / 0.551 / 0.575 | 0.177 / 0.156 / 0.153 / 0.157 |
| barrier | 2,117 | 737 / 1,055 / 1,164 / 1,216 | 0.104 / 0.236 / 0.275 / 0.290 | 0.323 / 0.439 / 0.467 / 0.479 | 0.151 / 0.155 / 0.115 / 0.115 |

<details>
<summary><strong>TP error — default (recall @0.10)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 14,790 / 14,790 / 14,790 / 14,790 | 0.129 / 0.145 / 0.156 / 0.167 | 0.064 / 0.082 / 0.095 / 0.099 | 0.129 / 0.133 / 0.134 / 0.135 | 0.151 / 0.153 / 0.154 / 0.155 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 1,639 / 1,639 / 1,639 / 1,639 | 0.161 / 0.202 / 0.244 / 0.276 | 0.029 / 0.033 / 0.037 / 0.039 | 0.134 / 0.144 / 0.150 / 0.154 | 0.360 / 0.375 / 0.382 / 0.385 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 546 / 546 / 546 / 546 | 0.132 / 0.163 / 0.196 / 0.203 | 0.057 / 0.059 / 0.061 / 0.064 | 0.094 / 0.100 / 0.107 / 0.107 | 0.126 / 0.130 / 0.131 / 0.131 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 273 / 273 / 273 / 273 | 0.146 / 0.154 / 0.155 / 0.158 | 0.079 / 0.081 / 0.082 / 0.083 | 0.195 / 0.197 / 0.197 / 0.197 | 0.593 / 0.594 / 0.595 / 0.596 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 3,201 / 3,201 / 3,201 / 3,201 | 0.110 / 0.118 / 0.133 / 0.160 | 0.455 / 0.459 / 0.464 / 0.469 | 0.238 / 0.239 / 0.239 / 0.240 | 0.245 / 0.245 / 0.246 / 0.248 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 1,247 / 1,247 / 1,247 / 1,247 | 0.152 / 0.181 / 0.225 / 0.389 | 0.292 / 0.291 / 0.290 / 0.293 | 0.283 / 0.283 / 0.283 / 0.284 | 0.027 / 0.027 / 0.027 / 0.027 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 232 / 232 / 232 / 232 | 0.235 / 0.339 / 0.389 / 0.459 | 0.390 / 0.353 / 0.349 / 0.349 | 0.319 / 0.353 / 0.362 / 0.370 | 0.026 / 0.028 / 0.028 / 0.028 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — medium (recall @0.40)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 55,126 / 55,126 / 55,126 / 55,126 | 0.143 / 0.165 / 0.181 / 0.196 | 0.084 / 0.109 / 0.126 / 0.131 | 0.139 / 0.143 / 0.144 / 0.145 | 0.164 / 0.165 / 0.166 / 0.166 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 6,109 / 6,109 / 6,109 / 6,109 | 0.177 / 0.231 / 0.287 / 0.332 | 0.037 / 0.042 / 0.047 / 0.049 | 0.144 / 0.155 / 0.164 / 0.169 | 0.386 / 0.403 / 0.410 / 0.414 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 2,037 / 2,037 / 2,037 / 2,037 | 0.149 / 0.192 / 0.229 / 0.239 | 0.056 / 0.060 / 0.064 / 0.069 | 0.102 / 0.108 / 0.115 / 0.115 | 0.149 / 0.150 / 0.149 / 0.150 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 1,019 / 1,019 / 1,019 / 1,019 | 0.151 / 0.161 / 0.163 / 0.168 | 0.090 / 0.093 / 0.094 / 0.096 | 0.201 / 0.204 / 0.204 / 0.204 | 0.618 / 0.619 / 0.621 / 0.622 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 11,931 / 11,931 / 11,931 / 11,931 | 0.115 / 0.126 / 0.145 / 0.183 | 0.487 / 0.491 / 0.496 / 0.503 | 0.243 / 0.244 / 0.245 / 0.245 | 0.261 / 0.260 / 0.261 / 0.264 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 4,651 / 4,651 / 4,651 / 4,651 | 0.161 / 0.197 / 0.257 / 0.463 | 0.318 / 0.316 / 0.315 / 0.318 | 0.288 / 0.287 / 0.287 / 0.288 | 0.028 / 0.028 / 0.028 / 0.028 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 867 / 867 / 867 | 1.000 / 0.374 / 0.445 / 0.567 | 1.000 / 0.395 / 0.385 / 0.383 | 1.000 / 0.407 / 0.414 / 0.425 | 1.000 / 0.028 / 0.029 / 0.029 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — optimal</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 100,294 / 109,159 / 113,989 / 115,141 | 0.159 / 0.194 / 0.227 / 0.255 | 0.120 / 0.160 / 0.190 / 0.196 | 0.148 / 0.154 / 0.156 / 0.157 | 0.172 / 0.175 / 0.178 / 0.178 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 8,931 / 10,558 / 11,357 / 11,896 | 0.184 / 0.254 / 0.327 / 0.403 | 0.044 / 0.054 / 0.063 / 0.072 | 0.150 / 0.167 / 0.177 / 0.186 | 0.393 / 0.411 / 0.422 / 0.428 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 3,256 / 3,944 / 4,259 / 4,307 | 0.162 / 0.238 / 0.301 / 0.332 | 0.056 / 0.067 / 0.080 / 0.096 | 0.106 / 0.120 / 0.130 / 0.131 | 0.153 / 0.154 / 0.158 / 0.159 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 1,785 / 1,879 / 1,880 / 1,887 | 0.153 / 0.168 / 0.168 / 0.179 | 0.096 / 0.105 / 0.105 / 0.108 | 0.202 / 0.207 / 0.207 / 0.207 | 0.628 / 0.634 / 0.635 / 0.637 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 20,949 / 21,293 / 21,452 / 21,637 | 0.119 / 0.132 / 0.153 / 0.200 | 0.497 / 0.502 / 0.508 / 0.516 | 0.245 / 0.246 / 0.247 / 0.247 | 0.270 / 0.269 / 0.270 / 0.274 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 5,511 / 6,135 / 6,340 / 6,540 | 0.156 / 0.191 / 0.247 / 0.443 | 0.303 / 0.308 / 0.309 / 0.312 | 0.284 / 0.286 / 0.286 / 0.286 | 0.028 / 0.028 / 0.028 / 0.028 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 605 / 817 / 935 / 960 | 0.247 / 0.359 / 0.432 / 0.527 | 0.422 / 0.386 / 0.380 / 0.378 | 0.351 / 0.379 / 0.397 / 0.407 | 0.026 / 0.027 / 0.028 / 0.028 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

---

**JPNTaxi_Gen2**: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (10,687 frames)


**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.6595**

| class_name | GTs | num_match@0.5/1.0/2.0/4.0 | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | :---- | :---- | :---- | :---- |
| car | 50,954 | 45,644 / 48,317 / 49,034 / 49,324 | 0.858 / 0.925 / 0.946 / 0.948 | 0.900 / 0.942 / 0.948 / 0.950 | 0.249 / 0.199 / 0.191 / 0.161 |
| truck | 18,624 | 15,109 / 16,753 / 17,761 / 18,114 | 0.729 / 0.852 / 0.930 / 0.955 | 0.818 / 0.891 / 0.934 / 0.948 | 0.299 / 0.215 / 0.181 / 0.180 |
| bus | 3,853 | 3,182 / 3,437 / 3,532 / 3,538 | 0.769 / 0.850 / 0.875 / 0.875 | 0.811 / 0.863 / 0.882 / 0.883 | 0.047 / 0.038 / 0.036 / 0.036 |
| bicycle | 3,768 | 3,351 / 3,417 / 3,418 / 3,420 | 0.764 / 0.783 / 0.783 / 0.784 | 0.796 / 0.802 / 0.803 / 0.803 | 0.243 / 0.243 / 0.243 / 0.243 |
| pedestrian | 70,699 | 66,162 / 67,089 / 67,449 / 67,816 | 0.886 / 0.901 / 0.911 / 0.917 | 0.875 / 0.886 / 0.891 / 0.897 | 0.148 / 0.148 / 0.146 / 0.148 |
| traffic_cone | 12,525 | 7,726 / 8,331 / 8,729 / 8,980 | 0.258 / 0.290 / 0.323 / 0.354 | 0.457 / 0.480 / 0.503 / 0.524 | 0.138 / 0.135 / 0.135 / 0.140 |
| barrier | 2,009 | 0 / 0 / 0 / 0 | 0.000 / 0.000 / 0.000 / 0.000 | nan / nan / nan / nan | nan / nan / nan / nan |

<details>
<summary><strong>TP error — default (recall @0.10)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 5,604 / 5,604 / 5,604 / 5,604 | 0.104 / 0.112 / 0.114 / 0.119 | 0.021 / 0.021 / 0.022 / 0.022 | 0.103 / 0.105 / 0.106 / 0.106 | 0.349 / 0.353 / 0.356 / 0.356 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 2,048 / 2,048 / 2,048 / 2,048 | 0.135 / 0.162 / 0.183 / 0.197 | 0.016 / 0.016 / 0.017 / 0.017 | 0.115 / 0.121 / 0.126 / 0.127 | 0.406 / 0.412 / 0.423 / 0.427 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 423 / 423 / 423 / 423 | 0.154 / 0.163 / 0.168 / 0.169 | 0.042 / 0.041 / 0.041 / 0.041 | 0.105 / 0.109 / 0.110 / 0.110 | 0.303 / 0.315 / 0.317 / 0.318 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 414 / 414 / 414 / 414 | 0.126 / 0.129 / 0.129 / 0.131 | 0.096 / 0.098 / 0.098 / 0.099 | 0.183 / 0.184 / 0.184 / 0.184 | 0.878 / 0.878 / 0.879 / 0.880 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 7,776 / 7,776 / 7,776 / 7,776 | 0.106 / 0.109 / 0.113 / 0.122 | 0.249 / 0.250 / 0.251 / 0.254 | 0.227 / 0.228 / 0.228 / 0.228 | 0.426 / 0.427 / 0.427 / 0.428 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 1,377 / 1,377 / 1,377 / 1,377 | 0.145 / 0.170 / 0.260 / 0.408 | 0.702 / 0.711 / 0.715 / 0.702 | 0.300 / 0.302 / 0.302 / 0.301 | 0.100 / 0.100 / 0.099 / 0.098 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — medium (recall @0.40)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 20,891 / 20,891 / 20,891 / 20,891 | 0.110 / 0.120 / 0.123 / 0.130 | 0.022 / 0.024 / 0.025 / 0.025 | 0.106 / 0.109 / 0.109 / 0.109 | 0.380 / 0.384 / 0.388 / 0.387 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 7,635 / 7,635 / 7,635 / 7,635 | 0.143 / 0.174 / 0.202 / 0.223 | 0.017 / 0.017 / 0.018 / 0.018 | 0.118 / 0.125 / 0.130 / 0.133 | 0.474 / 0.478 / 0.490 / 0.494 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 1,579 / 1,579 / 1,579 / 1,579 | 0.157 / 0.171 / 0.180 / 0.181 | 0.034 / 0.034 / 0.034 / 0.034 | 0.108 / 0.113 / 0.116 / 0.116 | 0.359 / 0.369 / 0.371 / 0.372 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 1,544 / 1,544 / 1,544 / 1,544 | 0.131 / 0.135 / 0.135 / 0.138 | 0.097 / 0.099 / 0.099 / 0.100 | 0.190 / 0.191 / 0.191 / 0.191 | 0.819 / 0.822 / 0.824 / 0.824 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 28,986 / 28,986 / 28,986 / 28,986 | 0.111 / 0.115 / 0.121 / 0.135 | 0.252 / 0.253 / 0.255 / 0.259 | 0.230 / 0.230 / 0.230 / 0.230 | 0.439 / 0.439 / 0.439 / 0.441 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 5,135 / 5,135 / 5,135 / 5,135 | 0.161 / 0.200 / 0.316 / 0.500 | 0.758 / 0.759 / 0.763 / 0.751 | 0.307 / 0.309 / 0.310 / 0.309 | 0.098 / 0.097 / 0.097 / 0.097 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — optimal</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 44,376 / 46,850 / 47,216 / 47,606 | 0.123 / 0.149 / 0.156 / 0.170 | 0.031 / 0.036 / 0.037 / 0.040 | 0.114 / 0.120 / 0.121 / 0.121 | 0.432 / 0.444 / 0.452 / 0.457 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 14,516 / 16,104 / 17,017 / 17,294 | 0.154 / 0.204 / 0.260 / 0.301 | 0.018 / 0.020 / 0.022 / 0.023 | 0.125 / 0.136 / 0.146 / 0.151 | 0.532 / 0.559 / 0.590 / 0.602 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 3,080 / 3,354 / 3,439 / 3,443 | 0.166 / 0.201 / 0.224 / 0.228 | 0.033 / 0.034 / 0.034 / 0.034 | 0.116 / 0.132 / 0.139 / 0.139 | 0.414 / 0.419 / 0.419 / 0.422 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 2,874 / 2,897 / 2,900 / 2,900 | 0.134 / 0.138 / 0.139 / 0.141 | 0.105 / 0.105 / 0.105 / 0.106 | 0.193 / 0.194 / 0.194 / 0.194 | 0.806 / 0.809 / 0.810 / 0.810 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 59,982 / 60,714 / 61,271 / 61,531 | 0.118 / 0.127 / 0.140 / 0.172 | 0.271 / 0.274 / 0.279 / 0.286 | 0.235 / 0.235 / 0.236 / 0.236 | 0.451 / 0.451 / 0.451 / 0.454 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 5,720 / 6,079 / 6,376 / 6,515 | 0.156 / 0.190 / 0.300 / 0.475 | 0.748 / 0.753 / 0.755 / 0.743 | 0.305 / 0.308 / 0.307 / 0.306 | 0.097 / 0.097 / 0.096 / 0.096 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan |

</details>

---

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.5440**

| class_name | GTs | num_match@0.5/1.0/2.0/4.0 | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | :---- | :---- | :---- | :---- |
| car | 41,196 | 33,758 / 37,076 / 38,510 / 38,977 | 0.744 / 0.836 / 0.874 / 0.885 | 0.802 / 0.853 / 0.869 / 0.873 | 0.229 / 0.170 / 0.161 / 0.161 |
| truck | 22,942 | 13,827 / 16,980 / 19,337 / 20,306 | 0.471 / 0.638 / 0.771 / 0.817 | 0.626 / 0.731 / 0.809 / 0.833 | 0.224 / 0.192 / 0.160 / 0.129 |
| bus | 3,033 | 1,552 / 2,141 / 2,485 / 2,543 | 0.286 / 0.506 / 0.667 / 0.693 | 0.455 / 0.616 / 0.712 / 0.727 | 0.078 / 0.069 / 0.074 / 0.074 |
| bicycle | 2,310 | 1,862 / 1,959 / 1,963 / 1,977 | 0.584 / 0.633 / 0.633 / 0.643 | 0.665 / 0.689 / 0.689 / 0.695 | 0.160 / 0.142 / 0.142 / 0.142 |
| pedestrian | 36,881 | 33,364 / 33,924 / 34,094 / 34,294 | 0.764 / 0.779 / 0.790 / 0.799 | 0.772 / 0.781 / 0.785 / 0.791 | 0.159 / 0.148 / 0.148 / 0.148 |
| traffic_cone | 7,183 | 4,785 / 4,935 / 5,020 / 5,158 | 0.317 / 0.337 / 0.366 / 0.399 | 0.492 / 0.504 / 0.523 / 0.541 | 0.135 / 0.135 / 0.125 / 0.125 |
| barrier | 1,847 | 0 / 0 / 0 / 0 | 0.000 / 0.000 / 0.000 / 0.000 | nan / nan / nan / nan | nan / nan / nan / nan |

<details>
<summary><strong>TP error — default (recall @0.10)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 4,531 / 4,531 / 4,531 / 4,531 | 0.141 / 0.154 / 0.162 / 0.171 | 0.034 / 0.040 / 0.043 / 0.045 | 0.137 / 0.139 / 0.140 / 0.141 | 0.319 / 0.346 / 0.360 / 0.366 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 2,523 / 2,523 / 2,523 / 2,523 | 0.197 / 0.237 / 0.300 / 0.332 | 0.024 / 0.028 / 0.032 / 0.035 | 0.146 / 0.154 / 0.164 / 0.168 | 0.509 / 0.562 / 0.605 / 0.624 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 333 / 333 / 333 / 333 | 0.224 / 0.314 / 0.405 / 0.440 | 0.030 / 0.036 / 0.039 / 0.040 | 0.130 / 0.143 / 0.155 / 0.156 | 0.272 / 0.289 / 0.289 / 0.290 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 254 / 254 / 254 / 254 | 0.186 / 0.198 / 0.199 / 0.245 | 0.094 / 0.094 / 0.094 / 0.091 | 0.230 / 0.231 / 0.231 / 0.229 | 0.847 / 0.861 / 0.861 / 0.861 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 4,056 / 4,056 / 4,056 / 4,056 | 0.116 / 0.121 / 0.133 / 0.162 | 0.296 / 0.299 / 0.303 / 0.307 | 0.261 / 0.262 / 0.262 / 0.262 | 0.464 / 0.464 / 0.465 / 0.468 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 790 / 790 / 790 / 790 | 0.164 / 0.177 / 0.305 / 0.502 | 0.490 / 0.497 / 0.497 / 0.493 | 0.234 / 0.235 / 0.234 / 0.233 | 0.108 / 0.108 / 0.107 / 0.107 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — medium (recall @0.40)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 16,890 / 16,890 / 16,890 / 16,890 | 0.151 / 0.169 / 0.181 / 0.193 | 0.041 / 0.049 / 0.053 / 0.056 | 0.142 / 0.145 / 0.146 / 0.147 | 0.379 / 0.410 / 0.429 / 0.435 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 9,406 / 9,406 / 9,406 / 9,406 | 0.201 / 0.261 / 0.346 / 0.394 | 0.031 / 0.036 / 0.041 / 0.044 | 0.151 / 0.162 / 0.175 / 0.181 | 0.612 / 0.669 / 0.712 / 0.735 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 1,243 / 1,243 / 1,243 / 1,243 | 0.244 / 0.365 / 0.483 / 0.537 | 0.055 / 0.052 / 0.052 / 0.054 | 0.137 / 0.153 / 0.166 / 0.168 | 0.375 / 0.347 / 0.337 / 0.337 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 947 / 947 / 947 / 947 | 0.188 / 0.205 / 0.207 / 0.264 | 0.116 / 0.115 / 0.115 / 0.109 | 0.238 / 0.238 / 0.238 / 0.237 | 0.884 / 0.895 / 0.896 / 0.899 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 15,121 / 15,121 / 15,121 / 15,121 | 0.121 / 0.128 / 0.142 / 0.182 | 0.314 / 0.318 / 0.323 / 0.328 | 0.262 / 0.263 / 0.263 / 0.263 | 0.486 / 0.485 / 0.487 / 0.491 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 2,945 / 2,945 / 2,945 / 2,945 | 0.171 / 0.188 / 0.318 / 0.540 | 0.468 / 0.477 / 0.474 / 0.473 | 0.232 / 0.234 / 0.234 / 0.233 | 0.102 / 0.102 / 0.102 / 0.102 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — optimal</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 30,777 / 33,945 / 34,775 / 34,956 | 0.164 / 0.200 / 0.222 / 0.240 | 0.054 / 0.071 / 0.078 / 0.080 | 0.149 / 0.155 / 0.156 / 0.157 | 0.447 / 0.505 / 0.535 / 0.541 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 12,711 / 15,140 / 17,099 / 18,052 | 0.205 / 0.278 / 0.393 / 0.478 | 0.033 / 0.043 / 0.054 / 0.063 | 0.154 / 0.168 / 0.188 / 0.200 | 0.655 / 0.740 / 0.812 / 0.867 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 1,191 / 1,652 / 1,886 / 1,924 | 0.238 / 0.367 / 0.492 / 0.552 | 0.036 / 0.052 / 0.052 / 0.055 | 0.132 / 0.153 / 0.167 / 0.168 | 0.297 / 0.341 / 0.345 / 0.344 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 1,384 / 1,483 / 1,484 / 1,496 | 0.188 / 0.203 / 0.205 / 0.265 | 0.111 / 0.114 / 0.114 / 0.111 | 0.239 / 0.241 / 0.241 / 0.239 | 0.885 / 0.894 / 0.895 / 0.899 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 27,185 / 28,060 / 28,214 / 28,437 | 0.123 / 0.134 / 0.150 / 0.199 | 0.327 / 0.335 / 0.340 / 0.348 | 0.263 / 0.264 / 0.264 / 0.265 | 0.500 / 0.504 / 0.505 / 0.510 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 3,298 / 3,377 / 3,633 / 3,756 | 0.166 / 0.180 / 0.314 / 0.523 | 0.475 / 0.481 / 0.478 / 0.475 | 0.234 / 0.236 / 0.235 / 0.235 | 0.103 / 0.103 / 0.102 / 0.102 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan |

</details>

---

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.3700**

| class_name | GTs | num_match@0.5/1.0/2.0/4.0 | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | :---- | :---- | :---- | :---- |
| car | 17,510 | 13,179 / 15,034 / 15,801 / 16,115 | 0.586 / 0.706 / 0.755 / 0.770 | 0.663 / 0.735 / 0.759 / 0.764 | 0.164 / 0.129 / 0.125 / 0.125 |
| truck | 14,707 | 6,206 / 9,176 / 11,392 / 12,747 | 0.207 / 0.459 / 0.642 / 0.755 | 0.436 / 0.622 / 0.732 / 0.786 | 0.185 / 0.185 / 0.152 / 0.124 |
| bus | 2,997 | 1,331 / 1,867 / 2,079 / 2,176 | 0.168 / 0.369 / 0.448 / 0.487 | 0.412 / 0.538 / 0.581 / 0.601 | 0.049 / 0.046 / 0.046 / 0.046 |
| bicycle | 566 | 386 / 433 / 446 / 450 | 0.227 / 0.284 / 0.330 / 0.342 | 0.417 / 0.461 / 0.484 / 0.491 | 0.132 / 0.132 / 0.132 / 0.132 |
| pedestrian | 16,580 | 14,554 / 14,780 / 14,883 / 15,028 | 0.679 / 0.697 / 0.704 / 0.720 | 0.712 / 0.720 / 0.725 / 0.734 | 0.136 / 0.136 / 0.136 / 0.136 |
| traffic_cone | 769 | 236 / 251 / 256 / 273 | 0.003 / 0.005 / 0.007 / 0.010 | 0.142 / 0.150 / 0.155 / 0.168 | 0.102 / 0.102 / 0.148 / 0.154 |
| barrier | 566 | 0 / 0 / 0 / 0 | 0.000 / 0.000 / 0.000 / 0.000 | nan / nan / nan / nan | nan / nan / nan / nan |

<details>
<summary><strong>TP error — default (recall @0.10)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 1,926 / 1,926 / 1,926 / 1,926 | 0.189 / 0.215 / 0.237 / 0.254 | 0.053 / 0.066 / 0.071 / 0.073 | 0.183 / 0.187 / 0.189 / 0.189 | 0.568 / 0.597 / 0.617 / 0.622 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 1,617 / 1,617 / 1,617 / 1,617 | 0.239 / 0.347 / 0.444 / 0.602 | 0.030 / 0.032 / 0.036 / 0.042 | 0.179 / 0.193 / 0.203 / 0.223 | 0.621 / 0.620 / 0.643 / 0.653 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 329 / 329 / 329 / 329 | 0.252 / 0.359 / 0.439 / 0.498 | 0.027 / 0.033 / 0.036 / 0.041 | 0.135 / 0.142 / 0.155 / 0.162 | 0.188 / 0.219 / 0.248 / 0.256 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 62 / 62 / 62 / 62 | 0.188 / 0.233 / 0.295 / 0.376 | 0.273 / 0.256 / 0.247 / 0.266 | 0.236 / 0.245 / 0.266 / 0.264 | 0.925 / 0.985 / 1.007 / 0.996 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 1,823 / 1,823 / 1,823 / 1,823 | 0.133 / 0.140 / 0.150 / 0.206 | 0.245 / 0.248 / 0.252 / 0.261 | 0.283 / 0.283 / 0.283 / 0.284 | 0.561 / 0.562 / 0.563 / 0.568 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 84 / 84 / 84 / 84 | 0.214 / 0.252 / 0.380 / 0.734 | 1.068 / 1.069 / 1.057 / 1.038 | 0.369 / 0.369 / 0.368 / 0.363 | 0.191 / 0.190 / 0.192 / 0.196 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — medium (recall @0.40)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 7,179 / 7,179 / 7,179 / 7,179 | 0.202 / 0.238 / 0.266 / 0.290 | 0.066 / 0.085 / 0.090 / 0.093 | 0.187 / 0.192 / 0.194 / 0.194 | 0.633 / 0.667 / 0.691 / 0.698 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 6,029 / 6,029 / 6,029 / 6,029 | 0.244 / 0.378 / 0.512 / 0.679 | 0.068 / 0.045 / 0.048 / 0.055 | 0.191 / 0.202 / 0.214 / 0.235 | 0.768 / 0.733 / 0.741 / 0.752 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 1,228 / 1,228 / 1,228 / 1,228 | 0.270 / 0.368 / 0.460 / 0.525 | 0.030 / 0.037 / 0.041 / 0.048 | 0.141 / 0.148 / 0.163 / 0.172 | 0.198 / 0.254 / 0.284 / 0.293 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 232 / 232 / 232 / 232 | 0.194 / 0.249 / 0.291 / 0.366 | 0.306 / 0.275 / 0.267 / 0.279 | 0.224 / 0.234 / 0.248 / 0.246 | 0.939 / 1.037 / 1.056 / 1.050 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 6,797 / 6,797 / 6,797 / 6,797 | 0.139 / 0.148 / 0.162 / 0.233 | 0.281 / 0.284 / 0.288 / 0.300 | 0.280 / 0.280 / 0.280 / 0.281 | 0.574 / 0.574 / 0.576 / 0.582 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — optimal</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 10,800 / 12,644 / 13,151 / 13,260 | 0.205 / 0.254 / 0.291 / 0.316 | 0.068 / 0.095 / 0.102 / 0.105 | 0.187 / 0.194 / 0.196 / 0.196 | 0.661 / 0.706 / 0.737 / 0.745 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 5,665 / 8,075 / 9,757 / 10,776 | 0.241 / 0.377 / 0.531 / 0.726 | 0.043 / 0.045 / 0.053 / 0.068 | 0.185 / 0.204 / 0.220 / 0.244 | 0.716 / 0.746 / 0.803 / 0.839 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 1,062 / 1,422 / 1,542 / 1,587 | 0.253 / 0.354 / 0.439 / 0.498 | 0.032 / 0.035 / 0.039 / 0.044 | 0.132 / 0.142 / 0.156 / 0.163 | 0.194 / 0.240 / 0.273 / 0.278 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 221 / 245 / 257 / 261 | 0.185 / 0.238 / 0.286 / 0.362 | 0.251 / 0.234 / 0.225 / 0.256 | 0.229 / 0.237 / 0.254 / 0.252 | 0.957 / 1.037 / 1.058 / 1.044 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 11,283 / 11,409 / 11,480 / 11,627 | 0.141 / 0.150 / 0.164 / 0.243 | 0.284 / 0.287 / 0.291 / 0.307 | 0.280 / 0.280 / 0.280 / 0.281 | 0.578 / 0.577 / 0.578 / 0.586 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 167 / 177 / 127 / 133 | 0.210 / 0.249 / 0.392 / 0.736 | 1.018 / 1.021 / 1.158 / 1.135 | 0.365 / 0.365 / 0.381 / 0.374 | 0.190 / 0.191 / 0.198 / 0.204 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan |

</details>

---

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.5785**

| class_name | GTs | num_match@0.5/1.0/2.0/4.0 | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | :---- | :---- | :---- | :---- |
| car | 109,660 | 92,745 / 100,651 / 103,627 / 104,735 | 0.783 / 0.872 / 0.902 / 0.913 | 0.829 / 0.878 / 0.890 / 0.894 | 0.229 / 0.164 / 0.161 / 0.159 |
| truck | 56,273 | 35,229 / 43,057 / 48,764 / 51,517 | 0.495 / 0.675 / 0.803 / 0.862 | 0.647 / 0.762 / 0.837 / 0.865 | 0.249 / 0.192 / 0.166 / 0.136 |
| bus | 9,883 | 6,077 / 7,470 / 8,128 / 8,289 | 0.457 / 0.619 / 0.703 / 0.721 | 0.586 / 0.693 / 0.741 / 0.752 | 0.056 / 0.049 / 0.049 / 0.049 |
| bicycle | 6,644 | 5,610 / 5,822 / 5,840 / 5,859 | 0.666 / 0.697 / 0.701 / 0.705 | 0.723 / 0.736 / 0.738 / 0.740 | 0.202 / 0.198 / 0.202 / 0.202 |
| pedestrian | 124,160 | 114,192 / 115,885 / 116,521 / 117,246 | 0.831 / 0.851 / 0.856 / 0.868 | 0.824 / 0.833 / 0.839 / 0.845 | 0.148 / 0.148 / 0.148 / 0.148 |
| traffic_cone | 20,477 | 12,760 / 13,531 / 14,021 / 14,440 | 0.261 / 0.288 / 0.318 / 0.349 | 0.455 / 0.473 / 0.494 / 0.514 | 0.135 / 0.135 / 0.135 / 0.138 |
| barrier | 4,422 | 0 / 0 / 0 / 0 | 0.000 / 0.000 / 0.000 / 0.000 | nan / nan / nan / nan | nan / nan / nan / nan |

<details>
<summary><strong>TP error — default (recall @0.10)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 12,062 / 12,062 / 12,062 / 12,062 | 0.122 / 0.134 / 0.140 / 0.148 | 0.027 / 0.031 / 0.033 / 0.034 | 0.119 / 0.122 / 0.123 / 0.124 | 0.361 / 0.374 / 0.383 / 0.386 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 6,190 / 6,190 / 6,190 / 6,190 | 0.171 / 0.216 / 0.267 / 0.315 | 0.020 / 0.022 / 0.025 / 0.027 | 0.134 / 0.144 / 0.152 / 0.159 | 0.476 / 0.504 / 0.534 / 0.549 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 1,087 / 1,087 / 1,087 / 1,087 | 0.178 / 0.225 / 0.268 / 0.286 | 0.034 / 0.036 / 0.038 / 0.038 | 0.115 / 0.123 / 0.130 / 0.131 | 0.313 / 0.324 / 0.330 / 0.331 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 730 / 730 / 730 / 730 | 0.142 / 0.149 / 0.151 / 0.165 | 0.100 / 0.101 / 0.101 / 0.102 | 0.195 / 0.197 / 0.197 / 0.197 | 0.841 / 0.848 / 0.849 / 0.850 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 13,657 / 13,657 / 13,657 / 13,657 | 0.111 / 0.115 / 0.121 / 0.141 | 0.260 / 0.262 / 0.264 / 0.269 | 0.239 / 0.239 / 0.239 / 0.240 | 0.445 / 0.446 / 0.446 / 0.449 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 2,252 / 2,252 / 2,252 / 2,252 | 0.153 / 0.175 / 0.279 / 0.453 | 0.637 / 0.645 / 0.648 / 0.638 | 0.279 / 0.281 / 0.281 / 0.280 | 0.103 / 0.103 / 0.102 / 0.102 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — medium (recall @0.40)</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 44,960 / 44,960 / 44,960 / 44,960 | 0.132 / 0.149 / 0.157 / 0.168 | 0.032 / 0.037 / 0.039 / 0.041 | 0.126 / 0.129 / 0.131 / 0.131 | 0.401 / 0.418 / 0.429 / 0.433 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 23,071 / 23,071 / 23,071 / 23,071 | 0.182 / 0.243 / 0.313 / 0.378 | 0.024 / 0.027 / 0.030 / 0.033 | 0.141 / 0.153 / 0.163 / 0.172 | 0.564 / 0.592 / 0.622 / 0.638 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 4,052 / 4,052 / 4,052 / 4,052 | 0.198 / 0.269 / 0.331 / 0.358 | 0.033 / 0.037 / 0.039 / 0.041 | 0.123 / 0.134 / 0.144 / 0.145 | 0.350 / 0.354 / 0.360 / 0.362 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 2,724 / 2,724 / 2,724 / 2,724 | 0.151 / 0.161 / 0.164 / 0.185 | 0.110 / 0.111 / 0.111 / 0.112 | 0.207 / 0.208 / 0.209 / 0.209 | 0.828 / 0.838 / 0.840 / 0.840 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 50,905 / 50,905 / 50,905 / 50,905 | 0.117 / 0.122 / 0.131 / 0.160 | 0.269 / 0.273 / 0.275 / 0.282 | 0.244 / 0.244 / 0.244 / 0.245 | 0.464 / 0.464 / 0.465 / 0.469 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 8,395 / 8,395 / 8,395 / 8,395 | 0.166 / 0.198 / 0.321 / 0.530 | 0.659 / 0.664 / 0.669 / 0.661 | 0.281 / 0.283 / 0.284 / 0.284 | 0.101 / 0.101 / 0.100 / 0.100 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |

</details>

<details>
<summary><strong>TP error — optimal</strong></summary>

| class_name | num_match@0.5/1.0/2.0/4.0 | ATE@0.5/1.0/2.0/4.0 | AOE@0.5/1.0/2.0/4.0 | ASE@0.5/1.0/2.0/4.0 | AVE@0.5/1.0/2.0/4.0 | AEE@0.5/1.0/2.0/4.0 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| car | 85,327 / 93,303 / 94,826 / 95,316 | 0.148 / 0.182 / 0.198 / 0.214 | 0.044 / 0.057 / 0.061 / 0.063 | 0.136 / 0.142 / 0.144 / 0.144 | 0.464 / 0.503 / 0.523 / 0.527 | 1.000 / 1.000 / 1.000 / 1.000 |
| truck | 32,549 / 39,457 / 43,979 / 46,340 | 0.188 / 0.269 / 0.373 / 0.472 | 0.026 / 0.034 / 0.041 / 0.048 | 0.145 / 0.162 / 0.179 / 0.192 | 0.610 / 0.671 / 0.725 / 0.764 | 1.000 / 1.000 / 1.000 / 1.000 |
| bus | 5,296 / 6,439 / 6,895 / 6,991 | 0.201 / 0.280 / 0.349 / 0.382 | 0.036 / 0.042 / 0.044 / 0.046 | 0.123 / 0.139 / 0.150 / 0.152 | 0.350 / 0.367 / 0.378 / 0.381 | 1.000 / 1.000 / 1.000 / 1.000 |
| bicycle | 4,446 / 4,536 / 4,537 / 4,547 | 0.152 / 0.161 / 0.164 / 0.186 | 0.110 / 0.110 / 0.110 / 0.109 | 0.209 / 0.210 / 0.211 / 0.210 | 0.830 / 0.839 / 0.841 / 0.840 | 1.000 / 1.000 / 1.000 / 1.000 |
| pedestrian | 98,679 / 99,889 / 100,492 / 101,325 | 0.122 / 0.131 / 0.145 / 0.189 | 0.288 / 0.292 / 0.296 / 0.306 | 0.248 / 0.248 / 0.249 / 0.249 | 0.480 / 0.479 / 0.480 / 0.485 | 1.000 / 1.000 / 1.000 / 1.000 |
| traffic_cone | 9,207 / 9,578 / 10,010 / 10,328 | 0.161 / 0.187 / 0.306 / 0.503 | 0.656 / 0.663 / 0.664 / 0.656 | 0.281 / 0.283 / 0.283 / 0.282 | 0.100 / 0.100 / 0.099 / 0.100 | 1.000 / 1.000 / 1.000 / 1.000 |
| barrier | 0 / 0 / 0 / 0 | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan | nan / nan / nan / nan |

</details>

</details>

---

### BEVFusion-LiDAR base/2.7.0

<details>
<summary> Changes  </summary>

- Train by min-max normalization (x, y, z, intensity, time_lag) into [0, 1], and then mapping it to fourier features [[1]](https://arxiv.org/pdf/2006.10739).
</details>

<details>
<summary> Artifacts </summary>

- Deployed onnx and ROS parameter files (for internal)
  - [WebAuto](https://evaluation.tier4.jp/evaluation/mlpackages/46f8188d-e3be-4f2f-b989-fd27002610d7/releases/51628f64-9c15-4029-b3c5-5bf501d879e2?project_id=zWhWRzei)
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-l/t4base/v2.7.0/deployment.zip)
  - [Google drive](https://drive.google.com/file/d/1zopj68qxLmI244qi3NgxB0ELT997V4W3/view?usp=drive_link)
- Logs (for internal)
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-l/t4base/v2.7.0/logs.zip)
  - [Google drive](https://drive.google.com/file/d/1-OIvsmsB69a5L_4sqjOSJ9IOltRWFDIv/view?usp=drive_link)
- Pytorch Best checkpoints:
  - [model-zoo](https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/bevfusion/bevfusion-l/t4base/v2.7.0/best_epoch_48.pth)
  - [Google drive](https://drive.google.com/file/d/1b8iwwLBLAmn0NwqRaTJOWHMINfS9p_fc/view?usp=drive_link)

</details>

<details>
<summary> Training configs </summary>

- [Config file path](https://github.com/KSeangTan/AWML/blob/0f5b5888148efcd2aac5af2315befd9301907745/projects/BEVFusion/configs/t4dataset/BEVFusion-L/bevfusion_lidar_voxel_second_secfpn_50e_8xb8_base_120m.py)
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

**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.8817**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 107,309 | 0.9131 | 0.862 / 0.914 / 0.933 / 0.943 | 0.905 / 0.935 / 0.942 / 0.945 | 0.233 / 0.192 / 0.159 / 0.142 |
| truck | 24,206 | 0.8552 | 0.711 / 0.843 / 0.919 / 0.948 | 0.795 / 0.877 / 0.918 / 0.934 | 0.297 / 0.225 / 0.192 / 0.180 |
| bus | 5,712 | 0.9081 | 0.829 / 0.912 / 0.945 / 0.947 | 0.876 / 0.916 / 0.931 / 0.932 | 0.312 / 0.146 / 0.146 / 0.146 |
| bicycle | 4,060 | 0.8357 | 0.813 / 0.840 / 0.844 / 0.846 | 0.857 / 0.868 / 0.869 / 0.870 | 0.210 / 0.194 / 0.194 / 0.194 |
| pedestrian | 77,369 | 0.8966 | 0.877 / 0.895 / 0.903 / 0.911 | 0.857 / 0.867 / 0.874 / 0.878 | 0.148 / 0.148 / 0.148 / 0.147 |
| **ALL** | 218,656 | 0.8817 | — | — | — |

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.7002**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 94,080 | 0.8174 | 0.708 / 0.817 / 0.864 / 0.881 | 0.782 / 0.844 / 0.867 / 0.872 | 0.212 / 0.166 / 0.164 / 0.161 |
| truck | 27,651 | 0.6660 | 0.463 / 0.626 / 0.759 / 0.815 | 0.612 / 0.714 / 0.787 / 0.812 | 0.229 / 0.190 / 0.154 / 0.130 |
| bus | 4,761 | 0.6414 | 0.393 / 0.602 / 0.775 / 0.795 | 0.554 / 0.691 / 0.798 / 0.807 | 0.324 / 0.219 / 0.181 / 0.138 |
| bicycle | 2,365 | 0.6430 | 0.586 / 0.658 / 0.663 / 0.666 | 0.683 / 0.715 / 0.716 / 0.717 | 0.141 / 0.141 / 0.141 / 0.141 |
| pedestrian | 37,523 | 0.7331 | 0.711 / 0.730 / 0.741 / 0.750 | 0.732 / 0.742 / 0.748 / 0.753 | 0.145 / 0.145 / 0.145 / 0.144 |
| **ALL** | 166,380 | 0.7002 | — | — | — |

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.5600**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 36,895 | 0.6578 | 0.498 / 0.656 / 0.726 / 0.751 | 0.626 / 0.714 / 0.750 / 0.760 | 0.168 / 0.143 / 0.137 / 0.132 |
| truck | 17,759 | 0.5131 | 0.206 / 0.450 / 0.648 / 0.749 | 0.439 / 0.611 / 0.720 / 0.775 | 0.240 / 0.193 / 0.134 / 0.124 |
| bus | 2,852 | 0.5178 | 0.313 / 0.520 / 0.608 / 0.630 | 0.534 / 0.659 / 0.704 / 0.714 | 0.244 / 0.166 / 0.140 / 0.140 |
| bicycle | 519 | 0.4296 | 0.315 / 0.421 / 0.491 / 0.491 | 0.503 / 0.563 / 0.592 / 0.592 | 0.180 / 0.180 / 0.180 / 0.180 |
| pedestrian | 17,091 | 0.6815 | 0.660 / 0.678 / 0.687 / 0.700 | 0.698 / 0.708 / 0.712 / 0.719 | 0.126 / 0.126 / 0.126 / 0.126 |
| **ALL** | 75,116 | 0.5600 | — | — | — |

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.7777**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 238,284 | 0.8504 | 0.760 / 0.851 / 0.888 / 0.903 | 0.818 / 0.868 / 0.886 / 0.890 | 0.219 / 0.184 / 0.161 / 0.158 |
| truck | 69,616 | 0.7065 | 0.492 / 0.671 / 0.802 / 0.861 | 0.641 / 0.752 / 0.822 / 0.851 | 0.251 / 0.216 / 0.173 / 0.136 |
| bus | 13,325 | 0.7443 | 0.575 / 0.735 / 0.827 / 0.840 | 0.703 / 0.791 / 0.843 / 0.849 | 0.345 / 0.181 / 0.181 / 0.146 |
| bicycle | 6,944 | 0.7538 | 0.714 / 0.761 / 0.769 / 0.771 | 0.776 / 0.797 / 0.800 / 0.801 | 0.186 / 0.176 / 0.176 / 0.176 |
| pedestrian | 131,983 | 0.8332 | 0.813 / 0.831 / 0.840 / 0.849 | 0.802 / 0.812 / 0.818 / 0.824 | 0.144 / 0.145 / 0.145 / 0.145 |
| **ALL** | 460,152 | 0.7777 | — | — | — |

---

**LargeBus**: db_largebus_v1 + db_largebus_v2 + db_largebus_v3 (1,228 frames)  

**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.8876**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 14,883 | 0.9176 | 0.876 / 0.916 / 0.934 / 0.944 | 0.917 / 0.943 / 0.947 / 0.949 | 0.245 / 0.154 / 0.154 / 0.154 |
| truck | 1,193 | 0.8727 | 0.747 / 0.873 / 0.926 / 0.944 | 0.829 / 0.900 / 0.924 / 0.928 | 0.269 / 0.206 / 0.157 / 0.157 |
| bus | 336 | 0.9443 | 0.824 / 0.975 / 0.989 / 0.989 | 0.878 / 0.974 / 0.984 / 0.984 | 0.439 / 0.338 / 0.269 / 0.269 |
| bicycle | 740 | 0.8396 | 0.764 / 0.848 / 0.869 / 0.877 | 0.833 / 0.862 / 0.866 / 0.871 | 0.194 / 0.194 / 0.182 / 0.182 |
| pedestrian | 5,059 | 0.8639 | 0.848 / 0.863 / 0.869 / 0.876 | 0.837 / 0.845 / 0.850 / 0.853 | 0.167 / 0.167 / 0.167 / 0.154 |
| **ALL** | 22,211 | 0.8876 | — | — | — |

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.7392**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 10,994 | 0.8425 | 0.745 / 0.846 / 0.883 / 0.896 | 0.810 / 0.869 / 0.886 / 0.891 | 0.210 / 0.170 / 0.153 / 0.153 |
| truck | 1,011 | 0.7288 | 0.537 / 0.722 / 0.818 / 0.838 | 0.670 / 0.784 / 0.834 / 0.840 | 0.184 / 0.158 / 0.113 / 0.113 |
| bus | 143 | 0.8580 | 0.589 / 0.944 / 0.944 / 0.956 | 0.730 / 0.929 / 0.929 / 0.929 | 0.510 / 0.463 / 0.463 / 0.463 |
| bicycle | 463 | 0.5826 | 0.477 / 0.607 / 0.622 / 0.625 | 0.606 / 0.667 / 0.671 / 0.673 | 0.118 / 0.112 / 0.102 / 0.102 |
| pedestrian | 3,754 | 0.6839 | 0.664 / 0.681 / 0.690 / 0.702 | 0.698 / 0.705 / 0.711 / 0.717 | 0.121 / 0.117 / 0.117 / 0.117 |
| **ALL** | 16,365 | 0.7392 | — | — | — |

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.5572**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 3,018 | 0.7091 | 0.556 / 0.712 / 0.776 / 0.792 | 0.665 / 0.747 / 0.778 / 0.786 | 0.205 / 0.181 / 0.181 / 0.181 |
| truck | 602 | 0.6393 | 0.365 / 0.651 / 0.760 / 0.781 | 0.553 / 0.730 / 0.789 / 0.798 | 0.208 / 0.208 / 0.152 / 0.152 |
| bus | 60 | 0.6121 | 0.420 / 0.637 / 0.696 / 0.696 | 0.583 / 0.725 / 0.765 / 0.765 | 0.275 / 0.197 / 0.197 / 0.197 |
| bicycle | 85 | 0.3386 | 0.244 / 0.355 / 0.378 / 0.378 | 0.446 / 0.514 / 0.524 / 0.524 | 0.181 / 0.181 / 0.137 / 0.137 |
| pedestrian | 1,121 | 0.4870 | 0.473 / 0.483 / 0.490 / 0.502 | 0.579 / 0.586 / 0.591 / 0.593 | 0.137 / 0.137 / 0.137 / 0.137 |
| **ALL** | 4,886 | 0.5572 | — | — | — |

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.8086**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 28,895 | 0.8789 | 0.806 / 0.881 / 0.909 / 0.919 | 0.853 / 0.896 / 0.908 / 0.911 | 0.245 / 0.185 / 0.176 / 0.170 |
| truck | 2,806 | 0.7783 | 0.597 / 0.778 / 0.859 / 0.880 | 0.714 / 0.824 / 0.865 / 0.870 | 0.206 / 0.206 / 0.157 / 0.155 |
| bus | 539 | 0.8898 | 0.718 / 0.931 / 0.952 / 0.958 | 0.808 / 0.931 / 0.937 / 0.937 | 0.382 / 0.354 / 0.354 / 0.354 |
| bicycle | 1,288 | 0.7288 | 0.641 / 0.744 / 0.762 / 0.768 | 0.729 / 0.769 / 0.773 / 0.776 | 0.176 / 0.176 / 0.176 / 0.172 |
| pedestrian | 9,934 | 0.7670 | 0.749 / 0.765 / 0.772 / 0.782 | 0.757 / 0.765 / 0.771 / 0.775 | 0.137 / 0.137 / 0.137 / 0.137 |
| **ALL** | 43,462 | 0.8086 | — | — | — |

---

**J6Gen2**: db_j6gen2_v1 + db_j6gen2_v2 + db_j6gen2_v3 + db_j6gen2_v4 + db_j6gen2_v5 + db_j6gen2_v6 + db_j6gen2_v7 + db_j6gen2_v8 + db_j6gen2_v9 (3,951 frames)

**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.8776**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 49,637 | 0.8907 | 0.841 / 0.890 / 0.909 / 0.922 | 0.896 / 0.924 / 0.931 / 0.934 | 0.269 / 0.199 / 0.159 / 0.135 |
| truck | 5,754 | 0.8438 | 0.718 / 0.833 / 0.894 / 0.930 | 0.794 / 0.862 / 0.893 / 0.915 | 0.222 / 0.194 / 0.171 / 0.171 |
| bus | 1,939 | 0.9473 | 0.878 / 0.942 / 0.983 / 0.986 | 0.925 / 0.963 / 0.981 / 0.982 | 0.206 / 0.140 / 0.140 / 0.140 |
| bicycle | 639 | 0.8665 | 0.854 / 0.871 / 0.871 / 0.871 | 0.867 / 0.875 / 0.875 / 0.875 | 0.176 / 0.176 / 0.176 / 0.176 |
| pedestrian | 14,362 | 0.8397 | 0.813 / 0.836 / 0.849 / 0.861 | 0.806 / 0.817 / 0.824 / 0.831 | 0.169 / 0.151 / 0.151 / 0.165 |
| **ALL** | 72,331 | 0.8776 | — | — | — |

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.6805**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 47,568 | 0.7957 | 0.662 / 0.795 / 0.851 / 0.875 | 0.760 / 0.838 / 0.866 / 0.874 | 0.212 / 0.184 / 0.164 / 0.164 |
| truck | 4,090 | 0.6451 | 0.451 / 0.622 / 0.729 / 0.778 | 0.606 / 0.711 / 0.768 / 0.789 | 0.234 / 0.205 / 0.176 / 0.165 |
| bus | 1,935 | 0.7955 | 0.571 / 0.760 / 0.912 / 0.938 | 0.694 / 0.815 / 0.906 / 0.916 | 0.345 / 0.240 / 0.182 / 0.168 |
| bicycle | 295 | 0.5394 | 0.494 / 0.552 / 0.554 / 0.557 | 0.628 / 0.669 / 0.669 / 0.669 | 0.137 / 0.138 / 0.138 / 0.138 |
| pedestrian | 6,529 | 0.6266 | 0.591 / 0.622 / 0.639 / 0.654 | 0.661 / 0.676 / 0.682 / 0.689 | 0.140 / 0.140 / 0.140 / 0.140 |
| **ALL** | 60,417 | 0.6805 | — | — | — |

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.4902**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 17,353 | 0.6483 | 0.452 / 0.639 / 0.734 / 0.768 | 0.608 / 0.712 / 0.760 / 0.774 | 0.168 / 0.153 / 0.143 / 0.132 |
| truck | 2,570 | 0.4871 | 0.209 / 0.419 / 0.619 / 0.702 | 0.425 / 0.578 / 0.700 / 0.746 | 0.199 / 0.127 / 0.126 / 0.124 |
| bus | 316 | 0.5172 | 0.246 / 0.532 / 0.626 / 0.665 | 0.433 / 0.640 / 0.701 / 0.721 | 0.173 / 0.100 / 0.100 / 0.089 |
| bicycle | 70 | 0.4406 | 0.382 / 0.438 / 0.471 / 0.471 | 0.584 / 0.619 / 0.637 / 0.637 | 0.193 / 0.193 / 0.193 / 0.193 |
| pedestrian | 1,673 | 0.3578 | 0.344 / 0.354 / 0.362 / 0.371 | 0.492 / 0.496 / 0.500 / 0.505 | 0.137 / 0.107 / 0.107 / 0.111 |
| **ALL** | 21,982 | 0.4902 | — | — | — |

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.7822**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 114,558 | 0.8292 | 0.725 / 0.826 / 0.872 / 0.894 | 0.800 / 0.859 / 0.881 / 0.888 | 0.232 / 0.194 / 0.164 / 0.158 |
| truck | 12,414 | 0.7169 | 0.534 / 0.691 / 0.795 / 0.847 | 0.665 / 0.760 / 0.816 / 0.843 | 0.251 / 0.194 / 0.166 / 0.151 |
| bus | 4,190 | 0.8590 | 0.703 / 0.840 / 0.938 / 0.955 | 0.790 / 0.874 / 0.929 / 0.936 | 0.345 / 0.186 / 0.182 / 0.168 |
| bicycle | 1,004 | 0.7505 | 0.724 / 0.758 / 0.760 / 0.760 | 0.781 / 0.798 / 0.799 / 0.799 | 0.176 / 0.176 / 0.176 / 0.176 |
| pedestrian | 22,564 | 0.7556 | 0.727 / 0.752 / 0.766 / 0.778 | 0.744 / 0.756 / 0.763 / 0.770 | 0.152 / 0.151 / 0.151 / 0.151 |
| **ALL** | 154,730 | 0.7822 | — | — | — |

---

**JPNTaxi_Gen2**: db_jpntaxigen2_v1 + db_jpntaxigen2_v2 (9,975 frames)

**Total BEV Center Distance mAP (eval range = 0.0 - 50.0m): 0.8837**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 42,789 | 0.9393 | 0.882 / 0.945 / 0.964 / 0.967 | 0.911 / 0.946 / 0.954 / 0.955 | 0.211 / 0.168 / 0.142 / 0.142 |
| truck | 17,259 | 0.8587 | 0.709 / 0.846 / 0.926 / 0.954 | 0.795 / 0.881 / 0.926 / 0.941 | 0.371 / 0.243 / 0.234 / 0.189 |
| bus | 3,437 | 0.8802 | 0.798 / 0.889 / 0.916 / 0.918 | 0.850 / 0.886 / 0.898 / 0.899 | 0.369 / 0.146 / 0.128 / 0.128 |
| bicycle | 2,681 | 0.8268 | 0.816 / 0.830 / 0.831 / 0.831 | 0.865 / 0.871 / 0.872 / 0.872 | 0.219 / 0.219 / 0.219 / 0.219 |
| pedestrian | 57,948 | 0.9135 | 0.896 / 0.912 / 0.919 / 0.926 | 0.872 / 0.882 / 0.889 / 0.893 | 0.148 / 0.140 / 0.143 / 0.140 |
| **ALL** | 124,114 | 0.8837 | — | — | — |

**Total BEV Center Distance mAP (eval range = 50.0 - 90.0m): 0.6901**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 35,518 | 0.8382 | 0.757 / 0.838 / 0.874 / 0.885 | 0.803 / 0.847 / 0.862 / 0.865 | 0.212 / 0.165 / 0.162 / 0.161 |
| truck | 22,550 | 0.6676 | 0.462 / 0.623 / 0.762 / 0.823 | 0.611 / 0.711 / 0.788 / 0.816 | 0.247 / 0.193 / 0.154 / 0.130 |
| bus | 2,683 | 0.5007 | 0.240 / 0.447 / 0.649 / 0.667 | 0.421 / 0.581 / 0.708 / 0.717 | 0.242 / 0.151 / 0.144 / 0.144 |
| bicycle | 1,607 | 0.6794 | 0.635 / 0.692 / 0.695 / 0.697 | 0.719 / 0.740 / 0.742 / 0.743 | 0.146 / 0.141 / 0.141 / 0.141 |
| pedestrian | 27,240 | 0.7645 | 0.745 / 0.762 / 0.772 / 0.780 | 0.753 / 0.764 / 0.769 / 0.773 | 0.156 / 0.144 / 0.145 / 0.145 |
| **ALL** | 89,598 | 0.6901 | — | — | — |

**Total BEV Center Distance mAP (eval range = 90.0 - 121.0m): 0.5750**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 16,524 | 0.6601 | 0.539 / 0.665 / 0.710 / 0.727 | 0.643 / 0.715 / 0.740 / 0.745 | 0.138 / 0.108 / 0.108 / 0.109 |
| truck | 14,587 | 0.5131 | 0.200 / 0.448 / 0.649 / 0.756 | 0.438 / 0.613 / 0.721 / 0.779 | 0.248 / 0.193 / 0.134 / 0.124 |
| bus | 2,476 | 0.5145 | 0.318 / 0.515 / 0.602 / 0.623 | 0.547 / 0.661 / 0.704 / 0.714 | 0.244 / 0.163 / 0.152 / 0.148 |
| bicycle | 364 | 0.4541 | 0.324 / 0.439 / 0.527 / 0.527 | 0.504 / 0.567 / 0.604 / 0.604 | 0.174 / 0.171 / 0.171 / 0.171 |
| pedestrian | 14,297 | 0.7331 | 0.711 / 0.730 / 0.739 / 0.753 | 0.731 / 0.742 / 0.746 / 0.754 | 0.126 / 0.126 / 0.126 / 0.126 |
| **ALL** | 48,248 | 0.5750 | — | — | — |

**Total BEV Center Distance mAP (eval range = 0.0 - 121.0m): 0.7715**

| class_name | GTs | mAP | AP@0.5/1.0/2.0/4.0 | max_f1@0.5/1.0/2.0/4.0 | optimal_conf@0.5/1.0/2.0/4.0 |
| :---- | ---: | ---: | :---- | :---- | :---- |
| car | 94,831 | 0.8661 | 0.785 / 0.869 / 0.900 / 0.910 | 0.828 / 0.871 / 0.884 / 0.887 | 0.198 / 0.165 / 0.150 / 0.141 |
| truck | 54,396 | 0.7010 | 0.478 / 0.662 / 0.800 / 0.864 | 0.632 / 0.747 / 0.821 / 0.852 | 0.273 / 0.216 / 0.173 / 0.134 |
| bus | 8,596 | 0.6721 | 0.500 / 0.665 / 0.756 / 0.768 | 0.648 / 0.737 / 0.792 / 0.798 | 0.326 / 0.151 / 0.146 / 0.146 |
| bicycle | 4,652 | 0.7611 | 0.731 / 0.766 / 0.773 / 0.775 | 0.790 / 0.805 / 0.809 / 0.809 | 0.186 / 0.187 / 0.187 / 0.187 |
| pedestrian | 99,485 | 0.8573 | 0.838 / 0.855 / 0.864 / 0.872 | 0.820 / 0.830 / 0.836 / 0.841 | 0.145 / 0.143 / 0.145 / 0.143 |
| **ALL** | 261,960 | 0.7715 | — | — | — |

</details>

---

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
