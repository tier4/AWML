# YOLOX-L Scene Selection Results

Here we present the results of preliminary experiments performed using YOLOX-L scene selection on TIER-IV's internal dataset.

## Results Summary
|                    | mAP  | car  | truck | bus  | bicycle | pedestrian |
|--------------------|------|------|-------|------|---------|------------|
| Baseline           | 64.9 | 81.1 | 52.6  | 68.9 | 59.2    | 62.6       |
| YOLOX-L Selection  | 50.2 | 64.6 | 44.8  | 64.3 | 30.6    | 46.5       |
| Random Sample 1    | 52.5 | 65.0 | 43.5  | 71.0 | 37.2    | 45.6       |
| Random Sample 2    | 51.0 | 65.2 | 44.2  | 66.6 | 31.4    | 47.7       |

**The baseline model performs better than fine-tuning, indicating issues with the fine-tuning process. The learning rate may need to be reduced. This document serves as a reference for evaluating scene selection methodology.**

## Sampling Details

We sampled datasets from DBv1.3 using three different approaches:

1. YOLOX-L Selection: Selected 100 datasets with highest average objects per scene (21,705 2D bounding boxes total)
2. Random Sample 1: 160 randomly sampled datasets matching YOLOX-L bbox count
3. Random Sample 2: 182 randomly sampled datasets matching YOLOX-L bbox count

We fine-tuned `TransFusion-L base/0.3` model checkpoint for 10 epochs using these datasets and compared results.

### Training Set Distribution
#### YOLOX-L Selection
```
Category-wise Instances:
+------------+--------+
| category   | number |
+------------+--------+
| car        | 29987  |
| truck      | 18295  |
| bus        | 3180   |
| bicycle    | 721    |
| pedestrian | 24688  |
+------------+--------+
```

#### Random Sampling 1 (with same number of objects as YOLOX-L)
```
Category-wise Instances:
+------------+--------+
| category   | number |
+------------+--------+
| car        | 34076  |
| truck      | 18888  |
| bus        | 5283   |
| bicycle    | 807    |
| pedestrian | 22349  |
+------------+--------+
```

#### Random Sampling 2 (with same number of objects as YOLOX-L)
```
Category-wise Instances:
+------------+--------+
| category   | number |
+------------+--------+
| car        | 35783  |
| truck      | 23963  |
| bus        | 5333   |
| bicycle    | 758    |
| pedestrian | 24761  |
+------------+--------+
```

---

## Results on Test Dataset

### YOLOX-L Selection
```
Total mAP: 0.502
------------- T4Metric results -------------
| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---- | ---- | ---- | ---- | ---- | ---- |
| car        | 64.6 | 48.6    | 66.2    | 71.2    | 72.6    |
| truck      | 44.8 | 21.2    | 41.0    | 54.1    | 62.8    |
| bus        | 64.3 | 42.9    | 64.5    | 73.4    | 76.3    |
| bicycle    | 30.6 | 26.3    | 31.6    | 32.3    | 32.4    |
| pedestrian | 46.5 | 39.8    | 43.8    | 49.1    | 53.4    |

```

### Random Sampling 1 (with same number of objects as YOLOX-L)
```
Total mAP: 0.525
------------- T4Metric results -------------
| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---- | ---- | ---- | ---- | ---- | ---- |
| car        | 65.0 | 45.7    | 67.3    | 72.8    | 74.4    |
| truck      | 43.5 | 21.9    | 39.9    | 52.3    | 59.7    |
| bus        | 71.0 | 50.7    | 72.0    | 78.6    | 82.9    |
| bicycle    | 37.2 | 30.6    | 39.2    | 39.4    | 39.6    |
| pedestrian | 45.6 | 40.1    | 44.0    | 47.6    | 50.7    |

```

### Random Sampling 2 (with same number of objects as YOLOX-L)
```
Total mAP: 0.510
------------- T4Metric results -------------
| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---- | ---- | ---- | ---- | ---- | ---- |
| car        | 65.2 | 48.8    | 66.8    | 71.9    | 73.2    |
| truck      | 44.2 | 22.5    | 39.8    | 52.7    | 61.6    |
| bus        | 66.6 | 40.5    | 66.6    | 78.0    | 81.3    |
| bicycle    | 31.4 | 17.1    | 35.8    | 36.1    | 36.5    |
| pedestrian | 47.7 | 41.7    | 45.8    | 49.8    | 53.6    |

```

### Baseline model (without training with db_jpntaxi_v4 )
```
Total mAP: 0.649
------------- T4Metric results -------------
| class_name | mAP  | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m |
| ---- | ---- | ---- | ---- | ---- | ---- |
| car        | 81.1 | 62.3    | 82.5    | 88.8    | 90.8    |
| truck      | 52.6 | 28.4    | 50.3    | 62.2    | 69.7    |
| bus        | 68.9 | 48.6    | 69.4    | 76.3    | 81.5    |
| bicycle    | 59.2 | 53.9    | 59.8    | 61.3    | 61.8    |
| pedestrian | 62.6 | 56.1    | 60.8    | 65.1    | 68.3    |


```
---

### Impressions

It seems that the YOLOX-L Selection is not better compared to random sampling. But nevertheless, it can be used to pick scenes with specific objects as necessary. For example, if we want to pick scenes with more bicycles in a particular area, we can use the YOLOX-L Selection.

---

