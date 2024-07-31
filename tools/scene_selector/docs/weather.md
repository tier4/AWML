# Select scene for weather

- We use BLIP-2 for filtering the condition of weather

## Example result

- Rain for with Nuscenes data

![](./fig/n008-2018-09-18-12-07-26-0400__CAM_FRONT__1537287126112404.jpg)

```
{'question': 'how is the weather', 'pred_answer': 'rain'}
```

- Snow for with [DAWN dataset](https://ar5iv.labs.arxiv.org/html/2008.05402)

![](./fig/snow_storm-004.jpg)

```
{'question': 'how is the weather', 'pred_answer': 'snow'}
```

- Fog for with [DAWN dataset](https://ar5iv.labs.arxiv.org/html/2008.05402)

![](./fig/foggy-049.jpg)

```
{'question': 'how is the weather', 'pred_answer': 'fog'}
```
