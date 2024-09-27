# (TBD) Select scene with VLM QA

## Example usage
### Scene selection for weather

- We use BLIP-2 for filtering the condition of weather
- Rain data from Nuscenes data

![](./docs/n008-2018-09-18-12-07-26-0400__CAM_FRONT__1537287126112404.jpg)

```
{'question': 'how is the weather', 'pred_answer': 'rain'}
```

- Snow data from [DAWN dataset](https://ar5iv.labs.arxiv.org/html/2008.05402)

![](./docs/snow_storm-004.jpg)

```
{'question': 'how is the weather', 'pred_answer': 'snow'}
```

- Fog data from [DAWN dataset](https://ar5iv.labs.arxiv.org/html/2008.05402)

![](./docs/foggy-049.jpg)

```
{'question': 'how is the weather', 'pred_answer': 'fog'}
```
